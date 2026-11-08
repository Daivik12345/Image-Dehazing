"""Bounded cold-process evaluation with immutable jobs and paired comparisons."""

import argparse
import csv
import itertools
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from btp.config import ROOT
from btp.evaluation import (
    PROTOCOL,
    LPIPSScorer,
    load_manifest,
    paired_metrics,
    read_rgb,
    sha256,
)
from btp.experiment import revision, write_json
from btp.memory import CACHE_POLICIES, POLICIES, resolve_cache_policy


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Reject unsupported execution modes and unbounded or ambiguous matrices."""
    allowed = {
        "schema_version",
        "policies",
        "variants",
        "reference_variant",
        "seeds",
        "repeats",
        "modes",
        "timeout_seconds",
        "max_jobs",
        "split",
        "execution",
        "scene_ids",
        "order_seed",
        "profile",
        "lpips",
    }
    if set(config) - allowed or config.get("schema_version") != 1:
        raise ValueError("Invalid config schema or unknown settings")
    if ("policies" in config) == ("variants" in config):
        raise ValueError("Specify exactly one of policies or variants")
    if "variants" in config:
        variants = config["variants"]
        if not isinstance(variants, list) or not variants:
            raise ValueError("Variants must be a nonempty list")
        ids, combinations = set(), set()
        for v in variants:
            if not isinstance(v, dict) or set(v) != {"id", "policy", "cache_policy"}:
                raise ValueError("Invalid variant schema")
            if not isinstance(v["id"], str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", v["id"]
            ):
                raise ValueError("Invalid variant ID")
            if v["cache_policy"] not in CACHE_POLICIES:
                raise ValueError("Each variant needs an explicit cache policy")
            cache = resolve_cache_policy(v["policy"], v["cache_policy"])
            combination = (v["policy"], cache)
            if v["id"] in ids or combination in combinations:
                raise ValueError("Duplicate variant ID or placement/cache combination")
            ids.add(v["id"])
            combinations.add(combination)
        if config.get("reference_variant") not in ids:
            raise ValueError("reference_variant must name a declared variant")
    elif "reference_variant" in config:
        raise ValueError("reference_variant requires explicit variants")
    fields = [("modes", ("accsamp", "spaced"))]
    if "policies" in config:
        fields.append(("policies", POLICIES))
    for name, choices in fields:
        values = config.get(name)
        if (
            not isinstance(values, list)
            or not values
            or len(values) != len(set(values))
            or any(v not in choices for v in values)
        ):
            raise ValueError(f"Invalid {name}")
    seeds = config.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("Invalid seeds")
    for name in ("repeats", "timeout_seconds", "max_jobs"):
        if type(config.get(name)) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config.get("execution") != "cold":
        raise ValueError(
            "Only cold execution is implemented; repeated CLI runs are not resident inference"
        )
    if config.get("split") not in ("development", "validation", "test"):
        raise ValueError("Invalid evaluation split")
    if type(config.get("order_seed", 42)) is not int:
        raise ValueError("Invalid order seed")
    for name in ("profile", "lpips"):
        if type(config.get(name, False)) is not bool:
            raise ValueError(f"{name} must be boolean")
    ids = config.get("scene_ids")
    if ids is not None and (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(x, str) for x in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Invalid scene_ids")
    return dict(config)


def execute_job(job: dict, command: list[str], log_path: Path, timeout: int) -> dict:
    """Run one fresh process and terminate its process group on timeout."""
    start = time.perf_counter()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
            env=env,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
            status = "completed" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            finally:
                # The leader may exit while a descendant ignores SIGTERM.
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code, status = process.returncode, "timeout"
    if status == "failed":
        tail = log_path.read_text(errors="replace")[-20000:]
        if "out of memory" in tail.lower():
            status = "oom"
    return {
        "status": status,
        "returncode": code,
        "cold_process_seconds": time.perf_counter() - start,
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize(jobs: list[dict], reference_variant: str = "M0") -> dict:
    """Aggregate scenes equally and compare only matched successful jobs."""
    groups = defaultdict(list)
    for job in jobs:
        groups[job.get("variant", job["policy"]) + "/" + job["mode"]].append(job)
    policies, comparisons = {}, {}
    for key, rows in groups.items():
        good = [r for r in rows if r["status"] == "completed"]
        scene_rows = defaultdict(list)
        for row in good:
            scene_rows[row["scene_id"]].append(row)
        stats = {
            "requested": len(rows),
            "completed": len(good),
            "failed": len(rows) - len(good),
            "successful_scenes": len(scene_rows),
            "profiled": any(r["profiled"] for r in rows),
        }
        for metric in ("psnr_db", "ssim", "lpips"):
            means = [
                _mean(
                    [
                        r["metrics"][metric]
                        for r in rs
                        if r["metrics"].get(metric) is not None
                    ]
                )
                for rs in scene_rows.values()
            ]
            stats[metric] = _mean([v for v in means if v is not None])
        stats["identical_outputs"] = sum(r["metrics"]["identical"] for r in good)
        if stats["identical_outputs"]:
            stats["psnr_db"] = None
        for field in (
            "lifetime_peak_allocated_bytes",
            "lifetime_peak_reserved_bytes",
            "host_peak_rss_bytes",
        ):
            values = [r.get("telemetry", {}).get(field) for r in rows]
            stats[field + "_max"] = max(
                (v for v in values if v is not None), default=None
            )
        for field in ("cold_process_seconds", "sampling_seconds"):
            medians = [
                float(np.median([r[field] for r in rs if r.get(field) is not None]))
                for rs in scene_rows.values()
                if any(r.get(field) is not None for r in rs)
            ]
            stats[field + "_scene_median"] = (
                float(np.median(medians)) if medians else None
            )
            stats[field + "_p95_across_scenes"] = (
                float(np.percentile(medians, 95)) if len(medians) >= 20 else None
            )
        policies[key] = stats
        if key == reference_variant + "/" + rows[0]["mode"]:
            continue
        baseline = {
            (r["scene_id"], r["seed"], r["repeat"]): r
            for r in groups.get(reference_variant + "/" + rows[0]["mode"], [])
            if r["status"] == "completed"
        }
        pairs = [
            (r, baseline[(r["scene_id"], r["seed"], r["repeat"])])
            for r in good
            if (r["scene_id"], r["seed"], r["repeat"]) in baseline
        ]
        comparison = {
            "reference_variant": reference_variant,
            "common_success_pairs": len(pairs),
            "candidate_failed": len(rows) - len(good),
            "baseline_failed": sum(
                r["status"] != "completed"
                for r in groups.get(reference_variant + "/" + rows[0]["mode"], [])
            ),
            "scene_deltas": {},
        }
        for metric in ("psnr_db", "ssim", "lpips"):
            deltas = defaultdict(list)
            for candidate, base in pairs:
                a, b = candidate["metrics"].get(metric), base["metrics"].get(metric)
                if a is not None and b is not None:
                    deltas[candidate["scene_id"]].append(a - b)
            means = {scene: float(np.mean(v)) for scene, v in deltas.items()}
            comparison["scene_deltas"][metric] = means
            comparison[metric + "_delta"] = _mean(list(means.values()))
            comparison[metric + "_valid_pairs"] = sum(len(v) for v in deltas.values())
            if len(means) >= 2:
                rng = np.random.default_rng(42)
                samples = rng.choice(
                    list(means.values()), size=(2000, len(means)), replace=True
                ).mean(axis=1)
                comparison[metric + "_scene_bootstrap_ci95"] = np.percentile(
                    samples, [2.5, 97.5]
                ).tolist()
        differences = []
        for candidate, base in pairs:
            diff = paired_metrics(
                read_rgb(Path(base["prediction"])),
                read_rgb(Path(candidate["prediction"])),
            )
            differences.append(
                {
                    "scene_id": candidate["scene_id"],
                    "seed": candidate["seed"],
                    "repeat": candidate["repeat"],
                    "mae": diff["mae"],
                    "max_abs_error": diff["max_abs_error"],
                }
            )
        comparison["output_differences"] = differences
        for field in ("sampling_seconds", "cold_process_seconds"):
            ratios = defaultdict(list)
            for candidate, base in pairs:
                a, b = candidate.get(field), base.get(field)
                if a is not None and b is not None and a > 0 and b > 0:
                    ratios[candidate["scene_id"]].append(b / a)
            comparison[field + "_speedup_scene_median"] = (
                float(np.median([np.median(v) for v in ratios.values()]))
                if ratios
                else None
            )
        comparisons[key] = comparison
    return {
        "policies": policies,
        "comparisons": comparisons,
        "note": "Success-only quality with failure counts; equal scene weighting. Null aggregate PSNR means no valid outputs or at least one exact match; inspect counts. Exact-match PSNR pairs are excluded from finite delta statistics and valid pair counts are explicit. Profiled runs are diagnostic, not timing benchmarks. No automatic policy acceptance.",
    }


def run_matrix(
    manifest_path: Path,
    config: dict,
    weights: Path,
    output: Path,
    executor: Callable = execute_job,
) -> dict:
    """Validate, run a bounded randomized matrix, then score outside inference."""
    config = validate_config(config)
    variants = config.get("variants") or [
        {"id": policy, "policy": policy, "cache_policy": resolve_cache_policy(policy)}
        for policy in config["policies"]
    ]
    manifest = load_manifest(manifest_path)
    scenes = [s for s in manifest["scenes"] if s["split"] == config["split"]]
    if config.get("scene_ids"):
        scenes = [s for s in scenes if s["scene_id"] in config["scene_ids"]]
        if {s["scene_id"] for s in scenes} != set(config["scene_ids"]):
            raise ValueError("Requested scene IDs missing from selected split")
    if len({s["geometry_tier"] for s in scenes}) > 1:
        raise ValueError(
            "Run each geometry tier separately; mixed-tier aggregation is unsupported"
        )
    count = (
        len(scenes)
        * len(variants)
        * len(config["modes"])
        * len(config["seeds"])
        * config["repeats"]
    )
    if not count or count > config["max_jobs"]:
        raise ValueError(f"Matrix has {count} jobs; limit {config['max_jobs']}")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    write_json(output / "config.json", config)
    write_json(
        output / "protocol.json",
        {
            "metric": PROTOCOL,
            "manifest_sha256": sha256(manifest_path),
            "dataset": manifest["dataset"],
            "release": manifest["release"],
            "source_commit": revision(ROOT),
            "source_manifest_sha256": sha256(ROOT / "source-manifest.json")
            if (ROOT / "source-manifest.json").exists()
            else None,
            "geometry_tiers": sorted({s["geometry_tier"] for s in scenes}),
            "preview_every": 0,
            "execution": "cold",
            "order_seed": config.get("order_seed", 42),
        },
    )
    write_json(output / "dataset-manifest.json", manifest)
    jobs = []
    for scene, variant, mode, seed, repeat in itertools.product(
        scenes,
        variants,
        config["modes"],
        config["seeds"],
        range(config["repeats"]),
    ):
        jobs.append(
            {
                "job_id": f"job-{len(jobs):05d}",
                "scene_id": scene["scene_id"],
                "split": scene["split"],
                "geometry_tier": scene["geometry_tier"],
                "input_sha256": scene["hazy_sha256"],
                "target_sha256": scene["clean_sha256"],
                "policy": variant["policy"],
                "variant": variant["id"],
                "cache_policy": variant["cache_policy"],
                "mode": mode,
                "seed": seed,
                "repeat": repeat,
                "input": str((manifest_path.parent / scene["hazy"]).resolve()),
                "target": str((manifest_path.parent / scene["clean"]).resolve()),
                "profiled": config.get("profile", False),
                "status": "pending",
            }
        )
    random.Random(config.get("order_seed", 42)).shuffle(jobs)

    def persist():
        temporary = output / "jobs.jsonl.tmp"
        temporary.write_text(
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in jobs)
        )
        temporary.replace(output / "jobs.jsonl")

    persist()
    for index, job in enumerate(jobs):
        job["status"] = "running"
        job["launch_index"] = index
        job["started_unix_seconds"] = time.time()
        persist()
        command = [
            sys.executable,
            "-u",
            "-m",
            "btp.reference",
            "--input",
            job["input"],
            "--weights",
            str(weights.resolve()),
            "--output",
            str(output / job["job_id"]),
            "--mode",
            job["mode"],
            "--seed",
            str(job["seed"]),
            "--dataset-version",
            sha256(manifest_path),
            "--preview-every",
            "0",
            "--memory-policy",
            job["policy"],
        ]
        if "variants" in config:
            command.extend(["--cache-policy", job["cache_policy"]])
        if job["profiled"]:
            command.append("--profile")
        job["command"] = command
        try:
            job.update(
                executor(
                    job,
                    command,
                    output / "logs" / f"{job['job_id']}.log",
                    config["timeout_seconds"],
                )
            )
        except Exception as error:
            job.update(status="runner_error", error=str(error))
        job["finished_unix_seconds"] = time.time()
        telemetry_file = output / job["job_id"] / "telemetry.json"
        if telemetry_file.is_file():
            try:
                job["telemetry"] = json.loads(telemetry_file.read_text())
            except (ValueError, OSError) as error:
                job["telemetry_error"] = str(error)
        persist()
        print(
            json.dumps(
                {
                    "job": index + 1,
                    "total": count,
                    "policy": job["policy"],
                    "variant": job["variant"],
                    "scene": job["scene_id"],
                    "status": job["status"],
                }
            ),
            flush=True,
        )
    scorer = None
    if config.get("lpips"):
        try:
            scorer = LPIPSScorer()
            write_json(output / "lpips-provenance.json", scorer.provenance())
        except Exception as error:
            write_json(output / "lpips-error.json", {"error": str(error)})
            for job in jobs:
                job["lpips_error"] = str(error)
    for job in jobs:
        if job["status"] != "completed":
            continue
        try:
            directory = output / job["job_id"]
            if (
                json.loads((directory / "status.json").read_text())["state"]
                != "completed"
            ):
                raise ValueError("Inference did not report completion")
            prediction = directory / "final/input.png"
            clean, pred = read_rgb(Path(job["target"])), read_rgb(prediction)
            job["metrics"] = paired_metrics(clean, pred)
            if scorer:
                try:
                    value = float(scorer(clean, pred))
                    if not np.isfinite(value):
                        raise ValueError("Non-finite LPIPS result")
                    job["metrics"]["lpips"] = value
                except Exception as error:
                    job["lpips_error"] = str(error)
            telemetry = json.loads((directory / "telemetry.json").read_text())
            job["sampling_seconds"] = telemetry["sampling"].get("wall_seconds")
            job["telemetry"] = telemetry
            job["prediction"] = str(prediction)
            job["prediction_sha256"] = sha256(prediction)
        except Exception as error:
            job.update(status="invalid_output", error=str(error))
        persist()
    summary = summarize(jobs, config.get("reference_variant", "M0"))
    write_json(output / "summary.json", summary)
    fields = [
        "job_id",
        "scene_id",
        "policy",
        "variant",
        "cache_policy",
        "mode",
        "seed",
        "repeat",
        "status",
        "cold_process_seconds",
        "sampling_seconds",
        "psnr_db",
        "ssim",
        "lpips",
        "identical",
    ]
    with (output / "per_image.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for job in jobs:
            writer.writerow({**job, **job.get("metrics", {})})
    return summary


def main() -> None:
    """Run a manifest/config pair; output must be a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "config", "weights", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run_matrix(
        args.manifest, json.loads(args.config.read_text()), args.weights, args.output
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
