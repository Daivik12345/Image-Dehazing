"""Repeatability evidence and conservative eligibility for staged offload studies."""

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from btp.benchmark import validate_config
from btp.evaluation import paired_metrics, read_rgb, sha256
from btp.experiment import write_json


def _read_run(directory: Path) -> tuple[dict, dict, list[dict]]:
    config = validate_config(json.loads((directory / "config.json").read_text()))
    protocol = json.loads((directory / "protocol.json").read_text())
    rows = [
        json.loads(line) for line in (directory / "jobs.jsonl").read_text().splitlines()
    ]
    if (
        not config.get("variants")
        or not config.get("scene_ids")
        or config.get("profile")
        or any(r.get("profiled") for r in rows)
    ):
        raise ValueError("Expected explicit scenes/variants and unprofiled records")
    expected = set(
        itertools.product(
            [v["id"] for v in config["variants"]],
            config["scene_ids"],
            config["seeds"],
            range(config["repeats"]),
            config["modes"],
        )
    )
    actual = [
        (r["variant"], r["scene_id"], r["seed"], r["repeat"], r["mode"]) for r in rows
    ]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Missing, duplicate or unexpected jobs")
    variants = {v["id"]: v for v in config["variants"]}
    for row in rows:
        variant = variants[row["variant"]]
        if (
            row.get("policy") != variant["policy"]
            or row.get("cache_policy") != variant["cache_policy"]
        ):
            raise ValueError("Job policy/cache does not match config")
        if Path(row["job_id"]).name != row["job_id"] or row["job_id"] in {".", ".."}:
            raise ValueError("Invalid job directory")
    return config, protocol, rows


def _prediction(directory: Path, row: dict) -> Path:
    path = directory / row["job_id"] / "final/input.png"
    if row.get("prediction_sha256") and sha256(path) != row["prediction_sha256"]:
        raise ValueError("Prediction hash changed")
    return path


def _runtime(directory: Path, row: dict) -> dict:
    doc = json.loads((directory / row["job_id"] / "manifest.json").read_text())
    fields = (
        "checkpoint_sha256",
        "gpu",
        "torch",
        "cuda",
        "cudnn",
        "tf32_matmul",
        "tf32_cudnn",
        "deterministic_algorithms",
        "cudnn_benchmark",
        "precision",
        "steps",
        "input_size",
    )
    if any(field not in doc for field in fields):
        raise ValueError("Missing inference provenance")
    return {field: doc[field] for field in fields}


def analyze_repeatability(directory: Path) -> dict[str, Any]:
    """Measure same-seed M0 variability; require every planned repeat to succeed."""
    config, protocol, rows = _read_run(directory)
    if (
        len(config["variants"]) != 1
        or config["variants"][0]["policy"] != "M0"
        or config["variants"][0]["cache_policy"] != "guidance"
        or config["repeats"] < 3
        or config["split"] != "development"
        or any(r["status"] != "completed" for r in rows)
    ):
        raise ValueError(
            "Repeatability requires complete development M0/guidance with >=3 repeats"
        )
    runtime = _runtime(directory, rows[0])
    if any(_runtime(directory, r) != runtime for r in rows):
        raise ValueError("Runtime or geometry changed within repeatability run")
    differences, groups = [], []
    for scene, seed, mode in itertools.product(
        config["scene_ids"], config["seeds"], config["modes"]
    ):
        subset = [
            r
            for r in rows
            if (r["scene_id"], r["seed"], r["mode"]) == (scene, seed, mode)
        ]
        durations = [r.get("sampling_seconds") for r in subset]
        if any(
            not isinstance(t, (float, int)) or not np.isfinite(t) or t <= 0
            for t in durations
        ):
            raise ValueError("Invalid sampling duration")
        groups.append(
            {
                "scene_id": scene,
                "seed": seed,
                "mode": mode,
                "median_sampling_seconds": float(np.median(durations)),
                "min_sampling_seconds": min(durations),
                "max_sampling_seconds": max(durations),
            }
        )
        for a, b in itertools.combinations(subset, 2):
            scores = paired_metrics(
                read_rgb(_prediction(directory, a)), read_rgb(_prediction(directory, b))
            )
            differences.append(
                {
                    "scene_id": scene,
                    "seed": seed,
                    "mode": mode,
                    "repeat_a": a["repeat"],
                    "repeat_b": b["repeat"],
                    "mae": scores["mae"],
                    "max_abs_error": scores["max_abs_error"],
                }
            )
    return {
        "state": "complete",
        "jobs": len(rows),
        "repeat_pairs": len(differences),
        "mae": max(d["mae"] for d in differences),
        "max_abs_error": max(d["max_abs_error"] for d in differences),
        "groups": groups,
        "differences": differences,
        "runtime": runtime,
        "protocol": protocol,
        "note": "Observed saved-PNG variation only; not a universal numerical tolerance or proof of latent-gradient equivalence.",
    }


def screen_report(directory: Path, baseline_directory: Path) -> dict[str, Any]:
    """Check eligibility for further experiments, never declare an optimization win."""
    baseline = analyze_repeatability(baseline_directory)
    base_config, _, _ = _read_run(baseline_directory)
    config, protocol, rows = _read_run(directory)
    for field in ("manifest_sha256", "source_manifest_sha256"):
        if not protocol.get(field) or protocol[field] != baseline["protocol"].get(
            field
        ):
            raise ValueError(
                "Source or dataset provenance differs from repeatability run"
            )
    for field in ("scene_ids", "seeds", "modes", "split"):
        if config[field] != base_config[field]:
            raise ValueError("Screening panel differs from repeatability panel")
    control = next(
        v for v in config["variants"] if v["id"] == config["reference_variant"]
    )
    if control["policy"] != "M0" or control["cache_policy"] != "guidance":
        raise ValueError("Placement screen must use M0/guidance as control")
    controls = {
        (r["scene_id"], r["seed"], r["repeat"], r["mode"]): r
        for r in rows
        if r["variant"] == control["id"]
    }
    if any(r["status"] != "completed" for r in controls.values()):
        raise ValueError("Incomplete M0 screening control")
    for row in controls.values():
        if _runtime(directory, row) != baseline["runtime"]:
            raise ValueError("Control runtime differs from repeatability run")
        read_rgb(_prediction(directory, row))
    result = {}
    for variant in config["variants"]:
        selected = [r for r in rows if r["variant"] == variant["id"]]
        reasons, deltas = [], []
        for row in selected:
            if row["status"] != "completed":
                reasons.append(f"{row['job_id']}: {row['status']}")
                continue
            try:
                if _runtime(directory, row) != baseline["runtime"]:
                    raise ValueError("Runtime differs from baseline")
                match = controls[
                    (row["scene_id"], row["seed"], row["repeat"], row["mode"])
                ]
                diff = paired_metrics(
                    read_rgb(_prediction(directory, match)),
                    read_rgb(_prediction(directory, row)),
                )
                deltas.append(
                    {
                        "scene_id": row["scene_id"],
                        "mae": diff["mae"],
                        "max_abs_error": diff["max_abs_error"],
                    }
                )
                if (
                    diff["mae"] > baseline["mae"] + 1e-12
                    or diff["max_abs_error"] > baseline["max_abs_error"] + 1e-12
                ):
                    reasons.append(
                        f"{row['job_id']}: exceeds observed M0 repeat variation"
                    )
            except (OSError, ValueError, KeyError) as error:
                reasons.append(f"{row['job_id']}: {error}")
        result[variant["id"]] = {
            "policy": variant["policy"],
            "cache_policy": variant["cache_policy"],
            "eligible_for_cache_test": not reasons,
            "reasons": reasons,
            "output_differences": deltas,
        }
    return {
        "variants": result,
        "baseline_mae_limit": baseline["mae"],
        "baseline_max_abs_error_limit": baseline["max_abs_error"],
        "note": "Eligibility permits further development testing only. It is not a speedup, quality, or memory-headroom acceptance gate. Select placement manually; do not tune thresholds after seeing candidate outputs.",
    }


def main() -> None:
    """Write repeatability or screening reports outside timed inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = (
        screen_report(args.run, args.baseline)
        if args.baseline
        else analyze_repeatability(args.run)
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
