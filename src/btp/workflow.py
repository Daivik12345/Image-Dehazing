"""Explicit, restart-safe selection from validated optimization evidence."""

import argparse
from pathlib import Path
from typing import Any

from btp.evaluation import paired_metrics, read_rgb, sha256
from btp.experiment import write_json
from btp.optimization import (
    _prediction,
    _read_run,
    _runtime,
    analyze_repeatability,
    screen_report,
)


def select_placement(
    baseline: Path, screen: Path, policy: str, output: Path
) -> dict[str, Any]:
    """Recompute eligibility and save an immutable, explicit placement decision.

    Never infer the session from file modification times or trust a cached report.
    Rejected policies include their evidence reasons in the error message.
    """
    if output.exists():
        raise FileExistsError(f"Preserve the existing decision: {output}")
    baseline, screen = baseline.resolve(), screen.resolve()
    report = screen_report(screen, baseline)
    matches = [
        (key, value)
        for key, value in report["variants"].items()
        if value["policy"] == policy and value["cache_policy"] == "guidance"
    ]
    if len(matches) != 1:
        raise ValueError(f"Policy {policy!r} not present exactly once in {screen}")
    variant, evidence = matches[0]
    if not evidence["eligible_for_cache_test"]:
        raise ValueError(
            f"{variant} rejected in {screen}: " + "; ".join(evidence["reasons"])
        )
    result = {
        "schema_version": 1,
        "policy": policy,
        "variant": variant,
        "baseline": str(baseline),
        "screen": str(screen),
        "evidence_hashes": {
            f"{name}/{file}": sha256(directory / file)
            for name, directory in (("baseline", baseline), ("screen", screen))
            for file in ("config.json", "protocol.json", "jobs.jsonl")
        },
        "repeatability": analyze_repeatability(baseline),
        "eligibility": report,
        "scope": "Eligible for cache testing only; not final acceptance",
    }
    write_json(output, result)
    return result


def select_cache(
    baseline: Path, screen: Path, cache_run: Path, policy: str, cache: str, output: Path
) -> dict[str, Any]:
    """Recompute placement and cache gates before permitting confirmation."""
    if output.exists():
        raise FileExistsError(output)
    placement = screen_report(screen, baseline)
    selected = [v for v in placement["variants"].values() if v["policy"] == policy]
    if len(selected) != 1 or not selected[0]["eligible_for_cache_test"]:
        raise ValueError(f"Placement rejected: {selected}")
    repeatability = analyze_repeatability(baseline)
    base_config, _, _ = _read_run(baseline)
    config, protocol, rows = _read_run(cache_run)
    for field in ("manifest_sha256", "source_manifest_sha256"):
        if protocol.get(field) != repeatability["protocol"].get(field):
            raise ValueError("Cache provenance differs from baseline")
    for field in ("scene_ids", "seeds", "modes", "split"):
        if config[field] != base_config[field]:
            raise ValueError("Cache panel differs from baseline")
    control = next(
        v for v in config["variants"] if v["id"] == config["reference_variant"]
    )
    if control["policy"] != policy or control["cache_policy"] != "guidance":
        raise ValueError("Cache control must use the selected placement/guidance")
    candidates = [
        v
        for v in config["variants"]
        if v["policy"] == policy and v["cache_policy"] == cache
    ]
    if len(candidates) != 1:
        raise ValueError("Cache candidate not present exactly once")
    candidate = candidates[0]

    def key(r):
        return (r["scene_id"], r["seed"], r["repeat"], r["mode"])

    controls = {key(r): r for r in rows if r["variant"] == control["id"]}
    differences = []
    for row in rows:
        if row["variant"] != candidate["id"]:
            continue
        reference = controls[key(row)]
        for record in (reference, row):
            if record["status"] != "completed":
                raise ValueError(f"Cache job {record['job_id']}: {record['status']}")
            if _runtime(cache_run, record) != repeatability["runtime"]:
                raise ValueError("Cache runtime differs from baseline")
        diff = paired_metrics(
            read_rgb(_prediction(cache_run, reference)),
            read_rgb(_prediction(cache_run, row)),
        )
        if (
            diff["mae"] > repeatability["mae"] + 1e-12
            or diff["max_abs_error"] > repeatability["max_abs_error"] + 1e-12
        ):
            raise ValueError(
                f"Cache job {row['job_id']} exceeds observed M0 repeat variation"
            )
        differences.append(
            {
                "job_id": row["job_id"],
                "mae": diff["mae"],
                "max_abs_error": diff["max_abs_error"],
            }
        )
    if not differences:
        raise ValueError("No paired cache differences")
    result = {
        "policy": policy,
        "cache_policy": cache,
        "differences": differences,
        "protocol": protocol,
        "scope": "Eligible for confirmation, not accepted",
    }
    write_json(output, result)
    return result


def verify_current_provenance(protocol: dict, source: Path, manifest: Path) -> None:
    """Reject restored evidence from a different current source or dataset."""
    for name, path in (
        ("source_manifest_sha256", source),
        ("manifest_sha256", manifest),
    ):
        if protocol.get(name) != sha256(path):
            raise ValueError(f"Current {name} differs from saved evidence")


def main() -> None:
    """Select a policy using explicit saved run directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--screen", required=True, type=Path)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-run", type=Path)
    parser.add_argument("--cache")
    args = parser.parse_args()
    if args.cache_run is not None:
        select_cache(
            args.baseline,
            args.screen,
            args.cache_run,
            args.policy,
            args.cache,
            args.output,
        )
    else:
        select_placement(args.baseline, args.screen, args.policy, args.output)
    print(f"Decision saved to {args.output}")


if __name__ == "__main__":
    main()
