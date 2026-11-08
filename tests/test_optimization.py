"""Fail-closed baseline repeatability and cache-stage eligibility."""

import json

import pytest
from PIL import Image

from btp.optimization import analyze_repeatability, screen_report


def write_run(root, variants, repeats):
    root.mkdir()
    config = {
        "schema_version": 1,
        "variants": [
            {
                "id": v,
                "policy": "M0" if v == "baseline" else "M2-U",
                "cache_policy": "guidance",
            }
            for v in variants
        ],
        "reference_variant": "baseline",
        "scene_ids": ["a", "b"],
        "seeds": [42],
        "modes": ["accsamp"],
        "repeats": repeats,
        "max_jobs": len(variants) * 2 * repeats,
        "timeout_seconds": 60,
        "execution": "cold",
        "split": "development",
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "protocol.json").write_text(
        json.dumps({"manifest_sha256": "data", "source_manifest_sha256": "source"})
    )
    rows = []
    for v in variants:
        for scene in ["a", "b"]:
            for repeat in range(repeats):
                jid = f"job-{len(rows):05d}"
                out = root / jid
                (out / "final").mkdir(parents=True)
                Image.new("RGB", (16, 16), (80, 90, 100)).save(out / "final/input.png")
                (out / "manifest.json").write_text(
                    json.dumps(
                        {
                            "checkpoint_sha256": {"model": "hash"},
                            "gpu": "fake",
                            "torch": "2.6",
                            "cuda": "11.8",
                            "precision": "float32",
                            "steps": 50,
                            "input_size": [16, 16],
                            "cudnn": 90100,
                            "tf32_matmul": False,
                            "tf32_cudnn": True,
                            "deterministic_algorithms": False,
                            "cudnn_benchmark": False,
                        }
                    )
                )
                rows.append(
                    {
                        "job_id": jid,
                        "scene_id": scene,
                        "variant": v,
                        "policy": "M0" if v == "baseline" else "M2-U",
                        "cache_policy": "guidance",
                        "seed": 42,
                        "repeat": repeat,
                        "mode": "accsamp",
                        "status": "completed",
                        "profiled": False,
                        "sampling_seconds": 10 + repeat,
                    }
                )
    (root / "jobs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def test_repeatability_and_cache_gate(tmp_path):
    baseline = write_run(tmp_path / "baseline", ["baseline"], 3)
    screen = write_run(tmp_path / "screen", ["baseline", "unet"], 1)
    report = analyze_repeatability(baseline)
    assert report["max_abs_error"] == 0
    assert report["repeat_pairs"] == 6
    result = screen_report(screen, baseline)
    assert result["variants"]["unet"]["eligible_for_cache_test"]
    Image.new("RGB", (16, 16), (81, 90, 100)).save(screen / "job-00002/final/input.png")
    assert not screen_report(screen, baseline)["variants"]["unet"][
        "eligible_for_cache_test"
    ]


def test_repeatability_rejects_missing_failed_or_profiled_records(tmp_path):
    baseline = write_run(tmp_path / "baseline", ["baseline"], 3)
    p = baseline / "jobs.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    for altered in [
        rows[:-1],
        [{**r, "status": "oom"} if i == 0 else r for i, r in enumerate(rows)],
        [{**r, "profiled": True} for r in rows],
    ]:
        p.write_text("".join(json.dumps(r) + "\n" for r in altered))
        with pytest.raises(ValueError):
            analyze_repeatability(baseline)


def test_screen_keeps_failed_candidate_ineligible(tmp_path):
    baseline = write_run(tmp_path / "baseline", ["baseline"], 3)
    screen = write_run(tmp_path / "screen", ["baseline", "unet"], 1)
    p = screen / "jobs.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    rows[-1]["status"] = "oom"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert not screen_report(screen, baseline)["variants"]["unet"][
        "eligible_for_cache_test"
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("cudnn", 90200),
        ("tf32_matmul", True),
        ("tf32_cudnn", False),
        ("deterministic_algorithms", True),
        ("cudnn_benchmark", True),
    ],
)
def test_numerical_runtime_changes_invalidate_repeats_and_candidates(
    tmp_path, field, value
):
    baseline = write_run(tmp_path / "baseline", ["baseline"], 3)
    screen = write_run(tmp_path / "screen", ["baseline", "unet"], 1)
    candidate = screen / "job-00002/manifest.json"
    doc = json.loads(candidate.read_text())
    doc[field] = value
    candidate.write_text(json.dumps(doc))
    assert not screen_report(screen, baseline)["variants"]["unet"][
        "eligible_for_cache_test"
    ]
    control = screen / "job-00000/manifest.json"
    doc = json.loads(control.read_text())
    doc[field] = value
    control.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="runtime"):
        screen_report(screen, baseline)
    repeat = baseline / "job-00001/manifest.json"
    doc = json.loads(repeat.read_text())
    doc[field] = value
    repeat.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="Runtime"):
        analyze_repeatability(baseline)
