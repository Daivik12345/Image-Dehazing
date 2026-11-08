"""Regression checks for bounded jobs and failure-preserving comparisons."""

import json
import shutil

import pytest
from test_evaluation import fixture_manifest

from btp.benchmark import run_matrix, validate_config


def test_config_rejects_duplicates_and_unbounded_work():
    config = {
        "schema_version": 1,
        "policies": ["M0", "M1"],
        "seeds": [42],
        "repeats": 1,
        "modes": ["accsamp"],
        "timeout_seconds": 120,
        "max_jobs": 4,
        "split": "development",
        "execution": "cold",
    }
    assert validate_config(config)["policies"] == ["M0", "M1"]
    with pytest.raises(ValueError):
        validate_config({**config, "policies": ["M0", "M0"]})
    with pytest.raises(ValueError):
        validate_config({**config, "timeout_seconds": 0})


def test_matrix_retains_failures_and_does_not_score_missing_outputs(tmp_path):
    manifest, _ = fixture_manifest(tmp_path)
    config = {
        "schema_version": 1,
        "policies": ["M0", "M1"],
        "seeds": [42],
        "repeats": 1,
        "modes": ["accsamp"],
        "timeout_seconds": 120,
        "max_jobs": 4,
        "split": "development",
        "execution": "cold",
    }

    def fake(job, command, log_path, timeout):
        if job["policy"] == "M1":
            return {"status": "timeout", "returncode": None}
        out = tmp_path / "result" / job["job_id"]
        (out / "final").mkdir(parents=True)
        shutil.copy2(tmp_path / "clean.png", out / "final/input.png")
        (out / "status.json").write_text('{"state":"completed"}')
        (out / "telemetry.json").write_text(
            '{"sampling":{"wall_seconds":1.0},"profiled":false}'
        )
        return {"status": "completed", "returncode": 0}

    summary = run_matrix(manifest, config, tmp_path, tmp_path / "result", executor=fake)
    assert summary["policies"]["M0/accsamp"]["completed"] == 1
    assert summary["policies"]["M1/accsamp"]["failed"] == 1
    assert summary["comparisons"]["M1/accsamp"]["common_success_pairs"] == 0
    records = [
        json.loads(x) for x in (tmp_path / "result/jobs.jsonl").read_text().splitlines()
    ]
    assert {r["status"] for r in records} == {"completed", "timeout"}
    with pytest.raises(FileExistsError):
        run_matrix(manifest, config, tmp_path, tmp_path / "result", executor=fake)


def test_summary_weights_scenes_equally_and_keeps_failures():
    from btp.benchmark import summarize

    rows = []
    for scene, score in [("a", 10), ("a", 10), ("b", 30)]:
        rows.append(
            {
                "scene_id": scene,
                "policy": "M0",
                "mode": "accsamp",
                "status": "completed",
                "profiled": False,
                "metrics": {"psnr_db": score, "ssim": 0.8, "identical": False},
                "sampling_seconds": score,
                "repeat": 0,
                "seed": 42,
            }
        )
    rows.append(
        {
            "scene_id": "c",
            "policy": "M0",
            "mode": "accsamp",
            "status": "oom",
            "profiled": False,
        }
    )
    stats = summarize(rows)["policies"]["M0/accsamp"]
    assert stats["psnr_db"] == 20
    assert stats["failed"] == 1
    assert stats["sampling_seconds_scene_median"] == 20
    assert stats["sampling_seconds_p95_across_scenes"] is None


def test_subprocess_timeout_is_recorded(tmp_path):
    import sys

    from btp.benchmark import execute_job

    result = execute_job(
        {},
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path / "job.log",
        1,
    )
    assert result["status"] == "timeout"
    assert result["cold_process_seconds"] < 15


def test_exact_matches_do_not_disappear_from_mean_psnr():
    from btp.benchmark import summarize

    rows = [
        {
            "scene_id": "a",
            "policy": "M0",
            "mode": "accsamp",
            "status": "completed",
            "profiled": False,
            "metrics": {"psnr_db": None, "ssim": 1.0, "identical": True},
        },
        {
            "scene_id": "b",
            "policy": "M0",
            "mode": "accsamp",
            "status": "completed",
            "profiled": False,
            "metrics": {"psnr_db": 20.0, "ssim": 0.8, "identical": False},
        },
    ]
    stats = summarize(rows)["policies"]["M0/accsamp"]
    assert stats["psnr_db"] is None
    assert stats["identical_outputs"] == 1


def test_nonfinite_lpips_preserves_other_metrics(tmp_path, monkeypatch):
    import btp.benchmark as benchmark

    manifest, _ = fixture_manifest(tmp_path)
    config = {
        "schema_version": 1,
        "policies": ["M0"],
        "seeds": [42],
        "repeats": 1,
        "modes": ["accsamp"],
        "timeout_seconds": 120,
        "max_jobs": 1,
        "split": "development",
        "execution": "cold",
        "lpips": True,
    }

    class Scorer:
        def provenance(self):
            return {}

        def __call__(self, *args):
            return float("nan")

    monkeypatch.setattr(benchmark, "LPIPSScorer", Scorer)

    def fake(job, command, log_path, timeout):
        out = tmp_path / "result" / job["job_id"]
        (out / "final").mkdir(parents=True)
        shutil.copy2(tmp_path / "clean.png", out / "final/input.png")
        (out / "status.json").write_text('{"state":"completed"}')
        (out / "telemetry.json").write_text('{"sampling":{"wall_seconds":1}}')
        return {"status": "completed", "returncode": 0}

    result = run_matrix(manifest, config, tmp_path, tmp_path / "result", executor=fake)
    assert result["policies"]["M0/accsamp"]["completed"] == 1
    row = json.loads((tmp_path / "result/jobs.jsonl").read_text())
    assert "lpips_error" in row and "lpips" not in row["metrics"]


def test_timeout_kills_group_after_leader_exits(tmp_path, monkeypatch):
    import signal
    import subprocess

    import btp.benchmark as benchmark

    signals = []

    class Process:
        pid = 12345
        returncode = -15
        calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("worker", 1)
            return self.returncode

    monkeypatch.setattr(benchmark.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(benchmark.os, "killpg", lambda pid, sig: signals.append(sig))
    benchmark.execute_job({}, ["worker"], tmp_path / "log", 1)
    assert signals == [signal.SIGTERM, signal.SIGKILL]
