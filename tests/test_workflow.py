"""Recovery must recompute gates and preserve explicit session identity."""

import json

import pytest
from test_optimization import write_run

from btp.workflow import select_placement


def test_selection_persists_explicit_paths(tmp_path):
    base = write_run(tmp_path / "base", ["baseline"], 3)
    screen = write_run(tmp_path / "screen", ["baseline", "unet"], 1)
    result = select_placement(base, screen, "M2-U", tmp_path / "decision.json")
    assert result["policy"] == "M2-U"
    assert result["screen"] == str(screen.resolve())
    assert json.loads((tmp_path / "decision.json").read_text()) == result


def test_selection_explains_rejection_and_ignores_cached_approval(tmp_path):
    base = write_run(tmp_path / "base", ["baseline"], 3)
    screen = write_run(tmp_path / "screen", ["baseline", "unet"], 1)
    path = screen / "jobs.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["status"] = "oom"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (screen / "eligibility.json").write_text('{"approved": true}')
    with pytest.raises(ValueError, match="oom"):
        select_placement(base, screen, "M2-U", tmp_path / "decision.json")
    assert not (tmp_path / "decision.json").exists()


def test_selection_rejects_unknown_and_existing_decision(tmp_path):
    base = write_run(tmp_path / "base", ["baseline"], 3)
    screen = write_run(tmp_path / "screen", ["baseline", "unet"], 1)
    with pytest.raises(ValueError, match="not present"):
        select_placement(base, screen, "M2-R", tmp_path / "decision.json")
    decision = tmp_path / "decision.json"
    decision.write_text("original")
    with pytest.raises(FileExistsError):
        select_placement(base, screen, "M0", decision)
    assert decision.read_text() == "original"


def test_cache_selection_recomputes_predictions(tmp_path):
    from PIL import Image

    from btp.workflow import select_cache

    base = write_run(tmp_path / "base", ["baseline"], 3)
    screen = write_run(tmp_path / "screen", ["baseline", "unet"], 1)
    cache = write_run(tmp_path / "cache", ["baseline", "unet"], 1)
    config = json.loads((cache / "config.json").read_text())
    config["variants"][0].update(policy="M2-U")
    config["variants"][1].update(cache_policy="never")
    (cache / "config.json").write_text(json.dumps(config))
    rows = [json.loads(x) for x in (cache / "jobs.jsonl").read_text().splitlines()]
    for row in rows:
        row["policy"] = "M2-U"
        row["cache_policy"] = "guidance" if row["variant"] == "baseline" else "never"
    (cache / "jobs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = select_cache(base, screen, cache, "M2-U", "never", tmp_path / "ok.json")
    assert len(result["differences"]) == 2
    Image.new("RGB", (16, 16), (100, 100, 100)).save(
        cache / "job-00002/final/input.png"
    )
    (cache / "summary.json").write_text(
        '{"common_success_pairs":2,"output_differences":[]}'
    )
    with pytest.raises(ValueError, match="variation"):
        select_cache(base, screen, cache, "M2-U", "never", tmp_path / "bad.json")


def test_current_source_must_match_restored_evidence(tmp_path):
    from btp.evaluation import sha256
    from btp.workflow import verify_current_provenance

    source, data = tmp_path / "source.json", tmp_path / "manifest.json"
    source.write_text("source-v1")
    data.write_text("data-v1")
    protocol = {
        "source_manifest_sha256": sha256(source),
        "manifest_sha256": sha256(data),
    }
    verify_current_provenance(protocol, source, data)
    source.write_text("source-v2")
    with pytest.raises(ValueError, match="source_manifest"):
        verify_current_provenance(protocol, source, data)
