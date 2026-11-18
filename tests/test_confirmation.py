import pytest

from btp.confirmation import paired_summary


def test_pairs_by_scene_then_bootstraps_scenes():
    baseline = [
        {"scene_id": s, "seed": 42, "repeat": r, "value": 10.0}
        for s in ("a", "b")
        for r in (0, 1)
    ]
    candidate = [dict(row, value=8.0) for row in reversed(baseline)]
    result = paired_summary(
        baseline,
        candidate,
        planned_runs=[(r["scene_id"], r["seed"], r["repeat"]) for r in baseline],
        seed=42,
        samples=100,
    )
    assert result["mean_delta"] == -2
    assert result["ci95"] == [-2, -2]
    assert result["independent_scenes"] == 2
    assert result["matched_runs"] == 4


def test_missing_duplicate_and_nonfinite_rejected():
    rows = [{"scene_id": "a", "seed": 42, "repeat": 0, "value": 1.0}]
    for bad in ([], rows * 2, [dict(rows[0], value=float("nan"))]):
        with pytest.raises(ValueError):
            paired_summary(rows, bad, planned_runs=[("a", 42, 0)])


def test_both_tables_missing_planned_scene_rejected():
    rows = [{"scene_id": "a", "seed": 42, "repeat": 0, "value": 1.0}]
    with pytest.raises(ValueError):
        paired_summary(rows, rows, planned_runs=[("a", 42, 0), ("b", 42, 0)])
