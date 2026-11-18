"""Fail-closed research contracts and held-out boundaries."""

import pytest

from btp.protocols import (
    DecisionRecord,
    ExperimentSpec,
    RunRecord,
    validate_run_records,
)


def experiment(**changes):
    values = dict(
        experiment_id="trial-a",
        parent_id=None,
        source_commit="a" * 40,
        source_bundle_sha256="b" * 64,
        data_manifest_sha256="c" * 64,
        protocol_id="btp-research-v1",
        split="development",
        scene_ids=("scene-1",),
        seeds=(42,),
        geometry="native",
        model_id="student-a0",
        checkpoint_sha256="d" * 64,
        policy="student",
        precision="float32",
        budget={"max_jobs": 1, "max_seconds": 60},
        output_directory="runs/trial-a",
        gate_family="quality",
        purpose="screening",
    )
    values.update(changes)
    return ExperimentSpec(**values)


def record(**changes):
    values = dict(
        experiment_id="trial-a",
        scene_id="scene-1",
        seed=42,
        repeat=0,
        input_sha256="e" * 64,
        output_sha256="f" * 64,
        status="complete",
        source_bundle_sha256="b" * 64,
        data_manifest_sha256="c" * 64,
        model_seconds=0.1,
        warm_end_to_end_seconds=0.2,
    )
    values.update(changes)
    return RunRecord(**values)


def test_contract_roundtrip_and_manifest_membership():
    spec = experiment()
    assert ExperimentSpec.from_dict(spec.to_dict()) == spec
    spec.validate_manifest(
        {"scenes": [{"scene_id": "scene-1", "split": "development"}]}
    )
    with pytest.raises(ValueError, match="split"):
        spec.validate_manifest({"scenes": [{"scene_id": "scene-1", "split": "test"}]})


@pytest.mark.parametrize(
    "changes",
    [
        {"source_commit": "unknown"},
        {"source_bundle_sha256": "bad"},
        {"scene_ids": ("scene-1", "scene-1")},
        {"seeds": (-1,)},
        {"output_directory": "../escape"},
        {"budget": {"max_jobs": 0, "max_seconds": 60}},
        {"split": "test"},
        {"purpose": "training", "split": "development"},
        {
            "purpose": "heldout",
            "split": "test",
            "finalist_sha256": "d" * 64,
            "tuning": True,
        },
    ],
)
def test_experiment_rejects_ineligible_or_incomplete_contract(changes):
    with pytest.raises(ValueError):
        experiment(**changes)


def test_heldout_requires_finalist_and_forbids_selection():
    spec = experiment(split="test", purpose="heldout", finalist_sha256="d" * 64)
    assert not spec.tuning
    with pytest.raises(ValueError, match="held-out"):
        DecisionRecord(
            decision_id="choice",
            rule_version="v1",
            selected_candidate="a",
            rejected_alternatives={},
            evidence={"report.json": "a" * 64},
            data_scope="test",
            author="reviewer",
            timestamp="2026-11-16T00:00:00Z",
            action="select",
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"model_seconds": float("nan")},
        {"load_seconds": -1},
        {"status": "complete", "output_sha256": None},
        {"status": "failed"},
        {"status": "failed", "error": "OOM", "output_sha256": "f" * 64},
        {"allocated_peak_bytes": -1},
    ],
)
def test_run_rejects_invalid_measurement_or_status(changes):
    with pytest.raises(ValueError):
        record(**changes)


def test_run_coverage_rejects_stale_duplicate_and_missing_results():
    spec = experiment()
    assert validate_run_records(spec, [record()]) == {"complete": 1}
    for rows in ([], [record(), record()], [record(source_bundle_sha256="a" * 64)]):
        with pytest.raises(ValueError):
            validate_run_records(spec, rows)
    failed = record(status="failed", error="OOM", output_sha256=None)
    assert validate_run_records(spec, [failed]) == {"failed": 1}
    assert failed.sampling_seconds is None


def test_schema_version_and_unknown_fields_fail_closed():
    data = experiment().to_dict()
    data["schema_version"] = 2
    with pytest.raises(ValueError, match="schema"):
        ExperimentSpec.from_dict(data)
    data["schema_version"] = 1
    data["surprise"] = True
    with pytest.raises(ValueError):
        ExperimentSpec.from_dict(data)


def test_experiment_budget_cannot_be_changed_after_validation():
    spec = experiment()
    with pytest.raises(TypeError):
        spec.budget["max_jobs"] = 0


def test_protocol_load_checks_frozen_source_membership_and_hash():
    from pathlib import Path

    from btp.protocols import load_protocol

    protocol = load_protocol(
        Path("configs/research/protocol-v1.json"),
        Path("configs/evaluation/nh_haze_manifest.json"),
    )
    assert len(protocol["evaluation"]["development_scene_ids"]) == 10
    assert len(protocol["evaluation"]["heldout_scene_ids"]) == 45


def test_decision_evidence_is_immutable():
    decision = DecisionRecord(
        decision_id="decision-1",
        rule_version="v1",
        selected_candidate="a",
        rejected_alternatives={},
        evidence={"result.json": "a" * 64},
        data_scope="development",
        author="analyst",
        timestamp="2026-11-17T00:00:00Z",
        action="select",
    )
    with pytest.raises(TypeError):
        decision.evidence["result.json"] = "b" * 64
    assert DecisionRecord.from_dict(decision.to_dict()) == decision
