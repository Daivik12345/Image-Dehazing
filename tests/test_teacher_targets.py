import json

import pytest
from PIL import Image

from btp.evaluation import sha256
from btp.teacher_targets import TargetBudget, generate_targets, shard_for


class Teacher:
    identity = "fixture-teacher"
    calls = 0
    closed = False

    def predict(self, path, output, seed):
        self.calls += 1
        output.mkdir()
        prediction = output / "prediction.png"
        Image.open(path).save(prediction)
        record = {
            "teacher_identity": self.identity,
            "input_sha256": sha256(path),
            "seed": seed,
            "status": "completed",
            "output_path": str(prediction),
            "output_sha256": sha256(prediction),
        }
        (output / "record.json").write_text(json.dumps(record))
        return record


def fixture(tmp_path):
    rows = []
    for i in range(2):
        for role in ("hazy", "clean"):
            Image.new("RGB", (3, 4), (i * 20, 30 if role == "clean" else 40, 0)).save(
                tmp_path / f"{role}{i}.png"
            )
        rows.append(
            {
                "pair_id": str(i),
                "source_scene_id": str(i),
                "split": "train",
                "registration_status": "verified",
                "source_size": [3, 4],
                "hazy": f"hazy{i}.png",
                "clean": f"clean{i}.png",
                "hazy_sha256": sha256(tmp_path / f"hazy{i}.png"),
                "clean_sha256": sha256(tmp_path / f"clean{i}.png"),
            }
        )
    return {
        "schema_version": 1,
        "dataset": "fixture",
        "release": "v1",
        "source_archive": "fixture",
        "archive_sha256": "0" * 64,
        "license": "fixture",
        "scenes": rows,
    }


def test_targets_idempotent_and_hash_locked(tmp_path):
    document = fixture(tmp_path)
    teacher = Teacher()
    kwargs = dict(
        document=document,
        root=tmp_path,
        output=tmp_path / "targets",
        teacher=teacher,
        approved_pair_ids=["0", "1"],
        budget=TargetBudget(4, 30),
        evaluation_scenes=[{"scene_id": "heldout"}],
        seed=42,
    )
    first = generate_targets(**kwargs)
    assert teacher.calls == 2
    assert [r["status"] for r in first["items"].values()] == ["completed", "completed"]
    generate_targets(**kwargs)
    assert teacher.calls == 2
    assert shard_for("scene", 7) == shard_for("scene", 7)
    assert 0 <= shard_for("scene", 7) < 7
    document["release"] = "changed"
    with pytest.raises(ValueError, match="lock"):
        generate_targets(**kwargs)


def test_partial_recovery_and_failure_retention(tmp_path):
    document = fixture(tmp_path)
    teacher = Teacher()
    kwargs = dict(
        document=document,
        root=tmp_path,
        output=tmp_path / "targets",
        teacher=teacher,
        approved_pair_ids=["0"],
        budget=TargetBudget(4, 30),
        evaluation_scenes=[{"scene_id": "heldout"}],
        seed=42,
    )
    state = generate_targets(**kwargs)
    state["items"]["0"]["status"] = "running"
    (tmp_path / "targets" / "manifest.json").write_text(json.dumps(state))
    recovered = generate_targets(**kwargs)
    assert teacher.calls == 2
    assert recovered["attempts"] == 2
    assert any(
        row["status"] == "interrupted" for row in recovered["items"]["0"]["history"]
    )

    class Broken(Teacher):
        def predict(self, *args):
            raise RuntimeError("broken fixture")

    kwargs.update(output=tmp_path / "failed", teacher=Broken())
    state = generate_targets(**kwargs)
    assert state["items"]["0"]["status"] == "failed"
    assert "broken fixture" in state["items"]["0"]["history"][0]["error"]


def test_hard_interruption_does_not_reset_budget(tmp_path):
    document = fixture(tmp_path)
    teacher = Teacher()
    kwargs = dict(
        document=document,
        root=tmp_path,
        output=tmp_path / "targets",
        teacher=teacher,
        approved_pair_ids=["0", "1"],
        budget=TargetBudget(4, 30),
        evaluation_scenes=[{"scene_id": "heldout"}],
        seed=42,
    )
    state = generate_targets(**kwargs)
    state["items"]["0"]["status"] = "running"
    state["time_reserved"] = True
    (tmp_path / "targets" / "manifest.json").write_text(json.dumps(state))
    recovered = generate_targets(**kwargs)
    assert teacher.calls == 2
    assert recovered["elapsed_seconds"] >= 30
    assert recovered["items"]["0"]["status"] == "interrupted"


def test_rejects_validation_pair_and_corrupt_completed_target(tmp_path):
    document = fixture(tmp_path)
    teacher = Teacher()
    kwargs = dict(
        document=document,
        root=tmp_path,
        output=tmp_path / "targets",
        teacher=teacher,
        approved_pair_ids=["0"],
        budget=TargetBudget(4, 30),
        evaluation_scenes=[{"scene_id": "heldout"}],
        seed=42,
    )
    state = generate_targets(**kwargs)
    (tmp_path / "targets" / state["items"]["0"]["target"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        generate_targets(**kwargs)
    document["scenes"][0]["split"] = "validation"
    kwargs["output"] = tmp_path / "validation"
    with pytest.raises(ValueError, match="training pair"):
        generate_targets(**kwargs)


@pytest.mark.parametrize(
    "field", ["teacher_identity", "input_sha256", "seed", "missing_record"]
)
def test_rejects_wrong_provenance_or_missing_record(tmp_path, field):
    document = fixture(tmp_path)

    class Wrong(Teacher):
        def predict(self, path, output, seed):
            record = super().predict(path, output, seed)
            if field == "missing_record":
                (output / "record.json").unlink()
            else:
                record[field] = "wrong"
                (output / "record.json").write_text(json.dumps(record))
            return record

    state = generate_targets(
        document=document,
        root=tmp_path,
        output=tmp_path / "targets",
        teacher=Wrong(),
        approved_pair_ids=["0"],
        budget=TargetBudget(2, 30),
        evaluation_scenes=[{"scene_id": "heldout"}],
        seed=42,
    )
    assert state["items"]["0"]["status"] == "failed"


def test_reuse_rechecks_record_hash(tmp_path):
    document = fixture(tmp_path)
    kwargs = dict(
        document=document,
        root=tmp_path,
        output=tmp_path / "targets",
        teacher=Teacher(),
        approved_pair_ids=["0"],
        budget=TargetBudget(2, 30),
        evaluation_scenes=[{"scene_id": "heldout"}],
        seed=42,
    )
    generate_targets(**kwargs)
    (tmp_path / "targets" / "attempt-0000" / "record.json").write_text("{}")
    with pytest.raises(ValueError, match="record"):
        generate_targets(**kwargs)
