"""Bounded, training-only offline targets with immutable identity and recovery.

This module never selects a corpus, downloads weights, or launches training.
Callers must obtain approval for the exact input list and compute budget first.
"""

import hashlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from btp.evaluation import confined_path, sha256
from btp.training_data import audit_training_manifest


@dataclass(frozen=True)
class TargetBudget:
    """Total inference attempts and cumulative seconds across all resumptions."""

    max_attempts: int = 100
    max_seconds: float = 7200

    def __post_init__(self) -> None:
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 100:
            raise ValueError("target shards allow 1..100 total attempts")
        if not math.isfinite(self.max_seconds) or not 0 < self.max_seconds <= 7200:
            raise ValueError("target shard time must be finite and at most two hours")


def shard_for(source_scene_id: str, shard_count: int) -> int:
    """Stable source-scene assignment keeps crops/variants in one shard."""
    if not source_scene_id or type(shard_count) is not int or shard_count < 1:
        raise ValueError("scene ID and positive shard count required")
    digest = hashlib.sha256(source_scene_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def generate_targets(
    *,
    document: dict[str, Any],
    root: Path,
    output: Path,
    teacher: Any,
    approved_pair_ids: Sequence[str],
    budget: TargetBudget,
    evaluation_scenes: Sequence[dict[str, Any]],
    seed: int,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """Generate the exact approved training list and archive each attempt immediately.

    Audit includes clean-file identities/geometry, but only the hazy path is given
    to the teacher. Failed pairs remain failed unless retry_failed is explicit.
    Crashed running items become interrupted and may resume in a fresh attempt
    directory; previous files are retained. Budget checks occur between images:
    one inference can overrun the deadline. Hard-killed time cannot be measured,
    so an interrupted attempt conservatively consumes the remaining time budget
    unless its completed duration was already saved in the manifest.
    """
    root, output = Path(root), Path(output)
    if not evaluation_scenes:
        raise ValueError("evaluation scene inventory is required for leakage audit")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be uint32")
    audit_training_manifest(document, root, evaluation_scenes=evaluation_scenes)
    if not approved_pair_ids or len(set(approved_pair_ids)) != len(approved_pair_ids):
        raise ValueError("approved pair list must be nonempty and unique")
    by_id = {row["pair_id"]: row for row in document["scenes"]}
    if any(
        pair_id not in by_id or by_id[pair_id]["split"] != "train"
        for pair_id in approved_pair_ids
    ):
        raise ValueError("only explicit training pair IDs may generate targets")
    lock = {
        "teacher_identity": teacher.identity,
        "manifest_sha256": hashlib.sha256(
            json.dumps(document, sort_keys=True).encode()
        ).hexdigest(),
        "approved_pair_ids": list(approved_pair_ids),
        "seed": seed,
        "budget": asdict(budget),
        "evaluation_sha256": hashlib.sha256(
            json.dumps(list(evaluation_scenes), sort_keys=True).encode()
        ).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        state = json.loads(path.read_text())
        if state.get("schema_version") != 1 or state.get("lock") != lock:
            raise ValueError(
                "target shard lock mismatch; choose a new output directory"
            )
    else:
        state = {
            "schema_version": 1,
            "lock": lock,
            "attempts": 0,
            "elapsed_seconds": 0.0,
            "items": {
                pair_id: {"status": "pending", "history": []}
                for pair_id in approved_pair_ids
            },
        }
        _write(path, state)
    if (
        type(state.get("attempts")) is not int
        or not 0 <= state["attempts"] <= budget.max_attempts
        or not math.isfinite(state.get("elapsed_seconds", float("nan")))
        or state["elapsed_seconds"] < 0
        or set(state["items"]) != set(approved_pair_ids)
    ):
        raise ValueError("invalid target shard progress")
    started, previous = time.monotonic(), state["elapsed_seconds"]

    def save() -> None:
        state["elapsed_seconds"] = previous + time.monotonic() - started
        _write(path, state)

    for pair_id in approved_pair_ids:
        item, row = state["items"][pair_id], by_id[pair_id]
        image_seed = int.from_bytes(
            hashlib.sha256(
                f"{seed}:{row['source_scene_id']}:{pair_id}".encode()
            ).digest()[:4],
            "big",
        )
        expected_provenance = {
            "teacher_identity": teacher.identity,
            "input_sha256": row["hazy_sha256"],
            "seed": image_seed,
            "status": "completed",
        }
        if item["status"] == "completed":
            target = confined_path(output, item["target"])
            if not target.is_file() or sha256(target) != item["target_sha256"]:
                raise ValueError(
                    "completed target hash mismatch; preserve and inspect shard"
                )
            record_path = confined_path(output, item.get("record"))
            if not record_path.is_file() or sha256(record_path) != item.get(
                "record_sha256"
            ):
                raise ValueError("completed target record hash mismatch")
            stored_record = json.loads(record_path.read_text())
            if (
                any(stored_record.get(k) != v for k, v in expected_provenance.items())
                or stored_record.get("output_sha256") != item["target_sha256"]
            ):
                raise ValueError("completed target record provenance mismatch")
            continue
        if item["status"] == "running":
            item["history"].append(
                {
                    "status": "interrupted",
                    "error": "previous process did not finalize item",
                }
            )
            item["status"] = "interrupted"
            # Reserve the full time budget before launching an attempt; a crash
            # leaves it reserved. Completed attempts replace it with measured time.
            if state.get("time_reserved", False):
                previous = budget.max_seconds
                state["time_reserved"] = False
        if item["status"] == "failed" and not retry_failed:
            continue
        if (
            state["attempts"] >= budget.max_attempts
            or previous + time.monotonic() - started >= budget.max_seconds
        ):
            save()
            break
        if getattr(teacher, "closed", False):
            item["status"] = "unexecuted"
            item["reason"] = "teacher worker closed; explicit restart required"
            save()
            continue
        attempt_dir = output / f"attempt-{state['attempts']:04d}"
        if attempt_dir.exists():
            raise ValueError(
                "unexpected attempt directory; inspect preserved partial state"
            )
        item["status"] = "running"
        state["attempts"] += 1
        state["time_reserved"] = True
        save()
        result: dict[str, Any]
        try:
            result = teacher.predict(
                confined_path(root, row["hazy"]), attempt_dir, image_seed
            )
            if result["status"] != "completed":
                raise RuntimeError(result.get("error", "teacher prediction failed"))
            if any(result.get(k) != v for k, v in expected_provenance.items()):
                raise ValueError("teacher record provenance mismatch")
            record_path = attempt_dir / "record.json"
            if (
                not record_path.is_file()
                or json.loads(record_path.read_text()) != result
            ):
                raise ValueError(
                    "teacher record missing or differs from returned record"
                )
            # The resident contract fixes prediction.png; don't trust arbitrary
            # returned paths to redirect hashes outside this shard.
            target = attempt_dir / "prediction.png"
            if sha256(target) != result.get("output_sha256"):
                raise ValueError("teacher output hash mismatch")
            with (
                Image.open(target) as prediction,
                Image.open(confined_path(root, row["hazy"])) as hazy,
            ):
                prediction.load()
                if prediction.mode != "RGB" or prediction.size != hazy.size:
                    raise ValueError("teacher target RGB/geometry mismatch")
            item.update(
                status="completed",
                target=str(target.relative_to(output)),
                target_sha256=sha256(target),
                source_scene_id=row["source_scene_id"],
                hazy_sha256=row["hazy_sha256"],
                clean_sha256=row["clean_sha256"],
                seed=image_seed,
                record=str(record_path.relative_to(output)),
                record_sha256=sha256(record_path),
            )
            item["history"].append(
                {
                    "status": "completed",
                    "attempt": state["attempts"],
                    "record": str((attempt_dir / "record.json").relative_to(output)),
                }
            )
        except Exception as error:
            item["status"] = "failed"
            item["history"].append(
                {
                    "status": "failed",
                    "attempt": state["attempts"],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        finally:
            state["time_reserved"] = False
            save()
    save()
    return state
