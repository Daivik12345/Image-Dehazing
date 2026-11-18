"""Versioned research records with explicit provenance and held-out boundaries.

These contracts validate record structure and coverage, not scientific success.
Quality changes and numerical systems equivalence use separate gate families.
"""

import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from btp.evaluation import sha256


def _hash(value: str, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a SHA256 digest")


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")


def _relative(value: str) -> None:
    _text(value, "path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError("Artifact path must be relative and confined")


class _Record:
    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible data with an explicit schema version."""
        result = {}
        for field in fields(self):
            value = getattr(self, field.name)
            result[field.name] = dict(value) if isinstance(value, Mapping) else value
        result["schema_version"] = 1
        return result

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "_Record":
        """Read only schema v1; reject unknown fields rather than reinterpret them."""
        values = dict(document)
        if values.pop("schema_version", None) != 1:
            raise ValueError("Unsupported record schema; explicit migration required")
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError(f"Invalid {cls.__name__} fields: {error}") from error


@dataclass(frozen=True)
class ExperimentSpec(_Record):
    """Immutable run identity; a budget is a cap, never compute authorization."""

    experiment_id: str
    parent_id: str | None
    source_commit: str
    source_bundle_sha256: str
    data_manifest_sha256: str
    protocol_id: str
    split: str
    scene_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    geometry: str
    model_id: str
    checkpoint_sha256: str
    policy: str
    precision: str
    budget: dict[str, int | float]
    output_directory: str
    gate_family: str
    purpose: str
    finalist_sha256: str | None = None
    tuning: bool = False

    def __post_init__(self) -> None:
        for name in ("experiment_id", "protocol_id", "geometry", "model_id", "policy"):
            _text(getattr(self, name), name)
        if self.parent_id is not None:
            _text(self.parent_id, "parent_id")
            if self.parent_id == self.experiment_id:
                raise ValueError("Experiment cannot parent itself")
        if not isinstance(self.source_commit, str) or not re.fullmatch(
            "[0-9a-f]{40}|[0-9a-f]{64}", self.source_commit
        ):
            raise ValueError("source_commit must identify an exact commit")
        for name in (
            "source_bundle_sha256",
            "data_manifest_sha256",
            "checkpoint_sha256",
        ):
            _hash(getattr(self, name), name)
        if isinstance(self.scene_ids, str) or isinstance(self.seeds, str):
            raise ValueError("Scenes and seeds must be sequences")
        object.__setattr__(self, "scene_ids", tuple(self.scene_ids))
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if not self.scene_ids or len(set(self.scene_ids)) != len(self.scene_ids):
            raise ValueError("Scene IDs must be unique and nonempty")
        for scene in self.scene_ids:
            _text(scene, "scene ID")
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(type(s) is not int or s < 0 for s in self.seeds)
        ):
            raise ValueError("Seeds must be unique nonnegative integers")
        if self.split not in {"train", "validation", "development", "test"}:
            raise ValueError("Invalid split")
        if self.gate_family not in {"systems", "quality"}:
            raise ValueError("Choose distinct systems or quality gate family")
        if self.purpose not in {"training", "screening", "confirmation", "heldout"}:
            raise ValueError("Invalid experiment purpose")
        if self.precision not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Explicit supported precision required")
        if self.purpose == "training" and self.split != "train":
            raise ValueError("Training may only use corpus train scenes")
        if self.split == "test" or self.purpose == "heldout":
            if self.split != "test" or self.purpose != "heldout" or self.tuning:
                raise ValueError(
                    "Held-out evaluation forbids tuning and requires heldout purpose"
                )
            _hash(self.finalist_sha256, "finalist_sha256")
        elif self.finalist_sha256 is not None:
            _hash(self.finalist_sha256, "finalist_sha256")
        if type(self.tuning) is not bool:
            raise ValueError("tuning must be boolean")
        if not isinstance(self.budget, dict) or set(self.budget) != {
            "max_jobs",
            "max_seconds",
        }:
            raise ValueError("Budget requires max_jobs and max_seconds")
        if type(self.budget["max_jobs"]) is not int or self.budget["max_jobs"] <= 0:
            raise ValueError("max_jobs must be a positive integer")
        seconds = self.budget["max_seconds"]
        if (
            type(seconds) not in {int, float}
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError("max_seconds must be finite and positive")
        if self.budget["max_jobs"] < len(self.scene_ids) * len(self.seeds):
            raise ValueError("Job budget cannot cover planned scenes and seeds")
        object.__setattr__(self, "budget", MappingProxyType(dict(self.budget)))
        _relative(self.output_directory)

    def validate_manifest(self, manifest: Mapping[str, Any]) -> None:
        """Enforce exact source scene membership against an already hashed manifest."""
        rows = manifest.get("scenes", [])
        membership = {}
        for row in rows:
            scene = row.get("source_scene_id", row.get("scene_id"))
            if scene is None or membership.setdefault(
                scene, row.get("split")
            ) != row.get("split"):
                raise ValueError("Manifest source scene crosses splits")
        if any(membership.get(scene) != self.split for scene in self.scene_ids):
            raise ValueError("Requested scene missing or assigned to another split")


@dataclass(frozen=True)
class RunRecord(_Record):
    """Per-scene measurements; unavailable phases stay null, failures stay visible."""

    experiment_id: str
    scene_id: str
    seed: int
    repeat: int
    input_sha256: str
    output_sha256: str | None
    status: str
    source_bundle_sha256: str
    data_manifest_sha256: str
    error: str | None = None
    load_seconds: float | None = None
    preprocessing_seconds: float | None = None
    conditioning_seconds: float | None = None
    sampling_seconds: float | None = None
    model_seconds: float | None = None
    postprocess_seconds: float | None = None
    warm_end_to_end_seconds: float | None = None
    allocated_peak_bytes: int | None = None
    reserved_peak_bytes: int | None = None
    host_rss_bytes: int | None = None

    def __post_init__(self) -> None:
        _text(self.experiment_id, "experiment ID")
        _text(self.scene_id, "scene ID")
        for name in ("seed", "repeat"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("input_sha256", "source_bundle_sha256", "data_manifest_sha256"):
            _hash(getattr(self, name), name)
        if self.status == "complete":
            _hash(self.output_sha256, "output_sha256")
            if self.error is not None:
                raise ValueError("Completed run cannot contain an error")
        elif self.status in {"failed", "unexecuted"}:
            _text(self.error, "failure reason")
            if self.output_sha256 is not None:
                raise ValueError(
                    "Failed/unexecuted run cannot declare a completed output"
                )
        else:
            raise ValueError("Invalid run status")
        for name, value in asdict(self).items():
            if name.endswith("_seconds") and value is not None:
                if (
                    type(value) not in {int, float}
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError("Timings must be finite and nonnegative")
            if name.endswith("_bytes") and value is not None:
                if type(value) is not int or value < 0:
                    raise ValueError("Memory measurements must be nonnegative integers")


@dataclass(frozen=True)
class DecisionRecord(_Record):
    """Evidence-linked decision; held-out results may be reported, never selected on."""

    decision_id: str
    rule_version: str
    selected_candidate: str | None
    rejected_alternatives: dict[str, str]
    evidence: dict[str, str]
    data_scope: str
    author: str
    timestamp: str
    action: str
    reviewer: str | None = None

    def __post_init__(self) -> None:
        for name in ("decision_id", "rule_version", "author"):
            _text(getattr(self, name), name)
        if self.data_scope not in {"train", "validation", "development", "test"}:
            raise ValueError("Invalid decision data scope")
        if self.action not in {"select", "reject", "inconclusive", "report"}:
            raise ValueError("Invalid decision action")
        if self.data_scope == "test" and self.action != "report":
            raise ValueError("Cannot select or tune using held-out results")
        if self.action == "select" and not self.selected_candidate:
            raise ValueError("Selection must name candidate")
        if self.selected_candidate is not None:
            _text(self.selected_candidate, "selected candidate")
        if not self.evidence:
            raise ValueError("Decisions require immutable evidence references")
        for path, digest in self.evidence.items():
            _relative(path)
            _hash(digest, "evidence hash")
        for candidate, reason in self.rejected_alternatives.items():
            _text(candidate, "rejected candidate")
            _text(reason, "rejection reason")
        if self.selected_candidate in self.rejected_alternatives:
            raise ValueError("Selected candidate cannot also be rejected")
        try:
            stamp = datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))
        except (TypeError, ValueError, AttributeError) as error:
            raise ValueError("Timestamp must be ISO8601") from error
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))
        object.__setattr__(
            self,
            "rejected_alternatives",
            MappingProxyType(dict(self.rejected_alternatives)),
        )
        if stamp.tzinfo is None:
            raise ValueError("Timestamp must include timezone")


def validate_run_records(
    spec: ExperimentSpec,
    records: Sequence[RunRecord],
    *,
    repeats: int = 1,
) -> dict[str, int]:
    """Reject stale/mixed evidence and incomplete coverage, retaining failures."""
    if type(repeats) is not int or repeats <= 0:
        raise ValueError("repeats must be positive")
    expected = {
        (scene, seed, repeat)
        for scene in spec.scene_ids
        for seed in spec.seeds
        for repeat in range(repeats)
    }
    if len(expected) > spec.budget["max_jobs"]:
        raise ValueError("Planned repeats exceed job budget")
    seen = set()
    for record in records:
        if (
            record.experiment_id != spec.experiment_id
            or record.source_bundle_sha256 != spec.source_bundle_sha256
            or record.data_manifest_sha256 != spec.data_manifest_sha256
        ):
            raise ValueError("Stale or mixed experiment provenance")
        identity = (record.scene_id, record.seed, record.repeat)
        if identity not in expected or identity in seen:
            raise ValueError("Unexpected or duplicate run identity")
        seen.add(identity)
    if seen != expected:
        raise ValueError("Incomplete planned job coverage")
    return dict(Counter(record.status for record in records))


def load_protocol(path: Path, frozen_manifest: Path) -> dict[str, Any]:
    """Load registered rules only when exact frozen source membership agrees.

    Geometry derivatives keep these memberships but have their own manifest hash.
    This verifies the registration, not evidence eligibility or compute approval.
    """
    protocol = json.loads(path.read_text())
    if (
        protocol.get("schema_version") != 1
        or protocol.get("protocol_id") != "btp-research-v1"
    ):
        raise ValueError("Unsupported protocol schema or ID")
    evaluation = protocol["evaluation"]
    if sha256(frozen_manifest) != evaluation["frozen_manifest_sha256"]:
        raise ValueError("Frozen evaluation manifest hash mismatch")
    manifest = json.loads(frozen_manifest.read_text())
    membership = {}
    for row in manifest["scenes"]:
        if row["scene_id"] in membership:
            raise ValueError("Duplicate frozen source scene")
        membership[row["scene_id"]] = row["split"]
    development = evaluation["development_scene_ids"]
    heldout = evaluation["heldout_scene_ids"]
    if (
        len(development) != 10
        or len(set(development)) != 10
        or len(heldout) != 45
        or len(set(heldout)) != 45
        or set(development) & set(heldout)
        or membership
        != {
            **dict.fromkeys(development, "development"),
            **dict.fromkeys(heldout, "test"),
        }
    ):
        raise ValueError("Frozen split membership differs")
    if protocol["systems_gate"]["implementation"] != "btp.optimization.screen_report":
        raise ValueError("Systems gate implementation changed")
    if protocol["heldout_policy"]["allow_tuning"] is not False:
        raise ValueError("Held-out tuning is forbidden")
    return protocol
