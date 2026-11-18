"""Bounded float32 AdamW training with deterministic serial sampling and safe resume.

This loop consumes audited in-memory training pairs. Corpus acquisition, augmentation,
validation loaders and GPU-job authorization belong to the caller. No training starts
on import; defaults describe a proposed pilot, not permission to run it.
"""

import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from btp.student.inference import file_sha256
from btp.student.losses import restoration_loss
from btp.student.model import CompactDehaze, StudentConfig, validate_rgb


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable total budget, including any profiling updates and previous runs."""

    max_updates: int = 2000
    max_seconds: float = 7200
    accumulation: int = 4
    warmup_updates: int = 100
    learning_rate: float = 1e-4
    minimum_lr: float = 1e-6
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    seed: int = 42
    checkpoint_interval: int = 250

    def __post_init__(self) -> None:
        for value in (
            self.max_updates,
            self.accumulation,
            self.warmup_updates,
            self.checkpoint_interval,
        ):
            if type(value) is not int or value < 1:
                raise ValueError(
                    "update counts and accumulation must be positive integers"
                )
        values = (
            self.max_seconds,
            self.learning_rate,
            self.minimum_lr,
            self.gradient_clip,
        )
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(
                "time, rates and gradient clip must be finite and positive"
            )
        if (
            self.minimum_lr > self.learning_rate
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("invalid learning rate bounds or weight decay")


@dataclass
class TrainingPair:
    """One audited microbatch (size one); aliases must use canonical source_scene_id."""

    pair_id: str
    source_scene_id: str
    split: str
    hazy: Tensor
    clean: Tensor
    teacher: Tensor | None = None


class BoundedTrainer:
    """Resumable serial sampler; use forbidden scene IDs from the evaluation registry."""

    @classmethod
    def create(
        cls,
        config: TrainingConfig,
        pairs: Sequence[TrainingPair],
        *,
        model_config: StudentConfig | None = None,
        provenance: dict[str, str],
        forbidden_scene_ids: Sequence[str] = (),
    ) -> "BoundedTrainer":
        """Seed before constructing a fresh model for matched random-initialization runs.

        Use the regular constructor to preserve an explicitly supplied pretrained
        model. Its actual initial weight hash is recorded without resetting weights.
        """
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        return cls(
            CompactDehaze(model_config),
            config,
            pairs,
            provenance=provenance,
            forbidden_scene_ids=forbidden_scene_ids,
        )

    def __init__(
        self,
        model: CompactDehaze,
        config: TrainingConfig,
        pairs: Sequence[TrainingPair],
        *,
        provenance: dict[str, str],
        forbidden_scene_ids: Sequence[str] = (),
    ):
        if (
            not pairs
            or not provenance.get("dataset_version")
            or not provenance.get("source_commit")
        ):
            raise ValueError(
                "training requires pairs, dataset_version and source_commit"
            )
        ids: set[str] = set()
        hashes: set[str] = set()
        fingerprint = hashlib.sha256()
        for pair in pairs:
            if not pair.pair_id or not pair.source_scene_id or pair.pair_id in ids:
                raise ValueError("empty or duplicate pair identity")
            if pair.split != "train" or pair.source_scene_id in forbidden_scene_ids:
                raise ValueError("non-training or held-out scene rejected")
            ids.add(pair.pair_id)
            content = hashlib.sha256()
            for tensor in (pair.hazy, pair.clean, pair.teacher):
                if tensor is not None:
                    validate_rgb(tensor)
                    if (
                        tensor.shape != pair.hazy.shape
                        or tensor.shape[0] != 1
                        or tensor.dtype != torch.float32
                    ):
                        raise ValueError(
                            "paired float32 microbatches must have matching shapes and batch one"
                        )
                    content.update(str(tuple(tensor.shape)).encode())
                    content.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            digest = content.hexdigest()
            if digest in hashes:
                raise ValueError("duplicate training pair content")
            hashes.add(digest)
            fingerprint.update(
                f"{pair.pair_id}:{pair.source_scene_id}:{digest}".encode()
            )
        # Own detached snapshots so caller mutation cannot invalidate the audit.
        self.model, self.config = model, config
        self.pairs = tuple(
            TrainingPair(
                pair.pair_id,
                pair.source_scene_id,
                pair.split,
                pair.hazy.detach().clone(),
                pair.clean.detach().clone(),
                pair.teacher.detach().clone() if pair.teacher is not None else None,
            )
            for pair in pairs
        )
        initial = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            initial.update(name.encode())
            initial.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            initial.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        self.initialization_sha256 = initial.hexdigest()
        self.provenance = dict(provenance)
        self.data_sha256 = fingerprint.hexdigest()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
        self.updates = 0
        self.data_position = 0
        self.exposures = 0
        self.elapsed_seconds = 0.0
        self.history: list[dict] = []
        self.best: tuple[float, float, int] | None = None
        random.seed(config.seed)
        torch.manual_seed(config.seed)

    def _rate(self) -> float:
        step = self.updates + 1
        if step <= self.config.warmup_updates:
            return self.config.learning_rate * step / self.config.warmup_updates
        span = max(1, self.config.max_updates - self.config.warmup_updates)
        fraction = (step - self.config.warmup_updates) / span
        return (
            self.config.minimum_lr
            + (self.config.learning_rate - self.config.minimum_lr)
            * (1 + math.cos(math.pi * fraction))
            / 2
        )

    def run(
        self,
        *,
        until_update: int | None = None,
        checkpoint_dir: Path | None = None,
        validate: Callable[[CompactDehaze], tuple[float, float]] | None = None,
    ) -> None:
        """Run up to a bounded absolute update; stop at microbatch boundaries on time.

        A deadline may overrun by one microbatch/validation/save operation. Incomplete
        accumulation is discarded and its sampler position rolled back. Budget time
        still counts. Validation must use training-corpus validation, never held-out data.
        """
        limit = (
            self.config.max_updates
            if until_update is None
            else min(until_update, self.config.max_updates)
        )
        if limit < self.updates:
            raise ValueError("until_update precedes current progress")
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        previous = self.elapsed_seconds

        def expired() -> bool:
            self.elapsed_seconds = previous + time.monotonic() - started
            return self.elapsed_seconds >= self.config.max_seconds

        self.model.train()
        try:
            while self.updates < limit and not expired():
                position = self.data_position
                self.optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                completed = 0
                for _ in range(self.config.accumulation):
                    if expired():
                        break
                    pair = self.pairs[self.data_position % len(self.pairs)]
                    device = next(self.model.parameters()).device
                    teacher = (
                        pair.teacher.to(device) if pair.teacher is not None else None
                    )
                    loss = restoration_loss(
                        self.model(pair.hazy.to(device)), pair.clean.to(device), teacher
                    )
                    if not torch.isfinite(loss):
                        raise ValueError("non-finite training loss")
                    (loss / self.config.accumulation).backward()
                    total_loss += loss.detach().item()
                    self.data_position += 1
                    completed += 1
                if completed != self.config.accumulation or expired():
                    self.data_position = position
                    self.optimizer.zero_grad(set_to_none=True)
                    break
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                    error_if_nonfinite=True,
                )
                rate = self._rate()
                for group in self.optimizer.param_groups:
                    group["lr"] = rate
                self.optimizer.step()
                self.updates += 1
                self.exposures += completed
                self.history.append(
                    {"update": self.updates, "loss": total_loss / completed, "lr": rate}
                )
                if self.updates % self.config.checkpoint_interval == 0:
                    if validate is not None:
                        self.model.eval()
                        with torch.inference_mode():
                            psnr, ssim = validate(self.model)
                        self.model.train()
                        if not math.isfinite(psnr) or not math.isfinite(ssim):
                            raise ValueError("validation returned non-finite metrics")
                        score = (psnr, ssim, -self.updates)
                        if self.best is None or score > self.best:
                            self.best = score
                            expired()
                            if checkpoint_dir is not None:
                                self.save(checkpoint_dir / "best.pt")
                    expired()
                    if checkpoint_dir is not None:
                        self.save(checkpoint_dir / "last.pt")
        finally:
            # Only complete optimizer updates consume sampler positions.
            self.data_position = self.exposures
            self.optimizer.zero_grad(set_to_none=True)
            expired()
            if checkpoint_dir is not None:
                self.save(checkpoint_dir / "last.pt")

    def save(self, path: Path) -> None:
        """Save resume state plus required .budget.json ledger including save duration.

        Copy both files to resume. A crash between writes fails closed on restore.
        The final tiny ledger write itself is outside the recorded duration.
        """
        started = time.monotonic()
        state = {
            "schema": 2,
            "initialization_sha256": self.initialization_sha256,
            "architecture": self.model.config.to_dict(),
            "config": asdict(self.config),
            "provenance": self.provenance,
            "data_sha256": self.data_sha256,
            "weights": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
            "scheduler": {"completed_updates": self.updates},
            "data_position": self.data_position,
            "exposures": self.exposures,
            "elapsed_seconds": self.elapsed_seconds,
            "history": self.history,
            "best": self.best,
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
        }
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)
        digest = file_sha256(path)
        self.elapsed_seconds += time.monotonic() - started
        # The checkpoint cannot include its own serialization duration. A paired
        # atomic ledger accounts for it; missing/stale ledgers fail closed.
        ledger = path.with_suffix(path.suffix + ".budget.json")
        ledger_tmp = ledger.with_name(ledger.name + ".tmp")
        ledger_tmp.write_text(
            json.dumps(
                {
                    "checkpoint_sha256": digest,
                    "elapsed_seconds": self.elapsed_seconds,
                }
            )
        )
        ledger_tmp.replace(ledger)

    def restore(self, path: Path) -> None:
        """Reject changed data/config/provenance and restore optimizer/RNG/progress."""
        path = Path(path)
        ledger_path = path.with_suffix(path.suffix + ".budget.json")
        try:
            ledger = json.loads(ledger_path.read_text())
            if ledger["checkpoint_sha256"] != file_sha256(path):
                raise ValueError("checkpoint budget ledger hash mismatch")
            budget_elapsed = ledger["elapsed_seconds"]
            if not math.isfinite(budget_elapsed) or budget_elapsed < 0:
                raise ValueError("invalid checkpoint budget ledger")
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("missing or invalid checkpoint budget ledger") from exc
        state = torch.load(path, map_location="cpu", weights_only=True)
        expected = {
            "schema": 2,
            "architecture": self.model.config.to_dict(),
            "config": asdict(self.config),
            "provenance": self.provenance,
            "data_sha256": self.data_sha256,
        }
        if any(state.get(k) != value for k, value in expected.items()):
            raise ValueError("resume architecture/config/data/provenance mismatch")
        identity = state.get("initialization_sha256")
        if (
            not isinstance(identity, str)
            or len(identity) != 64
            or any(c not in "0123456789abcdef" for c in identity)
        ):
            raise ValueError("invalid initialization hash")
        if budget_elapsed < state["elapsed_seconds"]:
            raise ValueError("checkpoint budget ledger rolls elapsed time backwards")
        state["elapsed_seconds"] = budget_elapsed
        updates, elapsed = state["updates"], state["elapsed_seconds"]
        if (
            not 0 <= updates <= self.config.max_updates
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError("invalid resume budget counters")
        if (
            state["scheduler"] != {"completed_updates": updates}
            or state["exposures"] != updates * self.config.accumulation
            or state["data_position"] != state["exposures"]
        ):
            raise ValueError("inconsistent resume sampler/scheduler counters")
        self.initialization_sha256 = identity
        self.model.load_state_dict(state["weights"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        for key in (
            "updates",
            "data_position",
            "exposures",
            "elapsed_seconds",
            "history",
            "best",
        ):
            setattr(self, key, state[key])
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if state["cuda_rng"]:
            if not torch.cuda.is_available():
                raise ValueError("CUDA resume requires the original CUDA runtime")
            torch.cuda.set_rng_state_all(state["cuda_rng"])
