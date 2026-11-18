"""Two-image, two-update CPU correctness smoke; not a trained research model.

Run with an explicitly provisioned Torch runtime and PYTHONPATH=src. No downloads.
"""

import json
import tempfile
from pathlib import Path

import torch

from btp.student.inference import export_student, load_student
from btp.student.model import CompactDehaze, StudentConfig
from btp.student.trainer import BoundedTrainer, TrainingConfig, TrainingPair


def main() -> None:
    """Audit default parameter count and execute only a tiny width-two fixture."""
    torch.set_num_threads(1)
    torch.manual_seed(42)
    candidate = CompactDehaze()
    count = candidate.parameter_count()
    del candidate
    pairs = [
        TrainingPair(
            str(i), str(i), "train", torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8)
        )
        for i in range(2)
    ]
    trainer = BoundedTrainer(
        CompactDehaze(StudentConfig(width=2)),
        TrainingConfig(max_updates=2, max_seconds=30, accumulation=1, warmup_updates=1),
        pairs,
        provenance={
            "dataset_version": "synthetic-fixture-v1",
            "source_commit": "local-uncommitted-smoke",
        },
    )
    trainer.run()
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "inference.pt"
        digest = export_student(trainer.model, checkpoint)
        predictor = load_student(checkpoint, trainer.model.config, "cpu", sha256=digest)
        torch.testing.assert_close(
            predictor(pairs[0].hazy), trainer.model(pairs[0].hazy).clamp(0, 1)
        )
    print(
        json.dumps(
            {
                "status": "CPU fixture only; no trained research model",
                "torch_version": torch.__version__,
                "seed": 42,
                "default_width16_parameters": count,
                "default_float32_parameter_bytes": count * 4,
                "fixture_width": 2,
                "updates": trainer.updates,
                "exposures": trainer.exposures,
                "finite_losses": all(
                    torch.isfinite(torch.tensor(row["loss"])) for row in trainer.history
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
