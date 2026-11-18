"""Optional Torch CPU contract checks (also runnable with unittest)."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional Torch runtime absent")
class StudentTests(unittest.TestCase):
    def setUp(self):
        import torch

        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_shapes_losses_and_roundtrip(self):
        import torch

        from btp.student.inference import export_student, load_student
        from btp.student.losses import restoration_loss, ssim
        from btp.student.model import CompactDehaze, StudentConfig

        model = CompactDehaze(StudentConfig(width=2))
        for shape in [(1, 3, 1, 1), (1, 3, 17, 19)]:
            x = torch.rand(shape)
            y = model(x)
            self.assertEqual(y.shape, x.shape)
            self.assertTrue(torch.isfinite(y).all())
        target = torch.rand_like(x, requires_grad=True)
        loss = restoration_loss(y, x, target)
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(
            all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
        )
        self.assertAlmostEqual(ssim(x, x).item(), 1.0, places=5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "student.pt"
            digest = export_student(model, path)
            predictor = load_student(path, model.config, "cpu", sha256=digest)
            torch.testing.assert_close(predictor(x), model(x).clamp(0, 1))
            with self.assertRaises(ValueError):
                load_student(path, StudentConfig(width=4), "cpu", sha256=digest)
            with self.assertRaises(ValueError):
                load_student(path, model.config, "cpu", sha256="0" * 64)
        with self.assertRaises(ValueError):
            model(torch.rand(1, 1, 3, 3))
        with self.assertRaises(ValueError):
            StudentConfig(width=0)

    def test_resume_and_budget(self):
        import torch

        from btp.student.model import CompactDehaze, StudentConfig
        from btp.student.trainer import BoundedTrainer, TrainingConfig, TrainingPair

        pairs = [
            TrainingPair(
                str(i), str(i), "train", torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8)
            )
            for i in range(2)
        ]
        config = TrainingConfig(
            max_updates=2, max_seconds=60, accumulation=1, warmup_updates=1
        )
        torch.manual_seed(42)
        whole = BoundedTrainer(
            CompactDehaze(StudentConfig(width=2)),
            config,
            pairs,
            provenance={"dataset_version": "fixture", "source_commit": "fixture"},
        )
        initial = {k: v.clone() for k, v in whole.model.state_dict().items()}
        whole.run()
        self.assertEqual(whole.updates, 2)
        self.assertTrue(
            any(
                not torch.equal(initial[k], v)
                for k, v in whole.model.state_dict().items()
            )
        )
        torch.manual_seed(42)
        split = BoundedTrainer(
            CompactDehaze(StudentConfig(width=2)),
            config,
            pairs,
            provenance=whole.provenance,
        )
        split.run(until_update=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resume.pt"
            split.save(path)
            resumed = BoundedTrainer(
                CompactDehaze(StudentConfig(width=2)),
                config,
                pairs,
                provenance=whole.provenance,
            )
            resumed.restore(path)
            self.assertGreaterEqual(resumed.elapsed_seconds, split.elapsed_seconds)
            resumed.run()
            for k, v in whole.model.state_dict().items():
                torch.testing.assert_close(
                    v, resumed.model.state_dict()[k], rtol=0, atol=0
                )
            self.assertEqual(resumed.exposures, 2)
            resumed.elapsed_seconds = 61
            resumed.run()
            self.assertEqual(resumed.updates, 2)
        pairs[0].split = "held-out"
        with self.assertRaises(ValueError):
            BoundedTrainer(
                CompactDehaze(StudentConfig(width=2)),
                config,
                pairs,
                provenance=whole.provenance,
            )

    def test_failed_accumulation_checkpoint_is_resumable(self):
        import torch

        from btp.student.model import CompactDehaze, StudentConfig
        from btp.student.trainer import BoundedTrainer, TrainingConfig, TrainingPair

        pairs = [
            TrainingPair(
                str(i), str(i), "train", torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8)
            )
            for i in range(2)
        ]
        trainer = BoundedTrainer(
            CompactDehaze(StudentConfig(width=2)),
            TrainingConfig(max_updates=1, accumulation=2),
            pairs,
            provenance={"dataset_version": "fixture", "source_commit": "fixture"},
        )
        # Exercise failure after one microbatch without mutating caller data.
        from unittest.mock import patch

        original_forward = trainer.model.forward
        calls = 0

        def failing_forward(image):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("fixture forward failure")
            return original_forward(image)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(trainer.model, "forward", failing_forward):
                with self.assertRaises(ValueError):
                    trainer.run(checkpoint_dir=Path(tmp))
            self.assertEqual(trainer.data_position, 0)
            trainer.restore(Path(tmp) / "last.pt")
            self.assertEqual(trainer.updates, 0)

    def test_budget_and_data_rejection(self):
        import torch

        from btp.student.model import CompactDehaze, StudentConfig
        from btp.student.trainer import BoundedTrainer, TrainingConfig, TrainingPair

        pair = TrainingPair(
            "a", "scene", "train", torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8)
        )
        config = TrainingConfig(max_updates=2, max_seconds=1)
        provenance = {"dataset_version": "fixture", "source_commit": "fixture"}
        trainer = BoundedTrainer(
            CompactDehaze(StudentConfig(width=2)), config, [pair], provenance=provenance
        )
        trainer.elapsed_seconds = 2
        trainer.run()
        self.assertEqual(trainer.updates, 0)
        with self.assertRaises(ValueError):
            BoundedTrainer(trainer.model, config, [pair, pair], provenance=provenance)
        with self.assertRaises(ValueError):
            BoundedTrainer(
                trainer.model,
                config,
                [pair],
                provenance=provenance,
                forbidden_scene_ids=["scene"],
            )
        for kwargs in (
            {"max_seconds": float("nan")},
            {"max_updates": 0},
            {"accumulation": 0},
        ):
            with self.assertRaises(ValueError):
                TrainingConfig(**kwargs)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional Torch runtime absent")
class TrainingBoundaryTests(unittest.TestCase):
    def fixture(self):
        import torch

        from btp.student.model import StudentConfig
        from btp.student.trainer import TrainingConfig, TrainingPair

        torch.set_num_threads(1)
        return (
            StudentConfig(width=2),
            TrainingConfig(max_updates=2, accumulation=1, warmup_updates=1),
            [
                TrainingPair(
                    "a",
                    "scene",
                    "train",
                    torch.full((1, 3, 8, 8), 0.2),
                    torch.full((1, 3, 8, 8), 0.4),
                )
            ],
            {"dataset_version": "fixture", "source_commit": "fixture"},
        )

    def test_seeded_factory_repeats_initialization_and_restore_retains_identity(self):
        import torch

        from btp.student.model import CompactDehaze
        from btp.student.trainer import BoundedTrainer

        architecture, config, pairs, provenance = self.fixture()
        first = BoundedTrainer.create(
            config, pairs, model_config=architecture, provenance=provenance
        )
        torch.rand(100)
        second = BoundedTrainer.create(
            config, pairs, model_config=architecture, provenance=provenance
        )
        self.assertEqual(first.initialization_sha256, second.initialization_sha256)
        for key, value in first.model.state_dict().items():
            torch.testing.assert_close(
                value, second.model.state_dict()[key], atol=0, rtol=0
            )
        first.run(until_update=1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "resume.pt"
            first.save(checkpoint)
            other = BoundedTrainer(
                CompactDehaze(architecture), config, pairs, provenance=provenance
            )
            other.restore(checkpoint)
            self.assertEqual(first.initialization_sha256, other.initialization_sha256)

    def test_caller_mutation_cannot_change_audited_training_snapshot(self):
        from btp.student.model import CompactDehaze
        from btp.student.trainer import BoundedTrainer

        architecture, config, pairs, provenance = self.fixture()
        trainer = BoundedTrainer(
            CompactDehaze(architecture), config, pairs, provenance=provenance
        )
        pairs[0].split = "held-out"
        pairs[0].clean.fill_(float("nan"))
        trainer.run(until_update=1)
        self.assertEqual(trainer.updates, 1)
        self.assertEqual(trainer.pairs[0].split, "train")

    def test_checkpoint_save_time_is_persisted_and_missing_ledger_rejected(self):
        import time
        from unittest.mock import patch

        import torch

        from btp.student.model import CompactDehaze
        from btp.student.trainer import BoundedTrainer

        architecture, config, pairs, provenance = self.fixture()
        trainer = BoundedTrainer(
            CompactDehaze(architecture), config, pairs, provenance=provenance
        )
        original_save = torch.save

        def delayed_save(*args, **kwargs):
            time.sleep(0.05)
            return original_save(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "resume.pt"
            with patch("torch.save", delayed_save):
                trainer.save(checkpoint)
            resumed = BoundedTrainer(
                CompactDehaze(architecture), config, pairs, provenance=provenance
            )
            resumed.restore(checkpoint)
            self.assertGreaterEqual(resumed.elapsed_seconds, 0.05)
            checkpoint.with_suffix(".pt.budget.json").unlink()
            with self.assertRaisesRegex(ValueError, "budget"):
                resumed.restore(checkpoint)


if __name__ == "__main__":
    unittest.main()
