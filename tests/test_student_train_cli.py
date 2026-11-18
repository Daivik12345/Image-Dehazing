"""Audited pilot defaults and optional tiny CPU orchestration."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from btp.evaluation import sha256

ROOT = Path(__file__).resolve().parents[1]


class PilotTests(unittest.TestCase):
    def fixture(self, root):
        rows = []
        for i, split in enumerate(("train", "validation")):
            row = dict(
                pair_id=f"pair{i}",
                source_scene_id=f"fixture{i}",
                split=split,
                source_size=[16, 16],
                registration_status="verified",
            )
            for role in ("hazy", "clean"):
                path = root / f"{i}-{role}.png"
                Image.new(
                    "RGB",
                    (16, 16),
                    (40 + i * 20 + (10 if role == "clean" else 0), 70, 90),
                ).save(path)
                row[role], row[role + "_sha256"] = path.name, sha256(path)
            rows.append(row)
        path = root / "manifest.json"
        path.write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    dataset="fixture",
                    release="v1",
                    source_archive="fixture.zip",
                    archive_sha256="a" * 64,
                    license="fixture only",
                    scenes=rows,
                )
            )
        )
        return dict(
            manifest=path,
            data_root=root,
            evaluation_manifest=ROOT / "configs/evaluation/nh_haze_manifest.json",
            protocol=ROOT / "configs/research/protocol-v1.json",
            output=root / "output",
            source_commit="fixture",
            crop_size=16,
        )

    def test_default_audits_without_training_and_enforces_boundaries(self):
        from btp.student.train_cli import run_pilot

        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            record = run_pilot(**args)
            self.assertEqual(record["status"], "audit_only")
            self.assertEqual(record["max_segment_seconds"], 1200)
            self.assertEqual(record["audit"]["counts"], {"train": 1, "validation": 1})
            self.assertFalse((args["output"] / "last.pt").exists())
            args["output"] = Path(directory) / "other"
            for changes in (
                {"max_host_bytes": 1},
                {"max_segment_seconds": float("nan")},
                {"max_segment_seconds": 0},
                {"config": {"max_updates": 2001}},
                {"ablation": "A2"},
            ):
                with self.assertRaises((ValueError, MemoryError)):
                    run_pilot(**dict(args, **changes))
            doc = json.loads(args["manifest"].read_text())
            doc["scenes"][1]["source_scene_id"] = "nh-haze-01"
            args["manifest"].write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "evaluation"):
                run_pilot(**args)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "optional Torch runtime absent"
    )
    def test_tiny_run_exports_selected_artifact_and_preserves_resume_pair(self):
        import torch

        from btp.student.train_cli import run_pilot

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            record = run_pilot(
                **args,
                run=True,
                architecture={"width": 2},
                config={
                    "max_updates": 2,
                    "max_seconds": 30,
                    "accumulation": 1,
                    "warmup_updates": 1,
                    "checkpoint_interval": 1,
                },
                until_update=1,
            )
            self.assertEqual(record["updates"], 1)
            self.assertEqual(record["segment"]["max_segment_seconds"], 1200)
            self.assertGreaterEqual(record["segment"]["elapsed_seconds"], 0)
            self.assertEqual(record["runtime"]["torch"], str(torch.__version__))
            self.assertIsNone(record["allocated_peak_bytes"])
            self.assertIsNone(record["reserved_peak_bytes"])
            self.assertGreaterEqual(record["training_wall_seconds"], 0)
            self.assertEqual(record["selection"], "corpus_validation_best")
            self.assertTrue((args["output"] / "last.pt.budget.json").exists())
            self.assertTrue((args["output"] / "inference.pt").exists())
            args["resume"] = args["output"] / "last.pt"
            args["output"] = Path(directory) / "resume-output"
            continued = run_pilot(
                **args,
                run=True,
                architecture={"width": 2},
                config={
                    "max_updates": 2,
                    "max_seconds": 30,
                    "accumulation": 1,
                    "warmup_updates": 1,
                    "checkpoint_interval": 1,
                },
                until_update=2,
                max_segment_seconds=30,
            )
            self.assertEqual(continued["updates"], 2)
            self.assertEqual(continued["config"], record["config"])
            self.assertEqual(continued["segment"]["max_segment_seconds"], 30)
            self.assertGreaterEqual(
                continued["elapsed_seconds"], record["elapsed_seconds"]
            )
            persisted = json.loads((args["output"] / "pilot.json").read_text())
            self.assertEqual(persisted["segment"], continued["segment"])
            self.assertEqual(
                continued["initialization_sha256"], record["initialization_sha256"]
            )

    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "optional Torch runtime absent"
    )
    def test_a1_requires_all_teacher_targets_and_labels_unselected_export(self):
        import torch

        from btp.student.train_cli import run_pilot

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "teacher"):
                run_pilot(**args, ablation="A1")
            document = json.loads(args["manifest"].read_text())
            train = document["scenes"][0]
            train["teacher"] = train["clean"]
            train["teacher_sha256"] = train["clean_sha256"]
            args["manifest"].write_text(json.dumps(document))
            record = run_pilot(
                **args,
                run=True,
                ablation="A1",
                architecture={"width": 2},
                config={
                    "max_updates": 1,
                    "max_seconds": 30,
                    "accumulation": 1,
                    "warmup_updates": 1,
                },
            )
            self.assertEqual(record["updates"], 1)
            self.assertEqual(
                record["selection"], "last_unselected_no_validation_checkpoint"
            )
            self.assertEqual(record["validation_history"], [])
