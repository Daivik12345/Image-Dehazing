"""Optional CPU integration of calibration and real student evaluation artifacts."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional Torch runtime absent")
class ResearchTorchTests(unittest.TestCase):
    def test_variance_matches_guidance_and_student_evaluation(self):
        import numpy as np
        import torch
        from PIL import Image

        from btp.calibration import local_variance
        from btp.evaluation import sha256
        from btp.guidance import confidence_maps
        from btp.student.inference import export_student
        from btp.student.model import CompactDehaze, StudentConfig
        from btp.student_evaluation import evaluate

        torch.set_num_threads(1)
        torch.manual_seed(42)
        a = np.random.default_rng(42).random((17, 19, 3), dtype=np.float32)
        t = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0)
        actual = confidence_maps(t, epsilon_var=0.001).variance[0, 0].numpy()
        np.testing.assert_allclose(actual, local_variance(a), atol=1e-8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray((a * 255).astype(np.uint8)).save(root / "image.png")
            digest = sha256(root / "image.png")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "dataset": "fixture",
                        "release": "v1",
                        "scenes": [
                            {
                                "scene_id": "fixture",
                                "split": "development",
                                "geometry_tier": "native",
                                "hazy": "image.png",
                                "clean": "image.png",
                                "hazy_sha256": digest,
                                "clean_sha256": digest,
                            }
                        ],
                    }
                )
            )
            model = CompactDehaze(StudentConfig(width=2))
            checkpoint = export_student(model, root / "model.pt")
            (root / "architecture.json").write_text(json.dumps(model.config.to_dict()))
            report = evaluate(
                root / "manifest.json",
                root / "model.pt",
                root / "architecture.json",
                checkpoint,
                root / "result",
                scenes=["fixture"],
                device="cpu",
                source_commit="a" * 40,
            )
            self.assertEqual(report["status"], "completed")
            self.assertEqual(len(report["records"][0]["times_seconds"]), 5)
            self.assertEqual(report["records"][0]["size"], [19, 17])
