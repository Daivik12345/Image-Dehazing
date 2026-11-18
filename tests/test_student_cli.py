"""Optional end-to-end CPU artifact and native geometry integration."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional Torch runtime absent")
class StudentCLITests(unittest.TestCase):
    def test_native_and_tiled_artifacts(self):
        import numpy as np
        import torch
        from PIL import Image

        from btp.student.cli import run_student
        from btp.student.inference import export_student
        from btp.student.model import CompactDehaze, StudentConfig

        torch.set_num_threads(1)
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = CompactDehaze(StudentConfig(width=2))
            digest = export_student(model, root / "fixture.pt")
            (root / "architecture.json").write_text(json.dumps(model.config.to_dict()))
            Image.fromarray(np.full((19, 23, 3), 128, dtype=np.uint8)).save(
                root / "input.png"
            )
            for tiled in (False, True):
                output = root / str(tiled)
                record = run_student(
                    root / "input.png",
                    root / "fixture.pt",
                    root / "architecture.json",
                    digest,
                    output,
                    tiled=tiled,
                    tile_size=16,
                    overlap=4,
                )
                self.assertEqual(record["output_size"], [23, 19])
                self.assertEqual(record["checkpoint_sha256"], digest)
                self.assertTrue((output / "prediction.png").is_file())
                with self.assertRaises(FileExistsError):
                    run_student(
                        root / "input.png",
                        root / "fixture.pt",
                        root / "architecture.json",
                        digest,
                        output,
                    )
            with self.assertRaisesRegex(ValueError, "SHA256"):
                run_student(
                    root / "input.png",
                    root / "fixture.pt",
                    root / "architecture.json",
                    "0" * 64,
                    root / "bad",
                )
            self.assertFalse((root / "bad").exists())
            with self.assertRaises(MemoryError):
                run_student(
                    root / "input.png",
                    root / "fixture.pt",
                    root / "architecture.json",
                    digest,
                    root / "oom",
                    tiled=True,
                    max_host_bytes=1,
                )
            failure = json.loads((root / "oom/record.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["error_type"], "MemoryError")


if __name__ == "__main__":
    unittest.main()
