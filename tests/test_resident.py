"""Resident lifecycle tests require no model checkpoints or GPU."""

import random

import pytest
from PIL import Image

from btp.resident import ResidentConfig, ResidentTeacher


class FakeBackend:
    loads = 0

    def __init__(self, config, weights, device):
        type(self).loads += 1
        self.conditioning = None
        self.closed = False

    def predict(self, path, seed):
        assert self.conditioning is None
        random.seed(seed)
        image = Image.open(path).convert("RGB")
        self.conditioning = image.getpixel((0, 0))
        if self.conditioning[0] == 255:
            raise RuntimeError("fixture failure")
        result = Image.new(
            "RGB", image.size, (self.conditioning[0], random.randrange(256), 0)
        )
        return result, {"sampling_seconds": 0.0}

    def reset(self):
        self.conditioning = None

    def close(self):
        self.closed = True


def test_one_load_a_b_a_seed_and_exception_cleanup(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("fixture")
    weights = tmp_path / "weights"
    weights.mkdir()
    for name in ("v2-1_512-ema-pruned.ckpt", "stage2.pt"):
        (weights / name).write_bytes(b"fixture")
    FakeBackend.loads = 0
    session = ResidentTeacher(
        ResidentConfig(cfg), weights, "cpu", backend_factory=FakeBackend
    )
    paths = []
    for index, color in enumerate((10, 20, 10, 255, 10)):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (3, 4), (color, 0, 0)).save(path)
        paths.append(path)
    records = [
        session.predict(path, tmp_path / f"out{i}", 42) for i, path in enumerate(paths)
    ]
    assert FakeBackend.loads == 1
    assert (
        records[0]["output_sha256"]
        == records[2]["output_sha256"]
        == records[4]["output_sha256"]
    )
    assert records[0]["output_sha256"] != records[1]["output_sha256"]
    assert records[3]["status"] == "failed"
    assert records[0]["warm_end_to_end_seconds"] >= 0
    assert len(records[0]["source_sha256"]) == 64
    assert len(records[0]["checkpoint_sha256"]) == 2
    cfg.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        session.predict(paths[0], tmp_path / "changed", 42)
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.predict(paths[0], tmp_path / "closed", 42)


def test_oom_closes_worker_and_checkpoint_change_rejected(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("fixture")
    for name in ("v2-1_512-ema-pruned.ckpt", "stage2.pt"):
        (tmp_path / name).write_bytes(b"fixture")
    image = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(image)

    class OOMBackend(FakeBackend):
        def predict(self, path, seed):
            raise MemoryError("out of memory fixture")

    session = ResidentTeacher(
        ResidentConfig(cfg), tmp_path, "cpu", backend_factory=OOMBackend
    )
    record = session.predict(image, tmp_path / "oom", 42)
    assert record["status"] == "failed"
    assert session.closed
    with pytest.raises(RuntimeError, match="closed"):
        session.predict(image, tmp_path / "later", 42)
    session = ResidentTeacher(
        ResidentConfig(cfg), tmp_path, "cpu", backend_factory=FakeBackend
    )
    (tmp_path / "stage2.pt").write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="changed"):
        session.predict(image, tmp_path / "changed", 42)


def test_opt_in_guidance_config_and_identity(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("fixture")
    for name in ("v2-1_512-ema-pruned.ckpt", "stage2.pt"):
        (tmp_path / name).write_bytes(b"fixture")
    assert ResidentConfig(cfg).guidance_mode == "B0"
    for mode in ("confidence", "uniform", "cap"):
        with pytest.raises(ValueError, match="epsilon"):
            ResidentConfig(cfg, guidance_mode=mode)
    with pytest.raises(ValueError, match="cap"):
        ResidentConfig(cfg, guidance_mode="cap", epsilon_var=1e-4)
    with pytest.raises(ValueError):
        ResidentConfig(cfg, guidance_mode="B0", max_l2_norm=1)
    baseline = ResidentTeacher(
        ResidentConfig(cfg), tmp_path, "cpu", backend_factory=FakeBackend
    )
    confidence = ResidentTeacher(
        ResidentConfig(cfg, guidance_mode="confidence", epsilon_var=1e-4),
        tmp_path,
        "cpu",
        backend_factory=FakeBackend,
    )
    assert baseline.identity != confidence.identity


def test_runtime_settings_change_teacher_identity(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("fixture")
    for name in ("v2-1_512-ema-pruned.ckpt", "stage2.pt"):
        (tmp_path / name).write_bytes(b"fixture")

    class RuntimeBackend(FakeBackend):
        runtime_identity = {"torch": "fixture", "tf32_matmul": False}

    a = ResidentTeacher(
        ResidentConfig(cfg), tmp_path, "cpu", backend_factory=RuntimeBackend
    )
    RuntimeBackend.runtime_identity = {"torch": "fixture", "tf32_matmul": True}
    b = ResidentTeacher(
        ResidentConfig(cfg), tmp_path, "cpu", backend_factory=RuntimeBackend
    )
    assert a.identity != b.identity
    with pytest.raises(ValueError, match="runtime"):
        a.predict(tmp_path / "unused.png", tmp_path / "unused", 42)
