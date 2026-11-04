"""Boundaries for reference preprocessing and CPU-safe launch validation."""

from pathlib import Path

import pytest

from btp.reference import reference_dimensions, run_reference, sha256_file


@pytest.mark.parametrize(
    "original,expected",
    [
        ((640, 427), (767, 512)),
        ((128, 128), (512, 512)),
        ((1024, 768), (1024, 768)),
        ((512, 512), (512, 512)),
        ((100, 200), (512, 1024)),
    ],
)
def test_reference_resize_matches_short_edge_rule(original, expected):
    assert reference_dimensions(*original) == expected


def test_invalid_dimensions():
    with pytest.raises(ValueError):
        reference_dimensions(0, 10)


def test_reference_missing_files_fail_without_gpu(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_reference(
            tmp_path / "missing.png",
            tmp_path,
            tmp_path / "output",
            "hazegen",
            42,
            "test-input-v1",
        )


def test_reference_refuses_overwriting_a_run(tmp_path):
    with pytest.raises(FileExistsError):
        run_reference(Path("unused.png"), tmp_path, tmp_path, "spaced", 42, "test-v1")


def test_hash_is_streamed_and_reproducible(tmp_path):
    sample = tmp_path / "sample"
    sample.write_bytes(b"abc")
    assert (
        sha256_file(sample)
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
