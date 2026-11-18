"""Additive appearance correction does not scale local detail."""

import numpy as np
import pytest

from btp.fusion import align_global_appearance


def test_equal_inputs_remain_identical():
    local = np.random.default_rng(9).random((7, 11, 3), dtype=np.float32)
    result = align_global_appearance(local, local, sigma=2)
    np.testing.assert_array_equal(result.image, local)
    assert result.clipped_fraction == 0


def test_constant_offset_is_capped_without_variance_scaling():
    local = np.full((9, 13, 3), 0.3, dtype=np.float32)
    guide = np.full_like(local, 0.9)
    result = align_global_appearance(local, guide, max_correction=0.05)
    np.testing.assert_allclose(result.image, 0.35, atol=1e-7)
    assert result.clipped_fraction == 1
    assert np.max(np.abs(result.image - local)) <= 0.050001


def test_detail_is_retained_for_uniform_appearance_difference():
    local = np.random.default_rng(2).uniform(0.2, 0.6, (9, 13, 3)).astype(np.float32)
    result = align_global_appearance(local, local + 0.02, sigma=2)
    np.testing.assert_allclose(result.image, local + 0.02, atol=1e-7)
    assert result.clipped_fraction == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma": 0},
        {"sigma": float("nan")},
        {"max_correction": -1},
        {"max_correction": float("inf")},
    ],
)
def test_invalid_filter_and_bound_rejected(kwargs):
    with pytest.raises(ValueError):
        align_global_appearance(np.zeros((2, 3, 3)), np.zeros((2, 3, 3)), **kwargs)


def test_invalid_guide_geometry_range_and_finiteness_rejected():
    local = np.zeros((2, 3, 3))
    for guide in [
        np.zeros((3, 2, 3)),
        np.full_like(local, np.nan),
        np.full_like(local, 2),
    ]:
        with pytest.raises(ValueError):
            align_global_appearance(local, guide)
