import numpy as np
import pytest

from btp.research_eval import predict_tiled, seam_metrics


def test_tiled_identity_preserves_native_odd_geometry():
    image = np.random.default_rng(42).random((31, 47, 3), dtype=np.float32)
    actual = predict_tiled(image, lambda x: x, tile_size=16, overlap=4)
    np.testing.assert_allclose(actual, image, atol=2e-7)


def test_seam_error_detects_border_artifact():
    clean = np.full((32, 32, 3), 0.5)
    broken = clean.copy()
    broken[:, 16] = 1
    metrics = seam_metrics(clean, broken, tile_size=16, overlap=0, band=1)
    assert metrics["seam_mae"] > metrics["interior_mae"]
    assert metrics["gradient_error"] > 0


def test_invalid_prediction_fails():
    with pytest.raises(ValueError):
        predict_tiled(
            np.zeros((16, 16, 3)), lambda x: x * np.nan, tile_size=16, overlap=4
        )


def test_integer_gradient_matches_float():
    clean = np.zeros((2, 2, 3), dtype=np.uint8)
    predicted = clean.copy()
    predicted[:, 1] = 1
    integer = seam_metrics(clean, predicted, tile_size=2, overlap=0)
    floating = seam_metrics(
        clean.astype(float), predicted.astype(float), tile_size=2, overlap=0
    )
    assert integer["gradient_error"] == floating["gradient_error"]
