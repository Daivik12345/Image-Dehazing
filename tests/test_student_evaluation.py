import numpy as np

from btp.student_evaluation import predict_variant


def test_identity_tiled_and_global_keep_geometry():
    image = np.random.default_rng(42).random((19, 23, 3), dtype=np.float32)
    for variant in ["full", "tiled"]:
        result, details = predict_variant(
            image, lambda x: x, variant, tile_size=16, overlap=4
        )
        np.testing.assert_allclose(result, image, atol=2e-7)
    result, details = predict_variant(
        image, lambda x: x, "global", tile_size=16, overlap=4
    )
    assert result.shape == image.shape
    assert details["global_size"] == [16, 13]
    assert details["clipping"]
