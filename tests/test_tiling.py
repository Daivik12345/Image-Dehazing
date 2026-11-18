"""Native coverage and streamed reconstruction contracts."""

import numpy as np
import pytest

from btp.tiling import extract_tile, fuse_tiles, make_tiles, smooth_weights, tile_seed


@pytest.mark.parametrize("width,height", [(1, 1), (2, 9), (9, 2), (13, 17), (8, 8)])
def test_identity_preserves_every_native_pixel(width, height):
    image = np.random.default_rng(7).random((height, width, 3), dtype=np.float32)
    tiles = make_tiles(width, height, tile_size=8, overlap=3)
    predictions = (extract_tile(image, tile) for tile in tiles)
    result = fuse_tiles(tiles, predictions, (width, height), max_host_bytes=1_000_000)
    assert result.shape == image.shape
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, image, atol=2e-7)


def test_terminal_positions_and_padding_are_explicit():
    tiles = make_tiles(13, 3, tile_size=8, overlap=2)
    assert [
        (t.x, t.y, t.width, t.height, t.pad_right, t.pad_bottom) for t in tiles
    ] == [(0, 0, 8, 3, 0, 5), (5, 0, 8, 3, 0, 5)]
    weights = smooth_weights(8)
    assert np.all(weights > 0)
    assert weights[3, 3] > weights[0, 0]
    np.testing.assert_array_equal(weights, weights[::-1, ::-1])


def test_seeds_repeat_and_distinguish_scene_and_position():
    tiles = make_tiles(13, 8, 8, 2)
    seed = tile_seed(42, "scene-a", tiles[0])
    assert seed == tile_seed(42, "scene-a", tiles[0])
    assert (
        len(
            {
                seed,
                tile_seed(42, "scene-b", tiles[0]),
                tile_seed(42, "scene-a", tiles[1]),
            }
        )
        == 3
    )


@pytest.mark.parametrize(
    "args", [(0, 5, 8, 2), (5, 5, 0, 0), (5, 5, 8, 8), (5, 5, 8, -1), (5.5, 5, 8, 2)]
)
def test_invalid_geometry_rejected(args):
    with pytest.raises(ValueError):
        make_tiles(*args)


def test_budget_rejected_before_consuming_predictions():
    def predictions():
        pytest.fail("budget must be checked before model output consumption")
        yield

    with pytest.raises(MemoryError, match="host"):
        fuse_tiles(make_tiles(8, 8, 8, 2), predictions(), (8, 8), max_host_bytes=1)


def test_missing_tiles_and_bad_predictions_rejected():
    tiles = make_tiles(13, 8, 8, 2)
    with pytest.raises(ValueError):
        fuse_tiles(tiles, [np.ones((8, 8, 3))], (13, 8))
    with pytest.raises(ValueError, match="coverage"):
        fuse_tiles(tiles[:1], [np.ones((8, 8, 3))], (13, 8))
    for prediction in [np.ones((7, 8, 3)), np.full((8, 8, 3), np.nan)]:
        with pytest.raises(ValueError):
            fuse_tiles(tiles[:1], [prediction], (8, 8))


def test_overlap_blending_preserves_constant_predictions_and_streams():
    import weakref

    tiles = make_tiles(13, 8, 8, 2)
    refs = []

    def predictions():
        for _ in tiles:
            assert all(ref() is None for ref in refs)
            value = np.full((8, 8, 3), 0.7, dtype=np.float32)
            refs.append(weakref.ref(value))
            yield value
            del value

    output = fuse_tiles(tiles, predictions(), (13, 8))
    np.testing.assert_allclose(output, 0.7, atol=1e-7)


def test_edge_padding_repeats_border_pixels_and_extra_predictions_fail():
    image = np.array([[[0.2, 0.3, 0.4]]], dtype=np.float32)
    tiles = make_tiles(1, 1, 4, 1)
    padded = extract_tile(image, tiles[0])
    np.testing.assert_allclose(padded, np.broadcast_to(image, (4, 4, 3)))
    with pytest.raises(ValueError, match="extra"):
        fuse_tiles(tiles, [padded, padded], (1, 1))
