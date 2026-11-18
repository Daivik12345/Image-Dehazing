import numpy as np

from btp.data_review import candidate_pairs, perceptual_hash


def test_identical_cross_split_is_reported():
    image = np.random.default_rng(42).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    value = perceptual_hash(image)
    rows = [
        dict(id="a", split="train", phash=value),
        dict(id="b", split="validation", phash=value),
    ]
    assert candidate_pairs(rows)[0]["distance"] == 0
    assert not candidate_pairs([rows[0], dict(rows[1], split="train")])
