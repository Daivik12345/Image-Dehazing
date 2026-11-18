import numpy as np
import pytest

from btp.resident_validation import compare_predictions


def test_gate_rejects_drift_and_requires_repeats():
    image = np.zeros((12, 12, 3))
    assert compare_predictions([image] * 3, [image, image])["eligible"]
    altered = image.copy()
    altered[0, 0, 0] = 1 / 255
    assert not compare_predictions([image] * 3, [altered])["eligible"]
    with pytest.raises(ValueError):
        compare_predictions([image] * 2, [image])
