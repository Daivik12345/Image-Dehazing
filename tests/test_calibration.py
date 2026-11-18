import numpy as np
import pytest

from btp.calibration import local_variance


def test_constant_variance_and_invalid_input():
    assert np.max(local_variance(np.ones((2, 3, 3)))) < 1e-12
    with pytest.raises(ValueError):
        local_variance(np.ones((2, 3, 3)) * 2)
    with pytest.raises(ValueError):
        local_variance(np.ones((2, 3, 3)), window=2)
