import util
import numpy as np


def test_offdiag_indices():
    i, j = util.offdiag_indices(100)
    assert np.all(i != j), "indices must not be diagonal"


def test_square_size():
    i, j = util.offdiag_indices(100)
    assert util.square_size(len(i)) == 100, "unexpected result"
