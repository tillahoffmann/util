import util
import numpy as np
import pytest


@pytest.fixture(params=[True, False])
def directed(request):
    return request.param


def test_offdiag_indices(directed):
    i, j = util.offdiag_indices(100, directed)
    assert np.all(i != j), "indices must not be diagonal"
    if directed:
        assert len(i) == 100 * 99, "unexpected length"
    else:
        assert len(i) == 100 * 99 / 2, "unexpected length"


def test_square_size(directed):
    i, _ = util.offdiag_indices(100, directed)
    assert util.square_size(len(i), directed) == 100, "unexpected result"


def test_square_array(directed):
    x = np.random.normal(0, 1, (100 * 99) if directed else (100 * 99 // 2))
    square = util.square_array(x, directed)
    if not directed:
        np.testing.assert_equal(square, square.T, 'square array must be symmetric')
    assert square.shape == (100, 100), "unexpected shape"


def test_linear_array(directed):
    x = np.random.normal(0, 1, (100, 100))
    linear = util.linear_array(x, directed)
    if directed:
        assert len(linear) == 100 * 99, "unexpected length"
    else:
        assert len(linear) == 100 * 99 // 2, "unexpected length"


def test_linear_square_linear_array(directed):
    desired = np.random.normal(0, 1, (100 * 99) if directed else (100 * 99 // 2))
    square = util.square_array(desired, directed)
    actual = util.linear_array(square, directed)
    np.testing.assert_equal(actual, desired, 'could not recover linear array')
