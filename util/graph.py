import numpy as np


def square_size(condensed_size):
    """
    Compute the size of a square matrix from a condensed matrix without diagonal elements
    (size `num_nodes * num_nodes`).
    """
    x = 0.5 * (1 + np.sqrt(1 + 4 * condensed_size))
    assert x == int(x), "expected an integer but got %f" % x
    return int(x)


def offdiag_indices(n):
    """
    Return the off diagonal indices of a `n` by `n` matrix.
    """
    return np.nonzero(1 - np.eye(n))
