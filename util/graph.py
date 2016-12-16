import numpy as np


def square_size(linear_size, directed=True):
    """
    Compute the size of a square matrix from a condensed matrix without diagonal elements.

    Parameters
    ----------
    linear_size : int
        size of a condensed matrix
    directed : bool
        whether the condensed matrix is directed

    Returns
    -------
    square_size : int
        size of the corresponding square matrix
    """
    if not directed:
        linear_size *= 2
    x = 0.5 * (1 + np.sqrt(1 + 4 * linear_size))
    assert x == int(x), "expected an integer but got %f" % x
    return int(x)


def offdiag_indices(n, directed=True):
    """
    Return the off diagonal indices of a `n` by `n` matrix.

    Parameters
    ----------
    n : int
        size of the square matrix
    directed : bool
        whether the condensed matrix is directed

    Returns
    -------
    i : np.ndarray
        offdiagonal index along the first dimension
    j : np.ndarray
        offdiagonal index along the second dimension
    """
    if directed:
        return np.nonzero(1 - np.eye(n))
    else:
        return np.triu_indices(n, 1)


def square_array(linear, directed=True):
    """
    Convert a linear vector to a square matrix.

    Parameters
    ----------
    linear : np.ndarray
        condensed matrix
    directed : bool
        whether the condensed matrix is directed

    Returns
    -------
    square_array : np.ndarray
        square matrix
    """
    n = square_size(len(linear), directed)
    i, j = offdiag_indices(n, directed)
    square = np.zeros((n, n), linear.dtype)
    square[i, j] = linear
    if not directed:
        square += square.T
    return square


def linear_array(square, directed=True):
    """
    Convert a square matrix to a linear vector.

    Parameters
    ----------
    square_array : np.ndarray
        square matrix
    directed : bool
        whether the condensed matrix is directed

    Returns
    -------
    linear : np.ndarray
        condensed matrix
    """
    i, j = offdiag_indices(square.shape[0], directed)
    return square[i, j]
