import util
import numpy as np


def test_acorr():
    x = np.random.normal(0, 1, 1000)
    y = util.acorr(x)
    assert x.size - 1 == y.size, "unexpected autocorrelation size"
