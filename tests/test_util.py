import util
import numpy as np
import pytest
import tensorflow as tf


def test_acorr():
    x = np.random.normal(0, 1, 1000)
    y = util.acorr(x)
    assert x.size - 1 == y.size, "unexpected autocorrelation size"


@pytest.mark.parametrize('value, desired', [
    ('Hello World!', False),
    ([1, 2, 3], True),
    ((1, 2, 3), True),
    (np.ones(10), True),
    (tf.ones(10), False)
])
def test_iterable(value, desired):
    assert util.iterable(value) == desired, "incorrect result for `iterable`"
