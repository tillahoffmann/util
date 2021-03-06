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


@pytest.mark.parametrize('value, reference, desired', [
    ("a", "bdaef", 2),
    ("ab", "bdaef", [2, 0])
])
def test_list_index(value, reference, desired):
    assert util.list_index(reference, *value) == desired, "incorrect result for `list_index`"
