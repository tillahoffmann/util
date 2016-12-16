import pytest
import numpy as np
import tensorflow as tf
import util


class LinearModel(util.Model):
    """
    Linear model assuming unit variance Gaussian noise.
    """
    def __init__(self, X, y, parameters):
        self._X = X
        self._y = y
        super(LinearModel, self).__init__(parameters)

    def build_likelihood(self):
        self.X = tf.constant(self._X)
        self.y = tf.constant(self._y)
        self.predictor = tf.reduce_sum(self.X * self.parameters, -1)
        return - tf.reduce_sum((self.predictor - self.y) ** 2) / 2


@pytest.fixture
def data():
    n, p = 10000, 3
    X = np.random.normal(size=(n, p)).astype(np.float32)
    parameters = np.random.normal(size=p).astype(np.float32)
    y = np.dot(X, parameters) + np.random.normal(size=n).astype(np.float32)
    return X, y, parameters


@pytest.fixture
def model(data):
    X, y, parameters = data
    model = LinearModel(X, y, parameters)
    model.desired = parameters
    return model


def test_optimize(model):
    parameters, posterior = model.optimize(100)
    np.testing.assert_allclose(parameters, model.desired, atol=0.1, err_msg='unexpected parameter values')


def test_posterior_cov(model):
    cov = model.run('posterior_cov')
    assert np.all(np.linalg.eigvalsh(cov) > 0), "posterior covariance must be positive definite"
