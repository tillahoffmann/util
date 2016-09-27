from util.util import log_gaussian
from util import sampling
import pytest
import numpy as np
import functools


@pytest.fixture(params=[1, 3, 5])
def num_parameters(request):
    return request.param


def test_metropolis(num_parameters):
    # Define parameters
    mean = np.random.normal(0, 1, num_parameters)
    covariance = np.diag(1 + np.random.gamma(1, size=num_parameters))

    # Create a sampler
    sampler = sampling.MetropolisSampler(lambda x: log_gaussian(x, mean, covariance)[0],
                                         covariance / num_parameters)
    # Start the sampler at the mean
    sample = sampler.sample(mean, 1000)
    assert sample.shape == (num_parameters, ), "incorrect shape for return value of `sample`"
    assert sampler.samples.shape == (1000, num_parameters), "incorrect shape for attribute `samples`"
    sample_mean = np.mean(sampler.samples, axis=0)
    np.testing.assert_array_less(np.abs(sample_mean - mean), np.diag(covariance), "unexpected sample mean")
