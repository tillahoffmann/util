from util import sampling, log_gaussian, hpd_interval
import pytest
import numpy as np


args = []

for num_parameters in [1, 3, 5]:
    # Define parameters
    mean = np.random.normal(0, 1, num_parameters)
    covariance = np.diag(1 + np.random.gamma(1, size=num_parameters))

    # Create a metropolis sampler
    sampler = sampling.MetropolisSampler(
        lambda x, mean=mean, covariance=covariance: log_gaussian(x, mean, covariance)[0],
        covariance / num_parameters
    )
    args.append((mean, covariance, sampler))

    # Create an adaptive metropolis sampler
    sampler = sampling.AdaptiveMetropolisSampler(
        lambda x, mean=mean, covariance=covariance: log_gaussian(x, mean, covariance)[0]
    )
    args.append((mean, covariance, sampler))

    # Create a Hamiltonian metropolis sampler
    sampler = sampling.HamiltonianSampler(
        lambda x, mean=mean, covariance=covariance: log_gaussian(x, mean, covariance)[0],
        jac=lambda x, mean=mean, covariance=covariance: log_gaussian(x, mean, covariance)[1],
        mass=covariance
    )
    args.append((mean, covariance, sampler))


@pytest.mark.parametrize('mean, covariance, sampler', args)
def test_sampling(mean, covariance, sampler):
    # Start the sampler at the mean
    sample = sampler.sample(mean, 1000)
    assert sample.shape == mean.shape, "incorrect shape for return value of `sample`"
    assert sampler.samples.shape == (1000, ) + mean.shape, "incorrect shape for attribute `samples`"
    assert sampler.acceptance_rate() > 0, "no samples were accepted"
    sample_mean = np.mean(sampler.samples, axis=0)
    np.testing.assert_array_less(np.abs(sample_mean - mean), np.diag(covariance), "unexpected sample mean")


@pytest.fixture(params=['grid_density_plot', 'autocorrelation_plot', 'trace_plot'])
def plotting_method(request):
    return request.param


@pytest.mark.parametrize('mean, covariance, sampler', args)
def test_plotting(mean, covariance, sampler, plotting_method):
    method = getattr(sampler, plotting_method)
    method()


def test_hpd_interval():
    x = np.random.normal(0, 1, 10000)
    lower, upper, continuous = hpd_interval(x)
    assert lower < 0, "expected negative lower bound"
    assert upper > 0, "expected positive upper bound"
    assert upper - lower > 3, "expected larger interval"
    assert upper - lower < 5, "expected smaller interval"
    assert continuous, "expected a continuous interval"


def test_foo():
    pass
