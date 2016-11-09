#!/usr/bin/env python

import numpy as np
from util import AdaptiveMetropolisSampler, log_gaussian
from matplotlib import pyplot as plt


def __main__():
    np.random.seed(4)
    # Generate parameters
    num_dims = 3
    mu = np.random.normal(0, 3, num_dims)
    cov = np.diag(np.random.gamma(.5, size=num_dims))
    # Create a sampler
    sampler = AdaptiveMetropolisSampler(lambda x: -log_gaussian(x, mu, cov)[0])
    # Draw samples
    sampler.sample(mu, 3000)
    # Show the trace
    sampler.trace_plot(values=mu)
    plt.show()


if __name__ == '__main__':
    __main__()
