from scipy import stats
import numpy as np
from util.sampling import AdaptiveRejectionSampler
from util.util import log_gaussian
from matplotlib import pyplot as plt


def __main__():
    np.random.seed(4)

    mu, sigma = 1.2, 1.5
    # Create an adaptive rejection sampler
    ars = AdaptiveRejectionSampler(log_gaussian, (-1, 2), (mu, sigma))

    # Draw 16 samples and add new abscissas in the process
    fig, axes = plt.subplots(4, 4, True, True)
    samples = []
    for ax in axes.ravel():
        ars.plot(ax=ax)
        ax.scatter(samples, np.zeros_like(samples), color='c')
        samples.append(ars.sample())

    # Summary of the data for a larger sample
    samples = ars.sample(100)
    print("Sample mean: {}".format(np.mean(samples)))
    print("Sample std : {}".format(np.std(samples)))

    # Try a hypothesis test to check that we have zero-mean data
    result = stats.ttest_1samp(samples, mu)
    print("p-value for sample mean t-test: {}".format(result.pvalue))

    plt.show()
