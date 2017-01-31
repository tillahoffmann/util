import logging
import functools as ft
import numpy as np
import pandas as pd
from scipy import stats

from ..display import trace_plot, grid_density_plot, comparison_plot, autocorrelation_plot
from ..util import autospace


logger = logging.getLogger('util.sampling')


class BaseSampler(object):
    """
    Base class for MCMC samplers.

    Parameters
    ----------
    fun : callable
        log-posterior or log-likelihood function taking a vector of parameters as its first argument
    args : array_like
        additional arguments to pass to `fun`
    parameter_names : list
        list of parameter names
    break_on_interrupt : bool
        stop the sampler rather than throwing an exception upon a keyboard interrupt
    """
    def __init__(self, fun, args=None, parameter_names=None, break_on_interrupt=True):
        if not callable(fun):
            raise ValueError("`fun` must be callable")

        self.fun = fun
        self.args = [] if args is None else args
        self.parameter_names = parameter_names
        self.break_on_interrupt = break_on_interrupt

        self._samples = []
        self._fun_values = []

    def get_parameter_name(self, index):
        """
        Get a parameter name.

        Parameters
        ----------
        index : int
            index of the parameter for which to get a name
        """
        return str(index) if self.parameter_names is None else self.parameter_names[index]

    def _parameter_dict(self, parameters):
        if parameters is None:
            parameters = range(self.num_parameters)
        if not isinstance(parameters, dict):
            parameters = {i: self.get_parameter_name(i) for i in parameters}
        return parameters

    @ft.wraps(trace_plot)
    def trace_plot(self, burn_in=0, parameters=None, values=None):
        return trace_plot(self.samples, self.fun_values, burn_in, self._parameter_dict(parameters), values)

    @ft.wraps(grid_density_plot)
    def grid_density_plot(self, burn_in=0, parameters=None, values=None, nrows=None, ncols=None, bins=10, **kwargs):
        return grid_density_plot(self.samples, burn_in, self._parameter_dict(parameters), values, nrows, ncols, bins,
                                 **kwargs)

    @ft.wraps(comparison_plot)
    def comparison_plot(self, values, burn_in=0, parameters=None, ax=None, **kwargs):
        return comparison_plot(self.samples, values, burn_in, self._parameter_dict(parameters), ax, **kwargs)

    @ft.wraps(autocorrelation_plot)
    def autocorrelation_plot(self, burn_in=0, lag=50, parameters=None, ax=None, **kwargs):
        return autocorrelation_plot(self.samples, burn_in, self._parameter_dict(parameters), lag, ax, **kwargs)

    def acceptance_rate(self, burn_in=0):
        """
        Evaluate the acceptance rate.

        Parameters
        ----------
        burn_in : int
            number of initial values to discard
        """
        samples = self.samples[burn_in:]
        return np.mean(samples[1:] != samples[:-1])

    def sample(self, parameters, steps=1, callback=None):
        """
        Draw samples from the distribution.

        Parameters
        ----------
        parameters : array_like
            current state of the chain
        steps : int or iterable
            number of steps
        callback : callable
            callback after each step
        """
        raise NotImplementedError

    def describe(self, burn_in=0, parameters=None):
        """
        Get a description of the parameters.

        Parameters
        ----------
        burn_in : int
            number of initial values to discard
        parameters : iterable
            indices of the parameters to plot (default is all)
        """
        if parameters is None:
            parameters = np.arange(self.samples.shape[1])

        # Use pandas to get a description
        columns = map(self.get_parameter_name, parameters)
        frame = pd.DataFrame(self.samples[burn_in:, parameters], columns=columns)
        description = frame.describe()

        name = self.__class__.__name__

        description = "{}\n{}\n{}".format(name, '=' * len(name), description)

        return description

    @property
    def samples(self):
        """
        Get the samples.
        """
        return np.asarray(self._samples)

    @property
    def fun_values(self):
        """
        Get the function values.
        """
        return np.asarray(self._fun_values)

    @property
    def num_parameters(self):
        if not self._samples:
            raise RuntimeError("cannot determine number of parameters if no samples have been drawn")
        return len(self._samples[0])


def hpd_interval(samples, alpha=0.05, lin=200):
    """
    Compute the highest posterior density interval using a Gaussian kernel density estimate.

    Parameters
    ----------
    samples : np.ndarray
    alpha : float
    lin : np.ndarray or int

    Returns
    -------
    lower
    upper
    continuous
    """
    # Get a linear spacing
    if isinstance(lin, int):
        lin = autospace(samples, lin)

    # Compute the KDE
    kde = stats.gaussian_kde(samples)
    y = kde(lin)

    # Compute the cumulative distribution
    idx = np.argsort(y)
    cumulative = np.cumsum(y[idx])

    # Get indices after applying a threshold
    idx = np.sort(idx[cumulative > cumulative[-1] * alpha])

    return lin[idx[0]], lin[idx[-1]], np.all(np.diff(idx) == 1)
