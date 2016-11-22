import logging
import numpy as np
from .base import BaseSampler


logger = logging.getLogger(__name__)


class MetropolisSampler(BaseSampler):
    """
    Standard Metropolis sampler.

    Parameters
    ----------
    fun : callable
        log-posterior or log-likelihood function taking a vector of parameters as its first argument
    proposal_covariance : array_like
        covariance of the Gaussian proposal distribution
    args : array_like
        additional arguments to pass to `fun`
    parameter_names : list
        list of parameter names
    break_on_interrupt : bool
        stop the sampler rather than throwing an exception upon a keyboard interrupt
    mode : str
        'update' to reuse the function value from the previous sample, 'reevaluate' to reevaluate the function value
        at every sampling step
    """
    def __init__(self, fun, proposal_covariance, args=None, parameter_names=None, break_on_interrupt=True,
                 mode='update'):
        super(MetropolisSampler, self).__init__(fun, args, parameter_names, break_on_interrupt)
        self.mode = mode
        self.proposal_covariance = proposal_covariance

    def sample(self, parameters, steps=1, callback=None):
        parameters = np.asarray(parameters)

        try:
            for step in steps if hasattr(steps, '__iter__') else range(steps):
                # Evaluate the current function value
                if len(self._fun_values) == 0 or self.mode == 'reevaluate':
                    fun_current = self.fun(parameters, *self.args)
                else:
                    fun_current = self._fun_values[-1]

                # Make a proposal
                proposal = np.random.multivariate_normal(parameters, self.proposal_covariance)

                # Compute the function at the proposed sample
                fun_proposal = self.fun(proposal, *self.args)
                # Accept or reject the step
                if fun_proposal - fun_current > np.log(np.random.uniform()):
                    # Update the log posterior and the parameter values
                    fun_current = fun_proposal
                    parameters = proposal

                # Save the parameters
                self._samples.append(parameters)
                self._fun_values.append(fun_current)

                if callback:
                    callback(parameters)

        # Reraise if we are not breaking on interrupt
        except KeyboardInterrupt:
            logger.info('sampling cancelled by keyboard interrupt')
            if not self.break_on_interrupt:
                raise

        return parameters


class AdaptiveMetropolisSampler(MetropolisSampler):
    """
    Adaptive Metropolis sampler.

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
    mode : str
        'update' to reuse the function value from the previous sample, 'reevaluate' to reevaluate the function value
        at every sampling step
    threshold : int
        number of iterations before the adaptive algorithm starts
    epsilon : array_like
        initial proposal covariance
    scale : float
        factor to apply to the sample covariance to get the proposal covariance
    """
    def __init__(self, fun, args=None, parameter_names=None, break_on_interrupt=True, mode='update', threshold=100,
                 epsilon=1e-5, scale=5.76):
        super(AdaptiveMetropolisSampler, self).__init__(fun, None, args, parameter_names, break_on_interrupt, mode)

        self.threshold = threshold
        self.epsilon = epsilon
        self.scale = scale

        # Dummy parameters
        self.num_parameters = None
        self.covariance0 = None
        self.sample_covariance = 0
        self.sample_mean = 0

    def sample(self, parameters, steps=1, callback=None):
        # Get additional information
        if self.num_parameters is None:
            # Store derived information
            self.num_parameters = len(parameters)
            # Initialise the proposal scale
            self.scale /= self.num_parameters
            # Initialise the covariance matrix
            self.covariance0 = np.dot(np.eye(self.num_parameters), self.epsilon)
            # Initialise the running mean and variance
            self.sample_mean = np.zeros(self.num_parameters)

        try:
            for step in range(steps):
                # Make a proposal with the initial covariance or the scaled sample covariance
                self.proposal_covariance= self.covariance0 if len(self._samples) < self.threshold \
                    else self.sample_covariance + self.covariance0

                # Sample
                parameters = super(AdaptiveMetropolisSampler, self).sample(parameters, 1, callback)

                # Update the sample mean...
                previous_mean = self.sample_mean
                self.sample_mean = (parameters + (len(self._samples) - 1) * previous_mean) / len(self._samples)

                # ...and the sample covariance
                self.sample_covariance = ((len(self._samples) - 1) * self.sample_covariance + parameters *
                                          parameters[:, None] + (len(self._samples) - 1) * previous_mean *
                                          previous_mean[:, None]) / len(self._samples) - \
                                         self.sample_mean * self.sample_mean[:, None]
        except KeyboardInterrupt:
            logger.info('sampling cancelled by keyboard interrupt')
            if not self.break_on_interrupt:
                raise

        return parameters
