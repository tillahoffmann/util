from .base import BaseSampler
import numpy as np
from matplotlib import pyplot as plt
from warnings import warn
import logging


logger = logging.getLogger('util.sampling')


class HamiltonianSampler(BaseSampler):
    """
    Hamiltonian Monte Carlo sampler.

    Parameters
    ----------
    fun : callable
        log-posterior or log-likelihood function taking a vector of parameters as its first argument and its derivative
        if `jac` is not given
    args : array_like
        additional arguments to pass to `fun`
    parameter_names : list
        list of parameter names
    break_on_interrupt : bool
        stop the sampler rather than throwing an exception upon a keyboard interrupt
    jac : callable
        derivative of `fun`. If `jac` is not given, `fun` must return the function value and the derivative.
    mass : array_like
        mass (vector) associated with each parameter or the variance of the momenta
    epsilon : float
        size of the leapfrog steps
    leapfrog_steps : int
        number of leapfrog steps for each sample
    """
    def __init__(self, fun, args=None, parameter_names=None, break_on_interrupt=True, jac=None, mass=1.0, epsilon=0.1,
                 leapfrog_steps=10):
        super(HamiltonianSampler, self).__init__(fun, args, parameter_names, break_on_interrupt)
        self.jac = jac
        if isinstance(mass, str):
            self.mass = np.loadtxt(mass)
        else:
            self.mass = np.asarray(mass)
        assert np.all(np.isfinite(mass)), '`mass` must be finite'
        self.epsilon = epsilon
        self.leapfrog_steps = leapfrog_steps

        # Do some sanity checks
        assert callable(jac), "`jac` must be callable"
        assert self.mass.ndim < 3, "`mass` must be a scalar, vector, or matrix"

        # Compute the inverse mass
        if self.mass.ndim < 2:
            self.inv_mass = 1.0 / self.mass
        else:
            self.inv_mass = np.linalg.inv(self.mass)

    def evaluate_kinetic(self, momentum):
        """
        Evaluate the kinetic energy.

        Parameters
        ----------
        momentum : array_like
            momentum of the parameters in the parameter space
        """
        if self.mass.ndim < 2:
            return -0.5 * np.sum(momentum ** 2 * self.inv_mass)
        else:
            return -0.5 * momentum.dot(self.inv_mass).dot(momentum)

    def estimate_mass(self):
        """
        Estimate the mass matrix from the samples.

        Notes
        -----
        The optimal mass matrix is the inverse sample covariance (see Gelman et al.).
        """
        if len(self._samples) < 2:
            warn('the number of samples is too small to estimate the mass matrix')
            return 1.0

        cov = np.cov(self.samples, rowvar=False)
        mass = np.linalg.inv(cov)
        return mass

    def save_mass(self, filename):
        """
        Save an estimate of the optimal mass matrix to a file.

        Parameters
        ----------
        filename : str
            the filename to save the mass matrix to
        """
        mass = self.estimate_mass()
        np.savetxt(filename, mass)

    def sample(self, parameters, steps=1, callback=None, full=False, epsilon=None, leapfrog_steps=None):
        """
        Draw samples from the distribution.

        Parameters
        ----------
        parameters : array_like
            current state of the chain
        steps : int, iterable
            number of steps
        callback : callable
            callback after each step
        full : bool
            whether to return a full report of the dynamics (only allowed if `steps == 1`)
        epsilon : float
            size of the leapfrog steps (default uses the value passed to `__init__`)
        leapfrog_steps : int
            number of leapfrog steps for each sample (default uses the value passed to `__init__`)
        """
        # Basic setup of parameters
        parameters = np.asarray(parameters)
        p = len(parameters)
        leapfrog_steps = leapfrog_steps or self.leapfrog_steps
        epsilon = epsilon or self.epsilon

        try:
            for step in steps if hasattr(steps, '__iter__') else range(steps):
                # Sample the momentum
                if self.mass.ndim < 2:
                    momentum = np.random.normal(0, 1, p) * np.sqrt(self.mass)
                else:
                    momentum = np.random.multivariate_normal(np.zeros(p), self.mass)
                # Compute the kinetic energy
                kinetic = self.evaluate_kinetic(momentum)

                # Evaluate the Jacobian
                if self.jac:
                    jac = self.jac(parameters, *self.args)
                    fun_value = self.fun(parameters, *self.args)
                else:
                    fun_value, jac = self.fun(parameters, *self.args)

                # Evolve the variables
                fun_value_end = None
                parameters_end = parameters
                # Initialise the sequence for full output
                if full:
                    params_sequence = [parameters_end.copy()]
                    energy_sequence = [(fun_value, kinetic)]

                for leapfrog_step in range(leapfrog_steps):
                    # Make a half step for the leapfrog algorithm
                    momentum += 0.5 * epsilon * jac
                    # Update the position
                    if self.mass.ndim < 2:
                        parameters_end += epsilon * self.inv_mass * momentum
                    else:
                        parameters_end += epsilon * self.inv_mass.dot(momentum)
                    # Evaluate the Jacobian
                    if self.jac:
                        jac = self.jac(parameters_end, *self.args)
                    else:
                        fun_value_end, jac = self.fun(parameters_end, *self.args)
                    # Make another half-step
                    momentum += 0.5 * epsilon * jac

                    if full:
                        # Append parameters
                        params_sequence.append(parameters_end)
                        # Evaluate the function value if necessary
                        if fun_value_end is None:
                            fun_value_end = self.fun(parameters_end, *self.args)
                        # Evaluate the kinetic energy
                        kinetic_end = self.evaluate_kinetic(momentum)
                        energy_sequence.append((fun_value_end, kinetic_end))
                        # Make sure the value gets recomputed in the next iteration
                        fun_value_end = None

                # Evaluate the function value if necessary
                if fun_value_end is None:
                    fun_value_end = self.fun(parameters_end, *self.args)

                # Evaluate the kinetic energy
                kinetic_end = self.evaluate_kinetic(momentum)

                # Accept or reject the step
                if np.log(np.random.uniform()) < fun_value_end + kinetic_end - fun_value - kinetic:
                    parameters = parameters_end
                    fun_value = fun_value_end

                self._samples.append(parameters.copy())
                self._fun_values.append(fun_value)

                if callback:
                    callback(parameters)
        except KeyboardInterrupt as ex:
            logger.info('sampling cancelled by keyboard interrupt after %d steps', step)
            # Reraise the exception if we are not just breaking on interrupt
            if not self.break_on_interrupt:
                raise ex

        if not full:
            return parameters

        if steps == 1:
            return parameters, np.asarray(params_sequence), np.asarray(energy_sequence)

        raise ValueError("`full=True` is only allowed for single steps")

    def dynamics_plot(self, parameters, epsilon=None, leapfrog_steps=None):
        """
        Plot the dynamics of the parameters.

        Parameters
        ----------
        parameters : array_like
            current state of the chain
        epsilon : float
            size of the leapfrog steps (default uses the value passed to `__init__`)
        leapfrog_steps : int
            number of leapfrog steps for each sample (default uses the value passed to `__init__`)
        """
        # Run a simulation with full output
        leapfrog_steps = leapfrog_steps or self.leapfrog_steps
        epsilon = epsilon or self.epsilon
        params_end, params_sequence, energy_sequence = self.sample(parameters, 1, None, True, epsilon, leapfrog_steps)

        fig, (ax1, ax2) = plt.subplots(1, 2, True)

        # Plot the  parameter sequence
        time = np.arange(leapfrog_steps + 1) * epsilon
        for parameter in range(len(parameters)):
            ax1.plot(time, params_sequence[:, parameter], label=self.get_parameter_name(parameter))

        # Plot the energy
        potential, kinetic = np.transpose(energy_sequence)
        ax2.plot(time, potential - np.max(potential), label='potential')
        ax2.plot(time, kinetic, label='kinetic')
        ax2.plot(time, potential + kinetic - np.max(potential), label='total')

        unit_steps = 1
        while unit_steps < np.max(time):
            ax1.axvline(unit_steps, color='k', ls='dotted')
            ax2.axvline(unit_steps, color='k', ls='dotted')
            unit_steps += 1

        ax1.legend(loc=0, frameon=False)
        ax2.legend(loc=0, frameon=False)

        return fig, (ax1, ax2)
