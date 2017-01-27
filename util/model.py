import tensorflow as tf
import numpy as np

from .util import iterable


class Model:
    """
    Base class for tensorflow models with a single tensor of parameters.

    Parameters
    ----------
    parameters : np.ndarray
        parameters for the model
    prior : str or callable
        prior to apply to parameters
    learning_rate : float
        learning rate of the optimizer used to maximize the posterior
    """
    def __init__(self, parameters, prior='flat', learning_rate=0.01, floatX=tf.float32):
        parameters = np.asarray(parameters)
        self.floatX = floatX
        # Store parameters
        self._prior = prior
        self._learning_rate = learning_rate
        # Build the graph
        self.graph = self.create_graph()
        with self.graph.as_default():
            # Initialise parameters
            self.parameters = tf.Variable(parameters, name='parameters', dtype=self.floatX)
            self.likelihood = self.build_likelihood()
            self.prior = self.build_prior()
            self.posterior = tf.add(self.likelihood, self.prior, 'posterior')

            # Add the posterior gradient
            self.posterior_grad, = tf.gradients(self.posterior, self.parameters)

            # Add the posterior covariance
            if parameters.ndim == 1:
                self.posterior_hess, = tf.hessians(self.posterior, self.parameters)
                self.posterior_cov = - tf.matrix_inverse(self.posterior_hess)

            # Set up the optimizer
            self.optimizer = self.create_optimizer()
            self.train_op = self.optimizer.minimize(-self.posterior)

        self.session = self.create_session()
        self.initialize_variables()

    def __del__(self):
        # Close the associated session cleanly
        session = getattr(self, 'session', None)
        if session:
            session.close()

    def create_graph(self):
        """Create a tensorflow graph."""
        return tf.Graph()

    def create_session(self):
        """
        Create a tensorflow session.
        """
        return tf.Session(graph=self.graph, config=self.session_config)

    def initialize_variables(self):
        with self.graph.as_default():
            init_op = tf.global_variables_initializer()
            self.session.run(init_op)

    @property
    def session_config(self):
        """tf.ConfigProto : session configuration"""
        gpu_options = tf.GPUOptions(allow_growth=True)
        return tf.ConfigProto(gpu_options=gpu_options)

    def build_prior(self):
        """
        Build the log prior graph.

        Returns
        -------
        prior : tf.Tensor
            log prior for the parameters
        """
        if callable(self._prior):
            return self._prior(self.parameters)
        elif self._prior == 'flat':
            return tf.constant(0, dtype=self.floatX, name='flat_prior')
        else:
            raise KeyError(self._prior)

    def build_likelihood(self):
        """
        Build the log likelihood graph.

        Returns
        -------
        likelihood : tf.Tensor
            log likelihood for the data given the parameters
        """
        raise NotImplementedError

    def create_optimizer(self):
        """
        Create a tensorflow optimizer.
        """
        self.learning_rate = tf.Variable(self._learning_rate, False, name='learning_rate', dtype=self.floatX)
        return tf.train.AdamOptimizer(self.learning_rate)

    def as_operation(self, operation):
        """
        Ensure the input is a tensorflow operation.

        Parameters
        ----------
        operation : tf.Tensor or str
            input to convert to an operation if necessary

        Returns
        -------
        operation : tf.Tensor or str
            operation
        """
        return getattr(self, operation) if isinstance(operation, str) and hasattr(self, operation) else operation

    def run(self, fetches, feed_dict=None, **kwargs):
        """
        Run one or more operations.

        Parameters
        ----------
        fetches : str, tf.Tensor or list
            one or more operations to fetch
        feed_dict : dict
            dictionary of values keyed by operations or operation names to substitute
        kwargs : dict
            convenience interface to fill the `feed_dict`
        """
        # Convert the fetches to operations
        if iterable(fetches):
            fetches = [self.as_operation(fetch) for fetch in fetches]
        else:
            fetches = self.as_operation(fetches)

        # Update the feed dict
        feed_dict = feed_dict or {}
        for key, value in kwargs.items():
            assert key not in feed_dict, "'%s' is defined in both `feed_dict` and in `kwargs`" % key
            feed_dict[key] = value

        # Map the feed dict
        feed_dict = {self.as_operation(key): value for key, value in (feed_dict or {}).items()}

        # Run the tensorflow session
        return self.session.run(fetches, feed_dict)

    def evaluate_posterior(self, parameters=None):
        """
        Evaluate the posterior of the model.

        Parameters
        ----------
        parameters : np.ndarray
            parameters at which to evaluate the posterior or `None` to use the current state

        Returns
        -------
        posterior : float
            posterior of the model
        """
        feed_dict = {} if parameters is None else {self.parameters: parameters}
        return self.session.run(self.posterior, feed_dict)

    def optimize(self, steps=1, feed_dict=None, **kwargs):
        """
        Take one or more steps to maximize the posterior.

        Parameters
        ----------
        steps : int
            how many steps to take
        feed_dict : dict
            dictionary of values keyed by operations or operation names to substitute
        kwargs : dict
            convenience interface to fill the `feed_dict`

        Returns
        -------
        parameters : np.ndarray
            parameter values after the optimization step
        posterior : float
            posterior after the optimization step
        """
        parameters, posterior = None, None
        for _ in range(steps):
            _, parameters, posterior = self.run(['train_op', 'parameters', 'posterior'], feed_dict, **kwargs)
            assert np.all(np.isfinite(parameters)), "not all parameters are finite"
            assert np.isfinite(posterior), "the posterior is not finite"
        return parameters, posterior
