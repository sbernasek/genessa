from os.path import exists, join, isdir
from os import mkdir
import numpy as np


class TimeSeries:
    """
    Base class defining a collection of dynamic trajectories.

    Attributes:

        t (np.ndarray[double]) - timepoints

        states (np.ndarray[int]) - state values, shape (num_trials, N, t)

    Properties:

        mean (np.ndarray[double]) - mean across trajectories, shape (N, t)

        var (np.ndarray[double]) - variance across trajectories, shape (N, t)

        peak_indices (np.ndarray[int]) - indices of maxima, length N

        peaks (np.ndarray[int]) - maxima, length N

    """

    def __init__(self, times, states):
        """
        Args:

            t (np.ndarray[double]) - timepoints

            states (np.ndarray[int]) - state values, shape (num_trials, N, t)

        """
        self.t = times
        self.states = states

    @property
    def N(self):
        """ Number of state dimensions. """
        return self.states.shape[1]

    @property
    def mean(self):
        """ Mean value for each dimension at each timepoint. """
        return self.states.mean(axis=0)

    @property
    def var(self):
        """ Variance for each dimension at each timepoint. """
        var = self.states.var(axis=0)
        var[var == 0] = 1e-30
        return var

    @property
    def peak_indices(self):
        """ Indices of maximum mean values for each dimension. """
        return self.mean.argmax(axis=1)

    @property
    def peaks(self):
        """ Maximum mean values for each dimension. """
        return self.mean[np.arange(self.N), self.peak_indices]

    @property
    def initial(self):
        """ Returns vector of mean initial state values. """
        return self.mean[:, 0]

    @property
    def final(self):
        """ Returns vector of mean final state values. """
        return self.mean[:, -1]

    @property
    def lower(self):
        """ Lower bound of trajectories. """
        return self.states.min(axis=0)

    @property
    def upper(self):
        """ Upper bound of trajectories. """
        return self.states.max(axis=0)

    @staticmethod
    def _index(value, array):
        """
        Returns nearest indices of specified value in an array. Excludes endpoints.

        Args:

            value (float) - target value

            array (np.ndarray[float]) - 1D vector of values

        Returns:

            indices (np.ndarray[int]) - indices of value in array

        """
        return np.diff(np.sign(array-np.ones(array.size)*value)).nonzero()[0]

    def index(self, value, dim, mode='mean'):
        """
        Returns nearest indices of specified value in the specified array. Excludes endpoints.

        Args:

            value (float) - target value

            dim (int) - state dimension to be indexed

            mode (str) - array to be indexed, e.g. 'lower', 'mean', or 'upper'

        Returns:

            indices (np.ndarray[int]) - indices of value in specified array

        """

        # get array
        if mode == 'mean':
            array = self.mean[dim]
        elif mode == 'lower':
            array = self.lower[dim]
        elif mode == 'upper':
            array = self.upper[dim]

        return self._index(value, array)

    @classmethod
    def load(cls, path):
        """
        Load from file.

        Args:

            path (str) - path to saved object

        Returns:

            timeseries (TimeSeries derivative)

        """
        times = np.load(join(path, 'times.npy'))
        states = np.load(join(path, 'states.npy'))
        return cls(times, states)

    def save(self, path):
        """
        Save to file.

        Args:

            path (str)

        """

        # make a directory
        if not isdir(path):
            mkdir(path)

        # save trajectories
        np.save(join(path, 'times.npy'), self.states)
        np.save(join(path, 'states.npy'), self.states)

    def evaluate_quantile(self, q):
        """
        Returns specified quantile for a specified dimension.

        Args:

            q (float) - quantile of distribution, 0 to 100

        Returns:

            trajectory (np.ndarray) - time series for specified quantile

        """
        return np.percentile(self.states, q=q, axis=0)

    def get_deviations(self, values=None):
        """
        Returns new timeseries of deviations about a specified value.

        Args:

            values (np.ndarray[float]) - state valus used to compute deviations

        Returns:

            model (TimeSeries derivative) - timeseries for deviations

        """

        # if no values provided, use mean initial values
        if values is None:
            values = self.initial
        values = values.reshape(-1, 1)

        # convert means and states to deviation form
        self.states - values

        # construct new model
        model = self.__class__(self.times, self.states-values)

        return model
