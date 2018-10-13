import numpy as np
from scipy.stats import norm

from .base import TimeSeries


class GaussianModel(TimeSeries):

    """
    Class defines a collection of dynamic trajectories with a gaussian model fit to them.

    Attributes:

        norm (scipy.stats.norm) - gaussian fit to dynamic trajectories

        bandwidth (float) - width of confidence band, 0 to 1

    Inherited attributes:

        t (np.ndarray[double]) - timepoints

        states (np.ndarray[int]) - state values, shape (num_trials, N, t)

    Properties:

        mean (np.ndarray[double]) - mean across trajectories, shape (N, t)

        var (np.ndarray[double]) - variance across trajectories, shape (N, t)

        peak_indices (np.ndarray[int]) - indices of maxima, length N

        peaks (np.ndarray[int]) - maxima, length N

    """

    def __init__(self, times, states, bandwidth=0.98):
        """
        Instantiate gaussian fit to dynamic trajectories.

        Args:

            t (np.ndarray[double]) - timepoints

            states (np.ndarray[int]) - state values, shape (num_trials, N, t)

            bandwidth (float) - width of confidence band, 0 to 1

        """
        super().__init__(times, states)
        self.fit_gaussian()
        self.bandwidth = bandwidth

    @property
    def lower(self):
        """ Lower bound of trajectories. """
        return self.norm.ppf((1-self.bandwidth)/2)

    @property
    def upper(self):
        """ Upper bound of trajectories. """
        return self.norm.ppf((1+self.bandwidth)/2)

    @classmethod
    def from_timeseries(cls, timeseries, bandwidth=0.98):
        """ Instantiate from TimeSeries. """
        return cls(timeseries.t, timeseries.states, bandwidth=bandwidth)

    def fit_gaussian(self):
        """ Fits a gaussian across each dimension at each time point. """
        self.norm = norm(loc=self.mean, scale=np.sqrt(self.var))

    def evaluate_quantile(self, q):
        """
        Returns specified quantile for a specified dimension.

        Args:

            q (float) - quantile of distribution, 0 to 100

        Returns:

            trajectory (np.ndarray) - time series for specified quantile

        """
        return self.norm.ppf(q/100)

    def evaluate_cdf(self, threshold):
        """
        Returns CDF for a specified dimension at a specified threshold value.

        Args:

            threshold (np.ndarray[float] or float) - threshold(s) values

        Returns:

            cdf (np.ndarray[float]) - cdf values

        """

        return self.norm.cdf(threshold)
