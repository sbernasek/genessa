import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


class TimeSeries:
    """
    Class defines a collection of time series.

    Attributes:

        t (np.ndarray[double]) - timepoints

        states (np.ndarray[int]) - state values, shape (num_trials, N, t)

        mean (np.ndarray[double]) - mean across trajectories, shape (N, t)

        var (np.ndarray[double]) - variance across trajectories, shape (N, t)

    """

    def __init__(self, times, states):
        """
        Args:

            t (np.ndarray[double]) - timepoints

            states (np.ndarray[int]) - state values, shape (num_trials, N, t)

        """
        self.t = times
        self.states = states
        self.mean = self.get_mean(states)
        self.var = self.get_variance(states)

    def to_json(self, retall=False):
        """
        Serialize TimeSeries.

        Args:

            retall (bool) - if True, write states to file

        Returns:

            js (dict) - json serialization of TimeSeries

        """
        js =  {
            't': self.t.tolist(),
            'mean': self.mean.tolist(),
            'var': self.var.tolist()}

        if retall is True:
            js['states'] = self.states.tolist()

        return js

    @staticmethod
    def from_json(js):
        """
        Instantiate from serialized TimeSeries.

        Args:

            js (dict) - deserialized json object

        Returns:

            timeseries (TimeSeries)

        """

        # unpack serialized values
        t = np.array(js['t'])

        # if states are included, unpack them
        if 'states' in js.keys():
            states = np.array(js['states'])
        else:
            raise ValueError('State trajectories were not found. Consider using a GaussianModel.')

        # instantiate recovered time series
        timeseries = TimeSeries(t=t, states=states)

        return timeseries

    @staticmethod
    def get_mean(states):
        """ Returns mean trajectory for each state dimension. """
        return states.mean(axis=0)

    @staticmethod
    def get_variance(states):
        """ Returns trajectory variance for each state dimension. """
        var = states.var(axis=0)
        var[var == 0] = 1e-30
        return var

    def get_time_of_value(self, dim, percentile, value):
        """
        Returns time (and index in time vector) at which specified percentile of trajectory distribution first reaches
        the specified value.

        Args:

            dim (int) - dimension of state space

            percentile (float) - percentile of distribution, 0 to 100

            value (float) - target value

        Returns:

            index (int) - index of of timepoint

            time (float) - time at which value occurs

        """

        index = (abs(np.percentile(self.states[:, dim, :], q=percentile, axis=0)-value)).argmin()
        time = self.t[index]
        return index, time

    def get_percentile(self, dim, percentile):
        """
        Returns desired percentile for the specified dimension.

        Args:

            dim (int) - dimension of state space

            percentile (float) - percentile of distribution, 0 to 100

        Returns:

            trajectory (np.ndarray) - time series for specified percentile

        """
        trajectory = np.percentile(self.states[:, dim, :], q=percentile, axis=0)
        return trajectory

    def plot(self,
             dims=None,
             samples=True,
             mean=False,
             interval=False,
             colors=None,
             ax=None):
        """
        Plot time series for a specified dimension.

        Args:

            dims (list) - dimensions of state space to be plotted

            samples (bool) - if True, plot individual trajectories

            mean (bool) - if True, plot mean trajectory

            interval (bool) - if True, plot SEM

            colors (list) - colors for each dimension

            ax (axes object) - if None, create one

        """

        # if no dimension specified, select highest
        if dims is None:
            dims = (self.states.shape[1] - 1,)

        # define color cycle
        if colors is None:
            colors = ('c', 'm', 'y', 'k', 'r', 'g', 'b')

        # create axes
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))

        # iterate across all specified dimensions
        for i, dim in enumerate(dims):

            color = colors[i % len(colors)]

            # add trajectories to plot
            if self.states is None:
                raise AttributeError('Trajectory data not available.')

            # plot samples
            if samples:
                for trajectory in self.states[:, dim, :]:
                    ax.plot(self.t, trajectory, '-', color=color, lw=1, alpha=0.5)

            # plot SEM interval
            if interval:
                sem = self.var[dim, :]/self.states.shape[0]
                lbound = self.mean[dim, :] - sem
                ubound = self.mean[dim, :] + sem
                ax.fill_between(self.t, lbound, ubound, color=color, alpha=0.5)

            # plot mean
            if mean:
                ax.plot(self.t, self.mean[dim, :], '-', color=color, lw=1, alpha=1)


class GaussianModel(TimeSeries):
    """
    Class inherits a time series and fits a gaussian parametric model.

    Inherited Attributes:

        t (np.ndarray[double]) - timepoints

        states (np.ndarray[int]) - state values, shape (num_trials, N, t)

        mean (np.ndarray[double]) - mean across trajectories, shape (N, t)

        var (np.ndarray[double]) - variance across trajectories, shape (N, t)

    """

    def __init__(self, t, mean, var, states=None):
        self.t = t
        self.mean = mean
        self.var = var
        self.states = states

        # fit gaussian
        self.fit_gaussian_model()

        # find peaks
        self.peak_indices = self.mean.argmax(axis=1)
        self.peaks = self.mean[range(self.mean.shape[0]), self.peak_indices]

    @staticmethod
    def from_timeseries(timeseries):
        return GaussianModel(t=timeseries.t,
                             mean=timeseries.mean,
                             var=timeseries.var,
                             states=timeseries.states)

    @staticmethod
    def from_json(js):
        """
        Instantiate from serialized TimeSeries.
        """

        # unpack serialized values
        t = np.array(js['t'])
        mean = np.array(js['mean'])
        var = np.array(js['var'])

        # if states are included, unpack them
        if 'states' in js.keys():
            states = np.array(js['states'])
        else:
            states = None

        # instantiate recovered time series
        model = GaussianModel(t=t, mean=mean, var=var, states=states)

        return model

    def fit_gaussian_model(self):
        """ Constructs gaussian across each dimension at each time point. """
        self.norm = norm(loc=self.mean, scale=np.sqrt(self.var))

    def get_time_of_value(self, dim, percentile, value, after_peak=False):
        """
        Returns time (and index in time vector) at which specified percentile of trajectory distribution first reaches
        the specified value.

        Args:

            dim (int) - dimension of state space

            percentile (float) - percentile of distribution, 0 to 100

            value (float) - target value

            after_peak (bool) - if True, begin counting from peak mean value

        Returns:

            index (np.ndarray of int) - index of of timepoint

            time (float) - time at which value occurs

        """

        # get trajectory of specified percentile
        trajectory = self.norm.ppf(percentile/100)[dim]

        # if after_peak is true, slice values after peak
        index_offset = 0
        if after_peak is True:
            index_offset = self.peak_indices[dim]
            trajectory = trajectory[index_offset:]

        # compute index and time of closest-matching value
        index = np.argmin(abs(trajectory-value)) + index_offset
        time = self.t[index]

        return index, time

    def get_percentile(self, percentile):
        """
        Returns desired percentile for the specified dimension using gaussian model.

        Args:

            percentile (float) - percentile of distribution, 0 to 100

        Returns:

            trajectory (np.ndarray) - time series for specified percentile

        """
        trajectory = self.norm.ppf(percentile/100)
        return trajectory

    def evaluate_cdf(self, dim, threshold):
        """
        Evaluates cumulative distribution function at a specified threshold value.

        Args:

            dim (int) - dimension of interest

            threshold (np.ndarray or float) - threshold(s) at which cdf is evaluated

        Returns:

            cdf_vector (np.ndarray) - time series for cdf

        """

        cdf_vector = self.norm.cdf(threshold)[dim]
        return cdf_vector

    def plot_confidence_interval(self,
                                 ax=None,
                                 confidence=0.98,
                                 dims=None,
                                 colors=None,
                                 mean=False,
                                 alpha=0.5):
        """
        Plot confidence interval.

        Args:

            ax (axes object) - if None, create one

            confidence (float) - confidence interval

            dims (iterable) - dimensions of state space to be plotted

            colors (iterable) - colors for confidence interval shading

            mean (bool) - if True, plot mean

            alpha (float) - transparency of fill

        """

        # define color cycle
        if colors is None:
            colors = ('c', 'm', 'y', 'k', 'r', 'g', 'b')

        # create axes
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))

        # if no dimension specified, select highest
        if dims is None:
            dims = (self.mean.shape[0] - 1,)

        # plot confidence interval
        lbound = self.norm.ppf((1-confidence)/2)
        ubound = self.norm.ppf((1+confidence)/2)

        # iterate across dimensions
        for dim, color in zip(dims, colors):

            # add confidence interval
            ax.fill_between(self.t, lbound[dim], ubound[dim], color=color, alpha=alpha, linewidth=0)

            # add mean to plot
            if mean is True:
                ax.plot(self.t, self.mean[dim, :], '-', color=color, linewidth=1, alpha=1)

    def get_deviation_model(self, steady_state=None):
        """
        Returns GaussianModel of deviations from specified steady state.

        Args:

            steady_state (np.ndarray) - steady state for each dimension

        Returns:

            model (GaussianModel object)

        """

        # if no steady state provided, use final mean value
        if steady_state is None:
            steady_state = self.mean[:, -1].reshape(-1, 1)

        # convert means and states to deviation form
        dev_mean = self.mean - steady_state
        dev_states = self.states.astype(np.float64)
        if dev_states is not None:
            dev_states -= steady_state

        # construct new model
        model = super().__class__(t=self.t,
                                  mean=dev_mean,
                                  var=self.var,
                                  states=dev_states)

        return model

    def get_normalized_model(self, steady_state=None):
        """
        Returns GaussianModel model normalized by steady state.

        Args:

            steady_state (np.ndarray) - normalization basis

        Returns:

            model (GaussianModel object)

        """

        # if no steady state provided, use final mean value
        if steady_state is None:
            steady_state = self.mean[:, -1].reshape(-1, 1)

        # check for zero in normalizatiom basis
        if np.any(steady_state==0):
            raise ValueError('Cannot normalize by steady state of zero.')

        # convert means, variances, and states to deviation form
        norm_mean = self.mean / steady_state
        norm_var = self.var / (steady_state**2)
        norm_states = self.states.astype(np.float64)
        if norm_states is not None:
            norm_states /= steady_state

        # construct new model
        model = GaussianModel(t=self.t, mean=norm_mean, var=norm_var, states=norm_states)

        return model
