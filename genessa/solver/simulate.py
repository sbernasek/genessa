import numpy as np
import types
from array import array
from copy import deepcopy

#import warnings
#warnings.filterwarnings('error')

from .ssa import cSSA
from .timeseries import TimeSeries
from ..parameters import conditions
from ..signals.signals import cSignal


class Simulation:
    """
    Python wrapper for SSA solver.
    """

    def __init__(self, network, condition):
        """
        Instantiate SSA solver.

        Args:

            network (Network)

            condition (str) - environmental conditions affecting rates

        """

        network.sort_rxns()

        # apply any rate scaling conditions
        if condition is not None:
            if condition not in conditions.keys():
                raise ValueError('Condition not recognized.')
            else:
                self.apply_rate_scaling(network, condition)

        # store system dimensions
        self.N = network.N
        self.M = network.M
        self.I = network.I

        # instantiate cython solver
        self.cSSA = cSSA.from_network(network)

    @staticmethod
    def apply_rate_scaling(network, condition):
        """
        Applies rate scaling to all condition-sensitive reactions in a network.

        Args:

            network (Network)

            condition (str) - environmental conditions affecting rates

        """

        # get rate scaling factors
        rate_scaling = conditions[condition]

        # apply scaling factors
        if 'temperature' in rate_scaling.keys():
            for rxn in network.reactions:
                rxn.rate_constant *= rate_scaling['temperature']**rxn.temperature_sensitive

        if 'metabolic_rate' in rate_scaling.keys():
            for rxn in network.reactions:
                rxn.rate_constant *= rate_scaling['metabolic_rate']**rxn.atp_sensitive

        if 'translation_capacity' in rate_scaling.keys():
            for rxn in network.reactions:
                rxn.rate_constant *= rate_scaling['translation_capacity']**rxn.ribosome_sensitive

    def odeint(self,
               input_function=None,
               ic=None,
               dt=1,
               duration=100):
        """
        Simulates dynamic system using Scipy's ode object.

        Args:

            input_function (function) - returns input value(s) for given time

            ic (np array) - initial conditions, defaults to zero

            dt (float) - time step

            duration (float) - simulation end time

        Returns:

            timeseries (TimeSeries)

        """
        from scipy.integrate import odeint

        # if no initial condition is provided, assume all states are zero
        if ic is None:
            ic = np.zeros(self.N, dtype=np.uint32)
        elif type(ic) == list or type(ic) == tuple:
            ic = np.array(ic, dtype=np.uint32)
        if len(ic) != self.N:
            raise RuntimeError('IC dimensions inconsistent with system.')

        # if no input function, use zeros
        if input_function is None:
            input_value = lambda t: array('d', np.zeros(self.I, dtype=np.float64))
        else:
            input_value = lambda t: input_function(t)

        # cumulative is not supported
        cumulative = array('d', np.zeros(self.N, dtype=np.float64))

        # define rate function
        def derivative(x, t):
            states = array('d', x)
            inputs = input_value(t)
            dxdt = self.cSSA.evaluate_species_rates(states, inputs, cumulative)
            return dxdt

        # run solver and compile timeseries
        times = np.arange(0, duration, dt)
        solout = odeint(derivative, y0=ic, t=times)
        timeseries = TimeSeries(times, solout.reshape(1, *solout.T.shape))
        return timeseries

    def solve_ivp(self,
                  input_function=None,
                  ic=None,
                  dt=1,
                  duration=100,
                  method='BDF'):
        """
        Simulates dynamic system using Scipy's ode object.

        Args:

            input_function (function) - returns input value(s) for given time

            ic (np array) - initial conditions, defaults to zero

            dt (float) - time step

            duration (float) - simulation end time

            method (str) - solver method

        Returns:

            timeseries (TimeSeries)

        """
        from scipy.integrate import solve_ivp
        from scipy.interpolate import interp1d

        # if no initial condition is provided, assume all states are zero
        if ic is None:
            ic = np.zeros(self.N, dtype=np.uint32)
        elif type(ic) == list or type(ic) == tuple:
            ic = np.array(ic, dtype=np.uint32)
        if len(ic) != self.N:
            raise RuntimeError('IC dimensions inconsistent with system.')

        # if no input function, use zeros
        if input_function is None:
            input_value = lambda t: array('d', np.zeros(self.I, dtype=np.float64))
        else:
            input_value = lambda t: input_function(t)

        # cumulative is not supported
        cumulative = array('d', np.zeros(self.N, dtype=np.float64))

        # define rate function
        def derivative(t, x):
            states = array('d', x)
            inputs = input_value(t)
            dxdt = self.cSSA.evaluate_species_rates(states, inputs, cumulative)
            return dxdt

        # run solver and compile timeseries
        solout = solve_ivp(derivative, (0, duration), ic, method=method)
        t, y = solout['t'], solout['y']
        interpolator = interp1d(t, y)
        times = np.arange(0, duration, dt)
        timeseries = TimeSeries(times, interpolator(times).reshape(1, self.N, times.size))

        return timeseries

    def ssa(self,
            ic,
            input_function=None,
            integrator_ic=None,
            dt=1.,
            duration=100.):
        """
        Run stochastic simulation.

        Args:

            ic (np array) - initial conditions, defaults to zero

            input_function (function) - returns input value(s) for given time

            integrator_ic (np array) - initial condition for integrator

            dt (float) - time step

            duration (float) - simulation end time

        Returns:

            times (np.ndarray) - timepoints (1 by t)

            states (np.ndarray) - state values at each time point (N by t)

        """

        # if no initial condition provided, assume initial states are all zero
        if ic is None:
            ic = np.zeros(self.N, dtype=np.uint32)
        elif type(ic) == list or type(ic) == tuple:
            ic = np.array(ic, dtype=np.uint32)
        else:
            ic = ic.astype(np.uint32)

        # check that initial condition has the correct dimensions
        assert (len(ic)==self.N), 'Wrong IC dimensions.'

        # set input function
        if input_function is None:
            input_function = cSignal([0 for _ in range(self.I)])
        else:
            input_function = deepcopy(input_function)

        # set initial condition for integrator
        if integrator_ic is None:
            integrator_ic = np.zeros(self.N, dtype=np.float64)
        else:
            integrator_ic = integrator_ic.astype(np.float64)

        # run SSA
        times, states = self.cSSA.run(ic,
                               input_function,
                               integrator_ic,
                               dt=dt,
                               duration=duration)

        return times, states


class MonteCarloSimulation(Simulation):
    """
    Class defines multiple trials of a simulation procedure.
    """

    def __init__(self,
                 network,
                 ic=None,
                 integrator_ic=None,
                 condition=None):
        """
        Each instance has a fixed set of simulation parameters for which samples are generated.

        Args:

            network (Network)

            ic (array like) - initial conditions, defaults to zero

            integrator_ic (array like) - integrator initialization

            condition (str) - environmental condition affecting rates

        """

        # instantiate simulation
        super().__init__(network, condition)

        # set initial condition generating functions
        self.set_initial_conditions(ic, integrator_ic)

    def set_initial_conditions(self,
                               ic=None,
                               integrator_ic=None):
        """
        Set initial conditions.

        Args:

            ic (array like or tuple) - initial state

            integrator_ic (array like or tuple) - initial integrator state

        """

        if ic is None:
            ic = np.zeros(self.N)

        self.get_ic = self.build_ic_generator(ic, self.N)
        self.get_integrator_ic = self.build_ic_generator(integrator_ic, self.N)

    @staticmethod
    def build_ic_generator(ic, N):
        """
        Build initial condition generating function.

        Args:

            ic (array like or tuple) - initial condition

            N (int) - number of states

        Returns:

            f (func) - returns vector of initial values

        """
        if type(ic) == tuple:
            m, v = ic
            f = lambda n: np.random.normal(m, np.sqrt(v), size=(n, N)).astype(int)
        elif type(ic) == types.FunctionType:
            f = lambda n: [ic() for _ in range(n)]
        else:
            f = lambda n: [ic for _ in range(n)]
        return f

    def run(self,
            input_function=None,
            num_trials=100,
            duration=100,
            dt=1):
        """
        Runs multiple monte carlo trials and stores results.

        Args:

            input_function (function) - generates input value(s) for each time

            num_trials (int) - number of independent trials

            duration (float) - simulation end time

            dt (float) - time step

        Returns:

            timeseries (TimeSeries) - system state dynamics

        """

        # get initial conditions
        ics = self.get_ic(num_trials)
        integrator_ics = self.get_integrator_ic(num_trials)

        # run each trial, appending sample to a list of all samples
        samples = []
        for trial in range(num_trials):

            # get initial condition and check that it's positive
            ic = ics[trial]
            integrator_ic = integrator_ics[trial]

            # check that initial condition is positive
            assert (ic<0).sum() == 0, 'Failed on Trial {:d}'.format(trial)

            # run simulation
            times, states = self.ssa(ic=ic,
                                     input_function=input_function,
                                     integrator_ic=integrator_ic,
                                     dt=dt,
                                     duration=duration)
            samples.append(states)

        # instantiate time series object
        timeseries = TimeSeries(times, np.array(samples))

        return timeseries
