__author__ = 'Sebi'

from .reactions import RateFunction
from .solver.ssa import Solver as cySolver
from .timeseries import TimeSeries
from .parameters import conditions
from copy import deepcopy
import numpy as np
import types

#import warnings
#warnings.filterwarnings('error')


class Integrator:
    """
    Class defines determinsitic ODE integration procedure.
    """
    def __init__(self, network, condition=None):
        """
        Parameters:
            network (Network object)
            condition (str) - environmental conditions affecting rate. either None, 'cold', 'hot', 'diabetic', or 'minute'
        """
        self.network = network
        self.num_reactions = len(network.reactions)
        self.network.sort_rxns()

        # apply any rate scaling conditions
        if condition is not None:
            if condition not in conditions.keys():
                raise ValueError('Condition not recognized.')
            else:
                self.apply_rate_scaling(condition)


    def apply_rate_scaling(self, condition):
        """
        Applies rate scaling to all condition-sensitive reactions:

        Parameters:
            condition (str) - environmental conditions affecting rate. either None, 'cold', 'hot', 'diabetic', or 'minute'
        """

        # get rate scaling factors
        rate_scaling = conditions[condition]

        # apply scaling factors
        if 'temperature' in rate_scaling.keys():
            for rxn in self.network.reactions:
                rxn.rate_constant *= rate_scaling['temperature']**rxn.temperature_sensitive

        if 'metabolic_rate' in rate_scaling.keys():
            for rxn in self.network.reactions:
                rxn.rate_constant *= rate_scaling['metabolic_rate']**rxn.atp_sensitive

        if 'translation_capacity' in rate_scaling.keys():
            for rxn in self.network.reactions:
                rxn.rate_constant *= rate_scaling['translation_capacity']**rxn.ribosome_sensitive

    def odeint(self, input_function=None, ic=None, dt=1, duration=100):
        """
        Simulates dynamic system using Scipy's ode object.
        """
        from scipy.integrate import odeint

        # if no initial condition is provided, assume all states are zero
        if ic is None:
            ic = np.zeros(self.network.nodes.size, dtype=np.int64)
        elif type(ic) == list or type(ic) == tuple:
            ic = np.array(ic, dtype=np.int64)
        if len(ic) != self.network.nodes.size:
            raise RuntimeError('IC dimensions inconsistent with system.')

        times = np.arange(0, duration, dt)
        rate_function = RateFunction(self.network)

        def derivative(x, t):
            dxdt = rate_function(x, input_function(t))
            return dxdt

        # run solver and compile timeseries
        solout = odeint(derivative, y0=ic, t=times)
        timeseries = TimeSeries(times, solout.reshape(1, *solout.T.shape))
        return timeseries

    def solve_ivp(self, input_function=None, ic=None, duration=100, method='BDF'):
        """
        Simulates dynamic system using Scipy's ode object.
        """
        from scipy.integrate import solve_ivp

        # if no initial condition is provided, assume all states are zero
        if ic is None:
            ic = np.zeros(self.network.nodes.size, dtype=np.int64)
        elif type(ic) == list or type(ic) == tuple:
            ic = np.array(ic, dtype=np.int64)
        if len(ic) != self.network.nodes.size:
            raise RuntimeError('IC dimensions inconsistent with system.')

        rate_function = RateFunction(self.network)

        def derivative(t, x):
            dxdt = rate_function(x, input_function(t))
            return dxdt

        # run solver and compile timeseries
        solout = solve_ivp(derivative, (0, duration), ic, method=method)

        t, y = solout['t'], solout['y']
        timeseries = TimeSeries(t, y.reshape(1, *y.shape))

        return timeseries


class Simulation(Integrator):
    """
    Class defines a simulation procedure.
    """

    def __init__(self, network, condition=None):
        """
        Parameters:
            network (Network object)
            condition (str) - environmental conditions affecting rate. either None, 'cold', 'hot', 'diabetic', or 'minute'
        """
        Integrator.__init__(self, network, condition)
        self.solver = cySolver(self.network)

    def simulate(self, ic=None, input_function=None, integrator_ic=None, dt=1, duration=100, pure_ssa=True):
        """
        Simulates dynamic system using stochastic solver.

        Parameters:
            ic (np array) - initial conditions, defaults to zero
            input_function (function) - returns input value(s) for given time
            integrator_ic (np array) - initial condition for integrator
            dt (float) - time step
            duration (float) - simulation end time
            pure_ssa (bool) - if True, use pure ssa

        Returns:
            times (np array) - timepoints (1 by t)
            states (np array) - state values at each time point (N by t)
        """

        # sort reactions
        #self.network.sort_rxns()
        #self.network.compile_stoichiometry()
        #self.network.compile_rate_dependencies()

        # if no initial condition is provided, assume all states are initially zero
        if ic is None:
            ic = np.zeros(self.network.nodes.size, dtype=np.int64)
        elif type(ic) == list or type(ic) == tuple:
            ic = np.array(ic, dtype=np.int64)
        if len(ic) != self.network.nodes.size:
            raise RuntimeError('IC dimensions inconsistent with system.')

        # run solver
        solout = self.solver.solve(ic=ic, input_function=input_function, integrator_ic=integrator_ic, dt=dt, duration=duration, use_pure_ssa=pure_ssa)

        return solout


class MonteCarloSimulation(Simulation):
    """
    Class defines multiple trials of a simulation procedure.
    """

    def __init__(self, system, ic=None, integrator_ic=None, condition=None):
        """
        Each instance has a fixed set of simulation parameters for which samples are generated.

        Parameters:
            system (network object)
            ic (np array) - initial conditions, defaults to zero
            integrator_ic (np array) - integrator initialization

            condition (str) - environmental conditions affecting rates
        """

        Simulation.__init__(self, system, condition)

        # set initial condition generating functions
        self.set_initial_conditions(ic, integrator_ic)

    def __repr__(self):
        self.network.print_reactions()

    def set_initial_conditions(self, ic=None, integrator_ic=None):
        self.get_ic = self.get_ic_generator(ic)
        self.get_integrator_ic = self.get_ic_generator(integrator_ic)

    @staticmethod
    def get_ic_generator(ic):
        """ Create initial condition generating function. """
        if type(ic) == tuple:
            m, v = ic
            dim = system.nodes.size
            f = lambda n: np.random.normal(m, np.sqrt(v), size=(n, dim)).astype(int)
        elif type(ic) == types.FunctionType:
            f = lambda n: [ic() for _ in range(n)]
        else:
            f = lambda n: [ic for _ in range(n)]
        return f

    def run(self, input_function=None, num_trials=100, duration=100, dt=1, pure_ssa=True):
        """
        Runs multiple monte carlo trials and stores results.

        Parameters:
            input_function (function) - generates input value(s) for each time
            num_trials (int) - number of independent trials
            duration (float) - simulation end time
            dt (float) - time step
            pure_ssa (bool) - if True, use pure Gillespie ssa

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
            if (ic < 0).sum() != 0:
                raise ValueError('Failed at trial {:d} with IC {}'.format(trial, ic))

            # run simulation
            solout = self.simulate(ic=ic, input_function=input_function, integrator_ic=integrator_ic, dt=dt, duration=duration, pure_ssa=pure_ssa)
            t, s = solout
            samples.append(s)

        # instantiate time series object
        timeseries = TimeSeries(t, np.array(samples))

        return timeseries
