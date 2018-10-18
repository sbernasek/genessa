# cython external imports
cimport numpy as np
from cpython.array cimport array

# python external imports
import numpy as np
from copy import deepcopy
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# import intra-package cython dependencies
from .rates cimport cRates
from .stoichiometry cimport cStoichiometry
from .deterministic cimport cDeterministicSystem

# import intra-package python dependencies
from ..parameters import conditions
from ..signals.signals import cSignal
from .rates import Rates
from ..timeseries.base import TimeSeries


# ============================= CYTHON CODE ===================================


cdef class cDeterministicSystem:
    """
    Class defines a network of nodes that interact via deterministic reactions.

    Attributes:

        N (unsigned int) - number of nodes

        M (unsigned int) - number of reactions

        I (unsigned int) - number of external inputs

        S (cStoichiometry) - stoichiometry for all reactions

        R (cRates) - rate function for all reactions

    """

    def __cinit__(self, network, *args, **kwargs):
        """
        Instantiate system of reactions.

        Args:

            network (Network) - system of nodes and reactions

        """

        # sort rxns and compile stoichiometry
        network.sort_rxns()
        network.resize_inputs()
        network.compile_stoichiometry()

        # typecast network features
        self.N = network.N
        self.M = network.M
        self.I = network.I

        # get cythonized rate function and network
        self.S = cStoichiometry.from_array(network.stoichiometry)
        self.R = Rates.compile_c_rate_function(network)

    cdef double* get_rxn_rates(self):
        """ Returns current reaction rates. """
        return self.R.rates

    cdef double get_total_rxn_rate(self) nogil:
        """ Returns current total reaction rate. """
        return self.R.total_rate

    cpdef double[:] c_evaluate_species_rates(self,
        double[::1] states,
        double[::1] inputs):
        """
        Evaluate rate of change for all species. States and integrator values are combined into the same c-contiguous memory block.

        Args:

            states (double[:]) - state and integrator values, length 2N

            inputs (double[:]) - input signal values, length I

            Note: both arguments must be c-contiguous

        Returns:

            rates (double[:]) - species rates, e.g. dX/dt, length 2N

        """

        cdef unsigned int rxn
        cdef double rxn_rate
        cdef unsigned int N, index, count, species
        cdef int coefficient
        cdef double[:] rxn_rates

        # instantiate array of zeros (double length for adding integrator)
        cdef double[:] rates = np.zeros(2*self.N, dtype=np.float64)

        # evaluate reaction rates
        rxn_rates = self.R.c_evaluate_rxn_rates(states[:self.N],
                                                inputs,
                                                states[self.N:])

        # for each reaction
        for rxn in xrange(self.M):
            rxn_rate = rxn_rates[rxn]
            N = self.S.lengths[rxn]
            index = self.S.index[rxn]

            # update each active state
            for count in xrange(N):
                species = self.S.species[index]
                coefficient = self.S.coefficients[index]
                rates[species] += (coefficient * rxn_rate)
                index += 1

        # set integrator rates
        for count in xrange(self.N):
            rates[self.N+count] = states[count]

        return rates


# ============================= PYTHON CODE ===================================


class DeterministicSimulation:
    """
    Python class for simulating gene regulatory network dynamics by solving a deterministic system of ordinary differential equations.

    Attributes:

        network (Network) - python Network instance

        solver (cDeterministicSystem) - cython-based deterministic system

    Properties:

        N (unsigned int) - number of nodes

        M (unsigned int) - number of reactions

        I (unsigned int) - number of external inputs

    """

    def __init__(self, network, condition):
        """
        Instantiate deterministic simulation for a given network.

        Args:

            network (Network) - python Network instance

            condition (str) - environmental conditions affecting rates

        """

        network = deepcopy(network)

        # apply any rate scaling conditions
        if condition is not None:
            if condition not in conditions.keys():
                raise ValueError('Condition not recognized.')
            else:
                self.apply_rate_scaling(network, condition)

        # store system
        self.network = network

        # instantiate solver
        self.set_solver(network)

    @property
    def N(self):
        """ Number of nodes in network. """
        return self.network.N

    @property
    def M(self):
        """ Number of reactions in network. """
        return self.network.M

    @property
    def I(self):
        """ Number of input channels in network. """
        return self.network.I

    def set_solver(self, network):
        """
        Set deterministic cython-based solver.

        Args:

            network (Network) - python Network instance

        """
        self.solver = cDeterministicSystem(network)

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
                rxn.k *= rate_scaling['temperature']**rxn.temperature_sensitive

        if 'metabolic_rate' in rate_scaling.keys():
            for rxn in network.reactions:
                rxn.k *= rate_scaling['metabolic_rate']**rxn.atp_sensitive

        if 'translation_capacity' in rate_scaling.keys():
            for rxn in network.reactions:
                rxn.k *= rate_scaling['translation_capacity']**rxn.ribosome_sensitive

    def differentiate(self, x, inputs):
        """
        Evaluate and return rate of change in state space.

        Args:

            x (np.ndarray[double]) - state and integrator values, length 2N

            inputs (np.ndarray[double]) - input signal values, length I

        Returns:

            dxdt (np.ndarray[double]) - rate of change for states and integrators, length 2N

        """

        # ensure state and integrator values are c-contiguous
        if not x.data.c_contiguous:
            x = np.ascontiguousarray(x)

        # evaluate derivatives
        dxdt = self.solver.c_evaluate_species_rates(states=x, inputs=inputs)

        return dxdt

    def solve_ivp(self,
                  ic=None,
                  integrator_ic=None,
                  signal=None,
                  duration=100,
                  dt=1,
                  method='BDF'):
        """
        Simulates dynamic system using Scipy's ode object.

        Args:

            ic (np.ndarray[double]) - initial conditions

            integrator_ic (np.ndarray[double]) - integrator initialization

            signal (cSignalType) - returns signal value(s) at each time

            duration (float) - simulation duration

            dt (float) - sampling interval

            method (str) - solver method

        Returns:

            timeseries (TimeSeries)

        """

        # if no initial condition is provided, assume all states are zero
        if ic is None:
            ic = self.network.ic.astype(np.float64)
        else:
            ic = ic.astype(np.float64)
        assert (ic.size==self.N), 'Initial Condition is the wrong size.'

        # apply network constraints to initial condition
        self.network.constrain_ic(ic)

        # if no initial integrator condition is provided, assume zeros
        if integrator_ic is None:
            integrator_ic = np.zeros(self.N, dtype=np.float64)
        else:
            integrator_ic = integrator_ic.astype(np.float64)
        assert (integrator_ic.size==self.N), 'Integrator IC is the wrong size.'

        # combine initial condition with initial integrator condition
        ic = np.hstack((ic, integrator_ic))

        # if no input function, use zeros
        if signal is None:
            signal = cSignal()
            assert self.I == 1, 'Signal dimensions do not match system inputs.'

        # define derivative function
        dxdt = lambda t, x: self.differentiate(x, signal.get_values(t))

        # run solver
        t_eval = np.arange(0, duration, dt)
        t_span = (0, duration)
        solout = solve_ivp(dxdt,
                           t_span,
                           ic,
                           method,
                           t_eval=t_eval,
                           vectorized=False)

        # instantiate timeseries
        states = solout['y'][:self.N].reshape(1, self.N, t_eval.size)
        timeseries = TimeSeries(t_eval, states)

        return timeseries
