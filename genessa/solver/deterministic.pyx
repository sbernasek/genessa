# cython external imports
cimport numpy as np
from cpython.array cimport array

# python external imports
import numpy as np
import ctypes
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
from .timeseries import TimeSeries


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

    def __cinit__(self, network):
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

    cpdef array c_evaluate_species_rates(self,
        np.ndarray states,
        array inputs,
        np.ndarray cumulative):
        """
        Evaluate rate of change for all species.

        Args:

            states (np.ndarray[double]) - state values

            inputs (array[double]) - input values

            cumulative (np.ndarray[double]) - integrator values

        Returns:

            rates (array[double]) - species rates, e.g. dX/dt

        """

        cdef unsigned int rxn
        cdef double rxn_rate
        cdef unsigned int N, index, count, species
        cdef int coefficient
        cdef array rxn_rates

        # instantiate array of zeros (double length for adding integrator)
        cdef array rates = array('d', self.N*[0.])

        # evaluate reaction rates
        rxn_rates = self.R.c_evaluate_rxn_rates(states, inputs, cumulative)

        # for each reaction
        for rxn in xrange(self.M):
            rxn_rate = rxn_rates.data.as_doubles[rxn]
            N = self.S.lengths[rxn]
            index = self.S.index[rxn]

            # update each active state
            for count in xrange(N):
                species = self.S.species[index]
                coefficient = self.S.coefficients[index]
                rates.data.as_doubles[species] += (coefficient * rxn_rate)
                index += 1

        return rates


# ============================= PYTHON CODE ===================================


class DeterministicSimulation:
    """
    Python class for simulating gene regulatory network dynamics by solving a deterministic system of ordinary differential equations.

    Attributes:

        N (unsigned int) - number of nodes

        M (unsigned int) - number of reactions

        I (unsigned int) - number of external inputs

        solver (cDeterministicSystem) - cython-based deterministic system

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

        # store system dimensions
        self.N = network.N
        self.M = network.M
        self.I = network.I

        # instantiate solver
        self.set_solver(network)

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

            x (np.ndarray[float]) - state and integrator values, length 2N

            inputs (array[double]) - input signal values, length I

        Returns:

            dxdt (np.ndarray[float]) - rate of change for states and integrators, length 2N

        """

        # split states and integrator into separate c-contiguous arrays
        states = np.ascontiguousarray(x[:self.N])
        cumulative = np.ascontiguousarray(x[self.N:])

        # evaluate state derivatives
        dxdt = self.solver.c_evaluate_species_rates(states, inputs, cumulative)

        return np.hstack((dxdt, cumulative))

    def solve_ivp(self,
                  ic=None,
                  signal=None,
                  duration=100,
                  dt=1,
                  method='BDF'):
        """
        Simulates dynamic system using Scipy's ode object.

        Args:

            ic (np.ndarray) - initial conditions

            signal (cSignalType) - returns signal value(s) at each time

            duration (float) - simulation duration

            dt (float) - sampling interval

            method (str) - solver method

        Returns:

            timeseries (TimeSeries)

        """

        # if no initial condition is provided, assume all states are zero
        if ic is None:
            ic = np.zeros(self.N, dtype=np.float64)
        else:
            ic = ic.astype(np.float64)
        assert (ic.size==self.N), 'Initial condition has wrong dimensions.'

        # if no input function, use zeros
        if signal is None:
            signal = cSignal([0. for _ in range(self.I)])


        times = np.arange(0, duration, dt)

        # run solver
        tspan = (0, duration)
        dxdt = lambda t, x: self.differentiate(x, signal(t))
        solout = solve_ivp(dxdt, tspan, ic, method, t_eval=times)

        # interpolate onto regularly sampled time interval
        #interpolator = interp1d(solout['t'], solout['y'])
        #times = np.arange(0, duration, dt)
        #states = interpolator(times)
        states = solout['y']

        # instantiate timeseries
        timeseries = TimeSeries(times, states.reshape(1, self.N, times.size))

        return timeseries
