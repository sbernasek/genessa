# cython: profile=False

# cython intra-package imports
from ..signals.signals cimport cSignalType, cSignal
from .deterministic cimport cDeterministicSystem
from .stochastic cimport cStochasticSystem
from .stochastic cimport evaluate_timestep, sum_double_arr, rand_open

# python intra-package imports
from .deterministic import DeterministicSimulation
from ..timeseries.base import TimeSeries

# cython external imports
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time
from libc.math cimport ceil
from cpython.mem cimport PyMem_Malloc, PyMem_Free
cimport numpy as np
from cpython.array cimport array

# python external imports
import numpy as np
from array import array
from types import FunctionType


# seed random number generator
srand(time(NULL))


# ============================= CYTHON CODE ===================================


cdef class cStochasticSystem(cDeterministicSystem):
    """
    Class defines a network of nodes that interact via stochastic reactions.

    Attributes:

        states (unsigned int*) - current state values for each node

        inputs (double*) - current input values for each signal channel

        cumulative (double*) - current integrator values for each node

        integrate (boolean int) - boolean flag for updating integrator

        rxn_order (unsigned int*) - reaction sort indices

        samples (array[unsigned int]) - sampled state values for each node

    Inherited Attributes:

        N (unsigned int) - number of nodes

        M (unsigned int) - number of reactions

        I (unsigned int) - number of external inputs

        S (cStoichiometry) - stoichiometry for all reactions

        R (cRates) - rate function for all reactions

    """

    def __init__(self, network, seed=-1):
        """
        Instantiate stochastic simulation.

        Args:

            network (Network) - python network instance

            seed (int) - seed for random number generator

        """

        cdef unsigned int i
        cdef int simulator_seed = <int> seed

        # seed random number generator
        if seed != -1:
            srand(simulator_seed)

        # set flag for integrator
        if self.R.icontrol.M == 0:
            self.integrate = 0
        else:
            self.integrate = 1

        # initialize simulation variables
        for i in xrange(self.N):
            self.states[i] = 0
            self.cumulative[i] = 0.
        for i in xrange(self.I):
            self.inputs[i] = 0.
        for i in xrange(self.M):
            self.rxn_order[i] = i

    def __cinit__(self, network, *args, **kwargs):
        """ Allocate memory for simulation variables. """
        self.allocate_memory()

    def __dealloc__(self):
        """ Deallocate memory from all array attributes. """
        PyMem_Free(self.rxn_order)
        PyMem_Free(self.states)
        PyMem_Free(self.inputs)
        PyMem_Free(self.cumulative)

    cdef void allocate_memory(self):
        """ Allocate memory for all array attributes. """

        cdef unsigned int size

        # allocate memory for reaction sort indices
        size = self.M * sizeof(unsigned int)
        self.rxn_order = <unsigned int*> PyMem_Malloc(size)
        if not self.rxn_order:
            raise MemoryError('Reaction order memory block not allocated.')

        # allocate memory for states vector
        size = self.N * sizeof(unsigned int)
        self.states = <unsigned int*> PyMem_Malloc(size)
        if not self.states:
            raise MemoryError('States memory block not allocated.')

        # allocate memory for input values vector
        size = self.I * sizeof(double)
        self.inputs = <double*> PyMem_Malloc(size)
        if not self.inputs:
            raise MemoryError('Inputs memory block not allocated.')

        # allocate memory for integrator values vector
        size = self.N * sizeof(double)
        self.cumulative = <double*> PyMem_Malloc(size)
        if not self.cumulative:
            raise MemoryError('Integrator memory block not allocated.')

    cdef void set_states(self, unsigned int[:] values) nogil:
        """ Set state values. """
        cdef unsigned int index
        for index in xrange(self.N):
            self.states[index] = values[index]

    cdef void set_inputs(self, double[:] values) nogil:
        """ Set input values. """
        cdef unsigned int index
        for index in xrange(self.I):
            self.inputs[index] = values[index]

    cdef void set_cumulative(self, double[:] values) nogil:
        """ Set integrator values. """
        cdef unsigned int index
        for index in xrange(self.N):
            self.cumulative[index] = values[index]

    cdef void set_rxn_order(self, double* rates):
        """
        Order reactions by reaction rate.

        Args:

            rates (double*) - reaction rates

        """
        cdef unsigned int index = 0
        cdef unsigned int rxn
        cdef np.ndarray[double, ndim=1] rates_array
        cdef np.ndarray[unsigned int, ndim=1] order

        # store rates as numpy array and get sorting index
        rates_array = np.asarray(<double[:self.M]> rates)
        order = np.argsort(rates_array).astype(np.uint32)[::-1]

        # copy to reaction order
        for index in xrange(self.M):
            self.rxn_order[index] = order[index]

    cpdef tuple run(self,
                    unsigned int[:] ic,
                    double[:] integrator_ic,
                    cSignalType signal,
                    double duration=100,
                    double sampling_interval=1.,
                    int seed=-1):
        """
        Python interface for stochastic simulation.

        Args:

            ic (unsigned int[:]) - initial state values

            integrator_ic (double[:]) - initial integrator values

            signal (cSignalType) - function returning signal value(s)

            duration (double) - simulation length

            sampling_interval (double) - sampling interval

            seed (int) - seed for random number generator

        Returns:

            times (np.ndarray[double]) - timepoints, length T

            states (np.ndarray[long]) - interpolated state values, shaped (N,T)

        """

        # seed random number generator
        if seed != -1:
            srand(seed)

        # set initial conditions
        self.set_states(ic)
        self.set_cumulative(integrator_ic)

        # initialize signal
        self.null_input = 0
        if signal is None:
            self.null_input = 1
            signal = cSignal()

        # set initial input signal values
        if self.null_input == 0:
            signal.reset()
            self.set_inputs(signal.get_values(0))

        # initialize all rates and sort order
        self.R.reset_rates()
        self.R.update_all(self.states, self.inputs, self.cumulative)
        self.set_rxn_order(self.get_rxn_rates())

        # declare variables for sampling
        self.sampling_interval = sampling_interval
        self.sample_time = 0
        self.sample_index = 0

        # preallocate samples array to record simulation history
        cdef unsigned int T = <unsigned int>ceil(duration/sampling_interval)
        samples = np.zeros((T, self.N), dtype=np.uint32)
        self.samples = array('I', samples.flatten())

        # run stochastic simulation algorithm
        self.ssa(signal=signal,
                 duration=duration,
                 sampling_interval=sampling_interval)

        #return numpy arrays
        cdef np.ndarray times = np.arange(0, duration, sampling_interval)
        cdef np.ndarray states = np.frombuffer(self.samples, dtype=np.uint32)

        return times, states.reshape(T, self.N).T

    cdef void ssa(self,
                    cSignalType signal,
                    double duration,
                    double sampling_interval) nogil:
        """
        Run Gillespie SSA.

        Args:

            signal (cSignalType) - function returning signal value(s)

            duration (double) - simulation length

            sampling_interval (double) - sampling interval

        """

        # declare variable for simulation
        cdef unsigned int index
        cdef unsigned int s_index
        cdef double t = 0.
        cdef unsigned int rxn = 0
        cdef double tau
        cdef double next_time

        # declare random float
        cdef double random_float

        # initialize input
        cdef bint changed
        if self.null_input == 0:
            signal.reset()

        cdef unsigned int state
        cdef double input_value

        # ================================================================
        # BEGIN SIMULATION
        # ================================================================
        while True:

            # record state values until current time
            self.record(t)

            # update input value
            if self.null_input == 0:

                # update input value
                signal.update(t)

                # check if input changed and input rates accordingly
                for index in xrange(self.I):
                    changed = signal.compare_value(self.inputs, index)
                    if changed == 1:
                        self.inputs[index] = signal.value[index]
                        self.R.update_after_input_change(self.states,
                                                       self.inputs,
                                                       self.cumulative,
                                                       index)

            # update reaction rates
            self.R.update_after_rxn_fired(self.states,
                                            self.inputs,
                                            self.cumulative,
                                            rxn)

            # if total rate is zero, keep stepping until input changes
            if self.R.total_rate <= 0:

                # set to zero (mitigates floating point error)
                self.R.total_rate = 0

                # if there is no input, jump to end
                if self.null_input == 1:
                    break
                else:
                    # skip to next change in input
                    changed = 0
                    while t <= duration:
                        signal.update(t)
                        changed = signal.compare_all(self.inputs)
                        if changed == 1:
                            break
                        else:
                            t += sampling_interval

                    # if simulation is complete, end it
                    if t >= duration:
                        break
                    else:
                        continue

            # evaluate timestep
            random_float = rand_open() / (RAND_MAX*1.0)
            tau = evaluate_timestep(self.R.total_rate, random_float)
            next_time = t + tau

            # if input change comes before next reaction, jump to that
            if next_time > signal.next_update:
                t = signal.next_update
                continue

            # fire reaction if it occurs within the simulation duration
            elif next_time < duration:

                # choose a reaction
                rxn = self.choose_rxn(random_float)

                # fire reaction
                self.fire_reaction(rxn, 1, self.states)

                # increment time and update cumulative state values
                t = next_time
                if self.integrate == 1:
                    self.update_cumulative(self.states, self.cumulative, tau)

            # otherwise, end the simulation
            else:
                break

        # ================================================================
        # END SIMULATION
        # ================================================================

        # record remaining timepoints
        self.record(duration)

    cdef unsigned int choose_rxn(self, double random) nogil:
        """
        Select a reaction with probabilities weighted by reaction rates.

        Args:

            random (double) - random float on [0, 1) interval

        Returns:

            rxn (unsigned int) - chosen reaction index

        """
        cdef double r = self.R.total_rate * random
        cdef double rate = 0
        cdef unsigned int index, rxn

        # select a reaction
        for index in xrange(self.M):
            rate = self.R.rates[self.rxn_order[index]]
            r -= rate
            if r <= 0:
                break

        # if procedure failed due to roundoff error, recurse with total rate
        if r > 0:
            self.R.total_rate = sum_double_arr(self.R.rates, self.M)
            rxn = self.choose_rxn(random)
        else:
            rxn = self.rxn_order[index]

        return rxn

    cdef void fire_reaction(self,
                            unsigned int rxn,
                            unsigned int extent,
                            unsigned int *states) nogil:
        """
        Fire a specified reaction by updating state values.

        Args:

            rxn (unsigned int) - reaction index

            extent (unsigned int) - reaction extent

            states (unsigned int*) - current state values

        """
        cdef unsigned int N = self.S.lengths[rxn]
        cdef unsigned int index = self.S.index[rxn]
        cdef unsigned int count, species
        cdef int coefficient

        # update each state
        for count in xrange(N):
            species = self.S.species[index]
            coefficient = self.S.coefficients[index]
            states[species] += (coefficient * extent)
            index += 1

    cdef void update_cumulative(self,
                                unsigned int *states,
                                double *cumulative,
                                double tau) nogil:
        """

        Update state integrator values.

        Args:

            states (unsigned int*) - state values

            cumulative (double*) - integrator values

            tau (double) - time step

        """
        cdef unsigned int i
        for i in xrange(self.N):
            cumulative[i] += (tau * states[i])

    cdef void sample(self) nogil:
        """ Sample current state levels. """

        # record states
        cdef unsigned int i
        cdef unsigned int row = self.sample_index*self.N
        for i in xrange(self.N):
            self.samples.data.as_uints[row+i] = self.states[i]

        # increment sampling index
        self.sample_index += 1

    cdef void record(self, double end_time) nogil:
        """

        Sample constant state levels until specified time.

        Args:

            end_time (double) - time to stop recording

        """
        while self.sample_time <= end_time:
            self.sample()
            self.sample_time += self.sampling_interval


# ============================= PYTHON CODE ===================================


class StochasticSimulation(DeterministicSimulation):
    """
    Python class for simulating gene regulatory network dynamics using the Gillespie stochastic simulation algorithm.

    Attributes:

        solver (cStochasticSystem) - cython-based stochastic system

        seed (int) - seed for random number generator

    Inherited attributes:

        network (Network) - python Network instance

    Properties:

        N (unsigned int) - number of nodes

        M (unsigned int) - number of reactions

        I (unsigned int) - number of external inputs

    """
    def __init__(self, network, condition, seed=None):
        """
        Instantiate deterministic simulation for a given network.

        Args:

            network (Network) - python Network instance

            condition (str) - environmental conditions affecting rates

            seed (int) - seed for random number generator

        """
        if seed is None:
            seed = -1

        self.seed = seed
        super().__init__(network, condition)

    def set_solver(self, network):
        """
        Set stochastic cython-based solver.

        Args:

            network (Network) - python Network instance

        """
        self.solver = cStochasticSystem(network, self.seed)

    def simulate(self,
            ic=None,
            integrator_ic=None,
            signal=None,
            duration=100.,
            dt=1.,
            seed=None):
        """
        Run stochastic simulation.

        Args:

            ic (np.ndarray[np.uint32]) - initial conditions

            integrator_ic (np.ndarray[np.float64]) - integrator initialization

            signal (cSignalType) - function returning signal value(s)

            duration (float) - simulation duration

            dt (float) - sampling interval

            seed (int) - seed for random number generator

        Returns:

            times (np.ndarray[np.float64]) - timepoints, shape (1,t)

            states (np.ndarray[np.uint32]) - state values, shape (N,t)

        """

        # cast initial condition to unsigned 32-bit integer
        if ic is None:
            ic = self.network.ic.astype(np.uint32)
        else:
            ic = ic.astype(np.uint32)

        # apply network constraints to initial condition
        self.network.constrain_ic(ic)

        # cast integrator initial condition to 64-bit float
        if integrator_ic is None:
            integrator_ic = np.zeros(self.N, dtype=np.float64)
        else:
            integrator_ic = integrator_ic.astype(np.float64)

        # check that initial conditions have the correct dimensions
        assert (ic.size==self.N), 'Wrong IC dimensions.'
        assert (integrator_ic.size==self.N), 'Wrong Integrator IC dimensions.'

        # set default seed (not used if negative)
        if seed is None:
            seed = -1

        # run stochastic simulation
        dynamics = self.solver.run(
            ic=ic,
            integrator_ic=integrator_ic,
            signal=signal,
            duration=duration,
            sampling_interval=dt,
            seed=seed)

        return dynamics


class MonteCarloSimulation(StochasticSimulation):
    """
    Python class for simulating multiple gene regulatory network dynamic trajectories using the Gillespie stochastic simulation algorithm.

    Attributes:

        ic (function) - returns initial value vector

        integrator_ic (function) - returns initial integrator value vectors

    Inherited Attributes:

        N (unsigned int) - number of nodes

        M (unsigned int) - number of reactions

        I (unsigned int) - number of external inputs

        solver (cStochasticSystem) - cython-based stochastic system

        seed (int) - seed for random number generator

    """

    def __init__(self,
                 network,
                 condition=None,
                 ic=None,
                 integrator_ic=None,
                 seed=-1):
        """
        Each instance has a fixed set of simulation parameters for which samples are generated.

        Args:

            network (Network) - python Network instance

            condition (str) - environmental condition affecting rates

            ic (array like, tuple, or func) - initial conditions

            integrator_ic (array like, tuple, or func) - integrator initial conditions

            seed (int) - seed for random number generator

        """

        # instantiate simulation
        super().__init__(network, condition, seed=seed)

        # set initial condition generating functions
        self.ic = self.build_ic_generator(ic)
        self.integrator_ic = self.build_ic_generator(integrator_ic)

    @staticmethod
    def build_ic_generator(ic):
        """
        Build initial condition generating function. Behavior is determined by the initial condition data type provided:

        If initial condition is a tuple, treat it as a (mean, variance) pair that defines a normal distribution. Initial conditions are randomly sampled from the distribution for each trial.

        If initial condition is a function, the function output becomes the initial condition for each simulation.

        If initial condition is array like, the same values serve as the initial condition for each simulation.

        Args:

            ic (tuple, function, or array like) - initial condition

        Returns:

            f (func) - function that returns an initial value vector

        """

        # if ic is a (mean, var) tuple, sample initial conditions from gaussian
        if type(ic) == tuple:
            m, v = ic
            f = lambda: np.random.normal(loc=m, scale=np.sqrt(v)).astype(int)

        # otherwise if ic is a function, call it for each initial condition
        elif type(ic) == FunctionType:
            f = lambda: ic()

        # otherwise repeat the same initial condition
        else:
            f = lambda: ic

        return f

    def run(self,
            N=100,
            signal=None,
            duration=100,
            dt=1,
            seed=None):
        """
        Run multiple stochastic simulations.

        Args:

            N (int) - number of independent simulation trajectories

            signal (cSignalType) - function returning signal value(s)

            duration (float) - simulation duration

            dt (float) - sampling interval

            seed (int) - seed for random number generator

        Returns:

            timeseries (TimeSeries) - system dynamics

        """

        # run each trial, appending sample to a list of all samples
        samples = []
        for i in range(N):

            # generate initial conditions
            ic = self.ic()
            integrator_ic = self.integrator_ic()

            # check that initial condition is positive
            assert (ic<0).sum() == 0, 'Negative IC on trial {:d}'.format(i)

            # run simulation
            times, states = self.simulate(
                ic=ic,
                integrator_ic=integrator_ic,
                signal=signal,
                duration=duration,
                dt=dt,
                seed=seed)

            # append states vector
            samples.append(states)

        # instantiate time series object
        timeseries = TimeSeries(times, np.array(samples, dtype=np.uint32))

        return timeseries
