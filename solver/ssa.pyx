# cython: profile=False

from rxndiffusion.solver.cyRNG import cyRNG
from rxndiffusion.solver.rxns import RateFunction
from rxndiffusion.solver.rxns cimport cRateFunction
cimport rxndiffusion.solver.cython_sum as cython_sum
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport log, fabs, ceil
import numpy as np
cimport numpy as np
cimport cython
from rxndiffusion.solver.signals cimport cSquarePulse, cMultiPulse, cSquareWave, cSignal

from cpython.array cimport array, clone
from cpython.array cimport copy as copyarray
from array import array
from copy import deepcopy


cdef class cNetwork:
    cdef unsigned int N, M, I # network has N nodes, M reactions, I inputs
    cdef cStoichiometry S
    cdef cRateFunction R

    def __init__(self, unsigned int N, unsigned int M, unsigned int I, cStoichiometry S, cRateFunction R):
        self.N = N
        self.M = M
        self.I = I
        self.S = S
        self.R = R

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef array get_rxn_rates(self):
        return self.R.rates

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double get_total_rate(self) nogil:
        return self.R.total_rate

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_all(self, array states, array inputs, array cumulative) nogil:
        self.R.update_all(states, inputs, cumulative)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_input(self, array states, array inputs, array cumulative, unsigned int dim) nogil:
        self.R.update_input(states, inputs, cumulative, dim)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update(self, array states, array inputs, array cumulative, unsigned int rxn_fired) nogil:
        self.R.update(states, inputs, cumulative, rxn_fired)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cset_species_rates(self, array states, array inputs, array cumul, array rates):

        cdef unsigned int rxn
        cdef double rxn_rate
        cdef unsigned int N, index, count, species
        cdef int coefficient

        # get reaction rates
        self.R.cupdate(states, inputs, cumul)

        # for each reaction
        for rxn in xrange(self.M):
            rxn_rate = self.R.rates.data.as_doubles[rxn]
            N = self.S.lengths.data.as_uints[rxn]
            index = self.S.index.data.as_uints[rxn]

            # update each active state
            for count in xrange(N):
                species = self.S.species.data.as_uints[index]
                coefficient = self.S.coefficients.data.as_longs[index]
                rates.data.as_doubles[species] += (coefficient * rxn_rate)
                index += 1

cdef class cSolver:
    cdef cNetwork network
    cdef object rng
    cdef bint integrate
    cdef bint null_input
    cdef array states, inputs, cumul, rxn_order
    cdef array rstates

    def __init__(self, cNetwork network):

        # add network
        self.network = network

        # seed random number generator
        self.rng = cyRNG(100)

        # set flags
        if network.R.icontrol.M == 0:
            self.integrate = 0
        else:
            self.integrate = 1

        # instantiate arrays for simulation variables
        self.states = array('I', np.zeros(network.N, dtype=np.uint32))
        self.inputs = array('d', np.zeros(network.I, dtype=np.int64))
        self.cumul = array('d', np.zeros(network.N, dtype=np.int64))
        self.rxn_order = array('I', np.arange(network.M, dtype=np.uint32))

        # initialize array for regular states
        self.rstates = array('I', np.zeros(network.N, dtype=np.uint32))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef array get_sp_rates(self, array states, array inputs, array cumul):
        """ Returns continuous species rates. """
        cdef array rates = array('d', self.network.N*[0.])
        self.network.cset_species_rates(states, inputs, cumul, rates)
        return rates

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_states(self, array x) nogil:
        cdef unsigned int index
        for index in xrange(self.network.N):
            self.states.data.as_uints[index] = x.data.as_uints[index]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_inputs(self, array x) nogil:
        cdef unsigned int index
        for index in xrange(self.network.I):
            self.inputs.data.as_doubles[index] = x.data.as_doubles[index]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_cumul(self, array x) nogil:
        cdef unsigned int index
        for index in xrange(self.network.N):
            self.cumul.data.as_doubles[index] = x.data.as_doubles[index]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_rxn_order(self, array rates):
        cdef unsigned int index = 0
        cdef unsigned int rxn
        cdef array order = array('I', np.argsort(rates).astype(np.uint32)[::-1])
        for index in xrange(self.network.M):
            self.rxn_order.data.as_uints[index] = order.data.as_uints[index]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple ssa(self, unsigned int[::1] ic, cSignal signal,
                double[::1] integrator_ic,
                double dt=1., double duration=100):
        """ Python interface for SSA. """

        # initialize times and state lists, simulation counters
        self.set_states(array('I', ic))
        self.set_cumul(array('d', integrator_ic))

        # initialize input
        self.null_input = 0
        if signal is None:
            self.null_input = 1
            signal = cSignal(0.)
        self.set_inputs(signal.get_signal(0))

        # initialize all rates and sort order
        self.network.update_all(self.states, self.inputs, self.cumul)
        cdef array rxn_rates = self.network.get_rxn_rates()
        self.set_rxn_order(rxn_rates)

        # preallocated regular states array to record simulation history
        cdef unsigned int num_timepoints = <unsigned int>ceil(duration/dt)
        self.rstates = array('I', np.empty((self.network.N, num_timepoints), dtype=np.uint32).flatten())

        self.c_ssa(signal=signal, dt=dt, duration=duration)

        #return numpy arrays
        cdef np.ndarray times = np.arange(0, duration, dt)
        cdef np.ndarray states = np.frombuffer(self.rstates, dtype=np.uint32).reshape(self.network.N, num_timepoints)

        return times, states

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void c_ssa(self, cSignal signal, double dt=1., double duration=100) nogil:
        """
        Run gillespie ssa solver.

        Args:
        signal (cSignal object)
        dt (double) - timestep used for interpolation
        duration (double) - simulation duration
        """

        # declare variables for time management
        cdef unsigned int index
        cdef unsigned int t_index = 0
        cdef unsigned int s_index
        cdef double t = 0.
        cdef double threshold = 0.
        cdef unsigned int num_timepoints = <unsigned int>ceil(duration/dt)

        # declare items used throughout simulation
        cdef unsigned int rxn = 0
        cdef double tau

        # declare random number
        cdef double rfloat

        # initialize input
        cdef bint changed
        if self.null_input == 0:
            signal.reset()

        # ================================================================
        # BEGIN SIMULATION
        # ================================================================
        while t < duration:

            # update stored state values
            while t >= threshold:
                for s_index in xrange(self.network.N):
                    self.rstates.data.as_uints[s_index*num_timepoints+t_index] = self.states.data.as_uints[s_index]
                threshold += dt
                t_index += 1
                #self.set_rxn_order(self.network.R.rates)

            # update input value
            if self.null_input == 0:

                # update input value
                signal.update(t)

                # check if input changed and input rates accordingly
                for index in xrange(self.network.I):
                    changed = signal.compare_value(self.inputs, index)
                    if changed == 1:
                        self.inputs.data.as_doubles[index] = signal.value.data.as_doubles[index]
                        self.network.update_input(self.states, self.inputs, self.cumul, index)

            # update reaction rates
            self.network.update(self.states, self.inputs, self.cumul, rxn)

            # if total rate is zero, keep stepping until input changes
            if self.network.R.total_rate == 0:

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
                            t += dt
                    continue

            # choose a reaction
            rfloat = rand()/(RAND_MAX*1.0)
            tau = get_timestep(self.network.R.total_rate, rfloat)
            rxn = choose_rxn(self.rxn_order, self.network.R.rates, self.network.M, self.network.R.total_rate, rfloat)

            # fire reaction
            self.fire_reaction(rxn, 1, self.states)

            # increment time and update cumulative state values
            t += tau
            if self.integrate == 1:
                self.update_cumulative(self.states, self.cumul, tau)

        # ================================================================
        # END SIMULATION
        # ================================================================

        # interpolate any later values
        while threshold < duration:
            for s_index in xrange(self.network.N):
                self.rstates.data.as_uints[s_index*num_timepoints+t_index] = self.states.data.as_uints[s_index]
            threshold += dt
            t_index += 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef unsigned int choose_rxn(self, array order, array rxn_rates, double total_rate, double rfloat) nogil:
        """ Select reaction from given pre-sorted rates (faster). """
        return choose_rxn(order, rxn_rates, self.network.M, total_rate, rfloat)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef array get_input_value(self, cSignal input_function, double t):
        return input_function.get_signal(t)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void fire_reaction(self, unsigned int rxn, unsigned int extent, array states) nogil:
        cdef unsigned int N = self.network.S.lengths.data.as_uints[rxn]
        cdef unsigned int index = self.network.S.index.data.as_uints[rxn]
        cdef unsigned int count, species
        cdef int coefficient

        # update each state
        for count in xrange(N):
            species = self.network.S.species.data.as_uints[index]
            coefficient = self.network.S.coefficients.data.as_longs[index]
            states.data.as_uints[species] += (coefficient * extent)
            index += 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_cumulative(self, array states, array cumulative, double tau) nogil:
        cdef unsigned int i
        for i in xrange(self.network.N):
            cumulative.data.as_doubles[i] += (tau * states.data.as_uints[i])


cdef class cStoichiometry:
    cdef array index
    cdef array lengths
    cdef array species
    cdef array coefficients

    def __init__(self, unsigned int[:] index,
                       unsigned int[:] lengths,
                       unsigned int[:] species,
                       long[:] coefficients):

        self.index = array('I', index)
        self.lengths = array('I', lengths)
        self.species = array('I', species)
        self.coefficients = array('l', coefficients)

    @staticmethod
    def from_array(np.ndarray s):
        rxns, species = s.T.nonzero()
        lengths = np.bincount(rxns).astype(np.uint32)
        index = np.hstack((np.zeros(1), np.cumsum(lengths))).astype(np.uint32)
        coefficients = s.T[(rxns, species)].astype(np.int64)
        return cStoichiometry(index, lengths, species.astype(np.uint32), coefficients)


cdef inline double get_timestep(double total_rate, double random) nogil:
    """ Sample time until next reaction from exponential distribution. """
    return (1/total_rate) * log(1/random)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef unsigned int choose_rxn(array order, array rates, unsigned int num_rxns, double total_rate, double random) nogil:
    """ Probabilistically selects reaction based on rate. """
    cdef double rate = 0
    cdef double r
    cdef unsigned int index

    # NOTE: if random number is high and last reaction puts rate over total, the r<=0 comparison is never activated and the index isn't incremented by the subsequent loop. solution is to correct index following the comparison
    r = total_rate * random
    for index in xrange(num_rxns):
        rate = rates.data.as_doubles[order.data.as_uints[index]]
        if r <= 0:
            index -= 1
            break
        r -= rate
    return order.data.as_uints[index]


class Solver:
    """ Python rapper for hybrid-ssa solver. """

    def __init__(self, network):

        # sort rxns and compile stoichiometry
        network.sort_rxns()
        network.resize_inputs()
        network.compile_stoichiometry()

        # typecast network features
        N = network.nodes.size
        M = len(network.reactions)
        I = network.input_size
        self.N = N
        self.M = M
        self.I = I
        stoichiometry = array('l', network.stoichiometry.astype(np.int64).tobytes())

        # get cythonized rate function and network
        #stoich_dict = self.get_stoichiometry_dict(network)
        S = cStoichiometry.from_array(network.stoichiometry)
        R = RateFunction(network).cRateFunction
        c_network = cNetwork(N, M, I, S, R)

        self.c_solver = cSolver(c_network)

    @staticmethod
    def get_stoichiometry_dict(network):
        adict = {}
        for i, rxn in enumerate(network.reactions):
            adict[i] = {s: rxn.stoichiometry[s] for s in rxn.stoichiometry.nonzero()[0]}
        return adict

    def solve(self, ic, input_function=None, integrator_ic=None, dt=1., duration=100., use_pure_ssa=False):
        """
        Run hybrid-ssa simulation algorithm.

        Args:
        ic (np.ndarray[np.int]) - initial condition
        input_function (func) - returns input value(s) as np.ndarray[np.float]
        dt (float) - timestep used for interpolation onto regular grid
        duration (float) - simulation end time
        use_pure_ssa (bint) - if True, use pure SSA algorithm

        Returns:
        solout (times_regular, states_regular) where:
            times_regular (np.ndarray[float]) - interpolated time vector
            states_regular (np.ndarray[float]) - interpolation of each state
        """

        # set input function
        if input_function is None:
            input_function = None#cSignal([0 for _ in range(self.I)])
        else:
            input_function = deepcopy(input_function)

        # check initial condition type
        ic = ic.astype(np.uint32)

        # set initial condition for integrator
        if integrator_ic is None:
            integrator_ic = np.zeros(self.N, dtype=np.float64)
        else:
            integrator_ic = integrator_ic.astype(np.float64)

        # run solver
        if use_pure_ssa:
            solout = self.c_solver.ssa(ic, input_function, integrator_ic, dt=dt, duration=duration)
        else:
            raise ValueError('Hybrid-SSA/Tau leap not yet implemented.')

        return solout
