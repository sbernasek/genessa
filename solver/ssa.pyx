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
    cdef array stoichiometry
    cdef long[:] min_coeff_per_rxn
    cdef long[:] reactant_species
    cdef unsigned int num_reactant_species
    cdef cRateFunction rate_function
    cdef cStoichiometry stoich

    def __init__(self, unsigned int N, unsigned int M, unsigned int I, array stoichiometry, long[:] min_coeff_per_rxn, long[:] reactant_species, int num_reactant_species, cRateFunction rate_function, cStoichiometry stoich):
        self.N = N
        self.M = M
        self.I = I
        self.stoichiometry = stoichiometry
        self.min_coeff_per_rxn = min_coeff_per_rxn
        self.reactant_species = reactant_species
        self.num_reactant_species = num_reactant_species
        self.rate_function = rate_function
        self.stoich = stoich

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef array get_rxn_rates(self):
        return self.rate_function.rates

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double get_total_rate(self):
        return self.rate_function.total_rate

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_all(self, array states, array input_value, array cumulative) nogil:
        self.rate_function.update_all(states, input_value, cumulative)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_input(self, array states, array input_value, array cumulative, unsigned int dim) nogil:
        self.rate_function.update_input(states, input_value, cumulative, dim)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update(self, array states, array input_value, array cumulative, unsigned int rxn_fired) nogil:
        self.rate_function.update(states, input_value, cumulative, rxn_fired)


cdef class cSolver:
    cdef cNetwork network
    cdef object rng
    cdef array int_template
    cdef array float_template

    def __init__(self, cNetwork network):

        # add network
        self.network = network

        # seed random number generator
        self.rng = cyRNG(100)

        # create zeros template for extents
        self.int_template = array('l', np.zeros(self.network.N, dtype=np.int64))
        self.float_template = array('d', np.zeros(self.network.N, dtype=np.float64))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ssa(self, unsigned int[::1] ic, cSignal input_function,
                double[::1] integrator_ic,
                double dt=1., double duration=100, bint integrate=0):
        """ Python interface for SSA. """
        cdef np.ndarray times = np.arange(0, duration, dt)
        cdef np.ndarray states
        states = self.c_ssa(ic, input_function, integrator_ic, dt, duration, integrate)
        return times, states

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray c_ssa(self, unsigned int[::1] ic, cSignal input_function,
                double[::1] integrator_ic,
                double dt=1., double duration=100, bint integrate=0):
        """
        Run gillespie ssa solver.

        Args:
        ic (np.ndarray[long]) - initial condition
        input_function (func returning np.ndarray[double]) - input signal
        integrator_ic (np.ndarray[double]) - initial condition for integrator
        dt (double) - timestep used for interpolation
        duration (double) - simulation duration

        Returns:
        times_regular (np.ndarray[float]) - interpolated time vector
        states_regular (np.ndarray[float]) - interpolation of each state
        """

        # initialize times and state lists, simulation counters
        cdef double t = 0.
        cdef array states = array('I', ic)
        cdef array cumulative = array('d', integrator_ic)

        # for input checking loop
        cdef unsigned int index
        cdef bint input_changed

        # simulation history
        #cdef array times_regular = array('d', np.arange(0, duration, dt))
        cdef long num_timepoints = <long>ceil(duration/dt)
        cdef array states_regular = array('I', np.empty((self.network.N, num_timepoints), dtype=np.uint32).flatten())
        cdef double threshold = 0.
        cdef unsigned int t_index = 0
        cdef unsigned int s_index
        cdef np.ndarray trajectories

        # initialize input
        cdef array input_value, new_input_value
        cdef bint null_input = 0
        if input_function is None:
            null_input = 1
            input_value = array('d', np.zeros(self.network.I))
        else:
            input_value = input_function.get_signal(0)

        # declare items used throughout simulation
        cdef array rxn_rates
        cdef array rxn_order
        cdef double total_rate
        cdef unsigned int rxn_fired = 0
        cdef double tau

        # initialize random number
        cdef double rfloat

        # initialize all rates and sort order
        self.network.update_all(states, input_value, cumulative)
        rxn_rates = self.network.get_rxn_rates()
        rxn_order = array('I', np.argsort(rxn_rates).astype(np.uint32)[::-1])

        # ================================================================
        # BEGIN SIMULATION
        # ================================================================
        while t < duration:

            # update stored state values
            while t >= threshold:
                for s_index in xrange(self.network.N):
                    states_regular.data.as_uints[s_index*num_timepoints+t_index] = states.data.as_uints[s_index]
                threshold += dt
                t_index += 1
                rxn_order = array('I', np.argsort(rxn_rates).astype(np.uint32)[::-1])

            if null_input == 0:

                # compute input value
                new_input_value = input_function.get_signal(t)

                # check if input changed and input rates accordingly
                for index in xrange(self.network.I):
                    if new_input_value.data.as_doubles[index] != input_value.data.as_doubles[index]:
                        input_value.data.as_doubles[index] = new_input_value.data.as_doubles[index]
                        self.network.update_input(states, input_value, cumulative, index)

            # update reaction rates
            self.network.update(states, input_value, cumulative, rxn_fired)

            # get rates
            rxn_rates = self.network.get_rxn_rates()
            total_rate = self.network.get_total_rate()

            # if total rate is zero, keep stepping until input changes
            if total_rate == 0:

                # if there is no input, jump to end
                if null_input == 1:
                    break
                else:
                    # skip to next change in input
                    input_changed = 0
                    while t <= duration:

                        # if input value changes, break loop
                        new_input_value = input_function.get_signal(t)
                        for index in xrange(self.network.I):
                            if new_input_value.data.as_doubles[index] != input_value.data.as_doubles[index]:
                                input_changed = 1
                                break
                        if input_changed == 1:
                            break

                        # otherwise, jump specified time
                        else:
                            t += dt
                    continue

            # choose a reaction
            rfloat = rand()/(RAND_MAX*1.0)
            tau = get_timestep(total_rate, rfloat)
            rxn_fired = choose_rxn(rxn_order, rxn_rates, self.network.M, total_rate, rfloat)

            # fire reaction
            self.fire_reaction(rxn_fired, 1, states)

            # update times and states (TODO: only update those that changed)
            t += tau
            if integrate == 1:
                self.update_cumulative(states, cumulative, tau)

        # ================================================================
        # END SIMULATION
        # ================================================================

        # interpolate any skipped values
        while threshold < duration:
            for s_index in xrange(self.network.N):
                states_regular.data.as_uints[s_index*num_timepoints+t_index] = states.data.as_uints[s_index]
            threshold += dt
            t_index += 1

        #return numpy arrays
        trajectories = np.frombuffer(states_regular, dtype=np.uint32).reshape(self.network.N, num_timepoints)

        return trajectories

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
        cdef unsigned int N = self.network.stoich.lengths.data.as_uints[rxn]
        cdef unsigned int index = self.network.stoich.index.data.as_uints[rxn]
        cdef unsigned int count, species
        cdef int coefficient

        # update each state
        for count in xrange(N):
            species = self.network.stoich.species.data.as_uints[index]
            coefficient = self.network.stoich.coefficients.data.as_longs[index]
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
        stoich = cStoichiometry.from_array(network.stoichiometry)
        rate_function = RateFunction(network).cRateFunction
        reactant_species = np.where(network.stoichiometry.min(axis=1) <= 0)[0]
        num_reactant_species = reactant_species.size
        min_coeff_per_rxn = network.stoichiometry.min(axis=0)
        c_network = cNetwork(N, M, I, stoichiometry, min_coeff_per_rxn, reactant_species, num_reactant_species, rate_function, stoich)

        self.c_solver = cSolver(c_network)

    @staticmethod
    def get_stoichiometry_dict(network):
        adict = {}
        for i, rxn in enumerate(network.reactions):
            adict[i] = {s: rxn.stoichiometry[s] for s in rxn.stoichiometry.nonzero()[0]}
        return adict

    def solve(self, ic, input_function=None, integrator_ic=None, dt=1., duration=100., use_pure_ssa=False, integrate=False):
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
            solout = self.c_solver.ssa(ic, input_function, integrator_ic, dt=dt, duration=duration, integrate=int(integrate))
        else:
            raise ValueError('Hybrid-SSA/Tau leap not yet implemented.')

        return solout
