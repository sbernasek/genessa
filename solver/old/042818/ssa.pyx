# cython: profile=True

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
    cdef int N, M, I # network has N nodes, M reactions, I inputs
    cdef array stoichiometry
    cdef long[:] min_coeff_per_rxn
    cdef long[:] reactant_species
    cdef int num_reactant_species
    cdef cRateFunction rate_function
    cdef dict stoich

    def __init__(self, int N, int M, int I, array stoichiometry, long[:] min_coeff_per_rxn, long[:] reactant_species, int num_reactant_species, cRateFunction rate_function, dict stoich_dict):
        self.N = N
        self.M = M
        self.I = I
        self.stoichiometry = stoichiometry
        self.min_coeff_per_rxn = min_coeff_per_rxn
        self.reactant_species = reactant_species
        self.num_reactant_species = num_reactant_species
        self.rate_function = rate_function
        self.stoich = stoich_dict

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef array get_rxn_rates(self):
        return self.rate_function.get_rxn_rates()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double get_total_rate(self):
        return self.rate_function.total_rate

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef update_all(self, array states, array input_value, array cumulative):
        return self.rate_function.update_all(states, input_value, cumulative)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef update_input(self, array states, array input_value, array cumulative, int dim):
        return self.rate_function.update_input(states, input_value, cumulative, dim)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef update(self, array states, array input_value, array cumulative, int rxn_fired):
        return self.rate_function.update(states, input_value, cumulative, rxn_fired)


cdef class cSolver:
    cdef cNetwork network
    cdef object rng
    cdef array mzeros
    cdef array int_template
    cdef array float_template

    def __init__(self, cNetwork network):

        # add network
        self.network = network

        # seed random number generator
        self.rng = cyRNG(100)

        # create zeros template for extents
        self.mzeros = array('l', np.zeros(self.network.M, dtype=np.int64))
        self.int_template = array('l', np.zeros(self.network.N, dtype=np.int64))
        self.float_template = array('d', np.zeros(self.network.N, dtype=np.float64))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef ssa(self, long[::1] ic, cSignal input_function,
                double[::1] integrator_ic,
                double dt=1., double duration=100, int integrate=0):
        """ Python interface for SSA. """
        cdef np.ndarray times = np.arange(0, duration, dt)
        cdef np.ndarray states
        states = self.c_ssa(ic, input_function, integrator_ic, dt, duration, integrate)
        return times, states

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray c_ssa(self, long[::1] ic, cSignal input_function,
                double[::1] integrator_ic,
                double dt=1., double duration=100, int integrate=0):
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
        cdef array states = array('l', ic)
        cdef array cumulative = array('d', integrator_ic)

        # for input checking loop
        cdef int index
        cdef bint input_changed

        # simulation history
        #cdef array times_regular = array('d', np.arange(0, duration, dt))
        cdef long num_timepoints = <long>ceil(duration/dt)
        cdef array states_regular = array('l', np.empty((self.network.N, num_timepoints), dtype=np.int64).flatten())
        cdef double threshold = 0.
        cdef int t_index = 0
        cdef int s_index
        cdef np.ndarray trajectories

        # initialize input
        cdef bint null_input = 0
        if input_function is None:
            null_input = 1

        # declare items used throughout simulation
        cdef array input_value, new_input_value
        cdef array rxn_rates
        cdef array rxn_order
        cdef double total_rate
        cdef int rxn_fired = 0
        cdef double tau

        # initialize all rates and sort order
        input_value = input_function.get_signal(0)
        self.network.update_all(states, input_value, cumulative)
        rxn_rates = self.network.get_rxn_rates()
        rxn_order = array('l', np.argsort(rxn_rates)[::-1])

        # ================================================================
        # BEGIN SIMULATION
        # ================================================================
        while t < duration:

            # update stored state values
            while t >= threshold:
                for s_index in xrange(self.network.N):
                    states_regular.data.as_longs[s_index*num_timepoints+t_index] = states.data.as_longs[s_index]
                threshold += dt
                t_index += 1
                rxn_order = array('l', np.argsort(rxn_rates)[::-1])

            # compute input value
            new_input_value = self.get_input_value(input_function, t)

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

            # choose and fire a reaction
            tau = get_timestep(total_rate)
            #rxn_fired = self._choose_rxn(rxn_rates, total_rate)
            rxn_fired = self.choose_rxn(rxn_order, rxn_rates, total_rate)
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
                states_regular.data.as_longs[s_index*num_timepoints+t_index] = states.data.as_longs[s_index]
            threshold += dt
            t_index += 1

        #return numpy arrays
        trajectories = np.frombuffer(states_regular, dtype=np.int64).reshape(self.network.N, num_timepoints)
        #times = np.frombuffer(times_regular, dtype=np.float64)

        return trajectories

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double get_total_rate(self, array rxn_rates):
        return cython_sum.sum_double_arr(rxn_rates, self.network.M)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _choose_rxn(self, array rxn_rates, double total_rate):
        """ Probabalistically select a single reaction given rates. """
        return choose_rxn(rxn_rates, self.network.M, total_rate)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int choose_rxn(self, array order, array rxn_rates, double total_rate):
        """ Select reaction from given pre-sorted rates (faster). """
        return choose_rxn_sorted(order, rxn_rates, self.network.M, total_rate)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef array get_input_value(self, cSignal input_function, double t):
        return input_function.get_signal(t)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int choose_critical_rxn(self, array rates, array critical):
        """ Choose a single critical reaction to be fired. """
        cdef array critical_rates = copyarray(rates)
        cdef int i
        cdef int rxn
        cdef double total_rate

        # if reaction is not critical, zero its rate
        for i in xrange(self.network.M):
            if critical.data.as_longs[i] == 0:
                critical_rates.data.as_doubles[i] = 0.

        total_rate = cython_sum.sum_double_arr(critical_rates, self.network.M)
        if total_rate > 0.:
            rxn = choose_rxn(critical_rates, self.network.M, total_rate)

        return rxn

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef fire_reaction(self, int rxn, int extent, array states):
        cdef int i
        cdef int coefficient
        for i, coefficient in self.network.stoich[rxn].items():
            states.data.as_longs[i] += (coefficient * extent)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef update_cumulative(self, array states, array cumulative, double tau):
        cdef int i
        for i in xrange(self.network.N):
            cumulative.data.as_doubles[i] += (tau * states.data.as_longs[i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef fire_noncritical_rxns(self, array rates, array non_critical, double tau, array states):
        """ Draw extents for noncritical reactions from Poisson dist. """
        cdef int rxn, extent
        for rxn in xrange(self.network.M):
            if non_critical.data.as_longs[rxn] == 1:
                if rates.data.as_doubles[rxn] != 0:
                    extent = self.rng.GetPoisson(rates.data.as_doubles[rxn]*tau)
                    self.fire_reaction(rxn, extent, states)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef array get_critical_rxns(self, array states):
        """ Returns boolean vector identifying critical reactions. """

        cdef int i # indes for states
        cdef int j # index for reactions
        cdef long stoich_coeff, min_coeff
        cdef double max_firings

        # preallocate boolean array (as ints) for critical reactions
        cdef array critical_rxns = clone(self.int_template, self.network.M, zero=1)

        # iterate across states for each reaction, finding the one with the largest

        # iterate across reactions
        for j in xrange(self.network.M):
            max_firings_for_rxn = self.nc_threshold + 1
            min_coeff = self.network.min_coeff_per_rxn[j]

            # if no consumption, mark reaction as non critical and conditnue
            if min_coeff >= 0:
                critical_rxns.data.as_longs[j] = 0
                continue

            # iterate across states
            for i in xrange(self.network.N):
                stoich_coeff = self.network.stoichiometry.data.as_longs[i*self.network.M+j]
                if stoich_coeff < 0:
                    max_firings = states.data.as_longs[i]/(stoich_coeff/min_coeff)
                    if max_firings < self.nc_threshold:
                        critical_rxns.data.as_longs[j] = 1
                        break

        return critical_rxns

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double get_candidate_leap(self, array states, array rxn_rates, array non_critical_rxns, double max_leap):
        """ Proposes candidate leap time based on expected state change. """

        cdef int i, j
        cdef int index
        cdef long species_index
        cdef double rate
        cdef double candidate_leap = max_leap
        cdef double tolerable_change
        cdef array extent_mean = clone(self.float_template, self.network.N, zero=1)
        cdef array extent_std = clone(self.float_template, self.network.N, zero=1)
        cdef double proposed_1, proposed_2

        # get mean and variance of expected state changes
        for j in xrange(self.network.M):
            if non_critical_rxns.data.as_longs[j] == 1:
                rate = rxn_rates.data.as_doubles[j]
                if rate > 0:
                    for i in xrange(self.network.N):
                         extent_mean[i] += rate*self.network.stoichiometry.data.as_longs[i*self.network.M+j]
                         extent_std[i] += rate*((self.network.stoichiometry.data.as_longs[i*self.network.M+j])**2)

        # take absolute value
        for i in xrange(self.network.N):
            extent_mean[i] = fabs(extent_mean[i])

        # iterate across reactive species
        for index in xrange(self.network.num_reactant_species):
            species_index = self.network.reactant_species[index]

            # get numerator
            tolerable_change = self.epsilon * states.data.as_longs[species_index]
            if tolerable_change < 1.:
                tolerable_change = 1.

            # try first proposed leap (if valid)
            if extent_mean[species_index] != 0.:
                proposed_1 = tolerable_change / extent_mean[species_index]
                if proposed_1 < candidate_leap:
                    candidate_leap = proposed_1

            # try second proposed leap (if valid)
            if extent_std[species_index] != 0.:
                proposed_2 = (tolerable_change**2) / extent_std[species_index]
                if proposed_2 < candidate_leap:
                    candidate_leap = proposed_2

        return candidate_leap


cdef inline double get_timestep(double total_rate):
    """ Sample time until next reaction from exponential distribution. """
    cdef double random_float = rand()/(RAND_MAX*1.0)
    return (1/total_rate) * log(1/random_float)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int choose_rxn(array rates, int num_rxns, double total_rate) nogil:
    """ Probabilistically selects reaction based on rate. """
    cdef double rate, r
    cdef int index
    #cdef double total_rate = cython_sum.sum_double_arr(rates, num_rxns)

    r = rand()/(RAND_MAX*1.0)
    for index in xrange(num_rxns):
        rate = rates.data.as_doubles[index]
        if r < 0:
            index -= 1
            break
        r -= rate / total_rate
    return index

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int choose_rxn_sorted(array order, array rates, int num_rxns, double total_rate) nogil:
    """ Probabilistically selects reaction based on rate. """
    cdef double rate = 0
    cdef double r
    cdef int index

    # NOTE: if random number is high and last reaction puts rate over total, the r<=0 comparison is never activated and the index isn't incremented by the subsequent loop. solution is to correct index following the comparison
    r = rand()/(RAND_MAX*1.0)
    for index in xrange(num_rxns):
        rate = rates.data.as_doubles[order.data.as_longs[index]]
        if r <= 0:
            index -= 1
            break
        r -= rate / total_rate
    return order.data.as_longs[index]

cdef double float_sum(np.ndarray[np.float_t, ndim=1] vals):
    """ Return sum of an array of type double. """
    return cython_sum.compute_float_sum(vals, vals.size)

cdef long int_sum(np.ndarray[np.int_t, ndim=1] vals):
    """ Return sum of an array of type double. """
    return cython_sum.compute_long_sum(vals, vals.size)

cdef double min_float(np.ndarray[np.float_t, ndim=1] vals):
    """ Return minimum from array of type long. """
    return cython_sum.get_min_float(vals, vals.size)

cdef long min_int(np.ndarray[np.int_t, ndim=1] vals):
    """ Return minimum from array of type long. """
    return cython_sum.get_min_int(vals, vals.size)

cdef array get_logical_not(array arr, int arr_size):
    """ Returns boolean mask for logical_not of a type long memoryview. """
    cdef int i
    cdef long val
    cdef array opp = clone(arr, arr_size, zero=1)

    for i in xrange(arr_size):
        val = arr.data.as_longs[i]
        if val == 0:
            opp.data.as_longs[i] = 1
    return opp

cdef double sum_array_subset(array arr, array mask, int arr_size) nogil:
    """ Sum a slice of a 1d array of type double. """
    cdef double subset_sum = 0
    cdef int index
    for i in xrange(arr_size):
        if mask.data.as_longs[i] == 1:
            subset_sum += arr.data.as_doubles[i]
    return subset_sum


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
        stoich_dict = self.get_stoichiometry_dict(network)
        rate_function = RateFunction(network).cRateFunction
        reactant_species = np.where(network.stoichiometry.min(axis=1) <= 0)[0]
        num_reactant_species = reactant_species.size
        min_coeff_per_rxn = network.stoichiometry.min(axis=0)
        c_network = cNetwork(N, M, I, stoichiometry, min_coeff_per_rxn, reactant_species, num_reactant_species, rate_function, stoich_dict)

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
            input_function = cSignal([0 for _ in range(self.I)])
        else:
            input_function = deepcopy(input_function)

        # check initial condition type
        ic = ic.astype(np.int64)

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
