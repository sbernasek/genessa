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
from rxndiffusion.solver.signals cimport cSquarePulse, cMultiPulse

from cpython.array cimport array, clone
from cpython.array cimport copy as copyarray
from array import array


cdef class cNetwork:
    cdef int N, M, I # network has N nodes, M reactions, I inputs
    cdef array stoichiometry
    cdef long[:] min_coeff_per_rxn
    cdef long[:] reactant_species
    cdef int num_reactant_species
    cdef cRateFunction rate_function

    def __init__(self, int N, int M, int I, array stoichiometry, long[:] min_coeff_per_rxn, long[:] reactant_species, int num_reactant_species, cRateFunction rate_function):
        self.N = N
        self.M = M
        self.I = I
        self.stoichiometry = stoichiometry
        self.min_coeff_per_rxn = min_coeff_per_rxn
        self.reactant_species = reactant_species
        self.num_reactant_species = num_reactant_species
        self.rate_function = rate_function

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef array get_rxn_rates(self, array states, array input_value, array cumulative):
        return self.rate_function.get_rxn_rates(states, input_value, cumulative)


cdef class cSolver:
    cdef cNetwork network
    cdef object rng
    cdef array mzeros
    cdef array int_template
    cdef array float_template
    cdef double epsilon
    cdef int nc_threshold
    cdef double separation
    cdef int holding_period

    def __init__(self, cNetwork network):

        # add network
        self.network = network

        # seed random number generator
        self.rng = cyRNG(100)

        # create zeros template for extents
        self.mzeros = array('l', np.zeros(self.network.M, dtype=np.int64))
        self.int_template = array('l', np.zeros(self.network.N, dtype=np.int64))
        self.float_template = array('d', np.zeros(self.network.N, dtype=np.float64))

        # hybrid-ssa algorithm parameters
        self.epsilon = 0.03 # weights for candidate leap (~0.03)
        self.nc_threshold = 5 # threshold for critical rxns (~5 firings)
        self.separation = 10 # threshold events per step (~10 firings)
        self.holding_period = 100 # pure-ssa step interval (~100 steps)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef solve(self, long[::1] ic, cMultiPulse input_function,
                double[::1] integrator_ic,
                double dt=1., double duration=100, bint use_pure_ssa=0):
        """
        Run hybrid-ssa solver.

        Args:
        ic (np.ndarray[long]) - initial condition
        input_function (func returning np.ndarray[double]) - input signal
        integrator_ic (np.ndarray[double]) - initial condition for integrator
        dt (double) - timestep used for interpolation
        duration (double) - simulation duration
        use_pure_ssa (bint) - if True, use pure SSA algorithm

        Returns:
        times_regular (np.ndarray[float]) - interpolated time vector
        states_regular (np.ndarray[float]) - interpolation of each state
        """

        # initialize times and state lists, simulation counters
        cdef double t = 0.
        cdef array states = array('l', ic)
        cdef array new_states
        cdef array cumulative = array('d', integrator_ic)

        # for input checking loop
        cdef int index
        cdef bint input_changed

        # simulation history
        cdef array times_regular = array('d', np.arange(0, duration, dt))
        cdef long num_timepoints = <long>ceil(duration/dt)
        cdef array states_regular = array('l', np.empty((self.network.N, num_timepoints), dtype=np.int64).flatten())
        cdef double threshold = 0.
        cdef int t_index = 0
        cdef int s_index
        cdef np.ndarray times, trajectories

        # gilespie
        cdef int pure_ssa_count = 0
        if use_pure_ssa == 1:
            pure_ssa_count = 1
        cdef bint shrink_step_size = 0

        # initialize input
        cdef bint null_input = 0
        if input_function is None:
            null_input = 1

        # declare items used throughout simulation
        cdef array input_value, new_input_value
        cdef array rxn_rates
        cdef double total_rxn_rate, total_critical_rxn_rate
        cdef array rxn_extents = clone(self.int_template, self.network.M, zero=1)
        cdef int rxn_fired
        cdef array critical_rxns, non_critical_rxns
        cdef double candidate_leap, alternate_leap, tau

        ###### begin dynamic simulation ######
        while t < duration:

            # update stored state values
            while t >= threshold:
                for s_index in xrange(self.network.N):
                    states_regular.data.as_longs[s_index*num_timepoints+t_index] = states.data.as_longs[s_index]
                threshold += dt
                t_index += 1

            # compute input value
            input_value = input_function.get_signal(t)

            # get current states and reaction rates
            rxn_rates = self.network.rate_function.get_rxn_rates(states, input_value, cumulative)

            # compute total reaction rate
            total_rxn_rate = cython_sum.sum_double_arr(rxn_rates, self.network.M)

            if total_rxn_rate == 0:

                # if there is no input and all rates are zero, jump to end
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

            # initialize reaction extents
            for index in xrange(self.network.M):
                rxn_extents.data.as_longs[index] = self.mzeros.data.as_longs[index]

            # if pure_ssa_count is active take a single SSA step
            if pure_ssa_count > 0:
                tau = get_timestep(total_rxn_rate)
                rxn_fired = choose_rxn(rxn_rates, self.network.M)
                rxn_extents.data.as_longs[rxn_fired] = 1

                # if using pure SSA, keep counter at 1
                if use_pure_ssa == 0:
                    pure_ssa_count -= 1

            else:

                if shrink_step_size == 0:

                    # determine which reactions are critical
                    critical_rxns = self.get_critical_rxns(states)
                    non_critical_rxns = get_logical_not(critical_rxns, self.network.M)

                    # if all reactions are critical, leap to end (rejected...)
                    if cython_sum.sum_long_arr(non_critical_rxns, self.network.M) == 0:
                        candidate_leap = duration - t

                    # propose candidate leap based on max tolerable change
                    else:
                        candidate_leap = self.get_candidate_leap(states, rxn_rates, non_critical_rxns, duration-t)

                # if candidate leap is too short, execute 100 pure ssa steps
                if candidate_leap < (self.separation / total_rxn_rate):
                    pure_ssa_count = self.holding_period
                    continue

                else:

                    # compute tau for critical reactions
                    total_critical_rxn_rate = sum_array_subset(rxn_rates, critical_rxns, self.network.M)

                    # if total critical rate is zero, jump to end
                    if total_critical_rxn_rate == 0:
                        alternate_leap = duration - t

                    # otherwise, compute time until next critical firing
                    else:
                        alternate_leap = get_timestep(total_critical_rxn_rate)

                    if candidate_leap <= alternate_leap or total_critical_rxn_rate == 0.:
                        tau = candidate_leap
                    else:
                        # fire critical reaction
                        tau = alternate_leap
                        rxn_extents = self.fire_critical_rxns(rxn_rates, critical_rxns, rxn_extents)

                    # use tau leap for non critical reactions
                    rxn_extents = self.fire_noncritical_rxns(rxn_rates, non_critical_rxns, tau, rxn_extents)

            # update states
            new_states = self.get_new_states(states, rxn_extents)

            # check if step was too large
            if cython_sum.min_long_arr(new_states, self.network.N) < 0:
                candidate_leap = tau / 2
                shrink_step_size = 1
                continue
            else:
                shrink_step_size = 0

            # update times and states
            t += tau
            for index in xrange(self.network.N):
                states.data.as_longs[index] = new_states.data.as_longs[index]
                cumulative.data.as_doubles[index] += (dt * new_states.data.as_longs[index])

        ###### end dynamic simulation ######

        # interpolate any skipped values
        while threshold < duration:
            for s_index in xrange(self.network.N):
                states_regular.data.as_longs[s_index*num_timepoints+t_index] = states.data.as_longs[s_index]
            threshold += dt
            t_index += 1

        #return numpy arrays
        times = np.frombuffer(times_regular, dtype=np.float64)
        trajectories = np.frombuffer(states_regular, dtype=np.int64).reshape(self.network.N, num_timepoints)
        return times, trajectories

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef array get_new_states(self, array states, array extents):
        cdef array new_states
        cdef int i, j
        cdef long extent

        # copy current states
        new_states = clone(self.int_template, self.network.N, zero=1)
        for i in xrange(self.network.N):
            new_states.data.as_longs[i] = states.data.as_longs[i]

        # apply reactions
        for j in xrange(self.network.M):
            extent = extents.data.as_longs[j]

            # skip reactions that did not fire
            if extent != 0:
                for i in xrange(self.network.N):
                    new_states.data.as_longs[i] += (self.network.stoichiometry.data.as_longs[i*self.network.M+j] * extent)
        return new_states

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef array fire_noncritical_rxns(self, array rxn_rates, array non_critical, double tau, array rxn_extents):
        """ Draw extents for noncritical reactions from Poisson dist. """
        cdef int i
        for i in xrange(self.network.M):
            if non_critical.data.as_longs[i] == 1:
                if rxn_rates.data.as_doubles[i] != 0:
                    rxn_extents.data.as_longs[i] = self.rng.GetPoisson(rxn_rates.data.as_doubles[i]*tau)
        return rxn_extents

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef array fire_critical_rxns(self, array rxn_rates, array critical, array rxn_extents):
        """ Fires a single critical reaction. """

        cdef array critical_rxn_rates = copyarray(rxn_rates)
        cdef int i
        cdef int rxn_fired

        # if reaction is not critical, zero its rate
        for i in xrange(self.network.M):
            if critical.data.as_longs[i] == 0:
                critical_rxn_rates.data.as_doubles[i] = 0.

        if cython_sum.sum_double_arr(critical_rxn_rates, self.network.M) > 0.:
            rxn_fired = choose_rxn(critical_rxn_rates, self.network.M)
            rxn_extents.data.as_longs[rxn_fired] = 1

        return rxn_extents

cdef inline double get_timestep(double total_rxn_rate):
    """ Sample time until next reaction from exponential distribution. """
    cdef double random_float = rand()/(RAND_MAX*1.0)
    return (1/total_rxn_rate) * log(1/random_float)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int choose_rxn(array rates, int num_rxns):
    """ Probabilistically selects reaction based on rate. """
    cdef double rate, r
    cdef int index = 0
    cdef double total_rate = cython_sum.sum_double_arr(rates, num_rxns)

    r = rand()/(RAND_MAX*1.0)
    for index in xrange(num_rxns):
        rate = rates.data.as_doubles[index]
        if r < 0:
            break
        r -= rate / total_rate
        index += 1 # bug - remove on next compilation. doesnt affect anything
    return index - 1

cpdef double float_sum(np.ndarray[np.float_t, ndim=1] vals):
    """ Return sum of an array of type double. """
    return cython_sum.compute_float_sum(vals, vals.size)

cpdef long int_sum(np.ndarray[np.int_t, ndim=1] vals):
    """ Return sum of an array of type double. """
    return cython_sum.compute_long_sum(vals, vals.size)

cpdef double min_float(np.ndarray[np.float_t, ndim=1] vals):
    """ Return minimum from array of type long. """
    return cython_sum.get_min_float(vals, vals.size)

cpdef long min_int(np.ndarray[np.int_t, ndim=1] vals):
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

cdef double sum_array_subset(array arr, array mask, int arr_size):
    """ Sum a slice of a 1d array of type double. """
    cdef double subset_sum = 0
    cdef int index
    for i in xrange(arr_size):
        if mask.data.as_longs[i] == 1:
            subset_sum += arr.data.as_doubles[i]
    return subset_sum


class Solver:
    """ Python rapper for hybrid-ssa solver. """

    def __init__(self, network, ):

        # sort rxns and compile stoichiometry
        network.sort_rxns()
        network.compile_stoichiometry()

        # typecast network features
        N, M = network.stoichiometry.shape
        I = network.input_size
        stoichiometry = array('l', network.stoichiometry.astype(np.int64).tobytes())
        self.N = N

        # get cythonized rate function and network
        rate_function = RateFunction(network.reactions).cRateFunction
        reactant_species = np.where(network.stoichiometry.min(axis=1) <= 0)[0]
        num_reactant_species = reactant_species.size
        min_coeff_per_rxn = network.stoichiometry.min(axis=0)
        c_network = cNetwork(N, M, I, stoichiometry, min_coeff_per_rxn, reactant_species, num_reactant_species, rate_function)

        self.c_solver = cSolver(c_network)

    def solve(self, ic, input_function, integrator_ic=None, dt=1., duration=100., use_pure_ssa=False):
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
            input_function = lambda t: np.zeros(self.N, dtype=np.float64)
        else:
            input_function = input_function

        # check initial condition type
        ic = ic.astype(np.int64)

        # set initial condition for integrator
        if integrator_ic is None:
            integrator_ic = np.zeros(self.N, dtype=np.float64)
        else:
            integrator_ic = integrator_ic.astype(np.float64)

        # run solver
        solout = self.c_solver.solve(ic, input_function, integrator_ic, dt=dt, duration=duration, use_pure_ssa=int(use_pure_ssa))

        return solout
