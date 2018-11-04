# cython: profile=False

# cython intra-package imports
from ..signals.signals cimport cSignalType
from .stochastic cimport cStochasticSystem
from .stochastic cimport evaluate_timestep, sum_double_arr, rand_open

# python intra-package imports
from .stochastic import MonteCarloSimulation

# cython external imports
from libc.stdlib cimport rand, srand, RAND_MAX
#from cpython.array cimport array

# python external imports
#from array import array


# ============================= CYTHON CODE ===================================


cdef class cDebug(cStochasticSystem):
    """
    Debugging extension of a class that defines a network of nodes that interact via stochastic reactions.

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

    cdef void ssa(self,
                    cSignalType signal,
                    double duration,
                    double sampling_interval) with gil:
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

            print(t)

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

    cdef unsigned int choose_rxn(self, double random) with gil:
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
                            unsigned int *states) with gil:
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
                                double tau) with gil:
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

    cdef void sample(self) with gil:
        """ Sample current state levels. """

        # record states
        cdef unsigned int i
        cdef unsigned int row = self.sample_index*self.N
        for i in xrange(self.N):
            self.samples.data.as_uints[row+i] = self.states[i]

        # increment sampling index
        self.sample_index += 1

    cdef void record(self, double end_time) with gil:
        """

        Sample constant state levels until specified time.

        Args:

            end_time (double) - time to stop recording

        """
        while self.sample_time <= end_time:
            self.sample()
            self.sample_time += self.sampling_interval



# ============================= PYTHON CODE ===================================


class Debugger(MonteCarloSimulation):
    """
    Debugging version of python class for simulating multiple gene regulatory network dynamic trajectories using the Gillespie stochastic simulation algorithm.

    Attributes:


    Inherited Attributes:

        ic (function) - returns initial value vector

        integrator_ic (function) - returns initial integrator value vectors

        N (unsigned int) - number of nodes

        M (unsigned int) - number of reactions

        I (unsigned int) - number of external inputs

        solver (cStochasticSystem) - cython-based stochastic system

        seed (int) - seed for random number generator

    """
    def set_solver(self, network):
        """
        Set stochastic cython-based solver.

        Args:

            network (Network) - python Network instance

        """
        self.solver = cDebug(network, self.seed)
