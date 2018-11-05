# cython: profile=False

# cython intra-package imports
from ..signals.signals cimport cSignalType, cSignal
from .stochastic cimport cStochasticSystem
from .stochastic cimport evaluate_timestep, sum_double_arr, rand_open

# python intra-package imports
from .stochastic import MonteCarloSimulation

# cython external imports
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.math cimport ceil
cimport numpy as np
from cpython.array cimport array

# python external imports
from array import array
import numpy as np


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
        print('CALL TO SSA.')
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

        print('BEGINNING SSA ALGORITHM.')

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
                    print('EXITING ON ZERO RATE WITHOUT INPUT')
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
                        print('EXITING ON ZERO RATE WITH INPUT')
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
                print('EXITING AT END OF SIMULATION DURATION')
                break

        # ================================================================
        # END SIMULATION
        # ================================================================

        print('FINISHED SSA ALGORITHM.')

        # record remaining timepoints
        self.record(duration)

        print('FINISHED RECORDING.')


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
        print('SETTING SOLVER.')
        self.solver = cDebug(network, self.seed)
