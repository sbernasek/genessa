# cython external imports
from libc.math cimport log
from cpython.array cimport array

# import intra-package cython dependencies
from ..signals.signals cimport cSignalType
from .deterministic cimport cDeterministicSystem


cdef class cStochasticSystem(cDeterministicSystem):

    # attributes
    cdef bint integrate
    cdef bint null_input
    cdef array rstates
    cdef unsigned int *rxn_order
    cdef unsigned int *states
    cdef double *inputs
    cdef double *cumulative

    # methods
    cdef void allocate_memory(self)

    cdef void set_states(self, unsigned int[:] x) nogil

    cdef void set_inputs(self, double[:] x) nogil

    cdef void set_cumulative(self, double[:] x) nogil

    cdef void set_rxn_order(self, double *rates)

    cpdef tuple run(self,
        unsigned int[:] ic,
        double[:] integrator_ic,
        cSignalType signal,
        double duration=*,
        double dt=*)

    cdef void ssa(self,
        cSignalType signal,
        double duration,
        double dt) with gil

    cdef void fire_reaction(self,
        unsigned int rxn,
        unsigned int extent,
        unsigned int *states) nogil

    cdef void update_cumulative(self,
        unsigned int *states,
        double *cumulative,
        double tau) nogil

    cdef void record_states(self,
        unsigned int t_index) nogil


# ======================== STANDALONE FUNCTIONS ===============================


cdef inline double evaluate_timestep(double total_rate, double random) nogil:
    """
    Evaluate and return time until next reaction. Time interval is sampled from an exponential distribution.

    Args:

        total_rate (double) - total reaction rate

        random (double) - random float on [0, 1) interval

    Returns:

        tau (double) - time until next reaction event

    """
    return (1/total_rate) * log(1/random)


cdef inline unsigned int choose_rxn(unsigned int* order,
                                     double* rates,
                                     unsigned int num_rxns,
                                     double total_rate,
                                     double random) nogil:
    """
    Select a reaction from a list, with probabilities proportionally weighted by the rate of each reaction.

    Args:

        order (unsigned int*) - reaction sort order

        rates (double*) - reaction rates

        num_rxns (unsigned int) - number of reactions

        total_rate (double) - total reaction rate

        random (double) - random float on [0, 1) interval

    Returns:

        rxn (unsigned int) - chosen reaction index

    Notes:

        - If the random number is high and the previous reaction puts the rate over the total rate, the r<=0 comparison is never activated and the index isn't incremented by the subsequent loop. The solution implemented here is to correct index following the comparison.

    """
    cdef double rate = 0
    cdef double r
    cdef unsigned int index

    r = total_rate * random
    for index in xrange(num_rxns):
        rate = rates[order[index]]
        if r <= 0:
            index -= 1
            break
        r -= rate
    return order[index]
