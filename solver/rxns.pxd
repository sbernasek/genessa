from cpython.array cimport array
from array import array
cimport numpy as np

cdef class cActive:
    cdef int N # number of active states
    cdef long[:] active # indices of active states

cdef class cActiveInputs(cActive):
    cdef int M # number of active inputs
    cdef long[:] active_inputs # indices of active inputs

cdef class cSumRxn(cActive):
    cdef double k # rate constant
    cdef long[:] propensity # propensity

    # methods
    cdef double get_rate_doubles(self, array values)
    cdef double get_rate_longs(self, array values)

cdef class cMassAction(cActiveInputs):
    cdef double k # rate constant
    cdef long[:] propensity, input_dependence # propensity and input dep.

    # methods
    cdef double get_rate(self, array states, array input_value) nogil

cdef class cRepressor(cActiveInputs):
    cdef double k_m, n # half maximal conc., hill coefficient
    cdef double[:] propensity, input_dependence # propensity and input dep.

    # methods
    cdef double get_occupancy(self, array states, array input_value) nogil

cdef class cHill(cActiveInputs):
    cdef int R # number of repressors
    cdef double k, k_m, n, baseline # rate constant, half maximal conc., hill coefficient, basal induction
    cdef double[:] propensity, input_dependence # propensity and input dep.
    cdef cRepressor[:] repressors
    cdef double[:] rate_modifier

    # methods
    cdef double get_rate(self, array states, array input_value)

cdef class cCoupling(cActive):
    cdef int R # number of repressors
    cdef double k # rate constant
    cdef long[:] propensity # propensity
    cdef double a # coupling strength
    cdef double w # edge weights
    cdef cRepressor[:] repressors

    # methods
    cdef double get_rate(self, array states, array input_value)

cdef class cRateFunction:

    # attributes
    cdef cActive[:] rxns
    cdef int M
    cdef array rxn_types,
    cdef dict rxn_map
    cdef dict input_map
    # cdef long[:] dependence,
    # cdef long[:] input_dependence
    cdef array rates
    cdef double total_rate

    # methods
    cdef array get_rxn_rates(self)
    cdef update_input(self, array states, array input_value, array cumulative, int input_dim)
    cdef update(self, array states, array input_value, array cumulative, int fired)
    cdef update_all(self, array states, array input_value, array cumulative)
    cdef set_rate(self, array states, array input_value, array cumulative, int rxn)
