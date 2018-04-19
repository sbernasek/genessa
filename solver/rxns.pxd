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

    # methods
    cdef double get_rate(self, array states, array input_value)

cdef class cRateFunction:

    # attributes
    cdef cMassAction[:] massaction
    cdef cHill[:] hill
    cdef cSumRxn[:] icontrol
    cdef cSumRxn[:] pcontrol
    cdef int n_massaction, n_hill, n_icontrol, n_pcontrol
    cdef array rates

    # methods
    cdef array get_rxn_rates(self, array states, array input_value, array cumulative)
