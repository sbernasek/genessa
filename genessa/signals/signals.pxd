cimport numpy as np


cdef class cSignal:

    # attributes
    cdef unsigned int I
    cdef double *value
    cdef double next_update

    # methods
    cpdef double get_value(self, unsigned int index, double t)
    cpdef np.ndarray get_values(self, double t)
    cdef void update(self, double t) nogil
    cdef void reset(self) nogil
    cdef void set_value(self, double *values) nogil
    cdef bint compare_value(self, double *values, unsigned int index) nogil
    cdef bint compare_all(self, double *values) nogil


cdef class cSquarePulse(cSignal):

    # attributes
    cdef double t_on, t_off
    cdef double on, off


cdef class cSquareWave(cSignal):

    # attributes
    cdef double period
    cdef double halfperiod
    cdef double on, off


cdef class cMultiPulse(cSignal):

    # attributes
    cdef double *t_on
    cdef double *t_off
    cdef double *on
    cdef double *off
    cdef double *update_times


ctypedef fused cSignalType:
    cSignal
    cSquarePulse
    cSquareWave
    cMultiPulse
