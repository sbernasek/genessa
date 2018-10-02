from cpython.array cimport array


cdef class cSignal:
    cdef unsigned int I
    cdef array value
    cdef array get_signal(self, double t)
    cdef void update(self, double t) nogil
    cdef void reset(self) nogil
    cdef void set_value(self, array values) nogil
    cdef bint compare_value(self, array values, unsigned int index) nogil
    cdef bint compare_all(self, array values) nogil

cdef class cSquarePulse(cSignal):
    cdef double t_on, t_off
    cdef array on, off

cdef class cMultiPulse(cSignal):
    cdef array t_on, t_off
    cdef array on, off

cdef class cSquareWave(cSignal):
    cdef double period
    cdef array on, off
