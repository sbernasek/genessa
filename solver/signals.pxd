from cpython.array cimport array
from array import array


cdef class cSquarePulse:
    cdef double t_on, t_off
    cdef array off, on
    cdef array get_signal(self, double t)


cdef class cMultiPulse:
    cdef long I
    cdef array t_on, t_off
    cdef array off, on
    cdef array get_signal(self, double t)
