from cpython.array cimport array
from array import array


cdef class cSquarePulse:
    cdef double t_on, t_off
    cdef array off, on
    cdef array get_signal(self, double t)


cdef class cMultiPulse:
    cdef long I
    #cdef double[::1] t_on, t_off
    cdef array t_on, t_off
    cdef array off, on
    cdef array get_signal(self, double t)
