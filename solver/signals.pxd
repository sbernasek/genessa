from cpython.array cimport array
from array import array

cdef class cSignal:
    cdef array on
    cdef array get_signal(self, double t)

cdef class cSquarePulse(cSignal):
    cdef double t_on, t_off
    cdef array off

cdef class cMultiPulse(cSignal):
    cdef long I
    cdef array t_on, t_off
    cdef array off

cdef class cSquareWave(cSignal):
    cdef double period
    cdef array off
