# cython: profile=False

import numpy as np
cimport numpy as np

from cpython.array cimport array
from cpython.array cimport copy as copyarray
from array import array

cdef class cSignal:
    """ Class defines a single channel square pulse signal. """

    def __init__(self, on=0, ndim=1):
        levels = [on for _ in range(ndim)]
        self.on = array('d', np.array(levels, dtype=np.float64))

    def __call__(self, double t):
        return self.get_signal(t)

    cdef array get_signal(self, double t):
        return self.on


cdef class cSquarePulse(cSignal):
    """ Class defines a single channel square pulse signal. """

    def __init__(self, t_on=0., t_off=3., off=0, on=1):
        self.t_on = t_on
        self.t_off = t_off
        self.off = array('d', np.array([off], dtype=np.float64).flatten())
        self.on = array('d', np.array([on], dtype=np.float64).flatten())

    def __call__(self, double t):
        return self.get_signal(t)

    cdef array get_signal(self, double t):
        """ Get pulse value at a given time. """
        if t < self.t_off:
            if t >= self.t_on:
                return self.on
        return self.off


cdef class cMultiPulse(cSignal):
    """ Class defines a multi channel square pulse signal. """

    def __init__(self, t_on, t_off, off, on):
        self.I = len(off)
        self.t_on = array('d', np.array(t_on, dtype=np.float64))
        self.t_off = array('d', np.array(t_off, dtype=np.float64))
        self.off = array('d', np.array(off, dtype=np.float64))
        self.on = array('d', np.array(on, dtype=np.float64))

    def __call__(self, double t):
        return self.get_signal(t)

    cdef array get_signal(self, double t):
        """ Get pulse values at a given time. """
        cdef long index
        cdef array values = copyarray(self.off)

        # get input from each channel
        for index in xrange(self.I):
            if t < self.t_off.data.as_doubles[index]:
                if t >= self.t_on.data.as_doubles[index]:
                    values.data.as_doubles[index] = self.on.data.as_doubles[index]
        return values


cdef class cSquareWave(cSignal):
    """ Class defines a single channel square wave signal. """

    def __init__(self, period=1., off=0., on=1.):
        self.period = period
        self.off = array('d', np.array([off], dtype=np.float64).flatten())
        self.on = array('d', np.array([on], dtype=np.float64).flatten())

    def __call__(self, double t):
        return self.get_signal(t)

    cdef array get_signal(self, double t):
        """ Get pulse value at a given time. """

        if (t // self.period) % 2 == 0:
            return self.off
        else:
            return self.on
