# cython: profile=False

# cython external imports
cimport numpy as np
from cpython.array cimport array

# python external imports
import numpy as np

# cython intra-package imports
from .signals cimport cSignal, cSquarePulse, cMultiPulse, cSquareWave


cdef class cSignal:
    """ Class defines a single channel square pulse signal. """

    def __init__(self, value=0, ndim=1):
        self.I = ndim
        levels = [value for _ in range(ndim)]
        self.value = array('d', np.array(levels, dtype=np.float64))

    def __call__(self, double t):
        return self.get_signal(t)

    cdef array get_signal(self, double t):
        self.update(t)
        return self.value

    cdef void update(self, double t) nogil:
        pass

    cdef void reset(self) nogil:
        pass

    cdef void set_value(self, array values) nogil:
        cdef unsigned int index
        for index in xrange(self.I):
            self.value.data.as_doubles[index] = values.data.as_doubles[index]

    cdef bint compare_value(self, array values, unsigned int index) nogil:
        cdef bint not_equal = 0
        if self.value.data.as_doubles[index] != values.data.as_doubles[index]:
            not_equal = 1
        return not_equal

    cdef bint compare_all(self, array values) nogil:
        cdef bint not_equal = 0
        cdef unsigned int index
        cdef double value
        for index in xrange(self.I):
            value = values.data.as_doubles[index]
            if self.value.data.as_doubles[index] != value:
                not_equal = 1
                break
        return not_equal


cdef class cSquarePulse(cSignal):
    """ Class defines a single channel square pulse signal. """

    def __init__(self, t_on=0., t_off=3., off=0, on=1):
        cSignal.__init__(self, value=off)
        self.t_on = t_on
        self.t_off = t_off
        self.off = array('d', np.array([off], dtype=np.float64).flatten())
        self.on = array('d', np.array([on], dtype=np.float64).flatten())
        self.I = len(self.off)
        self.reset()

    cdef void reset(self) nogil:
        self.set_value(self.off)

    cdef void update(self, double t) nogil:
        if t >= self.t_on:
            if t < self.t_off:
                self.set_value(self.on)
            else:
                self.set_value(self.off)
        else:
            pass


cdef class cMultiPulse(cSignal):
    """ Class defines a multi channel square pulse signal. """

    def __init__(self, t_on, t_off, off, on):
        self.I = len(off)
        self.t_on = array('d', np.array(t_on, dtype=np.float64))
        self.t_off = array('d', np.array(t_off, dtype=np.float64))
        self.off = array('d', np.array(off, dtype=np.float64))
        self.on = array('d', np.array(on, dtype=np.float64))
        self.reset()

    cdef void reset(self) nogil:
        self.set_value(self.off)

    cdef void update(self, double t) nogil:
        cdef unsigned int index

        # get input from each channel
        for index in xrange(self.I):
            if t >= self.t_on.data.as_doubles[index]:
                if t < self.t_off.data.as_doubles[index]:
                    self.value.data.as_doubles[index] = self.on.data.as_doubles[index]
                else:
                    self.value.data.as_doubles[index] = self.off.data.as_doubles[index]
            else:
                pass


cdef class cSquareWave(cSignal):
    """ Class defines a single channel square wave signal. """

    def __init__(self, period=1., off=0., on=1.):
        self.period = period
        self.off = array('d', np.array([off], dtype=np.float64).flatten())
        self.on = array('d', np.array([on], dtype=np.float64).flatten())
        self.I = len(self.off)
        self.reset()

    cdef void reset(self) nogil:
        self.set_value(self.off)

    cdef void update(self, double t) nogil:
        """ Get pulse value at a given time. """

        if (t // self.period) % 2 == 0:
            self.set_value(self.off)
        else:
            self.set_value(self.on)
