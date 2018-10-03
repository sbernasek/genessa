# cython: profile=False

# cython external imports
cimport numpy as np
from cpython.array cimport array

# python external imports
import numpy as np

# cython intra-package imports
from .signals cimport cSignal, cSquarePulse, cMultiPulse, cSquareWave


cdef class cSignal:
    """
    Class defines a constant signal.

    Attributes:

        value (array[double]) - constant signal value

        I (int) - number of signal channels

    """

    def __init__(self, value=0, I=1):
        """
        Instantiate a constant signal.

        Args:

            value (float) - constant signal values

            I (int) - number of signal channels

        """
        self.I = I
        levels = [value for _ in range(I)]
        self.value = array('d', np.array(levels, dtype=np.float64))

    def __call__(self, double t):
        """
        Returns signal values for specified time.

        Args:

            t (double) - time

        Returns:

            values (array[double]) - signal values

        """
        return self.get_signal(t)

    cdef array get_signal(self, double t):
        """
        Returns signal values.

        Args:

            t (double) - time

        Returns:

            values (array[double]) - signal values

        """
        self.update(t)
        return self.value

    cdef void update(self, double t) nogil:
        pass

    cdef void reset(self) nogil:
        pass

    cdef void set_value(self, array values) nogil:
        """
        Set signal values.

        Args:

            values (array[double]) - new signal values

        """
        cdef unsigned int index
        for index in xrange(self.I):
            self.value.data.as_doubles[index] = values.data.as_doubles[index]

    cdef bint compare_value(self, array values, unsigned int index) nogil:
        """
        Compare individual signal value with a reference.

        Args:

            values (array[double]) - reference values

            index (unsigned int) - signal channel compared

        Returns:

            not_equal (bint) - if True, signal values are not equal

        """
        cdef bint not_equal = 0
        if self.value.data.as_doubles[index] != values.data.as_doubles[index]:
            not_equal = 1
        return not_equal

    cdef bint compare_all(self, array values) nogil:
        """
        Compare all signal values with a reference.

        Args:

            values (array[double]) - reference values

        Returns:

            not_equal (bint) - if True, signal values differ from reference

        """
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
    """
    Class defines a single-channel square pulse signal.

    Attributes:

        t_on (float) - pulse onset time

        t_off (float) - pulse off time

        off (float) - signal off value

        on (float) - signal on value

    Inherited Attributes:

        value (array[double]) - signal off-values

        I (int) - number of signal channels (always one)

    """

    def __init__(self, t_on=0., t_off=3., off=0, on=1):
        """
        Instantiate a single-channel square pulse signal.

        Args:

            t_on (float) - pulse onset time

            t_off (float) - pulse off time

            off (float) - signal off value

            on (float) - signal on value

        """
        cSignal.__init__(self, value=off)
        self.t_on = t_on
        self.t_off = t_off
        self.off = array('d', np.array([off], dtype=np.float64).flatten())
        self.on = array('d', np.array([on], dtype=np.float64).flatten())
        self.I = len(self.off)
        self.reset()

    cdef void reset(self) nogil:
        """ Reset signal values to off values. """
        self.set_value(self.off)

    cdef void update(self, double t) nogil:
        """
        Update signal values for specified time.

        Args:

            t (double) - time

        """
        if t >= self.t_on:
            if t < self.t_off:
                self.set_value(self.on)
            else:
                self.set_value(self.off)
        else:
            pass


cdef class cMultiPulse(cSignal):
    """
    Class defines a multi channel square pulse signal.

    Attributes:

        t_on (array[float]) - pulse onset time

        t_off (array[float]) - pulse off time

        off (array[float]) - signal off value

        on (array[float]) - signal on value

    Inherited Attributes:

        value (array[double]) - signal off-values

        I (int) - number of signal channels (always one)

    """

    def __init__(self, t_on, t_off, off, on):
        """
        Instantiate a multi-channel square pulse signal.

        Args:

            t_on (array[float]) - pulse onset times

            t_off (array[float]) - pulse off times

            off (array[float]) - signal off values

            on (array[float]) - signal on values

        """
        self.I = len(off)
        self.t_on = array('d', np.array(t_on, dtype=np.float64))
        self.t_off = array('d', np.array(t_off, dtype=np.float64))
        self.off = array('d', np.array(off, dtype=np.float64))
        self.on = array('d', np.array(on, dtype=np.float64))
        self.reset()

    cdef void reset(self) nogil:
        """ Reset signal values to off values. """
        self.set_value(self.off)

    cdef void update(self, double t) nogil:
        """
        Update signal values for specified time.

        Args:

            t (double) - time

        """

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
    """
    Class defines a multi-channel square wave signal.

    Attributes:

        period (float) - signal oscillation period

        off (array[float]) - signal off values

        on (array[float]) - signal on values

    Inherited Attributes:

        value (array[double]) - signal off-values

        I (int) - number of signal channels (always one)

    """

    def __init__(self, period=1., off=0., on=1.):
        """
        Instantiate a multi-channel square wave signal. Note that all channels must follow the same oscillation frequency.

        Args:

            period (float) - signal oscillation period

            off (array[float]) - signal off values

            on (array[float]) - signal on values

        """
        self.period = period
        self.off = array('d', np.array([off], dtype=np.float64).flatten())
        self.on = array('d', np.array([on], dtype=np.float64).flatten())
        self.I = len(self.off)
        self.reset()

    cdef void reset(self) nogil:
        """ Reset signal values to off values. """
        self.set_value(self.off)

    cdef void update(self, double t) nogil:
        """
        Update signal values for specified time.

        Args:

            t (double) - time

        """

        if (t // self.period) % 2 == 0:
            self.set_value(self.off)
        else:
            self.set_value(self.on)
