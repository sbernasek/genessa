# cython: profile=False

# cython external imports
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free

# python external imports
import numpy as np
import ctypes

# cython intra-package imports
from .signals cimport cSignal, cSquarePulse, cMultiPulse, cSquareWave


cdef class cSignal:
    """
    Class defines a constant signal.

    Attributes:

        I (unsigned int) - number of signal channels

        value (double*) - signal values

    """

    def __init__(self, value):
        """
        Instantiate a constant signal.

        Args:

            value (float) - constant signal value

        """
        assert self.I==1, 'Too many signal channels specified.'
        self.value[0] = value

    def __cinit__(self, argument, *args, **kwargs):
        """
        Allocate memory for signal values.

        The shape of the first positioal argument is used to determine the number of signal channels.

        Args:

            argument (float or array like) - first positional argument

            args: arguments reserved for subclassing

            kwargs: keyword arguments reserved for subclassing

        """

        # check whether value is a collection (for multi-valued subclasses)
        try:
            I = len(argument)
        except TypeError:
            I = 1

        self.I = I
        self.value = <double*> PyMem_Malloc(I * sizeof(double))
        if not self.value:
            raise MemoryError('Could not allocate memory for signal values.')

    def __dealloc__(self):
        """ Deallocate memory from array attributes. """
        PyMem_Free(self.value)

    def __call__(self, double t):
        """
        Returns signal value at specified time.

        Args:

            t (double) - time

        Returns:

            value (double) - signal value

        """
        return self.get_value(0, t)

    cpdef double get_value(self, unsigned int index, double t):
        """
        Returns individual signal value at specified time.

        Args:

            index (unsigned int) - signal channel

            t (double) - time

        Returns:

            value (double) - signal value

        """
        self.update(t)
        return self.value[index]

    cpdef np.ndarray get_values(self, double t):
        """
        Returns array of all signal value(s) at specified time.

        Args:

            t (double) - time

        Returns:

            values (np.ndarray[double]) - signal value(s)

        Note: returned values share their memory with the signal.value pointer, and the returned array values will change each time the signal is updated.

        """
        self.update(t)
        return np.asarray(<np.float64_t[:self.I]> self.value)

    cdef void update(self, double t) nogil:
        pass

    cdef void reset(self) nogil:
        pass

    cdef void set_value(self, double *values) nogil:
        """
        Set signal values.

        Args:

            values (double*) - signal values

        """
        cdef unsigned int index
        for index in xrange(self.I):
            self.value[index] = values[index]

    cdef bint compare_value(self, double *values, unsigned int index) nogil:
        """
        Compare individual signal value with a reference.

        Args:

            values (double*) - reference values

            index (unsigned int) - signal channel compared

        Returns:

            not_equal (bint) - if True, signal values are not equal

        """
        cdef bint not_equal = 0
        if self.value[index] != values[index]:
            not_equal = 1
        return not_equal

    cdef bint compare_all(self, double *values) nogil:
        """
        Compare all signal values with a reference.

        Args:

            values (double*) - reference values

        Returns:

            not_equal (bint) - if True, signal values differ from reference

        """
        cdef bint not_equal = 0
        cdef unsigned int index
        for index in xrange(self.I):
            if self.value[index] != values[index]:
                not_equal = 1
                break
        return not_equal


cdef class cSquarePulse(cSignal):
    """
    Class defines a single-channel square pulse signal.

    Attributes:

        t_on (double) - pulse onset time

        t_off (double) - pulse off time

        off (double) - signal off value

        on (double) - signal on value

    Inherited Attributes:

        I (unsigned int) - number of signal channels (always one)

        value (double*) - signal value(s)

    """

    def __init__(self, t_on, t_off=3, off=0, on=1):
        """
        Instantiate a single-channel square pulse signal.

        Args:

            t_on (double) - pulse onset time

            t_off (double) - pulse off time

            off (double) - signal off value

            on (double) - signal on value

        """
        super().__init__(off)
        self.t_on = t_on
        self.t_off = t_off
        self.off = off
        self.on = on
        self.reset()

    cdef void reset(self) nogil:
        """ Reset signal values to off values. """
        self.set_value(&self.off)

    cdef void update(self, double t) nogil:
        """
        Update signal values for specified time.

        Args:

            t (double) - time

        """
        if t >= self.t_on:
            if t < self.t_off:
                self.set_value(&self.on)
            else:
                self.set_value(&self.off)
        else:
            self.set_value(&self.off)


cdef class cSquareWave(cSignal):
    """
    Class defines a single-channel square wave signal.

    Attributes:

        period (double) - signal oscillation period

        off (double) - signal off value

        on (double) - signal on value

    Inherited Attributes:

        I (unsigned int) - number of signal channels (must be 1)

        value (double*) - signal value(s)

    """

    def __init__(self, period, off=0, on=1):
        """
        Instantiate a single-channel square wave signal. Note that all channels must follow the same oscillation frequency.

        Args:

            period (double) - signal oscillation period

            off (double) - signal off value

            on (double) - signal on value

        """
        super().__init__(off)
        self.period = period
        self.off = off
        self.on = on

    cdef void reset(self) nogil:
        """ Reset signal to off value. """
        self.set_value(&self.off)

    cdef void update(self, double t) nogil:
        """
        Update signal value for specified time.

        Args:

            t (double) - time

        """
        if (t // (self.period/2)) % 2 == 0:
            self.set_value(&self.off)
        else:
            self.set_value(&self.on)


cdef class cMultiPulse(cSignal):
    """
    Class defines a multi channel square pulse signal.

    Attributes:

        t_on (double*) - pulse onset time(s)

        t_off (double*) - pulse off time(s)

        off (double*) - signal off value(s)

        on (double*) - signal on value(s)

    Inherited Attributes:

        I (unsigned int) - number of signal channels

        value (double*) - signal off-value(s)

    """

    def __init__(self, t_on, t_off=None, off=None, on=None):
        """
        Instantiate a multi-channel square pulse signal.

        Args:

            t_on (array like [float]) - pulse onset time(s)

            t_off (array like [float]) - pulse off time(s)

            off (array like [float]) - signal off value(s)

            on (array like [float]) - signal on value(s)

        """

        # set defaults
        if t_off is None:
            t_off = np.ones(self.I, dtype=np.float64)
        if off is None:
            off = np.zeros(self.I, dtype=np.float64)
        if on is None:
            on = np.ones(self.I, dtype=np.float64)

        # populate memory with constant signal values
        for i in xrange(self.I):
            self.t_on[i] = t_on[i]
            self.t_off[i] = t_off[i]
            self.off[i] = off[i]
            self.on[i] = on[i]

        # set values to off state
        self.reset()

    def __cinit__(self, t_on, *args, **kwargs):
        """
        Allocate memory for signal values.

        Args:

            t_on (array like [float]) - pulse onset time(s)

        """
        self.t_on = <double*> PyMem_Malloc(self.I * sizeof(double))
        if not self.t_on:
            raise MemoryError('Could not allocate memory for on-time values.')

        self.t_off = <double*> PyMem_Malloc(self.I * sizeof(double))
        if not self.t_off:
            raise MemoryError('Could not allocate memory for off-time values.')

        self.off = <double*> PyMem_Malloc(self.I * sizeof(double))
        if not self.off:
            raise MemoryError('Could not allocate memory for off values.')

        self.on = <double*> PyMem_Malloc(self.I * sizeof(double))
        if not self.on:
            raise MemoryError('Could not allocate memory for on values.')

    def __dealloc__(self):
        """ Deallocate memory from array attributes. """
        PyMem_Free(self.t_on)
        PyMem_Free(self.t_off)
        PyMem_Free(self.off)
        PyMem_Free(self.on)

    def __call__(self, double t):
        """
        Returns all signal values at specified time.

        Args:

            t (double) - time

        Returns:

            values (np.ndarray[double]) - signal value(s)

        Note: returned values share their memory with the signal.value pointer, and the returned array values will change each time the signal is updated.

        """
        return self.get_values(t)

    cdef void reset(self) nogil:
        """ Reset signal values to off values. """
        self.set_value(self.off)

    cdef void update(self, double t) nogil:
        """
        Update signal value(s) for specified time.

        Args:

            t (double) - time

        """

        cdef unsigned int index

        # get input from each channel
        for index in xrange(self.I):
            if t >= self.t_on[index]:
                if t < self.t_off[index]:
                    self.value[index] = self.on[index]
                else:
                    self.value[index] = self.off[index]
            else:
                self.value[index] = self.off[index]
