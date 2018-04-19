from cpython.array cimport array
from array import array

cdef inline double sum_double_arr(array arr, int length):
    cdef double _sum = arr.data.as_doubles[0]
    cdef int i
    for i in xrange(1, length):
        _sum += arr.data.as_doubles[i]
    return _sum

cdef inline long sum_long_arr(array arr, int length):
    cdef long _sum = arr.data.as_longs[0]
    cdef int i
    for i in xrange(1, length):
        _sum += arr.data.as_longs[i]
    return _sum

cdef inline int sum_int_arr(array arr, int length):
    cdef int _sum = arr.data.as_ints[0]
    cdef int i
    for i in xrange(1, length):
        _sum += arr.data.as_ints[i]
    return _sum

cdef inline long min_long_arr(array arr, int length):
    cdef int i
    cdef long val
    cdef long min = arr.data.as_longs[0]
    for i in xrange(length-1):
        val = arr.data.as_longs[i+1]
        if val < min:
            min = val
    return min

cdef inline double compute_float_sum(double[:] y, int N):
    cdef double x = y[0]
    cdef int i
    for i in xrange(1,N):
        x += y[i]
    return x

cdef inline int compute_int_sum(int[:] y, int N):
    cdef int x = y[0]
    cdef int i
    for i in xrange(1,N):
        x += y[i]
    return x

cdef inline long compute_long_sum(long[:] y, int N):
    cdef long x = y[0]
    cdef int i
    for i in xrange(1,N):
        x += y[i]
    return x

cdef inline double get_min_float(double[:] array, int N):
    """ Returns min value from array of N floats """
    cdef int i
    cdef double val
    cdef double min = array[0]
    for i in xrange(N-1):
        val = array[i+1]
        if val < min:
            min = val
    return min

cdef inline long get_min_int(long[:] array, int N):
    """ Returns min value from array of N integers """
    cdef int i
    cdef long val
    cdef long min = array[0]
    for i in xrange(N-1):
        val = array[i+1]
        if val < min:
            min = val
    return min

