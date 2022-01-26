# cython external imports

# import intra-package cython dependencies
from ..signals.signals cimport cSignalType
from .stochastic cimport cStochasticSystem


cdef class cDebug(cStochasticSystem):

    # methods
    cpdef tuple run(self,
        unsigned int[:] ic,
        double[:] integrator_ic,
        cSignalType signal,
        double duration=*,
        double sampling_interval=*,
        int seed=*)

    cdef void ssa(self,
        cSignalType signal,
        double duration,
        double sampling_interval) with gil
