# cython external imports

# import intra-package cython dependencies
from ..signals.signals cimport cSignalType
from .stochastic cimport cStochasticSystem


cdef class cDebug(cStochasticSystem):

    # methods
    cdef void ssa(self,
        cSignalType signal,
        double duration,
        double sampling_interval) with gil

    cdef unsigned int choose_rxn(self,
        double random) with gil

    cdef void fire_reaction(self,
        unsigned int rxn,
        unsigned int extent,
        unsigned int *states) with gil

    cdef void update_cumulative(self,
        unsigned int *states,
        double *cumulative,
        double tau) with gil

    cdef void sample(self) with gil

    cdef void record(self,
        double end_time) with gil
