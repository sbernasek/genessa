# cython external imports
cimport numpy as np

# import intra-package cython dependencies
from .rates cimport cRates
from .stoichiometry cimport cStoichiometry


cdef class cDeterministicSystem:

    # attributes
    cdef unsigned int N, M, I
    cdef cStoichiometry S
    cdef cRates R

    # methods
    cdef double* get_rxn_rates(self)

    cdef double get_total_rxn_rate(self) nogil

    cpdef double[:] c_evaluate_species_rates(self,
        double[::1] states,
        double[::1] inputs,
        double[::1] cumulative)
