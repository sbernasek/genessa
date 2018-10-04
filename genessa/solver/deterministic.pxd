# cython external imports
from cpython.array cimport array
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

    cpdef array c_evaluate_species_rates(self,
        np.ndarray states,
        array inputs,
        np.ndarray cumulative)
