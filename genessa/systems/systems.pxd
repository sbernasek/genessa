# cython external imports
from cpython.array cimport array

# import intra-package cython dependencies
from .rates cimport cRates


cdef class cStoichiometry:

    # attributes
    cdef array index
    cdef array lengths
    cdef array species
    cdef array coefficients


cdef class cSystem:

    # attributes
    cdef unsigned int N, M, I
    cdef cStoichiometry S
    cdef cRates R

    # methods
    cdef array get_rxn_rates(self)

    cdef double get_total_rxn_rate(self) nogil

    cdef array c_evaluate_species_rates(self,
                                       array states,
                                       array inputs,
                                       array cumulative)
