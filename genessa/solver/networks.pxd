# cython external imports
from cpython.array cimport array

# import intra-package cython dependencies
from .rxns cimport cRateFunction


cdef class cStoichiometry:

    cdef array index
    cdef array lengths
    cdef array species
    cdef array coefficients


cdef class cNetwork:

    cdef unsigned int N, M, I
    cdef cStoichiometry S
    cdef cRateFunction R

    cdef array get_rxn_rates(self)

    cdef double get_total_rate(self) nogil

    cdef void update_all(self,
                         array states,
                         array inputs,
                         array cumulative) nogil

    cdef void update_input(self,
                           array states,
                           array inputs,
                           array cumulative,
                           unsigned int dim) nogil

    cdef void update(self,
                     array states,
                     array inputs,
                     array cumulative,
                     unsigned int rxn_fired) nogil

    cdef void cset_species_rates(self,
                                 array states,
                                 array inputs,
                                 array cumul,
                                 array rates)
