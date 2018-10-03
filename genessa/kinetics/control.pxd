from cpython.array cimport array

# cython intra-package imports
from .base cimport cSpeciesDependent


cdef class cPController(cSpeciesDependent):

    # methods
    @staticmethod
    cdef cPController get_blank_cPController()
    @staticmethod
    cdef cPController from_list(list rxns)
    cdef double get_species_activity(self, unsigned int rxn, array states) nogil
    cdef double evaluate_rxn_rate(self, unsigned int rxn, array states) nogil
    cdef double c_evaluate_rate(self, unsigned int rxn, array states) nogil


cdef class cIController(cPController):

    # methods
    @staticmethod
    cdef cIController get_blank_cIController()
    @staticmethod
    cdef cIController from_list(list rxns)
    cdef double get_species_activity_sum(self, unsigned int rxn, array cumul) nogil
