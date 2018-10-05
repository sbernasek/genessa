from cpython.array cimport array

# cython intra-package imports
from .base cimport cSpeciesDependent


cdef class cPController(cSpeciesDependent):

    # methods
    @staticmethod
    cdef cPController get_blank_cPController()

    @staticmethod
    cdef cPController from_list(list rxns)

    cdef double evaluate_rxn_rate(self,
        unsigned int rxn,
        unsigned int *controlled) nogil

    cdef double c_evaluate_rate(self,
        unsigned int rxn,
        double *controlled) nogil


cdef class cIController(cSpeciesDependent):

    # methods
    @staticmethod
    cdef cIController get_blank_cIController()

    @staticmethod
    cdef cIController from_list(list rxns)

    cdef double evaluate_integrator_sum(self,
        unsigned int rxn,
        double *controlled) nogil

    cdef double evaluate_rxn_rate(self,
        unsigned int rxn,
        double *controlled) nogil

    cdef double c_evaluate_rate(self,
        unsigned int rxn,
        double *controlled) nogil
