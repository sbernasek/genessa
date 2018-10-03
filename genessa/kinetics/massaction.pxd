from cpython.array cimport array

# cython intra-package imports
from .base cimport cInputDependent


cdef class cMassAction(cInputDependent):

    # methods
    @staticmethod
    cdef cMassAction get_blank_cMassAction()

    @staticmethod
    cdef cMassAction from_list(list rxns)

    cdef double evaluate_rxn_rate(self,
        unsigned int rxn,
        unsigned int *states,
        double *inputs) nogil

    cdef double c_evaluate_rate(self,
        unsigned int rxn,
        array states,
        array inputs) nogil
