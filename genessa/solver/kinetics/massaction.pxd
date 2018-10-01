from cpython.array cimport array

# cython intra-package imports
from .base cimport cInputDependent


cdef class cMassAction(cInputDependent):
    # methods
    @staticmethod
    cdef cMassAction get_blank_cMassAction()
    @staticmethod
    cdef cMassAction from_list(list rxns)
    cdef double update(self, unsigned int rxn, array states, array inputs) nogil
    cdef double cget_rate(self, unsigned int rxn, array states, array input_values) nogil
