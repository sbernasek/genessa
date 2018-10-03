from cpython.array cimport array

# cython intra-package imports
from .base cimport cInputDependent


cdef class cIDRepressor(cInputDependent):

    # attributes
    cdef array k_m, n

    # methods
    @staticmethod
    cdef cIDRepressor get_blank_cIDRepressor()

    @staticmethod
    cdef cIDRepressor from_list(list rxns)

    cdef double get_species_activity(self,
        unsigned int rep,
        unsigned int *states) nogil

    cdef double get_input_activity(self,
        unsigned int rep,
        double *inputs) nogil

    cdef double get_occupancy(self,
        unsigned int rep,
        unsigned int *states,
        double *inputs) nogil

    cdef double cget_occupancy(self,
        array states,
        array input_values,
        unsigned int rep) nogil


cdef class cHill(cIDRepressor):

    # attributes
    cdef cIDRepressor rep_obj
    cdef array repressors_ind, n_repressors

    # methods
    @staticmethod
    cdef cHill get_blank_cHill()

    @staticmethod
    cdef cHill from_list(list rxns)

    cdef double get_species_activity(self,
        unsigned int rxn,
        unsigned int *states) nogil

    cdef double get_input_activity(self,
        unsigned int rxn,
        double *inputs) nogil

    cdef double get_availability(self,
        unsigned int rxn,
        unsigned int *states,
        double *inputs)nogil

    cdef double evaluate_rxn_rate(self,
        unsigned int rxn,
        unsigned int *states,
        double *inputs) nogil

    cdef double c_evaluate_rate(self,
        unsigned int rxn,
        array states,
        array inputs) nogil
