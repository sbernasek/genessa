from cpython.array cimport array

# cython intra-package imports
from .base cimport cSpeciesDependent


cdef class cSDRepressor(cSpeciesDependent):

    # attributes
    cdef array k_m, n
    cdef cRxnMap rxn_map
    cdef array occupancies

    # methods
    @staticmethod
    cdef cSDRepressor get_blank_cSDRepressor(unsigned int M)

    @staticmethod
    cdef cSDRepressor from_list(list rxns, dict rxn_map)

    cdef double get_species_activity(self,
        unsigned int rep,
        unsigned int *states) nogil

    cdef void set_occupancy(self,
        unsigned int rep,
        unsigned int *states) nogil

    cdef void update(self,
        unsigned int *states,
        unsigned int fired) nogil

    cdef double cget_occupancy(self,
        double *states,
        unsigned int rep) nogil


cdef class cCoupling(cSpeciesDependent):

    # attributes
    cdef array weight
    cdef cSDRepressor rep_obj
    cdef array repressors_ind, n_repressors
    cdef array edges, edge_to_rxn
    cdef cRxnMap rxn_map
    cdef array activity

    # methods
    @staticmethod
    cdef cCoupling get_blank_cCoupling(unsigned int M)

    @staticmethod
    cdef cCoupling from_list(list rxns, dict edge_map, dict repressor_map)

    cdef double get_availability(self,
        unsigned int rxn,
        unsigned int *states) nogil

    cdef void update_edge(self,
        unsigned int edge,
        unsigned int *states) nogil

    cdef void update_edges(self,
        unsigned int *states,
        unsigned int fired) nogil

    cdef double evaluate_rxn_rate(self,
        unsigned int rxn,
        unsigned int *states) nogil

    cdef double c_evaluate_rate(self,
        unsigned int rxn,
        double *states) nogil


ctypedef void (*cSetOccupancy)(cSDRepressor, unsigned int, unsigned int*) nogil
ctypedef void (*cSetEdge)(cCoupling, unsigned int, unsigned int*) nogil


cdef class cRxnMap:

    # attributes
    cdef array ind, lengths, values

    # methods
    cdef void app(self,
        cCoupling coupling_obj,
        unsigned int key,
        cSetEdge f,
        unsigned int *states) nogil

    cdef void app_rep(self,
        cSDRepressor rep_obj,
        unsigned int key,
        cSetOccupancy f,
        unsigned int *states) nogil
