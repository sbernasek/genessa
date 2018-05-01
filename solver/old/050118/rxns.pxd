from cpython.array cimport array
from array import array
cimport numpy as np


cdef class cSpeciesDependent:
    cdef int M
    cdef array k
    cdef array species_ind, n_active_species, species, species_dependence
    cdef array rates

    # methods
    cdef double get_rate(self, int rxn) nogil
    cdef double get_species_activity(self, int rxn, array states) nogil
    cdef int get_species_activity_product(self, int rxn, array states) nogil
    cdef double get_species_activity_sum(self, int rxn, array states) nogil


cdef class cPController(cSpeciesDependent):
    # methods
    @staticmethod
    cdef cPController get_blank_cPController()
    @staticmethod
    cdef cPController from_list(list rxns)
    cdef double get_species_activity(self, int rxn, array states) nogil
    cdef double update(self, int rxn, array states) nogil


cdef class cIController(cPController):
    # methods
    @staticmethod
    cdef cIController get_blank_cIController()
    @staticmethod
    cdef cIController from_list(list rxns)
    cdef double get_species_activity_sum(self, int rxn, array cumul) nogil


cdef class cInputDependent(cSpeciesDependent):
    cdef array inputs_ind, n_active_inputs, inputs, input_dependence

    # methods
    cdef double get_input_activity(self, int rxn, array inputs) nogil
    cdef double get_input_activity_product(self, int rxn, array input_values) nogil
    cdef double get_input_activity_sum(self, int rxn, array input_values) nogil


cdef class cMassAction(cInputDependent):
    # methods
    @staticmethod
    cdef cMassAction get_blank_cMassAction()
    @staticmethod
    cdef cMassAction from_list(list rxns)
    cdef double update(self, int rxn, array states, array inputs) nogil


cdef class cSDRepressor(cSpeciesDependent):
    cdef array k_m, n

    # methods
    @staticmethod
    cdef cSDRepressor get_blank_cSDRepressor()
    @staticmethod
    cdef cSDRepressor from_list(list rxns)
    cdef double get_species_activity(self, int rep, array states) nogil
    cdef double get_occupancy(self, int rep, array states) nogil


cdef class cIDRepressor(cInputDependent):
    cdef array k_m, n

    # methods
    @staticmethod
    cdef cIDRepressor get_blank_cIDRepressor()
    @staticmethod
    cdef cIDRepressor from_list(list rxns)
    cdef double get_species_activity(self, int rep, array states) nogil
    cdef double get_input_activity(self, int rep, array inputs) nogil
    cdef double get_occupancy(self, int rep, array states, array inputs) nogil


cdef class cHill(cIDRepressor):
    cdef cIDRepressor rep_obj
    cdef array repressors_ind, n_repressors

    # methods
    @staticmethod
    cdef cHill get_blank_cHill()
    @staticmethod
    cdef cHill from_list(list rxns)
    cdef double get_species_activity(self, int rxn, array states) nogil
    cdef double get_input_activity(self, int rxn, array inputs) nogil
    cdef double get_availability(self, int rxn, array states, array inputs)nogil
    cdef double update(self, int rxn, array states, array inputs) nogil


cdef class cCoupling(cSpeciesDependent):
    cdef array a, w
    cdef cSDRepressor rep_obj
    cdef array repressors_ind, n_repressors

    # methods
    @staticmethod
    cdef cCoupling get_blank_cCoupling()
    @staticmethod
    cdef cCoupling from_list(list rxns)
    cdef double get_species_activity(self, int rxn, array states) nogil
    cdef double get_availability(self, int rxn, array states) nogil
    cdef double update(self, int rxn, array states) nogil


ctypedef void (*cSetRate)(cRateFunction, int, array, array, array) nogil
cdef class cRxnMap:
    cdef array ind, lengths, values

    # methods
    cdef void app(self, cRateFunction rf, int key, cSetRate f, array states, array inputs, array cumul) nogil


cdef class cRateFunction:
    cdef cCoupling coupling
    cdef cMassAction massaction
    cdef cHill hill
    cdef cIController icontrol
    cdef cPController pcontrol
    cdef int M
    cdef array rxn_types, rxn_keys, rates
    cdef double total_rate
    cdef cRxnMap rxn_map, input_map

    # methods
    cdef array get_rxn_rates(self)
    cdef double evaluate(self, int rxn, array states, array inputs, array cumul) nogil
    cdef void set_rate(self, int rxn, array states, array inputs, array cumul) nogil
    cdef void update_input(self, array states, array inputs, array cumul, int dim) nogil
    cdef void update(self, array states, array inputs, array cumul, int fired) nogil
    cdef void update_all(self, array states, array inputs, array cumul) nogil

