from cpython.array cimport array


cdef class cSpeciesDependent:
    cdef unsigned int M
    cdef array k
    cdef array species_ind, n_active_species, species, species_dependence
    cdef array rates

    # methods
    cdef double get_species_activity(self, unsigned int rxn, array states) nogil
    cdef double get_species_activity_product(self, unsigned int rxn, array states) nogil
    cdef double get_species_activity_sum(self, unsigned int rxn, array states) nogil


cdef class cPController(cSpeciesDependent):
    # methods
    @staticmethod
    cdef cPController get_blank_cPController()
    @staticmethod
    cdef cPController from_list(list rxns)
    cdef double get_species_activity(self, unsigned int rxn, array states) nogil
    cdef double update(self, unsigned int rxn, array states) nogil
    cdef double cget_rate(self, unsigned int rxn, array states) nogil


cdef class cIController(cPController):
    # methods
    @staticmethod
    cdef cIController get_blank_cIController()
    @staticmethod
    cdef cIController from_list(list rxns)
    cdef double get_species_activity_sum(self, unsigned int rxn, array cumul) nogil


cdef class cInputDependent(cSpeciesDependent):
    cdef array inputs_ind, n_active_inputs, inputs, input_dependence

    # methods
    cdef double get_input_activity(self, unsigned int rxn, array inputs) nogil
    cdef double get_input_activity_product(self, unsigned int rxn, array input_values) nogil
    cdef double get_input_activity_sum(self, unsigned int rxn, array input_values) nogil


cdef class cMassAction(cInputDependent):
    # methods
    @staticmethod
    cdef cMassAction get_blank_cMassAction()
    @staticmethod
    cdef cMassAction from_list(list rxns)
    cdef double update(self, unsigned int rxn, array states, array inputs) nogil
    cdef double cget_rate(self, unsigned int rxn, array states, array input_values) nogil


cdef class cSDRepressor(cSpeciesDependent):
    cdef array k_m, n
    cdef cRxnMap rxn_map
    cdef array occupancies

    # methods
    @staticmethod
    cdef cSDRepressor get_blank_cSDRepressor(unsigned int M)
    @staticmethod
    cdef cSDRepressor from_list(list rxns, dict rxn_map)
    cdef double get_species_activity(self, unsigned int rep, array states) nogil
    cdef void set_occupancy(self, unsigned int rep, array states) nogil
    cdef void update(self, array states, unsigned int fired) nogil
    cdef double cget_occupancy(self, array states, unsigned int rep) nogil


cdef class cIDRepressor(cInputDependent):
    cdef array k_m, n

    # methods
    @staticmethod
    cdef cIDRepressor get_blank_cIDRepressor()
    @staticmethod
    cdef cIDRepressor from_list(list rxns)
    cdef double get_species_activity(self, unsigned int rep, array states) nogil
    cdef double get_input_activity(self, unsigned int rep, array inputs) nogil
    cdef double get_occupancy(self, unsigned int rep, array states, array inputs) nogil
    cdef double cget_occupancy(self, array states, array input_values, unsigned int rep) nogil


cdef class cHill(cIDRepressor):
    cdef cIDRepressor rep_obj
    cdef array repressors_ind, n_repressors

    # methods
    @staticmethod
    cdef cHill get_blank_cHill()
    @staticmethod
    cdef cHill from_list(list rxns)
    cdef double get_species_activity(self, unsigned int rxn, array states) nogil
    cdef double get_input_activity(self, unsigned int rxn, array inputs) nogil
    cdef double get_availability(self, unsigned int rxn, array states, array inputs)nogil
    cdef double update(self, unsigned int rxn, array states, array inputs) nogil
    cdef double cget_rate(self, unsigned int rxn, array states, array input_values) nogil


cdef class cCoupling(cSpeciesDependent):
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
    cdef double get_availability(self, unsigned int rxn, array states) nogil
    cdef void update_activity(self, unsigned int edge, array states) nogil
    cdef void update_activities(self, array states, unsigned int fired) nogil
    cdef double update(self, unsigned int rxn, array states) nogil
    cdef double cget_rate(self, unsigned int rxn, array states) nogil


ctypedef void (*cSetRate)(cRateFunction, unsigned int, array, array, array) nogil
ctypedef void (*cSetOccupancy)(cSDRepressor, unsigned int, array) nogil
ctypedef void (*cSetEdge)(cCoupling, unsigned int, array) nogil

cdef class cRxnMap:
    cdef array ind, lengths, values

    # methods
    cdef void app(self, cRateFunction rf, unsigned int key, cSetRate f, array states, array inputs, array cumul) nogil
    cdef void app_rep(self, cSDRepressor rep_obj, unsigned int key, cSetOccupancy f, array states) nogil
    cdef void app_coup(self, cCoupling coupling_obj, unsigned int key, cSetEdge f, array states) nogil


cdef class cRateFunction:
    cdef cCoupling coupling
    cdef cMassAction massaction
    cdef cHill hill
    cdef cIController icontrol
    cdef cPController pcontrol
    cdef unsigned int M
    cdef array rxn_types, rxn_keys, rates
    cdef double total_rate
    cdef cRxnMap rxn_map, input_map

    # methods
    cdef double evaluate(self, unsigned int rxn, array states, array inputs, array cumul) nogil
    cdef void set_rate(self, unsigned int rxn, array states, array inputs, array cumul) nogil
    cdef void update_input(self, array states, array inputs, array cumul, unsigned int dim) nogil
    cdef void update(self, array states, array inputs, array cumul, unsigned int fired) nogil
    cdef void update_all(self, array states, array inputs, array cumul) nogil
    #cdef void cupdate(self, array states, array inputs, array cumul) nogil
    cpdef void cupdate(self, array states, array inputs, array cumul) with gil
    cpdef array cget_rxn_rates(self, array states, array inputs, array cumul)



