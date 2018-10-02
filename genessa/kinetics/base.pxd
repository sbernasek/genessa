from cpython.array cimport array


cdef class cSpeciesDependent:

    # attributes
    cdef unsigned int M
    cdef array k
    cdef array species_ind, n_active_species, species, species_dependence
    cdef array rates

    # methods
    cdef double get_species_activity(self, unsigned int rxn, array states) nogil
    cdef double get_species_activity_product(self, unsigned int rxn, array states) nogil
    cdef double get_species_activity_sum(self, unsigned int rxn, array states) nogil


cdef class cInputDependent(cSpeciesDependent):

    # attributes
    cdef array inputs_ind, n_active_inputs, inputs, input_dependence

    # methods
    cdef double get_input_activity(self, unsigned int rxn, array inputs) nogil
    cdef double get_input_activity_product(self, unsigned int rxn, array input_values) nogil
    cdef double get_input_activity_sum(self, unsigned int rxn, array input_values) nogil
