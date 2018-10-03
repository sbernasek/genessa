# cython external imports
cimport numpy as np
from cpython.array cimport array

# python external imports
import numpy as np
from array import array

# cython intra-package imports
from .base cimport cSpeciesDependent, cInputDependent


cdef class cSpeciesDependent:

    def __init__(self,
                 unsigned int M,
                 double[:] k,
                 unsigned int[:] species_ind,
                 unsigned int[:] species,
                 double[:] species_dependence):

        # store number of reactions and rate constants
        self.M = M
        self.k = array('d', k)

        # add state dependence
        self.species_ind = array('I', species_ind)
        self.n_active_species = array('I', np.diff(species_ind).astype(np.uint32))
        self.species = array('I', species)
        self.species_dependence = array('d', species_dependence)

        # initialize rate vector
        self.rates = array('d', np.zeros(M, dtype=np.float64))

    cdef double get_species_activity(self,
                                     unsigned int rxn,
                                     unsigned int *states) nogil:
        return self.get_species_activity_product(rxn, states)

    cdef double get_species_activity_product(self,
                                             unsigned int rxn,
                                             unsigned int *states) nogil:
        """ Integrate species activity for specified reaction. """
        cdef unsigned int n
        cdef unsigned int count, state
        cdef double k
        cdef unsigned int index = self.species_ind.data.as_uints[rxn]
        cdef unsigned int N = self.n_active_species.data.as_uints[rxn]
        cdef double activity = 1

        # integrate species activity
        for count in xrange(N):
            state = self.species.data.as_uints[index]
            n = states[state]
            k = self.species_dependence.data.as_doubles[index]
            if k == 1:
                activity *= n
            elif k == 2:
                activity *= (n*(n-1)/2)
            else:
                activity = 0
            index += 1

        return activity

    cdef double get_species_activity_sum(self,
                                         unsigned int rxn,
                                         unsigned int *states) nogil:
        """ Integrate species activity for specified reaction. """
        cdef unsigned int n
        cdef unsigned int count, state
        cdef double k
        cdef unsigned int index = self.species_ind.data.as_uints[rxn]
        cdef unsigned int N = self.n_active_species.data.as_uints[rxn]
        cdef double activity = 0

        # integrate species activity
        for count in xrange(N):
            state = self.species.data.as_uints[index]
            n = states[state]
            k = self.species_dependence.data.as_doubles[index]
            activity += (n * k)
            index += 1

        return activity


cdef class cInputDependent(cSpeciesDependent):

    def __init__(self,
                 int M,
                 double[:] k,
                 unsigned int[:] species_ind,
                 unsigned int[:] species,
                 double[:] species_dependence,
                 unsigned int[:] inputs_ind,
                 unsigned int[:] inputs,
                 double[:] input_dependence):

        # add species dependence
        cSpeciesDependent.__init__(self, M, k, species_ind, species, species_dependence)

        # add input dependence
        self.inputs_ind = array('I', inputs_ind)
        self.n_active_inputs = array('I', np.diff(inputs_ind).astype(np.uint32))
        self.inputs = array('I', inputs)
        self.input_dependence = array('d', input_dependence)

    cdef double get_input_activity(self,
                                   unsigned int rxn,
                                   double *inputs) nogil:
        return self.get_input_activity_product(rxn, inputs)

    cdef double get_input_activity_product(self,
                                           unsigned int rxn,
                                           double *inputs) nogil:
        """ Integrate input activity for specified reaction. """
        cdef unsigned int count, dim
        cdef double n, k
        cdef unsigned int index = self.inputs_ind.data.as_uints[rxn]
        cdef unsigned int I = self.n_active_inputs.data.as_uints[rxn]
        cdef double activity = 1.

        # integrate input activity
        for count in xrange(I):
            dim = self.inputs.data.as_uints[index]
            n = inputs[dim]
            k = self.input_dependence.data.as_doubles[index]
            activity *= (n**k)
            index += 1

        return activity

    cdef double get_input_activity_sum(self,
                                       unsigned int rxn,
                                       double *inputs) nogil:
        """ Integrate input activity for specified reaction. """
        cdef unsigned int count, dim
        cdef double n, k
        cdef unsigned int index = self.inputs_ind.data.as_uints[rxn]
        cdef unsigned int I = self.n_active_inputs.data.as_uints[rxn]
        cdef double activity = 0.

        # integrate input activity
        for count in xrange(I):
            dim = self.inputs.data.as_uints[index]
            n = inputs[dim]
            k = self.input_dependence.data.as_doubles[index]
            activity += (n*k)
            index += 1

        return activity
