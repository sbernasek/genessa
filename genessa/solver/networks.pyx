# cython external imports
cimport numpy as np
from cpython.array cimport array

# python external imports
import numpy as np

# import intra-package cython dependencies
from .networks cimport cStoichiometry, cNetwork
from .rxns cimport cRateFunction


cdef class cStoichiometry:
    """
    Cython hash table containing stoichiometric coefficients for all reactions in a network.

    Attributes:

        index (array[unsigned int]) - start index for each reaction

        lengths (array[unsigned int]) - number of nonzero coefficients for each reaction

        species (array[unsigned int]) - node index for each coefficient

        coefficients (array[long]) - coefficient values

    """

    def __init__(self, unsigned int[:] index,
                       unsigned int[:] lengths,
                       unsigned int[:] species,
                       long[:] coefficients):

        self.index = array('I', index)
        self.lengths = array('I', lengths)
        self.species = array('I', species)
        self.coefficients = array('l', coefficients)

    @staticmethod
    def from_array(np.ndarray s):
        """ Instantiate from N x M stoichiometry array. """
        rxns, species = s.T.nonzero()
        lengths = np.bincount(rxns).astype(np.uint32)
        index = np.hstack((np.zeros(1), np.cumsum(lengths))).astype(np.uint32)
        coefficients = s.T[(rxns, species)].astype(np.int64)
        return cStoichiometry(index, lengths, species.astype(np.uint32), coefficients)



cdef class cNetwork:
    """
    Class defines a network of interacting nodes.

    Attributes:

        N (unsigned int) - number of nodes

        M (unsigned int) - number of reactions

        I (unsigned int) - number of external inputs

        S (cStoichiometry) - stoichiometry for all reactions

        R (cRateFunction) - rate function for all reactions

    """

    def __init__(self,
                 unsigned int N,
                 unsigned int M,
                 unsigned int I,
                 cStoichiometry S,
                 cRateFunction R):
        self.N = N
        self.M = M
        self.I = I
        self.S = S
        self.R = R

    cdef array get_rxn_rates(self):
        return self.R.rates

    cdef double get_total_rate(self) nogil:
        return self.R.total_rate

    cdef void update_all(self,
                         array states,
                         array inputs,
                         array cumulative) nogil:
        self.R.update_all(states, inputs, cumulative)

    cdef void update_input(self,
                           array states,
                           array inputs,
                           array cumulative,
                           unsigned int dim) nogil:
        self.R.update_input(states, inputs, cumulative, dim)

    cdef void update(self,
                     array states,
                     array inputs,
                     array cumulative,
                     unsigned int rxn_fired) nogil:
        self.R.update(states, inputs, cumulative, rxn_fired)

    cdef void cset_species_rates(self,
                                 array states,
                                 array inputs,
                                 array cumul,
                                 array rates):

        cdef unsigned int rxn
        cdef double rxn_rate
        cdef unsigned int N, index, count, species
        cdef int coefficient

        # get reaction rates
        self.R.cupdate(states, inputs, cumul)

        # for each reaction
        for rxn in xrange(self.M):
            rxn_rate = self.R.rates.data.as_doubles[rxn]
            N = self.S.lengths.data.as_uints[rxn]
            index = self.S.index.data.as_uints[rxn]

            # update each active state
            for count in xrange(N):
                species = self.S.species.data.as_uints[index]
                coefficient = self.S.coefficients.data.as_longs[index]
                rates.data.as_doubles[species] += (coefficient * rxn_rate)
                index += 1
