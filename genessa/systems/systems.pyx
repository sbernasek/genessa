# cython external imports
cimport numpy as np
from cpython.array cimport array

# python external imports
import numpy as np

# import intra-package cython dependencies
from .systems cimport cStoichiometry, cSystem
from .rates cimport cRates

# import intra-package python dependencies
from .rates import Rates


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
    def from_array(np.ndarray stoichiometry):
        """ Instantiate from N x M stoichiometry array. """
        rxns, species = stoichiometry.T.nonzero()
        lengths = np.bincount(rxns).astype(np.uint32)
        index = np.hstack((np.zeros(1), np.cumsum(lengths))).astype(np.uint32)
        coefficients = stoichiometry.T[(rxns, species)].astype(np.int64)
        return cStoichiometry(index, lengths, species.astype(np.uint32), coefficients)


cdef class cSystem:
    """
    Class defines a network of interacting nodes.

    Attributes:

        N (unsigned int) - number of nodes

        M (unsigned int) - number of reactions

        I (unsigned int) - number of external inputs

        S (cStoichiometry) - stoichiometry for all reactions

        R (cRates) - rate function for all reactions

    """

    def __init__(self,
                 unsigned int N,
                 unsigned int M,
                 unsigned int I,
                 cStoichiometry S,
                 cRates R):

        self.N = N
        self.M = M
        self.I = I
        self.S = S
        self.R = R

    @staticmethod
    def from_network(network):
        """
        Instantiate from python Network.

        Args:

            network (Network)

        Returns:

            c_system (cSystem)

        """

        # sort rxns and compile stoichiometry
        network.sort_rxns()
        network.resize_inputs()
        network.compile_stoichiometry()

        # typecast network features
        N = network.N
        M = network.M
        I = network.I

        # get cythonized rate function and network
        S = cStoichiometry.from_array(network.stoichiometry)
        R = Rates.compile_c_rate_function(network)

        return cSystem(N, M, I, S, R)

    cdef double* get_rxn_rates(self):
        """ Returns current reaction rates. """
        return self.R.rates

    cdef double get_total_rxn_rate(self) nogil:
        """ Returns current total reaction rate. """
        return self.R.total_rate

    cdef array c_evaluate_species_rates(self,
                                       array states,
                                       array inputs,
                                       array cumulative):
        """
        Evaluate rate of change for all species.

        Args:

            states (array[double]) - state values

            inputs (array[double]) - input values

            cumulative (array[double]) - integrator values

        Returns:

            rates (array[double]) - species rates, e.g. dX/dt

        """

        cdef unsigned int rxn
        cdef double rxn_rate
        cdef unsigned int N, index, count, species
        cdef int coefficient
        cdef array rxn_rates

        # instantiate array of zeros
        cdef array rates = array('d', self.N*[0.])

        # evaluate reaction rates
        rxn_rates = self.R.c_evaluate_rxn_rates(states, inputs, cumulative)

        # for each reaction
        for rxn in xrange(self.M):
            rxn_rate = rxn_rates.data.as_doubles[rxn]
            N = self.S.lengths.data.as_uints[rxn]
            index = self.S.index.data.as_uints[rxn]

            # update each active state
            for count in xrange(N):
                species = self.S.species.data.as_uints[index]
                coefficient = self.S.coefficients.data.as_longs[index]
                rates.data.as_doubles[species] += (coefficient * rxn_rate)
                index += 1

        return rates
