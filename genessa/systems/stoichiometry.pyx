# cython external imports
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free

# python external imports
import numpy as np

# import intra-package cython dependencies
from .stoichiometry cimport cStoichiometry


cdef class cStoichiometry:
    """
    Cython hash table containing stoichiometric coefficients for all reactions in a network.

    Attributes:

        M (unsigned int) - number of reactions

        Nc (unsigned int) - total number of nonzero coefficients

        index (unsigned int*) - start index for each reaction

        lengths (unsigned int*) - number of nonzero coefficients per reaction

        species (unsigned int*) - node index for each coefficient

        coefficients (int*) - coefficient values

    """

    def __init__(self,
        unsigned int M,
        unsigned int Nc,
        unsigned int[:] index,
        unsigned int[:] lengths,
        unsigned int[:] species,
        long[:] coefficients):
        """
        Instantiate cStoichiometry object.

        Args:

            M (unsigned int) - number of reactions

            Nc (unsigned int) - total number of nonzero coefficients

            index (unsigned int[:]) - start index for each reaction

            lengths (unsigned int[:]) - num. nonzero coefficients per reaction

            species (unsigned int[:]) - node index for each coefficient

            coefficients (int[:]) - coefficient values

        """

        cdef unsigned int i

        # store system size
        self.M = M
        self.Nc = Nc

        # allocate and populate memory for array attributes
        self.allocate_memory()
        for i in xrange(self.M):
            self.index[i] = index[i]
            self.lengths[i] = lengths[i]
        for i in xrange(self.Nc):
            self.species[i] = species[i]
            self.coefficients[i] = coefficients[i]

    def __dealloc__(self):
        """ Deallocate memory from all array attributes. """
        PyMem_Free(self.index)
        PyMem_Free(self.lengths)
        PyMem_Free(self.species)
        PyMem_Free(self.coefficients)

    cdef void allocate_memory(self):
        """ Allocate memory for all array attributes."""

        cdef unsigned int size

        # allocate memory for reaction sort indices
        size = self.M * sizeof(unsigned int)
        self.index = <unsigned int*> PyMem_Malloc(size)
        if not self.index:
            raise MemoryError('Could not allocate index memory.')

        # allocate memory for reaction sort indices
        size = self.M * sizeof(unsigned int)
        self.lengths = <unsigned int*> PyMem_Malloc(size)
        if not self.lengths:
            raise MemoryError('Could not allocate lengths memory.')

        # allocate memory for reaction sort indices
        size = self.Nc * sizeof(unsigned int)
        self.species = <unsigned int*> PyMem_Malloc(size)
        if not self.species:
            raise MemoryError('Could not allocate species memory.')

        # allocate memory for reaction sort indices
        size = self.Nc * sizeof(int)
        self.coefficients = <int*> PyMem_Malloc(size)
        if not self.coefficients:
            raise MemoryError('Could not allocate coefficients memory.')

    @staticmethod
    def from_array(np.ndarray stoichiometry):
        """ Instantiate from N x M stoichiometry array. """

        # extract nonzero stoichiometric coefficients
        rxns, species = stoichiometry.T.nonzero()
        lengths = np.bincount(rxns).astype(np.uint32)
        index = np.hstack((np.zeros(1), np.cumsum(lengths))).astype(np.uint32)
        coefficients = stoichiometry.T[(rxns, species)].astype(np.int64)

        # determine shapes
        num_rxns = len(lengths)
        num_coefficients = len(coefficients)

        return cStoichiometry(<unsigned int> num_rxns,
                              <unsigned int> num_coefficients,
                              index,
                              lengths,
                              species.astype(np.uint32),
                              coefficients)
