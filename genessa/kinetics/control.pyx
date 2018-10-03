# cython external imports
cimport numpy as np
from cpython.array cimport array

# python external imports
import numpy as np
from array import array
from functools import reduce
from operator import add
from ..utilities import name_parameter

# cython intra-package imports
from .base cimport cSpeciesDependent
from .control cimport cPController, cIController


cdef class cPController(cSpeciesDependent):

    @staticmethod
    cdef cPController get_blank_cPController():
        cdef np.ndarray xf = np.zeros(1, dtype=np.float64)
        cdef np.ndarray xl = np.zeros(1, dtype=np.uint32)
        return cPController(0, xf, xl, xl, xf)

    @staticmethod
    cdef cPController from_list(list rxns):
        """ Instantiate from list of reactions. """
        cdef unsigned int M
        cdef np.ndarray k
        cdef np.ndarray species_ind, species, species_dependence

        # if no controllers, return blank
        M = len(rxns)
        if M == 0:
            return cPController.get_blank_cPController()

        # get parameters
        k = np.array([rxn.k[0] for rxn in rxns], dtype=np.float64)

        # get species dependence
        species_ind = np.cumsum([0]+[rxn.num_active_species for rxn in rxns]).astype(np.uint32)
        species = np.hstack([rxn.active_species for rxn in rxns]).astype(np.uint32)
        species_dependence = np.hstack([rxn._propensity for rxn in rxns])
        return cPController(M, k, species_ind, species, species_dependence)

    cdef double get_species_activity(self,
                                     unsigned int rxn,
                                     array states) nogil:
        return self.get_species_activity_sum(rxn, states)

    cdef double evaluate_rxn_rate(self,
                                   unsigned int rxn,
                                   array states) nogil:
        """
        Evaluates and returns rate for specified reaction.

        Args:

            rxn (unsigned int) - index of reaction

            states (array[unsigned int]) - state values

        Returns:

            rate (double) - reaction rate

        """

        cdef double species_activity, rate

        # compute species activities
        species_activity = self.get_species_activity(rxn, states)

        # set rate
        rate = self.k.data.as_doubles[rxn] * species_activity
        if rate < 0:
            rate = 0.

        return rate

    cdef double c_evaluate_rate(self,
                                unsigned int rxn,
                                array states) nogil:
        """
        Evaluates and returns rate of specified reaction.

        Args:

            rxn (unsigned int) - reaction index

            states (array[double]) - state values

        Returns:

            rate (float) - reaction rate

        """

        cdef unsigned int count, ind
        cdef double n, value
        cdef unsigned int index
        cdef unsigned int N = self.n_active_species.data.as_uints[rxn]
        cdef double k = self.k.data.as_doubles[rxn]
        cdef double activity = 1

        # integrate species activity
        index = self.species_ind.data.as_uints[rxn]
        for count in xrange(N):
            ind = self.species.data.as_uints[index]
            value = states.data.as_doubles[ind]
            n = self.species_dependence.data.as_doubles[index]
            activity += (value*n)
            index += 1

        return k*activity


cdef class cIController(cPController):

    @staticmethod
    cdef cIController get_blank_cIController():
        cdef np.ndarray xf = np.zeros(1, dtype=np.float64)
        cdef np.ndarray xl = np.zeros(1, dtype=np.uint32)
        return cIController(0, xf, xl, xl, xf)

    @staticmethod
    cdef cIController from_list(list rxns):
        """ Instantiate from list of reactions. """
        cdef unsigned int M
        cdef np.ndarray k
        cdef np.ndarray species_ind, species, species_dependence

        # if no controllers, return blank
        M = len(rxns)
        if M == 0:
            return cIController.get_blank_cIController()

        # get parameters
        k = np.array([rxn.k[0] for rxn in rxns], dtype=np.float64)

        # get species dependence
        species_ind = np.cumsum([0]+[rxn.num_active_species for rxn in rxns]).astype(np.uint32)
        species = np.hstack([rxn.active_species for rxn in rxns]).astype(np.uint32)
        species_dependence = np.hstack([rxn._propensity for rxn in rxns])
        return cIController(M, k, species_ind, species, species_dependence)

    cdef double get_species_activity_sum(self,
                                         unsigned int rxn,
                                         array cumul) nogil:
        """ Integrate cumulative activity for specified reaction. """
        cdef unsigned int count, state
        cdef double n, k
        cdef unsigned int index = self.species_ind.data.as_uints[rxn]
        cdef unsigned int N = self.n_active_species.data.as_uints[rxn]
        cdef double activity = 0

        # integrate species activity
        for count in xrange(N):
            state = self.species.data.as_uints[index]
            n = cumul.data.as_doubles[state]
            k = self.species_dependence.data.as_doubles[index]
            activity += (n * k)
            index += 1

        return activity


#=============================== PYTHON CODE ==================================


class SumReaction:

    def __init__(self,
                 stoichiometry,
                 propensity,
                 k=1,
                 rxn_type=None,
                 parameters=None):

        self.rxn_type = rxn_type

        # compile stoichiometry as a vector of coefficients
        self.stoichiometry = np.array(stoichiometry, dtype=np.int64)

        # compile propensity as a vector of coefficients
        self.propensity = np.array(propensity, dtype=np.int64)
        self.input_dependence = np.zeros(1, dtype=np.int64)

        # set parameters
        if parameters is None:
            parameters = {}
        self.parameters = parameters
        k_value, k_name = name_parameter(k, 'k')
        if 'k' not in self.parameters.keys():
            self.parameters['k'] = k_name
        self.k = np.array([k_value], dtype=float)

        # define active species
        self.active_species = np.where(self.propensity != 0)[0]
        self.num_active_species = self.active_species.size
        self._propensity = self.propensity[self.active_species]

        # assign reaction rate sensitivities
        self.temperature_sensitive = True
        self.atp_sensitive = False
        self.ribosome_sensitive = False

    def shift(self, shift):
        """

        Expand stoichiometry and propensity vectors.

        Args:

            shift (int) - number of positions appended to beginning

        Returns:

            rxn (SumReaction) - updated reaction

        """
        s = np.hstack((np.zeros(shift, dtype=int), self.stoichiometry))
        p = np.hstack((np.zeros(shift, dtype=int), self.propensity))
        kw = dict(k=self.k[0],
                  rxn_type=self.rxn_type,
                  parameters=self.parameters)
        rxn = SumReaction(s, p, **kw)
        return rxn

    def evaluate_rate(self, states, input_value=0, discrete=True):
        """

        Compute and return current rate of a given pathway.

        Args:

            states (np array) - current state values

            discrete (bool) - if True, use discrete propensity function

        Returns:

            rate (float) - rate of reaction

        """

        # get active states and propensities
        _states = states[self.active_species]

        # get reactant activities
        activities = _states * self._propensity

        # incorporate input dependence
        rate = self.k * reduce(add, activities)

        if rate < 0:
            rate *= 0

        return rate


class ProportionalController(SumReaction):
    pass


class IntegralController(SumReaction):
    pass
