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

# python intra-package imports
from .base import Reaction


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

    cdef double evaluate_rxn_rate(self,
                                   unsigned int rxn,
                                   unsigned int *controlled) nogil:
        """
        Evaluates and returns rate for specified reaction.

        Args:

            rxn (unsigned int) - index of reaction

            controlled (unsigned int*) - controlled variable values

        Returns:

            rate (double) - reaction rate

        """

        cdef double activity, rate

        # compute species activities
        activity = self.get_species_activity_sum(rxn, controlled)

        # set rate
        rate = self.k.data.as_doubles[rxn] * activity
        if rate < 0:
            rate = 0.

        return rate

    cdef double c_evaluate_rate(self,
                                unsigned int rxn,
                                double *controlled) nogil:
        """
        Evaluates and returns rate of specified reaction.

        Args:

            rxn (unsigned int) - reaction index

            controlled (double*) - controlled variable values

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
            value = controlled[ind]
            n = self.species_dependence.data.as_doubles[index]
            activity += (value*n)
            index += 1

        return k*activity


cdef class cIController(cSpeciesDependent):

    @staticmethod
    cdef cIController get_blank_cIController():
        """ Returns blank cIController instance. """
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

    cdef double evaluate_integrator_sum(self,
                                     unsigned int rxn,
                                     double *controlled) nogil:
        """
        Evaluates and returns weighted sum of controlled variables.

        Args:

            rxn (unsigned int) - index of reaction

            controlled (double*) - controlled variable values

        Returns:

            activity (double) - weighted activity

        """
        cdef double n
        cdef unsigned int count, state
        cdef double k
        cdef unsigned int index = self.species_ind.data.as_uints[rxn]
        cdef unsigned int N = self.n_active_species.data.as_uints[rxn]
        cdef double activity = 0

        # integrate species activity
        for count in xrange(N):
            state = self.species.data.as_uints[index]
            n = controlled[state]
            k = self.species_dependence.data.as_doubles[index]
            activity += (n * k)
            index += 1

        return activity

    cdef double evaluate_rxn_rate(self,
                                   unsigned int rxn,
                                   double *controlled) nogil:
        """
        Evaluates and returns rate for specified reaction.

        Args:

            rxn (unsigned int) - index of reaction

            controlled (double*) - controlled variable values

        Returns:

            rate (double) - reaction rate

        """

        cdef double activity, rate

        # compute species activities
        activity = self.evaluate_integrator_sum(rxn, controlled)

        # set rate
        rate = self.k.data.as_doubles[rxn] * activity
        if rate < 0:
            rate = 0.

        return rate

    cdef double c_evaluate_rate(self,
                                unsigned int rxn,
                                double *controlled) nogil:
        """
        Evaluates and returns rate of specified reaction.

        Args:

            rxn (unsigned int) - reaction index

            controlled (double*) - controlled variable values

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
            value = controlled[ind]
            n = self.species_dependence.data.as_doubles[index]
            activity += (value*n)
            index += 1

        return k*activity



#=============================== PYTHON CODE ==================================


class SumReaction(Reaction):

    def __init__(self,
                 stoichiometry,
                 propensity,
                 k=1,
                 temperature_sensitive=True,
                 atp_sensitive=False,
                 ribosome_sensitive=False,
                 parameters=None,
                 labels={}):

        # call Reaction instantiation
        super().__init__(stoichiometry,
                         propensity,
                         temperature_sensitive=temperature_sensitive,
                         atp_sensitive=atp_sensitive,
                         ribosome_sensitive=ribosome_sensitive,
                         labels=labels)

        # set parameters
        if parameters is None:
            parameters = {}
        self.parameters = parameters
        k_value, k_name = name_parameter(k, 'k')
        if 'k' not in self.parameters.keys():
            self.parameters['k'] = k_name
        self.k = np.array([k_value], dtype=float)

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
                  temperature_sensitive=self.temperature_sensitive,
                  atp_sensitive=self.atp_sensitive,
                  ribosome_sensitive=self.ribosome_sensitive,
                  parameters=self.parameters,
                  labels=self.labels)
        rxn = self.__class__(s, p, **kw)
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
