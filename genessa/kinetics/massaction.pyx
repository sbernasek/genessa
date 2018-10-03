# cython external imports
cimport numpy as np
from cpython.array cimport array

# python external imports
from copy import copy
import numpy as np
from array import array
from scipy.misc import comb
from functools import reduce
from operator import mul, add
from ..utilities import name_parameter

# cython intra-package imports
from .base cimport cInputDependent
from .massaction cimport cMassAction


cdef class cMassAction(cInputDependent):
    """
    Class describes a set of multiple mass action kinetic pathways.
    """

    @staticmethod
    cdef cMassAction get_blank_cMassAction():
        """ Returns blank cMassAction instance. """
        cdef np.ndarray xf = np.zeros(1, dtype=np.float64)
        cdef np.ndarray xl = np.zeros(1, dtype=np.uint32)
        return cMassAction(0, xf, xl, xl, xf, xl, xl, xf)

    @staticmethod
    cdef cMassAction from_list(list rxns):
        """ Instantiate from list of reactions. """

        cdef unsigned int M
        cdef np.ndarray k
        cdef np.ndarray species_ind, species, species_dependence
        cdef np.ndarray inputs_ind, inputs, input_dependence

        # return blank if no reactions of this type
        M = len(rxns)
        if M == 0:
            return cMassAction.get_blank_cMassAction()

        # get parameters
        k = np.array([rxn.k[0] for rxn in rxns], dtype=np.float64)

        # get species dependence
        species_ind = np.cumsum([0]+[rxn.num_active_species for rxn in rxns]).astype(np.uint32)
        species = np.hstack([rxn.active_species for rxn in rxns]).astype(np.uint32)
        species_dependence = np.hstack([rxn._propensity for rxn in rxns])
        species_dependence = species_dependence.astype(np.float64)
        inputs_ind = np.cumsum([0]+[rxn.num_active_inputs for rxn in rxns]).astype(np.uint32)
        inputs = np.hstack([rxn.active_inputs for rxn in rxns]).astype(np.uint32)
        input_dependence = np.hstack([rxn._input_dependence for rxn in rxns])
        input_dependence = input_dependence.astype(np.float64)
        return cMassAction(M, k, species_ind, species, species_dependence, inputs_ind, inputs, input_dependence)

    cdef double evaluate_rxn_rate(self,
                                   unsigned int rxn,
                                   unsigned int *states,
                                   double *inputs) nogil:
        """
        Evaluates and returns rate for specified reaction.

        Args:

            rxn (unsigned int) - index of reaction

            states (unsigned int*) - state values

            inputs (double*) - input values

        Returns:

            rate (double) - reaction rate

        """
        cdef double species_activity, input_activity
        cdef double rate = self.k.data.as_doubles[rxn]

        # compute species activities
        rate *= self.get_species_activity(rxn, states)
        if self.n_active_inputs.data.as_uints[rxn] > 0:
            rate *= self.get_input_activity(rxn, inputs)

        return rate

    cdef double c_evaluate_rate(self,
                                unsigned int rxn,
                                array states,
                                array inputs) nogil:
        """
        Evaluates and returns rate of specified reaction.

        Args:

            rxn (unsigned int) - reaction index

            states (array[double]) - state values

            inputs (array[double]) - input values

        Returns:

            rate (float) - reaction rate

        """

        cdef unsigned int count, ind
        cdef double n, value
        cdef unsigned int index
        cdef unsigned int N = self.n_active_species.data.as_uints[rxn]
        cdef unsigned int I = self.n_active_inputs.data.as_uints[rxn]
        cdef double k = self.k.data.as_doubles[rxn]
        cdef double activity = 1

        # integrate species activity
        index = self.species_ind.data.as_uints[rxn]
        for count in xrange(N):
            ind = self.species.data.as_uints[index]
            value = states.data.as_doubles[ind]
            n = self.species_dependence.data.as_doubles[index]
            activity *= (value**n)
            index += 1

        # integrate input activity
        index = self.inputs_ind.data.as_uints[rxn]
        for count in xrange(I):
            ind = self.inputs.data.as_uints[index]
            value = inputs.data.as_doubles[ind]
            n = self.input_dependence.data.as_doubles[index]
            activity *= (value**n)
            index += 1

        return k*activity


#============================ PYTHON CODE ====================================

class MassAction:
    """
    Class describes a single mass action kinetic pathway.
    """

    def __init__(self,
                 stoichiometry=None,
                 propensity=None,
                 input_dependence=None,
                 k=1,
                 rxn_type=None,
                 temperature_sensitive=True,
                 atp_sensitive=False,
                 ribosome_sensitive=False,
                 parameters=None):
        """
        Class describes a single kinetic pathway.

        Args:

            stoichiometry (array like) - stoichiometric coefficients

            propensity (array like) - propensity coefficients

            input_dependence (float or array like) - order of input dependence

            k (float) - mass-action reaction rate constant

            rxn_type (str) - type of reaction

            temperature_sensitive (bool) - if True, rate scales with temp

            atp_sensitive (bool) - if True, rate scales with metabolism

            ribosome_sensitive (bool) - if True, rate scales with ribosomes

        """

        self.rxn_type = rxn_type

        # compile stoichiometry as a vector of coefficients
        if stoichiometry is None:
            stoichiometry = [0]
        self.stoichiometry = np.array(stoichiometry, dtype=np.int64)

        # compile propensity as a vector of coefficients
        if propensity is None:
            self.propensity = np.array([-s if s < 0 else 0 for s in self.stoichiometry], dtype=np.int64)
        else:
            self.propensity = np.array(propensity, dtype=np.int64)

        # convert input dependence to array
        if type(input_dependence) == int:
            input_dependence = [input_dependence]
        if input_dependence is not None:
            self.input_dependence = np.array(input_dependence, dtype=np.int64)
        else:
            self.input_dependence = np.zeros(1, dtype=np.int64)

        # set reaction parameters
        if parameters is None:
            parameters = {}
        self.parameters = parameters
        k_value, k_name = name_parameter(k, 'k')
        if 'k' not in self.parameters.keys():
            self.parameters['k'] = k_name
        self.k = np.array([k_value], dtype=float)

        # define active species and inputs
        self.active_species = np.where(self.propensity != 0)[0]
        self.active_inputs = np.where(self.input_dependence != 0)[0]
        self.num_active_species = self.active_species.size
        self.num_active_inputs = self.active_inputs.size

        # predefine active species mask
        self._propensity = self.propensity[self.active_species]
        self._input_dependence = self.input_dependence[self.active_inputs]

        # if kinetics are zeroth order, raise flag to skip rate computation
        self.zero_order = False
        if self.propensity.sum() == 0 and self.input_dependence is None:
            self.zero_order = True

        # assign reaction rate sensitivities
        self.temperature_sensitive = temperature_sensitive
        self.atp_sensitive = atp_sensitive
        self.ribosome_sensitive = ribosome_sensitive

    def shift(self, shift):
        """

        Expand stoichiometry and propensity vectors.

        Args:

            shift (int) - number of positions appended to beginning

        Returns:

            rxn (MassAction) - updated reaction

        """

        s = np.hstack((np.zeros(shift, dtype=int), self.stoichiometry))
        p = np.hstack((np.zeros(shift, dtype=int), self.propensity))
        i = self.input_dependence

        kw = dict(k=self.k[0],
                  rxn_type=self.rxn_type,
                  parameters=self.parameters)

        return MassAction(s, p, i, **kw)

    @staticmethod
    def _get_activities(states, propensities):
        """ Evaluate activities of active species. """
        return [comb(s, p, exact=True) if p > 1 else s**p for s, p in zip(states, propensities)]

    def evaluate_rate(self, states, input_state, discrete=True):
        """

        Returns rate for given state and input values.

        Args:

            states (np array) - current state values

            input_state (np array) - current input value(s)

            discrete (bool) - if True, use discrete propensity function

        Returns:

            rate (float) - rate of reaction

        """

        # get active states and propensities
        _states = states[self.active_species]

        # get reactant activities
        if discrete:
            activities = self._get_activities(_states, self._propensity)
        else:
            activities = _states**self._propensity

        # incorporate input dependence
        if self.num_active_species == 0:
            rate = self.k * reduce(mul, input_state**self.input_dependence)
        else:
            rate = self.k * reduce(mul, activities) * reduce(mul, input_state**self.input_dependence)

        return rate

    def get_propensity_coefficients(self, external_input=False):
        """
        Need to generalize and fix this POS.
        """

        if external_input is False:
            propensity = self.propensity
        else:
            propensity = np.hstack((np.array(self.input_dependence), self.propensity))

        i = [propensity]
        if np.max(propensity) < 2:
            a = [self.k]
        elif np.max(propensity) == 2 and len(np.where(propensity == 2)) == 1:
            a = copy(propensity)
            a[np.argmax(a)] -= 1
            i.append(a)
            a = [self.k/2, -self.k/2]
        else:
            print('Propensity function is too complex.')
        return i, a


class LinearReaction(MassAction):
    """
    Simplified version of MassAction class for linear rate laws.
    """
    def __init__(self, **kwargs):
        MassAction.__init__(self, **kwargs)
        if self._propensity.sum() != 1 or self._input_dependence.sum() != 0:
            print('prop', self._propensity)
            print('input dep', self._input_dependence)
            print(self.__class__)
            raise ValueError('Reaction is not linear.')

    def evaluate_rate(self, states, input_state, **kwargs):
        return self.k * states[self.active_species]


class LinearInput(MassAction):
    """
    Simplified version of MassAction class for linear input signal.
    """
    def __init__(self, **kwargs):
        MassAction.__init__(self, **kwargs)
        if self._propensity.sum() != 0 or self._input_dependence.sum() != 1:
            raise ValueError('Reaction does not have linear inputs.')

    def evaluate_rate(self, states, input_state, **kwargs):
        return self.k * input_state


class SecondOrderReaction(MassAction):
    """
    Simplified version of MassAction class for second order rate laws.
    """

    # WARNING: does not support multiple input channels (easy fix...)
    def __init__(self, **kwargs):
        MassAction.__init__(self, **kwargs)
        if (self._propensity.sum() + self._input_dependence.sum()) != 2:
            raise ValueError('Reaction is not second order.')

        if 2 in self._propensity or 2 in self._input_dependence:
            raise ValueError('Reaction is has a quadratic dependency.')

    def evaluate_rate(self, states, input_state, **kwargs):
        """ Compute and return current rate of a second order pathway. """
        if self.num_active_species > 0:
            rate = self.k * reduce(mul, states[self.active_species])
        else:
            rate = self.k
        if self.active_inputs.size > 0:
            rate *= input_state

        return rate
