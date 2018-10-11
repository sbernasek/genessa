# cython external imports
cimport numpy as np
from cpython.array cimport array

# python external imports
import numpy as np

# cython intra-package imports
from .massaction cimport cMassAction

# python intra-package imports
from .base import Reaction
from .massaction import MassAction


cdef class cFeedBack(cMassAction):
    """
    Class describes a set of multiple mass action kinetic pathways in which the reaction rate is not necessarily dependent upon the consumed species.
    """
    def __init__(self,
                 int M,
                 double[:] k,
                 unsigned int[:] species_ind,
                 unsigned int[:] species,
                 double[:] species_dependence,
                 unsigned int[:] inputs_ind,
                 unsigned int[:] inputs,
                 double[:] input_dependence,
                 unsigned int[:] targets_ind,
                 unsigned int[:] targets):

        # add species dependence
        super().__init__(M,
                         k,
                         species_ind,
                         species,
                         species_dependence,
                         inputs_ind,
                         inputs,
                         input_dependence)

        # add targets
        self.targets_ind = array('I', targets_ind)
        self.n_targets = array('I', np.diff(targets_ind).astype(np.uint32))
        self.targets = array('I', targets)

    @staticmethod
    cdef cFeedBack get_blank_cFeedBack():
        """ Returns blank cMassAction instance. """
        cdef np.ndarray xf = np.zeros(1, dtype=np.float64)
        cdef np.ndarray xl = np.zeros(1, dtype=np.uint32)
        return cFeedBack(0, xf, xl, xl, xf, xl, xl, xf, xl, xl)

    @staticmethod
    cdef cFeedBack from_list(list rxns):
        """ Instantiate from list of reactions. """

        cdef unsigned int M
        cdef np.ndarray k
        cdef np.ndarray species_ind, species, species_dependence
        cdef np.ndarray inputs_ind, inputs, input_dependence

        # return blank if no reactions of this type
        M = len(rxns)
        if M == 0:
            return cFeedBack.get_blank_cFeedBack()

        # get parameters
        k = np.array([rxn.k[0] for rxn in rxns], dtype=np.float64)

        # get species dependence
        species_ind = np.cumsum([0]+[rxn.num_active_species for rxn in rxns]).astype(np.uint32)
        species = np.hstack([rxn.active_species for rxn in rxns]).astype(np.uint32)
        species_dependence = np.hstack([rxn._propensity for rxn in rxns])
        species_dependence = species_dependence.astype(np.float64)

        # get input dependence
        inputs_ind = np.cumsum([0]+[rxn.num_active_inputs for rxn in rxns]).astype(np.uint32)
        inputs = np.hstack([rxn.active_inputs for rxn in rxns]).astype(np.uint32)
        input_dependence = np.hstack([rxn._input_dependence for rxn in rxns])
        input_dependence = input_dependence.astype(np.float64)

        # get targets
        targets_ind = np.cumsum([0]+[rxn.num_targets for rxn in rxns]).astype(np.uint32)
        targets = np.hstack([(rxn.stoichiometry<0).nonzero()[0] for rxn in rxns]).astype(np.uint32)

        return cFeedBack(M, k, species_ind, species, species_dependence, inputs_ind, inputs, input_dependence, targets_ind, targets)

    cdef double evaluate_rxn_rate(self,
                                   unsigned int rxn,
                                   unsigned int *states,
                                   double *inputs) nogil:
        """
        Evaluates and returns rate for specified reaction using internally stored

        Args:

            rxn (unsigned int) - index of reaction

            states (unsigned int*) - state values

            inputs (double*) - input values

        Returns:

            rate (double) - reaction rate

        """

        cdef unsigned int index, ind, count, value
        cdef unsigned int T = self.n_targets.data.as_uints[rxn]
        cdef double rate = cMassAction.evaluate_rxn_rate(self,
                                                         rxn,
                                                         states,
                                                         inputs)

        # if any of the target levels are zero, set rate to zero
        index = self.targets_ind.data.as_uints[rxn]
        for count in xrange(T):
            ind = self.targets.data.as_uints[index+count]
            value = states[ind]
            if value == 0:
                rate = 0

        return rate

    cdef double c_evaluate_rate(self,
                                unsigned int rxn,
                                double *states,
                                double *inputs) nogil:
        """
        Evaluates and returns rate of specified reaction. Accepts states as 'double' type as opposed to the "evaluate_rxn_rate" method which only accepts states with an 'unsigned int' type. This function serves as an interface for the scipy.integrate ODE solvers used for deterministic simulations.

        Args:

            rxn (unsigned int) - reaction index

            states (double*) - state values

            inputs (double*) - input values

        Returns:

            rate (float) - reaction rate

        """

        cdef unsigned int count, ind
        cdef double n, value
        cdef unsigned int index
        cdef unsigned int N = self.n_active_species.data.as_uints[rxn]
        cdef unsigned int I = self.n_active_inputs.data.as_uints[rxn]
        cdef unsigned int T = self.n_targets.data.as_uints[rxn]
        cdef double k = self.k.data.as_doubles[rxn]
        cdef double activity = 1

        # integrate species activity
        index = self.species_ind.data.as_uints[rxn]
        for count in xrange(N):
            ind = self.species.data.as_uints[index]
            value = states[ind]
            n = self.species_dependence.data.as_doubles[index]
            activity *= (value**n)
            index += 1

        # integrate input activity
        index = self.inputs_ind.data.as_uints[rxn]
        for count in xrange(I):
            ind = self.inputs.data.as_uints[index]
            value = inputs[ind]
            n = self.input_dependence.data.as_doubles[index]
            activity *= (value**n)
            index += 1

        # if any of the target levels are zero, set rate to zero
        index = self.targets_ind.data.as_uints[rxn]
        for count in xrange(T):
            ind = self.targets.data.as_uints[index+count]
            value = states[ind]
            if value == 0:
                activity = 0

        return k*activity


#============================ PYTHON CODE ====================================


class LinearFeedback(MassAction):
    """
    Class describes a single mass action kinetic pathway in which the consumed species (target) does not appear in the propensity function.

    Internally, this object functions the same as MassAction but with an extra step to zero the reaction rate when all targets have been consumed.
    """

    @property
    def targets(self):
        """ Species consumed by reaction. """
        return (self.stoichiometry<0).nonzero()[0]

    @property
    def num_targets(self):
        """ Number of species consumed by reaction. """
        return (self.stoichiometry<0).sum()

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

        rate = super().evaluate_rate(states, input_state, discrete=True)

        # if any targets are fully consumed, zero the rate
        if np.any(states[self.targets]==0):
            rate *= 0

        return rate
