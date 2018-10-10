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


#============================ PYTHON CODE ====================================


class Reaction:
    """
    Base class for all reactions.
    """

    def __init__(self,
                 stoichiometry,
                 propensity=None,
                 input_dependence=None,
                 temperature_sensitive=True,
                 atp_sensitive=False,
                 ribosome_sensitive=False,
                 labels={}):
        """
        Instantiate a single kinetic pathway.

        Args:

            stoichiometry (array like) - stoichiometric coefficients

            propensity (array like) - propensity coefficients

            input_dependence (float or array like) - order of input dependence

            temperature_sensitive (bool) - if True, rate scales with temp

            atp_sensitive (bool) - if True, rate scales with metabolism

            ribosome_sensitive (bool) - if True, rate scales with ribosomes

            labels (dict) - additional labels

        """

        # compile stoichiometry as a vector of coefficients
        self.stoichiometry = np.array(stoichiometry, dtype=np.int64)

         # compile propensity as a vector of coefficients
        if propensity is None:
            propensity = [0 for _ in range(self.N)]
        self.propensity = np.array(propensity, dtype=np.float64)

        # compile input dependence as a vector of coefficients
        if type(input_dependence) == int:
            input_dependence = np.array([input_dependence], dtype=np.float64)
        elif input_dependence is None:
            input_dependence = np.zeros(1, dtype=np.float64)
        else:
            input_dependence = np.array(input_dependence, dtype=np.float64)
        self.input_dependence = input_dependence

        # assign reaction rate sensitivities
        self.temperature_sensitive = temperature_sensitive
        self.atp_sensitive = atp_sensitive
        self.ribosome_sensitive = ribosome_sensitive

        # assign labels
        self.labels = labels

    @property
    def type(self):
        """ Reaction type. """
        return self.__class__.__name__

    @property
    def N(self):
        """ Dimensionality of state space. """
        return self.stoichiometry.size

    @property
    def active_species(self):
        """ Indices of active species. """
        return np.where(self.propensity != 0)[0]

    @property
    def num_active_species(self):
        """ Number of active species. """
        return self.active_species.size

    @property
    def active_inputs(self):
        """ Indices of active input channels. """
        return np.where(self.input_dependence != 0)[0]

    @property
    def num_active_inputs(self):
        """ Number of active input channels. """
        return self.active_inputs.size

    @property
    def _propensity(self):
        """ Propensity coefficients of active species. """
        return self.propensity[self.active_species]

    @property
    def _input_dependence(self):
        """ Input dependence coefficients of active species. """
        return self.input_dependence[self.active_inputs]

    @property
    def zero_order(self):
        """ Flag to skip rate computation for zero-order kinetics. """
        return (self.propensity.sum()==0 and self.input_dependence is None)

    @property
    def name(self):
        """ Reaction name. """
        return self.labels['name']

    @property
    def perturbed(self):
        """ Perturbation flag. """
        is_label = lambda x: x in self.labels.keys()
        return self.labels['perturbed'] if is_label('perturbed') else False
