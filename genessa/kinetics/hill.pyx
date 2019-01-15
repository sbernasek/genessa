# cython external imports
cimport numpy as np
from cpython.array cimport array

# python external imports
import numpy as np
from array import array
from functools import reduce
from operator import add, mul
from ..utilities import name_parameter

# cython intra-package imports
from .hill cimport cIDRepressor, cHill
from .base cimport cInputDependent

# python intra-package imports
from .base import Reaction


cdef class cIDRepressor(cInputDependent):

    def __init__(self,
        unsigned int M,
        double[:] k_m,
        double[:] n,
        unsigned int[:] species_ind,
        unsigned int[:] species,
        double[:] species_dependence,
        unsigned int[:] inputs_ind,
        unsigned int[:] inputs,
        double[:] input_dependence):

        # add input/species dependence
        vmax = array('d', np.ones(M, dtype=np.uint32))
        super().__init__(
            M,
            vmax,
            species_ind,
            species,
            species_dependence,
            inputs_ind,
            inputs,
            input_dependence)

        # add rate constants
        self.k_m = array('d', k_m)
        self.n = array('d', n)

    @staticmethod
    cdef cIDRepressor get_blank_cIDRepressor():
        """ Returns blank cIDRepressor instance. """
        cdef np.ndarray xf = np.zeros(1, dtype=np.float64)
        cdef np.ndarray xl = np.zeros(1, dtype=np.uint32)
        return cIDRepressor(0, xf, xf, xl, xl, xf, xl, xl, xf)

    @staticmethod
    cdef cIDRepressor from_list(list rxns):
        """ Instantiate repressor object from list of reactions. """
        cdef unsigned int M
        cdef list reps
        cdef np.ndarray k_m, n
        cdef np.ndarray species_ind, species, species_dependence
        cdef np.ndarray inputs_ind, inputs, input_dependence

        # get repressors
        reps = reduce(add, [rxn.repressors for rxn in rxns])

        # if there are no repressors, return empty arrays
        M = len(reps)
        if M == 0:
            return cIDRepressor.get_blank_cIDRepressor()

        # add parameters
        k_m = np.array([rxn.k_m for rxn in reps], dtype=np.float64)
        n = np.array([rxn.n for rxn in reps], dtype=np.float64)

        # add species dependence
        species_ind =np.cumsum([0]+[rxn.num_active_species for rxn in reps]).astype(np.uint32)
        species = np.hstack([rxn.active_species for rxn in reps]).astype(np.uint32)
        species_dependence = np.hstack([rxn._propensity for rxn in reps])
        inputs_ind = np.cumsum([0]+[rxn.num_active_inputs for rxn in reps]).astype(np.uint32)
        inputs = np.hstack([rxn.active_inputs for rxn in reps]).astype(np.uint32)
        input_dependence = np.hstack([rxn._input_dependence for rxn in reps])

        return cIDRepressor(M, k_m, n, species_ind, species, species_dependence, inputs_ind, inputs, input_dependence)

    cdef double get_species_activity(self,
                                     unsigned int rep,
                                     unsigned int *states) nogil:
        return self.get_species_activity_sum(rep, states)

    cdef double get_input_activity(self,
                                   unsigned int rep,
                                   double *inputs) nogil:
        return self.get_input_activity_sum(rep, inputs)

    cdef double get_occupancy(self,
                              unsigned int rep,
                              unsigned int *states,
                              double *inputs) nogil:
        """
        Evaluates and returns occupancy of specified repressor.

        Args:

            rep (unsigned int) - repressor index

            states (unsigned int*) - state values

            inputs (double*) - input values

        Returns:

            occupancy (double) - repressor occupancy

        """
        cdef double activity = 0
        cdef double k_m, n
        cdef double occupancy

        # compute species activities
        activity += self.get_species_activity(rep, states)
        if self.n_active_inputs.data.as_uints[rep] > 0:
            activity += self.get_input_activity(rep, inputs)

        # compute occupancy
        k_m = self.k_m.data.as_doubles[rep]
        n = self.n.data.as_doubles[rep]
        occupancy = (activity**n) / ( (activity**n) + (k_m**n) )

        return occupancy

    cdef double c_get_occupancy(self,
                               double *states,
                               double *inputs,
                               unsigned int rep) nogil:
        """
        Evaluates and returns occupancy of specified repressor. Accepts states as 'double' type as opposed to the "get_occupancy" method which only accepts states with an 'unsigned int' type. This function serves as an interface for the scipy.integrate ODE solvers used for deterministic simulations.

        Args:

            states (double*) - state values

            inputs (double*) - input values

            rep (unsigned int) - repressor index

        Returns:

            occupancy (double) - repressor occupancy

        """

        cdef unsigned int count, ind
        cdef double coefficient, value
        cdef unsigned int index
        cdef unsigned int N = self.n_active_species.data.as_uints[rep]
        cdef unsigned int I = self.n_active_inputs.data.as_uints[rep]
        cdef double k_m = self.k_m.data.as_doubles[rep]
        cdef double n = self.n.data.as_doubles[rep]
        cdef double activity = 0
        cdef double occupancy

        # integrate species activity
        index = self.species_ind.data.as_uints[rep]
        for count in xrange(N):
            ind = self.species.data.as_uints[index]
            value = states[ind]
            coefficient = self.species_dependence.data.as_doubles[index]
            activity += (value*coefficient)
            index += 1

        # integrate input activity
        index = self.inputs_ind.data.as_uints[rep]
        for count in xrange(I):
            ind = self.inputs.data.as_uints[index]
            value = inputs[ind]
            coefficient = self.input_dependence.data.as_doubles[index]
            activity += (value*coefficient)
            index += 1

        occupancy = (activity**n) / ( (activity**n) + (k_m**n) )

        return occupancy


cdef class cHill(cIDRepressor):


    def __init__(self,
            unsigned int M,
            double[:] vmax,
            double[:] k_m,
            double[:] n,
            unsigned int[:] species_ind,
            unsigned int[:] species,
            double[:] species_dependence,
            unsigned int[:] inputs_ind,
            unsigned int[:] inputs,
            double[:] input_dependence,

            cIDRepressor repressor_obj,
            unsigned int[:] repressors_ind):

        # add input/species dependence
        cInputDependent.__init__(self,
             M,
             vmax,
             species_ind,
             species,
             species_dependence,
             inputs_ind,
             inputs,
             input_dependence)

        self.k_m = array('d', k_m)
        self.n = array('d', n)

        # add repressor data
        self.rep_obj = repressor_obj
        self.repressors_ind = array('I', repressors_ind)
        n_repressors = np.diff(repressors_ind).astype(np.uint32)
        self.n_repressors = array('I', n_repressors)

    @staticmethod
    cdef cHill get_blank_cHill():
        """ Returns blank cHill instance. """
        cdef np.ndarray xf = np.zeros(1, dtype=np.float64)
        cdef np.ndarray xl = np.zeros(1, dtype=np.uint32)
        cdef cIDRepressor rep = cIDRepressor.get_blank_cIDRepressor()
        return cHill(0, xf, xf, xf, xl, xl, xf, xl, xl, xf, rep, xl)

    @staticmethod
    cdef cHill from_list(list rxns):
        """ Instantiate from list of reactions. """
        cdef unsigned int M
        cdef np.ndarray vmax, k_m, n
        cdef np.ndarray species_ind, species, species_dependence
        cdef np.ndarray inputs_ind, inputs, input_dependence
        cdef cIDRepressor repressor_obj
        cdef np.ndarray repressors_ind

        # if no reactions, add blank object
        M = len(rxns)
        if M == 0:
            return cHill.get_blank_cHill()

        # add parameters
        vmax = np.array([rxn.k[0] for rxn in rxns], dtype=np.float64)
        k_m = np.array([rxn.k_m for rxn in rxns], dtype=np.float64)
        n = np.array([rxn.n for rxn in rxns], dtype=np.float64)

        # add species and input dependence
        species_ind = np.cumsum([0]+[rxn.num_active_species for rxn in rxns]).astype(np.uint32)
        species = np.hstack([rxn.active_species for rxn in rxns]).astype(np.uint32)
        species_dependence = np.hstack([rxn._propensity for rxn in rxns])
        species_dependence = species_dependence.astype(np.float64)
        inputs_ind = np.cumsum([0]+[rxn.num_active_inputs for rxn in rxns]).astype(np.uint32)
        inputs = np.hstack([rxn.active_inputs for rxn in rxns]).astype(np.uint32)
        input_dependence = np.hstack([rxn._input_dependence for rxn in rxns])
        input_dependence = input_dependence.astype(np.float64)

        # add repressors
        repressor_obj = cIDRepressor.from_list(rxns)
        repressors_ind = np.cumsum([0]+[len(rxn.repressors) for rxn in rxns]).astype(np.uint32)

        return cHill(M, vmax, k_m, n, species_ind, species, species_dependence, inputs_ind, inputs, input_dependence, repressor_obj, repressors_ind)

    cdef double get_species_activity(self,
                                     unsigned int rxn,
                                     unsigned int *states) nogil:
        return self.get_species_activity_sum(rxn, states)

    cdef double get_input_activity(self,
                                   unsigned int rxn,
                                   double *inputs) nogil:
        return self.get_input_activity_sum(rxn, inputs)

    cdef double get_availability(self,
                                 unsigned int rxn,
                                 unsigned int *states,
                                 double *inputs) nogil:
        """
        Evaluates and returns availability of specified reaction.

        Args:

            rxn (unsigned int) - repressor index

            states (unsigned int*) - state values

            inputs (double*) - input values

        Returns:

            availability (double) - reaction availability

        """

        cdef unsigned int count
        cdef double occupancy
        cdef double availability = 1
        cdef unsigned int index = self.repressors_ind.data.as_uints[rxn]
        cdef unsigned int num_repressors = self.n_repressors.data.as_uints[rxn]

        # integrate repressor occupancies (multiplicative)
        for count in xrange(num_repressors):
            occupancy = self.rep_obj.get_occupancy(index, states, inputs)
            availability *= (1-occupancy)
            index += 1

        return availability

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
        cdef double activity = 0
        cdef double vmax, n, k_m
        cdef double availability
        cdef double rate

        # compute species activities
        activity += self.get_species_activity(rxn, states)
        if self.n_active_inputs.data.as_uints[rxn] > 0:
            activity += self.get_input_activity(rxn, inputs)

        # compute rate
        vmax = self.k.data.as_doubles[rxn]
        k_m = self.k_m.data.as_doubles[rxn]
        n = self.n.data.as_doubles[rxn]
        rate = vmax * (activity**n) / ( (activity**n) + (k_m**n) )

        # apply repressors
        availability = self.get_availability(rxn, states, inputs)
        rate *= availability

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
        cdef double coefficient, value
        cdef unsigned int index
        cdef unsigned int N = self.n_active_species.data.as_uints[rxn]
        cdef unsigned int I = self.n_active_inputs.data.as_uints[rxn]
        cdef unsigned int R = self.n_repressors.data.as_uints[rxn]
        cdef double vmax = self.k.data.as_doubles[rxn]
        cdef double k_m = self.k_m.data.as_doubles[rxn]
        cdef double n = self.n.data.as_doubles[rxn]
        cdef double activity = 0
        cdef double occupancy
        cdef double rate

        # integrate species activity
        index = self.species_ind.data.as_uints[rxn]
        for count in xrange(N):
            ind = self.species.data.as_uints[index]
            value = states[ind]
            coefficient = self.species_dependence.data.as_doubles[index]
            activity += (value*coefficient)
            index += 1

        # integrate input activity
        index = self.inputs_ind.data.as_uints[rxn]
        for count in xrange(I):
            ind = self.inputs.data.as_uints[index]
            value = inputs[ind]
            coefficient = self.input_dependence.data.as_doubles[index]
            activity += (value*coefficient)
            index += 1

        # compute rate
        rate = vmax * ( (activity**n) / ( (activity**n) + (k_m**n) ) )

        # integrate repressor occupancies (multiplicative)
        index = self.repressors_ind.data.as_uints[rxn]
        for count in xrange(R):
            occupancy = self.rep_obj.c_get_occupancy(states, inputs, index)
            rate *= (1-occupancy)
            index += 1

        return rate


#=============================== PYTHON CODE ==================================


class Hill(Reaction):

    def __init__(self,
                 stoichiometry,
                 propensity=None,
                 input_dependence=None,
                 k=1,
                 k_m=1,
                 n=1,
                 baseline=0,
                 repressors=None,
                 rate_modifier=None,
                 temperature_sensitive=False,
                 atp_sensitive=False,
                 ribosome_sensitive=False,
                 carbon_sensitive=False,
                 growth_dependence=0,
                 parameters=None,
                 labels={}):
        """

        Class describes a single hill-kinetic pathway.

        Args:

            stoichiometry (array like) - stoichiometric coefficients

            propensity (array like) - weights for activating substrates

            input_dependence (float or array) - order of input dependence

            k (float) - maximum reaction rate

            k_m (float) - michaelis constant

            n (float) - hill coefficient

            baseline (float) - baseline reaction rate

            repressors (list) - enzymatic repressors

            rate_modifier (array like) - rate constant sensitivity to inputs

            temperature_sensitive (bool) - if True, scale rate with temperature

            atp_sensitive (bool) - if True, scale rate with metabolism

            ribosome_sensitive (bool) - if True, scale rate with ribosomes

            carbon_sensitive (bool) - if True, scale rate with carbon availability

            growth_dependence (int) - log k / log growth

            labels (dict) - additional labels

        """

        # call Reaction instantiation
        super().__init__(stoichiometry,
                         propensity,
                         input_dependence,
                         temperature_sensitive=temperature_sensitive,
                         atp_sensitive=atp_sensitive,
                         ribosome_sensitive=ribosome_sensitive,
                         carbon_sensitive=carbon_sensitive,
                         growth_dependence=growth_dependence,
                         labels=labels)

        # set parameters
        if parameters is None:
            parameters = {}
        self.parameters = parameters

        # set rate constant
        k_value, k_name = name_parameter(k, 'k')
        if 'k' not in self.parameters.keys():
            self.parameters['k'] = k_name
        self.k = np.array([k_value], dtype=float)

        # set michaelis coefficient
        km_value, km_name = name_parameter(k_m, 'k_m')
        if 'k_m' not in self.parameters.keys():
            self.parameters['k_m'] = km_name
        self.k_m = km_value

        # set hill coefficient
        n_value, n_name = name_parameter(n, 'n')
        if 'n' not in self.parameters.keys():
            self.parameters['n'] = n_name
        self.n = n_value

        # set baseline value
        baseline_value, baseline_name = name_parameter(baseline, 'v0')
        if 'v0' not in self.parameters.keys():
            self.parameters['v0'] = baseline_name
        self.baseline = baseline_value

        # add repressors
        if repressors is None:
            repressors = []
        self.repressors = repressors

        # set rate modifier
        if rate_modifier is None:
            rate_modifier = np.zeros(1, dtype=np.int64)
        self.rate_modifier = rate_modifier

    @property
    def num_repressors(self):
        """ Number of repressors. """
        return len(self.repressors)

    @property
    def active_inputs(self):
        """ Indices of active input channels. """
        return np.where(np.logical_or(self.input_dependence != 0, self.rate_modifier != 0))[0]

    def shift(self, shift):
        """

        Expand stoichiometry and propensity vectors.

        Args:

            shift (int) - number of positions appended to beginning

        Returns:

            rxn (Hill) - updated reaction

        """

        s = np.hstack((np.zeros(shift, dtype=int), self.stoichiometry))
        p = np.hstack((np.zeros(shift, dtype=np.float64), self.propensity))
        i = self.input_dependence

        # shift repressors
        repressors = [rep.shift(shift) for rep in self.repressors]

        kw = dict(k=self.k[0],
                  k_m=self.k_m,
                  n=self.n,
                  baseline=self.baseline,
                  repressors=repressors,
                  rate_modifier=self.rate_modifier,
                  temperature_sensitive=self.temperature_sensitive,
                  atp_sensitive=self.atp_sensitive,
                  ribosome_sensitive=self.ribosome_sensitive,
                  carbon_sensitive=self.carbon_sensitive,
                  parameters=self.parameters,
                  labels=self.labels)

        return Hill(s, p, i, **kw)

    def add_promoter(self, species):
        """
        Adds promoter to enzymatic reaction.

        Args:

            species (int) - index of promoter within state space

        """
        self.propensity[species] += 1

    def add_repressor(self, repressor):
        """
        Adds repressor to reaction.

        Args:

            repressor (Repressor) - repressor to be added

        """
        self.repressors.append(repressor)

    def evaluate_rate(self, states, input_state, **kwargs):
        """
        Compute and return current rate of reaction.

        Args:

            states (np array) - current state values

            input_state (np array) - current input value(s)

        Returns:

            rate (float) - rate of reaction

        """

        # get substrate activity
        substrate_activity = (self._propensity * states[self.active_species]).sum() + (self.input_dependence*input_state).sum()

        # get repressor inhibition effects
        unoccupied_sites = 1
        if self.num_repressors > 0:
            unoccupied_sites = reduce(mul, [1-repressor.get_occupancy(states, input_state) for repressor in self.repressors])

        # get overall rate
        k = self.k + (self.rate_modifier * input_state).sum()
        rate = unoccupied_sites * (k * (substrate_activity**self.n)/(substrate_activity**self.n + self.k_m**self.n) + self.baseline)

        return rate


class Repressor(Reaction):

    def __init__(self,
                 propensity=None,
                 input_dependence=None,
                 k_m=1,
                 n=1,
                 parameters=None,
                 labels={}):
        """
        Class defines single instance of competitive enzyme occupancy.

        Args:

            propensity (array like) - weights for activating substrates

            input_dependence (float or np array) - order of input dependence

            k_m (float) - michaelis constant

            n (float) - hill coefficient

            labels (dict) - additional labels

        """

        # define rate law parameters
        if propensity is None:
            propensity = []
        self.propensity = np.array(propensity, dtype=np.float64)

        # define input dependence
        if type(input_dependence) == int:
            input_dependence = [input_dependence]
        elif input_dependence is None:
            input_dependence = [0]
        self.input_dependence = np.array(input_dependence, dtype=np.float64)

        # set parameters
        if parameters is None:
            parameters = {}
        self.parameters = parameters

        # set michaelis constant
        km_value, km_name = name_parameter(k_m, 'k_m')
        if 'k_m' not in self.parameters.keys():
            self.parameters['k_m'] = km_name
        self.k_m = km_value

        # set hill coefficient
        n_value, n_name = name_parameter(n, 'n')
        if 'n' not in self.parameters.keys():
            self.parameters['n'] = n_name
        self.n = n_value

        # assign labels
        self.labels = labels

    def shift(self, shift):
        """

        Expand propensity vector.

        Args:

            shift (int) - number of positions appended to beginning

        Returns:

            rxn (Repressor) - updated reaction

        """
        p = np.hstack((np.zeros(shift, dtype=np.float64), self.propensity))
        i = self.input_dependence
        kw = dict(k_m=self.k_m,
                  n=self.n,
                  parameters=self.parameters,
                  labels=self.labels)
        return Repressor(p, i, **kw)

    def get_occupancy(self, states, input_state):
        """
        Compute and return current occupancy of enzyme by repressive substrate.

        Args:

            states (np array) - current state values

            input_state (vector) - current input value(s)

        Returns:

            occupancy (float) - fraction of enzyme occupied by substrate

        """

        # get substrate activity
        substrate_activity = (self._propensity * states[self.active_species]).sum() + (self.input_dependence * input_state).sum()

        # get overall rate
        occupancy = (substrate_activity**self.n)/(substrate_activity**self.n + self.k_m**self.n)

        return occupancy
