# cython external imports
cimport numpy as np
from cpython.array cimport array

# python external imports
import numpy as np
from array import array
from functools import reduce
from operator import add

# cython intra-package imports
from .marbach cimport cRegulatoryModule, cTranscription
from .marbach cimport cRxnMap, get_binary_repr_size


cdef class cRegulatoryModule:

    def __init__(self,
                 unsigned int M,
                 unsigned int[:] nA,
                 unsigned int[:] nD,
                 unsigned int[:] bindsAsComplex,

                 double[:] k,
                 double[:] n,

                 unsigned int[:] species_ind,
                 unsigned int[:] species,
                 dict rxn_map):

        # store number of reactions and rate constants
        self.M = M
        self.nA = array('I', nA)
        self.nD = array('I', nD)
        self.bindsAsComplex = array('I', bindsAsComplex)

        # add state dependence
        self.species_ind = array('I', species_ind)
        self.n_active_species = array('I', np.diff(species_ind).astype(np.uint32))
        self.species = array('I', species)

        # store parameters
        self.k = array('d', k)
        self.n = array('d', n)

        # initialize fractional activation
        self.xi = array('d', np.zeros(len(k), dtype=np.float64))

        # initialize rxn_map and rate vector
        self.rxn_map = cRxnMap(rxn_map)
        self.activation = array('d', np.zeros(M, dtype=np.float64))

    @staticmethod
    cdef cRegulatoryModule get_blank_cRegulatoryModule(unsigned int M):
        """ Returns blank cRegulatoryModule instance. """
        cdef unsigned int i
        cdef np.ndarray xf = np.zeros(1, dtype=np.float64)
        cdef np.ndarray xl = np.zeros(1, dtype=np.uint32)
        cdef dict rxn_map = {i: [] for i in xrange(M)}
        return cRegulatoryModule(0, xl, xl, xl, xf, xf, xl, xl, rxn_map)

    @staticmethod
    cdef cRegulatoryModule from_list(list rxns,
                                     dict rxn_map):
        """ Instantiate from list of reactions. """
        cdef unsigned int M
        cdef list modules
        cdef np.ndarray nA, nD, bAC
        cdef np.ndarray k, n
        cdef np.ndarray species_ind, species, species_dependence

        # get repressors (add lists together)
        modules = reduce(add, [rxn.modules for rxn in rxns])

        # if there are no repressors, return empty arrays
        M = len(modules)
        if M == 0:
            return cRegulatoryModule.get_blank_cRegulatoryModule(len(rxn_map))

        # get parameters
        nA = np.array([m.nA for m in modules], dtype=np.uint32)
        nD = np.array([m.nD for m in modules], dtype=np.uint32)
        bAC = np.array([m.bindsAsComplex for m in modules], dtype=np.uint32)
        k = np.hstack([m.k for m in modules]).astype(np.float64)
        n = np.hstack([m.n for m in modules]).astype(np.float64)

        # get species dependence
        species_ind = np.cumsum([0]+[m.num_modifiers for m in modules]).astype(np.uint32)
        species = np.hstack([m.modifiers for m in modules]).astype(np.uint32)

        return cRegulatoryModule(M, nA, nD, bAC, k, n, species_ind, species, rxn_map)

    cdef double set_fractional_activation(self,
                                          unsigned int mod,
                                          unsigned int *states) nogil:
        """ Set fractional activation for modifiers of specified module. """
        cdef unsigned int count, state, modifier
        cdef double k, n
        cdef unsigned int index = self.species_ind.data.as_uints[mod]
        cdef unsigned int N = self.n_active_species.data.as_uints[mod]

        # set fractional activation for each modifier
        for count in xrange(N):
            modifier = self.species.data.as_uints[index]
            state = states[modifier]
            k = self.k.data.as_doubles[index]
            n = self.n.data.as_doubles[index]
            self.xi.data.as_doubles[index] = (state/k)**n
            index += 1

    cdef void set_activation(self,
                             unsigned int mod,
                             unsigned int *states) nogil:
        """
        Set activation for specified regulatory module.

        Args:

            mod (unsigned int) - index of regulatory module

            states (unsigned int*) - state values

        """

        cdef unsigned int i
        cdef unsigned int nA = self.nA.data.as_uints[mod]
        cdef unsigned int nD = self.nD.data.as_uints[mod]
        cdef unsigned int bAC = self.bindsAsComplex.data.as_uints[mod]
        cdef unsigned int index = self.species_ind.data.as_uints[mod]
        cdef double multiplyActivators = 1
        cdef double multiplyInputs = 1
        cdef double numerator = 1
        cdef double denominator = 1

        # update fractional activations
        self.set_fractional_activation(mod, states)

        # get numerator
        for i in xrange(nA):
            multiplyActivators *= self.xi.data.as_doubles[index+i]
        numerator = multiplyActivators

        # get denominator
        if bAC == 1:
            denominator += multiplyActivators
            if nD > 0:
                multiplyInputs = multiplyActivators
                for i in xrange(nD):
                    multiplyInputs *= self.xi.data.as_doubles[index+nA+i]
                denominator += multiplyInputs
        else:
            for i in xrange(nA+nD):
                denominator *= (1+self.xi.data.as_doubles[index+i])

        # set activation
        self.activation.data.as_doubles[mod] = numerator/denominator

    cdef double get_activation(self, unsigned int mod) nogil:
        """
        Returns activation for specified regulatory module.

        Args:

            mod (unsigned int) - index of regulatory module

        Returns:

            activation (double)

        """
        return self.activation.data.as_doubles[mod]

    cdef void update(self,
                     unsigned int *states,
                     unsigned int fired) nogil:
        """
        Update activities for regulatory modules that have changed.

        Args:

            states (unsigned int*) - state values

            fired (unsigned int) - index of reaction that fired

        """
        self.rxn_map.app(self, fired, self.set_activation, states)

    cdef double c_evaluate_activation(self,
                                double* states,
                                unsigned int mod) nogil:
        """
        Evaluates and returns activation of specified regulatory module.

        Args:

            states (double*) - state values

            mod (unsigned int) - index of regulatory module

        Returns:

            activation (double)

        """

        cdef unsigned int i, species
        cdef double k, n, value
        cdef unsigned int nA = self.nA.data.as_uints[mod]
        cdef unsigned int nD = self.nD.data.as_uints[mod]
        cdef unsigned int bAC = self.bindsAsComplex.data.as_uints[mod]
        cdef unsigned int index = self.species_ind.data.as_uints[mod]
        cdef double multiplyActivators = 1
        cdef double multiplyInputs = 1
        cdef double numerator = 1
        cdef double denominator = 1

        # get numerator
        for i in xrange(nA):
            species = self.species.data.as_uints[index+i]
            value = states[species]
            k = self.k.data.as_doubles[index+i]
            n = self.n.data.as_doubles[index+i]
            multiplyActivators *= ( (value/k)**n )
        numerator = multiplyActivators

        # get denominator
        if bAC == 1:
            denominator += multiplyActivators
            if nD > 0:
                multiplyInputs = multiplyActivators
                for i in xrange(nD):
                    species = self.species.data.as_uints[index+nA+i]
                    value = states[species]
                    k = self.k.data.as_doubles[index+i]
                    n = self.n.data.as_doubles[index+i]
                    multiplyInputs *= ( (value/k)**n )
                denominator += multiplyInputs
        else:
            for i in xrange(nA+nD):
                species = self.species.data.as_uints[index+i]
                value = states[species]
                k = self.k.data.as_doubles[index+i]
                n = self.n.data.as_doubles[index+i]
                denominator *= (1 + ((value/k)**n))

        return numerator/denominator


cdef class cTranscription:

    def __init__(self,
                 unsigned int M,

                 # parameters
                 double[:] k,
                 double[:] alpha,
                 unsigned int[:] alpha_ind,

                 # regulatory modules
                 cRegulatoryModule modules_obj,
                 unsigned int[:] modules_ind,

                 # input dependence
                 unsigned int[:] inputs_ind,
                 unsigned int[:] inputs,
                 double[:] input_dependence):

        # store number of reactions and vmax rate constants
        self.M = M
        self.k = array('d', k)

        # store alpha coefficients
        self.alpha_ind = array('I', alpha_ind)
        self.alpha = array('d', alpha)
        self.alpha_wt = array('d', alpha)
        self.num_alpha = array('I', np.diff(alpha_ind).astype(np.uint32))

        # add regulatory modules
        self.modules_obj = modules_obj
        self.modules_ind = array('I', modules_ind)
        self.num_modules = array('I', np.diff(modules_ind).astype(np.uint32))

        # add input dependence
        self.inputs_ind = array('I', inputs_ind)
        self.num_inputs = array('I', np.diff(inputs_ind).astype(np.uint32))
        self.inputs = array('I', inputs)
        self.input_dependence = array('d', input_dependence)

        # initialize rate vector
        self.rates = array('d', np.zeros(M, dtype=np.float64))

    @staticmethod
    cdef cTranscription get_blank_cTranscription(unsigned int M):
        """ Returns blank cTranscription instance. """
        cdef np.ndarray xf = np.zeros(1, dtype=np.float64)
        cdef np.ndarray xl = np.zeros(1, dtype=np.uint32)
        cdef cRegulatoryModule mod = cRegulatoryModule.get_blank_cRegulatoryModule(M)
        return cTranscription(0, xf, xf, xl, mod, xl, xl, xl, xf)

    @staticmethod
    cdef cTranscription from_list(list rxns,
                                  dict rxn_map):
        """ Instantiate from list of reactions. """

        cdef unsigned int M
        cdef np.ndarray k, alpha, alpha_ind
        cdef np.ndarray modules_ind
        cdef cRegulatoryModule modules_obj

        # return blank if no reactions of this type
        M = len(rxns)
        if M == 0:
            return cTranscription.get_blank_cTranscription(len(rxn_map))

        # get parameters
        k = np.array([rxn.k[0] for rxn in rxns], dtype=np.float64)
        alpha = np.hstack([rxn.alpha for rxn in rxns]).astype(np.float64)
        alpha_ind = np.cumsum([0]+[rxn.alpha.size for rxn in rxns]).astype(np.uint32)

        # get regulatory modules
        modules_obj = cRegulatoryModule.from_list(rxns, rxn_map)
        modules_ind = np.cumsum([0]+[rxn.num_modules for rxn in rxns]).astype(np.uint32)

        # get input dependence
        inputs_ind = np.cumsum([0]+[rxn.num_active_inputs for rxn in rxns]).astype(np.uint32)
        inputs = np.hstack([rxn.active_inputs for rxn in rxns]).astype(np.uint32)
        input_dependence = np.hstack([rxn._input_dependence for rxn in rxns])
        input_dependence = input_dependence.astype(np.float64)

        return cTranscription(M, k, alpha, alpha_ind, modules_obj, modules_ind, inputs_ind, inputs, input_dependence)

    cdef void apply_perturbation(self,
                                   unsigned int rxn,
                                   double ptb) nogil:
        """
        Apply perturbation to alpha values for specified reaction.

        Args:

            rxn (unsigned int) - reaction index

            ptb (double) - magnitude of perturbation

        """

        cdef unsigned int alpha_index = self.alpha_ind.data.as_uints[rxn]
        cdef unsigned int num_alpha = self.num_alpha.data.as_uints[rxn]
        cdef unsigned int i
        cdef double a_wt = self.alpha_wt.data.as_doubles[alpha_index]
        cdef double a_ptb

        # limit perturbation to truncate a_0 to [0-1] interval
        if (a_wt + ptb > 1):
            ptb = 1 - a_wt;
        elif (a_wt + ptb) < 0:
            ptb = 0 - a_wt

        # update alpha values
        for i in xrange(num_alpha):

            # increment wildtype alpha value
            a_wt = self.alpha_wt.data.as_doubles[alpha_index+i]
            a_ptb = a_wt + ptb

            # truncate to [0, 1] interval
            if a_ptb < 0:
                a_ptb = 0
            elif a_ptb > 1:
                a_ptb = 1

            # store perturbed alpha value
            self.alpha.data.as_doubles[alpha_index+i] = a_ptb

    cdef void remove_perturbation(self,
                                    unsigned int rxn) nogil:
        """
        Remove perturbation and restore specified reaction to wildtype alpha values.

        Args:

            rxn (unsigned int) - reaction index

        """

        cdef unsigned int alpha_index = self.alpha_ind.data.as_uints[rxn]
        cdef unsigned int num_alpha = self.num_alpha.data.as_uints[rxn]
        cdef unsigned int i
        cdef double alpha

        # copy wildtype values back into alpha array
        for i in xrange(num_alpha):
            alpha = self.alpha_wt.data.as_doubles[alpha_index+i]
            self.alpha.data.as_doubles[alpha_index+i] = alpha

    cdef double get_activation(self, unsigned int rxn) nogil:
        """
        Evaluate and return activation of specified reaction.

        Args:

            rxn (unsigned int) - reaction index

        Returns:

            activation (double)

        """

        cdef unsigned int mod_index = self.modules_ind.data.as_uints[rxn]
        cdef unsigned int num_modules = self.num_modules.data.as_uints[rxn]
        cdef unsigned int alpha_index = self.alpha_ind.data.as_uints[rxn]
        cdef unsigned int num_alpha = self.num_alpha.data.as_uints[rxn]

        cdef unsigned int i, j
        cdef double alpha
        cdef double factivation, p
        cdef double microstate_size
        cdef double activation = 0

        for i in xrange(num_alpha):
            microstate_size = get_binary_repr_size(i)
            p = 1
            for j in xrange(num_modules):
                factivation = self.modules_obj.get_activation(mod_index+j)
                if microstate_size-j-1 >= 0 and ((i >> j) & 1) == 1:
                    p *= factivation
                else:
                    p *= (1-factivation)

            alpha = self.alpha.data.as_doubles[alpha_index+i]
            activation += (alpha * p)

        return activation

    cdef double get_rate_modifier(self,
                                  unsigned int rxn,
                                  double *inputs) nogil:
        """
        Evaluate and return rate modifier by integrating input values.

        Args:

            rxn (unsigned int) - reaction index

            inputs (double*) - input values

        Returns:

            modifier (double)

        """
        cdef unsigned int index = self.inputs_ind.data.as_uints[rxn]
        cdef unsigned int num_inputs = self.num_inputs.data.as_uints[rxn]
        cdef unsigned int i, input_dim
        cdef double dependence, value
        cdef double modifier = 1

        # integrate rate modifiers
        for i in xrange(num_inputs):
            input_dim = self.inputs.data.as_uints[index+i]
            dependence = self.input_dependence.data.as_doubles[index+i]
            input_value = inputs[input_dim]
            modifier += (dependence*input_value)

        return modifier

    cdef double evaluate_rxn_rate(self, unsigned int rxn) nogil:
        """
        Evaluates and returns rate for specified reaction.

        Args:

            rxn (unsigned int) - index of reaction

        Returns:

            rate (double) - reaction rate

        """
        cdef double k = self.k.data.as_doubles[rxn]
        cdef double activation = self.get_activation(rxn)
        return k * activation

    cdef double c_evaluate_rate(self,
                                unsigned int rxn,
                                double* states) nogil:
        """
        Evaluates and returns rate of specified reaction.

        Args:

            rxn (unsigned int) - reaction index

            states (double*) - state values

        Returns:

            rate (float) - reaction rate

        """
        cdef double k = self.k.data.as_doubles[rxn]
        cdef unsigned int mod_index = self.modules_ind.data.as_uints[rxn]
        cdef unsigned int num_modules = self.num_modules.data.as_uints[rxn]
        cdef unsigned int alpha_index = self.alpha_ind.data.as_uints[rxn]
        cdef unsigned int num_alpha = self.num_alpha.data.as_uints[rxn]

        cdef unsigned int i, j
        cdef double alpha
        cdef double factivation, p
        cdef double microstate_size
        cdef double activation = 0

        # integrate activation across regulatory modules
        for i in xrange(num_alpha):
            microstate_size = get_binary_repr_size(i)
            p = 1
            for j in xrange(num_modules):
                factivation = self.modules_obj.c_evaluate_activation(states, mod_index+j)
                if microstate_size-j-1 >= 0 and ((i >> j) & 1) == 1:
                    p *= factivation
                else:
                    p *= (1-factivation)

            alpha = self.alpha.data.as_doubles[alpha_index+i]
            activation += (alpha * p)

        return k * activation


cdef class cRxnMap:

    def __init__(self, adict):
        ind, lengths, values = self.dict_to_array(adict)
        self.ind = array('I', ind)
        self.lengths =  array('I', lengths)
        self.values = array('I', values)

    @staticmethod
    def dict_to_array(adict):
        lengths, values = zip(*[(len(l), l) for l in adict.values()])
        indices = np.insert(np.cumsum(lengths), 0, 0)
        values = np.hstack(values).astype(np.uint32)
        return indices, lengths, values

    cdef void app(self,
                  cRegulatoryModule mod_obj,
                  unsigned int key,
                  cSetActivation f,
                  unsigned int *states) nogil:
        cdef unsigned int count, mod
        cdef unsigned int length = self.lengths.data.as_uints[key]
        cdef unsigned int index = self.ind.data.as_uints[key]

        if length != 0:
            for count in xrange(length):
                mod = self.values.data.as_uints[index]
                f(mod_obj, mod, states)
                index += 1


#=============================== PYTHON CODE ==================================


class RegulatoryModule:

    def __init__(self,
                 modifiers=None,
                 nA=0,
                 nD=0,
                 bindsAsComplex=False,
                 k=1,
                 n=1):

        if modifiers is None:
            modifiers = []

        self.modifiers = np.array(modifiers, dtype=np.uint32)
        self.nA = nA
        self.nD = nD
        self.nI = nA + nD
        self.bindsAsComplex = bindsAsComplex
        self.k = k
        self.n = n

        # predefine active species mask
        self.num_modifiers = self.modifiers.size

    def get_activation(self, x):
        """ x are the levels of active species """

        # fractional activations
        v = (x[self.modifiers]/self.k)**self.n

        # get numerator
        multiplyActivators = 1
        if self.nA > 0:
            multiplyActivators *= np.product(v[:self.nA])
        numerator = multiplyActivators

        # get denominator
        denominator = 1
        if self.bindsAsComplex:
            denominator += multiplyActivators
            if self.nD > 0:
                multiplyAllInputs = multiplyActivators * np.product(v[self.nA: self.nI])
                denominator += multiplyAllInputs
        else:
            denominator *= np.product(1+v)

        return numerator/denominator

    def shift(self, shift):
        """

        Expand list of modifiers.

        Args:

            shift (int) - number of positions appended to beginning

        Returns:

            rxn (RegulatoryModule) - updated reaction

        """

        modifiers = np.hstack((np.zeros(shift, dtype=np.int64), self.modifiers))
        nA = self.nA,
        nD = self.nD,
        bindsAsComplex = self.bindsAsComplex,
        k = self.k,
        n = self.n
        return RegulatoryModule(modifiers, nA, nD, bindsAsComplex, k, n)


class Transcription:

    def __init__(self,
                 stoichiometry=None,
                 modules=None,
                 k=1,
                 alpha=None,
                 perturbed=False,
                 input_dependence=None,
                 rxn_type=None,
                 parameters=None):

        self.rxn_type = rxn_type

        # compile stoichiometry as a vector of coefficients
        if stoichiometry is None:
            stoichiometry = [0]
        self.stoichiometry = np.array(stoichiometry, dtype=np.int64)

        # set reaction parameters
        if parameters is None:
            parameters = {}
        self.parameters = parameters

        # add k and alpha
        self.k = np.array([k], dtype=np.float64)
        self.alpha = np.array(alpha, dtype=np.float64)

        # set perturbation sensitivity flag
        self.perturbed = perturbed

        # add modules
        if modules is None:
            self.modules = []
        else:
            self.modules = modules
        self.num_modules = len(self.modules)

        # compile propensity as a boolean array
        self.propensity = np.zeros(len(stoichiometry), dtype=np.int64)
        for mod in self.modules:
            self.propensity[mod.modifiers] = 1

        # if kinetics are zeroth order, raise flag to skip rate computation
        self.zero_order = False
        if self.num_modules == 0:
            self.zero_order = True

        # set input dependence (currently have no influence)
        if input_dependence is None:
            input_dependence = np.zeros(1, dtype=np.uint32)
        self.input_dependence = input_dependence
        self.active_inputs = np.where(self.input_dependence != 0)[0]
        self.num_active_inputs = self.active_inputs.size
        self._input_dependence = self.input_dependence[self.active_inputs]

        # assign reaction rate sensitivities
        self.temperature_sensitive = True
        self.atp_sensitive = True
        self.ribosome_sensitive = False

    def shift(self, shift):
        """

        Expand stoichiometry vector.

        Args:

            shift (int) - number of positions appended to beginning

        Returns:

            rxn (Transcription) - updated reaction

        """

        s = np.hstack((np.zeros(shift, dtype=int), self.stoichiometry))
        modules = [mod.shift(shift) for mod in self.modules]

        kw = dict(k=self.k[0],
                  alpha=self.alpha,
                  perturbed=self.perturbed,
                  input_dependence=self.input_dependence,
                  rxn_type=self.rxn_type,
                  parameters=self.parameters)

        return Transcription(s, modules, **kw)

    def evaluate_rate(self, x, inputs):

        m = np.array([mod.get_activation(x) for mod in self.modules], dtype=np.float64)

        rate = 0
        for i, alpha in enumerate(self.alpha):
            s = np.binary_repr(i)
            p = 1
            for j in range(self.num_modules):
                if len(s)-j-1 >= 0 and s[len(s)-j-1] == '1':
                    p *= m[j]
                else:
                    p *= (1-m[j])
            rate += (alpha * p)

        #rate *= (self.k + (self._input_dependence*inputs).sum())
        return rate
