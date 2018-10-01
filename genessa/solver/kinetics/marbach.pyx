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
                                          array states) nogil:
        """ Set fractional activation for modifiers of specified module. """
        cdef unsigned int count, state, modifier
        cdef double k, n
        cdef unsigned int index = self.species_ind.data.as_uints[mod]
        cdef unsigned int N = self.n_active_species.data.as_uints[mod]

        # set fractional activation for each modifier
        for count in xrange(N):
            modifier = self.species.data.as_uints[index]
            state = states.data.as_uints[modifier]
            k = self.k.data.as_doubles[index]
            n = self.n.data.as_doubles[index]
            self.xi.data.as_doubles[index] = (state/k)**n
            index += 1

    cdef void set_activation(self,
                             unsigned int mod,
                             array states) nogil:
        """ Set activation for specified regulatory module. """

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

    cdef double get_activation(self,
                               unsigned int mod) nogil:
        """ Get activation for specified regulatory module. """
        return self.activation.data.as_doubles[mod]

    cdef void update(self,
                     array states,
                     unsigned int fired) nogil:
        """ Update occupancies for repressors that have changed. """
        self.rxn_map.app(self, fired, self.set_activation, states)

    cdef double cget_activation(self,
                                array states,
                                unsigned int mod) nogil:
        """ Get activation of specified regulatory module """

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
            value = states.data.as_doubles[species]
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
                    value = states.data.as_doubles[species]
                    k = self.k.data.as_doubles[index+i]
                    n = self.n.data.as_doubles[index+i]
                    multiplyInputs *= ( (value/k)**n )
                denominator += multiplyInputs
        else:
            for i in xrange(nA+nD):
                species = self.species.data.as_uints[index+i]
                value = states.data.as_doubles[species]
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

    cdef double apply_perturbation(self,
                                   unsigned int rxn,
                                   double ptb) nogil:
        """ Apply perturbation to alpha values for specified reaction. """

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

    cdef double remove_perturbation(self,
                                    unsigned int rxn) nogil:
        """ Restore specified reaction to wildtype alpha values. """

        cdef unsigned int alpha_index = self.alpha_ind.data.as_uints[rxn]
        cdef unsigned int num_alpha = self.num_alpha.data.as_uints[rxn]
        cdef unsigned int i
        cdef double alpha

        # copy wildtype values back into alpha array
        for i in xrange(num_alpha):
            alpha = self.alpha_wt.data.as_doubles[alpha_index+i]
            self.alpha.data.as_doubles[alpha_index+i] = alpha

    cdef double get_activation(self,
                               unsigned int rxn,
                               array states) nogil:
        """ Determine gene activation for specified states. """

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
                                  array input_values) nogil:
        """ Integrate input_values with input_dependence. """
        cdef unsigned int index = self.inputs_ind.data.as_uints[rxn]
        cdef unsigned int num_inputs = self.num_inputs.data.as_uints[rxn]
        cdef unsigned int i, input_dim
        cdef double dependence, value
        cdef double modifier = 1

        # integrate rate modifiers
        for i in xrange(num_inputs):
            input_dim = self.inputs.data.as_uints[index+i]
            dependence = self.input_dependence.data.as_doubles[index+i]
            input_value = input_values.data.as_doubles[input_dim]
            modifier += (dependence*input_value)

        return modifier

    cdef double update(self,
                       unsigned int rxn,
                       array states,
                       array input_values) nogil:
        """ Update rate of specified reaction. """
        cdef double k = self.k.data.as_doubles[rxn]
        cdef double activation = self.get_activation(rxn, states)
        return k * activation

    cdef double cget_rate(self,
                          unsigned int rxn,
                          array states,
                          array input_values) nogil:
        """ Get rate of specified reaction """
        cdef double k = self.k.data.as_doubles[rxn]
        cdef double activation = self.get_activation(rxn, states)

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
                  array states) nogil:
        cdef unsigned int count, mod
        cdef unsigned int length = self.lengths.data.as_uints[key]
        cdef unsigned int index = self.ind.data.as_uints[key]

        if length != 0:
            for count in xrange(length):
                mod = self.values.data.as_uints[index]
                f(mod_obj, mod, states)
                index += 1
