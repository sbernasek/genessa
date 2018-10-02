# cython: profile=False

"""
TO DO:
- convert .data.as_doubles to raw pointers
"""

# cython external imports
cimport cython
cimport numpy as np
from cpython.array cimport array

# python external imports
import numpy as np
from array import array
from functools import reduce
from operator import add

# cython intra-package imports
from ..kinetics.massaction cimport cMassAction
from ..kinetics.control cimport cPController, cIController
from ..kinetics.hill cimport cHill
from ..kinetics.marbach cimport cTranscription
from ..kinetics.coupling cimport cCoupling
from .rates cimport cRates, cRxnMap
from .cython_sum cimport sum_double_arr

# ============================= CYTHON CODE ===================================


cdef class cRates:

    def __init__(self,
                 cCoupling coupling,
                 cMassAction massaction,
                 cTranscription transcription,
                 cHill hill,
                 cIController icontrol,
                 cPController pcontrol,
                 unsigned int[:] rxn_types,
                 unsigned int[:] rxn_keys,
                 dict rxn_map,
                 dict input_map,
                 dict ptb_map):

        # store rate objects
        self.coupling = coupling
        self.massaction = massaction
        self.transcription = transcription
        self.hill = hill
        self.icontrol = icontrol
        self.pcontrol = pcontrol

        # store reaction types and dependencies
        self.M = len(rxn_types)
        self.rxn_types = array('I', rxn_types)
        self.rxn_keys = array('I', rxn_keys)
        self.rxn_map = cRxnMap(rxn_map)
        self.input_map = cRxnMap(input_map)
        self.ptb_map = cRxnMap(ptb_map)

        # initialize rates array
        self.rates = array('d', np.zeros(self.M, dtype=np.float64))
        self.total_rate = sum_double_arr(self.rates, self.M)

    cdef double evaluate(self,
                         unsigned int rxn,
                         array states,
                         array inputs,
                         array cumul) nogil:
        """ Update rate of individual reaction. """

        # get reaction rate
        if self.rxn_types.data.as_uints[rxn] == 0:
            return self.coupling.update(self.rxn_keys.data.as_uints[rxn], states)

        elif self.rxn_types.data.as_uints[rxn] == 1:
            return self.massaction.update(self.rxn_keys.data.as_uints[rxn], states, inputs)

        elif self.rxn_types.data.as_uints[rxn] == 2:
            return self.transcription.update(self.rxn_keys.data.as_uints[rxn], states, inputs)

        elif self.rxn_types.data.as_uints[rxn] == 3:
            return self.hill.update(self.rxn_keys.data.as_uints[rxn], states, inputs)

        elif self.rxn_types.data.as_uints[rxn] == 4:
            return self.icontrol.update(self.rxn_keys.data.as_uints[rxn], cumul)

        elif self.rxn_types.data.as_uints[rxn] == 5:
            return self.pcontrol.update(self.rxn_keys.data.as_uints[rxn], states)

        else:
            return 0.0

    cdef void set_rate(self,
                       unsigned int rxn,
                       array states,
                       array inputs,
                       array cumul) nogil:
        """ Set rate for individual reaction. """
        cdef double old_rate, rate

        # update rxn rate
        rate = self.evaluate(rxn, states, inputs, cumul)
        old_rate = self.rates.data.as_doubles[rxn]

        # update total rate
        self.rates.data.as_doubles[rxn] = rate
        self.total_rate += (rate - old_rate)

    cdef void apply_perturbation(self,
                                 unsigned int rxn,
                                 double ptb) nogil:
        """ Apply perturbation to individual reaction. """

        cdef unsigned int rxn_type = self.rxn_types.data.as_uints[rxn]
        cdef unsigned int rxn_key = self.rxn_keys.data.as_uints[rxn]

        # perturbations have only been implemented for other reaction types
        if rxn_type == 2:
            self.transcription.remove_perturbation(rxn_key)
            self.transcription.apply_perturbation(rxn_key, ptb)
        else:
            pass

    cdef void update_input(self,
                           array states,
                           array inputs,
                           array cumul,
                           unsigned int dim) nogil:
        """ Update rates for input-dependent reactions that have changed. """

        #  TEMPORARY *** apply first dimension of input as perturbation
        cdef double ptb = inputs.data.as_doubles[0]
        self.ptb_map.app_ptb(self, dim, self.apply_perturbation, ptb)

        # update input dependent reactions
        self.input_map.app(self, dim, self.set_rate, states, inputs, cumul)

    cdef void update(self,
                     array states,
                     array inputs,
                     array cumul,
                     unsigned int fired) nogil:
        """ Update rates for species-dependent reactions that have changed. """
        self.coupling.update_activities(states, fired)
        self.coupling.rep_obj.update(states, fired)
        self.transcription.modules_obj.update(states, fired)
        self.rxn_map.app(self, fired, self.set_rate, states, inputs, cumul)

    cdef void update_all(self,
                         array states,
                         array inputs,
                         array cumul) nogil:
        """ Update all reaction rates. """
        cdef unsigned int rxn
        cdef unsigned int input_dim

        # apply time-zero perturbations
        cdef double ptb = inputs.data.as_doubles[0]
        self.ptb_map.app_ptb(self, 0, self.apply_perturbation, ptb)

        # update all reaction rates
        for rxn in xrange(self.M):
            self.coupling.update_activities(states, rxn)
            self.coupling.rep_obj.update(states, rxn)
            self.transcription.modules_obj.update(states, rxn)
            self.set_rate(rxn, states, inputs, cumul)

    cpdef void cupdate(self,
                       array states,
                       array inputs,
                       array cumul) with gil:
        """ Update all reaction rates using continuous rate functions. """

        cdef unsigned int rxn
        cdef double rate
        cdef unsigned int rxn_type, rxn_key

        # iterate across reactions
        for rxn in xrange(self.M):

            # get reaction type
            rxn_type = self.rxn_types.data.as_uints[rxn]
            rxn_key = self.rxn_keys.data.as_uints[rxn]

            # get reaction rate
            if rxn_type == 0:
                rate = self.coupling.cget_rate(rxn_key, states)
            elif rxn_type == 1:
                rate = self.massaction.cget_rate(rxn_key, states, inputs)
            elif rxn_type == 2:
                rate = self.transcription.cget_rate(rxn_key, states, inputs)
            elif rxn_type == 3:
                rate = self.hill.cget_rate(rxn_key, states, inputs)
            elif rxn_type == 4:
                rate = self.icontrol.cget_rate(rxn_key, cumul)
            elif rxn_type == 5:
                rate = self.pcontrol.cget_rate(rxn_key, states)
            else:
                pass

            self.rates.data.as_doubles[rxn] = rate

    cpdef array evaluate_rxn_rates(self,
                                   array states,
                                   array inputs,
                                   array cumul):
        """
        Update rates and return current rate vector.

        Args:

            states (array[unsigned int]) - state values

            input_value (array[double]) - input values

            cumul (array[double]) - integrator values

        Returns:

            rates (array[double]) - reaction rates

        """
        self.cupdate(states, inputs, cumul)
        return self.rates


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
                  cRates rf,
                  unsigned int key,
                  cSetRate f,
                  array states,
                  array inputs,
                  array cumul) nogil:
        cdef unsigned int count, rxn
        cdef unsigned int length = self.lengths.data.as_uints[key]
        cdef unsigned int index = self.ind.data.as_uints[key]

        for count in xrange(length):
            rxn = self.values.data.as_uints[index]
            f(rf, rxn, states, inputs, cumul)
            index += 1

    cdef void app_ptb(self,
                      cRates rf,
                      unsigned int key,
                      cPerturb f,
                      double ptb) nogil:
        cdef unsigned int count, rxn
        cdef unsigned int length = self.lengths.data.as_uints[key]
        cdef unsigned int index = self.ind.data.as_uints[key]

        for count in xrange(length):
            rxn = self.values.data.as_uints[index]
            f(rf, rxn, ptb)
            index += 1

#============================== PYTHON CODE ===================================



class Rates:
    """
    Rate function.

    Attributes:

        N (int) - network dimensionality

        M (int) - number of reactions

        reactions (list) - list of python-based reaction objects

        cRates (cRates) - cython rate function

    """

    def __init__(self, network):
        """

        Args:

            network (Network)

        """
        self.N = network.nodes.size
        self.M = len(network.reactions)
        self.reactions = network.reactions

        # compile cRates
        self.cRates = self.compile_c_rate_function(network)


    def __call__(self, states, input_value, cumul):
        """
        Evaluate reaction rates.

        Args:

            states (array[unsigned int]) - state values

            input_value (array[double]) - input values

            cumul (array[double]) - integrator values

        Returns:

            rxn_rates (array[double]) - reaction rates

        """
        return self.c_evaluate_rxn_rates(states, input_value, cumul)

    def c_evaluate_rxn_rates(self, states, input_state, cumul):
        """
        Evaluate reaction rates using cRates object.

        Args:

            states (array[unsigned int]) - state values

            input_value (array[double]) - input values

            cumul (array[double]) - integrator values

        Returns:

            rxn_rates (array[double]) - reaction rates

        """
        return self.cRates.evaluate_rxn_rates(states, input_state, cumul)

    def c_evaluate_species_rates(self, states, input_state, cumul):
        """
        Evaluate species rates using cRates object.

        Args:

            states (array[unsigned int]) - state values

            input_value (array[double]) - input values

            cumul (array[double]) - integrator values

        Returns:

            species_rates (array[double]) - species rates

        """
        rates = np.zeros(self.N, dtype=np.float64)
        rxn_rates = self.cRates.evaluate_rxn_rates(states, input_state, cumul)
        for i, rxn in enumerate(self.reactions):
            rates += rxn_rates[i] * rxn.stoichiometry
        return rates

    def py_evaluate_rxn_rates(self, states, input_state):
        """
        Evaluate reaction rates using python rate functions.

        Args:

            states (array[unsigned int]) - state values

            input_value (array[double]) - input values

        Returns:

            rxn_rates (array[double]) - reaction rates

        """
        rates = np.zeros(self.M, dtype=np.float64)
        for i, rxn in enumerate(self.reactions):
            rates[i] = rxn.evaluate_rate(states, input_state)
        return rates

    def py_evaluate_species_rates(self, states, input_state):
        """
        Evaluate species rates using python rate functions.

        Args:

            states (array[unsigned int]) - state values

            input_value (array[double]) - input values

        Returns:

            species_rates (array[double]) - species rates

        """
        rates = np.zeros(self.N, dtype=np.float64)
        for rxn in self.reactions:
            rates += rxn.evaluate_rate(states, input_state) * rxn.stoichiometry
        return rates

    @classmethod
    def compile_c_rate_function(cls, network):
        """
        Compile cRates instance for given network.

        Args:

            network (Network)

        Returns:

            cRates (cRates)

        """

        coupling = []
        massaction = []
        transcription = []
        hill = []
        icontrol = []
        pcontrol = []

        rxn_types = []

        for rxn in network.reactions:

            if rxn.__class__.__name__ == 'Coupling':
                rxn_types.append(0)
                coupling.append(rxn)

            elif rxn.__class__.__name__ == 'Reaction':
                rxn_types.append(1)
                massaction.append(rxn)

            elif rxn.__class__.__name__ == 'Transcription':
                rxn_types.append(2)
                transcription.append(rxn)

            elif rxn.__class__.__name__ == 'Hill':
                rxn_types.append(3)
                hill.append(rxn)

            elif rxn.__class__.__name__ == 'IntegralController':
                rxn_types.append(4)
                icontrol.append(rxn)

            elif rxn.__class__.__name__ == 'ProportionalController':
                rxn_types.append(5)
                pcontrol.append(rxn)

            else:
                raise ValueError('{} reaction type not recognized.'.format(rxn.__class__.__name__))

        # get edge map
        edge_map = cls.get_rxn_map(network, maptype='edges')

        # get repressor map
        repressor_map = cls.get_rxn_map(network, maptype='repressors')

        # get modules map
        modules_map = cls.get_rxn_map(network, maptype='modules')

        # get rate objects
        coupling = cCoupling.from_list(coupling, edge_map, repressor_map)
        massaction = cMassAction.from_list(massaction)
        transcription = cTranscription.from_list(transcription, modules_map)
        hill = cHill.from_list(hill)
        icontrol = cIController.from_list(icontrol)
        pcontrol = cPController.from_list(pcontrol)

        # get reaction map
        rxn_map = cls.get_rxn_map(network, maptype='propensity')
        input_map = cls.get_input_map(network)
        perturbation_map = cls.get_perturbation_map(network)

        # set reaction lists
        rxn_types = np.array(rxn_types, dtype=np.uint32)
        rxn_keys = cls.get_rxn_keys(rxn_types)

        # instantiate cReactions object
        return cRates(coupling,
                     massaction,
                     transcription,
                     hill,
                     icontrol,
                     pcontrol,
                     rxn_types,
                     rxn_keys,
                     rxn_map,
                     input_map,
                     perturbation_map)

    @staticmethod
    def get_rxn_keys(rxn_types):
        keys = np.zeros_like(rxn_types)
        for rxn_type in np.unique(rxn_types):
            ind = (rxn_types==rxn_type)
            keys[ind] = np.cumsum(ind)[ind]-1
        return keys

    @classmethod
    def get_rxn_map(cls, network, maptype='propensity'):
        """
        Returns dictionary where keys are states and values are lists of  reaction indices whose rates depend upon each state.

        Args:

            network (Network)

            maptype (str) - type of rxn map


        Returns:

            adict (dict) - {state: index} pairs

        """

        if maptype == 'propensity':
            p_dict = cls.get_propensity_dict(network)
        elif maptype == 'repressors':
            p_dict = cls.get_repressor_dependence_dict(network)
        elif maptype == 'edges':
            p_dict = cls.get_edge_dict(network)
        elif maptype == 'modules':
            p_dict = cls.get_module_dependence_dict(network)

        adict = {i: [] for i in range(len(network.reactions))}
        for (i, rxn) in enumerate(network.reactions):
            list_of_lists = [p_dict[s] for s in rxn.stoichiometry.nonzero()[0]]
            if len(list_of_lists) > 0:
                alist = reduce(add, list_of_lists)
                adict[i].extend(alist)

        # remove duplicates
        for (k, v) in adict.items():
            adict[k] = list(set(v))

        return adict

    @staticmethod
    def get_repressor_dependence_dict(network):
        """
        Returns dictionary where keys are states and values are lists of  repressor indices whose occupancies depend upon each state.
        """
        adict = {i: [] for i in range(network.nodes.size)}
        get_name = lambda x: x.__class__.__name__
        rxns = [rxn for rxn in network.reactions if get_name(rxn)=='Coupling']

        if len(rxns) > 0:
            # store index of repressor i whose occupancy depends on state s
            repressors = reduce(add, [rxn.repressors for rxn in rxns])
            for i, repressor in enumerate(repressors):
                for s in repressor.active_substrates:
                    adict[s].append(i)

        return adict

    @staticmethod
    def get_module_dependence_dict(network):
        """
        Returns dictionary where keys are states and values are lists of  module indices whose occupancies depend upon each state.
        """
        adict = {i: [] for i in range(network.nodes.size)}
        rxns = [rxn for rxn in network.reactions if rxn.__class__.__name__=='Transcription']

        if len(rxns) > 0:
            # store index of module i whose occupancy depends on state s
            modules = reduce(add, [rxn.modules for rxn in rxns])
            for i, module in enumerate(modules):
                for s in module.modifiers:
                    adict[s].append(i)

        return adict

    @staticmethod
    def get_edge_dict(network):
        """
        Returns dictionary where keys are states and values are lists of edge indices whose occupancies depend upon each state.
        """
        adict = {i: [] for i in range(network.nodes.size)}
        rxns = [rxn for rxn in network.reactions if rxn.__class__.__name__=='Coupling']

        # store index of edge whose activity depends on state
        if len(rxns) > 0:
            dependents = np.hstack([rxn.active_species for rxn in rxns])
            for edge, state in enumerate(dependents):
                adict[state].append(edge)
        return adict

    @staticmethod
    def get_propensity_dict(network):
        """
        Returns dictionary where keys are states and values are lists of  reaction indices whose propensities depend upon each state.
        """
        adict = {i: [] for i in range(network.nodes.size)}
        for (i, rxn) in enumerate(network.reactions):

            # store index of reaction i whose propensity depends on state s
            for s in rxn.propensity.nonzero()[0]:
                adict[s].append(i)

            # store index of reaction i whose repression depends on state s
            rxn_type = rxn.__class__.__name__
            if rxn_type in ('Hill', 'Coupling'):
                for repressor in rxn.repressors:
                    for s in repressor.active_substrates:
                        adict[s].append(i)

            # store index of reaction i whose transcription depends on state s
            elif rxn_type == 'Transcription':
                for module in rxn.modules:
                    for s in module.modifiers:
                        adict[s].append(i)

        return adict

    @staticmethod
    def get_input_map(network):
        """ Map inputs to reactions. """
        adict = {i: [] for i in range(network.input_size)}
        for (j, rxn) in enumerate(network.reactions):
            rxn_type = rxn.__class__.__name__
            if rxn_type in ('Coupling', 'SumReaction', 'ProportionalController', 'IntegralController'):
                continue
            elif rxn_type == 'Transcription':
                if rxn.perturbed == True:
                    adict[0].append(j)
            else:
                for s in rxn.input_dependence.nonzero()[0]:
                    adict[s].append(j)
        return adict

    @staticmethod
    def get_perturbation_map(network):
        """ Map signal dimensions to perturbed reactions. """
        adict = {i: [] for i in range(network.input_size)}
        for (j, rxn) in enumerate(network.reactions):
            rxn_type = rxn.__class__.__name__
            if rxn_type == 'Transcription':
                if rxn.perturbed == True:
                    adict[0].append(j)
        return adict

