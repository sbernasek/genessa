import numpy as np
from scipy.misc import comb
import functools
from operator import mul, add
import copy
from .solver.rxns import RateFunction as cRateFunction


def name_parameter(parameter, default_name='k'):
    """ Get parameter name. """

    # set reaction parameters
    if type(parameter) in (tuple, list):
        value, name = parameter

    elif type(parameter) == dict:
        value = list(parameter.keys())[0]
        name = list(parameter.values())[0]

    # if only parameter value is provided, use default name
    elif type(parameter) in (int, float, np.float64, np.int64):
        value, name = parameter, default_name

    return value, name


class Reaction:

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

        s = np.hstack((np.zeros(shift, dtype=int), self.stoichiometry))
        p = np.hstack((np.zeros(shift, dtype=int), self.propensity))
        i = self.input_dependence

        kw = dict(k=self.k[0],
                  rxn_type=self.rxn_type,
                  parameters=self.parameters)

        return Reaction(s, p, i, **kw)

    @staticmethod
    def _get_activities(states, propensities):
        return [comb(s, p, exact=True) if p > 1 else s**p for s, p in zip(states, propensities)]

    def get_rate(self, states, input_state, discrete=True):
        """
        Compute and return current rate of a given pathway.

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
            rate = self.k * functools.reduce(mul, input_state**self.input_dependence)
        else:
            rate = self.k * functools.reduce(mul, activities) * functools.reduce(mul, input_state**self.input_dependence)

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
            a = copy.copy(propensity)
            a[np.argmax(a)] -= 1
            i.append(a)
            a = [self.k/2, -self.k/2]
        else:
            print('Propensity function is too complex.')
        return i, a

    @staticmethod
    def from_json(js):
        """
        Instantiate Rection object from json-serialized dictionary.
        """

        # create instance
        rxn = Reaction(
            stoichiometry=np.array(js['stoichiometry']),
            propensity=np.array(js['propensity']),
            input_dependence=js['input_dependence'],
            k=np.array([js['k']], dtype=float),
            rxn_type=js['rxn_type'],
            temperature_sensitive=js['temperature_sensitive'],
            atp_sensitive=js['atp_sensitive'],
            ribosome_sensitive=js['ribosome_sensitive']
        )
        return rxn

    def to_json(self):
        return {
            # return each attribute
            'stoichiometry': self.stoichiometry.tolist(),
            'input_dependence': self.input_dependence.tolist(),
            'k': self.k[0],
            'propensity': self.propensity.tolist(),
            'active_species': self.active_species.tolist(),
            'zero_order': self.zero_order,
            'rxn_type': self.rxn_type,
            'temperature_sensitive': self.temperature_sensitive,
            'atp_sensitive': self.atp_sensitive,
            'ribosome_sensitive': self.ribosome_sensitive
        }


class LinearReaction(Reaction):
    """ Simplified version of Reaction class for linear rate laws. """
    def __init__(self, **kwargs):
        Reaction.__init__(self, **kwargs)
        if self._propensity.sum() != 1 or self._input_dependence.sum() != 0:
            print('prop', self._propensity)
            print('input dep', self._input_dependence)
            print(self.__class__)
            raise ValueError('Reaction is not linear.')

    def get_rate(self, states, input_state, **kwargs):
        return self.k * states[self.active_species]


class LinearInput(Reaction):
    """ Simplified version of Reaction class for linear input signal. """
    def __init__(self, **kwargs):
        Reaction.__init__(self, **kwargs)
        if self._propensity.sum() != 0 or self._input_dependence.sum() != 1:
            raise ValueError('Reaction is not a linear input.')

    def get_rate(self, states, input_state, **kwargs):
        return self.k * input_state


class SecondOrderReaction(Reaction):
    """ Simplified version of Reaction class for second order rate laws. """

    # WARNING: does not support multiple input channels (easy fix...)
    def __init__(self, **kwargs):
        Reaction.__init__(self, **kwargs)
        if (self._propensity.sum() + self._input_dependence.sum()) != 2:
            raise ValueError('Reaction is not second order.')

        if 2 in self._propensity or 2 in self._input_dependence:
            raise ValueError('Reaction is has a quadratic dependency.')

    def get_rate(self, states, input_state, **kwargs):
        """ Compute and return current rate of a second order pathway. """
        if self.num_active_species > 0:
            rate = self.k * functools.reduce(mul, states[self.active_species])
        else:
            rate = self.k
        if self.active_inputs.size > 0:
            rate *= input_state

        return rate


class EnzymaticReaction:

    def __init__(self, stoichiometry=None, propensity=None, input_dependence=None, k=1, k_m=1, n=1, baseline=0, repressors=None, rxn_type='hill', rate_modifier=None, temperature_sensitive=False, atp_sensitive=False, ribosome_sensitive=False, parameters=None):
        """
        Class describes a single hill-kinetic pathway.

        Args:
            stoichiometry (array like) - list of stoichiometric coefficients for all species
            propensity (array like) - weights for activating substrates
            input_dependence (float or array) - order of rate dependence upon input
            k (float) - maximum reaction rate
            k_m (float) - michaelis constant
            n (float) - hill coefficient
            baseline (float) - baseline reaction rate in absense of any substrate
            repressors (list) - list of repressor objects affecting reaction pathway
            rxn_type (str) - name of reaction
            rate_modifier (array like) - coefficients by which input modules rate constant
            temperature_sensitive (bool) - if True, reaction rate is scalable with temperature
            atp_sensitive (bool) - if True, reaction rate is scalable with metabolic rate
            ribosome_sensitive (bool) - if True, reaction rate is scalable with translation capacity
        """

        self.rxn_type = rxn_type
        self.zero_order = False

        # define stoichiometry
        if stoichiometry is None:
            stoichiometry = [0]
        self.stoichiometry = np.array(stoichiometry, dtype=np.int64)

        # define rate law parameters
        if propensity is None:
            propensity = np.zeros(len(stoichiometry))
        self.propensity = np.array(propensity, dtype=np.float64)

        # convert input dependence to array
        if type(input_dependence) == int:
            input_dependence = [input_dependence]
        if input_dependence is not None:
            self.input_dependence = np.array(input_dependence, dtype=np.float64)
        else:
            self.input_dependence = np.zeros(1, dtype=np.float64)

        # set parameters
        if parameters is None:
            parameters = {}
        self.parameters = parameters
        k_value, k_name = name_parameter(k, 'k')
        if 'k' not in self.parameters.keys():
            self.parameters['k'] = k_name
        self.k = np.array([k_value], dtype=float)

        km_value, km_name = name_parameter(k_m, 'k_m')
        if 'k_m' not in self.parameters.keys():
            self.parameters['k_m'] = km_name
        self.k_m = km_value

        n_value, n_name = name_parameter(n, 'n')
        if 'n' not in self.parameters.keys():
            self.parameters['n'] = n_name
        self.n = n_value

        baseline_value, baseline_name = name_parameter(baseline, 'v0')
        if 'v0' not in self.parameters.keys():
            self.parameters['v0'] = baseline_name
        self.baseline = baseline_value

        # add repressors
        if repressors is None:
            self.repressors = []
        else:
            self.repressors = repressors
        self.num_repressors = len(self.repressors)

        # set rate modifier
        if rate_modifier is None:
            rate_modifier = np.zeros(1, dtype=np.int64)
        self.rate_modifier = rate_modifier

        # identify participating substrates
        self.active_substrates = np.where(self.propensity != 0)[0]
        self.active_inputs = np.where(np.logical_or(self.input_dependence != 0, self.rate_modifier != 0))[0]

        # predefine active species mask
        self.num_active_substrates = self.active_substrates.size
        self.num_active_inputs = self.active_inputs.size
        self._propensity = self.propensity[self.active_substrates]
        self._input_dependence = self.input_dependence[self.active_inputs]

        # assign reaction rate sensitivities
        self.temperature_sensitive = temperature_sensitive
        self.atp_sensitive = atp_sensitive
        self.ribosome_sensitive = ribosome_sensitive

    def shift(self, shift):
        """ Expand dimensionality of reaction. """

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
                  rxn_type=self.rxn_type,
                  rate_modifier=self.rate_modifier,
                  temperature_sensitive=self.temperature_sensitive,
                  atp_sensitive=self.atp_sensitive,
                  ribosome_sensitive=self.ribosome_sensitive,
                  parameters=self.parameters)

        return EnzymaticReaction(s, p, i, **kw)

    def add_promoter(self, species):
        """
        Adds promoter to enzymatic reaction.

        Parameters:
            species (int) - index of promoter within state space
        """
        self.propensity[species] += 1
        self.active_substrates = np.where(self.propensity != 0)[0]
        self._propensity = self.propensity[self.active_substrates]
        #self.active_substrates[species] += 1

    def add_repressor(self, repressor):
        """
        Adds repressor to reaction.

        Parameters:
            repressor (EnzymaticRepressor object) - repressor to be added to enzymatic reaction
        """
        self.repressors.append(repressor)
        self.num_repressors += 1

    def get_rate(self, states, input_state, **kwargs):
        """
        Compute and return current rate of reaction.

        Parameters:
            states (np array) - current state values
            input_state (np array) - current input value(s)

        Returns:
            rate (float) - rate of reaction
        """

        # get substrate activity
        substrate_activity = (self._propensity * states[self.active_substrates]).sum() + (self.input_dependence*input_state).sum()

        # get repressor inhibition effects
        unoccupied_sites = 1
        if self.num_repressors > 0:
            unoccupied_sites = functools.reduce(mul, [1-repressor.get_occupancy(states, input_state) for repressor in self.repressors])

        # get overall rate
        k = self.k + (self.rate_modifier * input_state).sum()
        rate = unoccupied_sites * (k * (substrate_activity**self.n)/(substrate_activity**self.n + self.k_m**self.n) + self.baseline)

        return rate

    def to_json(self):
        """
        Return json-compatible serialized dictionary of object.
        """
        return {
            # return each attribute
            'rxn_type': self.rxn_type,
            'stoichiometry': self.stoichiometry.tolist(),
            'propensity': self.propensity.tolist(),
            'input_dependence': self.input_dependence.tolist(),
            'k': self.k[0],
            'k_m': self.k_m,
            'n': self.n,
            'baseline': self.baseline,
            'temperature_sensitive': self.temperature_sensitive,
            'atp_sensitive': self.atp_sensitive,
            'ribosome_sensitive': self.ribosome_sensitive,
            'repressors': [repressor.to_json() for repressor in self.repressors]
            }

    @staticmethod
    def from_json(js):
        """
        Instantiate EnzymaticReaction object from json-serialized dictionary of object.
        """
        enzymatic_rxn = EnzymaticReaction(
            stoichiometry=np.array(js['stoichiometry']),
            propensity=np.array(js['propensity']),
            input_dependence=js['input_dependence'],
            k=np.array([js['k']], dtype=float),
            k_m=js['k_m'],
            n=js['n'],
            baseline=js['baseline'],
            repressors=[EnzymaticRepressor.from_json(repressor_js) for repressor_js in js['repressors']],
            rxn_type=js['rxn_type'],
            temperature_sensitive=js['temperature_sensitive'],
            atp_sensitive=js['atp_sensitive'],
            ribosome_sensitive=js['ribosome_sensitive'])
        return enzymatic_rxn


class EnzymaticRepressor:

    def __init__(self, propensity=None, input_dependence=None, k_m=1, n=1, parameters=None):
        """
        Class defines single instance of competitive enzyme occupancy.

        Parameters:
            propensity (array like) - weights for activating substrates
            input_dependence (float or np array) - order of rate dependence upon input
            k_m (float) - michaelis constant
            n (float) - hill coefficient
        """

        # define rate law parameters
        if propensity is None:
            propensity = []
        self.propensity = np.array(propensity, dtype=np.float64)

        # convert input dependence to array
        if type(input_dependence) == int:
            input_dependence = [input_dependence]
        if input_dependence is not None:
            self.input_dependence = np.array(input_dependence, dtype=np.float64)
        else:
            self.input_dependence = np.zeros(1, dtype=np.float64)

        # set parameters
        if parameters is None:
            parameters = {}
        self.parameters = parameters

        km_value, km_name = name_parameter(k_m, 'k_m')
        if 'k_m' not in self.parameters.keys():
            self.parameters['k_m'] = km_name
        self.k_m = km_value

        n_value, n_name = name_parameter(n, 'n')
        if 'n' not in self.parameters.keys():
            self.parameters['n'] = n_name
        self.n = n_value

        # determine active substrates
        self.active_substrates = np.where(self.propensity != 0)[0]
        self.active_inputs = np.where(self.input_dependence != 0)[0]

        # predefine active species mask
        self.num_active_substrates = self.active_substrates.size
        self.num_active_inputs = self.active_inputs.size
        self._propensity = self.propensity[self.active_substrates]
        self._input_dependence = self.input_dependence[self.active_inputs]

    def shift(self, shift):
        """ Expand dimensionality of reaction. """
        p = np.hstack((np.zeros(shift, dtype=np.float64), self.propensity))
        i = self.input_dependence
        kw = dict(k_m=self.k_m,
                  n=self.n,
                  parameters=self.parameters)
        return EnzymaticRepressor(p, i, **kw)

    def get_occupancy(self, states, input_state):
        """
        Compute and return current occupancy of enzyme by repressive substrate.

        Parameters:
            states (np array) - current state values
            input_state (vector) - current input value(s)

        Returns:
            occupancy (float) - fraction of enzyme occupied by repressive substrate
        """

        # get substrate activity
        substrate_activity = (self._propensity * states[self.active_substrates]).sum() + (self.input_dependence * input_state).sum()

        # get overall rate
        occupancy = (substrate_activity**self.n)/(substrate_activity**self.n + self.k_m**self.n)

        return occupancy

    def to_json(self):
        """
        Return json-compatible serialized dictionary of object.
        """
        return {
            # return each attribute
            'propensity': self.propensity.tolist(),
            'input_dependence': self.input_dependence.tolist(),
            'k_m': self.k_m,
            'n': self.n}

    @staticmethod
    def from_json(js):
        """
        Instantiate EnzymaticReaction object from json-serialized dictionary of object.
        """
        repressor = EnzymaticRepressor(
            propensity=np.array(js['propensity']),
            input_dependence=js['input_dependence'],
            k_m=js['k_m'],
            n=js['n'])
        return repressor


class SumReaction:

    def __init__(self, stoichiometry, propensity, k=1, rxn_type=None, parameters=None):

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
        s = np.hstack((np.zeros(shift, dtype=int), self.stoichiometry))
        p = np.hstack((np.zeros(shift, dtype=int), self.propensity))
        kw = dict(k=self.k[0],
                  rxn_type=self.rxn_type,
                  parameters=self.parameters)
        rxn = SumReaction(s, p, **kw)
        return rxn

    def get_rate(self, states, input_value=0, discrete=True):
        """
        Compute and return current rate of a given pathway.

        Parameters:
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
        rate = self.k * functools.reduce(add, activities)

        if rate < 0:
            rate *= 0

        return rate


class ProportionalController(SumReaction):
    pass


class IntegralController(SumReaction):
    pass


class Coupling:

    def __init__(self, stoichiometry=None, propensity=None, k=1, a=1, w=1,repressors=None, rxn_type='coupling', parameters=None):
        """
        Class describes a single coupling pathway.

        Parameters:
            stoichiometry (array like) - list of stoichiometric coefficients for all species
            propensity (array like) - weights for coupling comparison
            k (float) - baseline rxn rate
            a (float) - coupling strength
            w (float) - edge weights
            repressors (list) - list of repressor objects
            rxn_type (str) - name of reaction
        """

        self.rxn_type = rxn_type

        # define stoichiometry
        if stoichiometry is None:
            stoichiometry = [0]
        self.stoichiometry = np.array(stoichiometry, dtype=np.int64)

        # define input dependence (not used)
        self.input_dependence = np.zeros(1, dtype=np.int64)

        # define rate law parameters
        if propensity is None:
            propensity = np.zeros(len(stoichiometry))
        self.propensity = np.array(propensity, dtype=np.float64)

        # set parameters
        if parameters is None:
            parameters = {}
        self.parameters = parameters

        k_value, k_name = name_parameter(k, 'k')
        if 'k' not in self.parameters.keys():
            self.parameters['k'] = k_name
        self.k = np.array([k_value], dtype=float)

        a_value, a_name = name_parameter(a, 'a')
        if 'a' not in self.parameters.keys():
            self.parameters['a'] = a_name
        self.a = a_value

        w_value, w_name = name_parameter(w, 'w')
        if 'w' not in self.parameters.keys():
            self.parameters['w'] = w_name
        self.w = w_value

        # add repressors
        if repressors is None:
            self.repressors = []
        else:
            self.repressors = repressors
        self.num_repressors = len(self.repressors)

        # identify participating substrates
        self.active_species = np.where(self.propensity != 0)[0]
        self._propensity = self.propensity[self.active_species]
        self.num_active_species = self.active_species.size

        # assign reaction rate sensitivities
        self.temperature_sensitive = True
        self.atp_sensitive = False
        self.ribosome_sensitive = False

    def shift(self, shift):
        """ Expand dimensionality of reaction. """

        s = np.hstack((np.zeros(shift, dtype=int), self.stoichiometry))
        p = np.hstack((np.zeros(shift, dtype=np.float64), self.propensity))

        # shift repressors
        repressors = [rep.shift(shift) for rep in self.repressors]

        kw = dict(k=self.k[0],
                  a=self.a,
                  w=self.w,
                  parameter_names=self.parameter_names,
                  repressors=repressors,
                  rxn_type=self.rxn_type)

        return Coupling(s, p, **kw)

    def add_repressor(self, repressor):
        """
        Adds repressor to reaction.

        Parameters:
            repressor (EnzymaticRepressor object) - repressor to be added to enzymatic reaction
        """
        self.repressors.append(repressor)
        self.num_repressors += 1

    def get_rate(self, states, input_state, **kwargs):
        """
        Compute and return current rate of reaction.

        Parameters:
            states (np array) - current state values
            input_state (np array) - current input value(s)

        Returns:
            rate (float) - rate of reaction
        """

        # get substrate activity
        rate = 0
        if self._propensity.size != 0:
            rate += (self._propensity * states[self.active_species]).sum()
            N = self.active_species.size
            rate *= (self.a*self.w / (1+self.w * (N - 1)))

        # add constant term
        rate += self.k[0]

        # get repressor inhibition effects
        unoccupied_sites = 1
        if self.num_repressors > 0:
            unoccupied_sites = functools.reduce(mul, [1-repressor.get_occupancy(states, input_state) for repressor in self.repressors])

        # get overall rate
        rate = unoccupied_sites * rate

        return rate


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

        modifiers = np.hstack((np.zeros(shift, dtype=np.int64), self.modifiers))
        kw = dict(
            nA = self.nA,
            nD = self.nD,
            bindsAsComplex = self.bindsAsComplex,
            k = self.k,
            n = self.n)
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

        s = np.hstack((np.zeros(shift, dtype=int), self.stoichiometry))
        modules = [mod.shift(shift) for mod in self.modules]

        kw = dict(k=self.k[0],
                  alpha=self.alpha,
                  perturbed=self.perturbed,
                  input_dependence=self.input_dependence,
                  rxn_type=self.rxn_type,
                  parameters=self.parameters)

        return Transcription(s, modules, **kw)

    def get_rate(self, x, inputs):

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


class RateFunction:

    def __init__(self, network):
        self.N = network.nodes.size
        self.M = len(network.reactions)
        self.reactions = network.reactions
        self.cRateFunction = cRateFunction(network)

    def __call__(self, states, input_state):
        return self.get_rates(states, input_state)

    def get_rxn_rates(self, states, input_state):
        rates = np.zeros(self.M, dtype=np.float64)
        for i, rxn in enumerate(self.reactions):
            rates[i] = rxn.get_rate(states, input_state)
        return rates

    def get_rates(self, states, input_state):
        rates = np.zeros(self.N, dtype=np.float64)
        for rxn in self.reactions:
            rates += rxn.get_rate(states, input_state) * rxn.stoichiometry
        return rates

    def cget_rxn_rates(self, states, input_state, cumul):
        return self.cRateFunction(states, input_state, cumul)

    def cget_rates(self, states, input_state, cumul):
        rates = np.zeros(self.N, dtype=np.float64)
        rxn_rates = self.cRateFunction(states, input_state, cumul)
        for i, rxn in enumerate(self.reactions):
            rates += rxn_rates[i] * rxn.stoichiometry
        return rates
