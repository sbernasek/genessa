__author__ = 'Sebi'

import numpy as np
from scipy.misc import comb
import functools
from operator import mul, add
import copy


class Reaction:

    def __init__(self, stoichiometry=None, propensity=None, input_dependence=None, k=1, rxn_type=None, temperature_sensitive=True, atp_sensitive=False, ribosome_sensitive=False):
        """
        Class describes a single kinetic pathway.

        Parameters:
            stoichiometry (array like or tuple) - list of stoichiometric coefficients for all species. if Tuple,
                indices of participating species where negative indicates consumption
            propensity (array like or tuple) - list of reaction orders for all species, if Tuple,
                indices of species
            input_dependence (float or np array) - order of rate dependence upon input(s)
            k (float) - mass-action reaction rate constant
            rxn_type (str) - type of reaction
            temperature_sensitive (bool) - if True, reaction rate is scalable with temperature
            atp_sensitive (bool) - if True, reaction rate is scalable with metabolic rate
            ribosome_sensitive (bool) - if True, reaction rate is scalable with translation capacity
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
        self.rate_constant = np.array([k], dtype=float)
        self.active_species = np.where(self.propensity != 0)[0]
        self.active_inputs = np.where(self.input_dependence != 0)[0]
        self.num_active_species = self.active_species.size

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

        stoichiometry = np.hstack((np.zeros(shift, dtype=int), self.stoichiometry))
        propensity = np.hstack((np.zeros(shift, dtype=int), self.propensity))
        rxn = Reaction(stoichiometry, propensity, self.input_dependence, self.rate_constant[0], self.rxn_type)
        return rxn

    @staticmethod
    def _get_activities(states, propensities):
        return [comb(s, p, exact=True) if p > 1 else s**p for s, p in zip(states, propensities)]

    def get_rate(self, states, input_state, discrete=True):
        """
        Compute and return current rate of a given pathway.

        Parameters:
            states (np array) - current state values
            input_state (np array) - current input value(s)
            discrete (bool) - if True, use discrete propensity function
        Returns:
            rate (float) - rate of reaction
        """

        # get active states and propensities
        _states = states[self.active_species]

        # get reactant activities. if discrete, use combination functions, otherwise use continuous rate law
        if discrete:
            activities = self._get_activities(_states, self._propensity)
        else:
            activities = _states**self._propensity

        # incorporate input dependence
        if self.num_active_species == 0:
            rate = self.rate_constant * functools.reduce(mul, input_state**self.input_dependence)
        else:
            rate = self.rate_constant * functools.reduce(mul, activities) * functools.reduce(mul, input_state**self.input_dependence)

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
            a = [self.rate_constant]
        elif np.max(propensity) == 2 and len(np.where(propensity == 2)) == 1:
            a = copy.copy(propensity)
            a[np.argmax(a)] -= 1
            i.append(a)
            a = [self.rate_constant/2, -self.rate_constant/2]
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
            k=np.array([js['rate_constant']], dtype=float),
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
            'rate_constant': self.rate_constant[0],
            'propensity': self.propensity.tolist(),
            'active_species': self.active_species.tolist(),
            'zero_order': self.zero_order,
            'rxn_type': self.rxn_type,
            'temperature_sensitive': self.temperature_sensitive,
            'atp_sensitive': self.atp_sensitive,
            'ribosome_sensitive': self.ribosome_sensitive
        }


class LinearReaction(Reaction):
    # Simplified version of Reaction class for linear rate laws.
    def __init__(self, **kwargs):
        Reaction.__init__(self, **kwargs)
        if self._propensity.sum() != 1 or self._input_dependence.sum() != 0:
            print('prop', self._propensity)
            print('input dep', self._input_dependence)
            print(self.__class__)
            raise ValueError('Reaction is not linear.')

    def get_rate(self, states, input_state, **kwargs):
        return self.rate_constant * states[self.active_species]


class LinearInput(Reaction):
    # Simplified version of Reaction class for linear input signal.
    def __init__(self, **kwargs):
        Reaction.__init__(self, **kwargs)
        if self._propensity.sum() != 0 or self._input_dependence.sum() != 1:
            raise ValueError('Reaction is not a linear input.')

    def get_rate(self, states, input_state, **kwargs):
        return self.rate_constant * input_state


class SecondOrderReaction(Reaction):
    # Simplified version of Reaction class for second order rate laws.

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
            rate = self.rate_constant * functools.reduce(mul, states[self.active_species])
        else:
            rate = self.rate_constant
        if self.active_inputs.size > 0:
            rate *= input_state

        return rate


class EnzymaticReaction:

    def __init__(self, stoichiometry=None, propensity=None, input_dependence=None,
                 rate_constant=1, k_m=1, hill=1, baseline=0, repressors=None, rxn_type='hill',
                 temperature_sensitive=False, atp_sensitive=False, ribosome_sensitive=False):
        """
        Class describes a single hill-kinetic pathway.

        Parameters:
            stoichiometry (array like) - list of stoichiometric coefficients for all species
            propensity (array like) - weights for activating substrates
            input_dependence (float or array) - order of rate dependence upon input
            rate_constant (float) - maximum reaction rate
            k_m (float) - michaelis constant
            hill (float) - hill coefficient
            baseline (float) - baseline reaction rate in absense of any substrate
            repressors (list) - list of repressor objects affecting reaction pathway
            rxn_type (str) - name of reaction
            temperature_sensitive (bool) - if True, reaction rate is scalable with temperature
            atp_sensitive (bool) - if True, reaction rate is scalable with metabolic rate
            ribosome_sensitive (bool) - if True, reaction rate is scalable with translation capacity
        """

        self.rxn_type = rxn_type

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
        self.input_dependence = np.array(input_dependence, dtype=np.float64)

        self.rate_constant = np.array([rate_constant], dtype=float)
        self.k_m = k_m
        self.n = hill
        self.baseline = baseline
        self.zero_order = False

        # add repressors
        if repressors is None:
            self.repressors = []
        else:
            self.repressors = repressors
        self.num_repressors = len(self.repressors)

        # identify participating substrates
        self.active_substrates = np.where(self.propensity != 0)[0]
        self._propensity = self.propensity[self.active_substrates]
        self.active_inputs = np.where(self.input_dependence != 0)[0]

        # assign reaction rate sensitivities
        self.temperature_sensitive = temperature_sensitive
        self.atp_sensitive = atp_sensitive
        self.ribosome_sensitive = ribosome_sensitive

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
        rate = unoccupied_sites * (self.rate_constant * (substrate_activity**self.n)/(substrate_activity**self.n + self.k_m**self.n) + self.baseline)

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
            'rate_constant': self.rate_constant[0],
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
            rate_constant=np.array([js['rate_constant']], dtype=float),
            k_m=js['k_m'],
            hill=js['n'],
            baseline=js['baseline'],
            repressors=[EnzymaticRepressor.from_json(repressor_js) for repressor_js in js['repressors']],
            rxn_type=js['rxn_type'],
            temperature_sensitive=js['temperature_sensitive'],
            atp_sensitive=js['atp_sensitive'],
            ribosome_sensitive=js['ribosome_sensitive'])
        return enzymatic_rxn


class EnzymaticRepressor:

    def __init__(self, propensity=None, input_dependence=None, k_m=1, hill=1):
        """
        Class defines single instance of competitive enzyme occupancy.

        Parameters:
            propensity (array like) - weights for activating substrates
            input_dependence (float or np array) - order of rate dependence upon input
            k_m (float) - michaelis constant
            hill (float) - hill coefficient
        """

        # define rate law parameters
        if propensity is None:
            propensity = []
        self.propensity = np.array(propensity, dtype=np.float64)

        # convert input dependence to array
        if type(input_dependence) == int:
            input_dependence = [input_dependence]
        self.input_dependence = np.array(input_dependence, dtype=np.float64)

        self.k_m = k_m
        self.n = hill

        # determine active substrates
        self.active_substrates = np.where(self.propensity != 0)[0]
        self._propensity = self.propensity[self.active_substrates]
        self.active_inputs = np.where(self.input_dependence != 0)[0]

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
        substrate_activity = (self._propensity * states[self.active_substrates]).sum() + (self.input_dependence *input_state).sum()

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
            hill=js['n'])
        return repressor

class SumReaction:

    def __init__(self, stoichiometry, propensity, k=1, rxn_type=None):

        self.rxn_type = rxn_type

        # compile stoichiometry as a vector of coefficients
        self.stoichiometry = np.array(stoichiometry, dtype=np.int64)

        # compile propensity as a vector of coefficients
        self.propensity = np.array(propensity, dtype=np.int64)
        self.input_dependence = np.zeros(1, dtype=np.int64)

        # set reaction parameters
        self.rate_constant = np.array([k], dtype=float)
        self.active_species = np.where(self.propensity != 0)[0]
        self.num_active_species = self.active_species.size

        # predefine active species mask
        self._propensity = self.propensity[self.active_species]

        # assign reaction rate sensitivities
        self.temperature_sensitive = True
        self.atp_sensitive = False
        self.ribosome_sensitive = False

    def shift(self, shift):

        stoichiometry = np.hstack((np.zeros(shift, dtype=int), self.stoichiometry))
        propensity = np.hstack((np.zeros(shift, dtype=int), self.propensity))
        rxn = SumReaction(stoichiometry, propensity, self.rate_constant[0], self.rxn_type)
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
        rate = self.rate_constant * functools.reduce(add, activities)

        if rate < 0:
            rate *= 0

        return rate

class ProportionalController(SumReaction):
    pass

class IntegralController(SumReaction):
    pass
