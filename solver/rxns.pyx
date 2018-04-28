# cython: profile=True


"""
TO DO:

1. Can pre-allocate array of rxn types, with array of "start positions" in propensity arrays for each rxn time (to avoid massive sparseness)

2. Can pre-allocate array of propensity coefficients for each rxn type

3. For each update, pull rxn type, get index for corresponding propensity array, then compute propensity.

4. One "get_rate" function serves for all reactions of a given type

"""


from rxndiffusion.reactions import Reaction as pyMassAction
from rxndiffusion.reactions import EnzymaticReaction as pyHill
from rxndiffusion.reactions import IntegralController as pyIControl
from rxndiffusion.reactions import ProportionalController as pyPControl
from rxndiffusion.reactions import Coupling as pyCoupling
from rxns cimport cMassAction, cRepressor, cHill, cCoupling, cRateFunction
import numpy as np
cimport numpy as np
cimport cython
cimport rxndiffusion.solver.cython_sum as cython_sum

from cpython.array cimport array
from array import array

from functools import reduce
from operator import add


cdef class cActive:
    def __init__(self, int N, long[:] active):
        self.N = N
        self.active = active

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    cdef double get_rate(self, array states, array input_value, array cumulative):
        return 0.


cdef class cActiveInputs(cActive):
    def __init__(self, int N, long[:] active, int M, long[:] active_inputs):
        cActive.__init__(self, N, active)
        self.M = M
        self.active_inputs = active_inputs


cdef class cSumRxn(cActive):
    def __init__(self, int N, long[:] active, double k, long[:] propensity):
        cActive.__init__(self, N, active)
        self.k = k
        self.propensity = propensity

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double get_rate_longs(self, array values):
        """ Get mass action reaction rate. """

        cdef double rate = 0
        cdef int j
        cdef int s_ind

        # get species dependencies
        for j in xrange(self.N):
            s_ind = self.active[j]
            rate += self.propensity[s_ind]*values.data.as_longs[s_ind]

        if rate < 0:
            rate = 0.

        rate *= self.k
        return rate

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double get_rate_doubles(self, array values):
        """ Get reaction rate. """

        cdef double rate = 0
        cdef int j
        cdef int s_ind

        # get species dependencies
        for j in xrange(self.N):
            s_ind = self.active[j]
            rate += self.propensity[s_ind]*values.data.as_doubles[s_ind]

        if rate < 0:
            rate = 0.

        rate *= self.k
        return rate


cdef class cProportionalController(cSumRxn):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double get_rate(self, array states, array input_value, array cumulative):
        return self.get_rate_longs(states)


cdef class cIntegralController(cSumRxn):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double get_rate(self, array states, array input_value, array cumulative):
        return self.get_rate_doubles(cumulative)


cdef class cMassAction(cActiveInputs):

    def __init__(self, int N, long[:] active, int M, long[:] active_inputs,double k, long[:] propensity, long[:] input_dependence):
        cActiveInputs.__init__(self, N, active, M, active_inputs)
        self.k = k
        self.propensity = propensity
        self.input_dependence = input_dependence

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double get_rate(self, array states, array input_value, array cumulative):
        return self.c_get_rate(states, input_value)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double c_get_rate(self, array states, array input_value) nogil:
        """ Get mass action reaction rate. """
        cdef double rate
        cdef int i, j
        cdef int s_ind, i_ind
        cdef double k
        cdef long n
        rate = self.k

        # get input dependencies
        for i in xrange(self.M):
            i_ind = self.active_inputs[i]
            k = self.input_dependence[i_ind]

            if k > 0:
                rate *= input_value.data.as_doubles[i_ind] ** k

        # get species dependencies
        for j in xrange(self.N):
            s_ind = self.active[j]
            k = self.propensity[s_ind]
            n = states.data.as_longs[s_ind]

            if k == 1:
                rate *= n
            elif k  == 2:
                rate *= n*(n-1)/2.

        return rate


cdef class cRepressor(cActiveInputs):

    def __init__(self, int N, long[:] active, int M, long[:] active_inputs,
                 double k_m, double n,
                 double[:] propensity, double[:] input_dependence):
        cActiveInputs.__init__(self, N, active, M, active_inputs)
        self.k_m = k_m
        self.n = n
        self.propensity = propensity
        self.input_dependence = input_dependence

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double get_occupancy(self, array states, array input_value) nogil:
        """ Get occupancy by repressor. """

        cdef double occupancy
        cdef int i, j
        cdef int s_ind, i_ind
        cdef double activity = 0

        # compute occupancy (could pre-assign indices...)
        for i in xrange(self.N):
            s_ind = self.active[i]
            activity += (states.data.as_longs[s_ind] * self.propensity[s_ind])
        for j in xrange(self.M):
            i_ind = self.active_inputs[j]
            activity += input_value.data.as_doubles[i_ind] * self.input_dependence[i_ind]
        occupancy = (activity**self.n)/(activity**self.n + self.k_m**self.n)
        return occupancy


cdef class cHill(cActiveInputs):

    def __init__(self, int N, long[:] active, int M, long[:] active_inputs,
                 int R,
                 double k, double k_m, double n, double baseline,
                 double[:] propensity, double[:] input_dependence,
                 cRepressor[:] c_reps,
                 double[:] rate_modifier):
        cActiveInputs.__init__(self, N, active, M, active_inputs)
        self.R = R
        self.k = k
        self.k_m = k_m
        self.n = n
        self.baseline = baseline
        self.propensity = propensity
        self.input_dependence = input_dependence
        self.repressors = c_reps
        self.rate_modifier = rate_modifier

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double get_rate(self, array states, array input_value, array cumulative):
        return self.c_get_rate(states, input_value)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double c_get_rate(self, array states, array input_value):
        """ Get rate for Hill reaction. """

        cdef double rate = self.k
        cdef int i, j, k
        cdef int s_ind, i_ind
        cdef double n
        cdef double activity = 0
        cdef double available = 1
        cdef cRepressor repressor
        cdef double rate_modifier = 0

        # compute activity
        for i in xrange(self.N):
            s_ind = self.active[i]
            activity += (states.data.as_longs[s_ind] * self.propensity[s_ind])
        for j in xrange(self.M):
            i_ind = self.active_inputs[j]
            activity += input_value.data.as_doubles[i_ind] * self.input_dependence[i_ind]
            rate_modifier += input_value.data.as_doubles[i_ind] * self.rate_modifier[i_ind]

        # compute repressor occupancy
        for k in xrange(self.R):
            repressor = self.repressors[k]
            available *= (1 - repressor.get_occupancy(states, input_value))

        # compute overall rate
        rate += rate_modifier
        rate *= available * ((activity**self.n)/(activity**self.n + self.k_m**self.n) + self.baseline)

        return rate

cdef class cCoupling(cActive):
    def __init__(self, int N, long[:] active, int R,
                 double k, long[:] propensity,
                 double a, double w, cRepressor[:] c_reps):
        cActive.__init__(self, N, active)
        self.R = R
        self.k = k
        self.propensity = propensity
        self.a = a
        self.w = w
        self.repressors = c_reps

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double get_rate(self, array states, array input_value, array cumulative):
        return self.c_get_rate(states, input_value)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double c_get_rate(self, array states, array input_value):
        """ Get reaction rate for coupling mechanism. """

        cdef double rate = 0
        cdef int j
        cdef int s_ind
        cdef double available = 1
        cdef cRepressor repressor
        cdef double occupancy

        # if coupled
        if self.N > 0:

            # get species dependencies
            for j in xrange(self.N):
                s_ind = self.active[j]
                rate += self.propensity[s_ind]*states.data.as_longs[s_ind]

            # apply constants
            rate *= (self.a*self.w / (1+self.w * (self.N - 1)))
            rate += self.k
            if rate < 0:
                rate = 0.

        # if no coupling
        else:
            rate = self.k

        # compute repressor occupancy
        for j in xrange(self.R):
            repressor = self.repressors[j]
            occupancy = repressor.get_occupancy(states, input_value)
            rate *= (1 - occupancy)

        return rate


cdef class cRateFunction:

    def __init__(self,
                 cActive[:] rxns,
                 long[:] rxn_types,
                 dict rxn_map,
                 dict input_map):

        # store reactions
        self.rxns = rxns
        self.rxn_types = rxn_types #array('l', rxn_type_identifiers)
        self.M = len(self.rxns)
        self.rxn_map = rxn_map
        self.input_map = input_map

        # initialize rates array
        self.rates = array('d', np.zeros(self.M, dtype=np.float64))
        self.total_rate = cython_sum.sum_double_arr(self.rates, self.M)

    cdef array get_rxn_rates(self):
        """ Get current rate vector. """
        return self.rates

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_input(self, array states, array input_value, array cumulative, int input_dim):
        """ Update rates for reactions that have changed. """
        #cdef int i
        cdef rxn_ind
        cdef list indices = self.input_map[input_dim]
        for rxn_ind in indices:
            self.set_rate(states, input_value, cumulative, rxn_ind)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update(self, array states, array input_value, array cumulative, int fired):
        """ Update rates for reactions that have changed. """
        #cdef int i
        cdef rxn_ind
        cdef list indices = self.rxn_map[fired]
        for rxn_ind in indices:
            self.set_rate(states, input_value, cumulative, rxn_ind)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void update_all(self, array states, array input_value, array cumulative):
        """ Update all reaction rates. """
        cdef int rxn
        for rxn in xrange(self.M):
            self.set_rate(states, input_value, cumulative, rxn)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double evaluate(self, array states, array input_value, array cumulative, int rxn):
        """ Evaluate rate of individual reaction. """

        cdef int rxn_type
        cdef cMassAction mass_rxn
        cdef cHill hill_rxn
        cdef cSumRxn sum_rxn
        cdef cCoupling coupling_rxn

        # get reaction type
        rxn_type = self.rxn_types[rxn]

        # get reaction rate
        if rxn_type == 0:
            coupling_rxn = self.rxns[rxn]
            rate = coupling_rxn.c_get_rate(states, input_value)

        elif rxn_type == 1:
            mass_rxn = self.rxns[rxn]
            rate = mass_rxn.c_get_rate(states, input_value)

        elif rxn_type == 2:
            hill_rxn = self.rxns[rxn]
            rate = hill_rxn.c_get_rate(states, input_value)

        elif rxn_type == 3:
            sum_rxn = self.rxns[rxn]
            rate = sum_rxn.get_rate_doubles(cumulative)

        elif rxn_type == 4:
            sum_rxn = self.rxns[rxn]
            rate = sum_rxn.get_rate_longs(states)

        else:
            raise ValueError('Reaction type not recognized.')

        #rxn.get_rate(states, input_value, cumulative)

        return rate

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void set_rate(self, array states, array input_value, array cumulative, int rxn):
        """ Set rate for individual reaction. """

        cdef double old_rate, rate

        # update rxn rate
        rate = self.evaluate(states, input_value, cumulative, rxn)
        old_rate = self.rates.data.as_doubles[rxn]

        # update total rate
        self.rates.data.as_doubles[rxn] = rate
        self.total_rate += (rate - old_rate)


class MassActionReaction:
    """ Typecasting for cMassAction instances. """
    def __init__(self, py_rxn):
        """
        Args:
        py_rxn (nevosim.Reaction instance)
        """
        cdef cMassAction c_rxn
        N = int(py_rxn.active_species.size)
        active = py_rxn.active_species.astype(np.int64)
        M = int(py_rxn.active_inputs.size)
        active_inputs = py_rxn.active_inputs.astype(np.int64)
        k = py_rxn.k.astype(np.float64)
        propensity = py_rxn.propensity.astype(np.int64)
        input_dependence = py_rxn.input_dependence.astype(np.int64)
        c_rxn = cMassAction(N, active, M, active_inputs, k, propensity, input_dependence)
        self.c_rxn = c_rxn

    def get_rate(self, states, input_value):
        return self.c_rxn.get_rate(states, input_value)

class HillReaction:
    """ Typecasting for cHill instances. """
    def __init__(self, enz_rxn):
        """
        Args:
        enz_rxn (nevosim.EnzymaticReaction instance)
        """
        N = int(enz_rxn.active_substrates.size)
        active = enz_rxn.active_substrates.astype(np.int64)
        M = int(enz_rxn.active_inputs.size)
        active_inputs = enz_rxn.active_inputs.astype(np.int64)
        k = enz_rxn.k.astype(np.float64)
        k_m = np.float64(enz_rxn.k_m)
        n = np.float64(enz_rxn.n)
        baseline = np.float64(enz_rxn.baseline)
        propensity = enz_rxn.propensity.astype(np.float64)
        input_dependence = enz_rxn.input_dependence.astype(np.float64)
        rate_modifier = enz_rxn.rate_modifier.astype(np.float64)

        # get repressor objects
        if len(enz_rxn.repressors) > 0:
            reps = np.array([Repressor(Rep).c_rep for Rep in enz_rxn.repressors])
        else:
            reps = np.array([], dtype=np.obj2sctype(cRepressor))
        R = reps.size

        # get reaction object
        cdef cHill c_rxn
        c_rxn = cHill(N, active, M, active_inputs, R, k, k_m, n, baseline, propensity, input_dependence, reps, rate_modifier)
        self.c_rxn = c_rxn

    def get_rate(self, states, input_value):
        return self.c_rxn.get_rate(states, input_value)

class Repressor:
    """ Typecasting for cRepressor instances. """
    def __init__(self, py_repressor):
        """
        Args:
        py_repressor (nevosim.EnzymaticRepressor instance)
        """
        N = int(py_repressor.active_substrates.size)
        active = py_repressor.active_substrates.astype(np.int64)
        M = int(py_repressor.active_inputs.size)
        active_inputs = py_repressor.active_inputs.astype(np.int64)
        k_m = np.float64(py_repressor.k_m)
        n = np.float64(py_repressor.n)
        propensity = py_repressor.propensity.astype(np.float64)
        input_dependence = py_repressor.input_dependence.astype(np.float64)

        # get reaction object
        cdef cRepressor c_rep
        c_rep = cRepressor(N, active, M, active_inputs, k_m, n, propensity, input_dependence)
        self.c_rep = c_rep

    def get_occupancy(self, states, input_value):
        return self.c_rep.get_occupancy(states, input_value)

class SumReaction:
    """ Typecasting for cSumRxn instances. """
    def __init__(self, py_rxn):
        """
        Args:
        py_rxn (nevosim.SumReaction instance)
        """
        cdef cSumRxn c_rxn
        N = int(py_rxn.active_species.size)
        active = py_rxn.active_species.astype(np.int64)
        k = py_rxn.k.astype(np.float64)
        propensity = py_rxn.propensity.astype(np.int64)
        c_rxn = cSumRxn(N, active, k, propensity)
        self.c_rxn = c_rxn

    def get_rate(self, states):
        return self.c_rxn.get_rate(states)


class ProportionalController(SumReaction):
    def __init__(self, py_rxn):
        """ Args: py_rxn (ProportionalController instance) """
        cdef cProportionalController c_rxn
        N = int(py_rxn.active_species.size)
        active = py_rxn.active_species.astype(np.int64)
        k = py_rxn.k.astype(np.float64)
        propensity = py_rxn.propensity.astype(np.int64)
        c_rxn = cProportionalController(N, active, k, propensity)
        self.c_rxn = c_rxn

    def get_rate(self, states):
        return self.c_rxn.get_rate(states)


class IntegralController(SumReaction):
    def __init__(self, py_rxn):
        """ Args: py_rxn (IntegralController instance) """
        cdef cIntegralController c_rxn
        N = int(py_rxn.active_species.size)
        active = py_rxn.active_species.astype(np.int64)
        k = py_rxn.k.astype(np.float64)
        propensity = py_rxn.propensity.astype(np.int64)
        c_rxn = cIntegralController(N, active, k, propensity)
        self.c_rxn = c_rxn

    def get_rate(self, states):
        return self.c_rxn.get_rate(states)


class Coupling:
    """ Typecasting for cCoupling instances. """
    def __init__(self, rxn):
        """
        Args:
        rxn (nevosim.Coupling instance)
        """
        N = int(rxn.active_species.size)
        active = rxn.active_species.astype(np.int64)
        propensity = rxn.propensity.astype(int)
        k = rxn.k[0].astype(np.float64)
        a = np.float64(rxn.a)
        w = np.float64(rxn.w)

        # get repressor objects
        if len(rxn.repressors) > 0:
            reps = np.array([Repressor(Rep).c_rep for Rep in rxn.repressors])
        else:
            reps = np.array([], dtype=np.obj2sctype(cRepressor))
        R = reps.size

        # get reaction object
        cdef cCoupling c_rxn
        c_rxn = cCoupling(N, active, R, k, propensity, a, w, reps)
        self.c_rxn = c_rxn

    def get_rate(self, states, input_value):
        return self.c_rxn.get_rate(states, input_value)


class RateFunction:
    """ Python wrapper for c-based reaction rate computation."""
    def __init__(self, cell):
        """
        Args:
        rxns (list of reaction instances)
        """
        rxns, rxn_types = [], []

        for rxn in cell.reactions:
            if rxn.__class__ == pyCoupling:
                rxn_types.append(0)
                rxns.append(Coupling(rxn).c_rxn)
            elif rxn.__class__ == pyMassAction:
                rxn_types.append(1)
                rxns.append(MassActionReaction(rxn).c_rxn)
            elif rxn.__class__ == pyHill:
                rxn_types.append(2)
                rxns.append(HillReaction(rxn).c_rxn)
            elif rxn.__class__ == pyIControl:
                rxn_types.append(3)
                rxns.append(IntegralController(rxn).c_rxn)
            elif rxn.__class__ == pyPControl:
                rxn_types.append(4)
                rxns.append(ProportionalController(rxn).c_rxn)
            else:
                raise ValueError('{} reaction type not recognized.'.format(rxn.__class__.__name__))

        # get reaction map
        rxn_map = self.get_rxn_map(cell)
        input_map = self.get_input_map(cell)

        # set reaction lists
        rxns = np.array(rxns, dtype=np.obj2sctype(cActive))
        rxn_types = np.array(rxn_types, dtype=np.int64)

        # instantiate cReactions object
        self.cRateFunction = cRateFunction(rxns, rxn_types, rxn_map, input_map)


    @staticmethod
    def get_propensity_dict(network):
        """ Returns dictionary where keys are states and values are lists of  reaction indices whose propensities depend upon each state. """
        adict = {i: [] for i in range(network.nodes.size)}
        for (i, rxn) in enumerate(network.reactions):

            # store index of reaction i whose propensity depends on state s
            for s in rxn.propensity.nonzero()[0]:
                adict[s].append(i)

            # store index of reaction i whose repression depends on state s
            if type(rxn) in (pyHill, pyCoupling):
                for repressor in rxn.repressors:
                    for s in repressor.active_substrates:
                        adict[s].append(i)

        return adict

    @classmethod
    def get_rxn_map(cls, network):

        p_dict = cls.get_propensity_dict(network)

        adict = {i: [] for i in range(len(network.reactions))}
        for (i, rxn) in enumerate(network.reactions):
            list_of_lists = [p_dict[s] for s in rxn.stoichiometry.nonzero()[0]]
            alist = reduce(add, list_of_lists)
            adict[i].extend(alist)

        # remove duplicates
        for (k, v) in adict.items():
            adict[k] = list(set(v))

        return adict

    @staticmethod
    def get_input_map(network):
        adict = {i: [] for i in range(network.input_size)}
        for (j, rxn) in enumerate(network.reactions):
            if type(rxn) == Coupling or type(rxn) == SumReaction:
                continue
            for s in rxn.input_dependence.nonzero()[0]:
                adict[s].append(j)
        return adict

    def __call__(self, states, input_value, cumulative):
        """
        Get rate vector from cRateFunction.get_rxn_rate

        Args:
        states (np array, dtype=np.float64)
        input_value (np array, dtype=np.float64)
        """
        return self.get_rxn_rates(states, input_value, cumulative)

    def get_callable(self):
        """ Get callable cRateFunction.get_rxn_rates instance. """
        return self.cRateFunction.get_rxn_rates

    def get_rxn_rates(self, states, input_value, cumulative):
        """
        Call cRateFunction.get_rxn_rate.

        Args:
        states (np array, dtype=np.float64)
        input_value (np array, dtype=np.float64)
        """
        return self.cRateFunction.get_rxn_rates(states, input_value, cumulative)


