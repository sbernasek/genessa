# cython: profile=True

from rxndiffusion.reactions import Reaction as pyMassAction
from rxndiffusion.reactions import EnzymaticReaction as pyHill
from rxndiffusion.reactions import IntegralController as pyIControl
from rxndiffusion.reactions import ProportionalController as pyPControl

from rxns cimport cMassAction, cRepressor, cHill, cRateFunction
import numpy as np
cimport numpy as np
cimport cython

from cpython.array cimport array
from array import array

"""
IDEAS: can pre-specify indices not participating in reactions to drastically cut down iterations...
"""


cdef class cActive:
    def __init__(self, int N, long[:] active):
        self.N = N
        self.active = active


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


cdef class cMassAction(cActiveInputs):

    def __init__(self, int N, long[:] active, int M, long[:] active_inputs,double k, long[:] propensity, long[:] input_dependence):
        cActiveInputs.__init__(self, N, active, M, active_inputs)
        self.k = k
        self.propensity = propensity
        self.input_dependence = input_dependence

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double get_rate(self, array states, array input_value) nogil:
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
                 double[:] propensity,
                 double[:] input_dependence,
                 cRepressor[:] c_reps):
        cActiveInputs.__init__(self, N, active, M, active_inputs)
        self.R = R
        self.k = k
        self.k_m = k_m
        self.n = n
        self.baseline = baseline
        self.propensity = propensity
        self.input_dependence = input_dependence
        self.repressors = c_reps

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double get_rate(self, array states, array input_value):
        """ Get rate for Hill reaction. """

        cdef double rate = self.k
        cdef int i, j, k
        cdef int s_ind, i_ind
        cdef double n
        cdef double activity = 0
        cdef double available = 1
        cdef cRepressor repressor

        # compute activity (could pre-assign indices...)
        for i in xrange(self.N):
            s_ind = self.active[i]
            activity += (states.data.as_longs[s_ind] * self.propensity[s_ind])
        for j in xrange(self.M):
            i_ind = self.active_inputs[j]
            activity += input_value.data.as_doubles[i_ind] * self.input_dependence[i_ind]

        # compute repressor occupancy
        for k in xrange(self.R):
            repressor = self.repressors[k]
            available *= (1 - repressor.get_occupancy(states, input_value))

        # compute overall rate
        rate *= available * ((activity**self.n)/(activity**self.n + self.k_m**self.n) + self.baseline)

        return rate


cdef class cRateFunction:

    def __init__(self, cMassAction[:] massaction, cHill[:] hill, cSumRxn[:] icontrol, cSumRxn[:] pcontrol):

        # store reactions
        self.massaction = massaction
        self.hill = hill
        self.icontrol = icontrol
        self.pcontrol = pcontrol

        # determine number of reactions of each type
        self.n_massaction = len(massaction)
        self.n_hill = len(hill)
        self.n_icontrol = len(icontrol)
        self.n_pcontrol = len(pcontrol)

        # initialize rates for all reactions
        n = self.n_massaction + self.n_hill + self.n_icontrol + self.n_pcontrol
        self.rates = array('d', np.zeros(n, dtype=np.float64))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef array get_rxn_rates(self, array states, array input_value, array cumulative):
        cdef cMassAction rxn
        cdef cHill hill_rxn
        cdef cSumRxn controller
        cdef double rate
        cdef int index = 0
        cdef int i

        # EnzymaticReaction class
        for i in xrange(self.n_hill):
            hill_rxn = self.hill[i]
            rate = hill_rxn.get_rate(states, input_value)
            self.rates.data.as_doubles[index] = rate
            index += 1

        # IntegralController class
        for i in xrange(self.n_icontrol):
            controller = self.icontrol[i]
            rate = controller.get_rate_doubles(cumulative)
            self.rates.data.as_doubles[index] = rate
            index += 1

        # ProportionalController class
        for i in xrange(self.n_pcontrol):
            controller = self.pcontrol[i]
            rate = controller.get_rate_longs(states)
            self.rates.data.as_doubles[index] = rate
            index += 1

        # Reaction class
        for i in xrange(self.n_massaction):
            rxn = self.massaction[i]
            rate = rxn.get_rate(states, input_value)
            self.rates.data.as_doubles[index] = rate
            index += 1

        return self.rates


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
        k = py_rxn.rate_constant.astype(np.float64)
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
        k = enz_rxn.rate_constant.astype(np.float64)
        k_m = np.float64(enz_rxn.k_m)
        n = np.float64(enz_rxn.n)
        baseline = np.float64(enz_rxn.baseline)
        propensity = enz_rxn.propensity.astype(np.float64)
        input_dependence = enz_rxn.input_dependence.astype(np.float64)

        # get repressor objects
        if len(enz_rxn.repressors) > 0:
            reps = np.array([Repressor(Rep).c_rep for Rep in enz_rxn.repressors])
        else:
            reps = np.array([], dtype=np.obj2sctype(cRepressor))
        R = reps.size

        # get reaction object
        cdef cHill c_rxn
        c_rxn = cHill(N, active, M, active_inputs, R, k, k_m, n, baseline, propensity, input_dependence, reps)
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
        k = py_rxn.rate_constant.astype(np.float64)
        propensity = py_rxn.propensity.astype(np.int64)
        c_rxn = cSumRxn(N, active, k, propensity)
        self.c_rxn = c_rxn

    def get_rate(self, states):
        return self.c_rxn.get_rate(states)

class RateFunction:
    """ Python wrapper for c-based reaction rate computation."""
    def __init__(self, rxns):
        """
        Args:
        rxns (list of reaction instances)
        """

        # sort reactions alphabetically by class name to match stoichiometry
        massaction, hill, pcontrol, icontrol = [], [], [], []
        for rxn in rxns:
            if rxn.__class__ == pyMassAction:
                massaction.append(MassActionReaction(rxn).c_rxn)
            elif rxn.__class__ == pyHill:
                hill.append(HillReaction(rxn).c_rxn)
            elif rxn.__class__ == pyIControl:
                icontrol.append(SumReaction(rxn).c_rxn)
            elif rxn.__class__ == pyPControl:
                pcontrol.append(SumReaction(rxn).c_rxn)
            else:
                raise ValueError('{} reaction type not recognized.'.format(rxn.__class__.__name__))

        # set reaction lists
        massaction = np.array(massaction, dtype=np.obj2sctype(cMassAction))
        hill = np.array(hill, dtype=np.obj2sctype(cHill))
        icontrol = np.array(icontrol, dtype=np.obj2sctype(cSumRxn))
        pcontrol = np.array(pcontrol, dtype=np.obj2sctype(cSumRxn))

        # instantiate cReactions object
        self.cRateFunction = cRateFunction(massaction, hill, icontrol, pcontrol)

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


