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
from .coupling cimport cSDRepressor, cCoupling, cRxnMap
from .base cimport cSpeciesDependent


cdef class cSDRepressor(cSpeciesDependent):

    def __init__(self,
                 unsigned int M,
                 double[:] k_m,
                 double[:] n,
                 unsigned int[:] species_ind,
                 unsigned int[:] species,
                 double[:] species_dependence,
                 dict rxn_map):

        # add input/species dependence
        vmax = array('d', np.ones(M, dtype=np.uint32))
        cSpeciesDependent.__init__(self, M, vmax, species_ind, species, species_dependence)
        self.k_m = array('d', k_m)
        self.n = array('d', n)
        self.rxn_map = cRxnMap(rxn_map)
        self.occupancies = array('d', np.zeros(M, dtype=np.float64))

    @staticmethod
    cdef cSDRepressor get_blank_cSDRepressor(unsigned int M):
        """ Returns blank cSDRepressor instance. """
        cdef np.ndarray xf = np.zeros(1, dtype=np.float64)
        cdef np.ndarray xl = np.zeros(1, dtype=np.uint32)
        cdef dict rxn_map = {i: [] for i in xrange(M)}
        return cSDRepressor(0, xf, xf, xl, xl, xf, rxn_map)

    @staticmethod
    cdef cSDRepressor from_list(list rxns,
                                dict rxn_map):
        """ Instantiate from list of reactions. """
        cdef unsigned int M
        cdef list reps
        cdef np.ndarray k_m, n
        cdef np.ndarray species_ind, species, species_dependence

        # get repressors
        reps = reduce(add, [rxn.repressors for rxn in rxns])

        # if there are no repressors, return empty arrays
        M = len(reps)
        if M == 0:
            return cSDRepressor.get_blank_cSDRepressor(len(rxn_map))

        # get parameters
        k_m = np.array([rxn.k_m for rxn in reps], dtype=np.float64)
        n = np.array([rxn.n for rxn in reps], dtype=np.float64)

        # get species dependence
        species_ind = np.cumsum([0]+[rxn.num_active_substrates for rxn in reps]).astype(np.uint32)
        species = np.hstack([rxn.active_substrates for rxn in reps]).astype(np.uint32)
        species_dependence = np.hstack([rxn._propensity for rxn in reps])

        return cSDRepressor(M, k_m, n, species_ind, species, species_dependence, rxn_map)

    cdef double get_species_activity(self,
                                     unsigned int rep,
                                     unsigned int *states) nogil:
        return self.get_species_activity_sum(rep, states)

    cdef void set_occupancy(self,
                            unsigned int rep,
                            unsigned int *states) nogil:
        """
        Evaluate and set occupancy by specified repressor.

        Args:

            rep (unsigned int) - repressor index

            states (unsigned int*) - state values

        """
        cdef double activity
        cdef double k_m, n
        cdef double occupancy

        # compute species activities
        activity = self.get_species_activity(rep, states)

        # compute occupancy
        k_m = self.k_m.data.as_doubles[rep]
        n = self.n.data.as_doubles[rep]
        occupancy = (activity**n) / ( (activity**n) + (k_m**n) )

        self.occupancies.data.as_doubles[rep] = occupancy

    cdef void update(self,
                     unsigned int *states,
                     unsigned int fired) nogil:
        """
        Update occupancies following a reaction firing.

        Args:

            states (unsigned int*) - state values

            fired (unsigned int) - index of fired reaction

        """
        self.rxn_map.app_rep(self, fired, self.set_occupancy, states)

    cdef double cget_occupancy(self,
                               double* states,
                               unsigned int rep) nogil:
        """
        Evaluate and return occupancy by specified repressor.

        Args:

            states (double*) - state values

            rep (unsigned int) - repressor index

        Returns:

            occupancy (double)

        """

        cdef unsigned int count, ind
        cdef double coefficient, value
        cdef unsigned int index
        cdef unsigned int N = self.n_active_species.data.as_uints[rep]
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

        occupancy = (activity**n) / ( (activity**n) + (k_m**n) )

        return occupancy


cdef class cCoupling(cSpeciesDependent):
    def __init__(self,
                 unsigned int M,
                 double[:] k,
                 double[:] weight,

                 unsigned int[:] species_ind,
                 unsigned int[:] species,
                 double[:] species_dependence,

                 cSDRepressor repressor_obj,
                 unsigned int[:] repressors_ind,

                 dict rxn_map):

        # add input/species dependence
        cSpeciesDependent.__init__(self, M, k, species_ind, species, species_dependence)

        N = np.array(self.n_active_species)
        self.k = array('d', k)
        self.weight = array('d', weight)

        # add repressor data
        self.rep_obj = repressor_obj
        self.repressors_ind = array('I', repressors_ind)
        self.n_repressors = array('I', np.diff(repressors_ind).astype(np.uint32))

        # add edge data
        self.rxn_map = cRxnMap(rxn_map)
        self.edges = array('l', np.zeros(len(species), dtype=np.int64))
        self.edge_to_rxn = array('I', np.repeat(np.arange(M), self.n_active_species).astype(np.uint32))
        self.activity = array('l', np.zeros(M, dtype=np.int64))

    @staticmethod
    cdef cCoupling get_blank_cCoupling(unsigned int M):
        """ Returns blank cCoupling object. """
        cdef np.ndarray xf = np.zeros(1, dtype=np.float64)
        cdef np.ndarray xl = np.zeros(1, dtype=np.uint32)
        cdef cSDRepressor rep = cSDRepressor.get_blank_cSDRepressor(M)
        cdef unsigned int i
        cdef dict rxn_map = {i: [] for i in xrange(M)}
        return cCoupling(0, xf, xf, xl, xl, xf, rep, xl, rxn_map)

    @staticmethod
    cdef cCoupling from_list(list rxns,
                             dict edge_map,
                             dict repressor_map):
        """ Instantiate from list of reactions. """
        cdef unsigned int M
        cdef np.ndarray k, a, w, N, weight
        cdef np.ndarray species_ind, species, species_dependence
        cdef cSDRepressor repressor_obj
        cdef np.ndarray repressors_ind

        # if no reactions of this type, add blank
        M = len(rxns)
        if M == 0:
            return cCoupling.get_blank_cCoupling(len(edge_map))

        # add species dependence
        species_ind = np.cumsum([0]+[rxn.num_active_species for rxn in rxns]).astype(np.uint32)
        species = np.hstack([rxn.active_species for rxn in rxns]).astype(np.uint32)
        species_dependence = np.hstack([rxn._propensity for rxn in rxns])
        species_dependence = species_dependence.astype(np.float64)

        # get parameters
        k = np.array([rxn.k[0] for rxn in rxns], dtype=np.float64)
        a = np.array([rxn.a for rxn in rxns], dtype=np.float64)
        w = np.array([rxn.w for rxn in rxns], dtype=np.float64)
        N = np.diff(species_ind).astype(np.uint32)
        weight = (a*w/(1+w*(N-1)))

        # add repressors
        repressor_obj = cSDRepressor.from_list(rxns, repressor_map)
        repressors_ind = np.cumsum([0]+[len(rxn.repressors) for rxn in rxns]).astype(np.uint32)

        return cCoupling(M, k, weight, species_ind, species, species_dependence, repressor_obj, repressors_ind, edge_map)

    cdef double get_availability(self,
                                 unsigned int rxn,
                                 unsigned int *states) nogil:
        """
        Integrate repressor occupancies to determine overall availability for a specified promoter.

        Args:

            rxn (unsigned int) - index of transcription reaction

            states (unsigned int*) - state values (not used)

        Returns:

            availability (double) - total promoter availability

        """

        cdef unsigned int count
        cdef double occupancy
        cdef double availability = 1
        cdef unsigned int index = self.repressors_ind.data.as_uints[rxn]
        cdef unsigned int num_repressors = self.n_repressors.data.as_uints[rxn]

        # integrate repressor occupancies (multiplicative)
        for count in xrange(num_repressors):
            occupancy = self.rep_obj.occupancies.data.as_doubles[index]
            availability *= (1-occupancy)
            index += 1

        return availability

    cdef double evaluate_rxn_rate(self,
                                   unsigned int rxn,
                                   unsigned int *states) nogil:
        """
        Evaluates and returns rate for specified reaction.

        Args:

            rxn (unsigned int) - index of reaction

            states (unsigned int*) - state values

        Returns:

            rate (double) - reaction rate

        """
        cdef double coupling_strength
        cdef double rate

        # compute rate and apply repressors
        coupling_strength = self.activity.data.as_ints[rxn] * self.weight.data.as_doubles[rxn]
        rate = (self.k.data.as_doubles[rxn] + coupling_strength) * self.get_availability(rxn, states)

        # # update rate
        if rate < 0:
            rate = 0

        return rate

    cdef void update_edge(self,
                              unsigned int edge,
                              unsigned int *states) nogil:
        """
        Update edge weight.

        Args:

            edge (unsigned int) - index of edge

            states (unsigned int*) - state values

        """
        cdef int weight
        cdef unsigned int state_ind, state
        cdef int old_edge = self.edges.data.as_ints[edge]
        cdef int new_edge, activity, rxn

        # get new edge value
        weight = <int>self.species_dependence.data.as_doubles[edge]
        state_ind = self.species.data.as_uints[edge]
        new_edge = weight * states[state_ind]

        # update rxn activity
        rxn = self.edge_to_rxn.data.as_uints[edge]
        activity = self.activity.data.as_ints[rxn]
        self.activity.data.as_ints[rxn] = activity + (new_edge - old_edge)

        # update edge
        self.edges.data.as_uints[edge] = new_edge

    cdef void update_edges(self,
                                unsigned int *states,
                                unsigned int fired) nogil:
        """
        Update edge weights following a reaction event.

        Args:

            states (unsigned int*) - state values

            fired (unsigned int) - fired reaction

        """
        self.rxn_map.app(self, fired, self.update_edge, states)

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

        cdef unsigned int count, ind
        cdef double coefficient, value
        cdef unsigned int index
        cdef unsigned int N = self.n_active_species.data.as_uints[rxn]
        cdef unsigned int R = self.n_repressors.data.as_uints[rxn]
        cdef double k = self.k.data.as_doubles[rxn]
        cdef double weight = self.weight.data.as_doubles[rxn]
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

        # compute rate
        rate = k + (weight * activity)

        # integrate repressor occupancies (multiplicative)
        index = self.repressors_ind.data.as_uints[rxn]
        for count in xrange(R):
            occupancy = self.rep_obj.cget_occupancy(states, index)
            rate *= (1-occupancy)
            index += 1

        return rate


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

    cdef void app_rep(self,
                      cSDRepressor rep_obj,
                      unsigned int key,
                      cSetOccupancy f,
                      unsigned int *states) nogil:
        cdef unsigned int count, rep
        cdef unsigned int length = self.lengths.data.as_uints[key]
        cdef unsigned int index = self.ind.data.as_uints[key]

        if length != 0:
            for count in xrange(length):
                rep = self.values.data.as_uints[index]
                f(rep_obj, rep, states)
                index += 1

    cdef void app(self,
                   cCoupling coupling_obj,
                   unsigned int key,
                   cSetEdge f,
                   unsigned int *states) nogil:
        cdef unsigned int count, edge
        cdef unsigned int length = self.lengths.data.as_uints[key]
        cdef unsigned int index = self.ind.data.as_uints[key]

        if length != 0:
            for count in xrange(length):
                edge = self.values.data.as_uints[index]
                f(coupling_obj, edge, states)
                index += 1


#=============================== PYTHON CODE ==================================


class Coupling:

    def __init__(self,
                 stoichiometry=None,
                 propensity=None,
                 k=1,
                 a=1,
                 w=1,
                 repressors=None,
                 rxn_type='coupling',
                 parameters=None):
        """
        Class describes a single coupling pathway.

        Args:

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
        """

        Expand stoichiometry and propensity vectors.

        Args:

            shift (int) - number of positions appended to beginning

        Returns:

            rxn (Coupling) - updated reaction

        """

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

    def evaluate_rate(self, states, input_state, **kwargs):
        """
        Returns rate for given state and input values.

        Args:

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
            unoccupied_sites = reduce(mul, [1-repressor.get_occupancy(states, input_state) for repressor in self.repressors])

        # get overall rate
        rate = unoccupied_sites * rate

        return rate
