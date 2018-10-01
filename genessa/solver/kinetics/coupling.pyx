# cython external imports
cimport numpy as np
from cpython.array cimport array

# python external imports
import numpy as np
from array import array
from functools import reduce
from operator import add

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
                                     array states) nogil:
        return self.get_species_activity_sum(rep, states)

    cdef void set_occupancy(self,
                            unsigned int rep,
                            array states) nogil:
        """ Get occupancy by specified repressor. """
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
                     array states,
                     unsigned int fired) nogil:
        """ Update occupancies for repressors that have changed. """
        self.rxn_map.app_rep(self, fired, self.set_occupancy, states)

    cdef double cget_occupancy(self,
                               array states,
                               unsigned int rep) nogil:
        """ Get rate of specified reaction """

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
            value = states.data.as_doubles[ind]
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
                                 array states) nogil:
        """ Integrate all repressor activity to determine availability. """

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

    cdef double update(self,
                       unsigned int rxn,
                       array states) nogil:
        """ Update rate of specified reaction. """
        cdef double coupling_strength
        cdef double rate

        # compute rate and apply repressors
        coupling_strength = self.activity.data.as_ints[rxn] * self.weight.data.as_doubles[rxn]
        rate = (self.k.data.as_doubles[rxn] + coupling_strength) * self.get_availability(rxn, states)

        # # update rate
        if rate < 0:
            rate = 0

        return rate

    cdef void update_activity(self,
                              unsigned int edge,
                              array states) nogil:

        """ Get occupancy by specified repressor. """
        cdef int weight
        cdef unsigned int state_ind, state
        cdef int old_edge = self.edges.data.as_ints[edge]
        cdef int new_edge, activity, rxn

        # get new edge value
        weight = <int>self.species_dependence.data.as_doubles[edge]
        state_ind = self.species.data.as_uints[edge]
        new_edge = weight * states.data.as_uints[state_ind]

        # update rxn activity
        rxn = self.edge_to_rxn.data.as_uints[edge]
        activity = self.activity.data.as_ints[rxn]
        self.activity.data.as_ints[rxn] = activity + (new_edge - old_edge)

        # update edge
        self.edges.data.as_uints[edge] = new_edge

    cdef void update_activities(self,
                                array states,
                                unsigned int fired) nogil:
        """ Update occupancies for repressors that have changed. """
        self.rxn_map.app(self, fired, self.update_activity, states)

    cdef double cget_rate(self,
                          unsigned int rxn,
                          array states) nogil:
        """ Get rate of specified reaction """

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
            value = states.data.as_doubles[ind]
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
                      array states) nogil:
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
                   array states) nogil:
        cdef unsigned int count, edge
        cdef unsigned int length = self.lengths.data.as_uints[key]
        cdef unsigned int index = self.ind.data.as_uints[key]

        if length != 0:
            for count in xrange(length):
                edge = self.values.data.as_uints[index]
                f(coupling_obj, edge, states)
                index += 1
