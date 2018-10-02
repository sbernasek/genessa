from cpython.array cimport array

# cython intra-package imports
from ..kinetics.massaction cimport cMassAction
from ..kinetics.control cimport cPController, cIController
from ..kinetics.hill cimport cHill
from ..kinetics.marbach cimport cTranscription
from ..kinetics.coupling cimport cCoupling


cdef class cRates:

    # attributes
    cdef cCoupling coupling
    cdef cMassAction massaction
    cdef cTranscription transcription
    cdef cHill hill
    cdef cIController icontrol
    cdef cPController pcontrol
    cdef unsigned int M
    cdef array rxn_types, rxn_keys, rates
    cdef double total_rate
    cdef cRxnMap rxn_map, input_map, ptb_map

    # methods
    cdef double evaluate(self, unsigned int rxn, array states, array inputs, array cumul) nogil
    cdef void set_rate(self, unsigned int rxn, array states, array inputs, array cumul) nogil
    cdef void apply_perturbation(self, unsigned int rxn, double ptb) nogil
    cdef void update_input(self, array states, array inputs, array cumul, unsigned int dim) nogil
    cdef void update(self, array states, array inputs, array cumul, unsigned int fired) nogil
    cdef void update_all(self, array states, array inputs, array cumul) nogil
    cpdef void cupdate(self, array states, array inputs, array cumul) with gil
    cpdef array evaluate_rxn_rates(self, array states, array inputs, array cumul)


# define mappable function names
ctypedef void (*cSetRate)(cRates, unsigned int, array, array, array) nogil
ctypedef void (*cPerturb)(cRates, unsigned int, double) nogil


cdef class cRxnMap:

    # attributes
    cdef array ind, lengths, values

    # methods
    cdef void app(self, cRates rf, unsigned int key, cSetRate f, array states, array inputs, array cumul) nogil
    cdef void app_ptb(self, cRates rf, unsigned int key, cPerturb f, double ptb) nogil
