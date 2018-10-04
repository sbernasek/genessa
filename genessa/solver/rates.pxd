cimport numpy as np
from cpython.array cimport array

# cython intra-package imports
from ..kinetics.massaction cimport cMassAction
from ..kinetics.control cimport cPController, cIController
from ..kinetics.hill cimport cHill
from ..kinetics.marbach cimport cTranscription
from ..kinetics.coupling cimport cCoupling


cdef class cRates:

    # attributes
    cdef unsigned int M
    cdef double total_rate

    # attributes requiring memory
    cdef unsigned int *rxn_types
    cdef unsigned int *rxn_keys
    cdef double *rates

    # reaction object attributes
    cdef cCoupling coupling
    cdef cMassAction massaction
    cdef cTranscription transcription
    cdef cHill hill
    cdef cIController icontrol
    cdef cPController pcontrol
    cdef cRxnMap rxn_map, input_map, ptb_map

    # methods
    cdef void allocate_memory(self)

    cdef double evaluate_rxn_rate(self,
        unsigned int rxn,
        unsigned int *states,
        double *inputs,
        double *cumulative) nogil

    cdef void update_rxn_rate(self,
        unsigned int rxn,
        unsigned int *states,
        double *inputs,
        double *cumulative) nogil

    cdef void apply_perturbation(self,
        unsigned int rxn,
        double ptb) nogil

    cdef void update_after_input_change(self,
        unsigned int *states,
        double *inputs,
        double *cumulative,
        unsigned int dim) nogil

    cdef void update_after_rxn_fired(self,
        unsigned int *states,
        double *inputs,
        double *cumulative,
        unsigned int fired) nogil

    cdef void update_all(self,
        unsigned int *states,
        double *inputs,
        double *cumulative) nogil

    cpdef array c_evaluate_rxn_rates(self,
        np.ndarray[np.float64_t, ndim=1, mode='c'] states,
        array inputs,
        np.ndarray[np.float64_t, ndim=1, mode='c'] cumulative)


# define mappable function names
ctypedef void (*cSetRate)(cRates,
                          unsigned int,
                          unsigned int*,
                          double*,
                          double*) nogil

ctypedef void (*cPerturb)(cRates,
                          unsigned int,
                          double) nogil


cdef class cRxnMap:

    # attributes
    cdef array ind, lengths, values

    # methods
    cdef void app(self,
                  cRates rf,
                  unsigned int key,
                  cSetRate f,
                  unsigned int *states,
                  double *inputs,
                  double *cumulative) nogil

    cdef void app_ptb(self,
                      cRates rf,
                      unsigned int key,
                      cPerturb f,
                      double ptb) nogil
