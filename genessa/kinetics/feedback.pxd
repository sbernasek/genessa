from cpython.array cimport array

# cython intra-package imports
from .massaction cimport cMassAction


cdef class cFeedBack(cMassAction):

    # attributes
    cdef array targets_ind, n_targets, targets

    # methods
    @staticmethod
    cdef cFeedBack get_blank_cFeedBack()

    @staticmethod
    cdef cFeedBack from_list(list rxns)
