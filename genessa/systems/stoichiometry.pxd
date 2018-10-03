
cdef class cStoichiometry:

    # attributes
    cdef unsigned int M
    cdef unsigned int Nc
    cdef unsigned int *index
    cdef unsigned int *lengths
    cdef unsigned int *species
    cdef int *coefficients

    # methods
    cdef void allocate_memory(self)
