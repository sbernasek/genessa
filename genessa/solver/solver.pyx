
cdef class cSystem:
    """
    Cython class for running a stochastic simulation.

    Attributes:

        system (cSystem) - system of nodes and reactions

        states (unsigned int*) - state values for each node

        inputs (double*) - input values for each input channel

        cumulative (double*) - integrated values for each node

        integrate (boolean int) - boolean flag for running integrator

        rxn_order (unsigned int*) - order of reaction list

        rstates (array[unsigned int]) - recorded state values for each node

    """

    cdef cSystem system
    cdef bint integrate
    cdef bint null_input
    cdef array rstates
    cdef unsigned int* rxn_order
    cdef unsigned int* states
    cdef double* inputs
    cdef double* cumulative

    def __cinit__(self, cSystem system):
        """
        Instantiate stochastic simulation.

        Args:

            system (cSystem) - system of nodes and reactions

        """

        cdef unsigned int i

        # add network
        self.system = system

        # set flags
        if system.R.icontrol.M == 0:
            self.integrate = 0
        else:
            self.integrate = 1

        # initialize array for regular states
        self.rstates = array('I', np.zeros(system.N, dtype=np.uint32))

        # allocate and populate memory for simulation variables
        self.allocate_memory()
        for i in xrange(system.N):
            self.states[i] = 0
            self.cumulative[i] = 0.
        for i in xrange(system.I):
            self.inputs[i] = 0.
        for i in xrange(system.M):
            self.rxn_order[i] = i

    def __dealloc__(self):
        """ Deallocate memory from all array attributes. """
        PyMem_Free(self.rxn_order)
        PyMem_Free(self.states)
        PyMem_Free(self.inputs)
        PyMem_Free(self.cumulative)

    cdef void allocate_memory(self):
        """
        Allocate memory for all array attributes.

        Note:

            - memory allocation requires GIL

        """

        cdef unsigned int size

        # allocate memory for reaction sort indices
        size = self.system.M * sizeof(unsigned int)
        self.rxn_order = <unsigned int*> PyMem_Malloc(size)
        if not self.rxn_order:
            raise MemoryError('Reaction order memory block not allocated.')

        # allocate memory for states vector
        size = self.system.N * sizeof(unsigned int)
        self.states = <unsigned int*> PyMem_Malloc(size)
        if not self.states:
            raise MemoryError('States memory block not allocated.')

        # allocate memory for input values vector
        size = self.system.I * sizeof(double)
        self.inputs = <double*> PyMem_Malloc(size)
        if not self.inputs:
            raise MemoryError('Inputs memory block not allocated.')

        # allocate memory for integrator values vector
        size = self.system.N * sizeof(double)
        self.cumulative = <double*> PyMem_Malloc(size)
        if not self.cumulative:
            raise MemoryError('Integrator memory block not allocated.')

    @staticmethod
    def from_network(network):
        """
        Instantiate from python network.

        Args:

            network (Network)

        Returns:

            c_ssa (cSSA)

        """
        return cSSA(cSystem.from_network(network))





    @staticmethod
    def from_network(network):
        """
        Instantiate from python Network.

        Args:

            network (Network)

        Returns:

            c_system (cSystem)

        """

        # sort rxns and compile stoichiometry
        network.sort_rxns()
        network.resize_inputs()
        network.compile_stoichiometry()

        # typecast network features
        N = network.N
        M = network.M
        I = network.I

        # get cythonized rate function and network
        S = cStoichiometry.from_array(network.stoichiometry)
        R = Rates.compile_c_rate_function(network)

        return cSystem(N, M, I, S, R)

    cdef double* get_rxn_rates(self):
        """ Returns current reaction rates. """
        return self.R.rates

    cdef double get_total_rxn_rate(self) nogil:
        """ Returns current total reaction rate. """
        return self.R.total_rate

    cdef array c_evaluate_species_rates(self,
                                       array states,
                                       array inputs,
                                       array cumulative):
        """
        Evaluate rate of change for all species.

        Args:

            states (array[double]) - state values

            inputs (array[double]) - input values

            cumulative (array[double]) - integrator values

        Returns:

            rates (array[double]) - species rates, e.g. dX/dt

        """

        cdef unsigned int rxn
        cdef double rxn_rate
        cdef unsigned int N, index, count, species
        cdef int coefficient
        cdef array rxn_rates

        # instantiate array of zeros
        cdef array rates = array('d', self.N*[0.])

        # evaluate reaction rates
        rxn_rates = self.R.c_evaluate_rxn_rates(states, inputs, cumulative)

        # for each reaction
        for rxn in xrange(self.M):
            rxn_rate = rxn_rates.data.as_doubles[rxn]
            N = self.S.lengths[rxn]
            index = self.S.index[rxn]

            # update each active state
            for count in xrange(N):
                species = self.S.species[index]
                coefficient = self.S.coefficients[index]
                rates.data.as_doubles[species] += (coefficient * rxn_rate)
                index += 1

        return rates
