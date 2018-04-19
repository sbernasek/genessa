__author__ = 'Sebi'

from .timeseries import TimeSeries
from .parameters import conditions
from .solver.ssa import choose_rxn, float_sum, min_float, min_int
from .solver.rxns import RateFunction as RateFunction
from .solver.ssa import Solver as cySolver
from copy import deepcopy
import numpy as np
import random as rd
import scipy.integrate
import scipy.optimize
import itertools
import functools
import warnings
import matplotlib.pyplot as plt
from scipy import interpolate
import array
import types
#warnings.filterwarnings('error')


class Simulation:
    """
    Class defines a simulation procedure.
    """

    def __init__(self, network, condition=None):
        """
        Parameters:
            network (Network object)
            condition (str) - environmental conditions affecting rate. either None, 'cold', 'hot', 'diabetic', or 'minute'
        """
        self.network = deepcopy(network)
        self.network.sort_rxns()
        self.num_reactions = len(network.reactions)
        self.network.compile_stoichiometry()

        # apply any rate scaling conditions
        if condition is not None:
            if condition not in conditions.keys():
                raise ValueError('Condition not recognized.')
            else:
                self.apply_rate_scaling(condition)

    def apply_rate_scaling(self, condition):
        """
        Applies rate scaling to all condition-sensitive reactions:

        Parameters:
            condition (str) - environmental conditions affecting rate. either None, 'cold', 'hot', 'diabetic', or 'minute'
        """

        # get rate scaling factors
        rate_scaling = conditions[condition]

        # apply scaling factors
        if 'temperature' in rate_scaling.keys():
            for rxn in self.network.reactions:
                rxn.rate_constant *= rate_scaling['temperature']**rxn.temperature_sensitive

        if 'metabolic_rate' in rate_scaling.keys():
            for rxn in self.network.reactions:
                rxn.rate_constant *= rate_scaling['metabolic_rate']**rxn.atp_sensitive

        if 'translation_capacity' in rate_scaling.keys():
            for rxn in self.network.reactions:
                rxn.rate_constant *= rate_scaling['translation_capacity']**rxn.ribosome_sensitive

    @staticmethod
    def choose_rxn(rates):
        """ Call cython implementation of choose_rxn. """
        return choose_rxn(array.array('d', rates), rates.size)

    def get_rxn_rates(self, state, input_value, discrete=True):
        """
        Computes rate of each reaction.

        Parameters:
            state (np array) - Nx1 vector of current state
            input_value (np array) - current value(s) of input
            discrete (bool) - if True, use discrete propensity function
        Returns:
            rxn_rates (np array) - array of M current reaction rates
        """

        # initialize rate vector
        rxn_rates = np.empty(self.num_reactions)

        # compute rate of each reaction
        for i, rxn in enumerate(self.network.reactions):
            rate = rxn.get_rate(state, input_value, discrete=discrete)
            rxn_rates[i] = rate

        return rxn_rates

    def get_species_rates(self, t, state, input_function=None, discrete=True):
        """
        Computes net rate of change of each chemical species. This serves as the derivative function for the ODE solver.

        Parameters:
            t (float) - current time
            state (np array) - array of current state values
            input_function (function) - returns input value(s) for a given timepoint
            discrete (bool) - if True, use discrete propensity function

        Returns:
            species_rates (np array) - net rate of change of each species (N x 1)
        """

        # compute input value
        if input_function is not None:
            input_value = input_function(t)
        else:
            input_value = np.zeros(self.network.input_size)

        # compute reaction and species rates
        rxn_rates = self.get_rxn_rates(state, input_value, discrete=discrete)
        species_rates = np.dot(self.network.stoichiometry, rxn_rates)

        return species_rates

    def simulate(self, ic=None, input_function=None, integrator_ic=None, method='cy_hybrid', dt=1, duration=100, discrete=True, plot=False, ax=None):
        """
        Simulates dynamic system using Scipy's ode object.

        Parameters:
            ic (np array) - initial conditions, defaults to zero
            input_function (function) - returns input value(s) for each time
            integrator_ic (np array) - initial condition for integrator
            method (str) - solution method
            dt (float) - time step
            duration (float) - simulation end time

            discrete (bool) - if True, use discrete propensity function
            plot (bool) - if True, plot dynamics
            ax (axes object) - if None, create one

        Returns:
            times (np array) - time points corresponding to state values (1 by t)
            states (np array) - matrix of state values at each time point (N by t)
        """

        # sort reactions
        self.network.sort_rxns()

        # compile reindexing key, stoichiometric matrix
        self.network.compile_stoichiometry()

        # if no initial condition is provided, assume all states are initially zero
        if ic is None:
            ic = np.zeros(self.network.nodes.size, dtype=np.int64)
        elif type(ic) == list or type(ic) == tuple:
            ic = np.array(ic, dtype=np.int64)
        if len(ic) != self.network.nodes.size:
            raise RuntimeError('Initial condition dimensions are inconsistent with system dimensionality.')

        # select solutions algorithm (for deterministic solutions assume continuous state space)
        if method == 'deterministic':
            discrete = False
            times, states = self.run_deterministic_solver(ic, duration=duration, dt=dt, input_function=input_function, discrete=discrete)

        elif method == 'pd-leaping':
            times, states = self.run_tau_leaping_solver(ic, dt=dt, duration=duration, input_function=input_function, discrete=discrete)

        elif method == 'bd-leaping':
            times, states = self.run_binomial_leaping_solver(ic, dt=dt, duration=duration, input_function=input_function, discrete=discrete)

        elif method == 'ssa':
            times, states = self.run_pure_ssa_solver(ic, dt=dt, duration=duration, input_function=input_function, discrete=discrete)

        elif method == 'hybrid-ssa':
            times, states = self.run_hybrid_ssa_solver(ic, dt=dt, duration=duration, input_function=input_function, discrete=discrete)

        elif method == 'cy_hybrid':
            py_solver = cySolver(self.network)
            times, states = py_solver.solve(ic=ic, input_function=input_function, integrator_ic=integrator_ic, dt=dt, duration=duration)

        elif method == 'cy_ssa':
            py_solver = cySolver(self.network)
            times, states = py_solver.solve(ic=ic, input_function=input_function, integrator_ic=integrator_ic, dt=dt, duration=duration, use_pure_ssa=True)

        else:
            print('Method not recognized.')
            return None, None

        # plot output
        if plot is True:
            if ax is None:
                fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(times, states[self.network.output_node, :], linewidth=3)
            ax.set_xlabel('time (hr)', fontsize=15)
            ax.set_ylabel('output', fontsize=15)
            ax.tick_params(labelsize=14)

        return times, states

    def run_deterministic_solver(self, ic, duration=100, dt=0.01, input_function=None, discrete=True):
        """
        Solve system of ODEs using scipy ODE integrators.

        Parameters:
            ic (np array) - initial conditions for each species
            duration (float) - simulation end time
            dt (float) - time step for interpolation
            input_function (function) - returns input value(s) for a given timepoint
            discrete (bool) - if True, use discrete propensity function

        Returns:
            times (np array) - time points corresponding to state values (1 by t)
            states (np array) - matrix of state values at each time point (N by t)
        """

        # initialize solution list
        solution = []

        # first try dopri5 for non-stiff systems:
        try:
            solver = scipy.integrate.ode(self.get_species_rates).set_integrator('dopri5', method='bdf', nsteps=100000).set_f_params(input_function, discrete)
            solout = lambda t, y: solution.append([t] + [y_i for y_i in y])
            solver.set_solout(solout)
            solver.set_initial_value(ic, 0)
            solver.integrate(duration)

        # if dopri5 fails, use vode (slow, but more likely to work for stiff systems). if vode fails, return None
        except UserWarning:
            print('Defaulted to vode solver.')
            try:
                solver = scipy.integrate.ode(self.get_species_rates).set_integrator('vode', method='bdf', nsteps=100000).set_f_params(input_function, discrete)
                solver.set_initial_value(ic, 0)
                while solver.successful() and solver.t <= duration:
                    solver.integrate(duration, step=True)
                    solution.append([solver.t] + [y_i for y_i in solver.y])
            except:
                print('Determinstic ODE solver failed')
                return None, None

        # get solution and sort by time (sometimes solver produces erroneous time points)
        solution = np.array(solution).T
        sort_indices = np.argsort(solution[0, :])
        solution = solution[:, sort_indices]
        times = solution[0, :]
        states = solution[1:, :]

        # interpolate states onto regulate timepoints
        interpolator = interpolate.interp1d(times, states, kind='nearest', axis=1)
        duration = min(np.max(times), duration)
        times_regular = np.arange(0, duration, dt)
        states_regular = interpolator(times_regular)

        return times_regular, states_regular

    def run_tau_leaping_solver(self, ic, dt=0.1, duration=100, input_function=None, discrete=True):
        """
        Generate sample solution of stochastic ODEs using the tau leaping method.

        Parameters:
            ic (np array) -  initial conditions for each species
            dt (float) - time step
            duration (float) - simulation end time
            input_function (function) - returns input value(s) for a given timepoint
            discrete (bool) - if True, use discrete propensity function

        Returns:
            times (np array) - time points corresponding to state values (1 by t)
            states (np array) - matrix of state values at each time point (N by t)
        """

        # construct time vector
        times = np.arange(0, duration, dt)

        # initialize states array
        states = np.zeros((len(ic), len(times)))
        states[:, 0] = ic

        # begin dynamic simulation
        for i, t in enumerate(times[1:]):

            # compute input value
            if input_function is not None:
                input_value = input_function(t)
            else:
                input_value = np.zeros(self.network.input_size)

            # get reaction extents
            rxn_rates = self.get_rxn_rates(states[:, i], input_value, discrete)
            rxn_extents = np.array(list(map(lambda rate: np.random.poisson(abs(rate*dt)), rxn_rates)))
            species_extents = np.dot(self.network.stoichiometry, rxn_extents)

            # update states
            states[:, i+1] = states[:, i] + species_extents

        return times, states

    def run_binomial_leaping_solver(self, ic, dt=0.001, duration=100, input_function=None, discrete=True):
        """
        Generate sample solution of stochastic ODEs using the binomial distribution leaping method.

        Parameters:
            ic (np array) - initial conditions for each species
            dt (float) - time step
            duration (float) - simulation end time
            input_function (function) - returns input value(s) for a given timepoint
            discrete (bool) - if True, use discrete propensity function

        Returns:
            times (np array) - time points corresponding to state values (1 by t)
            states (np array) - matrix of state values at each time point (N by t)
        """

        # construct time vector
        times = np.arange(0, duration, dt)

        # initialize states array
        states = np.zeros((len(ic), len(times)))
        states[:, 0] = ic

        # determine whether each reaction contains any consumed species
        no_consumables = np.all(self.network.stoichiometry >= 0, axis=0)

        # begin dynamic simulation
        for i, t in enumerate(times[1:]):

            # compute input value
            if input_function is not None:
                input_value = input_function(t)
            else:
                input_value = np.zeros(self.network.input_size)

            # get reaction propensities (rates)
            rxn_rates = self.get_rxn_rates(states[:, i], input_value, discrete)

            # initialize reaction extent and temporary state vectors
            rxn_extents = np.zeros((len(rxn_rates)))
            states_temp = states[:, i].copy()

            # iterate across reactions (could randomize order)
            for j, (rate, is_template_rxn) in enumerate(zip(rxn_rates, no_consumables)):

                # if a reactant is not consumed, draw the extent from a poisson distribution
                if is_template_rxn:
                    rxn_extent = np.random.poisson(rate*dt)

                # if a reactant is consumed, draw the extent from a binomial distribution
                else:

                    # find limiting reactant for this reaction, along with maximum extent (k_j_max)
                    v_j = self.network.stoichiometry[:, j]
                    limiting_reactant = np.where(v_j < 0)[0][np.argmin(np.divide(states_temp[v_j < 0], -(v_j[v_j < 0])))]
                    k_j_max = abs(int(states_temp[limiting_reactant]/self.network.stoichiometry[limiting_reactant, j]))

                    # sample reaction extent (k_j) from a binomial distribution with p = rate*dt/k_j_max and n=k_j_max
                    if k_j_max == 0:
                        rxn_extent = 0
                    else:
                        p = min(rate*dt/k_j_max, 1)
                        rxn_extent = np.random.binomial(k_j_max, p)

                # update extent vector
                rxn_extents[j] = rxn_extent

                # update temp state vector
                states_temp += self.network.stoichiometry[:, j]*rxn_extent

            # update states
            states[:, i+1] = states[:, i] + np.dot(self.network.stoichiometry, rxn_extents)

        return times, states

    def run_pure_ssa_solver(self, ic, dt=0.01, duration=100, input_function=None, discrete=True):
        """
        Generate sample solution of stochastic ODEs using the pure gillespie algorithm.

        Parameters:
            ic (np array) - initial conditions for each species
            dt (float) - time step
            duration (float) - simulation end time
            input_function (function) - returns input value for a given timepoint
            discrete (bool) - if True, use discrete propensity function

        Returns:
            times (np array) - time points corresponding to state values (1 by t)
            states (np array) - matrix of state values at each time point (N by t)
        """

        # initialize times and state lists
        t, times, states = 0, [0], [ic.astype(np.int64)]
        search_step = min(1, dt)

        # determine input type (preallocation is ugly but faster)
        test_input = input_function(0)
        if type(test_input) == int or type(test_input) == float:
            def get_max_input(val):
                return val
        else:
            def get_max_input(vals):
                return vals.max()

        cy_rate_function = RateFunction(self.network.reactions).get_callable()

        # begin dynamic simulation
        while t <= duration:

            # compute input value
            if input_function is not None:
                input_value = input_function(t)
            else:
                input_value = np.zeros(self.network.input_size)

            # get current states and reaction rates
            rxn_rates = cy_rate_function(states[-1], input_value)

            # compute total reaction rate
            total_rxn_rate = rxn_rates.sum()

            # if all rates are zero jump to end
            if total_rxn_rate == 0:
                max_input = get_max_input(input_value)
                while max_input == 0 and t <= duration:
                    t += search_step
                    times.append(t)
                    states.append(states[-1])
                    input_value = input_function(t)
                    max_input = get_max_input(input_value)
                continue

            # initialize reaction extents
            rxn_extents = np.zeros(len(rxn_rates), dtype=np.int64)

            # take a single SSA step
            tau = 1/total_rxn_rate*np.log(1/rd.random())

            rxn_fired = self.choose_rxn(rxn_rates)
            rxn_extents[rxn_fired] = 1

            # update times and states
            t += tau
            times.append(t)
            new_states = states[-1] + np.dot(self.network.stoichiometry, rxn_extents)
            states.append(new_states)

        # convert samples to array
        times = np.array(times).T
        states = np.array(states).T

        # interpolate to nearest floor
        times_regular = np.arange(0, duration, dt)
        interpolation_indices = np.searchsorted(times, times_regular)-1
        interpolation_indices[interpolation_indices < 0] = 0
        states_regular = states[:, interpolation_indices]

        return times_regular, states_regular

    def run_hybrid_ssa_solver(self, ic, dt=1, duration=100, input_function=None, discrete=True):
        """
        Generate sample solution of stochastic ODEs using a hybrid tau-leaping/SSA method with variable tao selection.

        Parameters:
            ic (np array) - initial conditions for each species
            dt (float) - time step
            duration (float) - simulation end time
            input_function (function) - returns input value for a given timepoint
            discrete (bool) - if True, use discrete propensity function

        Returns:
            times (np array) - time points corresponding to state values (1 by t)
            states (np array) - matrix of state values at each time point (N by t)
        """

        # hybrid-ssa algorithm parameters
        nc_threshold = 5 # threshold for which reactions are critical (typically ~5 firings)
        separation = 10 # threshold average number of reaction events per timestep (typically ~10)
        epsilon = 0.03 # weight applied to each state in selecting candidate leap
        holding_period = 100 # number of pure-ssa steps taken

        # initialize times and state lists
        t, times, states = 0, [0], [ic.astype(np.int64)]

        # begin dynamic simulation
        pure_ssa_count = 0
        shrink_step_size = False

        # set input function
        if input_function is None:
            null_input = True
            self.get_input = lambda t: np.zeros(self.network.input_size)
        else:
            null_input = False
            self.get_input = input_function

        # instantiate cython rate function
        #cy_rate_function = RateFunction(self.network.reactions)

        # determine reactant species
        reactant_species = np.where(self.network.stoichiometry.min(axis=1) <= 0)[0]

        ### begin simulation
        while t <= duration:

            # compute input value
            input_value = self.get_input(t)

            # get current states and reaction rates
            rxn_rates = self.get_rxn_rates(states[-1], input_value, discrete) #cy_rate_function(states[-1], input_value)

            # compute total reaction rate
            total_rxn_rate = float_sum(rxn_rates)

            if total_rxn_rate == 0:

                # if there is no input and all rates are zero, jump to end
                if null_input:
                    times.append(duration)
                    states.append(states[-1])
                    break

                else:
                    # skip to next change in input
                    while t <= duration:
                        new_input = self.get_input(t)
                        if (new_input != input_value):
                            break
                        else:
                            t += dt
                            times.append(t)
                            states.append(states[-1])
                    continue

            # initialize reaction extents
            rxn_extents = np.zeros(self.num_reactions, dtype=np.int64)

            # if pure_ssa_count is active take a single SSA step
            if pure_ssa_count > 0:
                tau = 1/total_rxn_rate*np.log(1/rd.random())
                rxn_fired = self.choose_rxn(rxn_rates)
                rxn_extents[rxn_fired] = 1
                pure_ssa_count -= 1

            else:

                if shrink_step_size is False:

                    # determine which reactions are critical
                    with np.errstate(divide='ignore', invalid='ignore'):
                        div = np.divide(states[-1].reshape(-1, 1), self.network.stoichiometry)
                        div[div > 0] = 0
                        div[np.isnan(div)] = 0
                        div_max = abs(div).max(axis=0)

                    if np.all(np.isnan(div_max)):
                        critical_rxns = np.zeros(self.num_reactions)
                    else:
                        critical_rxns = np.logical_and(div_max > 0, div_max < nc_threshold)

                    non_critical_rxns = np.logical_not(critical_rxns)

                    # compute candidate leap
                    if non_critical_rxns.any() is False:
                        candidate_leap = duration - t
                    else:
                        extent_mean = abs(np.multiply(rxn_rates[non_critical_rxns], self.network.stoichiometry[:, non_critical_rxns]).sum(axis=1))

                        extent_std_deviation = np.multiply(rxn_rates[non_critical_rxns], self.network.stoichiometry[:, non_critical_rxns]**2).sum(axis=1)

                        numerator = epsilon*states[-1][reactant_species]
                        numerator[numerator < 1] = 1

                        with np.errstate(divide='ignore', invalid='ignore'):
                            candidate_leap = np.concatenate((numerator/extent_mean[reactant_species], (numerator**2)/extent_std_deviation[reactant_species]))
                            candidate_leap = min_float(candidate_leap)

                # if candidate leap is too short, execute 100 pure ssa steps
                if candidate_leap < separation / total_rxn_rate:
                    pure_ssa_count = holding_period
                    continue

                else:
                    # compute tau for critical reactions
                    total_critical_rxn_rate = (rxn_rates[critical_rxns]).sum()
                    if total_critical_rxn_rate == 0:
                        # if total rate is zero, jump to end of simulation
                        alternate_leap = duration - t
                    else:
                        alternate_leap = 1/total_critical_rxn_rate*np.log(1/rd.random())

                    if candidate_leap <= alternate_leap or total_critical_rxn_rate == 0.:
                        # do not fire critical reactions, use tau leap for non critical
                        tau = candidate_leap
                        non_critical_rxn_extents = np.random.poisson(rxn_rates[non_critical_rxns] * tau)
                        rxn_extents[non_critical_rxns] = non_critical_rxn_extents

                    else:
                        # fire one critical reaction, use tau leap for non critical
                        tau = alternate_leap

                        if critical_rxns.any():

                            critical_rxn_fired = np.where(critical_rxns)[0][self.choose_rxn(rxn_rates[critical_rxns])]

                            rxn_extents[critical_rxn_fired] = 1

                        # use tau leap for non critical reactions
                        non_critical_rxn_extents = np.random.poisson(rxn_rates[non_critical_rxns] * tau)
                        rxn_extents[non_critical_rxns] = non_critical_rxn_extents

            # update states
            new_states = states[-1] + np.dot(self.network.stoichiometry, rxn_extents)

            # check if step was too large
            if min_int(new_states) < 0:
                candidate_leap = tau / 2
                shrink_step_size = True
                continue
            else:
                shrink_step_size = False

            # update times and states
            t += tau
            times.append(t)
            states.append(new_states)

        # convert samples to array
        times = np.array(times).T
        states = np.array(states).T

        # interpolate states onto regulate timepoints by taking nearest floor value
        times_regular = np.arange(0, duration, dt)
        interpolation_indices = np.searchsorted(times, times_regular)-1
        interpolation_indices[interpolation_indices < 0] = 0
        states_regular = states[:, interpolation_indices]
        return times_regular, states_regular

    def run_stochastic_simulation(self, ic=None, input_function=None, method='ssa', num_trials=1000, duration=1, dt=0.01, ax=None):

        if ic is None:
            ic = np.zeros(self.network.nodes.size)

        # create figure if none provided
        if ax is None:
            fig, ax = plt.subplots(ncols=1, figsize=(4, 3))
        else:
            fig = plt.gcf()

        ax.set_xlabel('time', fontsize=14), ax.set_ylabel('abundance', fontsize=14)
        ax.set_title('Solution via: {:s} \n {:d} trials'.format(method, num_trials), fontsize=14)
        ax.tick_params(labelsize=12)

        # generate samples
        for _ in range(num_trials):
            t, x = self.simulate(ic=ic, method=method, dt=dt, duration=duration, input_function=input_function)
            _ = ax.plot(t, x[0, :], '-k', alpha=0.25)

        ax.set_ylim(0, 1.2*np.max(x))

        return fig, ax

    def compile_moment_vectors(self, max_moment, external_input=False):
        """
        Compiles dictionary of moment vectors along with corresponding indices.

        Parameters:
            max_moment (int) - maximum moment order considered
            external_input (bool) - if True, system includes an external input

        Returns:
            moment_vectors_dict (dict) - keys are vector indices and values are moment vectors
            column_indices_dict (dict) - keys are m_vectors (tuple) and values are row indices

        """

        if external_input is False:
            n = self.network.nodes.size
        else:
            n = self.network.nodes.size + 1

        # compile moment indices dictionary (m vectors)
        i = 0
        moment_vectors_dict = {}
        for m in range(1, max_moment+1):
            m_vectors = self.get_m_vectors(n, m)
            for m_vector in m_vectors:
                moment_vectors_dict[i] = m_vector
                i += 1
        column_indices_dict = {value: key for key, value in moment_vectors_dict.items()}

        return moment_vectors_dict, column_indices_dict

    def get_moment_matrix(self, max_moment=2, external_input=False):
        """
        Compiles moment dynamics matrix (mCn x mCn)

        Parameters:
            max_moment (int) - maximum moment order considered
            external_input (bool) - if True, system includes an external input

        Returns:
            moment_matrix (np array) - system interaction matrix for moment equations (mCn x mCn)
            column_indices_dict (dict) - keys are m_vectors (tuple) and values are row indices (int)
            ho_matrix (np array) - interaction matrix for higher order terms (mCn x ?)
            hod_columns_dict (dict) - keys are m_vectors (tuple), values are column indices
        """

        # compile moment vector with corresponding indices
        moment_vectors_dict, column_indices_dict = self.compile_moment_vectors(max_moment, external_input)

        # initialize moment dynamics matrix
        moment_matrix = np.zeros((len(moment_vectors_dict), len(moment_vectors_dict)))

        # initialize vector for higher order terms
        ho_matrix = np.zeros((len(moment_vectors_dict), 0))
        ho_columns_dict = {}

        # iterate across all moments
        for row, n in sorted(moment_vectors_dict.items()):
            n = np.array(n)

            # iterate across all reactions
            for l, rxn in enumerate(self.network.reactions):

                # iterate across all propensities
                for i, a in zip(*rxn.get_propensity_coefficients(external_input)):

                    # iterate across all moments of equal or lower order
                    for k in self.get_j_vectors(n):

                        # skip zeroth j_vector
                        k = np.array(k)
                        if sum(k) == 0:
                            continue

                        # if external input is included, append input to stoichiometric vector
                        if external_input is False:
                            stoichiometry = rxn.stoichiometry
                        else:
                            stoichiometry = np.hstack((0, rxn.stoichiometry))

                        # compute coefficient and column index
                        coefficient = a*functools.reduce(lambda x, y: x*y, stoichiometry**k * np.array([scipy.misc.comb(n_i, k_i) for n_i, k_i in zip(n, k)]))
                        column_index = n-k+i

                        if tuple(column_index) in column_indices_dict.keys():
                            # increment dynamics matrix
                            moment_matrix[row, column_indices_dict[tuple(column_index)]] += coefficient

                        elif tuple(column_index) in ho_columns_dict.keys():
                            # increment higher order dynamics matrix
                            ho_matrix[row, ho_columns_dict[tuple(column_index)]] += coefficient

                        else:
                            # if dependence not yet encountered, add it to higher order dynamics matrix
                            ho_columns_dict[tuple(column_index)] = np.shape(ho_matrix)[1]
                            new_column = np.zeros((len(moment_matrix), 1))
                            new_column[row, 0] = coefficient
                            ho_matrix = np.hstack((ho_matrix, new_column))

        return moment_matrix, column_indices_dict, ho_matrix, ho_columns_dict

    @staticmethod
    def get_m_vectors(n, m):
        """
        Generates list of moment vectors (tuples of length N containing moment orders for each species)

        Parameters:
            n (int) - system dimension (number of species)
            m (int) - maximum moment order included

        Returns:
            indices (list of tuples) - list of tuples describing the moment order of each species
        """
        indices = []
        for index in itertools.product(range(0, m+1), repeat=n):
            if sum(index) == m:
                indices.append(index)
        indices.reverse()
        return indices

    @staticmethod
    def get_j_vectors(m_vector):
        """
        Generates all sequential moment vectors up to the specified moment vector.

        Parameters:
            m_vector (tuple) - upper bound for j_vectors

        Returns:
            j_vectors (list of tuples) - ordered list of tuples describing each sequential vector up to m_vector
        """
        j_vectors = []
        for ind in itertools.product(*[range(0, index+1) for index in m_vector]):
            j_vectors.append(ind)
        return j_vectors

    def run_moment_dynamics_solver(self, m=2, ic=None, input_function=None, dt=0.01, constrain_positive=False, duration=100):
        """
        Solve the specified number of moment equations for a system of stochastic ODEs.

        Parameters:
            m (int) - max moment order considered
            ic (np array) - initial conditions for each species
            input_function (function) - returns input value for a given timepoint
            dt (float) - time step
            constrain_positive (bool) - if True, constrains all states to remain positive
            duration (float) - simulation end time

        Returns:
            times (np array) - time points corresponding to state values (1 by t)
            moments_dict (dict) - dictionary in which keys are species and values are (nCm x t) np arrays describing the
            time evolution of each moment
        """

        # compile reindexing key, stoichiometric matrix
        self.network.compile_stoichiometry()

        # define time course
        times = np.arange(0, duration, dt)

        # compile moment dynamics matrix
        if input_function is not None:
            external_input = True
        else:
            external_input = False
        moment_matrix, column_indices_dict, ho_matrix, ho_columns_dict = self.get_moment_matrix(m, external_input)

        # if no initial condition provided, initialize all moments as zero
        ic_moments = np.zeros((len(moment_matrix)))
        if ic is not None:
            for i in range(0, self.network.nodes.size):

                if input_function is not None:
                    new_index, new_dim = i+self.network.input_size, self.network.nodes.size + self.network.input_size
                else:
                    new_index, new_dim = i, self.network.nodes.size

                mean_keys = np.zeros(new_dim)
                mean_keys[new_index] = 1
                ic_moments[column_indices_dict[tuple(mean_keys)]] = ic[i]

        # try setting all other moments of a species to zero as well when one moment hits zero
        moments = self.solve_linear_ode(moment_matrix, times, ic=ic_moments, input_function=input_function, constrain_positive=constrain_positive, key=column_indices_dict)

        # ordered tuples
        indices_ordered = np.array(sorted(column_indices_dict.keys(), key=lambda x: column_indices_dict[x]))

        # compile dictionary for centered moment dynamics arrays
        moments_dict = {}
        for species in range(0, self.network.nodes.size):

            if input_function is not None:
                species_index = species + self.network.input_size
            else:
                species_index = species

            # find and store index of current moment for current species
            ordered_moment_indices = np.zeros((np.max(indices_ordered)), dtype=int)
            for m in range(1, np.max(indices_ordered)+1):
                index = np.intersect1d(np.where(indices_ordered[:, species_index] == m), np.where(np.sum(indices_ordered, axis=1) == m))
                ordered_moment_indices[m-1] = index

            # store ordered moment dynamics
            moments_dict[species] = moments[ordered_moment_indices, :]

        return times, moments_dict

    @staticmethod
    def solve_linear_ode(moment_dynamics, times, ic=None, input_function=None, constrain_positive=False, key=None):
        """
        Solves linear system of ordinary differential equations of the form X' = AX

        Parameters:
            moment_dynamics (np array) - linear coefficient matrix, n x n
            times (np array) - array of time points for which solutions are evaluated, t x 1
            ic (np array) - initial conditions for each moment, n x 1
            input_function (function) - returns input value for a given timepoint
            constrain_positive (bool) - if True, constrains all states to remain positive
            key (dict) - dictionary relating m_vectors to moment_dynamics matrix

        Returns:
            moments (np array) - solution evaluated at all specified times, t x n
        """

        # if no initial condition provided, assume zero
        if ic is None:
            ic = np.zeros((len(moment_dynamics)))

        # define linear system of ordinary differential equations
        #deriv = lambda x, *t: np.dot(moment_dynamics, x)

        def deriv(x_f, *t_f, input_f=None):
            if input_f is not None:
                x_f = np.hstack((input_f(t_f), x_f[1:]))
            return np.dot(moment_dynamics, x_f)
        deriv_ = functools.partial(deriv, input_f=input_function)

        # use manual shooting method if states are constrained to be positive
        if constrain_positive is False:
            moments = scipy.integrate.odeint(deriv_, ic, times)

        else:
            reverse_key = {index: m_vector for m_vector, index in key.items()}
            moments = np.zeros((len(times), len(moment_dynamics)))
            moments[0, :] = ic
            for i, t in enumerate(times[1:]):
                dt = t - times[i]
                x = moments[i, :]
                dx_dt = deriv(x, t, input_f=input_function)
                moments[i+1, :] = x + dx_dt*dt

                # set all negative states to zero
                moments[i+1, moments[i+1, :] < 0] = 0

                # find all higher order moments and set them to zero
                for j, moment in enumerate(moments[i+1, :]):
                    if moment <= 0 and sum(reverse_key[j]) == 1:
                        negative_state_index = np.argmax(reverse_key[j])

                        for m_vector, index in key.items():
                            if m_vector[negative_state_index] > 0:
                                moments[i+1, index] = 0

        return moments.T


class MonteCarloSimulation(Simulation):
    """
    Class defines multiple trials of a simulation procedure.
    """

    def __init__(self, system, ic=None, input_function=None, integrator_ic=None, dt=1, duration=100, condition=None):
        """
        Each instance has a fixed set of simulation parameters for which samples are generated.

        Parameters:
            system (network object)
            ic (np array) - initial conditions, defaults to zero
            input_function (function) - returns input value(s) for each time
            integrator_ic (np array) - integrator initialization
            dt (float) - time step
            duration (float) - simulation end time
            condition (str) - environmental conditions affecting rates
        """

        Simulation.__init__(self, system, condition)

        # set initial condition generator
        if type(ic) == tuple:
            m, v = ic
            dim = system.nodes.size
            self.get_ic = lambda n: np.random.normal(m, np.sqrt(v), size=(n, dim)).astype(int)

        elif type(ic) == types.FunctionType:
            self.get_ic = lambda n: [ic() for _ in range(n)]

        else:
            self.get_ic = lambda n: [ic for _ in range(n)]

        # set integrator initial condition generator
        if type(integrator_ic) == tuple:
            m, v = integrator_ic
            dim = system.nodes.size
            self.get_integrator_ic = lambda n: np.random.normal(m, np.sqrt(v), size=(n, dim)).astype(int)

        elif type(integrator_ic) == types.FunctionType:
            self.get_integrator_ic = lambda n: [integrator_ic() for _ in range(n)]

        else:
            self.get_integrator_ic = lambda n: [integrator_ic for _ in range(n)]

        self.dt = dt
        self.duration = duration
        self.input = input_function

    def __repr__(self):
        self.network.print_reactions()

    def run_trials(self, num_trials=10, method='bd-leaping', discrete=True, condition=None):
        """
        Runs multiple monte carlo trials and stores results.

        Parameters:
            num_trials (int) - number of independent trials
            method (str) - stochastic simulation algorithms used
            discrete (bool) - if True, use discrete rate laws
            condition (str) - environmental conditions affecting rate. either None, 'cold', 'hot', 'diabetic', or 'minute'

        Returns:
            timeseries (TimeSeries) - trajectories of system states
        """

        # apply any rate scaling conditions
        if condition is not None:
            simulation = Simulation(self.network, condition)
        else:
            simulation = self

        # get initial condition
        ics = self.get_ic(num_trials)
        integrator_ics = self.get_integrator_ic(num_trials)

        # run each trial, appending sample to a list of all samples
        samples = []
        for trial in range(num_trials):
            ic = ics[trial]
            integrator_ic = integrator_ics[trial]
            if (ic < 0).sum() != 0:
                print('FAIL')
                print(trial, ic)
                break
            times, states = simulation.simulate(ic, input_function=self.input, integrator_ic=integrator_ic, method=method, dt=self.dt, duration=self.duration,  discrete=discrete)
            samples.append(states)

        # instantiate time series object
        timeseries = TimeSeries(times, np.array(samples))

        return timeseries


def print_moment_vector(dynamics_ind, v_spacing=0.2):
    fig = plt.figure(figsize=(2, 2))
    ax = plt.gca()
    reverse = {val: key for key, val in dynamics_ind.items()}
    for row in range(0, len(dynamics_ind)):
        moment_text = '$<' + '\mu^{{{:s}}}'.format(str(reverse[row])) + '>$'
        ax.text(0, 1-row*v_spacing, moment_text, fontsize=18)
    _ = plt.axis('off')
    return fig
