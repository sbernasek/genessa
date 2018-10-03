# intra-package python imports
from ..kinetics.massaction import MassAction

"""
TO DO:
    1. add separate input nodes to graph
    2. add enzymatic reactions to mutation scheme

"""


class MutableNetwork(Network):
    """
    Class inherits a network to which it adds growth and mutation capabilities.

    Attributes:

        name (str) - name of class type

        unique_node_id (int) - counter for unique node numbers

    Inherited Attributes:

        nodes (np.ndarray) - vector of node indices

        node_key (dict) - {state dimension: node id} pairs

        reactions (list) - list of reaction objects

        stoichiometry (np.ndarray) - stoichiometric coefficients, (N,M)

        N (int) - number of nodes

        M (int) - number of reactions

        I (int) - number of inputs

    """

    def __init__(self, N=1, I=1, reactions=None):
        """
        Inherits a network and adds mutation capabilities.

        Args:

            N (int) - number of nodes

            I (int) - number of input channels

            reactions (list) - reactions

        """
        super().__init__(N, I, reactions)
        self.name = 'Cell'
        self.unique_node_id = N

    def divide(self):
        """
        Returns duplicate of network.
        """
        child = deepcopy(self)
        return child

    def mutate(self, mutation_rates, rate_constants=None):
        """
        Randomly undergo mutation.

        Args:

            mutation_rates (dict) - frequencies of each type of mutation

            rate_constants (dict) - default rate constants

        """

        # node additions
        num_node_additions = np.random.poisson(mutation_rates['add_node'])
        if num_node_additions > 0:
            self.add_nodes(num_node_additions)

        # node deletions
        if self.N > 1:
            num_node_removals = np.random.poisson(mutation_rates['remove_node'] * (self.N - 1))
            if num_node_removals > 0:
                self.remove_nodes(num_node_removals)

        # edge additions
        num_edge_additions = np.random.poisson(mutation_rates['add_edge'] * self.N)
        for _ in range(num_edge_additions):
            self.add_edge(rate_constants)

        # edge deletions
        num_edge_removals = np.random.poisson(mutation_rates['remove_edge'] * self.M)
        if num_edge_removals > 0:
            self.remove_edges(num_edge_removals)

        # rate modifications
        num_rate_modifications = np.random.poisson(mutation_rates['modify_rate'] * self.M)
        if num_rate_modifications > 0:
            self.modify_rates(num_rate_modifications)

        # recompile stoichiometry
        self.compile_stoichiometry()

        # resize input channels
        self.resize_inputs()

    def modify_rates(self, modifications):
        """
        Multiply a random rate constant by a factor between 0.5 and 2.

        Args:

            modifications (int) - number of rate modifications

        """

        rxn_indices = np.random.randint(self.M, size=modifications)
        factors = 2**np.random.uniform(low=-1, high=1, size=modifications)
        for index, factor in zip(rxn_indices, factors):
            self.reactions[index].k *= factor

    def add_nodes(self, additions=1):
        """
        Add a new nodes to the network.

        Args:

            additions (int) - number of nodes added

        """
        node_ids = np.arange(additions) + self.unique_node_id
        self.update_reaction_dimensions(added_node_ids=node_ids)
        self.nodes = np.insert(self.nodes, np.searchsorted(self.nodes, node_ids), node_ids)
        self.node_key = {index: int(node_id) for index, node_id in enumerate(self.nodes)}
        self.unique_node_id += additions

    def remove_nodes(self, removals=1, removed_nodes=None):
        """
        Remove nodes from the network.

        Args:

            removals (int) - number of nodes removed

            removed_nodes (array like) - specific nodes to be removed

        """

        # remove randomly selected nodes
        candidate_nodes = self.nodes[self.nodes != self.output_node]
        if candidate_nodes.size >= removals:
            node_ids = np.random.choice(candidate_nodes, size=removals, replace=False)
            self.remove_dependencies(node_ids)
            self.nodes = np.setdiff1d(self.nodes, node_ids)
            self.update_reaction_dimensions(removed_node_ids=node_ids)
            self.node_key = {index: int(node_id) for index, node_id in enumerate(self.nodes)}

        # remove specified nodes
        if removed_nodes is not None:
            removed_nodes = np.array(removed_nodes)
            self.remove_dependencies(removed_nodes)
            self.nodes = np.setdiff1d(self.nodes, removed_nodes)
            self.update_reaction_dimensions(removed_node_ids=removed_nodes)
            self.node_key = {index: int(node_id) for index, node_id in enumerate(self.nodes)}

    def add_edge(self, rate_constants=None):
        """
        Add a new edge to the network.

        Args:

            rate_constants (dict) - default rate constants

        """

        if rate_constants is None:
            rate_constants = {0: 1, 1: 1, 2: 1}

        # initialize stoichiometry and propensity (input padded left) vectors
        stoichiometry = np.zeros(self.N)
        propensity = np.zeros(self.N + self.I)

        # decide between 0th, 1st, or 2nd order with uniform probability
        rxn_order = np.random.randint(3)

        # constant increase/decrease of reactant
        if rxn_order == 0:
            stoichiometry[np.random.randint(0, len(stoichiometry))] = np.random.choice([-1, 1])

        if rxn_order == 1:
            reactant = np.random.randint(0, len(propensity))
            propensity[reactant] = 1

            # 50% chance of reactant consumption
            if np.random.randint(0, 2) == 0 and reactant != 0:
                stoichiometry[reactant-self.I] = -1

            # 50% chance of product formation
            if np.random.randint(0, 2) == 0:
                eligible_products = [candidate for candidate in np.arange(len(stoichiometry)) if candidate != (reactant-self.I)]
                if len(eligible_products) > 0:
                    product = eligible_products[np.random.randint(len(eligible_products))]
                    stoichiometry[product] = 1

        if rxn_order == 2:
            reactants = np.random.randint(len(propensity), size=2)
            for reactant in reactants:
                propensity[reactant] += 1

                # 50% chance of consumption for each reactant
                if np.random.randint(0, 2) == 0 and reactant >= self.I:
                    stoichiometry[reactant-self.I] += -1

            # 50% chance of product formation
            if np.random.randint(0, 2) == 0:
                eligible_products = [candidate for candidate in np.arange(len(stoichiometry)) if candidate not in reactants]
                if len(eligible_products) > 0:
                    product = eligible_products[np.random.randint(len(eligible_products))]
                    stoichiometry[product] = 1

        # add new reaction to network
        rxn = MassAction(stoichiometry,
                         propensity=propensity[self.I:],
                         input_dependence=propensity[0:self.I],
                         k=rate_constants[rxn_order])
        self.reactions.append(rxn)
        self.M += 1

    def remove_edges(self, removals=1):
        """
        Remove edges from the network.

        Args:

            removals (int) - number of edges removed

        """
        for _ in range(removals):
            if self.M > 0:
                _ = self.reactions.pop(np.random.randint(self.M))
                self.M -= 1

    def remove_dependencies(self, removed_node_ids):
        """
        Remove all downstream dependencies

        Args:

            removed_node_ids (np.ndarray) - nodes whose dependents are removed

        """

        removed_dimensions = np.array([dim for dim, node_id in self.node_key.items() if node_id in removed_node_ids])

        for rxn in self.reactions:
            stoich_contrib = np.sum(abs(rxn.stoichiometry[removed_dimensions])
            propensity_contrib = abs(rxn.propensity[removed_dimensions]))
            if stoich_contrib + propensity_contrib != 0:
                self.reactions.remove(rxn)
                self.M -= 1

    def merge_with_another(self, other_network):
        """
        Merges current network instance with another network instance.

        Args:

            other_network (MutableNetwork object) - network to be merged

        """

        # copy other network
        other_network = other_network.divide()

        # get node arrays before merge
        self_nodes = self.nodes
        other_nodes = other_network.nodes

        # determine nodes exclusive to each network
        self_exclusive = self_nodes[np.logical_not(np.in1d(self_nodes, other_nodes))]
        other_exclusive = other_nodes[np.logical_not(np.in1d(other_nodes, self_nodes))]

        # update system dimensions for each network (resize stoichiometry/propensity vectors)
        self.update_reaction_dimensions(added_node_ids=other_exclusive)
        other_network.update_reaction_dimensions(added_node_ids=self_exclusive)

        # incorporate nodes exclusive to other network into current network
        insertion_points = np.searchsorted(self_nodes, other_exclusive)
        self.nodes = np.insert(self_nodes, insertion_points, other_exclusive)
        self.node_key = {index: int(node_id) for index, node_id in enumerate(self.nodes)}

        # update unique_node_id
        self.unique_node_id = np.max(self.nodes)+1

        # incorporate reactions from other network
        self.reactions.extend(other_network.reactions)
        self.M += other_network.M
        self.compile_stoichiometry()
        self.resize_inputs()
