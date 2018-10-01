from .reactions import Reaction, EnzymaticReaction, SumReaction, Coupling
import numpy as np
import copy as copy
from tabulate import tabulate
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import gridspec

"""
TO DO:
    1. add separate input nodes to graph
    2. add enzymatic reactions to mutation scheme

"""


class Network:
    """
    Class defines a network of interacting nodes. Nodes interact via stochastic processes called reactions. Each reaction has a propensity that may depend upon the current state of one or more other nodes.

    Attributes:

        output_node (int) - index of output

        nodes (np array) - vector of node indices

        reactions (list) - list of reaction objects

        stoichiometry (np array) - N x M matrix of stoichiometric coefficients

    """

    def __init__(self, nodes=1, inputs=1, reactions=None, output_node=0):
        """
        Args:

            nodes (int) - number of nodes

            inputs (int) - number of input channels

            reactions (list or rxn object) - reaction objects

            output_node (int) - index of output

        """

        # initialize network species
        self.output_node = output_node
        self.nodes = np.arange(0, nodes)
        self.input_size = inputs

        # initialize reaction list
        if reactions is None:
            self.reactions = []
        elif type(reactions) == list:
            self.reactions = reactions
        else:
            self.reactions = [reactions]

        # initialize stoichiometric matrix
        self.stoichiometry = None

    def sort_rxns(self):
        """
        Sorts reactions list by type.
        """
        self.reactions = sorted(self.reactions, key=lambda rxn: str(rxn.__class__))

    def compile_stoichiometry(self):
        """
        Iterates through list of M reactions involving N species and constructs N x M stoichiometric matrix.
        """

        # initialize stoichiometric matrix as zeros
        self.stoichiometry = np.zeros((self.nodes.size, len(self.reactions)), dtype=np.int64)
        for rxn_num, rxn in enumerate(self.reactions):
            self.stoichiometry[:, rxn_num] = rxn.stoichiometry

    def compile_rate_dependencies(self):
        """
        Iterates through list of M reactions involving N species and I inputs and constructs M x N dependency matrices.
        """

        # compile state dependency matrix
        self.state_dependence = np.zeros((len(self.reactions), self.nodes.size), dtype=np.int64)
        for rxn_num, rxn in enumerate(self.reactions):
            self.state_dependence[rxn_num, :] = rxn.propensity

        # compile state dependency matrix
        self.input_dependence = np.zeros((len(self.reactions), self.input_size), dtype=np.int64)
        for rxn_num, rxn in enumerate(self.reactions):
            self.input_dependence[rxn_num, :] = rxn.input_dependence

    def resize_inputs(self):
        """
        Resize dependence vector of all reactions with no input dependence.
        """

        for rxn in self.reactions:

            # resize reactions
            if rxn.input_dependence is None:
                rxn.input_dependence = np.zeros(self.input_size)
            elif rxn.input_dependence.max() == 0:
                rxn.input_dependence = np.zeros(self.input_size)

            # resize repressors
            if type(rxn) in (EnzymaticReaction, Coupling):
                for repressor in rxn.repressors:
                    if repressor.input_dependence is None:
                        repressor.input_dependence = np.zeros(self.input_size)

    def get_input_dependents(self, ic=None):
        """
        Find all nodes dependent upon the input. Node B is dependent upon node A if B appears within the stoichiometry
        vector of a reaction whose propensity includes A.

        Return:
            dependents (set) - nodes affected by input
            ic (array like) - initial conditions used to determine which states may be accessed. if None, just check for kinetic dependency
        """

        # initialize dependent node sets
        dependents, dependents_prev = set(), set()

        # determine all nodes immediately dependent upon the input
        for rxn in self.reactions:
            if rxn.input_dependence.max() != 0:
                if ic is None:
                    _ = [dependents.add(dependent) for dependent, coeff in enumerate(rxn.stoichiometry) if coeff != 0]
                else:
                    for dependent, coeff in enumerate(rxn.stoichiometry):
                        if coeff != 0:

                            # get reactants
                            reactants = set([node for node, coeff in enumerate(rxn.propensity) if coeff != 0])

                            # if all reactants are currently accounted for, include dependent
                            if reactants.issubset(dependents):
                                dependents.add(dependent)

                            # if all reactants are either accounted for or have positive initial levels, include dependent
                            else:
                                inactive_reaction = [0 if ic[reactant] > 0 else 1 for reactant in reactants.difference(dependents)]
                                if sum(inactive_reaction) == 0:
                                    dependents.add(dependent)

        # add downstream dependencies to set
        dependents = self.get_node_dependents(dependents, ic)

        return dependents

    def get_node_dependents(self, node_set, ic=None):
        """
        Find all nodes dependent upon a set of nodes. Node B is dependent upon node A if B appears within the stoichiometry
        vector of a reaction whose propensity includes A.

        Parameters:
            node_set (set) - nodes whose dependents are desired
            ic (array like) - initial conditions used to determine which states may be accessed. if None, just check for kinetic dependency

        Return:
            dependents (set) - nodes affected by any node in node_set
        """

        # initialize list of dependent and candidate nodes
        dependents, dependents_prev = node_set, set()
        new_dependents = dependents.difference(dependents_prev)

        # recursively identify all downstream dependents
        while len(new_dependents) != 0:
            dependents_prev = copy.deepcopy(dependents)
            for parent in new_dependents:
                for rxn in self.reactions:
                    if parent in [node for node, coeff in enumerate(rxn.propensity) if coeff != 0]:

                        # if we only want kinetic dependency, add all dependents
                        if ic is None:
                            _ = [dependents.add(dependent) for dependent, coeff in enumerate(rxn.stoichiometry) if coeff != 0]

                        else:
                            for dependent, coeff in enumerate(rxn.stoichiometry):
                                if coeff != 0:

                                    # get reactants
                                    reactants = set([node for node, coeff in enumerate(rxn.propensity) if coeff != 0])

                                    # if all reactants are currently accounted for, include dependent
                                    if reactants.issubset(dependents):
                                        dependents.add(dependent)

                                    # if all reactants are either accounted for or have positive initial levels, include dependent
                                    else:
                                        inactive_reaction = [0 if ic[reactant] > 0 else 1 for reactant in reactants.difference(dependents)]
                                        if sum(inactive_reaction) == 0:
                                            dependents.add(dependent)

            new_dependents = dependents.difference(dependents_prev)

        return dependents

    def print_reactions(self):
        """
        Creates graph object and prints reactions.
        """
        Graph(self).show_reactions()


class MutableNetwork(Network):
    """
    Class inherits a network to which it adds growth and mutation capabilities.

    Attributes:

        name (str) - name of class type

        unique_node_id (int) - counter for unique node numbers

        node_key (dict) - maps state space dimension to unique node id

    """

    def __init__(self, nodes=1, inputs=1, reactions=None, output_node=0):
        """
        Inherits a network and adds mutation capabilities.

        Args:

            nodes (int) - number of nodes for initialized mutable network

            inputs (int) - number of input channels

            reactions (list) - reactions for initialized mutable network

            output_node (int) - index of output

        """
        Network.__init__(self, nodes, inputs, reactions, output_node)
        self.name = 'Cell'
        self.unique_node_id = nodes
        self.node_key = {num: num for num in range(nodes)}

    def __repr__(self):
        """
        Print all reactions and visualize graph.
        """
        graph = Graph(self)
        graph.show_reactions()
        graph.visualize_graph()
        return str(type(self))

    @staticmethod
    def from_json(js):

        # create instance
        network = MutableNetwork()

        # get each attribute from json dictionary
        network.output_node = js['output_node']
        network.nodes = np.array(js['nodes'])
        network.unique_node_id = js['unique_node_id']
        network.stoichiometry = np.array(js['stoichiometry'])
        network.node_key = {int(key): int(val) for key, val in js['node_key'].items()}

        # get attributes containing nested classes
        network.reactions = [Reaction.from_json(rxn) if rxn_type=='mass_action' else EnzymaticReaction.from_json(rxn)
                             for rxn, rxn_type in zip(js['reactions'], js['rxn_types'])]

        return network

    def to_json(self):
        return {
            # return each attribute
            'output_node': int(self.output_node),
            'nodes': self.nodes.tolist(),
            'unique_node_id': int(self.unique_node_id),
            'stoichiometry': self.stoichiometry.tolist(),
            'node_key': self.node_key,

            # return attributes containing nested classes
            'reactions': [rxn.to_json() for rxn in self.reactions],
            'rxn_types': ['mass_action' if isinstance(rxn, Reaction) else 'enzymatic' for rxn in self.reactions]}

    def divide(self):
        """
        Returns duplicate of network.
        """
        child = copy.deepcopy(self)
        return child

    def mutate(self, mutation_rates, rate_constants=None):
        """
        Randomly undergo mutation.

        Parameters:
            mutation_rates (dict) - contains frequencies of each type of mutation
            rate_constants (dict) - default zero, first, and second order rate constants
        """

        # node additions
        num_node_additions = np.random.poisson(mutation_rates['add_node'])
        if num_node_additions > 0:
            self.add_nodes(num_node_additions)

        # node deletions
        if self.nodes.size > 1:
            num_node_removals = np.random.poisson(mutation_rates['remove_node'] * (self.nodes.size - 1))
            if num_node_removals > 0:
                self.remove_nodes(num_node_removals)

        # edge additions
        num_edge_additions = np.random.poisson(mutation_rates['add_edge'] * self.nodes.size)
        for _ in range(num_edge_additions):
            self.add_edge(rate_constants)

        # edge deletions
        num_edge_removals = np.random.poisson(mutation_rates['remove_edge'] * len(self.reactions))
        if num_edge_removals > 0:
            self.remove_edges(num_edge_removals)

        # rate modifications
        num_rate_modifications = np.random.poisson(mutation_rates['modify_rate'] * len(self.reactions))
        if num_rate_modifications > 0:
            self.modify_rates(num_rate_modifications)

        # recompile stoichiometry
        self.compile_stoichiometry()

        # resize input channels
        self.resize_inputs()

    def modify_rates(self, modifications):
        """
        Multiply a random rate constant by a factor between 0.5 and 2.

        Parameters:
            modifications (int) - number of rate modifications
        """

        rxn_indices = np.random.randint(len(self.reactions), size=modifications)
        factors = 2**np.random.uniform(low=-1, high=1, size=modifications)
        for index, factor in zip(rxn_indices, factors):
            self.reactions[index].k *= factor

    def add_nodes(self, additions=1):
        """
        Add a new nodes to the network.

        Parameters:
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

        Parameters:
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

        Parameters:
            rate_constants (dict) - default zero, first, and second order rate constants
        """

        if rate_constants is None:
            rate_constants = {0: 1, 1: 1, 2: 1}

        # initialize stoichiometry and propensity (input in zero position) vectors
        stoichiometry = np.zeros(self.nodes.size)
        propensity = np.zeros(self.nodes.size + self.input_size)

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
                stoichiometry[reactant-self.input_size] = -1

            # 50% chance of product formation
            if np.random.randint(0, 2) == 0:
                eligible_products = [candidate for candidate in np.arange(len(stoichiometry)) if candidate != (reactant-self.input_size)]
                if len(eligible_products) > 0:
                    product = eligible_products[np.random.randint(len(eligible_products))]
                    stoichiometry[product] = 1

        if rxn_order == 2:
            reactants = np.random.randint(len(propensity), size=2)
            for reactant in reactants:
                propensity[reactant] += 1

                # 50% chance of consumption for each reactant
                if np.random.randint(0, 2) == 0 and reactant >= self.input_size:
                    stoichiometry[reactant-self.input_size] += -1

            # 50% chance of product formation
            if np.random.randint(0, 2) == 0:
                eligible_products = [candidate for candidate in np.arange(len(stoichiometry)) if candidate not in reactants]
                if len(eligible_products) > 0:
                    product = eligible_products[np.random.randint(len(eligible_products))]
                    stoichiometry[product] = 1

        # add new reaction to network
        rxn = Reaction(stoichiometry, propensity=propensity[self.input_size:], input_dependence=propensity[0:self.input_size], k=rate_constants[rxn_order])
        self.reactions.append(rxn)

    def remove_edges(self, removals=1):
        """
        Remove edges from the network.

        Parameters:
            removals (int) - number of edges removed
        """
        for _ in range(removals):
            if len(self.reactions) > 0:
                _ = self.reactions.pop(np.random.randint(len(self.reactions)))

    def remove_dependencies(self, removed_node_ids):
        """
        Remove all downstream dependencies

        Parameters:
            removed_node_ids (np array) - nodes whose dependencies are to be removed
        """

        removed_dimensions = np.array([dim for dim, node_id in self.node_key.items() if node_id in removed_node_ids])

        for rxn in self.reactions:
            if np.sum(abs(rxn.stoichiometry[removed_dimensions])+abs(rxn.propensity[removed_dimensions])) != 0:
                self.reactions.remove(rxn)

    def update_reaction_dimensions(self, added_node_ids=None, removed_node_ids=None):
        """
        Updates dimensions of stoichiometry and propensity vectors for each reaction.

        Parameters:
            added_node_ids (np array) - unique indices of nodes to be added
            removed_node_ids (np array) - unique indices of nodes to be removed
        """

        # add new nodes to stoichiometry and propensity vectors for each reaction
        if added_node_ids is not None:
            insertion_points = np.searchsorted(self.nodes, added_node_ids)
            for rxn in self.reactions:
                rxn.stoichiometry = np.insert(rxn.stoichiometry, insertion_points, np.zeros(insertion_points.size))
                rxn.propensity = np.insert(rxn.propensity, insertion_points, np.zeros(insertion_points.size))
                rxn.active_species = np.where(rxn.propensity != 0)[0]

        # remove expired nodes from stoichiometry and propensity vectors for each reaction
        if removed_node_ids is not None:
            removed_dimensions = np.array([dim for dim, node_id in self.node_key.items() if node_id in removed_node_ids])
            for rxn in self.reactions:
                rxn.stoichiometry = np.delete(rxn.stoichiometry, removed_dimensions)
                rxn.propensity = np.delete(rxn.propensity, removed_dimensions)
                rxn.active_species = np.where(rxn.propensity != 0)[0]

    def merge_with_another(self, other_network):
        """
        Merges current network instance with another network instance.

        Parameters:
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
        self.compile_stoichiometry()
        self.resize_inputs()


class Graph:
    """
    Class provides topological view of an individual network.

    Attributes:

        output_node (int) - index of output node

        nodes (np array) - vector of node indices

        input_size (int) - number of input channels

        reactions (list) - list of reaction objects

        stoichiometry (np array) - N x M matrix of stoichiometric coefficients

        node_key (dict) - maps state space dimension (key) to unique node id (value)

        edge_list (list) - list of edges, each defined as a (from, to, edge_dict) tuple

        up_edges (dict) - up-regulating edges in which keys are (from, to) tuples, values are edge weights

        down_edges (dict) - down-regulating edges in which keys are (from, to) tuples, values are edge weights

        graph (Networkx MultiDiGraph)

    """

    def __init__(self, network):
        """
        Inherits a mutable network then compiles an edge list and creates a networkx graph object.

        Args:

            network (network object)

        """

        # assign poperties
        self.output_node = network.output_node
        self.nodes = network.nodes
        self.input_size = network.input_size
        self.reactions = network.reactions
        self.stoichiometry = network.stoichiometry
        self.node_key = network.node_key

        # compile edge lists
        self.edge_list = self.get_edges()
        self.up_edges, self.down_edges = self.parse_edges()

        # create graph object
        self.graph = self.create_graph()

        # create node color dictionary
        self.node_colors = {
            'input': (206/256, 111/256, 111/256), # red
            'output': (138/256, 201/256, 228/256), # blue
            'reactive species': (138/256, 201/256, 228/256)} # blue

    def show_reactions(self):
        """
        Pretty-print table of all reactions within the network.
        """

        # create table of reactions
        rxn_table = []
        for num, rxn in enumerate(self.reactions):

            # assemble reactants
            reactants = [self.node_key[int(reactant)] for reactant, coeff in enumerate(rxn.stoichiometry) if coeff<0]
            reactants = ", ".join(str(reactant) for reactant in reactants)

            # assemble products
            products = [self.node_key[int(product)] for product, coeff in enumerate(rxn.stoichiometry) if coeff>0]
            products = ", ".join(str(product) for product in products)

            # for enzymatic reactions, use hill rate law
            if type(rxn) == EnzymaticReaction:
                rate_law = self.get_enzymatic_rate_law(rxn)

            elif type(rxn) == SumReaction:
                rate_law = self.get_sum_rxn_rate_law(rxn)

            elif type(rxn) == Coupling:
                rate_law = self.get_coupling_rate_law(rxn)

            else:
                rate_law = self.get_mass_action_rate_law(rxn)


            rate_constant = self.format_rate_constant(rxn)

            # assemble rate sensitivities to environmental conditions
            sensitivities = ''
            if rxn.temperature_sensitive is True:
                pass
                #sensitivities += 'Temp'
            if rxn.atp_sensitive is not False:
                if rxn.atp_sensitive is True or rxn.atp_sensitive==1:
                    sensitivities += 'ATP'
                else:
                    sensitivities += 'ATP^{:1.0f}'.format(rxn.atp_sensitive)
            if rxn.ribosome_sensitive is not False:
                if rxn.ribosome_sensitive is True or rxn.ribosome_sensitive==1:
                    sensitivities += ', RPs'
                else:
                    sensitivities += ', RPs^{:1.0f}'.format(rxn.ribosome_sensitive)

            # append reaction to table
            if rxn.rxn_type is None:
                name = 'Not Named'
            else:
                name = rxn.rxn_type
            rxn_table.append([name, reactants, products, rate_law, rate_constant])

            # for enzymatic reactions, add any repressors
            if type(rxn) in (EnzymaticReaction, Coupling):
                for repressor in rxn.repressors:
                    repressor_name = 'Repression of ' + rxn.rxn_type
                    repressor_rate_law = '1 - ' + self.get_enzymatic_rate_law(repressor)
                    rxn_table.append([repressor_name, '', '', repressor_rate_law, '', 'NA'])

        # print tables
        print(tabulate(rxn_table, headers=["Rxn", "Reactants", "Products", "Propensity", "Parameter"], numalign='center', stralign='center'))

    def get_sum_rxn_rate_law(self, rxn):
        rate_law = '[{:d}] - [{:d}]'.format(np.where(rxn.propensity == 1)[0][0], np.where(rxn.propensity == -1)[0][0])
        return rate_law

    def get_coupling_rate_law(self, rxn):
        if rxn.propensity.max() == 0:
            rate_law = ''
        else:
            base = np.where(rxn.propensity>0)[0][0]
            neighbors = np.where(rxn.propensity<0)[0]
            weight = (rxn.a * rxn.w) / (1+rxn.w*len(neighbors))

            if len(neighbors) > 1:
                coeff = '{:d}'.format(len(neighbors))
            else:
                coeff = ''

            rate_law = '{:0.3f} x ({:s}[{:d}]'.format(weight, coeff, base)
            for n in neighbors:
                rate_law += ' - [{:d}]'.format(n)
            rate_law += ')'
        return rate_law

    def get_mass_action_rate_law(self, rxn):
        """
        Compiles rate law in string form for a given reaction with mass-action kinetics

        Parameters:
            rxn (Reaction)

        Returns:
            rate_law (str)
        """

        # assemble mass-action rate law
        propensity = [('['+str(self.node_key[int(species)])+']')*int(coeff) for species, coeff in enumerate(rxn.propensity)]

        input_contribution = ''
        for i, dependence in enumerate(rxn.input_dependence):
            if dependence != 0:
                input_contribution += int(dependence) * '[IN_{:d}]'.format(i)

        rate_law = input_contribution + "".join(str(term) for term in propensity)

        return rate_law

    def format_rate_constant(self, rxn):
        rate_constant = '{:2.5f}'.format(rxn.k[0])
        if type(rxn) == EnzymaticReaction:
            for i, coeff in enumerate(rxn.rate_modifier):
                if coeff != 0:
                    rate_constant += ' + {:0.1f}[IN_{:d}]'.format(coeff, i)
        return rate_constant

    def get_enzymatic_rate_law(self, rxn):
        """
        Compiles rate law in string form for a given enzymatic reaction or repressor.

        Parameters:
            rxn (EnzymaticReaction or EnzymaticRepressor)

        Returns:
            rate_law (str) - in Hill form
        """
        substrate_contribution = ''
        propensity = ['['+str(self.node_key[int(species)])+']' if coeff!=0 else '' for species, coeff in enumerate(rxn.propensity)]
        weights = [str(int(coeff)) if coeff!=0 and coeff!=1 else '' for species, coeff in enumerate(rxn.propensity)]

        # combine weights and inputs
        for i, j in zip(weights, propensity):
            substrate_contribution += (i+j)

        # assemble substrates
        activity = ''
        for i, dependence in enumerate(rxn.input_dependence):
            if dependence != 0:
                if rxn.input_dependence.size == 1:
                    input_name = '[IN]'
                else:
                    input_name = '[IN_{:d}]'.format(i)

                coefficient = ''
                if dependence != 1:
                    coefficient = str(dependence)
                if i == 0:
                    activity += (coefficient + input_name)
                else:
                    activity += (coefficient + input_name)

        if len(rxn.active_substrates) > 0 and len(activity) > 0:
            activity += '+' + substrate_contribution
        elif len(rxn.active_substrates) > 0 and len(activity) == 0:
            activity += substrate_contribution

        # assemble rate law
        if rxn.n != 1:
            rate_law = activity+'^'+str(rxn.n)[:4] + '/(' + activity + '^' + str(rxn.n)[:4] + ' + ' + str(rxn.k_m)[:6]+ '^' + str(rxn.n)[:4]+')'
        else:
            rate_law = activity + '/(' + activity + ' + ' + str(rxn.k_m)[:6] + ')'

        return rate_law

    def get_direct_dependents(self, parent=None, input_dim=0):
        """
        Find all nodes directly dependent upon a single node. Node B is dependent upon node A if B appears within the stoichiometry
        vector of a reaction whose propensity includes A.

        Parameters:
            parent (int or None) - node whose dependents are desired, if None return nodes dependent on input(s)
            input_dim (int) - dimension of input vector used if parent is None

        Return:
            upregulated (set) - dependent nodes whose levels are increased by the parent node
            downregulated (set) - dependent nodes whose levels are decreased by the parent node
        """

        # initialize up/down regulated node lists
        upregulated, downregulated = set(), set()

        # iterate across all reactions
        for rxn in self.reactions:

            # initialize as non-dependent
            rxn_is_proportionally_dependent, rxn_is_inversely_dependent = False, False

            # check if reaction is input dependent
            if parent is None and rxn.input_dependence[input_dim] != 0:
                rxn_is_proportionally_dependent = True

            # check if parent node appears within reaction's propensity function
            elif parent is not None and parent in [self.node_key[node] for node, coeff in enumerate(rxn.propensity) if coeff != 0]:
                rxn_is_proportionally_dependent = True

            # if reaction is proportionally dependent upon the parent node, append all dependents to up/down regulated node lists
            if rxn_is_proportionally_dependent is True:
                upregulated = upregulated.union(set([self.node_key[dependent] for dependent, coeff in enumerate(rxn.stoichiometry) if coeff > 0]))
                downregulated = downregulated.union(set([self.node_key[dependent] for dependent, coeff in enumerate(rxn.stoichiometry) if coeff < 0]))

            # for enzymatic reactions, check all repressors
            if type(rxn) is EnzymaticReaction:
                for repressor in rxn.repressors:

                    # check if repressor is input dependent
                    if parent is None and repressor.input_dependence[input_dim] != 0:
                        rxn_is_inversely_dependent = True

                    # check if parent node appears within repressor's propensity function
                    elif parent is not None and parent in [self.node_key[node] for node, coeff in enumerate(repressor.propensity) if coeff != 0]:
                        rxn_is_inversely_dependent = True

            # if reaction is proportionally dependent upon the parent node, append all dependents to up/down regulated node lists
            if rxn_is_inversely_dependent is True:
                upregulated = upregulated.union(set([self.node_key[dependent] for dependent, coeff in enumerate(rxn.stoichiometry) if coeff < 0]))
                downregulated = downregulated.union(set([self.node_key[dependent] for dependent, coeff in enumerate(rxn.stoichiometry) if coeff > 0]))

        return upregulated, downregulated

    def get_edges(self):
        """
        Generates upregulating and downregulating edge sets for network visualization.

        Returns:
            edges (list) - each entry is a tuple containing (from, to, {edge_prop})
        """
        # initialize edge list
        edges = []

        # add input dependencies to edge list
        for input_dim in range(self.input_size):
            up, down = self.get_direct_dependents(parent=None, input_dim=input_dim)
            input_node = 'IN_' + str(input_dim)
            for node in up:
                edges.append((input_node, node, {'weight': 1}))
            for node in down:
                edges.append((input_node, node, {'weight': -1}))

        # add edges for all node-node dependencies
        for parent in self.nodes:
            up, down = self.get_direct_dependents(parent)
            for node in up:
                edges.append((parent, node, {'weight': 1}))
            for node in down:
                edges.append((parent, node, {'weight': -1}))

        return edges

    def parse_edges(self):
        """
        Sorts edge_list into up-regulating and down-regulating interactions.

        Returns:
            up_edges (dict) - keys are (from, to) tuples, values are edge weights
            down_edges (dict) - keys are (from, to) tuples, values are edge weights
        """

        up_edges, down_edges = {}, {}
        for edge in self.edge_list:
            if edge[2]['weight'] > 0:
                up_edges[(edge[0], edge[1])] = edge[2]['weight']
            else:
                down_edges[(edge[0], edge[1])] = edge[2]['weight']

        return up_edges, down_edges

    def get_degree_distributions(self):
        """
        Obtains in- and out- degree distributions for network excluding input node.

        Returns:
            in_degrees (np array) - distribution of in-degrees
            out_degrees (np array) - distribution of out-degrees
        """

        in_degrees = np.array([degree for node, degree in self.graph.in_degree_iter() if node != None])
        out_degrees = np.array([degree for node, degree in self.graph.out_degree_iter() if node != None])
        return in_degrees, out_degrees

    def create_graph(self):
        """
        Generates Networkx object of network topology.

        Returns:
            graph (Networkx MultiDiGraph)
        """

        # if network has no edges, abort
        if len(self.edge_list) == 0:
            print('Network has no edges.')

        # create directed graph with multiple parallel edges
        graph = nx.MultiDiGraph()

        # add nodes
        graph.add_nodes_from(['IN_'+str(input_dim) for input_dim in range(self.input_size)], node_type='input')
        if self.output_node is not None:
            graph.add_node(self.output_node, node_type='output')
        graph.add_nodes_from([node for node in self.nodes if node != self.output_node], node_type='reactive species')

        # add edges
        for edge in self.edge_list:
            graph.add_edge(edge[0], edge[1], weight=edge[2]['weight'])

        return graph

    def add_legend_to_network(self, ax, edge_alpha=0.5):
        """
        Adds legend to network visualization.

        Parameters:
            ax (axes object) - axis for legend
        """

        # create patch artists for legend
        legend_patches = []
        for node_type, color in sorted(self.node_colors.items()):
            if node_type != 'input':
                legend_patches.append(mpatches.Patch(color=color, label=node_type))

        # add legend
        upreg_line = mlines.Line2D([], [], color='g', linewidth=5, label='upregulation', alpha=edge_alpha)
        downreg_line = mlines.Line2D([], [], color='r', linewidth=5, label='downregulation', alpha=edge_alpha)
        _ = ax.legend(loc=10, handles=legend_patches+[upreg_line, downreg_line], ncol=2, prop={'size': 16})

    def visualize_graph(self, graph_layout='shell', ax=None, fig_size=(5, 4),
                        label_edges=False, show_legend=False, title=None, label_nodes=True,
                        node_size=2000, node_text_size=10, edge_width=3,
                        ):
        """
        Visualize network topology.

        Parameters:
            graph_layout (string) - method used to arrange network nodes in space
            label_edges (bool) - if True, add edge weight labels
            fig_size (tuple) - figure dimensions in inches
            show_legend (bool) - if True, add legend
            ax (axes object) - axis on which network is drawn
            title (str) - title to add to plot
        """

        if self.graph is None:
            print('No network object has been created.')
            return

        # display options
        node_alpha = 1
        edge_alpha = 0.5
        edge_text_pos = 0.4

        # create figure for visualization
        if ax is None:
            fig = plt.figure()
            fig.set_size_inches(*fig_size)
            ax = plt.gca()
        _ = ax.axis('off')

        # select graph layout scheme
        if graph_layout == 'spring':
            pos = nx.spring_layout(self.graph)
        elif graph_layout == 'spectral':
            pos = nx.spectral_layout(self.graph)
        elif graph_layout == 'random':
            pos = nx.random_layout(self.graph)
        elif graph_layout == 'circular':
            pos = nx.circular_layout(self.graph)
        elif graph_layout == 'FR':
            pos = nx.fruchterman_reingold_layout(self.graph)
        else:
            pos = nx.shell_layout(self.graph)

        # draw nodes)
        for node_type, color in self.node_colors.items():

            # get all nodes of corresponding type
            node_list = [node for node, attr in self.graph.node.items() if attr['node_type']==node_type]

            # networkx interprets 3 nodes as color mappable so may need to duplicate nodes if count=3
            if len(node_list) == 3:
                node_list *= 2

            # add nodes to graph
            nx.draw_networkx_nodes(self.graph, pos, ax=ax, nodelist=node_list, node_color=color, node_size=node_size, alpha=node_alpha)

        # add node labels
        node_labels = {'IN_'+str(input_dim): 'IN_'+str(input_dim) for input_dim in range(self.input_size)}
        for node in self.nodes:
            if node == self.output_node:
                node_labels[node] = 'output'
            else:
                node_labels[node] = str(node)
        if label_nodes is True:
            nx.draw_networkx_labels(self.graph, pos, ax=ax, labels=node_labels, font_size=node_text_size, fontweight='bold', color='k', ha='center')

        # add edges (green for upregulating, red for downregulating)
        nx.draw_networkx_edges(self.graph, pos, ax=ax, edgelist=self.up_edges.keys(), width=edge_width, alpha=edge_alpha, edge_color='g')
        nx.draw_networkx_edges(self.graph, pos, ax=ax, edgelist=self.down_edges.keys(), width=edge_width, alpha=edge_alpha, edge_color='r')

        # add edge labels
        if label_edges is True:
            nx.draw_networkx_edge_labels(self.graph, pos, ax=ax, edge_labels=self.up_edges, label_pos=edge_text_pos, font_size=10, fontweight='bold')
            nx.draw_networkx_edge_labels(self.graph, pos, ax=ax, edge_labels=self.down_edges, label_pos=edge_text_pos, font_size=10, fontweight='bold')

        # add legend
        if show_legend:
            fig = plt.gcf()
            fig.axes[0].change_geometry(2, 1, 1)
            gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
            ax.set_position(gs[0].get_position(fig))
            ax.set_subplotspec(gs[0])
            ax_legend = fig.add_subplot(gs[1])
            _, _ = ax.axis('off'), ax_legend.axis('off')
            self.add_legend_to_network(ax_legend)

        if title is not None:
            ax.set_title(title, fontsize=18)

        plt.tight_layout()

        return None

