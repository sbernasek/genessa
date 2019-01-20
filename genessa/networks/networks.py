import numpy as np
from copy import deepcopy
import pickle

# intra-package python imports
from .ratelaws import RateLaws


class Network:
    """
    Class defines a network of interacting nodes. Nodes interact via stochastic processes called reactions. Each reaction has a propensity that may depend upon the current state of one or more other nodes.

    Attributes:

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
        Instantiate network.

        Args:

            N (int) - number of nodes

            I (int) - number of input channels

            reactions (list or rxn object) - reaction objects

        """

        # initialize network species
        self.nodes = np.arange(0, N)

        # initialize reaction list
        if reactions is None:
            self.reactions = []
        elif type(reactions) == list:
            self.reactions = reactions
        else:
            self.reactions = [reactions]

        # initialize stoichiometric matrix
        self.stoichiometry = None

        # set system size attributes
        self.I = I

    def __repr__(self):
        """ Print tabulated summary of reactions. """
        return self.print_reactions()

    @property
    def N(self):
        """ Number of nodes. """
        return self.nodes.size

    @property
    def M(self):
        """ Number of reactions. """
        return len(self.reactions)

    @property
    def node_key(self):
        """ Dictionary mapping node positional indices to node IDs. """
        return dict(enumerate(self.nodes))

    @property
    def ic(self):
        """ Default initial condition. """
        return np.zeros(self.N, dtype=np.int64)

    @staticmethod
    def load(path):
        """
        Load network from file.

        Args:

            path (str) - file path

        Returns:

            network (Network derivative)

        """
        with open(path, 'rb') as file:
            network = pickle.load(file)
        return network

    def save(self, path):
        """
        Save network to file. Networks are saved as serialized pickle objects.

        Args:

            path (str) - save destination

        """
        with open(path, 'wb') as file:
            pickle.dump(self, file, protocol=-1)

    def print_reactions(self):
        """ Print tabulated summary of reactions. """
        rate_laws = RateLaws(self.node_key, self.reactions)
        return rate_laws.__repr__()

    def constrain_ic(self, ic):
        """
        Constrains initial condition.

        Args:

            ic (np.ndarray[double]) - initial condition

        """
        pass

    def get_ic(self, ic=None):
        """
        Get initial condition for cell.

        Args:

            ic (array like or tuple) - initial condition

        Returns:

            ic (np.ndarray) - initial condition

        """

        # if IC is none, assume all genes in ground state
        if ic is None:
            ic = np.zeros(self.N, dtype=np.int64)

        # if IC is mean,var tuple, sample ICs from gaussian
        elif type(ic) == tuple:
            mean, var = ic
            ic = np.random.normal(mean, np.sqrt(var), size=mean.size).astype(int)
            ic[ic<0] = 0

        return ic

    def sort_rxns(self):
        """ Sorts reactions list by type. """
        sort_key = lambda rxn: str(rxn.__class__.__name__)
        self.reactions = sorted(self.reactions, key=sort_key)

    def compile_stoichiometry(self):
        """ Constructs N x M stoichiometric matrix. """
        self.stoichiometry = np.zeros((self.N, self.M), dtype=np.int64)
        for rxn_num, rxn in enumerate(self.reactions):
            self.stoichiometry[:, rxn_num] = rxn.stoichiometry

    def compile_rate_dependencies(self):
        """ Constructs M x N and M x I dependency matrices. """

        # compile state dependency matrix
        self.state_dependence = np.zeros((self.M, self.N), dtype=np.int64)
        for rxn_num, rxn in enumerate(self.reactions):
            self.state_dependence[rxn_num, :] = rxn.propensity

        # compile state dependency matrix
        self.input_dependence = np.zeros((self.M, self.I), dtype=np.int64)
        for rxn_num, rxn in enumerate(self.reactions):
            self.input_dependence[rxn_num, :] = rxn.input_dependence

    def update_reaction_dimensions(self,
                                   added_node_ids=None,
                                   removed_node_ids=None):
        """
        Update dimensions of all stoichiometry and propensity vectors.

        Args:

            added_node_ids (np.ndarray) - indices of added nodes

            removed_node_ids (np.ndarray) - indices of removed nodes

        """

        # add new nodes to stoichiometry and propensity vectors
        if added_node_ids is not None:
            insertion_points = np.searchsorted(self.nodes, added_node_ids)
            for rxn in self.reactions:
                rxn.stoichiometry = np.insert(rxn.stoichiometry, insertion_points, np.zeros(insertion_points.size))
                rxn.propensity = np.insert(rxn.propensity, insertion_points, np.zeros(insertion_points.size))

        # remove expired nodes from stoichiometry and propensity vectors
        if removed_node_ids is not None:
            removed_dimensions = np.array([dim for dim, node_id in self.node_key.items() if node_id in removed_node_ids])
            for rxn in self.reactions:
                rxn.stoichiometry = np.delete(rxn.stoichiometry, removed_dimensions)
                rxn.propensity = np.delete(rxn.propensity, removed_dimensions)

    def resize_inputs(self):
        """ Resize input dependence vector for all unaffected reactions. """

        for rxn in self.reactions:

            # resize reactions
            if rxn.input_dependence is None:
                rxn.input_dependence = np.zeros(self.I)
            elif rxn.input_dependence.max() == 0:
                rxn.input_dependence = np.zeros(self.I)

            # resize repressors
            if rxn.__class__.__name__ in ('Hill', 'Coupling'):
                for repressor in rxn.repressors:
                    if repressor.input_dependence is None:
                        repressor.input_dependence = np.zeros(self.I)

    def get_input_dependents(self, ic=None):
        """
        Find all nodes dependent upon the input. Node B is dependent upon node A if B appears within the stoichiometry vector of a reaction whose propensity includes A.

        Args:

            ic (array like) - initial conditions

        Returns:

            dependents (set) - nodes affected by input

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
        Find all nodes dependent upon a set of nodes. Node B is dependent upon node A if B appears within the stoichiometry vector of a reaction whose propensity includes A.

        Args:

            node_set (set) - nodes whose dependents are desired

            ic (array like) - initial conditions used to determine which states may be accessed. if None, just check for kinetic dependency

        Returns:

            dependents (set) - nodes affected by any node in node_set

        """

        # initialize list of dependent and candidate nodes
        dependents, dependents_prev = node_set, set()
        new_dependents = dependents.difference(dependents_prev)

        # recursively identify all downstream dependents
        while len(new_dependents) != 0:
            dependents_prev = deepcopy(dependents)
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
