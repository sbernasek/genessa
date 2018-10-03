import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import networkx as nx

# intra-package python imports
from .ratelaws import RateLaws


class Graph:
    """
    Class provides topological view of an individual network.

    Attributes:

        output_node (int) - index of output node

        nodes (np.ndarray) - vector of node indices

        reactions (list) - list of reaction objects

        stoichiometry (np.ndarray) - stoichiometric coefficients, (N, M)

        N (int) - number of nodes

        M (int) - number of reactions

        I (int) - number of input channels

        node_key (dict) - maps state dimension (key) to unique node id (value)

        edge_list (list) - edges defined as a (from, to, edge_dict) tuple

        up_edges (dict) - up-regulating edges in which keys are (from, to) tuples, values are edge weights

        down_edges (dict) - down-regulating edges in which keys are (from, to) tuples, values are edge weights

        graph (Networkx MultiDiGraph)

    """

    def __init__(self, network):
        """
        Inherits a mutable network then compiles an edge list and creates a networkx graph object.

        Args:

            network (MutableNetwork)

        """

        # assign poperties
        self.output_node = network.output_node
        self.nodes = network.nodes
        self.reactions = network.reactions
        self.stoichiometry = network.stoichiometry
        self.node_key = network.node_key

        # define system size
        self.N = self.nodes.size
        self.M = len(self.reactions)
        self.I = network.input_size

        # compile edge lists
        self.edge_list = self.get_edges()
        self.up_edges, self.down_edges = self.parse_edges()

        # create graph object
        self.graph = self.create_graph()

        # create node color dictionary
        self.node_colors = {
            'input': (206/256, 111/256, 111/256), # red
            'output': (138/256, 201/256, 228/256), # blue
            'reactive': (138/256, 201/256, 228/256)} # blue








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
            if type(rxn) == Hill:
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
        for input_dim in range(self.I):
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
            in_degrees (np.ndarray) - distribution of in-degrees
            out_degrees (np.ndarray) - distribution of out-degrees
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
        graph.add_nodes_from(['IN_'+str(input_dim) for input_dim in range(self.I)], node_type='input')
        if self.output_node is not None:
            graph.add_node(self.output_node, node_type='output')
        graph.add_nodes_from([node for node in self.nodes if node != self.output_node], node_type='reactive')

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
        node_labels = {'IN_'+str(input_dim): 'IN_'+str(input_dim) for input_dim in range(self.I)}
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
