from .reactions import Reaction, LinearReaction, EnzymaticReaction, EnzymaticRepressor
from .networks import MutableNetwork, Graph
from .parameters import gene_params, dimer_params, repressor_params

import numpy as np
import networkx as nx


"""
TO DO:

1. expand print_reactions table
2. add repressors to graph

"""

class Cell(MutableNetwork):
    """
    Class inherits a mutable network to which it adds mRNA/protein identities to nodes.

    Attributes:
        name (str) - name of class type
        unique_node_id (int) - counter for unique node numbers
        node_key (dict) - maps state space dimension (key) to unique node id (value)
    """

    def __init__(self, coding_genes=1, non_coding_genes=0, output_node=1):
        """
        Inherits a network and adds mutation capabilities.

        Parameters:
            coding_genes (int) - number of transcript-protein pairs with which cell is initialized
            non_coding_genes (int) - number of non-coding transcript species with which cell is initialized
            output_node (int) - index of output
        """

        MutableNetwork.__init__(self, nodes=0, reactions=[], output_node=output_node)

        self.transcripts = []
        self.proteins = []

        # add coding genes (first one assumed input dependent)
        promoters, k_m = ('input',), 0.5
        for i in range(coding_genes):
            self.add_protein_coding_gene(promoters, k_m=k_m)
            promoters, k_m = (), None

        # add non-coding genes
        for _ in range(non_coding_genes):
                self.add_non_coding_gene()

    def __repr__(self):
        """
        Visualize graph.
        """
        self.show()
        return str(type(self))

    def show(self, **kwargs):
        """
        Visualize regulatory network.
        """
        graph = GRN(self)
        graph.visualize_graph(**kwargs)

    def to_json(self):
        """
        Return json-serialized cell object.
        """
        js = super(Cell, self).to_json()
        js['transcripts'] = self.transcripts
        js['proteins'] = self.proteins
        return js

    @staticmethod
    def from_json(js):
        """
        Instantiate Cell object from json-serialized dictionary.
        """
        cell = Cell(coding_genes=0)

        # get each attribute from json dictionary
        cell.output_node = js['output_node']
        cell.nodes = np.array(js['nodes'])
        cell.unique_node_id = js['unique_node_id']
        cell.stoichiometry = np.array(js['stoichiometry'])
        cell.node_key = {int(key): int(val) for key, val in js['node_key'].items()}
        cell.reactions = [Reaction.from_json(rxn) if rxn_type=='mass_action' else EnzymaticReaction.from_json(rxn)
                             for rxn, rxn_type in zip(js['reactions'], js['rxn_types'])]
        cell.transcripts = js['transcripts']
        cell.proteins = js['proteins']

        return cell

    def add_species(self, gamma, **kwargs):
        """
        Add new species to cell.

        Parameters:
            gamma (float) - degradation rate constant
            kwargs - keyword arguments for various rate scaling sensitivities

        Returns:
            species_id (int) - unique id of new species
        """

        # get species id
        species_id = self.unique_node_id

        # add new node to network
        self.add_nodes(additions=1)

        # add degradation
        if gamma > 0:
            stoichiometry = np.zeros(self.nodes.size)
            stoichiometry[-1] = -1
            input_dependence = np.zeros(self.input_size)
            degradation = LinearReaction(stoichiometry=stoichiometry, k=gamma, input_dependence=input_dependence, **kwargs)
            self.reactions.append(degradation)

        return species_id

    def add_promoter(self, gene, promoters=(), k_transcription=None, k_m=None, hill=None, baseline=None):
        """
        Add enzymatic transcription reaction.

        Parameters:
            gene (int) - index of mrna being transcribed
            promoters (array like) - list of indices of promoters
            k_transcription, k_m, hill (float) - maximal transcription rate, michaelis constant, and hill coefficient
            baseline (float) - baseline transcription rate in absence of promoter
        """

        # compile stoichiometric vector
        stoichiometry = np.zeros(self.nodes.size)
        stoichiometry[gene] = 1

        # parse promoters
        propensity, input_dependence = np.zeros(self.nodes.size), np.zeros(self.input_size)
        for promoter in promoters:
            if type(promoter) == str:
                if '_' in promoter:
                    input_dependence[int(promoter.split('_')[-1])] = 1
                else:
                    input_dependence[0] = 1
            else:
                propensity[promoter] += 1

        # get rate constants
        p = gene_params(k_transcription=k_transcription, k_m=k_m, hill=hill, baseline=baseline)

        # add enzymatic reaction to network
        transcription = EnzymaticReaction(stoichiometry=stoichiometry,
                                          propensity=propensity,
                                          input_dependence=input_dependence,
                                          rate_constant=p.k_transcription, k_m=p.k_m, hill=p.hill, baseline=p.baseline,
                                          rxn_type='transcription',
                                          temperature_sensitive=True, atp_sensitive=True)
        self.reactions.append(transcription)

    def add_protein_coding_gene(self, promoters=(), k_transcription=None, k_m=None, hill=None, baseline=None, k_translation=None, gamma_r=None, gamma_p=None, *kwargs):
        """
        Add new protein-coding gene to cell.

        Parameters:
            promoters (array like) - list of indices of promoters
            k_transcription, k_m, n (float) - maximal transcription rate, michaelis constant, and hill coefficient
            baseline (float) - base transcription rate in absence of promoter
            k_translation (float) - translation rate constant
            gamma_r, gamma_p (float) - transcript and protein degradation constants

        Returns:
            transcript_id, protein_id (int) - unique node ids of newly created species
        """

        # get rate constants
        p = gene_params(k_transcription=k_transcription, k_m=k_m, hill=hill, baseline=baseline, k_translation=k_translation,
                                           gamma_r=gamma_r, gamma_p=gamma_p)

        # add mrna and protein species
        transcript_id = self.add_species(gamma=p.gamma_r, rxn_type='transcript decay', temperature_sensitive=True)
        self.transcripts.append(transcript_id)
        protein_id = self.add_species(gamma=p.gamma_p, rxn_type='protein decay', temperature_sensitive=True, atp_sensitive=True)
        self.proteins.append(protein_id)

        # add transcription reaction
        if len(promoters) > 0:
            self.add_promoter(transcript_id, promoters, p.k_transcription, p.k_m, p.hill, p.baseline)

        # add translation
        stoichiometry = np.zeros(self.nodes.size)
        stoichiometry[protein_id] = 1
        propensity = np.zeros(self.nodes.size)
        propensity[transcript_id] = 1
        input_dependence = np.zeros(self.input_size)
        translation = LinearReaction(stoichiometry=stoichiometry, propensity=propensity, k=p.k_translation, rxn_type='translation', input_dependence=input_dependence,
                               temperature_sensitive=True, atp_sensitive=True, ribosome_sensitive=True)
        self.reactions.append(translation)

        return transcript_id, protein_id

    def add_non_coding_gene(self, promoters=(), k_transcription=None, k_m=None, hill=None, baseline=None, gamma_r=None):
        """
        Add non-coding gene to cell.

        Parameters:
            promoters (array like) - list of indices of promoters
            k_transcription, k_m, n (float) - maximal transcription rate, michaelis constant, and hill coefficient
            baseline (float) - base transcription rate in absence of promoter
            gamma_r (float) - transcript degradation rate

        Returns:
            transcript_id (int) - unique node id of newly created species
        """

        # get rate constants
        p = gene_params(k_transcription=k_transcription, k_m=k_m, hill=hill, baseline=baseline,
                                           gamma_r=gamma_r)

        # add mrna species
        transcript_id = self.add_species(gamma=p.gamma_r, rxn_type='transcript decay', temperature_sensitive=True)
        self.transcripts.append(transcript_id)

        # add transcription
        self.add_promoter(transcript_id, promoters,
                          k_transcription=p.k_transcription, k_m=p.k_m, hill=p.hill, baseline=p.baseline)

        return transcript_id

    def add_dimer(self, species1, species2, reversible=True, k_association=None, k_dissociation=None, gamma=None):
        """
        Adds dimer as well as association/dissociation reactions.

        Parameters:
            species1, species2 (int) - indices of dimer subcomponents
            reversible (bool) - if True, add reversible product to species list
            k_association, k_dissociation (float) - rate constants for association/dissociation reactions
            gamma (float) - degradation constant

        Returns:
            dimer_id (int) - id of newly created dimer species
        """

        # get dimerization rate parameters
        p = dimer_params(k_association=k_association, k_dissociation=k_dissociation, gamma=gamma)

        # create dimer
        dimer_id = None
        if reversible:
            dimer_id = self.add_species(gamma=p.gamma, rxn_type='dimer decay', temperature_sensitive=True, atp_sensitive=True)

            # determine whether species 1 and 2 are proteins or transcripts, then add dimer to appropriate species list
            if species1 in self.proteins and species2 in self.proteins:
                self.proteins.append(dimer_id)
            elif species1 in self.transcripts and species2 in self.transcripts:
                self.transcripts.append(dimer_id)
            else:
                raise ValueError('Species are not recognize or are of conflicting type.')

        # compile stoichiometry for forward and reverse reactions
        association_stoichiometry = np.zeros(self.nodes.size)
        if reversible:
            association_stoichiometry[dimer_id] += 1
        for component in (species1, species2):
            association_stoichiometry[component] -= 1
        dissociation_stoichiometry = -1 * association_stoichiometry

        # add association/dissociation reactions to network
        association = Reaction(stoichiometry=association_stoichiometry, k=p.k_association, rxn_type='association', temperature_sensitive=True)
        dissociation = Reaction(stoichiometry=dissociation_stoichiometry, k=p.k_dissociation, rxn_type='dissociation', temperature_sensitive=True)
        if reversible:
            self.reactions.extend([association, dissociation])
        else:
            self.reactions.append(association)

        return dimer_id

    def add_transcriptional_repressor(self, repressor, target, k_m=None, hill=None):
        """
        Adds transcriptional repressor to all reactions enzymatically synthesizing the specified target.

        Parameters:
            repressor (int) - substrate acting to repress transcription
            target (int) - gene under repression
            k_m (float) - michaelis constant
            hill (float) - hill coefficient
        """

        # get repressor parameters
        p = repressor_params(k_m=k_m, hill=hill)

        # determine substrate weights
        propensity = np.zeros(self.nodes.size)
        if type(repressor) == str:
            input_dependence = np.zeros(self.input_size)
            if '_' in repressor:
                input_dependence[int(repressor.split('_')[-1])] = 1
            else:
                input_dependence[0] = 1
        else:
            input_dependence = 0
            propensity[repressor] = 1

        # instantiate repressor
        repression = EnzymaticRepressor(propensity=propensity, input_dependence=input_dependence, k_m=p.k_m, hill=p.hill)

        # add repression to any reaction in which the target is enzymatically synthesized
        for enzymatic_rxn in [rxn for rxn in self.reactions if type(rxn) == EnzymaticReaction]:
            if enzymatic_rxn.stoichiometry[target] > 0:
                enzymatic_rxn.add_repressor(repression)

    def add_sequential_proteins(self, actuator, n=1, **kwargs):
        """
        Creates sequential cascade of protein synthesis via transcription-translation.

        Parameters:
            actuator (int) - index of protein initiating cascade
            n (int) - cascade length
            kwargs (dict) - rate parameters for intermediate gene expression

        Returns:
            sensor (int) - index of sensor at the end of the cascade
        """

        # create intermediates
        for stage in range(n):
            _, actuator = self.add_protein_coding_gene(promoters=(actuator,), **kwargs)

        return actuator

    def add_transcriptional_feedback(self, sensor=None, target=None, intermediates=0, k_m=None, hill=None):
        """
        Adds feedback loop imparting transcriptional repression.

        Parameters:
            sensor (int) - index of species whose level is measured
            target (int) - index of transcript whose synthesis is repressed
            intermediates (int) - number of intermediate gene-synthesis stages
            k_m, hill (float, float) - repressor michaelis constant and hill coefficient
        """

        # assume sensor is first protein if none specified
        if sensor is None:
            sensor = self.proteins[0]

        # assume target is first transcript if none specified
        if target is None:
            target = self.transcripts[0]

        # create intermediate signaling cascade
        actuator = self.add_sequential_proteins(sensor, intermediates)

        # implement transcriptional repression
        self.add_transcriptional_repressor(actuator, target, k_m=k_m, hill=hill)

    def add_post_transcriptional_feedback(self, sensor=None, target=None, intermediates=0, rate_constant=None, intermediate_kwargs={}):
        """
        Adds feedback loop imparting post-transcriptional repression via an mRNA-dimerization mechanism.

        Parameters:
            sensor (int) - index of species whose level is measured
            target (int) - index of target transcript
            intermediates (int) - number of intermediate gene-synthesis stages
            rate_constant (float) - rate of dimerization
            intermediate_kwargs (dict) - rate parameters for intermediate gene expression
        """

        # assume sensor is first protein if none specified
        if sensor is None:
            sensor = self.proteins[0]

        # assume target is first transcript if none specified
        if target is None:
            target = self.transcripts[0]

        # create intermediate signaling cascade
        actuator = self.add_sequential_proteins(sensor, intermediates, **intermediate_kwargs)

        # creates noncoding miRNA
        actuator = self.add_non_coding_gene(promoters=(actuator,))

        # implement antithetical destruction
        _ = self.add_dimer(target, actuator, reversible=False, k_association=rate_constant, k_dissociation=0)

    def add_post_translational_feedback(self, target=None, reversible=False, intermediates=1, rate_constant=None):
        """
        Adds feedback loop imparting post-translational repression via a dimerization mechanism.

        Parameters:
            target (int) - index of controlled protein
            reversible (bool) - if True, add a reversible dimer species
            intermediates (int) - number of intermediate gene-synthesis stages
            rate_constant (float) - rate of dimerization
        """

        # assume target is first protein if none specified
        if target is None:
            target = self.proteins[0]

        # create intermediate signaling cascade
        actuator = self.add_sequential_proteins(target, intermediates)

        # implement dimerization
        _ = self.add_dimer(target, actuator, reversible=reversible, k_association=rate_constant, k_dissociation=0)

    def add_catalytic_post_translational_feedback(self, target=None, intermediates=0, vmax=None, k_m=None, hill=None):
        """
        Adds feedback loop imparting post-translational repression via a catalytic degradation mechanism.

        Parameters:
            target (int) - index of controlled protein
            intermediates (int) - number of intermediate gene-synthesis stages
            vmax, k_m, hill (float) - catalytic degradation rate constants
        """

        # assume target is first protein if none specified
        if target is None:
            target = self.proteins[0]

        # create intermediate signaling cascade
        actuator = self.add_sequential_proteins(target, intermediates)

        # create catalytic degradation reaction
        stoichiometry = np.zeros(self.nodes.size)
        stoichiometry[target] = -1
        propensity = np.zeros(self.nodes.size)
        propensity[actuator] = 1

        # get reaction parameters
        p = repressor_params(vmax=vmax, k_m=k_m, hill=hill)

        # add catalytic degradation reaction
        catalytic_degradation = EnzymaticReaction(stoichiometry=stoichiometry, propensity=propensity,
                 rate_constant=p.vmax, k_m=p.k_m, hill=p.hill, baseline=0, rxn_type='enzymatic degradation',
                 temperature_sensitive=True)
        self.reactions.append(catalytic_degradation)

    def add_catalytic_degradation(self, actuator=None, target=None, vmax=None, k_m=None, hill=None):
        """
        Adds feedback loop imparting post-translational repression via a catalytic degradation mechanism.

        Parameters:
            actuator (int) - index of catalytic species
            target (int) - index of controlled protein
            vmax, k_m, hill (float) - catalytic degradation rate constants
        """

        # assume target is first protein if none specified
        if target is None:
            target = self.proteins[0]

        # create catalytic degradation reaction
        stoichiometry = np.zeros(self.nodes.size)
        stoichiometry[target] = -1
        propensity = np.zeros(self.nodes.size)
        propensity[actuator] = 1

        # get reaction parameters
        p = repressor_params(vmax=vmax, k_m=k_m, hill=hill)

        # add catalytic degradation reaction
        catalytic_degradation = EnzymaticReaction(stoichiometry=stoichiometry, propensity=propensity,
                 rate_constant=p.vmax, k_m=p.k_m, hill=p.hill, baseline=0, rxn_type='enzymatic degradation',
                 temperature_sensitive=True)
        self.reactions.append(catalytic_degradation)

    def add_linear_feedback(self, sensor, target, rate_constant=None, **kwargs):
        """
        Adds linear negative feedback applied to mRNA level.

        Parameters:
            sensor (int) - index of species whose level is measured
            target (int) - index of target species
            rate_constant (float) - feedback strength
            kwargs - rate sensitivity to environmental conditions
        """

        # determine stoichiometry and propensity
        stoichiometry, propensity = np.zeros(len(self.nodes)), np.zeros(len(self.nodes))
        stoichiometry[target] = -1
        propensity[sensor] = 1

        # set parameters
        p = repressor_params(k_linear=rate_constant)

        # add feedback reaction
        feedback = LinearReaction(stoichiometry=stoichiometry, propensity=propensity, k=p.k_linear, rxn_type='linear feedback', **kwargs)
        self.reactions.append(feedback)


class GRN(Graph):
    """
    Class inherits graph object and adds transcript/protein labels to provide topological view of an individual cell's
    gene regulatory network.

    Attributes:
        output_node (int) - index of output node
        nodes (np array) - vector of node indices
        reactions (list) - list of reaction objects
        stoichiometry (np array) - N x M matrix of stoichiometric coefficients
        node_key (dict) - maps state space dimension (key) to unique node id (value)
        transcripts (array like) - transcript ids
        proteins (array like) - protein ids
        edge_list (list) - list of edges, each defined as a (from, to, edge_dict) tuple
        up_edges (dict) - up-regulating edges in which keys are (from, to) tuples, values are edge weights
        down_edges (dict) - down-regulating edges in which keys are (from, to) tuples, values are edge weights
        graph (Networkx MultiDiGraph)
    """

    def __init__(self, cell):
        """
        Inherits a cell then compiles a GRN-edge list and creates a networkx graph object.

        Parameters:
            cell (Cell object)
        """
        self.transcripts = cell.transcripts
        self.proteins = cell.proteins
        Graph.__init__(self, cell)
        self.graph = self.create_graph()

        # node colors
        self.node_colors = {
            'input': (155/256, 220/256, 150/256), # green
            'gene': (238/256, 167/256, 246/256), # purple
            'transcript': (138/256, 201/256, 228/256), # blue
            'protein': (206/256, 111/256, 111/256)} # red

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

        #graph.add_node(self.output_node, node_type='input/output')
        graph.add_nodes_from([node for node in self.nodes if node not in self.transcripts + self.proteins], node_type='gene')
        graph.add_nodes_from(self.transcripts, node_type='transcript')
        graph.add_nodes_from(self.proteins, node_type='protein')

        #graph.add_nodes_from([node for node in self.proteins if node != self.output_node], node_type='protein')

        # add edges
        for edge in self.edge_list:
            graph.add_edge(edge[0], edge[1], weight=edge[2]['weight'])

        return graph

