import numpy as np

# intra-package python imports
from ..kinetics.massaction import MassAction
from .cells import Gene, Cell


class TwoStateGene:
    """
    Class defines a single gene coding a protein product.

    System dimensions:
        0: Gene Off
        1: Gene On
        2: mRNA
        3: Protein

    Attributes:

        nodes (np.ndarray) - node indices

        off_states (dict) - single (name: node_id) pair

        on_states (dict) - single (name:node_id) pair

        transcripts (dict) - single {name: node_id} pair

        proteins (dict) - single {name: node_id} pair

        reactions (list) - translation, mRNA decay, and protein decay reactions

    """
    def __init__(self, name='gene', k0=0, k1=1, k2=1, g0=1, g1=1, g2=1):
        """
        Create gene along with translation, mRNA decay, and protein decay reactions.

        Args:

            name (str) - gene name

            k0 (float) - gene on rate

            k1 (float) - transcription rate constant

            k2 (float) - translation rate constant

            g0 (float) - gene off rate

            g1 (float) - transcript decay rate constant

            g2 (float) - protein decay rate constant

        """

        self.nodes = np.arange(4)
        self.off_states = {name: 0}
        self.on_states = {name: 1}
        self.transcripts = {name: 2}
        self.proteins = {name: 3}

        # define gene names
        gene_name = name[0].lower()
        protein_name = name.upper()

        # define reactions
        self.reactions = [

            # gene activation
            MassAction([-1, 1, 0, 0],
                       [1, 0, 0, 0],
                       k=k0,
                       rxn_type=gene_name+' on rate',
                       atp_sensitive=False,
                       ribosome_sensitive=False),

            # gene deactivation
            MassAction([1, -1, 0, 0],
                       [0, 1, 0, 0],
                       k=g0,
                       rxn_type=gene_name+' off rate',
                       atp_sensitive=False,
                       ribosome_sensitive=False),

            # transcript synthesis
            MassAction([0, 0, 1, 0],
                       [0, 1, 0, 0],
                       k=k1,
                       rxn_type=gene_name+' transcription',,
                       atp_sensitive=True,
                       ribosome_sensitive=False),

            # transcript decay
            MassAction([0, 0, -1, 0],
                       [0, 0, 1, 0],
                       k=g1,
                       rxn_type=gene_name+' decay',
                       atp_sensitive=False,
                       ribosome_sensitive=False),

            # protein synthesis
            MassAction([0, 0, 0, 1],
                       [0, 0, 1, 0],
                       k=k2,
                       rxn_type=protein_name+' translation',
                       atp_sensitive=True,
                       ribosome_sensitive=True),

            # protein decay
            MassAction([0, 0, 0, -1],
                       [0, 0, 0, 1],
                       k=g2,
                       rxn_type=protein_name+' decay',
                       atp_sensitive=True,
                       ribosome_sensitive=True)]


class TwoStateModel(Cell):
    """
    Class defines a cell with one or more protein coding genes. Transcription is based on a twostate model.

    Attributes:

        off_states (dict) - {name: node_id} pairs

        on_states (dict) - {name: node_id} pairs

    Inherited Attributes:

        nodes (np.ndarray) - node indices

        reactions (list) - translation, mRNA decay, and protein decay reactions

        transcripts (dict) - {name: node_id} pairs

        proteins (dict) - {name: node_id} pairs

        phosphorylated (dict) - {name: node_id} pairs

    """
    def __init__(self,
                 genes=(),
                 num_inputs=1,
                 **kwargs):
        """
        Instantiate twostate cell with one or more protein coding genes.

        Args:

            genes (tuple) - names of genes

            num_inputs (int) - number of input channels

            kwargs: keyword arguments for add_genes

        """
        self.off_states = {}
        self.on_states = {}
        super().__init__(genes, num_inputs, **kwargs)

    def add_gene(self, **kwargs):
        """
        Add individual gene.

        kwargs: keyword arguments for Gene instantiation

        """

        gene = TwoStateGene(**kwargs)

        # update nodes and reactions
        shift = self.nodes.size
        added_node_ids = np.arange(shift, shift+gene.nodes.size)
        self.update_reaction_dimensions(added_node_ids=added_node_ids)

        # add new nodes
        self.nodes = np.append(self.nodes, added_node_ids)
        self.reactions.extend([rxn.shift(shift) for rxn in gene.reactions])

        # update dictionaries
        self.off_states.update({k: v+shift for k,v in gene.off_states.items()})
        self.on_states.update({k: v+shift for k,v in gene.on_states.items()})
        self.transcripts.update({k: v+shift for k,v in gene.transcripts.items()})
        self.proteins.update({k: v+shift for k,v in gene.proteins.items()})
        self.update()

    def add_activation(self,
                        gene,
                        activator,
                        k=1,
                        **kwargs):
        """
        Add transcript synthesis reaction.

        Args:

            gene (str) - target gene name

            activator (str) - names of activating protein

            k (float) - activation rate constant

            kwargs: keyword arguments for reaction

        """

        # define stoichiometry
        stoichiometry = np.zeros(self.nodes.size, dtype=np.int64)
        stoichiometry[self.on_states[gene]] = 1
        stoichiometry[self.off_states[gene]] = -1

        # define propensity
        propensity = np.zeros(self.nodes.size, dtype=np.int64)
        propensity[self.off_states[gene]] = 1
        input_dependence = np.zeros(self.input_size, dtype=np.int64)
        if 'IN' in activator:
            if '_' in activator:
                input_dependence[int(activator.split('_')[-1])] = 1
            else:
                input_dependence[0] = 1
        else:
            propensity[self.proteins[activator]] = 1

        # define reaction
        rxn = MassAction(
                 stoichiometry=stoichiometry,
                 propensity=propensity,
                 input_dependence=input_dependence,
                 k=k,
                 rxn_type=gene+' activation',
                 atp_sensitive=False,
                 ribosome_sensitive=False)

        # add reaction
        self.reactions.append(rxn)
        self.update()

      def add_transcriptional_repressor(self,
                                        actuator,
                                        target,
                                        k=1.,
                                        atp_sensitive=True,
                                        ribosome_sensitive=True):
        """
        Add transcriptional repressor.

        Args:

            actuator (str) - actuating substrate

            target (float) - target gene

            k (float) - maximum degradation rate

            atp_sensitive (bool) - scale rate with metabolism

            ribosome_sensitive (bool) - scale rate with ribosomes

            kwargs: keyword arguments for MassAction instantiation

        """

        # define stoichiometry
        stoichiometry = np.zeros(self.nodes.size, dtype=np.int64)
        stoichiometry[self.on_states[gene]] = -1
        stoichiometry[self.off_states[gene]] = 1

        # define propensity
        propensity = np.zeros(self.nodes.size, dtype=np.int64)
        propensity[self.on_states[gene]] = 1
        input_dependence = np.zeros(self.input_size, dtype=np.int64)
        if 'IN' in actuator:
            if '_' in actuator:
                input_dependence[int(actuator.split('_')[-1])] = 1
            else:
                input_dependence[0] = 1
        else:
            propensity[self.proteins[actuator]] = 1

        # define reaction
        rxn = MassAction(
                 stoichiometry=stoichiometry,
                 propensity=propensity,
                 input_dependence=input_dependence,
                 k=k,
                 rxn_type=gene+' repression',
                 atp_sensitive=atp_sensitive,
                 ribosome_sensitive=ribosome_sensitive)

        # add reaction
        self.reactions.append(rxn)
        self.update()
