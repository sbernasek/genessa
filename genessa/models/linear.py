import numpy as np

# intra-package python imports
from ..kinetics.massaction import MassAction
from .cells import Cell
from .genes import LinearGene


class LinearCell(Cell):
    """
    Class defines a cell with one or more protein coding genes. All reaction rates are based on linear propensity functions.

    Attributes:

        genes (dict) - {name: node_id} pairs - unused by default

    Inherited Attributes:

        transcripts (dict) - {name: node_id} pairs

        proteins (dict) - {name: node_id} pairs

        phosphorylated (dict) - {name: node_id} pairs

        nodes (np.ndarray) - vector of node indices

        node_key (dict) - {state dimension: node id} pairs

        reactions (list) - list of reaction objects

        stoichiometry (np.ndarray) - stoichiometric coefficients, (N,M)

        N (int) - number of nodes

        M (int) - number of reactions

        I (int) - number of inputs

    """

    def __init__(self, genes=(), I=1, **kwargs):
        """
        Instantiate linear cell with one or more protein coding genes.

        Args:

            genes (tuple) - names of genes

            I (int) - number of input channels

            kwargs: keyword arguments for add_genes

        """
        self.genes = {}
        super().__init__(genes, I, **kwargs)

    def add_gene(self, **kwargs):
        """
        Add individual gene.

        kwargs: keyword arguments for Gene instantiation

        """

        gene = LinearGene(**kwargs)

        # update nodes and reactions
        shift = self.nodes.size
        added_node_ids = np.arange(shift, shift+gene.nodes.size)
        self.update_reaction_dimensions(added_node_ids=added_node_ids)

        # add new nodes
        self.nodes = np.append(self.nodes, added_node_ids)
        self.reactions.extend([rxn.shift(shift) for rxn in gene.reactions])

        # update dictionaries
        self.genes.update({k: v+shift for k,v in gene.genes.items()})
        self.transcripts.update({k: v+shift for k,v in gene.transcripts.items()})
        self.proteins.update({k: v+shift for k,v in gene.proteins.items()})

    def add_activation(self,
            gene,
            activator,
            k=1,
            atp_sensitive=False,
            carbon_sensitive=False,
            ribosome_sensitive=False,                        
            **labels
        ):
        """
        Add gene activation reaction.

        Args:

            gene (str) - target gene name

            activator (str) - names of activating protein

            k (float) - activation rate constant

            labels (dict) - additional labels for reaction

        """

        # define reaction name
        labels['name'] = gene+' activation'

        # define stoichiometry
        stoichiometry = np.zeros(self.nodes.size, dtype=np.int64)
        stoichiometry[self.genes[gene]] = 1

        # define propensity
        propensity = np.zeros(self.nodes.size, dtype=np.int64)
        input_dependence = np.zeros(self.I, dtype=np.int64)
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
            atp_sensitive=atp_sensitive,
            carbon_sensitive=carbon_sensitive,
            ribosome_sensitive=ribosome_sensitive,
            labels=labels
        )

        # add reaction
        self.reactions.append(rxn)

    def add_transcriptional_promoter(
        self,
        gene,
        k=1,
        atp_sensitive=True,
        carbon_sensitive=True,
        ribosome_sensitive=False,
        **labels
        ):
        """
        Add transcriptional promoter reaction.

        Args:

            gene (str) - target gene name

            k (float) - activation rate constant

            labels (dict) - additional labels for reaction

        """

        # define reaction name
        labels['name'] = gene+' transcription'

        # define stoichiometry
        stoichiometry = np.zeros(self.nodes.size, dtype=np.int64)
        stoichiometry[self.transcripts[gene]] = 1

        # define propensity
        propensity = np.zeros(self.nodes.size, dtype=np.int64)
        propensity[self.genes[gene]] = 1

        # define reaction
        rxn = MassAction(
            stoichiometry=stoichiometry,
            propensity=propensity,
            input_dependence=None,
            k=k,
            atp_sensitive=atp_sensitive,
            carbon_sensitive=carbon_sensitive,
            ribosome_sensitive=ribosome_sensitive,
            labels=labels
        )

        # add reaction
        self.reactions.append(rxn)

    def add_translational_promoter(
        self,
        gene,
        k=1,
        atp_sensitive=True,
        carbon_sensitive=True,
        ribosome_sensitive=True,
        **labels
        ):
        """
        Add translational promoter reaction.

        Args:

            gene (str) - target gene name

            k (float) - activation rate constant

            labels (dict) - additional labels for reaction

        """

        # define reaction name
        labels['name'] = gene+' translation'

        # define stoichiometry
        stoichiometry = np.zeros(self.nodes.size, dtype=np.int64)
        stoichiometry[self.proteins[gene]] = 1

        # define propensity
        propensity = np.zeros(self.nodes.size, dtype=np.int64)
        propensity[self.transcripts[gene]] = 1

        # define reaction
        rxn = MassAction(
            stoichiometry=stoichiometry,
            propensity=propensity,
            input_dependence=None,
            k=k,
            atp_sensitive=atp_sensitive,
            carbon_sensitive=carbon_sensitive,
            ribosome_sensitive=ribosome_sensitive,
            labels=labels
        )

        # add reaction
        self.reactions.append(rxn)
