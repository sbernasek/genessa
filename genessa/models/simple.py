import numpy as np

# intra-package python imports
from ..kinetics.massaction import MassAction
from .cells import Cell
from .genes import SimpleGene


class SimpleCell(Cell):
    """
    Class defines a simple cell with one or more proteins. Each protein is abstracted as a single dimension in a state space, with transcription lumped into a single linear synthesis term. All reaction rates are based on linear propensity functions.

    Attributes:

    Inherited Attributes:

        transcripts (dict) - {name: node_id} pairs - unused by default

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
        Instantiate simple cell with one or more protein coding genes.

        Args:

            genes (tuple) - names of genes

            I (int) - number of input channels

            kwargs: keyword arguments for add_genes

        """
        super().__init__(genes, I, **kwargs)

    def add_gene(self, **kwargs):
        """
        Add individual protein.

        kwargs: keyword arguments for Protein instantiation

        """

        protein = SimpleGene(**kwargs)

        # update nodes and reactions
        shift = self.nodes.size
        added_node_ids = np.arange(shift, shift+protein.nodes.size)
        self.update_reaction_dimensions(added_node_ids=added_node_ids)

        # add new nodes
        self.nodes = np.append(self.nodes, added_node_ids)
        self.reactions.extend([rxn.shift(shift) for rxn in protein.reactions])

        # update dictionaries
        self.transcripts.update({k: v+shift for k,v in protein.transcripts.items()})
        self.proteins.update({k: v+shift for k,v in protein.proteins.items()})

    def add_activation(self,
            protein,
            activator,
            k=1,
            growth_dependence=0,
            **labels):
        """
        Add gene activation reaction.

        Args:

            protein (str) - target protein name

            activator (str) - names of activating protein

            k (float) - activation rate constant

            growth_dependence (int) - log k / log growth

            labels (dict) - additional labels for reaction

        """

        # define reaction name
        labels['name'] = protein + ' activation'

        # define stoichiometry
        stoichiometry = np.zeros(self.nodes.size, dtype=np.int64)
        stoichiometry[self.proteins[protein]] = 1

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
                 atp_sensitive=False,
                 carbon_sensitive=False,
                 ribosome_sensitive=False,
                 growth_dependence=growth_dependence,
                 labels=labels)

        # add reaction
        self.reactions.append(rxn)
