import numpy as np

# intra-package python imports
from ..kinetics.massaction import MassAction
from .cells import Cell
from .genes import LinearGene


class LinearModel(Cell):
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
                 rxn_type=gene+' activation',
                 atp_sensitive=False,
                 ribosome_sensitive=False)

        # add reaction
        self.reactions.append(rxn)
        self.update()

      def add_linear_feedback(self,
                            actuator,
                            target,
                            mode='protein',
                            k=1.,
                            atp_sensitive=True,
                            ribosome_sensitive=True):
        """
        Add linear feedback term.

        Args:

            actuator (str) - actuating substrate

            target (float) - target gene

            mode (str) - 'protein', 'transcript', or 'gene'

            k (float) - maximum degradation rate

            atp_sensitive (bool) - scale rate with metabolism

            ribosome_sensitive (bool) - scale rate with ribosomes

            kwargs: keyword arguments for MassAction instantiation

        """

        # define stoichiometry
        stoichiometry = np.zeros(self.nodes.size, dtype=np.int64)
        if mode == 'protein':
            stoichiometry[self.proteins[target]] = -1
        elif mode == 'transcript':
            stoichiometry[self.transcripts[target]] = -1
        elif mode == 'gene':
            stoichiometry[self.genes[target]] = -1

        # define propensity
        assert actuator in self.proteins.keys(), 'Actuator not recognized.'
        propensity = np.zeros(self.nodes.size, dtype=np.int64)
        propensity[self.proteins[actuator]] = 1

        # define reaction
        rxn = MassAction(stoichiometry,
                         propensity,
                         input_dependence=None,
                         k=k,
                         rxn_type=mode+'  feedback',
                         atp_sensitive=atp_sensitive,
                         ribosome_sensitive=ribosome_sensitive,
                         **kwargs)

        # add reaction
        self.reactions.append(rxn)
        self.update()
