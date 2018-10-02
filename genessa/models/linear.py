import numpy as np

# intra-package python imports
from ..kinetics.massaction import MassAction
from .cells import Gene, Cell


class LinearGene:
    """
    Class defines a single gene coding a protein product. All reaction rates are based on linear propensity functions.

    System dimensions:
        0: Gene
        1: mRNA
        2: Protein

    Attributes:

        nodes (np.ndarray) - node indices

        genes (dict) - single (name: node_id) pair

        transcripts (dict) - single {name: node_id} pair

        proteins (dict) - single {name: node_id} pair

        reactions (list) - translation, mRNA decay, and protein decay reactions

    """
    def __init__(self, name='gene', k1=1, k2=1, g0=1, g1=1, g2=1):
        """
        Create gene along with translation, mRNA decay, and protein decay reactions.

        Args:

            name (str) - gene name

            k1 (float) - transcription rate constant

            k2 (float) - translation rate constant

            g0 (float) - gene decay rate constant

            g1 (float) - transcript decay rate constant

            g2 (float) - protein decay rate constant

        """

        self.nodes = np.arange(3)
        self.genes = {name: 0}
        self.transcripts = {name: 1}
        self.proteins = {name: 2}

        # define gene names
        gene_name = name[0].lower()
        protein_name = name.upper()

        # define reactions
        self.reactions = [

            # gene decay
            MassAction([-1, 0, 0],
                       [1, 0, 0],
                       k=g0,
                       rxn_type=gene_name+' off rate',
                       atp_sensitive=False,
                       ribosome_sensitive=False),

            # transcript synthesis
            MassAction([0, 1, 0],
                       [1, 0, 0],
                       k=k1,
                       rxn_type=gene_name+' transcription',,
                       atp_sensitive=True,
                       ribosome_sensitive=False),

            # transcript decay
            MassAction([0, -1, 0],
                       [0, 1, 0],
                       k=g1,
                       rxn_type=gene_name+' decay',
                       atp_sensitive=False,
                       ribosome_sensitive=False),

            # protein synthesis
            MassAction([0, 0, 1],
                       [0, 1, 0],
                       k=k2,
                       rxn_type=protein_name+' translation',
                       atp_sensitive=True,
                       ribosome_sensitive=True),

            # protein decay
            MassAction([0, 0, -1],
                       [0, 0, 1],
                       k=g2,
                       rxn_type=protein_name+' decay',
                       atp_sensitive=True,
                       ribosome_sensitive=True)]


class LinearModel(Cell):
    """
    Class defines a cell with one or more protein coding genes. All reaction rates are based on linear propensity functions.

    Attributes:

        genes (dict) - {name: node_id} pairs - unused by default

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
        Instantiate linear cell with one or more protein coding genes.

        Args:

            genes (tuple) - names of genes

            num_inputs (int) - number of input channels

            kwargs: keyword arguments for add_genes

        """
        self.genes = {}
        super().__init__(genes, num_inputs, **kwargs)

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
