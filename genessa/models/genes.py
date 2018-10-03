import numpy as np

# intra-package python imports
from ..kinetics.massaction import MassAction


class Gene:
    """

    Class defines a single gene coding a protein product.

    System dimensions:
        0: mRNA
        1: Protein

    Attributes:

        nodes (np.ndarray) - node indices

        transcripts (dict) - single {name: node_id} pair

        proteins (dict) - single {name: node_id} pair

        reactions (list) - translation, mRNA decay, and protein decay reactions

    """
    def __init__(self, name='gene', k=1, g1=1, g2=1):
        """
        Create gene along with translation, mRNA decay, and protein decay reactions.

        Args:

            name (str) - gene name

            k (float) - translation rate constant

            g1 (float) - transcript decay rate constant

            g2 (float) - protein decay rate constant

        """

        self.nodes = np.arange(2)
        self.transcripts = {name: 0}
        self.proteins = {name: 1}

        # define gene names
        gene_name = name[0].lower()
        mrna_decay = gene_name + ' decay'
        protein_name = name.upper()
        translation = protein_name + ' translation'
        protein_decay = protein_name + ' decay'

        # define reactions
        self.reactions = [

            # transcript decay
            MassAction([-1, 0],
                       [1, 0],
                       k=g1,
                       rxn_type=mrna_decay,
                       atp_sensitive=False,
                       ribosome_sensitive=False),

            # protein synthesis
            MassAction([0, 1],
                       [1, 0],
                       k=k,
                       rxn_type=translation,
                       atp_sensitive=True,
                       ribosome_sensitive=True),

            # protein decay
            MassAction([0, -1],
                       [0, 1],
                       k=g2,
                       rxn_type=protein_decay,
                       atp_sensitive=True,
                       ribosome_sensitive=True)]


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
                       rxn_type=gene_name+' transcription',
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
                       rxn_type=gene_name+' transcription',
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