import numpy as np

# intra-package python imports
from ..kinetics.massaction import MassAction
from .cells import Cell
from .genes import TwoStateGene


class TwoStateCell(Cell):
    """
    Class defines a cell with one or more protein coding genes. Transcription is based on a twostate model.

    Attributes:

        off_states (dict) - {name: node_id} pairs

        on_states (dict) - {name: node_id} pairs

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
    def __init__(self,
                 genes=(),
                 I=1,
                 **kwargs):
        """
        Instantiate twostate cell with one or more protein coding genes.

        Args:

            genes (tuple) - names of genes

            I (int) - number of input channels

            kwargs: keyword arguments for add_genes

        """
        self.off_states = {}
        self.on_states = {}
        super().__init__(genes, I, **kwargs)

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

    def add_activation(self,
                        gene,
                        activator,
                        k=1,
                        labels={}):
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
        stoichiometry[self.on_states[gene]] = 1
        stoichiometry[self.off_states[gene]] = -1

        # define propensity
        propensity = np.zeros(self.nodes.size, dtype=np.int64)
        propensity[self.off_states[gene]] = 1
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
                 ribosome_sensitive=False,
                 labels=labels)

        # add reaction
        self.reactions.append(rxn)

    def add_transcriptional_repressor(self,
                                    actuator,
                                    target,
                                    k=1.,
                                    atp_sensitive=True,
                                    ribosome_sensitive=True,
                                    labels={}):
        """
        Add transcriptional repressor.

        Args:

            actuator (str) - actuating substrate

            target (float) - target gene

            k (float) - maximum degradation rate

            atp_sensitive (bool) - scale rate with metabolism

            ribosome_sensitive (bool) - scale rate with ribosomes

            labels (dict) - additional labels for reaction

        """

        # define reaction name
        labels['name'] = target+' repression'

        # define stoichiometry
        stoichiometry = np.zeros(self.N, dtype=np.int64)
        stoichiometry[self.on_states[target]] = -1
        stoichiometry[self.off_states[target]] = 1

        # define propensity
        propensity = np.zeros(self.N, dtype=np.int64)
        propensity[self.on_states[target]] = 1
        input_dependence = np.zeros(self.I, dtype=np.int64)
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
                 atp_sensitive=atp_sensitive,
                 ribosome_sensitive=ribosome_sensitive,
                 labels=labels)

        # add reaction
        self.reactions.append(rxn)
