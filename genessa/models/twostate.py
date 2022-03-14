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

        dosage (int) - dosage of each gene (used to set initial conditions)

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
                 dosage=2,
                 **kwargs):
        """
        Instantiate twostate cell with one or more protein coding genes.

        Args:

            genes (tuple) - names of genes

            I (int) - number of input channels

            dosage (int) - dosage of each gene (used to set initial conditions)

            kwargs: keyword arguments for add_genes

        """
        self.off_states = {}
        self.on_states = {}
        self.dosage = dosage
        super().__init__(genes, I, **kwargs)

    @property
    def ic(self):
        """ Default initial condition. """
        ic = np.zeros(self.N, dtype=np.int64)
        for off_state in self.off_states.values():
            ic[off_state] = self.dosage
        return ic

    def constrain_ic(self, ic):
        """
        Constrains initial condition to specified gene dosage.

        Args:

            ic (np.ndarray[double]) - initial condition

        """

        for gene in self.off_states.keys():

            # get current dosage specified by initial condition
            off_state = self.off_states[gene]
            on_state = self.on_states[gene]
            currrent_dosage = ic[off_state] + ic[on_state]

            # if dosage is correct, leave as is
            if currrent_dosage == self.dosage:
                continue

            # if dosage is too low, add to the off state
            elif currrent_dosage < self.dosage:
                ic[off_state] += (self.dosage - currrent_dosage)

            # if dosage is too high, remove from the on state first
            while ic[off_state] + ic[on_state] > self.dosage:
                if ic[on_state] > 0:
                    ic[on_state] -= 1
                else:
                    ic[off_state] -= 1

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
                 atp_sensitive=atp_sensitive,
                 carbon_sensitive=carbon_sensitive,
                 ribosome_sensitive=ribosome_sensitive,
                 labels=labels)

        # add reaction
        self.reactions.append(rxn)

    def add_transcriptional_repressor(self,
            actuator,
            target,
            k=1.,
            atp_sensitive=False,
            carbon_sensitive=False,
            ribosome_sensitive=False,
            **labels
        ):
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
                 carbon_sensitive=carbon_sensitive,
                 ribosome_sensitive=ribosome_sensitive,
                 labels=labels)

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
        propensity[self.on_states[gene]] = 1

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
