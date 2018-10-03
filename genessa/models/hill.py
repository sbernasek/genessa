import numpy as np

# intra-package python imports
from ..kinetics.hill import Hill, Repressor
from .cells import Cell


class HillModel(Cell):
    """
    Class defines a cell with one or more protein coding genes.

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

    def add_transcription(self,
                          gene,
                          promoters=(),
                          repressors=None,
                          k=1,
                          k_m=1,
                          n=1,
                          baseline=0.,
                          **kwargs):
        """
        Add transcript synthesis reaction.

        Args:

            gene (str) - target gene name

            promoters (array like) - names of activating proteins

            repressors (array like) - Repressor instances

            k (float) - maximum transcription rate

            k_m (float) - michaelis menten constant

            n (float) - hill coefficients

            baseline (float) - baseline transcription rate

            kwargs: keyword arguments for reaction

        """


        # define stoichiometry
        stoichiometry = np.zeros(self.nodes.size, dtype=np.int64)
        stoichiometry[self.transcripts[gene]] = 1

        # define propensity
        propensity = np.zeros(self.nodes.size, dtype=np.int64)
        input_dependence = np.zeros(self.I, dtype=np.int64)

        if type(promoters) == str:
            promoters = (promoters,)

        for promoter in promoters:
            if 'IN' not in promoter:
                propensity[self.proteins[promoter]] = 1
            else:
                if '_' in promoter:
                    input_dependence[int(promoter.split('_')[-1])] = 1
                else:
                    input_dependence[0] = 1

        # define reaction
        rxn = Hill(stoichiometry=stoichiometry,
                   propensity=propensity,
                   input_dependence=input_dependence,
                   k=k,
                   k_m=k_m,
                   n=n,
                   baseline=baseline,
                   repressors=repressors,
                   rxn_type=gene+' transcription',
                   atp_sensitive=True,
                   ribosome_sensitive=False,
                   **kwargs)

        # add reaction
        self.reactions.append(rxn)
        self.update()

    def add_transcriptional_repressor(self,
                                      actuators,
                                      target,
                                      k_m=1,
                                      n=1):
        """
        Add transcriptional repressor.

        Args:

            actuators (array like) - list of actuating protein names

            target (str) - target gene name

            k_m (float) - michaelis menten constant

            n (float) - hill coefficient

        """

        # define propensity
        propensity = np.zeros(self.nodes.size, dtype=np.int64)
        input_dependence = np.zeros(self.I, dtype=np.int64)


        if type(actuators) == str:
            actuators = (actuators,)

        for actuator in actuators:
            if 'IN' not in actuator:
                propensity[self.proteins[actuator]] = 1
            else:
                if '_' in actuator:
                    input_dependence[int(actuator.split('_')[-1])] = 1
                else:
                    input_dependence[0] = 1

        # define repressor
        repressor = Repressor(propensity=propensity,
                              input_dependence=input_dependence,
                              k_m=k_m,
                              n=n)

        # add repressor
        for rxn in self.reactions:
            if rxn.__class__.__name__ == 'Hill' and target in rxn.rxn_type:
                rxn.add_repressor(repressor)
