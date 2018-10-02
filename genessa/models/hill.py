import numpy as np

# intra-package python imports
from ..kinetics.hill import Hill, Repressor
from .cells import Cell


class HillModel(Cell):
    """
    Class defines a cell with one or more protein coding genes.

    Attributes:

        genes (dict) - {name: node_id} pairs - unused by default

        transcripts (dict) - {name: node_id} pairs

        proteins (dict) - {name: node_id} pairs

        phosphorylated (dict) - {name: node_id} pairs

    Inherited Attributes:

        nodes (np.ndarray) - node indices

        reactions (list) - translation, mRNA decay, and protein decay reactions

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
        input_dependence = np.zeros(self.input_size, dtype=np.int64)
        for promoter in promoters:
            if 'IN' not in promoter:
                propensity[self.proteins[promoter]] = 1
            else:
                input_dependence[int(promoter.split('_')[-1])] = 1

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
        input_dependence = np.zeros(self.input_size, dtype=np.int64)
        for actuator in actuators:
            if 'IN' not in actuator:
                propensity[self.proteins[actuator]] = 1
            else:
                input_dependence[int(actuator.split('_')[-1])] = 1

        # define repressor
        repressor = Repressor(propensity=propensity,
                              input_dependence=input_dependence,
                              k_m=k_m,
                              n=n)

        # add repressor
        for rxn in self.reactions:
            if rxn.__class__.__name__ == 'Hill' and target in rxn.rxn_type:
                rxn.add_repressor(repressor)
