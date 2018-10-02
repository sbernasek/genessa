import numpy as np

# intra-package python imports
from ..kinetics.marbach import Transcription
from .cells import Cell


class MarbachModel(Cell):
    """
    Class defines a cell with one or more protein coding genes. Transcriptional kinetics are dictated by the Marbach model.

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
                          modules=None,
                          k=1,
                          alpha=None,
                          perturbed=False,
                          **kwargs):
        """
        Add transcript synthesis reaction.

        Args:

            gene (str) - target gene name

            modules (list) - RegulatoryModule instances

            k (float) - transcription rate constant

            alpha (array like) - alpha values

            perturbed (bool) - if True, rate is subject to perturbation

            kwargs: keyword arguments for reaction

        """

        # define stoichiometry
        stoichiometry = np.zeros(self.nodes.size, dtype=np.int64)
        stoichiometry[self.transcripts[gene]] = 1

        # define synthesis reaction
        rxn = Transcription(stoichiometry,
                            modules,
                            k=k,
                            alpha=alpha,
                            perturbed=perturbed,
                            rxn_type=gene+' transcription',
                            atp_sensitive=True,
                            ribosome_sensitive=False,
                            **kwargs)

        # add reaction
        self.reactions.append(rxn)
        self.update()
