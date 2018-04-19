__author__ = 'Sebi'

time_scaling = 1/60

conditions = {
    'normal': {'temperature': 1},
    'cold': {'temperature': 0.5},
    'hot': {'temperature': 1.5},
    'diabetic': {'metabolic_rate': 0.5},
    'minute': {'translation_capacity': 0.5}}

mutation_rates = {
    'add_node': 0.1,  # additions per node per generation
    'remove_node': 0.01,  # removals per node per generation
    'add_edge': 0.5,  # additions per node per generation
    'remove_edge': 0.1,  # removals per edge per generation
    'modify_rate': 0.5  # modifications per edge per generation
}

# define default rate constants (keys are rxn orders, values are rate constants)
rate_constants = {
    0: 1,
    1: 0.1,
    2: 0.01
}


class GeneExpressionParameters:
    """
    Default set of rate parameters for synthesis and degradation of transcripts and proteins.
    """
    def __init__(self, k_transcription=None, k_translation=None,
                 k_m=None, hill=None, baseline=None,
                 gamma_r=None, gamma_p=None
                 ):

        # transcription
        self.k_transcription = k_transcription if k_transcription is not None else 1
        self.k_m = k_m if k_m is not None else 5000
        self.hill = hill if hill is not None else 1
        self.baseline = baseline if baseline is not None else 0

        # translation
        self.k_translation = k_translation if k_translation is not None else 1

        # degradation
        self.gamma_r = gamma_r if gamma_r is not None else 0.01
        self.gamma_p = gamma_p if gamma_p is not None else 0.001


def gene_params(**kwargs):
    """
    Wrapper for instantiating gene expression parameter set.
    """
    parameters = GeneExpressionParameters(**kwargs)
    return parameters


class DimerizationParameters:
    """
    Default set of rate parameters for dimer formation and dissociation.
    """
    def __init__(self, k_association=None, k_dissociation=None, gamma=None):

        # dimer association and dissociation
        self.k_association = k_association if k_association is not None else 1
        self.k_dissociation = k_dissociation if k_dissociation is not None else 1

        # dimer decay
        self.gamma = gamma if gamma is not None else 0


def dimer_params(**kwargs):
    """
    Wrapper for instantiating dimerization parameter set.
    """
    parameters = DimerizationParameters(**kwargs)
    return parameters


class RepressorParameters:
    """
    Default set of rate parameters for repression.
    """
    def __init__(self, vmax=None, k_m=None, hill=None, k_linear=None):

        # transcriptional repression
        self.vmax = vmax if vmax is not None else 1
        self.k_m = k_m if k_m is not None else 1
        self.hill = hill if hill is not None else 1
        self.k_linear = k_linear if k_linear is not None else 1


def repressor_params(**kwargs):
    """
    Wrapper for instantiating repressor parameter set.
    """
    parameters = RepressorParameters(**kwargs)
    return parameters