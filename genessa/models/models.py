from .cells import Cell

from .networks import MutableNetwork
from .algorithms import MonteCarloSimulation
from .timeseries import GaussianModel
from .reactions import Reaction, EnzymaticReaction
from .parameters import two_state_model_defaults, hill_model_defaults




class HillModel(MutableNetwork):
    """
    Creates a network exhibiting Hill transcription kinetics followed by linear protein synthesis.

    System dimensions:
        0: mRNA
        1: Protein
        2-9: null
    """

    def __init__(self, rate_constants=None):

        self.name = 'Hill Model'

        # if no rate constants provided, use defaults
        rc = rate_constants
        if rate_constants is None:
            rc = hill_model_defaults

        # assign rate constants
        vmax, k_m, n = rc['vmax'], rc['k_m'], rc['n']
        k_translation = rc['k_translation']
        gamma_r, gamma_p = rc['gamma_r'], rc['gamma_p']

        # define reactions
        rxns = [

            # transcript synthesis/decay
            EnzymaticReaction(stoichiometry=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], input_dependence=1, rate_constant=vmax, k_m=k_m, n=n,
                         baseline_rate=0, atp_sensitive=True),
            Reaction(stoichiometry=[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0], k=gamma_r),

            # protein synthesis/decay
            Reaction(stoichiometry=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0], propensity=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     k=k_translation, atp_sensitive=True, ribosome_sensitive=True),
            Reaction(stoichiometry=[0, -1, 0, 0, 0, 0, 0, 0, 0, 0], k=gamma_p)]

        # instantiate cell with two-state model
        MutableNetwork.__init__(self, nodes=10, reactions=rxns)


class FeedbackSystem(Cell):
    """
    Creates a network consisting of a single gene-protein pair subject to feedback mechanisms.

    System dimensions:
        0: target mRNA
        1: target protein
        2+: regulatory elements
    """

    def __init__(self, gene=0, transcript=0, protein=0, protein_catalytic=0):
        """
        Parameters:
            gene (int) - number of transcriptional regulation motifs
            transcript (int) - number of post-transcriptional regulation motifs
            protein (int) - number of post-translational regulation motifs utilizing a dimerization mechanism
            protein_catalytic (int) - number of post-translational regulation motifs utilizing catalytic degradation
        """

        # instantiate cell
        Cell.__init__(self, coding_genes=1, non_coding_genes=0, output_node=1)

        # add transcriptional feedback mechanisms
        for _ in range(gene):
            self.add_transcriptional_feedback(self.output_node, self.transcripts[0], intermediates=1, k_m=1000, hill=1)

        # add post-transcriptional feedback mechanisms
        for _ in range(transcript):
            self.add_post_transcriptional_feedback(self.output_node, self.transcripts[0], intermediates=0, k_association=0.001)

        # add post-translational feedback mechanisms utilizing dimerization mechanism
        for _ in range(protein):
            self.add_antithetic_post_translational_feedback(self.output_node, intermediates=1, k_association=0.001)

        # add post-translational feedback mechanisms utilizing catalytic degradation mechanism
        for _ in range(protein_catalytic):
            self.add_catalytic_post_translational_feedback(self.output_node, intermediates=1, vmax=10, k_m=1, hill=1)

    def get_trajectories(self, input_signal, ic=None, condition=None, num_trials=10, dt=1, duration=2400, method='bd-leaping'):
        """
        Run stochastic simulation.

        Parameters:
            input_signal (function) - returns input value for any timepoint
            condition (str) - rate scaling conditions, e.g. 'normal' or 'diabetic'
            ic (array like) - initial conditions
            num_trials (int) - number of independent samples
            dt (float) - time step
            duration (float) - simulation duration, in seconds
            method (str) - solution algorithm

        Returns:
            trajectories (np array) - time series for all states
            model (st.norm object) - GaussianModel object
        """

        simulation = MonteCarloSimulation(self, ic=ic, dt=dt, duration=duration, input_function=input_signal)
        trajectories = simulation.run_trials(num_trials=num_trials, condition=condition, method=method)
        model = GaussianModel.from_timeseries(trajectories)

        return trajectories, model
