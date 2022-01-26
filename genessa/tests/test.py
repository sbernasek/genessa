from unittest import TestCase

from genessa.kinetics.massaction import MassAction
from genessa.networks.networks import Network
from genessa.solver.stochastic import MonteCarloSimulation


class TestDecay(TestCase):
    """
    Tests for basic stochastic simulation of a first-order decay process.
    """

    @classmethod
    def setUpClass(cls):
        """ Initialize test Solver instance. """

        # instantiate decay reaction
        decay_rxn = MassAction(stoichiometry=[-1], k=2)

        # instantiate reaction network
        network = Network(N=1, reactions=[decay_rxn])

        # instantiate solver
        solver = MonteCarloSimulation(network, ic=[100])
        cls.solver = solver

    def test_solve(self):
        """ Run simulation and make sure initial state is non-zero.  """
        timeseries = self.solver.run(N=100, duration=3, dt=0.01)
        self.assertTrue(timeseries.mean[0].sum() > 0)
