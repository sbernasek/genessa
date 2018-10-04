import numpy as np
from copy import deepcopy

# intra-package python imports
from ..networks.networks import Network
from ..kinetics.hill import Repressor
from ..kinetics.coupling import Coupling
from .cells import Cell


class CoupledCell(Cell):
    """
    Class defines a cell with one or more protein coding genes. Transcriptional kinetics are dictated by coupling to other cells.

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
                          k,
                          coupled=[],
                          a=0,
                          w=0,
                          rep=None,
                          k_m=1,
                          n=1):
        """
        Add transcript synthesis reaction. Transcription rate is dependent upon coupling with other cells.

        Args:

            gene (str) - target gene name

            k (float) - transcription rate constant

            coupled (list) - target gene index in coupled cells

            a (float) - coupling strength

            w (float) - edge weight

            rep (str) - name of protein that represses transcription

            k_m (float) - michaelis constant for repressor

            n (float) - hill coefficient for repressor

        """

        # define stoichiometry
        stoichiometry = np.zeros(self.N, dtype=np.int64)
        stoichiometry[self.transcripts[gene]] = 1

        # define propensity as total difference between gene and neighbors
        propensity = np.zeros(self.N, dtype=np.int64)
        for neighbor in coupled:
            propensity[self.transcripts[gene]] -= 1
            propensity[neighbor] = 1

        # add repressors
        repressors = None
        if rep is not None:
            repressor_propensity = np.zeros(self.N, dtype=np.int64)
            repressor_propensity[self.proteins[rep]] = 1
            repressors = [Repressor(repressor_propensity, None, k_m=k_m, n=n)]

        # define synthesis reaction
        rxn = Coupling(stoichiometry,
                       propensity,
                       k=k,
                       a=a,
                       w=w,
                       repressors=repressors,
                       rxn_type=gene+' transcription')

        # add reaction
        self.reactions.append(rxn)
        self.update()


class CoupledCells(CoupledCell):
    """
    Collection of coupled cells.

    Attributes:

        template (CoupledCell) - template for cell population

        replicates (int) - number of cells in population

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

    def __init__(self, cell, replicates=1):

        # store template cell
        self.template = deepcopy(cell)

        # store dimensionality
        self.replicates = replicates

        # expand reactions
        reactions = self.get_reactions(cell, replicates)

        # instantiate population wide network
        Network.__init__(self, self.template.N*replicates, reactions=reactions)

        # expand labels
        self.expand_labels()

    def __repr__(self):
        """ Prints list of reactions for template cell. """
        print('{:d} cells of the form:'.format(self.replicates))
        return self.template.__repr__()

    def index(self, node_index):
        """
        Returns template node index for all cells.

        Args:

            node_index (int) - index of node in template cell

        Returns:

            node_indices (np.ndarray[int]) - same node for all cells

        """
        return node_index + np.arange(self.replicates)*self.template.N

    def expand_labels(self):
        """ Adds a replicate ID layer to transcript and protein labels."""
        expand = lambda states: {i: {k: v+self.template.N*i for k, v in states.items()} for i in range(self.replicates)}
        self.transcripts = expand(self.template.transcripts)
        self.proteins = expand(self.template.proteins)

    @staticmethod
    def get_reactions(cell, replicates=1):
        """
        Returns list of reactions for entire cell population.

        Args:

            cell (CoupledCell) - template cell

            replicates (int) - number of replicates

        Returns:

            reactions (list) - reactions with expanded dimensionality

        """

        dimensionality = cell.N * replicates
        reactions = []
        for i in range(replicates):
            ind = slice(i*cell.N, (i+1)*cell.N)
            cell_id = 'Cell {:d}: '.format(i)
            for rxn in cell.reactions:

                # create expanded vectors for current reactions
                stoichiometry = np.zeros(dimensionality, dtype=int)
                propensity = np.zeros(dimensionality, dtype=int)
                stoichiometry[ind] = rxn.stoichiometry
                propensity[ind] = rxn.propensity

                # compile kwargs for reactions
                kw = {'stoichiometry': stoichiometry,
                      'propensity': propensity,
                      'rxn_type': cell_id+rxn.rxn_type,
                      'parameters': rxn.parameters}

                # add MassAction reaction
                if rxn.__class__.__name__ == 'MassAction':
                    expanded_rxn = rxn.__class__(input_dependence=rxn.input_dependence, k=rxn.k[0], **kw)

                # add Hill reaction
                elif rxn.__class__.__name__ == 'Hill':

                    # shift repressors
                    repressors = []
                    for rep in rxn.repressors:
                        if rep.propensity.size > 0:
                            p = np.zeros(dimensionality, dtype=int)
                            p[ind] = rep.propensity
                        else:
                            p=None
                        i = rep.input_dependence
                        rep_kw = dict(k_m=rep.k_m,
                                      n=rep.n,
                                      parameters=rep.parameters)
                        repressor = Repressor(p, i, **rep_kw)
                        repressors.append(repressor)

                    expanded_rxn = rxn.__class__(input_dependence=rxn.input_dependence, repressors=repressors, k=rxn.k[0], k_m=rxn.k_m, n=rxn.n, baseline=rxn.baseline, rate_modifier=rxn.rate_modifier, **kw)

                # add Coupling reaction
                elif rxn.__class__.__name__ == 'Coupling':

                    # shift repressors
                    repressors = []
                    for rep in rxn.repressors:
                        if rep.propensity.size > 0:
                            p = np.zeros(dimensionality, dtype=int)
                            p[ind] = rep.propensity
                        else:
                            p=None
                        i = rep.input_dependence
                        rep_kw = dict(k_m=rep.k_m,
                                      n=rep.n,
                                      parameters=rep.parameters)
                        repressor = Repressor(p, i, **rep_kw)
                        repressors.append(repressor)

                    expanded_rxn = rxn.__class__(repressors=repressors, k=rxn.k[0], a=rxn.a, w=rxn.w, **kw)

                else:
                    raise TypeError('Reaction type {} not recognized.'.format(rxn.__class__))

                reactions.append(expanded_rxn)
        return reactions

    def add_transcription(self,
                          cell,
                          gene,
                          k,
                          coupled=[],
                          a=0,
                          w=0,
                          rep=None,
                          k_m=1,
                          n=1):
        """
        Add transcript synthesis reaction for an individual cell. Transcription rate is dependent upon coupling with other cells.

        Args:

            cell (int) - cell index

            gene (str) - target gene name

            k (float) - transcription rate constant

            coupled (list) - indices of coupled cells

            a (float) - coupling strength

            w (float) - edge weight

            rep (str) - name of protein that represses transcription

            k_m (float) - michaelis constant for repressor

            n (float) - hill coefficient for repressor

        """

        # define stoichiometry
        stoichiometry = np.zeros(self.N, dtype=np.int64)
        stoichiometry[self.transcripts[cell][gene]] = 1

        # define propensity as total difference between gene and neighbors
        propensity = np.zeros(self.N, dtype=np.int64)
        for neighbor in coupled:
            propensity[self.transcripts[cell][gene]] += -1
            propensity[self.transcripts[neighbor][gene]] = 1

        # add repressors
        repressors = None
        if rep is not None:
            repressor_propensity = np.zeros(self.N, dtype=np.int64)
            repressor_propensity[self.proteins[cell][rep]] = 1
            repressors = [Repressor(repressor_propensity, None, k_m=k_m, n=n)]

        # define synthesis reaction
        rxn = Coupling(stoichiometry,
                       propensity,
                       k=k,
                       a=a,
                       w=w,
                       repressors=repressors,
                       rxn_type='Cell {}: '.format(cell)+gene+' transcription')

        # add synthesis reaction
        self.reactions.append(rxn)
        self.update()

    def add_coupling(self, adjacency, **kwargs):
        """
        Add coupling from an adjacency matrix.

        Args:

            adjacency (np.ndarray[bool]) - adjacency matrix

            kwargs: keyword arguments for transcription

        """

        # add coupling to each cell in the population
        for cell, row in enumerate(adjacency):
            coupled = row.nonzero()[0]
            self.add_transcription(cell, coupled=coupled, **kwargs)

        # add equivalent synthesis reaction to the template cell
        self.template.add_transcription(coupled=[], **kwargs)

    def add_random_coupling(self, a=0.01, **kwargs):
        """

        Couple population randomly.

        Args:

            a (float) - coupling strength

            kwargs: keyword arguments for transcription reaction

        """

        # define coupling topology
        adjacency = np.random.randint(0, 2, size=2*(self.replicates,))
        np.fill_diagonal(adjacency, 0)

        # add coupling
        self.add_coupling(adjacency, a=a, **kwargs)

    def add_dense_coupling(self, a=0.01, undirected=True, **kwargs):
        """

        Couple population with dense links.

        Args:

            a (float) - coupling strength

            undirected (bool) - flag for reciprocal coupling

            kwargs: keyword arguments for transcription reaction

        """

        # define coupling topology
        adjacency = np.ones(2*(self.replicates,), dtype=int)
        np.fill_diagonal(adjacency, 0)
        if not undirected:
            adjacency[np.triu_indices(self.replicates)] = 0

        # add coupling
        self.add_coupling(adjacency, a=a, **kwargs)

    def add_sparse_coupling(self, a=0.01, **kwargs):
        """

        Add transcript synthesis without any coupling.

        Args:

            a (float) - coupling strength

            kwargs: keyword arguments for transcription reaction

        """

        # define coupling topology
        adjacency = np.zeros(2*(self.replicates,), dtype=int)

        # add coupling
        self.add_coupling(adjacency, a=a, **kwargs)
