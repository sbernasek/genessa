import numpy as np
import networkx as nx
from copy import copy

# intra-package python imports
from ..systems.networks import Network
from ..kinetics.massaction import MassAction
from ..kinetics.hill import Hill
from .genes import Gene


class Cell(Network):
    """
    Class defines a cell with one or more protein coding genes.

    Attributes:

        transcripts (dict) - {name: node_id} pairs

        proteins (dict) - {name: node_id} pairs

        phosphorylated (dict) - {name: node_id} pairs

    Inherited Attributes:

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
        Instantiate cell with one or more protein coding genes.

        Args:

            genes (tuple) - names of genes

            I (int) - number of input channels

            kwargs: keyword arguments for add_genes

        """

        # instantiate network
        super().__init__(0, I=I)

        # initialize species dictionaries
        self.transcripts = {}
        self.proteins = {}
        self.phosphorylated = {}

        # add genes
        self.add_genes(genes, **kwargs)

    def get_ic(self, ic=None):
        """
        Get initial condition for cell.

        Args:

            ic (array like or tuple) - initial condition

        Returns:

            ic (np.ndarray) - initial condition

        """

        # if IC is none, assume all genes in ground state
        if ic is None:
            ic = np.zeros(self.N, dtype=np.int64)

        # if IC is mean,var tuple, sample ICs from gaussian
        elif type(ic) == tuple:
            mean, var = ic
            ic = np.random.normal(mean, np.sqrt(var), size=mean.size).astype(int)
            ic[ic<0] = 0

        return ic

    def update(self):
        """ Update node key. """
        self.node_key = {i: int(n_id) for i, n_id in enumerate(self.nodes)}

    def add_genes(self, names, **kwargs):
        """
        Add multiple genes.

        Args:

            names (array like) - names of genes

            kwargs: keyword arguments for Gene instantiation

        """
        for name in names:
            self.add_gene(name=name, **kwargs)

    def add_gene(self, **kwargs):
        """
        Add individual gene.

        kwargs: keyword arguments for Gene instantiation

        """

        gene = Gene(**kwargs)

        # update nodes and reactions
        shift = self.N
        added_node_ids = np.arange(shift, shift+gene.nodes.size)
        self.update_reaction_dimensions(added_node_ids=added_node_ids)

        # add new nodes
        self.nodes = np.append(self.nodes, added_node_ids)
        self.reactions.extend([rxn.shift(shift) for rxn in gene.reactions])

        # update dictionaries
        self.transcripts.update({k: v+shift for k,v in gene.transcripts.items()})
        self.proteins.update({k: v+shift for k,v in gene.proteins.items()})
        self.update()

    def add_phosphorylation(self,
                            base,
                            kf=1.,
                            Kf=1.,
                            kr=1.,
                            Kr=1.,
                            g=0.,
                            name=None):
        """
        Add phosphorylated protein product.

        Args:

            base (str) - name of protein that is phosphorylated

            kf (float) - maximum phosphorylation rate

            Kf (float) - substrate conc. at half maximal phosphorylation rate

            kr (float) - maximum dephosphorylation rate

            Kr (float) - substrate conc. at half maximal dephosphorylation rate

            g (float) - phosphorylated protein decay rate constant

        """

        # update nodes and reactions
        node_id = self.N
        added_node_ids = np.arange(node_id, node_id+1)
        self.update_reaction_dimensions(added_node_ids=added_node_ids)

        # add new nodes
        self.nodes = np.append(self.nodes, added_node_ids)

        # update dictionaries
        if name is None:
            name = 'p'+base
        self.proteins.update({name: node_id})
        self.phosphorylated.update({base: node_id})

        # add forward reaction
        stoichiometry = np.zeros(self.N, dtype=np.int64)
        stoichiometry[self.proteins[base]] = -1
        stoichiometry[node_id] = 1
        propensity = np.zeros(self.N, dtype=np.int64)
        propensity[self.proteins[base]] = 1
        activation = Hill(stoichiometry,
                          propensity,
                          None,
                          k=kf,
                          k_m=Kf,
                          rxn_type=base.upper()+' phosphorylation')

        # add reverse reaction
        propensity = np.zeros(self.N, dtype=np.int64)
        propensity[node_id] = 1
        deactivation = Hill(-stoichiometry,
                            propensity,
                            None,
                            k=kr,
                            k_m=Kr,
                            rxn_type=name+' dephosphorylation')

        # add decay
        stoichiometry = np.zeros(self.N, dtype=np.int64)
        stoichiometry[node_id] = -1
        decay = MassAction(stoichiometry,
                           None,
                           None,
                           k=g,
                           rxn_type=name+' decay')

        # add reactions
        rxns = [rxn for rxn in [activation, deactivation, decay] if rxn.k != 0]
        self.reactions.extend(rxns)
        self.update()

    def add_dimer(self,
                  p1,
                  p2,
                  kf=1.,
                  kr=1.,
                  g=1.,
                  name=None):
        """
        Add dimerization product.

        Args:

            p1, p2 (str) - protein substrates

            kf (float) - forward rate constant

            kr (float) - reverse rate constant

            g (float) - dimer decay rate constant

            name (str) - dimer name, if None concatenate substrate names

        """

        # add new node for dimer
        if name is None:
            name = '-'.join([p1, p2])

        dimer_id = self.N
        added_node_ids = np.arange(dimer_id, dimer_id+1)
        self.update_reaction_dimensions(added_node_ids=added_node_ids)
        self.nodes = np.append(self.nodes, added_node_ids)
        self.proteins.update({name: dimer_id})

        # get base species
        base1, base2 = self.proteins[p1], self.proteins[p2]
        stoichiometry = np.zeros(self.N, dtype=np.int64)
        stoichiometry[base1] = -1
        stoichiometry[base2] = -1
        stoichiometry[dimer_id] = 1
        fwd_name, rev_name = name+' association', name+' dissociation'
        fwd = MassAction(stoichiometry, None, None, kf, rxn_type=fwd_name)
        rev = MassAction(-stoichiometry, None, None, kr, rxn_type=rev_name)

        # add decay
        s = np.zeros(self.N, dtype=np.int64)
        s[dimer_id] = -1
        decay = MassAction(s, k=g, rxn_type='{:s} decay'.format(name))

        # add reactions
        rxns = [rxn for rxn in [fwd, rev, decay] if rxn.k != 0]
        self.reactions.extend(rxns)
        self.update()

    def add_translocation(self,
                          base,
                          kf=1.,
                          kr=1.,
                          g=1.,
                          name=None):
        """
        Add translocation product.

        Args:

            base (str) - protein substrate

            kf (float) - forward rate constant

            kr (float) - reverse rate constant

            g (float) - translocation product decay rate constant

            name (str) - translocation product name, if None use 'n'+base

        """

        # add new node for dimer
        if name is None:
            name = 'n'+base

        node_id = self.N
        added_node_ids = np.arange(node_id, node_id+1)
        self.update_reaction_dimensions(added_node_ids=added_node_ids)
        self.nodes = np.append(self.nodes, added_node_ids)
        self.proteins.update({name: node_id})

        # get base species
        base_id = self.proteins[base]
        stoich = np.zeros(self.N, dtype=np.int64)
        stoich[base_id] = -1
        stoich[node_id] = 1
        fwd = MassAction(stoich, None, None, kf, rxn_type=base+' import')
        rev = MassAction(-stoich, None, None, kr, rxn_type=base+' export')

        # add decay
        stoich = np.zeros(self.N, dtype=np.int64)
        stoich[node_id] = -1
        decay = MassAction(stoich, k=g, rxn_type='{:s} decay'.format(name))

        # add reactions
        rxns = [rxn for rxn in [fwd, rev, decay] if rxn.k != 0]
        self.reactions.extend(rxns)
        self.update()

    def add_transcript_degradation(self,
                                   actuator,
                                   target,
                                   k=1.,
                                   Kd=1.):
        """
        Add transcript degradation term based on Michaelis Menten kinetics.

        Args:

            actuator (str) - actuating substrate

            target (float) - target gene

            k (float) - maximum degradation rate

            Kd (float) - substrate concentration for half maximal rate

        """

        # create reaction for target
        stoichiometry = np.zeros(self.N, dtype=np.int64)
        stoichiometry[self.transcripts[target]] = -1
        propensity = np.zeros(self.N, dtype=np.int64)

        # determine input dependence
        if 'IN' not in actuator:
            input_dependence = None
            propensity[self.transcripts[actuator]] = 1
        else:
            if '_' in actuator:
                input_dependence = np.zeros(self.I, dtype=float)
                input_dependence[int(actuator.split('_')[-1])] = 1
            else:
                input_dependence = 1

        # define reaction
        rxn = Hill(stoichiometry,
                   propensity,
                   input_dependence,
                   k=k,
                   k_m=Kd,
                   rxn_type=target+' transcript deg.')

        # add reaction
        self.reactions.append(rxn)
        self.update()

    def add_protein_degradation(self,
                                actuator,
                                target,
                                k=1.,
                                Kd=1.,
                                modulation=None):
        """
        Add protein degradation term based on Michaelis Menten kinetics.

        Args:

            actuator (str) - actuating substrate

            target (float) - target protein

            k (float) - maximum degradation rate

            Kd (float) - substrate concentration for half maximal rate

            modulation (tuple) - (input_dimension, modulation_factor) pair

        """

        # create reaction for target
        stoichiometry = np.zeros(self.N, dtype=np.int64)
        stoichiometry[self.proteins[target]] = -1
        propensity = np.zeros(self.N, dtype=np.int64)
        rxn_name = target+'  deg.'

        # determine input dependence
        if 'IN' not in actuator:
            input_dependence = None
            propensity[self.proteins[actuator]] = 1
        else:
            if '_' in actuator:
                input_dependence = np.zeros(self.I, dtype=float)
                input_dependence[int(actuator.split('_')[-1])] = 1
            else:
                input_dependence = 1

        # add input rate modifier
        rate_modifier = None
        if modulation is not None:
            rate_modifier = np.zeros(self.I, dtype=float)
            rate_modifier[modulation[0]] = modulation[1]

        # define reaction
        rxn = Hill(stoichiometry,
                   propensity,
                   input_dependence,
                   k=k,
                   k_m=Kd,
                   rxn_type=rxn_name,
                   rate_modifier=rate_modifier)

        # add reaction
        self.reactions.append(rxn)
        self.update()
