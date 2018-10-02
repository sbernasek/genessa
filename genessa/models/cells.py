import numpy as np
import networkx as nx
from copy import copy

# intra-package python imports
from ..systems.networks import MutableNetwork, Graph
from ..kinetics.massaction import MassAction
from ..kinetics.hill import Hill


class Gene:
    """
    Class defines a single gene coding a protein product.

    System dimensions:
        0: mRNA
        1: Protein

    Attributes:

        nodes (np.ndarray) - node indices

        transcripts (dict) - single {name: node_id} pair

        proteins (dict) - single {name: node_id} pair

        reactions (list) - translation, mRNA decay, and protein decay reactions

    """
    def __init__(self, name='gene', k=1, g1=1, g2=1):
        """
        Create gene along with translation, mRNA decay, and protein decay reactions.

        Args:

            name (str) - gene name

            k (float) - translation rate constant

            g1 (float) - transcript decay rate constant

            g2 (float) - protein decay rate constant

        """

        self.nodes = np.arange(2)
        self.transcripts = {name: 0}
        self.proteins = {name: 1}

        # define gene names
        gene_name = name[0].lower()
        mrna_decay = gene_name + ' decay'
        protein_name = name.upper()
        translation = protein_name + ' translation'
        protein_decay = protein_name + ' decay'

        # define reactions
        self.reactions = [

            # transcript decay
            MassAction([-1, 0],
                       [1, 0],
                       k=g1,
                       rxn_type=mrna_decay,
                       atp_sensitive=False,
                       ribosome_sensitive=False),

            # protein synthesis
            MassAction([0, 1],
                       [1, 0],
                       k=k,
                       rxn_type=translation,
                       atp_sensitive=True,
                       ribosome_sensitive=True),

            # protein decay
            MassAction([0, -1],
                       [0, 1],
                       k=g2,
                       rxn_type=protein_decay,
                       atp_sensitive=True,
                       ribosome_sensitive=True)]


class Cell(MutableNetwork):
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

    def __init__(self,
                 genes=(),
                 num_inputs=1,
                 **kwargs):
        """
        Instantiate cell with one or more protein coding genes.

        Args:

            genes (tuple) - names of genes

            num_inputs (int) - number of input channels

            kwargs: keyword arguments for add_genes

        """

        # instantiate network
        MutableNetwork.__init__(self, 0, inputs=num_inputs)

        # initialize species dictionaries
        self.transcripts = {}
        self.proteins = {}
        self.phosphorylated = {}

        # add genes
        self.add_genes(genes, **kwargs)

    def __repr__(self):
        """ Print list of reactions. """
        graph = Graph(self)
        graph.show_reactions()
        return ''

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
            ic = np.zeros(self.nodes.size, dtype=np.int64)

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
        shift = self.nodes.size
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
        node_id = self.nodes.size
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
        stoichiometry = np.zeros(self.nodes.size, dtype=np.int64)
        stoichiometry[self.proteins[base]] = -1
        stoichiometry[node_id] = 1
        propensity = np.zeros(self.nodes.size, dtype=np.int64)
        propensity[self.proteins[base]] = 1
        activation = Hill(stoichiometry,
                          propensity,
                          None,
                          k=kf,
                          k_m=Kf,
                          rxn_type=base.upper()+' phosphorylation')

        # add reverse reaction
        propensity = np.zeros(self.nodes.size, dtype=np.int64)
        propensity[node_id] = 1
        deactivation = Hill(-stoichiometry,
                            propensity,
                            None,
                            k=kr,
                            k_m=Kr,
                            rxn_type=name+' dephosphorylation')

        # add decay
        stoichiometry = np.zeros(self.nodes.size, dtype=np.int64)
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

        dimer_id = self.nodes.size
        added_node_ids = np.arange(dimer_id, dimer_id+1)
        self.update_reaction_dimensions(added_node_ids=added_node_ids)
        self.nodes = np.append(self.nodes, added_node_ids)
        self.proteins.update({name: dimer_id})

        # get base species
        base1, base2 = self.proteins[p1], self.proteins[p2]
        stoichiometry = np.zeros(self.nodes.size, dtype=np.int64)
        stoichiometry[base1] = -1
        stoichiometry[base2] = -1
        stoichiometry[dimer_id] = 1
        fwd_name, rev_name = name+' association', name+' dissociation'
        fwd = MassAction(stoichiometry, None, None, kf, rxn_type=fwd_name)
        rev = MassAction(-stoichiometry, None, None, kr, rxn_type=rev_name)

        # add decay
        s = np.zeros(self.nodes.size, dtype=np.int64)
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

        node_id = self.nodes.size
        added_node_ids = np.arange(node_id, node_id+1)
        self.update_reaction_dimensions(added_node_ids=added_node_ids)
        self.nodes = np.append(self.nodes, added_node_ids)
        self.proteins.update({name: node_id})

        # get base species
        base_id = self.proteins[base]
        stoich = np.zeros(self.nodes.size, dtype=np.int64)
        stoich[base_id] = -1
        stoich[node_id] = 1
        fwd = MassAction(stoich, None, None, kf, rxn_type=base+' import')
        rev = MassAction(-stoich, None, None, kr, rxn_type=base+' export')

        # add decay
        stoich = np.zeros(self.nodes.size, dtype=np.int64)
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
        stoichiometry = np.zeros(self.nodes.size, dtype=np.int64)
        stoichiometry[self.transcripts[target]] = -1
        propensity = np.zeros(self.nodes.size, dtype=np.int64)

        # determine input dependence
        if 'IN' not in actuator:
            input_dependence = None
            propensity[self.transcripts[actuator]] = 1
        else:
            if '_' in actuator:
                input_dependence = np.zeros(self.input_size, dtype=float)
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
        stoichiometry = np.zeros(self.nodes.size, dtype=np.int64)
        stoichiometry[self.proteins[target]] = -1
        propensity = np.zeros(self.nodes.size, dtype=np.int64)
        rxn_name = target+'  deg.'

        # determine input dependence
        if 'IN' not in actuator:
            input_dependence = None
            propensity[self.proteins[actuator]] = 1
        else:
            if '_' in actuator:
                input_dependence = np.zeros(self.input_size, dtype=float)
                input_dependence[int(actuator.split('_')[-1])] = 1
            else:
                input_dependence = 1

        # add input rate modifier
        rate_modifier = None
        if modulation is not None:
            rate_modifier = np.zeros(self.input_size, dtype=float)
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

    def add_transcription(self,
                          gene,
                          modules=None,
                          k=1,
                          alpha=None,
                          perturbed=False,
                          **kw):

        # define stoichiometry
        s = np.zeros(self.nodes.size, dtype=np.int64)
        s[self.transcripts[gene]] = 1

        # add synthesis reaction
        name = gene + ' transcription'
        rxn = Transcription(s,
                            modules,
                            k=k,
                            alpha=alpha,
                            perturbed=perturbed,
                            rxn_type=name,
                            **kw)
        self.reactions.append(rxn)
        self.update()
