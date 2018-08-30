from .reactions import Reaction, EnzymaticReaction, EnzymaticRepressor, Coupling, Transcription
from .networks import MutableNetwork, Graph
#from .parameters import gene_params, dimer_params, repressor_params

import numpy as np
import networkx as nx
from copy import copy


class Gene:
    """
    System dimensions:
        0: Gene
        1: mRNA
        2: Protein
    """
    def __init__(self, name='gene', k=1, g1=1, g2=1):

        self.nodes = np.arange(2)
        self.transcripts = {name: 0}
        self.proteins = {name: 1}

        gene_name = name[0].lower()

        mrna_decay = gene_name + ' decay'
        protein_name = name.upper()
        translation = protein_name + ' translation'
        protein_decay = protein_name + ' decay'

        # define reactions
        self.reactions = [

            # transcript decay
            Reaction([-1, 0], [1, 0], k=g1, rxn_type=mrna_decay),

            # protein synthesis/decay
            Reaction([0, 1], [1, 0], k=k, rxn_type=translation),
            Reaction([0, -1], [0, 1], k=g2, rxn_type=protein_decay)]


class Cell(MutableNetwork):

    def __init__(self, genes=(), num_inputs=1, **kwargs):
        MutableNetwork.__init__(self, 0, inputs=num_inputs)
        self.transcripts = {}
        self.proteins = {}
        self.phosphorylated = {}

        # add genes
        self.add_genes(genes, **kwargs)

    def __repr__(self):
        graph = Graph(self)
        graph.show_reactions()
        return ''

    def get_ic(self, ic=None):
        """ Get initial condition for cell. """

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
        self.node_key = {index: int(node_id) for index, node_id in enumerate(self.nodes)}
        #self.compile_stoichiometry()
        #self.resize_inputs()

    def add_genes(self, names, **kwargs):
        for name in names:
            self.add_gene(name=name, **kwargs)

    def add_gene(self, **kwargs):

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

    def add_phosphorylation(self, base, kf=1., Kf=1., kr=1., Kr=1., g=0., name=None):

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
        activation_name = base.upper() + ' phosphorylation'
        activation = EnzymaticReaction(stoichiometry, propensity, None, k=kf, k_m=Kf, rxn_type=activation_name)

        # add reverse reaction
        propensity = np.zeros(self.nodes.size, dtype=np.int64)
        propensity[node_id] = 1
        deactivation_name = name + ' dephosphorylation'
        deactivation = EnzymaticReaction(-stoichiometry, propensity, None, k=kr, k_m=Kr, rxn_type=deactivation_name)

        # add decay
        stoichiometry = np.zeros(self.nodes.size, dtype=np.int64)
        stoichiometry[node_id] = -1
        rxn_name = name+' decay'
        decay = Reaction(stoichiometry, None, None, k=g, rxn_type=rxn_name)

        # add reactions
        rxns = [rxn for rxn in [activation, deactivation, decay] if rxn.k != 0]
        self.reactions.extend(rxns)
        self.update()

    def add_transcription(self, gene, modules=None, k=1, alpha=None, perturbed=False, **kw):

        # define stoichiometry
        s = np.zeros(self.nodes.size, dtype=np.int64)
        s[self.transcripts[gene]] = 1

        # # define input dependence
        # if input_dependence is None:
        #     input_dependence = np.zeros(self.input_size, dtype=np.float64)

        # add synthesis reaction
        name = gene + ' transcription'
        rxn = Transcription(s, modules, k=k, alpha=alpha, perturbed=perturbed, rxn_type=name, **kw)
        self.reactions.append(rxn)
        self.update()

    def add_dimer(self, p1, p2, kf=1., kr=1., g=1., name=None):

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
        fwd = Reaction(stoichiometry, None, None, kf, rxn_type=fwd_name)
        rev = Reaction(-stoichiometry, None, None, kr, rxn_type=rev_name)

        # add decay
        s = np.zeros(self.nodes.size, dtype=np.int64)
        s[dimer_id] = -1
        decay = Reaction(s, k=g, rxn_type='{:s} decay'.format(name))

        # add reactions
        rxns = [rxn for rxn in [fwd, rev, decay] if rxn.k != 0]
        self.reactions.extend(rxns)
        self.update()

    def add_translocation(self, base, kf=1., kr=1., g=1., name=None):

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
        stoichiometry = np.zeros(self.nodes.size, dtype=np.int64)
        stoichiometry[base_id] = -1
        stoichiometry[node_id] = 1
        fwd = Reaction(stoichiometry, None, None, kf, rxn_type=base+' import')
        rev = Reaction(-stoichiometry, None, None, kr, rxn_type=base+' export')

        # add decay
        s = np.zeros(self.nodes.size, dtype=np.int64)
        s[node_id] = -1
        decay = Reaction(s, k=g, rxn_type='{:s} decay'.format(name))

        # add reactions
        rxns = [rxn for rxn in [fwd, rev, decay] if rxn.k != 0]
        self.reactions.extend(rxns)
        self.update()

    def add_transcript_degradation(self, actuator, target, k=1., Kd=1.):

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

        # add reverse reaction
        rxn = EnzymaticReaction(stoichiometry, propensity, input_dependence, k=k, k_m=Kd, rxn_type=target+' transcript deg.')
        self.reactions.append(rxn)
        self.update()

    def add_protein_degradation(self, actuator, target, k=1., Kd=1., modulation=None):

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

        # add reverse reaction
        rxn = EnzymaticReaction(stoichiometry, propensity, input_dependence, k=k, k_m=Kd, rxn_type=rxn_name, rate_modifier=rate_modifier)
        self.reactions.append(rxn)
        self.update()
