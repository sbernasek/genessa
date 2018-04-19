#from .parameters as params
from .cells import Cell
import os as os
import numpy as np
import json as json
import glob as glob

"""
TO DO:

1. Add topographical readout (degree distribution, clustering coefficient, etc)

"""


class Population:

    def __init__(self, cells=None):
        """
        Parameters:
            cells (list) - collection of cell objects
        """

        # initialize population as a dictionary of cells, each of which has a unique ID, generation, parent, and fitness
        if cells is None:
            cells = [Cell(3)]

        self.population = len(cells)
        self.unique_id = len(cells)
        self.cells = {cell_id:
                      {'cell': cell,
                       'generation': 0,
                       'parent': None,
                       'fitness': 0}
                      for cell_id, cell in enumerate(cells)}

    def grow(self, n, mode=None, generation=None):
        """
        Grow population back to specified level by reproduction of current cells.

        Parameters:
            n (int) - desired population
            mode (str) - asexual or sexual reproduction
            generation (int) - current generation. if None, increment parent's generation
        """

        while self.population < n:

            if mode == 'sexual' and self.population >= 2:

                # sexual reproduction: create child by merging two randomly selected parents
                parent_id = tuple([int(parent) for parent in np.random.choice(list(self.cells.keys()), size=2, replace=False)])
                mother, father = [self.cells[parent]['cell'] for parent in parent_id]

                # store number of nodes and edges (for pruning)
                node_count = int((mother.nodes.size + father.nodes.size)/2)
                edge_count = int(len(mother.reactions + father.reactions)/2)

                # create copy of mother then add all nodes/reactions from father
                child = mother.divide()
                child.merge_with_another(father)

                # prune network back to original size
                node_removals = child.nodes.size - node_count
                if node_removals > 0:
                    child.remove_nodes(node_removals)

                # prune edges back to original number
                edge_removals = len(child.reactions) - edge_count
                if edge_removals > 0:
                    child.remove_edges(edge_removals)

                # child inherits generation from youngest parent
                if generation is None:
                    generation = max([self.cells[parent]['generation'] for parent in parent_id]) + 1

            else:

                # asexual reproduction: create child by replicating randomly selected parent
                parent_id = int(list(self.cells.keys())[np.random.randint(0, len(self.cells))])
                parent = self.cells[parent_id]['cell']
                child = parent.divide()

                # set child's unique cell_id
                if generation is None:
                    generation = self.cells[parent_id]['generation'] + 1

            # mutate child while ensuring output is input-dependent
            accepted = False
            while accepted is False:

                # mutate child
                child.mutate(params.mutation_rates, params.rate_constants)

                # if its output is input-dependent, accept it
                dependent_nodes = child.get_input_dependents(ic=np.zeros(child.nodes.size))
                if child.output_node in dependent_nodes:
                    accepted = True

            # add child to populations
            child_id = self.unique_id
            self.cells[child_id] = {'cell': child, 'generation': generation, 'parent': parent_id, 'fitness': 0}
            self.population = len(self.cells)

            # increment unique cell_id count
            self.unique_id += 1

    def evaluate(self, fitness):
        """
        Evaluate fitness of each cell within population.

        Parameters:
            fitness (function) - fitness function
        """
        for cell_id, cell in self.cells.items():
            try:
                score = fitness(cell['cell'])
            except:
                score = 0
            cell['fitness'] = score

    def cull(self, percentile):
        """
        Remove cells from population on the basis of their fitness.

        Parameters:
            percentile (float) - fraction of population removed
        """
        # sort cells by fitness
        ranked_cell_ids = sorted(self.cells.keys(), key=lambda x: self.cells[x]['fitness'])

        # remove cells below specified percentile
        culled_cell_ids = ranked_cell_ids[:int(np.floor(self.population*(percentile)))]
        for cell_id in culled_cell_ids:
            _ = self.cells.pop(cell_id)

        # update population
        self.population = len(self.cells)

    def evolve(self, pressure, population=100, percentile=0.8, mode=None):
        """
        Runs single generation of growth and selection.

        Parameters:
            pressure (function) - fitness function
            population (int) - target population
            percentile (float) - fitness threshold for selection
            mode (str) - mode of cell reproduction, either 'asexual' or 'sexual'

        Return:
            f (np array) - cell fitness distribution
            n (np array) - cell node count distribution
            r (np array) - cell rxn count distribution
        """

        # run growth and evaluate fitness
        self.grow(n=population, mode=mode)
        self.evaluate(pressure)

        # get population statistics
        f = [cell['fitness'] for cell in self.cells.values()]
        n = [cell['cell'].nodes.size for cell in self.cells.values()]
        r = [len(cell['cell'].reactions) for cell in self.cells.values()]

        # run selection
        self.cull(percentile=percentile)

        return f, n, r


class Generation(Population):
    """
    Class inherits a population of cells to which it adds json read/write capability.
    """

    def __init__(self, *args):
        Population.__init__(self)

        # if instantiating from parent class, retain parent attributes
        if len(args) == 1 and isinstance(args[0], Population):
            self.cells = args[0].cells
            self.population = args[0].population
            self.unique_id = args[0].unique_id

    @staticmethod
    def from_json(js):

        # create instance
        generation = Generation()

        # get each attribute from json dictionary
        generation.population = int(js['population'])
        generation.unique_id = int(js['unique_id'])

        # get attributes containing nested classes
        generation.cells = {int(cell_id):
                            {'cell': Cell.from_json(cell_dict['cell']),
                             'generation': int(cell_dict['generation']),
                             'parent': int(cell_dict['parent']),
                             'fitness': float(cell_dict['fitness'])} for cell_id, cell_dict in js['cells'].items()}
        return generation

    def to_json(self):
        return {
            # return each attribute
            'population': self.population,
            'unique_id': self.unique_id,

            # return attributes containing nested classes
            'cells': {cell_id:
                      {'cell': cell_dict['cell'].to_json(),
                        'generation': cell_dict['generation'],
                        'parent': cell_dict['parent'],
                        'fitness': float(cell_dict['fitness'])} for cell_id, cell_dict in self.cells.items()}}

    def save(self, path, file_name=None):
        """
        Write generation to file in serialized json format.

        Parameters:
            path (str) - path to which json is saved
            file_name (int or str) - generation number or file name
        """

        # if target directory doesn't exist, make it
        if os.path.isdir(path) is False:
            os.mkdir(path)
            file_name = 'gen_0'

        # if target directory does exist and no file_name is specified, increment previous generation
        elif file_name is None:
            generations = []
            for f in glob.glob(os.path.join(path, '*.json')):
                generations.append(int(f.split('.')[0].split('/')[-1].split('_')[1]))
            file_name = 'gen_' + str(max(generations) + 1)

        destination = os.path.join(path, str(file_name)+'.json')
        with open(destination, mode='w', encoding='utf-8') as f:
            json.dump(self.to_json(), f)


    @staticmethod
    def load(path, file_name):
        """
        Create generation object from serialized json format.

        Parameters:
            path (str) - path to directory containing json
            file_name (int or str) - file name

        Returns:
            generation (Generation object)
        """
        target = os.path.join(path, str(file_name) + '.json')
        with open(target, mode='r', encoding='utf-8') as f:
            generation = Generation.from_json(json.load(f))
        return generation
