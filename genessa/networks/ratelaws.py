import numpy as np
from tabulate import tabulate


class RateLaws:
    """
    Class provides tabulated summary of reaction kinetics.

    Attributes:

        node_key (dict) - maps state dimension (key) to unique node id (value)

        reactions (list) - list of reaction objects

        table (list of lists) - rate law table, row for each reaction

    """

    def __init__(self,
                 node_key,
                 reactions):
        """
        Instantiate raw law table.

        Args:

            node_key (dict) - maps state dimension (key) to node id (value)

            reactions (list) - reaction instances

        """
        self.node_key = node_key
        self.reactions = reactions
        self.build_table()

    def __repr__(self):
        """
        Pretty-print rate law table.
        """
        self.print_table()
        return ''

    def print_table(self):
        """
        Pretty-print rate law table.
        """
        print(tabulate(self.table,
                       headers=["Rxn",
                                "Reactants",
                                "Products",
                                "Propensity",
                                "Parameter"],
                       numalign='center',
                       stralign='center'))

    def build_table(self):
        """
        Build rate law table.
        """
        self.table = []
        for rxn in self.reactions:
            self.add_rxn_to_table(rxn)

    def add_rxn_to_table(self, rxn):
        """
        Add reaction rate law to table.

        Args:

            rxn (reaction instance)

        """
        # assemble reactants
        reactants = self.assemble_reactants(rxn)

        # assemble products
        products = self.assemble_products(rxn)

        # assemble rate expression
        rate_law = self.assemble_rate_expression(rxn)

        # assemble rate constant
        rate_constant = self.assemble_rate_constant(rxn)

        # assemble sensitivities to environmental conditions
        sensitivities = self.assemble_sensitivities(rxn)

        # set reaction name
        if rxn.name is None:
            name = 'Not Named'
        else:
            name = rxn.name

        # add reaction to table
        self.table.append([name, reactants, products, rate_law, rate_constant])

        # add any repressors for Hill and Coupling reactions
        if rxn.type in ('Hill', 'Coupling'):
            for repressor in rxn.repressors:
                repressor_name = 'Repression of ' + rxn.name
                rate_law = self.get_enzymatic_rate_law(repressor)
                rate_law = '1 - ' + rate_law
                self.table.append([repressor_name, '', '', rate_law, '', 'NA'])

    def assemble_reactants(self, rxn):
        """
        Assemble list of reactants.

        Args:

            rxn (reaction instance)

        Returns:

            reactants (str) - species indices

        """
        reactants = []
        for reactant, coeff in enumerate(rxn.stoichiometry):
            if coeff < 0:
                reactants.append(str(self.node_key[int(reactant)]))
        return ", ".join(reactants)

    def assemble_products(self, rxn):
        """
        Assemble list of products.

        Args:

            rxn (reaction instance)

        Returns:

            products (str) - species indices

        """
        products = []
        for product, coeff in enumerate(rxn.stoichiometry):
            if coeff > 0:
                products.append(str(self.node_key[int(product)]))
        return ", ".join(products)

    def assemble_sensitivities(self, rxn):
        """
        Assemble list of sensitivities to environmental conditions.

        Args:

            rxn (reaction instance)

        Returns:

            sensitivities (str) - species indices

        """

        sensitivities = ''

        if rxn.temperature_sensitive is True:
            pass

        if rxn.atp_sensitive is not False:
            if rxn.atp_sensitive is True or rxn.atp_sensitive==1:
                sensitivities += 'ATP'
            else:
                sensitivities += 'ATP^{:1.0f}'.format(rxn.atp_sensitive)

        if rxn.ribosome_sensitive is not False:
            if rxn.ribosome_sensitive is True or rxn.ribosome_sensitive==1:
                sensitivities += ', RPs'
            else:
                sensitivities += ', RPs^{:1.0f}'.format(rxn.ribosome_sensitive)

        return sensitivities

    def assemble_rate_expression(self, rxn):
        """
        Assemble rate expression.

        Args:

            rxn (reaction instance)

        Returns:

            rate_expression (str)

        """

        # get reaction type
        if rxn.type in ('MassAction', 'LinearFeedback'):
            rate_expression = self.get_mass_action_rate_law(rxn)

        elif rxn.type == 'Hill':
            rate_expression = self.get_enzymatic_rate_law(rxn)

        elif rxn.type == 'SumReaction':
            rate_expression = self.get_sum_rxn_rate_law(rxn)

        elif rxn.type == 'Coupling':
            rate_expression = self.get_coupling_rate_law(rxn)

        else:
            rate_expression = 'Unknown Rxn Type'

        return rate_expression

    @staticmethod
    def assemble_rate_constant(rxn):
        """
        Returns rate constant expression for a reaction instance.

        Args:

            rxn (all types)

        Returns:

            rate_constant (str) - rate constant expression

        """

        rate_constant = '{:2.5f}'.format(rxn.k[0])

        if rxn.type == 'Hill':
            for i, coeff in enumerate(rxn.rate_modifier):
                if coeff != 0:
                    rate_constant += ' + {:0.1f}[IN_{:d}]'.format(coeff, i)

        return rate_constant

    def get_mass_action_rate_law(self, rxn):
        """
        Returns rate expression for MassAction instance.

        Args:

            rxn (MassAction)

        Returns:

            rate_law (str) - rate expression

        """

        fmt_term = lambda sp: ('['+str(self.node_key[int(sp)])+']')

        # add reactants
        p = [fmt_term(sp)*int(co) for sp, co in enumerate(rxn.propensity)]

        # add input contribution
        input_contribution = ''
        for i, dependence in enumerate(rxn.input_dependence):
            if dependence != 0:
                input_contribution += int(dependence) * '[IN_{:d}]'.format(i)

        # assemble rate law
        rate_law = input_contribution + "".join(str(term) for term in p)

        return rate_law

    def get_enzymatic_rate_law(self, rxn):
        """
        Returns rate expression for Hill or Repressor instance.

        Args:

            rxn (Hill or Repressor)

        Returns:

            rate_law (str) - rate expression

        """

        fmt = lambda sp: '['+str(self.node_key[int(sp)])+']'

        # add substrate contributions
        substrate_contribution = ''
        p = [fmt(sp) if co!=0 else '' for sp, co in enumerate(rxn.propensity)]
        w = [str(int(c)) if c!=0 and c!=1 else '' for c in rxn.propensity]

        # combine weights and inputs
        for i, j in zip(w, p):
            substrate_contribution += (i+j)

        # assemble substrates
        activity = ''
        for i, dependence in enumerate(rxn.input_dependence):
            if dependence != 0:
                if rxn.input_dependence.size == 1:
                    input_name = '[IN]'
                else:
                    input_name = '[IN_{:d}]'.format(i)

                coefficient = ''
                if dependence != 1:
                    coefficient = str(dependence)
                if i == 0:
                    activity += (coefficient + input_name)
                else:
                    activity += (coefficient + input_name)

        if rxn.num_active_species > 0 and len(activity) > 0:
            activity += '+' + substrate_contribution
        elif rxn.num_active_species > 0 and len(activity) == 0:
            activity += substrate_contribution

        # assemble rate law
        if rxn.n != 1:
            rate_law = activity+'^'+str(rxn.n)[:4] + '/(' + activity + '^' + str(rxn.n)[:4] + ' + ' + str(rxn.k_m)[:6]+ '^' + str(rxn.n)[:4]+')'
        else:
            rate_law = '{:s}/({:s}+{:s})'.format(activity, activity, str(rxn.k_m)[:6])

        return rate_law

    @staticmethod
    def get_sum_rxn_rate_law(rxn):
        """
        Returns rate expression for SumReaction instance.

        Args:

            rxn (SumReaction)

        Returns:

            rate_law (str) - rate expression

        """
        positive = np.where(rxn.propensity == 1)[0][0]
        negative = np.where(rxn.propensity == -1)[0][0]
        rate_law = '[{:d}] - [{:d}]'.format(positive, negative)
        return rate_law

    @staticmethod
    def get_coupling_rate_law(rxn):
        """
        Returns rate expression for Coupling instance.

        Args:

            rxn (Coupling)

        Returns:

            rate_law (str) - rate expression

        """
        if rxn.propensity.max() == 0:
            rate_law = ''
        else:
            base = np.where(rxn.propensity>0)[0][0]
            neighbors = np.where(rxn.propensity<0)[0]
            weight = (rxn.a * rxn.w) / (1+rxn.w*len(neighbors))

            if len(neighbors) > 1:
                coeff = '{:d}'.format(len(neighbors))
            else:
                coeff = ''

            rate_law = '{:0.3f} x ({:s}[{:d}]'.format(weight, coeff, base)
            for n in neighbors:
                rate_law += ' - [{:d}]'.format(n)
            rate_law += ')'
        return rate_law
