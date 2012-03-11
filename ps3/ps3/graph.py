# encoding: utf-8
"""
Object oriented graph structure code

"""

__all__ = ["Block", "Network", "Variable", "Factor"]

import re
import itertools

import numpy as np

class Block(object):
    def __init__(self, properties=[]):
        self.properties = properties

    def __getattr__(self, k):
        return self.properties[k]

    def __str__(self):
        return "Block(properties=%s)"%(str(self.properties))

    def __repr__(self):
        return self.__str__()

class Network(Block):
    """
    A network contains variables and factors.

    ## Argument

    * `name` (str): The name of the network.

    """
    def __init__(self, name, **kwargs):
        super(Network, self).__init__(**kwargs)
        self.name = name
        self.nodes = {}
        self.inactive_nodes = {}
        self.factors = []
        self.inactive_factors = []

    def add_node(self, variable):
        """Add a node to the network."""
        self.nodes[variable.name] = variable

    def remove_node(self, variable):
        self.inactive_nodes[variable.name] = self.nodes.pop(variable.name)

    def add_factor(self, factor):
        """Add a factor to the network."""
        for n in factor.nodes:
            assert n.name in self.nodes
        self.factors.append(factor)

    def remove_factors(self, factors):
        self.inactive_factors += \
                [self.factors.pop(self.factors.index(f)) for f in factors]

    def __str__(self):
        s = "network %s {\n"%(self.name)
        s += "".join([str(p)+"\n" for p in self.properties])
        s += "}\n"
        s += "".join([str(self.nodes[k]) for k in self.nodes])
        s += "".join([str(f) for f in self.factors])
        return s

    def to_file(self, fn):
        """
        Save the network to a BIF file.

        ## Arguments

        * `fn` (str): The file to write to.

        """
        f = open(fn, "w")
        f.write(str(self))
        f.close()

    @classmethod
    def from_file(cls, fn):
        """
        Parse a BIF document to produce a `Network` object.

        ## Arguments

        * `fn` (str): The filename to load.

        ## Returns

        * `network` (Network): The generated `Network`.

        """
        _block_re = re.compile("(.*?) (.*?) {")
        _var_re   = re.compile("type discrete.*?\[.*?\] {(.*?)}", re.M)
        _prob_re  = re.compile("\((.*?)(?:\|(.*?)|(?:))\)")
        _table_re = re.compile("table (.*?);", re.M)
        _cond_re  = re.compile("\((.*?)\)(.*?);", re.M)

        network = None
        current_block = ""
        # We'll keep track of the bracket level so that we can find the
        # "current block" as we go through the file line-by-line.
        level = 0

        f = open(fn, "r")
        for line in f:
            level += line.count("{")
            level -= line.count("}")
            current_block += line
            if level == 0: # parse the current block now
                b_type, b_info = _block_re.search(current_block).groups()
                if b_type == "network":
                    network = cls(b_info)
                elif b_type == "variable":
                    assert network is not None
                    vals = _var_re.search(current_block).groups()[0]\
                            .split(",")
                    var = Variable(b_info, [v.strip() for v in vals])
                    network.add_node(var)
                elif b_type == "probability":
                    assert network is not None
                    groups = _prob_re.search(b_info).groups()
                    var = [groups[0].strip()]
                    if groups[1] is None: # It's a prior probability table.
                        tab = np.array([v
                            for v in _table_re.search(current_block)\
                                .groups()[0].split(",")], dtype=float)
                        factor = Factor([network.nodes[var[0]]], tab)
                    else: # It's a conditional PDF.
                        var += [v.strip() for v in groups[1].split(",")]
                        nodes = [network.nodes[v] for v in var]

                        # NOTE: the probability table is indexed in _reverse
                        #       order_.
                        tab = np.zeros([len(n.states) for n in nodes[::-1]])
                        nstates = len(nodes[0].states)

                        # Loop over the entries in the table
                        for p in _cond_re.findall(current_block):
                            ind = [nodes[i+1].states.index(p0.strip())
                                    for i,p0 in enumerate(p[0].split(","))]
                            # Yep. Don't ask.
                            ind = [slice(i,i+1) for i in ind[::-1]]\
                                    + [slice(nstates)]
                            tab[ind] = np.array(p[1].split(","), dtype=float)
                        factor = Factor(nodes, tab)
                    network.add_factor(factor)
                else:
                    raise NotImplementedError("Unknown block type: %s"%b_type)
                current_block = ""
        f.close()

        return network

    def _get_set_of_nodes(self, factors, variable=None):
        nodes = []
        for f in factors:
            for n in f.nodes:
                if (variable is None or n.name != variable)\
                        and n not in nodes:
                    nodes.append(n)
        return nodes

    def greedy_ordering(self):
        unmarked = self.nodes.keys()
        factors = [f._node_names for f in self.factors]

        # Loop over the factors to collect which edges already exist.
        edges = {}
        for f in factors:
            for n in f:
                try:
                    edges[n] = edges[n] | set(f)
                except KeyError:
                    edges[n] = set(f)
        for k in edges:
            edges[k] = list(edges[k] - set(k))

        # An inline function to calculate the induced fill edges for all the
        # nodes.
        def get_fill_edges():
            fills = {}
            for k in unmarked:
                fills[k] = []
                for i in range(len(edges[k])):
                    for j in range(i+1, len(edges[k])):
                        if edges[k][i] not in edges[edges[k][j]]:
                            fills[k] += [[edges[k][i], edges[k][j]]]
            return fills

        order = []
        while len(unmarked) > 0:
            # Get the list of fill edges induced by each node.
            fills = get_fill_edges()

            # This gets the _first_ min-fill node.
            min_fill = min(fills, key=lambda x: len(fills[x]))

            # Induce the fill edges.
            for e in fills[min_fill]:
                if e[0] in unmarked and e[1] in unmarked:
                    edges[e[0]].append(e[1])
                    edges[e[1]].append(e[0])

            # Mark the chosen node.
            unmarked.remove(min_fill)
            order.append(min_fill)
            # print dict(map(lambda x: (x, len(fills[x])), fills))

        max_clique = edges[max(edges, key=lambda e: len(edges[e]))]

        self._order = order
        return order, max_clique

    @property
    def order(self):
        """Lazily calculate the elimination order."""
        try:
            return self._order
        except AttributeError:
            return self.greedy_ordering()[0]

    def eliminate(self, variable):
        node = self.nodes[variable]

        # The factors involving `variable`.
        factors = node.factors
        # Get the other nodes that are involved.
        nodes = self._get_set_of_nodes(factors, variable)

        # Calculate the table for the new factor.
        tab = np.zeros([len(n.states) for n in nodes[::-1]])
        for ind in itertools.product(*[n.states for n in nodes]):
            kwargs = dict([(nodes[i].name, ind[i]) for i in range(len(ind))])
            mask = [n.states.index(ind[i]) for i,n in enumerate(nodes)]
            mask = [slice(i,i+1) for i in mask[::-1]]
            for s in node.states:
                kwargs[variable] = s
                tab[mask] += np.prod([f.evaluate(**kwargs) for f in factors])

        # Normalize by the sum.
        tab /= np.sum(tab)

        # Remove all the factors.
        self.remove_node(node)
        self.remove_factors(factors)
        [n.remove_factors(factors) for n in nodes]

        # Add the new factor.
        self.add_factor(Factor(nodes, tab))

    def query(self, q, evidence={}):
        for k in evidence:
            self.nodes[k].add_evidence(evidence[k])
        for n in order:
            if n not in q:
                net.eliminate(n)
        print self.nodes[q[0]]
        for f in self.factors:
            print f.table
            print f.evaluate()
        print self.factors
        # print res

class Variable(Block):
    """
    A variable corresponds to a single node in a network.

    ## Arguments

    * `name` (str): The name of the variable. This is often how the variable
      is referenced and it is case sensitive.
    * `states` (list): The list of possible states that the variable can
      take. These should be strings.

    """
    def __init__(self, name, states, **kwargs):
        super(Variable, self).__init__(**kwargs)
        self.name = name
        self.states = states
        self.factors = []

    def __str__(self):
        s = "variable %s {\n"%(self.name)
        s += "  type discrete [ %d ] { %s };\n"%\
                (len(self.states), ", ".join([str(s) for s in self.states]))
        s += "}\n"
        return s

    def add_factor(self, factor):
        self.factors.append(factor)

    def remove_factors(self, factors):
        for f in factors:
            try:
                self.factors.remove(f)
            except ValueError:
                pass

    def add_evidence(self, state):
        [f.add_evidence(self, state) for f in self.factors]

class Factor(Block):
    """
    A factor quantifies the connections between nodes in a network.

    ## Arguments

    * `nodes` (list): A list of `Variable` objects. These are the nodes that
      involved in this factor.

    * `table` (numpy.ndarray): The value of the factor for each value of
      each variable. It should have one dimension for each variable and be
      indexed in _reverse order_.

    """
    def __init__(self, nodes, table, **kwargs):
        super(Factor, self).__init__(**kwargs)
        self.nodes = nodes
        self._node_names = [n.name for n in nodes]
        for n in self.nodes: # Keep track of the scope of the variables.
            n.add_factor(self)
        self.table = table
        self.evidence = {}

    def __str__(self):
        names = []
        for n in self.nodes:
            if n.name in self.evidence:
                names.append("%s = %s"%(n.name, self.evidence[n.name]))
            else:
                names.append(n.name)
        if len(self.nodes) == 1:
            s = "probability ( %s ) {\n  table %s;\n}\n"%\
                (names[0],
                 ", ".join([str(self.evaluate(s))
                     for s in self.nodes[0].states]))
        else:
            s = "probability ( %s | %s ) {\n"%(names[0], ", ".join(names[1:]))
            for ind in itertools.product(*[n.states for n in self.nodes[1:]]):
                s += "  ( %s ) "%(", ".join([str(i) for i in ind]))
                s += ", ".join([str(self.evaluate(s, *ind))
                    for s in self.nodes[0].states])
                s += ";\n"
            s += "}\n"
        return s

    def evaluate(self, *args, **kwargs):
        """
        Evaluate this factor for a particular set of values.

        ## Arguments

        * `*args` (list): A list of `str`s. This is the set of values that
          the factor will be evaluated for.

        ## Returns

        * `value` (float): The value of the factor at this position.

        """
        if len(args) == 0:
            args = [""]*len(self.nodes)
            for k in kwargs:
                try:
                    args[self._node_names.index(k)] = kwargs[k]
                except ValueError:
                    pass
        assert len(args) == len(self.nodes)
        for i,a in enumerate(args):
            try:
                if a != self.evidence[self._node_names[i]]:
                    return 0.0
            except KeyError:
                pass
        ind = [self.nodes[i].states.index(args[i]) for i in range(len(args))]
        # Don't forget that the indexing is in _reverse_ order.
        ind = [slice(i,i+1) for i in ind[::-1]]
        return float(self.table[ind])

    def add_evidence(self, node, state):
        self.evidence[node.name] = state

def test_io():
    import os
    fn = os.path.join(*(list(os.path.split(__file__)[:-2])+["data","test.bif"]))
    print "Asserting validity of file:", fn
    assert str(Network.from_file(fn)) == open(fn).read()

if __name__ == '__main__':
    net = Network.from_file("data/alarm.bif")
    order, max_clique = net.greedy_ordering()

    print "Elimination Ordering"
    print " -> ".join(order)

    print "\nWith induced max-clique:"
    print " - ".join(max_clique)
    print

    e = {"HYPOVOLEMIA": "TRUE", "ERRCAUTER": "TRUE", "PVSAT": "NORMAL",
            "DISCONNECT": "TRUE", "MINVOLSET": "LOW"}
    q = ["STROKEVOLUME"]

    net.query(q, e)

