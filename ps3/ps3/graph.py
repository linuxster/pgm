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
        self.factors = []

    def add_node(self, variable):
        """Add a node to the network."""
        self.nodes[variable.name] = variable

    def add_factor(self, factor):
        """Add a factor to the network."""
        self.factors.append(factor)

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
        # We'll keep track of the bracket level so that we can find the "current
        # block" as we go through the file line-by-line.
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
                    vals = _var_re.search(current_block).groups()[0].split(",")
                    var = Variable(b_info, [v.strip() for v in vals])
                    network.add_node(var)
                elif b_type == "probability":
                    assert network is not None
                    groups = _prob_re.search(b_info).groups()
                    var = [groups[0].strip()]
                    if groups[1] is None: # It's a prior probability table.
                        tab = np.array([v for v in _table_re.search(current_block)\
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
                            ind = [slice(i,i+1) for i in ind[::-1]]+[slice(nstates)]
                            tab[ind] = np.array(p[1].split(","), dtype=float)
                        factor = Factor(nodes, tab)
                    network.add_factor(factor)
                else:
                    raise NotImplementedError("Unknown block type: %s"%b_type)
                current_block = ""
        f.close()

        return network

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

    def __str__(self):
        s = "variable %s {\n"%(self.name)
        s += "  type discrete [ %d ] { %s };\n"%\
                (len(self.states), ", ".join([str(s) for s in self.states]))
        s += "}\n"
        return s

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
        self.table = table

    def __str__(self):
        if len(self.nodes) == 1:
            s = "probability ( %s ) {\n  table %s;\n}\n"%\
                (self.nodes[0].name,
                 ", ".join([str(v) for v in self.table]))
        else:
            s = "probability ( %s | %s ) {\n"%(self.nodes[0].name,
                      ", ".join([v.name for v in self.nodes[1:]]))
            for ind in itertools.product(*[n.states for n in self.nodes[1:]]):
                s += "  ( %s ) "%(", ".join([str(i) for i in ind]))
                s += ", ".join([str(self.evaluate(s, *ind))
                    for s in self.nodes[0].states])
                s += ";\n"
            s += "}\n"
        return s

    def evaluate(self, *args):
        """
        Evaluate this factor for a particular set of values.

        ## Arguments

        * `*args` (list): A list of `str`s. This is the set of values that
          the factor will be evaluated for.

        ## Returns

        * `value` (float): The value of the factor at this position.

        """
        assert len(args) == len(self.nodes)
        ind = [self.nodes[i].states.index(args[i]) for i in range(len(args))]
        # Don't forget that the indexing is in _reverse_ order.
        ind = [slice(i,i+1) for i in ind[::-1]]
        return float(self.table[ind])

def run_tests():
    import os
    fn = os.path.join(*(list(os.path.split(__file__)[:-2])+["data","test.bif"]))
    print "Asserting validity of file:", fn
    assert str(Network.from_file(fn)) == open(fn).read()

if __name__ == '__main__':
    run_tests()

