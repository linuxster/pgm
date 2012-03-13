#!/usr/bin/env python
# encoding: utf-8
"""
Problem 2 of problem set 3 of PGM (Spring 2012).

"""

__all__ = ["Factor", "Node", "read_uai"]

import numpy as np

class Factor(object):
    def __init__(self, variables, factor):
        self.variables = variables
        self.factor = np.array(factor)

    def get_factor(self, dir_edge):
        if np.all(self.variables == np.array(dir_edge)):
            return self.factor.reshape((2,2)).T
        return self.factor.reshape((2,2))

class Node(object):
    def __init__(self, variable):
        self.variable = variable
        self.psi = np.ones(2)
        self.parent = None
        self.children = {}
        self.factors = []

    def __str__(self):
        return "Node(%s)"%self.variable

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return self.variable

    def neighbors(self):
        return [(int(f.variables[f.variables != self.variable]), f)
                for f in self.factors]

    def add_child(self, child, f):
        assert child.parent is None
        factor = f.get_factor([self.variable, child.variable])
        child.parent = (self, factor.T)
        self.children[str(child.variable)] = (child, factor)

    def get_leaves(self):
        if len(self.children) == 0:
            return [self.variable]
        r = []
        for k in self.children:
            r += self.children[k][0].get_leaves()
        return r

    def upward_pass(self):
        if len(self.children) == 0:
            print "leaf:", self.variable, "<-", self.parent[0].variable
            return self.psi
        m = np.ones(2)
        for k in self.children:
            m *= self.children[k][0].upward_pass()
        message = self.psi[None,:] * m[:,None]
        if self.parent is not None:
             message *= self.parent[1]
        message = np.sum(message, axis=0)
        print self.variable,
        if self.parent is not None:
            print "<-", self.parent[0].variable
        else:
            print
        return message/np.sum(message)

def read_uai(fn, root_var):
    f = open(fn)
    f.readline() # ignore the first line

    nvars = int(f.readline())
    f.readline() # ignore the cardinalities
    ncliques = int(f.readline())

    cliques = [np.array(f.readline().split()[1:], dtype=int)
            for i in range(ncliques)]

    tables = []
    while True:
        l = f.readline()
        if len(l) > 1:
            n = int(l)
            tmp = []
            while len(tmp) < n:
                tmp += [float(i) for i in f.readline().split()]
            tables.append(tmp)
        elif len(l) == 0:
            break
    f.close()

    nodes = [Node(i) for i in range(nvars)]
    factors = []
    for c in zip(cliques, tables):
        if len(c[0]) == 1:
            nodes[c[0]].psi *= c[1]
        else:
            factors.append(Factor(*c))
            nodes[c[0][0]].factors.append(factors[-1])
            nodes[c[0][1]].factors.append(factors[-1])

    # Recursively build the tree from a chosen root node
    def build_node(var, parent=None):
        node = nodes[var]
        neighbors = node.neighbors()
        for n in neighbors:
            if parent is None or n[0] != parent:
                node.add_child(build_node(n[0], var), n[1])
        return node

    return build_node(root_var)

if __name__ == '__main__':
    root = read_uai("data/kitchen.uai", 85)
    print root.upward_pass()

