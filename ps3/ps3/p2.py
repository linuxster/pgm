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
            return [self]
        r = []
        for k in self.children:
            r += self.children[k][0].get_leaves()
        return r

    def up_message(self):
        try:
            return self._up_message
        except AttributeError:
            self._up_message = self.upward_pass()
            return self._up_message

    def upward_pass(self):
        if len(self.children) == 0:
            return self.psi
        m = np.ones(2)
        for k in self.children:
            m *= self.children[k][0].up_message()
        message = self.psi[None,:] * m[:,None]
        if self.parent is not None:
             message *= self.parent[1]
        message = np.sum(message, axis=0)
        return message/np.sum(message)

    def down_messages(self):
        try:
            return self._down_messages
        except AttributeError:
            self._down_messages = self.downward_pass()
            return self._down_messages

    def downward_pass(self):
        if self.parent is not None:
            dm = self.parent[0].down_messages()
            m_par = dm[str(self.variable)]
        else:
            m_par = np.ones(2)
        if len(self.children) == 0:
            m = np.sum(self.psi[None,:] * m_par, axis=0)
            return m/np.sum(m)
        m = dict([(k, m_par) for k in self.children])
        for k in self.children:
            for j in self.children:
                if j != k:
                    m[k] *= self.children[j][0].up_message()
            m[k] = self.psi[None,:] * m[k][:,None] * self.children[k][1]
            m[k] = np.sum(m[k], axis=0)
            m[k] /= np.sum(m[k])

        return m

    def full_pass(self):
        for n in self.get_leaves():
            n.down_messages()

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
    root.full_pass()

