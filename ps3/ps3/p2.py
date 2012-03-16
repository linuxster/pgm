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
        self.name = "No Name"

    def __str__(self):
        return "Node(%s)"%self.variable

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return self.variable

    def reset(self):
        try:
            del self._up_message
            del self._down_messages
        except AttributeError:
            pass
        [self.children[k][0].reset() for k in self.children]

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

    def up_message(self, do_map=False):
        try:
            return self._up_message
        except AttributeError:
            self._up_message = self.upward_pass(do_map)
            return self._up_message

    def upward_pass(self, do_map):
        if len(self.children) == 0:
            return self.psi
        m = np.ones(2)
        for k in self.children:
            m *= self.children[k][0].up_message(do_map=do_map)
        message = self.psi[None,:] * m[:,None]
        if self.parent is not None:
             message *= self.parent[1]
        if do_map:
            message = np.max(message, axis=0)
        else:
            message = np.sum(message, axis=0)
        return message/np.sum(message)

    def down_messages(self, do_map=False):
        try:
            return self._down_messages
        except AttributeError:
            self._down_messages = self.downward_pass(do_map)
            return self._down_messages

    def downward_pass(self, do_map):
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
                    m[k] *= self.children[j][0].up_message(do_map=do_map)
            m[k] = self.psi[None,:] * m[k][:,None] * self.children[k][1]
            if do_map:
                m[k] = np.max(m[k], axis=0)
            else:
                m[k] = np.sum(m[k], axis=0)
            m[k] /= np.sum(m[k]) # watching out for underflow!

        return m

    def full_pass(self, do_map):
        for n in self.get_leaves():
            n.down_messages(do_map=do_map)

    def prob(self):
        r = self.psi * self.up_message()
        dm = self.down_messages()
        for k in dm:
            # For some reason I'm getting underflow if I don't normalize here
            # too... strange.
            r *= dm[k] / np.sum(dm[k])
        return r / np.sum(r)

    def map(self):
        return np.argmax(self.prob())

def read_uai(fn, root_var, names=None):
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
    if names is not None:
        for i in range(len(names)):
            nodes[i].name = names[i]

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

    return build_node(root_var), nodes

def main():
    print "Problem 2"
    print "======= ="
    print

    names = [n.strip() for n in open("data/names.txt")]
    scenes = {"Office": "data/office.uai", "Kitchen": "data/kitchen.uai"}

    for scene in scenes:
        root, nodes = read_uai(scenes[scene], 85, names=names)

        print "For the \"%s\" scene:"%scene
        print "--- ---  " + "-"*len(scene) + "  -----"

        # Start by running the map query.
        root.full_pass(True)
        m = np.array([nodes[i].map() for i in range(len(names))], dtype=bool)
        print "The following objects were deemed present in the image"\
                + " based on their MAP assignment:"
        print ", ".join([nodes[i].name for i in np.arange(len(names))[m]])
        print

        # Reset and then calculate the marginals.
        root.reset()
        root.full_pass(False)
        p = np.array([nodes[i].prob() for i in range(len(names))])

        print "The following objects were observed with probability > 0.8:"
        print ", ".join([nodes[i].name
                for i in np.arange(len(names))[p[:,1]>0.8]])
        print
        if np.sum(p[:,1]>0.6) == np.sum(p[:,1]>0.8):
            print "No other objects were observed with probability "\
                    + "0.6 < p < 0.8."
        else:
            print "Besides these, the following objects were observed with "\
                    + "probability 0.6 < p < 0.8:"
            print ", ".join([nodes[i].name
                for i in np.arange(len(names))[(p[:,1]>0.6) * (p[:,1]<0.8)]])
        print

if __name__ == "__main__":
    main()

