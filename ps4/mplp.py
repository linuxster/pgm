"""
Implementation of the max-product linear program for PGM class.

Dan Foreman-Mackey (danfm@nyu.edu)

"""

import numpy as np

class Edge(object):
    def __init__(self, a, b, theta):
        self.a = a
        self.b = b

        self.theta = theta.reshape(len(a), len(b))
        self.messages = [np.zeros(len(a)), np.zeros(len(b))]

        self.a.add_neighbour(self.b, self)
        self.b.add_neighbour(self.a, self)

    def evaluate(self, ia, va, vb):
        if ia == self.a:
            return self.theta[va, vb]
        elif ia == self.b:
            return self.theta[vb, va]
        raise Exception("Not the right variable.")

    def get_thetabar(self):
        return self.theta - self.messages[0][:,None] - self.messages[1][None,:]

    def update_messages(self):
        dfa = self.a.thetabar - self.messages[0]
        dfb = self.b.thetabar - self.messages[1]
        arg = self.theta + dfb[None,:] + dfa[:,None]

        # Equation 1.19 from [1].
        self.messages[0] = -dfa + 0.5 * np.max(arg, axis=1)
        self.messages[1] = -dfb + 0.5 * np.max(arg, axis=0)

        self.a.set_message(self.b, self.messages[0])
        self.b.set_message(self.a, self.messages[1])

class Variable(object):
    def __init__(self, _id, theta):
        self.dim = len(theta)
        self._id = _id
        self.theta = np.array(theta)
        self.neighbours = []
        self.edges = []
        self.messages = []

    def __len__(self):
        return self.dim

    def __hash__(self):
        return self._id

    @property
    def thetabar(self):
        tb = self.theta + sum([m for m in self.messages])
        return tb

    def add_neighbour(self, other, edge):
        self.neighbours.append(other)
        self.edges.append(edge)
        self.messages.append(np.zeros(len(self)))

    def set_message(self, other, message):
        self.messages[self.neighbours.index(other)] = np.array(message)

class Network(object):
    def __init__(self):
        self.names = []
        self.variables = []
        self.edges = []

    def add_variable(self, name, theta):
        self.variables.append(Variable(name, theta))
        self.names.append(name)

    def add_edge(self, a, b, theta):
        ia, ib = self.names.index(a), self.names.index(b)
        self.edges.append(Edge(self.variables[ia], self.variables[ib], theta))

    def read_uai(self, fn, log=False):
        """
        Read a UAI formatted file. If `log` is True, the logarithms of the
        probability tables are given.

        """
        f = open(fn)

        # Start by reading in/ignoring the metadata at the top of the file.
        f.readline()
        nvars = int(f.readline())
        f.readline()
        ncliques = int(f.readline())

        # Read the list of cliques.
        variables = {}
        edges = {}
        for i in xrange(ncliques):
            l = f.readline().split()
            if int(l[0]) == 1:
                variables[l[1]] = i
            elif int(l[0]) == 2:
                edges[" ".join([l[1],l[2]])] = i
            else:
                raise NotImplementedError("Pairwise only")

        # Read in the table values.
        tables = []
        count = 0
        for line in f:
            if count == 0:
                try:
                    count = int(line)
                except ValueError:
                    pass
                else:
                    tables.append(np.zeros(count))
            else:
                for v in line.split():
                    if log:
                        val = float(v)
                    else:
                        val = np.log(np.max([float(v), 1e-8])) # Underflow
                    tables[-1][-count] = val
                    count -= 1

        f.close()

        # Build the Network.
        for i in xrange(nvars):
            self.add_variable(i, tables[variables[str(i)]])
        for k in sorted(edges, key=lambda k: edges[k]):
            i, j = k.split()
            self.add_edge(int(i), int(j), tables[edges[k]])

    def update_messages(self):
        [e.update_messages() for e in self.edges]

    def map_query(self):
        J = 0.0
        x = np.zeros(len(self.variables), dtype=int)

        # Add in the single node contributions.
        for i, v in enumerate(self.variables):
            x[i] = np.argmax(v.thetabar)
            J += v.thetabar[x[i]]

        # And the edge contributions too.
        for e in self.edges:
            J += np.max(e.get_thetabar())

        thtot = 0.0
        for i, v in enumerate(self.variables):
            thtot += v.theta[x[i]] \
                    + np.sum([v.edges[j].evaluate(v, x[i], x[self.variables.index(o)])
                        for j, o in enumerate(v.neighbours)])

        return x, J, thtot

    def run_mplp(self, maxiter=-1, tol=0.0002, outfile=None):
        if outfile is not None:
            f = open(outfile, "w")
            f.write("# iteration, dual objective, integer solution, integrality gap\n")
            f.close()
        i = 0
        best_x = None
        best_th = -1e10
        best_J  = -1e10
        best_iter = 0
        while True:
            self.update_messages()
            x, J, th = network.map_query()
            if outfile is not None:
                f = open(outfile, "a")
                f.write("%d, %e, %e, %e\n"%(i, J, th, J-th))
                f.close()

            if th > best_th:
                best_iter = i
                best_x  = x
                best_th = th
                best_J  = J

            if i > 0 and (np.abs(J-J0) < tol or (i >= maxiter and maxiter > 0)):
                break
            J0 = float(J)
            i += 1

        f = open(outfile, "a")
        if i == maxiter:
            f.write("MPLP didn't converge after %d iterations.\n"%i)
        else:
            f.write("MPLP converged after %d iterations.\n"%i)
        f.close()
        return best_x, best_J, best_th, best_iter

if __name__ == "__main__":
    names = [line.strip() for line in open("data/names.txt")]

    for room in ["kitchen", "office"]:
        network = Network()

        print "In the %s,"%room,
        network.read_uai("data/%s.uai"%room, log=False)
        x, J, th, i = network.run_mplp(outfile="%s.out"%room)
        if np.any(x[:len(names)]):
            print "we detected:",
            for i in range(len(names)):
                if x[i]:
                    print names[i]+",",
            print
        else:
            print "No objects detected."

    print
    for fn in ["1exm"]:
        print fn
        network = Network()
        network.read_uai("data/%s.UAI.LG"%fn, log=True)
        x, J, th, i = network.run_mplp(outfile="%s.out"%fn, maxiter=200)
        print "Best solution at iteration %d."%i
        print "The integrality gap is: %e."%np.abs(J-th)
        print "And the assignment is:"
        print x

