"""


"""

__all__ = ["Sentence"]

import numpy as np


_ntags = 10
_dimcats = [1, 2, 2, 201, 201]
_N = (np.sum(_dimcats) + _ntags) * _ntags


class Sentence(object):
    def __init__(self, fn):
        data = [line.split(",") for line in open(fn)]
        self.words = [d[0] for d in data]

        # The tags for each word.
        # `y.shape = (nwords,)`.
        self.y = np.array([d[1] for d in data], dtype=int) - 1
        assert self.y.shape == (len(self),)

        # The 5 features for each word
        # `x.shape = (nwords, 5)`.
        self.x = np.array([d[2:] for d in data], dtype=int) - 1
        assert self.x.shape == (len(self), len(_dimcats))

        # Calculate the true feature indicator vector given the correct
        # assignment `y`.
        # `f.shape == (4170,)`.
        self.f = self.feature_indicator(self.y)
        assert self.f.shape == (_N,)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return " ".join(self.words)

    def __len__(self):
        return len(self.words)

    def feature_indicator(self, y):
        """
        Given a set of assignments for the words and the observed features,
        compute the inficator vector `f(x, y)`.

        """
        f = np.zeros((_ntags + np.sum(_dimcats), _ntags))
        for w in range(len(self)):
            # First, add in the transitions from the current word to the
            # next one.
            if w < len(self) - 1:
                f[y[w + 1], y[w]] += 1

            # Then add in the features.
            n = _ntags  # The current location in the indicator vector.
            for i, x0 in enumerate(self.x[w]):
                f[n + x0, y[w]] += 1
                n += _dimcats[i]
        return f.flatten()

    def delta(self, y):
        """
        Given a proposal assignment `y`, calculate the update step for the
        Perceptron algorithm.

        """
        f = self.feature_indicator(y)
        return self.f - f

    def map_query(self, w):
        """
        Given a weight vector `w` and the observations `x`, compute the MAP
        assignment to `y`.

        """
        # The transition weights.
        wt = w[:_ntags ** 2].reshape((_ntags, _ntags))

        # The feature weights.
        wa = w[_ntags ** 2:].reshape((-1, _ntags))

        # Precompute the single node potentials.
        thetas = []
        for w in range(len(self)):
            n = 0
            thetas.append(np.zeros(_ntags))
            for i, x0 in enumerate(self.x[w]):
                thetas[-1] += wa[n + x0]
                n += _dimcats[i]

        # Do the first pass.
        messages1 = np.zeros((len(self) - 1, _ntags))
        for i in range(len(self) - 1):
            # Add in the edge potential to get `psi`.
            psi = wt + thetas[i][None, :]

            # Add the other messages. On the first pass, there
            # will only ever be the message from the previous pass.
            if i > 0:
                psi += messages1[i - 1][None, :]

            messages1[i] = np.max(psi, axis=1)

        # Do the second pass from the top down.
        messages2 = np.zeros((len(self) - 1, _ntags))
        for i in range(1, len(self))[::-1]:
            psi = wt.T + thetas[i][:, None]
            if i < len(self) - 1:
                psi += messages2[i][:, None]

            messages2[i - 1, :] = np.max(psi, axis=0)

        # Compute the max marginals.
        marginals = np.zeros((len(self), _ntags))
        for i in range(len(self)):
            theta = thetas[i]
            if i > 0:
                theta += messages1[i - 1]
            if i < len(self) - 1:
                theta += messages2[i]
            marginals[i, :] = theta

        return np.argmax(marginals, axis=1)


class Perceptron(object):
    def __init__(self):
        self.w = np.zeros(_N, dtype=float)
        self.w0 = np.zeros_like(self.w)

    def run(self, sentences, maxiter=50):
        N = sum([len(s) for s in sentences])
        for i in range(maxiter):
            err = 0
            for s in sentences:
                y = s.map_query(self.w)
                err += np.sum(y != s.y)
                delta = s.delta(y)
                self.w += delta
            print err / float(N)
            self.w0 = np.array(self.w)


if __name__ == "__main__":
    s = [Sentence("data/test-{0}.txt".format(i)) for i in range(1, 200)]
    machine = Perceptron()
    machine.run(s)
