#!/usr/bin/env python
"""
Latent Dirichlet allocation.

By: Dan Foreman-Mackey
Based on: Blei, Ng & Jordan (2003)
For: Probabilistic Graphical Models @ NYU 2012

"""

__all__ = ["LDA"]

import numpy as np

class LDA(object):
    def __init__(self, fn=None):
        # I like to let my samplers own their own random number generators
        # so here I'm initializing on from numpy.
        self._random = np.random.mtrand.RandomState()

        # Read alpha, beta and the word list in from a .ready file.
        f = open(fn, "r")
        ntopics = int(f.readline())
        alpha = np.array(f.readline().split(), dtype=float)
        assert len(alpha) == ntopics
        beta = []
        self._words = []
        for line in f:
            c = line.split()
            self._words.append(c[0])
            beta.append([v for v in c[1:]])
        f.close()
        beta = np.array(beta, dtype=float).T

        # Make sure that alpha and beta are properly normalized.

        # `alpha` should sum to 1. NOTE: `alpha.shape = (ntopics,)`.
        self.alpha = alpha/np.sum(alpha)

        # For a particular topic, `i`, `sum(beta[i,:]) == 1`.
        # NOTE: `beta.shape = (ntopics, nwords)`
        self.beta  = beta/(np.sum(beta, axis=1)[:, None])

        # The word distribution in these particular documents is just one
        # count of each word, I think.
        self.w = np.arange(self.beta.shape[1])

        # We can initialize the topic distribution as uniform.
        # NOTE: `theta.shape = (ntopics,)`
        self.theta = np.ones(len(self.alpha), dtype=float)/len(self.alpha)

        # The actually per-word topics are initially chosen to be random.
        # NOTE: `z.shape = (nwords,)`
        self.z = self._random.randint(len(self.alpha), size=len(self.w))

    def sample_theta(self):
        alpha2 = self.alpha+np.sum(self.z[:,None] == \
                np.arange(len(self.alpha))[None,:], axis=0)
        alpha2 /= np.sum(alpha2)
        self.theta = self._random.dirichlet(alpha2)

    def sample_topics(self):
        alpha2 = self.beta[:, self.w] * self.alpha[:, None]
        alpha2 /= np.sum(alpha2, axis=0)[None, :]
        for k in range(alpha2.shape[1]):
            mask = self._random.multinomial(1, alpha2[:,k]) == 1
            self.z[k] = int(np.arange(len(self.alpha))[mask])

    def gibbs_sample(self, iterations):
        """
        Sample the conditional distribution p ( theta, z | w, alpha, beta )
        using Gibbs sampling.

        """
        for i in xrange(iterations):
            self.sample_theta()
            self.sample_topics()

if __name__ == '__main__':
    lda = LDA(fn="data/abstract_nips21_NIPS2008_0517.txt.ready")

    lda.gibbs_sample(100)

