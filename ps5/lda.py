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
        self.z = self.random.randint(len(self.alpha), size=len(self.w))

    # @property
    # def words(self):
    #     """
    #     Return the list of words that appear in the generated documents.

    #     """
    #     if len(self._words) == 0:
    #         return self.w
    #     r = []
    #     for m in range(self.w.shape[0]):
    #         r.append([])
    #         for n in range(self.w.shape[1]):
    #             r[-1].append(self._words[self.w[m,n]])
    #     return r

    # def initialize(self, M, N):
    #     """
    #     Set up the model to get ready for sampling.

    #     """
    #     self.theta = np.ones(M)/float(M)
    #     self.z     = np.zeros([M, N], dtype=int)
    #     self.w     = np.zeros([M, N], dtype=int)

    # def generate_theta(self, M):
    #     """
    #     Sample the topic distribution from the Dirichlet distribution and
    #     save its state.

    #     """
    #     self.theta = self._random.dirichlet(self.alpha, size=M)
    #     return self.theta

    # def generate_topics(self, M, N):
    #     """
    #     Draw topic samples from the sample distribution.

    #     """
    #     for m in range(M):
    #         for n in range(N):
    #             mask = self._random.multinomial(1, self.theta[m, :])
    #             v = np.arange(self.theta.shape[1])[mask == 1]
    #             self.z[m, n] = int(v)

    # def generate_words(self, M, N):
    #     for m in range(M):
    #         for n in range(N):
    #             mask = self._random.multinomial(1, self.beta[self.z[m, n],:])
    #             v = np.arange(self.beta.shape[1])[mask == 1]
    #             self.w[m, n] = int(v)

    # def generate(self, M, N):
    #     """
    #     Generate `M` documents with `N` words each sampled from the LDA with
    #     the given `alpha` and `beta` values.

    #     """
    #     self.initialize(M, N)
    #     self.generate_theta(M)
    #     self.generate_topics(M, N)
    #     self.generate_words(M, N)

    def sample_theta(self):
        alpha2 = np.sum(np.sum(self.z[:,:,None] == \
                np.arange(len(self.alpha))[None,None,:], axis=0), axis=0)
        self.theta = self._random.dirichlet(alpha2)

    def sample_topics(self):
        pass

    def gibbs_sample(self, iterations):
        """
        Sample the conditional distribution p ( theta, z | w, alpha, beta )
        using Gibbs sampling.

        """
        self.w = np.atleast_2d(np.arange(self.beta.shape[1])).T

        self.initialize(*self.beta.shape)

        for i in xrange(iterations):
            self.sample_theta()
            self.sample_topics()

if __name__ == '__main__':
    lda = LDA(fn="data/abstract_nips21_NIPS2008_0517.txt.ready")

    # lda.gibbs_sample(1)

