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
    def __init__(self, fn=None, alpha=None, beta=None, rstate=None):
        assert (fn is None and alpha is not None and beta is not None) or \
               (fn is not None and alpha is None and beta is None)

        self._words = []
        if fn is not None:
            f = open(fn, "r")
            ntopics = int(f.readline())
            alpha = np.array(f.readline().split(), dtype=float)
            assert len(alpha) == ntopics
            beta = []
            for line in f:
                c = line.split()
                self._words.append(c[0])
                beta.append([v for v in c[1:]])
            f.close()
            beta = np.array(beta, dtype=float).T

        self.alpha = alpha/np.sum(alpha)
        self.beta  = beta/(np.sum(beta, axis=1)[:, None])

        # I like to let my samplers own their own random number generators
        # so here I'm initializing on from numpy.
        self._random = np.random.mtrand.RandomState()
        try:
            self._random.set_state(rstate)
        except:
            pass

    @property
    def words(self):
        """
        If there was a list of words provided, produce the list of words that
        appear in the generated documents.

        """
        if len(self._words) == 0:
            return self.w
        r = []
        for m in range(self.w.shape[0]):
            r.append([])
            for n in range(self.w.shape[1]):
                r[-1].append(self._words[self.w[m,n]])
        return r

    def initialize(self, M, N):
        """
        Set up the model to get ready for sampling.

        """
        self.theta = np.ones(M)/float(M)
        self.z     = np.zeros([M, N], dtype=int)
        self.w     = np.zeros([M, N], dtype=int)

    def sample_theta(self, M):
        """
        Sample the topic distribution from the Dirichlet distribution and
        save its state.

        """
        self.theta = self._random.dirichlet(self.alpha, size=M)
        return self.theta

    def sample_topics(self, M, N):
        """
        Draw topic samples from the sample distribution.

        """
        for m in range(M):
            for n in range(N):
                mask = self._random.multinomial(1, self.theta[m, :])
                v = np.arange(self.theta.shape[1])[mask == 1]
                self.z[m, n] = int(v)

    def sample_words(self, M, N):
        for m in range(M):
            for n in range(N):
                mask = self._random.multinomial(1, self.beta[self.z[m, n],:])
                v = np.arange(self.beta.shape[1])[mask == 1]
                self.w[m, n] = int(v)

    def generate(self, M, N):
        """
        Generate `M` documents with `N` words each sampled from the LDA with
        the given `alpha` and `beta` values.

        """
        self.initialize(M, N)
        self.sample_theta(M)
        self.sample_topics(M, N)
        self.sample_words(M, N)

if __name__ == '__main__':
    lda = LDA(fn="data/abstract_nips21_NIPS2008_0517.txt.ready")
    lda.generate(3, 10)
    print [" ".join(l) for l in lda.words]

    import matplotlib.pyplot as pl
    pl.hist(lda.w.flatten(), lda.beta.shape[-1], normed=True, histtype="step", color="k")

    truth = []
    for line in open("data/abstract_nips21_NIPS2008_0517.txt"):
        for w in line.split():
            try:
                truth.append(lda._words.index(w))
            except ValueError:
                pass
    pl.hist(truth, lda.beta.shape[-1], normed=True, histtype="step", color="r")

    pl.show()

