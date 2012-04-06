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
    """
    An object that implements the LDA topic model.

    *Note*: this has been special purposed for this particular NIPS abstract
    dataset. We're also assume that `alpha` and `beta` are known and given
    and that we're only trying to model the topic distribution in _one_
    abstract at a time.

    ## Arguments

    * `fn` (str): This should be the path to a `*.txt.ready` file.

    """
    def __init__(self, fn):
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

        # NOTE: `alpha.shape = (ntopics,)`.
        self.alpha = alpha

        # NOTE: `beta.shape = (ntopics, nwords)`
        self.beta  = beta

        # The word distribution in these particular documents is just one
        # count of each word, I think.
        self.w = np.arange(self.beta.shape[1])

    def reset(self):
        """
        Set `theta` and `z` to reasonable first guesses. This should be
        called before any algorithms are run.

        """
        # We can initialize the topic distribution as uniform.
        # NOTE: `theta.shape = (ntopics,)`
        self.theta = np.ones(len(self.alpha), dtype=float)/len(self.alpha)

        # The actually per-word topics are initially chosen to be random.
        # NOTE: `z.shape = (nwords,)`
        self.z = self._random.randint(len(self.alpha), size=len(self.w))

    def _sample_theta(self):
        """
        Used by the standard Gibbs sampling algorithm to draw samples from
        the posterior distribution on `theta`:

            p(theta | z, alpha) ~ Dirichlet(alpha2)

        where `alpha2` is given by the results from problem set 2:

            alpha2[k] = alpha[k] + sum(z == k)   .

        ## Returns

        * `theta` (numpy.ndarray): The sampled topic distribution. The shape
          of this object is `(ntopics,)`.

        """
        alpha2 = self.alpha+np.sum(self.z[:,None] == \
                np.arange(len(self.alpha))[None,:], axis=0)
        self.theta = self._random.dirichlet(alpha2)
        return self.theta

    def _sample_topics(self):
        """
        Used by the standard Gibbs sampling algorithm to sample the per-word
        topics from the posterior distribution:

            p(z[n] | w[n], beta, theta) ~ Multinomial(theta2)

        where `theta2` is given by `theta * beta[:, w[n]]` where you can
        think of `beta[:, w[n]]` as the _likelihood_ over topics given a
        particular word `w[n]` and `theta` as the _prior_ over topics. Also
        note that the multiplication in the above expression is
        _component-wise_.

        ## Returns

        * `topics` (numpy.ndarray): The sampled topics with shape `(nwords,)`.

        """
        alpha2 = self.beta[:, self.w] * self.alpha[:, None]
        alpha2 /= np.sum(alpha2, axis=0)[None, :]
        for k in range(alpha2.shape[1]):
            mask = self._random.multinomial(1, alpha2[:,k]) == 1
            self.z[k] = int(np.arange(len(self.alpha))[mask])
        return self.z

    def gibbs(self, iterations, burnin=50):
        """
        Sample the conditional distribution p ( theta, z | w, alpha, beta )
        using Gibbs sampling.

        ## Arguments

        * `iterations` (int): The number of steps of Gibbs sampling to run.

        ## Returns

        * `thetas` (numpy.ndarray): The samples of the `theta` distribution.
          This will have the shape `(iterations, ntopics)`.
        * `topics` (numpy.ndarray): The samples of the `z` distribution.
          This will have the shape `(iterations, nwords)`.

        """
        # Make sure that each algorithm starts from a similar first guess.
        self.reset()

        # Allocate memory for the Markov chain.
        thetas = np.zeros((iterations, len(self.theta)))
        topics = np.zeros((iterations, len(self.w)))

        for i in xrange(iterations):
            thetas[i,:] = self._sample_theta()
            topics[i,:] = self._sample_topics()

        return thetas[burnin:, :], topics[burnin:, :]

    def collapsed_gibbs(self, iterations, burnin=50):
        # Make sure that each algorithm starts from a similar first guess.
        self.reset()

        thetas = np.zeros((iterations-burnin-1, len(self.alpha)))
        topics = np.zeros((iterations, len(self.w)))

        for i in xrange(iterations):
            # Unfortunately, I think that we need to run the Gibbs sampling
            # in _series_ for the `z`s because they are *not* conditionally
            # independent. Slow!
            for n, zn in enumerate(self.z):
                m = np.arange(len(self.z)) != n
                inds = np.arange(len(self.alpha))[:,None]
                Nk = np.sum(self.z[m][None,:] == inds, axis=1)
                a = Nk + self.alpha

                b = self.beta[:, self.w[n]]

                p = a*b/np.sum(a*b)
                mask = self._random.multinomial(1, p) == 1
                self.z[n] = int(np.arange(len(self.alpha))[mask])
            topics[i,:] = self.z

            if i > burnin:
                # A little bit of magic to calculate `E[theta]`.
                T = i - burnin
                tmp = np.sum(topics[burnin:i, :, None] \
                        == np.arange(len(self.alpha))[None, None, :], axis=1)
                theta = T * self.alpha + np.sum(tmp, axis=0)
                theta /= T * (np.sum(self.alpha) + len(self.z))

                thetas[T-1, :] = theta

        return thetas, topics[burnin:,:]

if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as pl

    lda = LDA(fn="data/abstract_nips21_NIPS2008_0517.txt.ready")

    if "--truth" in sys.argv:
        theta, z = lda.collapsed_gibbs(500, burnin=50)
        true_th = theta[-1,:]
        f = open("truth.txt", "w")
        f.write(" ".join(["%e"%th for th in true_th]))
        f.close()
    else:
        true_th = np.array(open("truth.txt").readline().split(), dtype=float)

    print "Finished truth..."
    print np.argsort(true_th)[::-1]

    burnin = 1

    theta, z = lda.collapsed_gibbs(100, burnin=burnin)

    pl.plot(np.sum((theta-true_th[None, :])**2, axis=1))

    theta, z = lda.gibbs(100, burnin=burnin)

    # Compute the cumulative expectation value of `theta` given the Gibbs
    # chain.
    e_th = np.cumsum(theta, axis=0)/(np.arange(theta.shape[0]) + 1)[:, None]

    pl.plot(np.sum((e_th-true_th[None, :])**2, axis=1))

    # pl.yscale("log")

    pl.show()

    # mu = np.mean(theta, axis=0)
    # print np.argsort(mu)[::-1]

    # pl.plot(mu)

    # pl.figure()

    # for i in range(theta.shape[-1]):
    #     mask = theta[:,i] > 0
    #     y, x = np.histogram(theta[mask, i], 50, normed=True)

    #     pl.plot(0.5*(x[1:]+x[:-1]), np.log10(y), "k", alpha=0.5)

    # pl.ylabel(r"$\log_{10} \, \rho (\theta_k)$", fontsize=16.)
    # pl.xlabel(r"$\theta_k$", fontsize=16.)

    # pl.show()

