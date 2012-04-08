#!/usr/bin/env python
"""
Latent Dirichlet allocation.

By: Dan Foreman-Mackey
Based on: Blei, Ng & Jordan (2003)
For: Probabilistic Graphical Models @ NYU 2012

"""

__all__ = ["LDA"]

import numpy as np
import scipy.special as sp

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
        # NOTE: `alpha2.shape = (ntopics, nwords)`
        alpha2 = self.beta[:, self.w] * self.theta[:, None]
        alpha2 /= np.sum(alpha2, axis=0)[None, :]
        inds = np.arange(len(self.alpha))
        self.z = np.array([int(
                        inds[self._random.multinomial(1, alpha2[:,n]) == 1])
                for n in range(len(self.w))])
        return self.z

    def gibbs(self, iterations, burnin=50):
        """
        Sample the conditional distribution p ( theta, z | w, alpha, beta )
        using Gibbs sampling.

        ## Arguments

        * `iterations` (int): The number of steps of Gibbs sampling to run.

        ## Returns

        * `thetas` (numpy.ndarray): The samples of the cumulative expectation
          value for the `theta` distribution. This will have the shape
          `(iterations-burnin, ntopics)`.

        """
        # Make sure that each algorithm starts from a similar first guess.
        self.reset()

        # Allocate memory for the Markov chain.
        thetas = np.zeros((iterations, len(self.theta)))
        topics = np.zeros((iterations, len(self.w)))

        for i in xrange(iterations):
            thetas[i,:] = self._sample_theta()
            topics[i,:] = self._sample_topics()

        # Calculate the _cumulative_ expectation value of `theta`.
        thetas = np.cumsum(thetas[burnin:], axis=0)\
                / (np.arange(iterations-burnin) + 1)[:, None]

        return thetas

    def _approx_theta(self, topics):
        """
        Used by the collapsed Gibbs algorithm to estimate the expectation
        value of `theta` given a set of `z` samples.

        ## Arguments

        * `topics` (numpy.ndarray): The list of topic samples. This object
          should have the shape `(T, nwords)`, where `T` is the number of
          samples.

        ## Returns

        * `theta` (numpy.ndarray): The estimated `theta` with length
          `ntopics`.

        """
        T = topics.shape[0]
        tmp = np.sum(topics[:, :, None] \
                == np.arange(len(self.alpha))[None, None, :], axis=1)
        theta = T * self.alpha + np.sum(tmp, axis=0)
        theta /= T * (np.sum(self.alpha) + len(self.z))
        return theta

    def collapsed_gibbs(self, iterations, burnin=50, get_theta=True):
        """
        Run the collapsed Gibbs sampling algorithm for a given number of
        iterations.

        ## Arguments

        * `iterations` (int): The number of iterations to run.

        ## Keyword Arguments

        * `burnin` (int): The number of iterations to discard from the
          beginning of the chain. (default: 50)
        * `get_theta` (bool): Flag indicating whether or not the expectation
          value of `theta` will be calculated at each step in the chain. The
          default is `True` and otherwise, it will only be returned for the
          final step.

        ## Returns

        * `theta` (numpy.ndarray): If `get_theta` was true, this will return
          an array of shape `(iterations-burnin-1, ntopics)` with the
          _cumulative_ expectation value of `theta` as a function of
          iterations. Otherwise it will only return the _final_ estimated
          expectation value of `theta` (i.e. a list of length `ntopics`).

        """
        # Make sure that each algorithm starts from a similar first guess.
        self.reset()

        if get_theta:
            thetas = np.zeros((iterations-burnin-1, len(self.alpha)))
        topics = np.zeros((iterations, len(self.w)))

        for i in xrange(iterations):
            if not get_theta and i%500 == 0:
                print "Collapsed Gibbs: Iteration #%d"%i
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

            if get_theta and i > burnin:
                thetas[i-burnin-1, :] = self._approx_theta(topics[burnin:i])

        if not get_theta:
            return self._approx_theta(topics[burnin:])

        return thetas

    def variational(self, maxiter=500, tol=0):
        k, N  = len(self.alpha), len(self.w)
        phi   = np.ones((N, k), dtype=float) / k
        gamma = self.alpha + float(N)/k

        thetas = np.zeros((maxiter, k))

        for iteration in xrange(maxiter):
            phi = self.beta[:, self.w].T * np.exp(sp.psi(gamma))[None,:]
            phi /= np.sum(phi, axis=1)[:, None]
            gamma2 = self.alpha + np.sum(phi, axis=0)

            delta = np.sum(np.abs(gamma-gamma2))

            gamma = gamma2
            thetas[iteration,:] = gamma/np.sum(gamma)

            if delta <= tol:
                break

        return thetas[:iteration]

if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as pl

    lda = LDA(fn="data/abstract_nips21_NIPS2008_0517.txt.ready")

    if "--truth" in sys.argv:
        print "Generating truth.txt using collapsed Gibbs sampling..."
        theta = lda.collapsed_gibbs(10000, burnin=50, get_theta=False)
        true_th = np.array(theta)
        f = open("truth.txt", "w")
        f.write(" ".join(["%e"%th for th in true_th]))
        f.close()
    else:
        true_th = np.array(open("truth.txt").readline().split(), dtype=float)

    burnin = 50

    plot_l2 = lambda theta, fmt, label: \
            pl.plot(np.arange(theta.shape[0]),
                    np.sum((theta-true_th[None, :])**2, axis=1), fmt,
                    lw=2, label=label)

    print "Running collapsed Gibbs sampling..."
    theta = lda.collapsed_gibbs(1000, burnin=burnin)
    plot_l2(theta, "-k", "Collapsed Gibbs")

    print "Running standard Gibbs sampling..."
    theta = lda.gibbs(1000, burnin=burnin)
    plot_l2(theta, "--k", "Standard Gibbs")

    print "Running variational LDA..."
    theta = lda.variational()
    plot_l2(theta, ":k", "Variational")

    pl.legend()
    pl.xlabel(r"$\mathrm{iterations}$", fontsize=14)
    pl.ylabel(r"$L_2$", fontsize=14)

    pl.yscale("log")
    pl.xscale("log")

    pl.savefig("results.pdf")

