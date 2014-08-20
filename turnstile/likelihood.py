# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["GPLikelihood"]

import numpy as np
from emcee.autocorr import integrated_time
from scipy.linalg import cho_factor, cho_solve, LinAlgError

from .pipeline import Pipeline
from ._gp import compute_kernel_matrix


class GPLikelihood(Pipeline):

    def __init__(self, *args, **kwargs):
        kwargs["cache"] = kwargs.pop("cache", False)
        super(GPLikelihood, self).__init__(*args, **kwargs)

    def get_result(self, query, parent_response):
        lcs = map(LCWrapper, parent_response.light_curves)
        return dict(model_light_curves=lcs)


class LCWrapper(object):

    def __init__(self, lc, length_factor=10):
        self.time = lc.time
        self.flux = lc.flux
        self.ferr = lc.ferr
        self.predictors = lc.predictors - 1

        # Convert to PPM.
        self.flux = (self.flux - 1) * 1e6
        self.ferr *= 1e6

        # Estimate the hyperparameters.
        self.var = np.var(self.flux)
        scale = np.median(np.diff(self.time)) * integrated_time(self.flux)
        self.tau = scale ** 2
        x = self.predictors
        d = (x[np.random.randint(len(x), size=10000)] -
             x[np.random.randint(len(x), size=10000)])
        self.ell = length_factor * np.mean(np.sum(d**2, axis=1))

        # Build the kernel matrix.
        self.K = compute_kernel_matrix(self.var, self.tau, self.time, self.ell,
                                       x)
        Kobs = np.array(self.K)
        Kobs[np.diag_indices_from(Kobs)] += self.ferr ** 2
        self.factor = cho_factor(Kobs, overwrite_a=True)

        # Compute the likelihood of the null model.
        self.ll0 = 0.0
        self.ll0, _, _ = self.lnlike(order=1)

    def linear_maximum_likelihood(self, model=None, order=2):
        if model is None:
            model = np.zeros_like(self.time)
            order = 1
        else:
            model = model(self.time)
        m = np.vander(model, order)
        mT = m.T

        # Precompute some useful factors.
        Cf = cho_solve(self.factor, self.flux)
        Cm = cho_solve(self.factor, m)
        S = np.dot(mT, Cm)

        # Solve for the maximum likelihood model.
        factor = cho_factor(S, overwrite_a=True)
        w = cho_solve(factor, np.dot(mT, Cf), overwrite_b=True)
        sigma = cho_solve(factor, np.eye(len(S)), overwrite_b=True)

        return w, m, sigma, Cf, Cm

    def predict(self, model=None, order=2):
        try:
            w, m, sigma, Cf, Cm = self.linear_maximum_likelihood(model, order)
        except LinAlgError:
            w, m, sigma, Cf, Cm = self.linear_maximum_likelihood()
        return np.dot(m, w) + np.dot(self.K, Cf - np.dot(Cm, w))

    def lnlike(self, model=None, order=2):
        try:
            w, m, sigma, Cf, Cm = self.linear_maximum_likelihood(model, order)
        except LinAlgError:
            return 0.0, 0.0, 0.0

        depth = w[0]
        ivar = 1.0 / sigma[0, 0]

        # Compute the value of the likelihood at its maximum.
        dll = -0.5*np.dot(self.flux-np.dot(m, w), Cf-np.dot(Cm, w)) - self.ll0
        return dll, depth, ivar
