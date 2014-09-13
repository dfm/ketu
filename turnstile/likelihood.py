# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["GPLikelihood"]

import numpy as np
from emcee.autocorr import integrated_time
from scipy.linalg import cho_factor, cho_solve, LinAlgError

import george
from george.kernels import ExpSquaredKernel, Matern32Kernel

from .pipeline import Pipeline


class GPLikelihood(Pipeline):

    query_parameters = dict(
        matern=(False, False),
    )

    def __init__(self, *args, **kwargs):
        kwargs["cache"] = kwargs.pop("cache", False)
        super(GPLikelihood, self).__init__(*args, **kwargs)

    def get_result(self, query, parent_response):
        lcs = [LCWrapper(lc, matern=query["matern"])
               for lc in parent_response.light_curves]
        return dict(model_light_curves=lcs)


class LCWrapper(object):

    def __init__(self, lc, dist_factor=10.0, time_factor=0.1, matern=False):
        self.time = lc.time
        self.flux = lc.flux - 1.0
        self.ferr = lc.ferr

        # Convert to parts per thousand.
        self.flux *= 1e3
        self.ferr *= 1e3

        # Estimate the hyperparameters:
        # (a) the variance:
        self.var = np.var(self.flux)

        # (b) the time scale length:
        l = len(self.flux)
        tau = max(
            integrated_time(self.flux[i*l//n:(i+1)*l//n])
            for n in range(3) for i in range(n)
        )
        self.tau = time_factor * (np.median(np.diff(self.time)) * tau) ** 2
        if matern:
            self.tau *= 10.0

        # (c) the distance scale length:
        x = lc.predictors - 1
        d = (x[np.random.randint(len(x), size=10000)] -
             x[np.random.randint(len(x), size=10000)])
        self.ell = dist_factor * np.mean(np.sum(d**2, axis=1))

        # Include time as an input feature
        x = np.concatenate((np.atleast_2d(self.time).T, x), axis=1)
        ndim = x.shape[1]

        scale = np.append(self.tau, self.ell + np.zeros(ndim-1))
        if matern:
            self.kernel = self.var * Matern32Kernel(scale, ndim)
        else:
            self.kernel = self.var * ExpSquaredKernel(scale, ndim)
        self.gp = george.GP(self.kernel, solver=george.HODLRSolver)
        self.gp.compute(x, self.ferr)

        # Compute the likelihood of the null model.
        self.ll0 = 0.0
        self.ll0, _, _ = self.lnlike(order=1)

    def linear_maximum_likelihood(self, model=None, order=2, y=None):
        if y is None:
            y = self.flux

        if model is None:
            model = np.zeros_like(self.time)
            order = 1
        else:
            model = model(self.time)
        m = np.vander(model, order)
        mT = m.T

        # Precompute some useful factors.
        Cf = self.gp.solver.apply_inverse(y)
        Cm = self.gp.solver.apply_inverse(m)
        S = np.atleast_2d(np.dot(mT, Cm))

        # Solve for the maximum likelihood model.
        factor = cho_factor(S, overwrite_a=True)
        w = cho_solve(factor, np.dot(mT, Cf), overwrite_b=True)
        sigma = cho_solve(factor, np.eye(len(S)), overwrite_b=True)

        return w, m, sigma, Cf, Cm

    def predict(self, model=None, order=2, y=None):
        try:
            w, m, sigma, Cf, Cm = \
                self.linear_maximum_likelihood(model, order, y=y)
        except LinAlgError:
            w, m, sigma, Cf, Cm = self.linear_maximum_likelihood(y=y)

        if len(w) > 1:
            sig = m[:, 0] * w[0]
            bkg = m[:, 1] * w[1]
        else:
            bkg = m[:, 0] * w[0]
            sig = np.zeros_like(bkg)

        return sig, bkg + np.dot(self.kernel.value(self.gp._x),
                                 Cf - np.dot(Cm, w))

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
