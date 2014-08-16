# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["BasicLikelihood", "GPLikelihood"]

import numpy as np
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from emcee.autocorr import integrated_time

import george
from george.kernels import ExpSquaredKernel

from .pipeline import Pipeline


class LCWrapper(object):

    def __init__(self, lc):
        self.lc = lc
        self.time = lc.time
        self.flux = lc.flux
        self.ferr = lc.ferr
        self.ivar = 1.0 / self.ferr ** 2

    def lnlike(self, model):
        return -0.5 * np.sum((self.flux - model(self.time)) ** 2 * self.ivar)


class BasicLikelihood(Pipeline):

    def get_result(self, **kwargs):
        result = self.parent.query(**kwargs)
        result["data"] = map(LCWrapper, result.pop("data"))
        return result


class GPLCWrapper(LCWrapper):

    def __init__(self, lc):
        super(GPLCWrapper, self).__init__(lc)

        # Convert to PPM.
        self.flux = (self.flux - 1) * 1e6
        self.ferr *= 1e6

        # Estimate the hyperparameters.
        var = float(np.var(self.flux))
        scale = np.median(np.diff(self.time)) * integrated_time(self.flux)
        self.gp = george.GP(var * ExpSquaredKernel(scale ** 2))
        self.gp.compute(self.time, self.ferr)

        # Compute the ln-likelihood of the null hypothesis.
        self.ll0 = 0.0
        self.ll0, _, _ = self.lnlike(lambda t: np.zeros_like(t), order=1)

    def lnlike(self, model, order=2):
        # Precompute some useful factors.
        Cf = cho_solve(self.gp._factor, self.flux)
        m = np.vander(model(self.time), order)
        mT = m.T
        Cm = cho_solve(self.gp._factor, m)
        S = np.dot(mT, Cm)

        # Solve for the maximum likelihood depth.
        try:
            factor = cho_factor(S, overwrite_a=True)
            w = cho_solve(factor, np.dot(mT, Cf), overwrite_b=True)
        except LinAlgError:
            return 0.0, 0.0, 0.0
        else:
            ivar = 1.0 / cho_solve(factor, np.eye(len(S)),
                                   overwrite_b=True)[0, 0]
        depth = w[0]

        # Compute the value of the likelihood at its maximum.
        dll = -0.5*np.dot(self.flux-np.dot(m, w), Cf-np.dot(Cm, w)) - self.ll0
        return dll, depth, ivar


class GPLikelihood(Pipeline):

    def get_result(self, **kwargs):
        result = self.parent.query(**kwargs)
        result["data"] = map(GPLCWrapper, result.pop("data"))
        return result
