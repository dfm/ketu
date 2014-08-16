# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["BasicLikelihood", "GPLikelihood"]

import numpy as np
from scipy.linalg import cho_solve
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

        # Estimate the hyperparameters.
        var = float(np.var(lc.flux))
        scale = np.median(np.diff(lc.time)) * integrated_time(lc.flux)
        self.gp = george.GP(var * ExpSquaredKernel(scale ** 2))
        self.gp.compute(lc.time, lc.ferr)
        self.fm1 = self.flux - 1
        self.ll0 = -0.5*np.dot(self.fm1, cho_solve(self.gp._factor, self.fm1))

    def lnlike(self, model):
        Cf = cho_solve(self.gp._factor, self.fm1)
        m = model(self.time)
        Cm = cho_solve(self.gp._factor, m)
        ivar = np.dot(m, Cm)
        if ivar == 0.0:
            return 0.0, 0.0, 0.0
        depth = np.dot(m, Cf) / ivar
        dll = -0.5 * np.dot(self.fm1 - depth * m, Cf - depth * Cm) - self.ll0
        return dll, depth, ivar


class GPLikelihood(Pipeline):

    def get_result(self, **kwargs):
        result = self.parent.query(**kwargs)
        result["data"] = map(GPLCWrapper, result.pop("data"))
        return result
