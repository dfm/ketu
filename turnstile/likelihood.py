# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["BasicLikelihood", "GPLikelihood"]

import numpy as np
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
        self.gp = george.GP(var * ExpSquaredKernel(scale))
        self.gp.compute(lc.time, lc.ferr)

    def lnlike(self, model):
        return self.gp.lnlikelihood(self.flux - model(self.time),
                                    quiet=True)


class GPLikelihood(Pipeline):

    def get_result(self, **kwargs):
        result = self.parent.query(**kwargs)
        result["data"] = map(GPLCWrapper, result.pop("data"))
        return result
