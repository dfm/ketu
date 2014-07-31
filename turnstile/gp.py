# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["GPModel", "GPLightCurve"]

import numpy as np
from emcee.autocorr import integrated_time

import george
from george.kernels import ExpSquaredKernel

from .pipeline import Pipeline


class GPLightCurve(object):

    def __init__(self, lc):
        self.lc = lc

        # Estimate the hyperparameters.
        var = float(np.var(lc.flux))
        scale = np.median(np.diff(lc.time)) * integrated_time(lc.flux)

        self.kernel = var * ExpSquaredKernel(var)
        self.gp =


class GPModel(Pipeline):

    defaults = {
    }

    def get_result(self, **kwargs):
        pass
