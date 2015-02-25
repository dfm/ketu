# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["fit_traptransit"]

import emcee
import numpy as np
from .._traptransit import periodic_traptransit


def fit_traptransit(lc, period, t0, depth):
    # The probabilistic model for emcee.
    class TrapWalker(emcee.BaseWalker):
        def lnpriorfn(self, p):
            return 0.0
        def lnlikefn(self, p):
            pred = (periodic_traptransit(lc.time, p) - 1) * 1e3
            r = lc.flux - pred
            r -= lc.predict(r)
            return -0.5 * np.sum(r ** 2) * lc.ivar

    # Initialize the walkers.
    p0 = np.array([0.1, depth, 5., period, t0])
    nwalkers, ndim = 16, len(p0)
    coords = p0 + 1e-4 * np.random.randn(nwalkers, ndim)

    # Sample.
    ensemble = emcee.Ensemble(TrapWalker(), coords)
    assert np.all(np.isfinite(ensemble.lnprob))
    sampler = emcee.Sampler()
    sampler.run(ensemble, 2000)

    return sampler.get_coords(discard=500, flat=True)
