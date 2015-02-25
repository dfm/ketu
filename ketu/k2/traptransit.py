# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["fit_traptransit"]

import emcee
import numpy as np
from .._traptransit import periodic_traptransit

try:
    import matplotlib.pyplot as pl
except ImportError:
    pl = None


def fit_traptransit(lc, periods, t0s, depths):
    if pl is None:
        raise ImportError("matplotlib")

    # The probabilistic model for emcee.
    class TrapWalker(emcee.BaseWalker):
        def __init__(self, nplanets):
            self.nplanets = nplanets
        def lnpriorfn(self, p):
            for j in range(self.nplanets):
                i = 5 * j
                if 0 >= p[i]:  # Full duration
                    return -np.inf
                if 0 >= p[i+1]:  # Depth
                    return -np.inf
                if not (2 < p[i+2] < 30):  # Duration / ingress
                    return -np.inf
                if 0 >= p[i+3]:  # Period
                    return -np.inf
                if p[i+4] > p[i+3]:  # Phase
                    return -np.inf
            return 0.0
        def lnlikefn(self, p):
            pred = np.zeros(len(lc.time))
            for j in range(self.nplanets):
                i = 5 * j
                pred += (periodic_traptransit(lc.time, p[i:i+5]) - 1) * 1e3
            r = lc.flux - pred
            r -= lc.predict(r)
            return -0.5 * np.sum(r ** 2) * lc.ivar

    # Initialize the walkers.
    nplanets = len(periods)
    p0 = np.array([v for i in range(nplanets)
                   for v in (0.1, depths[i], 5., periods[i], t0s[i])])
    ndim = len(p0)
    nwalkers = 4 * ndim
    coords = p0 + 1e-4 * np.random.randn(nwalkers, ndim)

    # Sample.
    ensemble = emcee.Ensemble(TrapWalker(nplanets), coords)
    assert np.all(np.isfinite(ensemble.lnprob))
    sampler = emcee.Sampler()
    sampler.run(ensemble, 2000)

    # Find the best sample.
    samps = sampler.get_coords(flat=True)
    lp = sampler.get_lnprob(flat=True)
    p0 = samps[np.argmax(lp)]
    coords = p0 + 1e-4 * np.random.randn(nwalkers, ndim)
    ensemble = emcee.Ensemble(TrapWalker(nplanets), coords)

    # Run production chain.
    sampler.reset()
    sampler.run(ensemble, 2000)

    # Plot the best fit model.
    samps = sampler.get_coords(flat=True)
    lp = sampler.get_lnprob(flat=True)
    p0 = samps[np.argmax(lp)]

    pred = (periodic_traptransit(lc.time, p0) - 1) * 1e3
    r = lc.flux - pred
    bkg = lc.predict(r)
    t = (lc.time - p0[4] + 0.5 * p0[3]) % (p0[3]) - 0.5 * p0[3]
    fig, ax = pl.subplots()
    ax.plot(t, lc.flux - bkg, ".k")
    t = np.linspace(-1.5, 1.5, 5000)
    ax.plot(t, (periodic_traptransit(t + p0[4], p0) - 1) * 1e3, "g")
    ax.set_xlim(-1.5, 1.5)

    return sampler.get_coords(discard=500, flat=True), fig
