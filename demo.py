#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import kplr
import copy
import transit
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl

import george
from george.kernels import RBFKernel


class LightCurve(object):

    def __init__(self, time, flux, ferr, quality=None):
        # Mask bad data.
        m = np.isfinite(time) * np.isfinite(flux) * np.isfinite(ferr)
        if quality is not None:
            m *= quality
        self.time = np.atleast_1d(time)[m]
        self.flux = np.atleast_1d(flux)[m]
        self.ferr = np.atleast_1d(ferr)[m]

        # Normalize by the median.
        mu = np.median(self.flux)
        self.flux /= mu
        self.ferr /= mu

    def split(self, ts, normalize=True):
        ts = np.concatenate([[-np.inf], np.sort(np.atleast_1d(ts)), [np.inf]])
        datasets = []
        for i, t0 in enumerate(ts[:-1]):
            m = (np.isfinite(self.time) * (self.time >= t0)
                 * (self.time < ts[i + 1]))
            if np.any(m):
                ds = copy.deepcopy(self)
                ds.time = ds.time[m]
                ds.flux = ds.flux[m]
                ds.ferr = ds.ferr[m]
                if normalize:
                    mu = np.median(ds.flux)
                    ds.flux /= mu
                    ds.ferr /= mu
                datasets.append(ds)

        return datasets

    def autosplit(self, ttol, max_length=None):
        dt = self.time[1:] - self.time[:-1]
        m = dt > ttol
        ts = 0.5 * (self.time[1:][m] + self.time[:-1][m])
        datasets = self.split(ts)
        if max_length is not None:
            while any([len(d.time) > max_length for d in datasets]):
                datasets = [[d] if len(d.time) <= max_length
                            else d.split([d.time[int(0.5*len(d.time))]])
                            for d in datasets]
                datasets = [d for ds in datasets for d in ds]

        return datasets

    def optimize_hyperparams(self, p0, N=3):
        ts = np.linspace(self.time.min(), self.time.max(), N+2)
        lcs = self.split(ts[1:-1])
        return np.median(map(lambda l: l._op(p0), lcs), axis=0)

    def _op(self, p0):
        results = op.minimize(nll, np.log(p0), method="L-BFGS-B",
                              args=(self.time, self.flux, self.ferr))
        return np.exp(results.x)


def nll(p, t, f, fe):
    a, s = np.exp(p)
    gp0 = george.GaussianProcess(a * RBFKernel(s), tol=1e-16, nleaf=20)
    gp0.compute(t, fe)
    return -gp0.lnlikelihood(f-1)


def box_model(t, t0, duration, depth):
    m = np.ones_like(t)
    m[np.abs(t-t0) < 0.5*duration] -= depth
    return m


def median_detrend(x, y, dt=4.):
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    assert len(x) == len(y)
    r = np.empty(len(y))
    for i, t in enumerate(x):
        inds = np.abs(x-t) < 0.5 * dt
        r[i] = np.median(y[inds])
    return y / r


# Load the data.
client = kplr.API()
kic = client.star(2856200)
print("Kepmag = {0}".format(kic.kic_kepmag))
lcs = kic.get_light_curves(short_cadence=False)
data = lcs[5].read()
datasets = LightCurve(data["TIME"], data["SAP_FLUX"], data["SAP_FLUX_ERR"],
                      data["SAP_QUALITY"] == 0).autosplit(0.5)
lc = datasets[0]
lc.time -= np.min(lc.time)
ror = 0.1
truth = transit.ldlc_simple(lc.time, 0.4, 0.2, 40.0, 15., 0.4, ror,
                            0.0, kplr.EXPOSURE_TIMES[1]/86400, 0.1, 2)
lc.flux *= truth

# Optimize the hyperparameters.
print("Optimizing hyperparameters")
a, s = lc.optimize_hyperparams([1e-4, 2.1], N=8)
print("Finished. Found a={0}, s={1}".format(a, s))

# Plot the "raw" data.
print("Plotting raw data")
detrend = median_detrend(lc.time, lc.flux)
pl.clf()
pl.plot(lc.time, lc.flux, ".k", alpha=0.4)
pl.savefig("raw.png")

# Median filter the data.
print("Median filtering")
detrend = median_detrend(lc.time, lc.flux, dt=2.5)
print("Plotting 'detrended' data")
pl.clf()
pl.plot(lc.time, detrend, ".k", alpha=0.4)
pl.savefig("detrended.png")

# Set up final GP.
gp = george.GaussianProcess(a * RBFKernel(s), tol=1e-16, nleaf=20)
gp.compute(lc.time, lc.ferr)

times = np.linspace(4, 26, 500)
gp_lls = np.empty_like(times)
mf_lls = np.empty_like(times)
print("Running {0} hypothesis tests".format(len(times)))
for i, t0 in enumerate(times):
    mod = box_model(lc.time, t0, 0.4, ror**2)
    gp_lls[i] = gp.lnlikelihood(lc.flux - mod)
    mf_lls[i] = -0.5 * np.sum(((detrend-mod)/lc.ferr)**2)
print("Finished")

gp_null = gp.lnlikelihood(lc.flux - 1)
mf_null = -0.5 * np.sum(((detrend-1)/lc.ferr)**2)

pl.clf()
pl.plot(times, mf_lls - mf_null)
pl.plot(times, gp_lls - gp_null)
pl.gca().axhline(0.0, color="k")
pl.savefig("demo.png")
