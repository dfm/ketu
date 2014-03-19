#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import kplr
import transit
import numpy as np
import cPickle as pickle
# from scipy.spatial import cKDTree
import matplotlib.pyplot as pl

import george
from george.kernels import RBFKernel

from turnstile.data import LightCurve


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
# kic = client.star(12253474)
kic = client.star(12506954)
print(kic.kois)
print("Kepmag = {0}".format(kic.kic_kepmag))
lcs = kic.get_light_curves(short_cadence=False)
data = lcs[5].read()
datasets = LightCurve(data["TIME"], data["SAP_FLUX"], data["SAP_FLUX_ERR"],
                      data["SAP_QUALITY"] == 0).autosplit(0.5)
lc = datasets[0]
lc.time -= np.min(lc.time)
ror = 0.02
truth = transit.ldlc_simple(lc.time, 0.4, 0.2, 40.0, 15., 0.4, ror,
                            0.5, kplr.EXPOSURE_TIMES[1]/86400, 0.1, 3)
lc.flux *= truth

# Plot the "raw" data.
print("Plotting raw data")
detrend = median_detrend(lc.time, lc.flux)
pl.clf()
pl.plot(lc.time, lc.flux, ".k", alpha=0.4)
pl.savefig("raw.png")
pl.savefig("raw.pdf")

# Median filter the data.
print("Median filtering")
detrend = median_detrend(lc.time, lc.flux, dt=2.5)
print("Plotting 'detrended' data")
pl.clf()
pl.plot(lc.time, detrend, ".k", alpha=0.4)
pl.savefig("detrended.png")
pl.savefig("detrended.pdf")

# Optimize the hyperparameters.
print("Optimizing hyperparameters")
a, s = lc.optimize_hyperparams(N=3)
print("Finished. Found a={0}, s={1}".format(a, s))

truth = (a, s, 0.4, 0.2, 40.0, 15., 0.4, ror, 0.5)
pickle.dump((lc.time[100:], lc.flux[100:], lc.ferr[100:], detrend[100:],
             truth), open("demo.pkl", "wb"), -1)

# Set up final GP.
gp = george.GaussianProcess(a * RBFKernel(s), tol=1e-16, nleaf=20)
gp.compute(lc.time, lc.ferr)

times = np.linspace(2.5, 26, 1500)
gp_lls = np.empty_like(times)
mf_lls = np.empty_like(times)
print("Running {0} hypothesis tests".format(len(times)))
for i, t0 in enumerate(times):
    mod = box_model(lc.time, t0, 0.3, ror**2)
    # mod = box_model(lc.time, t0, 0.2, 0.02**2)
    gp_lls[i] = gp.lnlikelihood(lc.flux - mod)
    mf_lls[i] = -0.5 * np.sum(((detrend-mod)/lc.ferr)**2)
print("Finished")

gp_null = gp.lnlikelihood(lc.flux - 1)
mf_null = -0.5 * np.sum(((detrend-1)/lc.ferr)**2)

# tree = cKDTree(np.atleast_2d(lc.time).T)
# print(tree.query(np.array([[15.0], [40.0], ]), distance_upper_bound=0.1))

pl.clf()
pl.plot(times, mf_lls - mf_null, "k", lw=1.5)
pl.plot(times, gp_lls - gp_null, "r")
pl.gca().axhline(0.0, color="k")
pl.savefig("demo.png")
pl.savefig("demo.pdf")
