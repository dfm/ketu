#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import os
import h5py
import emcee
import numpy as np
import pandas as pd
import cPickle as pickle
import matplotlib.pyplot as pl
from IPython.parallel import Client

from turnstile.characterization.k2 import prepare_characterization

candidates = pd.read_csv("k2results/candidates/candidates_for_ben.csv")
names = list(set(candidates.kicid))
inds = np.arange(len(names))
np.random.shuffle(inds)

for ind in inds:
    nm = names[ind]
    epicid = nm.split()[1]
    fn = "../k2/lightcurves/c1/{0}00000/{1}000/ktwo{2}-c01_lpd-lc.fits" \
        .format(epicid[:4], epicid[4:6], epicid)
    basis = "../k2/lightcurves/c1-basis.h5"

    m = candidates.kicid == nm
    periods = np.array(candidates[m].period)
    t0s = np.array(candidates[m].t0) - 1975.0
    depths = np.array(candidates[m].depth)
    bs = 0.1 + np.zeros_like(depths)
    print(periods)

    model = prepare_characterization(fn, basis, periods, t0s,
                                     np.sqrt(depths * 1e-3), bs)
    print(np.sqrt(depths * 1e-3))
    print(model.pack())
    print(model.lnprior())
    print(model.lnlike())

    OUTDIR = "k2_characterization/{0}".format(epicid)
    if os.path.exists(os.path.join(OUTDIR, "results.h5")):
        print("skipping")
        continue
    try:
        os.makedirs(OUTDIR)
    except os.error:
        pass

    # Plot the initial fit.
    fig = model.plot()
    fig.savefig(os.path.join(OUTDIR, "plot.png"))
    pl.close(fig)

    # Check the packing and unpacking.
    p0 = model.pack()
    model.unpack(p0)
    assert np.allclose(p0, model.pack())

    def lnprob(p):
        return model.lnprob(p)

    # Set up the IPython client.
    c = Client()
    view = c[:]
    view.push({"lnprob": lnprob, "model": model})

    ndim, nwalkers = len(p0), 64
    pos = [p0 + 1e-8 * np.random.randn(ndim) for w in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=view)
    print(list(view.map(lnprob, pos)))

    print("Burning in")
    pos, lp, _ = sampler.run_mcmc(pos, 2000)
    # p0 = pos[np.argmax(lp)]
    # pos = [p0 + 1e-8 * np.random.randn(ndim) for w in range(nwalkers)]
    # pos, lp, _ = sampler.run_mcmc(pos, 2000)
    # p0 = pos[np.argmax(lp)]
    # pos = [p0 + 1e-8 * np.random.randn(ndim) for w in range(nwalkers)]
    # pos, lp, _ = sampler.run_mcmc(pos, 2000)
    # pl.clf()
    # pl.plot(sampler.lnprobability.T)
    # pl.savefig(os.path.join(OUTDIR, "lp.png"))
    # pl.close()

    # sampler.reset()
    # p0 = pos[np.argmax(lp)]
    # pos = [p0 + 1e-8 * np.random.randn(ndim) for w in range(nwalkers)]
    # pos, lp, _ = sampler.run_mcmc(pos, 2000)

    pl.clf()
    pl.plot(sampler.lnprobability.T)
    pl.savefig(os.path.join(OUTDIR, "lp-mid.png"))
    pl.close()

    p = pos[np.argmax(lp)]
    model.unpack(p)
    fig = model.plot()
    fig.savefig(os.path.join(OUTDIR, "plot-mid.png"))
    pl.close(fig)

    print(list(view.map(lnprob, pos)))

    print("Production run")
    sampler.reset()
    pos, lp, _ = sampler.run_mcmc(pos, 10000)

    pl.clf()
    pl.plot(sampler.lnprobability.T)
    pl.savefig(os.path.join(OUTDIR, "lp-final.png"))
    pl.close()

    with h5py.File(os.path.join(OUTDIR, "results.h5"), "w") as f:
        f.create_dataset("chain", data=sampler.chain)
        f.create_dataset("lnp", data=sampler.lnprobability)

    with open(os.path.join(OUTDIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f, -1)

    p = pos[np.argmax(lp)]
    model.unpack(p)
    fig = model.plot()
    fig.savefig(os.path.join(OUTDIR, "plot-final.png"))
    pl.close(fig)
