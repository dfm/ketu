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
from IPython.parallel import Client

from turnstile.characterization import prepare_characterization

kois = pd.read_hdf("kois.h5", "kois")

print(set(kois.kepid))

for kepid in set(kois.kepid):
    m = kois.kepid == kepid
    m &= np.isfinite(kois.koi_period)
    m &= np.isfinite(kois.koi_ror)
    koi = kois[m]
    if not len(koi):
        continue
    print(kepid)
    model = prepare_characterization(
        kepid,
        np.array(koi.koi_period),
        np.array(koi.koi_time0bk % koi.koi_period),
        np.array(koi.koi_ror),
        np.array(koi.koi_impact),
        es=np.array(koi.koi_eccen) + 1e-4)

    OUTDIR = "characterization/{0}".format(kepid)
    if os.path.exists(OUTDIR):
        print("skipping")
        continue
    try:
        os.makedirs(OUTDIR)
    except os.error:
        pass

    p0 = model.pack()
    model.unpack(p0)
    assert np.allclose(p0, model.pack())

    print(model.lnprob(p0))

    fig = model.plot(dy=1e-3)
    fig.savefig(os.path.join(OUTDIR, "plot.png"))

    def lnprob(p):
        return model.lnprob(p)

    # Set up the IPython client.
    c = Client()
    view = c[:]
    view.push({"lnprob": lnprob, "model": model})

    ndim, nwalkers = len(p0), 36
    pos = [p0 + 1e-5 * np.random.randn(ndim) for w in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=view)

    print(list(view.map(lnprob, pos)))

    pos, lp, _ = sampler.run_mcmc(pos, 1000)

    p0 = pos[np.argmax(lp)]
    model.unpack(p0)
    model.fit_star = True

    p0 = model.pack()
    print(len(p0))
    model.unpack(p0)
    assert np.allclose(p0, model.pack())

    ndim, nwalkers = len(p0), 36
    pos = [p0 + 1e-5 * np.random.randn(ndim) for w in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=view)

    view.push({"lnprob": lnprob, "model": model})
    print(list(view.map(lnprob, pos)))
    pos, lp, _ = sampler.run_mcmc(pos, 1000)

    p0 = pos[np.argmax(lp)]
    pos = [p0 + 1e-5 * np.random.randn(ndim) for w in range(nwalkers)]
    sampler.reset()

    print("Final burn-in")
    pos, lp, _ = sampler.run_mcmc(pos, 5000)
    sampler.reset()

    p = pos[np.argmax(lp)]
    model.unpack(p)
    fig = model.plot(dy=1e-3)
    fig.savefig(os.path.join(OUTDIR, "plot-mid.png"))

    print("Production run")
    pos, lp, _ = sampler.run_mcmc(pos, 20000)

    with h5py.File(os.path.join(OUTDIR, "results.h5"), "w") as f:
        f.create_dataset("chain", data=sampler.chain)
        f.create_dataset("lnp", data=sampler.lnprobability)

    with open(os.path.join(OUTDIR, "model.pkl"), "wb") as f:
        pickle.dump(model, f, -1)

    p = pos[np.argmax(lp)]
    model.unpack(p)
    fig = model.plot(dy=1e-3)
    fig.savefig(os.path.join(OUTDIR, "plot-final.png"))
