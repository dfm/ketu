#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import emcee
import numpy as np
import pandas as pd
from turnstile.characterization import prepare_characterization

kois = pd.read_hdf("kois.h5", "kois")
kepid = 8559644
kois = kois[kois.kepid == kepid]
model = prepare_characterization(
    kepid,
    np.array(kois.koi_period),
    np.array(kois.koi_time0bk % kois.koi_period),
    np.array(kois.koi_ror),
    np.array(kois.koi_impact),
    es=np.array(kois.koi_eccen) + 1e-4)

p0 = model.pack()
model.unpack(p0)
assert np.allclose(p0, model.pack())

print(model.lnprob(p0))

fig = model.plot()
fig.savefig("plot.png")


def lnprob(p):
    return model.lnprob(p)


ndim, nwalkers = len(p0), 36
pos = [p0 + 1e-5 * np.random.randn(ndim) for w in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
