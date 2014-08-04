#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as pl

import turnstile

r = 0.02
dfreq = 0.005 / (4.25 * 365.)
freq = np.arange(1.0 / 150., 1.0 / 50., dfreq)
periods = 1.0 / freq
print("Testing {0} periods".format(len(periods)))

q = dict(
    kicid=12253474,
    injections=[dict(period=114.0, t0=22., radius=r)],
    durations=0.35, depths=r**2,
    periods=periods.tolist(), dt=0.1,
)

pipe = turnstile.Download()
pipe = turnstile.Inject(pipe)
pipe = turnstile.Prepare(pipe)

if False:
    suffix = "basic"
    pipe = turnstile.Detrend(pipe)
    pipe = turnstile.BasicLikelihood(pipe)
else:
    suffix = "gp"
    pipe = turnstile.GPLikelihood(pipe)

pipe = turnstile.Hypotheses(pipe)
pipe = turnstile.Search(pipe)

results = pipe.query(**q)

z = results["grid"][:, :, 0, 0]
zimg = np.array(z)
zimg[np.isnan(zimg)] = 0.0
x = results["periods"]
y = np.arange(0, z.shape[1] * results["dt"], results["dt"])
X, Y = np.meshgrid(x, y, indexing="ij")

pl.figure(figsize=(8, 8))
pl.pcolormesh(X, Y, zimg, cmap="gray", vmin=-5000, vmax=3000)
true_period = results["injection"].bodies[0].period
true_t0 = (results["injection"].bodies[0].t0
           - results["mean_time"]) % true_period
pl.gca().axvline(true_period, color="r", alpha=0.1, lw=3)
pl.gca().axhline(true_t0, color="r", alpha=0.1, lw=3)
pl.colorbar()
pl.xlim(np.min(x), np.max(x))
pl.ylim(np.min(y), np.max(y))
pl.xlabel("period")
pl.ylabel("offset")
pl.savefig("results/grid-{0}.png".format(suffix), dpi=300)

pl.xlim(true_period - 20, true_period + 20)
pl.ylim(true_t0 - 20, true_t0 + 20)
pl.savefig("results/grid-{0}-zoom.png".format(suffix), dpi=300)

pl.clf()
z[np.isnan(z)] = -np.inf
pl.plot(x, np.max(z, axis=1), "k")
pl.gca().axvline(true_period, color="r", alpha=0.1, lw=3)
pl.ylim(-500, 3000)
pl.xlabel("period")
pl.savefig("results/periodogram-{0}.png".format(suffix))
