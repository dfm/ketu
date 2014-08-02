#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as pl

import turnstile

r = 0.02
periods = np.linspace(100, 120, 1000)

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
pl.pcolor(X, Y, zimg, cmap="gray", vmin=-5000, vmax=3000)
pl.gca().axvline(results["injection"].bodies[0].period, color="r", alpha=0.1,
                 lw=3)
pl.gca().axhline(results["injection"].bodies[0].t0, color="r", alpha=0.1,
                 lw=3)
pl.colorbar()
pl.xlabel("period")
pl.ylabel("offset")
pl.savefig("results/grid-{0}.png".format(suffix))

pl.clf()
z[np.isnan(z)] = -np.inf
pl.plot(x, np.max(z, axis=1), "k")
pl.gca().axvline(results["injection"].bodies[0].period, color="r", alpha=0.1,
                 lw=3)
pl.ylim(-500, 3000)
pl.xlabel("period")
pl.savefig("results/periodogram-{0}.png".format(suffix))
