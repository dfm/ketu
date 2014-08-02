#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as pl

import turnstile

q = dict(
    kicid=12253474, durations=0.5, depths=0.03**2,
    periods=np.linspace(100, 120, 500).tolist(), dt=0.3,
    injections=[dict(period=114.0, t0=10., radius=0.03)],
)

pipe = turnstile.Download()
pipe = turnstile.Inject(pipe)
pipe = turnstile.Prepare(pipe)

if False:
    pipe = turnstile.Detrend(pipe)
    pipe = turnstile.BasicLikelihood(pipe)
else:
    pipe = turnstile.GPLikelihood(pipe)

pipe = turnstile.Hypotheses(pipe)
pipe = turnstile.Search(pipe)

results = pipe.query(**q)
z = results["grid"][:, :, 0, 0]
x = results["periods"]
y = np.arange(0, z.shape[1] * results["dt"], results["dt"])
X, Y = np.meshgrid(x, y, indexing="ij")
xi, yi = np.unravel_index(np.argmax(z), z.shape)
print(x[xi], y[yi])

pl.figure(figsize=(8, 8))
pl.pcolor(X, Y, -z, cmap="gray")
pl.gca().axvline(results["injection"].bodies[0].period, color="r", alpha=0.1,
                 lw=3)
pl.gca().axhline(results["injection"].bodies[0].t0, color="r", alpha=0.1,
                 lw=3)
pl.xlabel("period")
pl.ylabel("offset")
pl.savefig("grid.png")
