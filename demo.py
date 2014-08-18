#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as pl

import turnstile

r = 0.015
pmin, pmax = 100, 400.
periods = np.exp(np.arange(np.log(pmin), np.log(pmax), 0.3*0.3/(4.1*365.)))
print("Testing {0} periods".format(len(periods)))
durations = np.array([0.3])  # np.arange(0.3, 0.7, 0.2)
print("Testing {0} durations".format(len(durations)))
depths = np.array([1.0])  # np.arange(0.01, 0.04, 0.02) ** 2
print("Testing {0} depths".format(len(durations)))

q1 = dict(
    # kicid=11415243,
    kicid=3542566,
    # kicid=12253474,
    injections=[dict(period=114.123, t0=12.5, radius=r)],
)
q2 = dict(
    durations=durations.tolist(), depths=depths.tolist(),
    periods=periods.tolist(), dt=0.3 * 0.3,
)

pipe = turnstile.Download()
pipe = turnstile.Inject(pipe)
pipe = turnstile.Prepare(pipe)

if False:
    suffix = "basic"
    pipe = turnstile.Detrend(pipe)
    pipe0 = turnstile.BasicLikelihood(pipe)
    q1["detrend_window"] = 2
else:
    suffix = "gp"
    pipe0 = turnstile.GPLikelihood(pipe)

pipe = turnstile.Hypotheses(pipe0)
pipe = turnstile.Search(pipe, cache=False)

q = dict(q1, **q2)
results = pipe.query(**q)

# # Plot the depths.
# times = results["times"]
# depths = results["depths"]
# d_ivars = results["d_ivars"]
# print(np.sqrt(np.median(depths ** 2 * d_ivars)))
# pl.plot(times, depths * np.sqrt(d_ivars), "k")
# # pl.gca().axhline(1e6 * r ** 2, color="r", lw=3, alpha=0.3)
# s = results["injection"].bodies[0]
# for t in s.t0 + np.arange(100) * s.period:
#     pl.gca().axvline(t, color="r", lw=1, alpha=0.3)
# pl.xlim(min(times), max(times))

# pl.savefig("results/depths.png")
# assert 0

z = results["bic1"][:, :, 0]  # - results["bic2"][:, :, 0]
zimg = np.array(z)
zimg[~np.isfinite(zimg)] = 0.0
x = results["periods"]
y = np.arange(0, z.shape[1] * results["dt"], results["dt"])
X, Y = np.meshgrid(x, y, indexing="ij")

# true_period = 168.814
true_period = results["injection"].bodies[0].period
true_t0 = (results["injection"].bodies[0].t0
           - results["mean_time"]) % true_period

pl.figure(figsize=(8, 8))
pl.pcolormesh(X, Y, zimg, cmap="gray")
pl.gca().axvline(true_period, color="r", alpha=0.1, lw=3)
pl.gca().axhline(true_t0, color="r", alpha=0.1, lw=3)
pl.colorbar()
pl.xlim(np.min(x), np.max(x))
pl.ylim(np.min(y), np.max(y))
pl.xlabel("period")
pl.ylabel("offset")
pl.savefig("results/grid-{0}.png".format(suffix), dpi=300)

pl.xlim(true_period - 10, true_period + 10)
pl.ylim(true_t0 - 10, true_t0 + 10)
pl.savefig("results/grid-{0}-zoom.png".format(suffix), dpi=300)

z2 = results["bic2"][:, :, 0]
z2[~np.isfinite(z2)] = -np.inf
z[~np.isfinite(z)] = -np.inf
xi, yi = np.unravel_index(np.argmax(z), z.shape)
period, t0 = x[xi], y[yi]
print(xi, yi, x[xi], y[yi], z[xi, yi])

# # Plot the data.
# lcs = pipe0.query(**q1)["data"]
# fig, axes = pl.subplots(4, 4, figsize=(10, 10))
# t = (t0 + results["mean_time"]) % period
# i = 0
# for lc in lcs:
#     while t < lc.time.min():
#         t += period
#     m = np.abs(lc.time - t) < 3
#     if np.any(m):
#         axes.flat[i].plot(lc.time[m], lc.flux[m], ".k")
#         axes.flat[i].set_xlim(t - 3, t + 3)
#         axes.flat[i].set_xticklabels([])
#         axes.flat[i].set_yticklabels([])
#         i += 1
#         if i >= len(axes.flat):
#             break
# fig.savefig("results/transits-{0}.png".format(suffix))

pl.figure()
i = (np.arange(len(x)), np.argmax(z, axis=1))
m = z[i] + 1 > z2[i]
x = np.array(x)
pl.plot(x, z[i], "k")
pl.plot(x, z2[i], "b", alpha=0.3)
pl.plot(x[m], z[i][m], ".r")
pl.gca().axvline(true_period, color="r", alpha=0.1, lw=3)
# pl.ylim(-500, 3000)
pl.xlabel("period")
pl.savefig("results/periodogram-{0}.png".format(suffix))

# pl.plot(x, np.min(z, axis=1), ".k")
pl.xlim(true_period - 0.5, true_period + 0.5)
pl.savefig("results/periodogram-{0}-zoom.png".format(suffix))
