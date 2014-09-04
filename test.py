import turnstile
import numpy as np
import matplotlib.pyplot as pl


pipe = turnstile.Download()
# q = dict(kicid=5709725)

# q = dict(kicid=10593626)
# period, t0 = 289.862, 133.70

# q = dict(kicid=11415243)
# period, t0 = 168.814, 211.45

q = dict(kicid=8644545)
period, t0 = 295.963, 138.91

# q = dict(kicid=3542566)
# period, t0 = 325.03, 156.06

# q = dict(kicid=2860283)

pipe = turnstile.Inject(pipe)
q["injections"] = [dict(radius=0.015, period=25., t0=12.)]

pipe = turnstile.Prepare(pipe)
pipe = turnstile.GPLikelihood(pipe)

pipe = turnstile.OneDSearch(pipe)
q["durations"] = [0.2, 0.4]

pipe = turnstile.TwoDSearch(pipe)
q["min_period"] = 100
q["max_period"] = 400
q["alpha"] = np.log(60000)-np.log(2*np.pi)

pipe = turnstile.PeakDetect(pipe)
pipe = turnstile.FeatureExtract(pipe, cache=False)

response = pipe.query(**q)

# pl.plot(response.periods, response.scaled_phic_same, "k")
# for peak in response.peaks:
#     pl.plot(peak["period"], peak["scaled_phic_same"], ".r")
# pl.gca().axhline(5 * response.rms, color="r", lw=3, alpha=0.3)

# pl.xlabel("period")
# pl.ylabel("scaled PHIC")
# pl.savefig("periodogram-kic-{0}.png".format(q["kicid"]))

# import json
# print(json.dumps(response.peaks, sort_keys=True, indent=4,
#                  separators=(',', ': ')))


# assert 0


# def time_warp(t):
#     return (t - t0 + 0.5 * period) % period - 0.5 * period


# def model(t):
#     t = time_warp(t)
#     r = np.zeros_like(t)
#     r[np.fabs(t) < 0.5 * duration] = -depth
#     return r


# fig1 = pl.figure()
# ax1 = fig1.add_subplot(111)
# fig2 = pl.figure()
# ax2 = fig2.add_subplot(111)

# count = 0
# offset = 0.001
# rng = (-3, 3)
# for lc in response.model_light_curves:
#     t = time_warp(lc.time)
#     m = (t < rng[1]) * (t > rng[0])
#     if not np.any(m):
#         continue
#     m = np.ones_like(t, dtype=bool)
#     ind = np.argsort(t[m])
#     t = t[m][ind]
#     y = lc.flux[m][ind]
#     mu = count*offset - np.median(y)
#     ax1.plot(t, y + mu, ".k", ms=2)

#     mean, bkg = lc.predict(y=lc.flux - model(lc.time))
#     ax1.plot(t, bkg[m][ind] + mu, "r")
#     ax1.plot(t, (model(lc.time) + bkg)[m][ind] + mu, ":r")

#     ax2.plot(t, y - bkg[m][ind], ".", ms=3)

#     count += 1

# ax1.set_xlim(rng)
# ax2.set_xlim(rng)
# ax2.axhline(-depth, color="r", alpha=0.2, lw=3)
# # ax1.set_ylim(0, 0.015)
# fig1.savefig("data.png")
# fig2.savefig("data2.png")
