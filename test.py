import turnstile
import numpy as np
import matplotlib.pyplot as pl

pipe = turnstile.Download()
# q = dict(kicid=11415243)
# period, t0 = 168.814, 211.45

q = dict(kicid=8644545)
period, t0 = 295.963, 138.91

# q = dict(kicid=3542566)
# period, t0 = 325.03, 156.06

# pipe = turnstile.Inject(pipe)
# q["injections"] = [dict(radius=0.03, period=365., t0=15.)]

pipe = turnstile.Prepare(pipe)
pipe = turnstile.GPLikelihood(pipe)

pipe = turnstile.OneDSearch(pipe)
q["durations"] = [0.2]

pipe = turnstile.TwoDSearch(pipe)
q["min_period"] = 100
q["max_period"] = 400
q["alpha"] = np.log(60000)-np.log(2*np.pi)

response = pipe.query(**q)

# Reject some points.
z1 = response.phic_same
z2 = response.phic_variable
depths = response.depth_2d
# depths[np.isnan(depths)] = 0.0
# z1[z2 > z1] = -np.inf
# z1[depths <= 0.0] = -np.inf

# Find the profile over duration.
DURATION = 0
# dur_inds = np.argmax(z1, axis=2)
z1 = z1[:, DURATION]
z2 = z2[:, DURATION]
depths = depths[:, DURATION]

periods = response.period_2d
t0s = response.t0_2d[:, DURATION]
i = np.argmax(z1)
period, t0 = periods[i], t0s[i]
duration = response.durations[DURATION]
depth = depths[i]
print(period, t0, duration, depth)

pl.plot(periods, z1, "k")
pl.plot(periods, z2, "r")
pl.gca().axvline(period, color="r", lw=3, alpha=0.3)
pl.xlabel("period")
pl.ylabel("PHIC")
pl.ylim(-100, 100)
pl.savefig("periodogram-kic-{0}.png".format(q["kicid"]))

# times_1d = np.arange(response.min_time_1d, response.max_time_1d,
#                      response.time_spacing)
# pl.clf()
# pl.plot(times_1d, response.dll_1d[:, DURATION], "k")
# # t = t0 + (period) * np.arange(9)
# # i = np.round((t - response.min_time_1d) / response.time_spacing).astype(int)
# # l = response.dll_1d[i] + 0.5 * np.log(response.depth_ivar_1d[i])
# # print(np.sum(l))

# for i in range(10):
#     t = t0 + i * period
#     pl.gca().axvline(t, color="r", lw=3, alpha=0.3)
# # pl.ylim(-10, 100)
# pl.xlim(times_1d.min(), times_1d.max())
# pl.savefig("dude.png")

# print(response.gp_light_curves[0])


def time_warp(t):
    return (t - t0 + 0.5 * period) % period - 0.5 * period


def model(t):
    t = time_warp(t)
    r = np.zeros_like(t)
    r[np.fabs(t) < 0.5 * duration] = -depth
    return r


fig1 = pl.figure()
ax1 = fig1.add_subplot(111)
fig2 = pl.figure()
ax2 = fig2.add_subplot(111)

count = 0
offset = 1000
rng = (-3, 3)
for lc in response.model_light_curves:
    t = time_warp(lc.time)
    m = (t < rng[1]) * (t > rng[0])
    if not np.any(m):
        continue
    m = np.ones_like(t, dtype=bool)
    ind = np.argsort(t[m])
    t = t[m][ind]
    y = lc.flux[m][ind]
    mu = count*offset - np.median(y)
    ax1.plot(t, y + mu, ".k", ms=2)

    mean, bkg = lc.predict(y=lc.flux - model(lc.time))
    ax1.plot(t, bkg[m][ind] + mu, "r")
    ax1.plot(t, (model(lc.time) + bkg)[m][ind] + mu, ":r")

    ax2.plot(t, y - bkg[m][ind], ".k", ms=2)

    count += 1

# ax1.set_xlim(rng)
# ax2.set_xlim(rng)
ax2.axhline(-depth, color="r", alpha=0.2, lw=3)
# pl.ylim(-1000, 1000)
fig1.savefig("data.png")
fig2.savefig("data2.png")
