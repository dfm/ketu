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
# q["injections"] = [dict(radius=0.02, period=period, t0=t0)]

pipe = turnstile.Prepare(pipe)
pipe = turnstile.GPLikelihood(pipe)

pipe = turnstile.OneDSearch(pipe)
q["durations"] = 0.2

pipe = turnstile.TwoDSearch(pipe, cache=False)
q["min_period"] = 100
q["max_period"] = 400
q["alpha"] = np.log(60000)-np.log(2*np.pi)

response = pipe.query(**q)

z1 = response.phic_same[:, :, 0]
z1[np.isnan(z1)] = -np.inf
z2 = response.phic_variable[:, :, 0]
z2[np.isnan(z2)] = -np.inf
depth = response.depth_2d[:, :, 0]
depth[np.isnan(depth)] = 0.0

z1[z2 > z1] = -np.inf
z1[depth <= 0.0] = -np.inf

periods = response.period_2d
y = response.t0_2d
for i in np.argsort(z1.flatten())[-10:]:
    xi, yi = np.unravel_index(i, z1.shape)
    period, t0 = periods[xi], y[yi]
    print(period, t0, depth[xi, yi], response.depth_ivar_2d[xi, yi, 0])

i = (np.arange(len(periods)), np.argmax(z1, axis=1))
pl.plot(periods, z1[i], "k")
# pl.plot(periods, z2[i], "r")
pl.gca().axvline(period, color="r", lw=3, alpha=0.3)
pl.xlabel("period")
pl.ylabel("PHIC")
pl.savefig("periodogram-kic-{0}.png".format(q["kicid"]))

times_1d = np.arange(response.min_time_1d, response.max_time_1d,
                     response.time_spacing)
pl.clf()
pl.plot(times_1d, response.dll_1d, "k")
# t = t0 + (period) * np.arange(9)
# i = np.round((t - response.min_time_1d) / response.time_spacing).astype(int)
# l = response.dll_1d[i] + 0.5 * np.log(response.depth_ivar_1d[i])
# print(np.sum(l))

for i in range(10):
    t = t0 + i * period
    pl.gca().axvline(t, color="r", lw=3, alpha=0.3)
# pl.ylim(-10, 100)
pl.xlim(times_1d.min(), times_1d.max())
pl.savefig("dude.png")

# print(response.gp_light_curves[0])


def time_warp(t):
    return (t - t0 + 0.5 * period) % period - 0.5 * period


def model(t):
    t = time_warp(t)
    r = np.zeros_like(t)
    r[np.fabs(t) < 0.5 * 0.2] = -1
    return r


pl.clf()

count = 0
offset = 1000
rng = (-3, 3)
for lc in response.model_light_curves:
    t = time_warp(lc.time)
    m = (t < rng[1]) * (t > rng[0])
    if not np.any(m):
        continue
    ind = np.argsort(t[m])
    t = t[m][ind]
    y = lc.flux[m][ind]
    mu = count*offset - np.median(y)
    pl.plot(t, y + mu, ".k", ms=2)
    pl.plot(t, lc.predict(model)[m][ind] + mu, "r")
    count += 1

pl.xlim(rng)
# pl.ylim(-1000, 1000)
pl.savefig("data.png")
