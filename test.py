import turnstile
import numpy as np
import matplotlib.pyplot as pl

pipe = turnstile.Download()
# q = dict(kicid=11415243)
# period, t0 = 168.814, 211.45

q = dict(kicid=3542566)
period, t0 = 325.03, 156.06

pipe = turnstile.Inject(pipe)
q["injections"] = [dict(radius=0.02, period=period, t0=t0)]

pipe = turnstile.Prepare(pipe)
pipe = turnstile.GPLikelihood(pipe)

pipe = turnstile.OneDSearch(pipe)
q["durations"] = 0.2

pipe = turnstile.TwoDSearch(pipe, cache=False)
q["min_period"] = 100
q["max_period"] = 400
# q["alpha"] = np.log(100000)-np.log(2*np.pi)

response = pipe.query(**q)

periods = response.periods
z1 = response.phic_same[:, :, 0]
z1[np.isnan(z1)] = -np.inf
z2 = response.phic_variable[:, :, 0]
z2[np.isnan(z2)] = -np.inf
depth = response.depth_2d[:, :, 0]
depth[np.isnan(depth)] = 0.0

z1[z2 > z1] = -np.inf
z1[depth <= 0.0] = -np.inf

i = (np.arange(len(periods)), np.argmax(z1, axis=1))
# # m = z1[i] > z2[i]
# # print(m)
# # pl.plot(periods, z2[i], "b", alpha=0.1)
pl.plot(periods, z1[i], "k")
# # pl.plot(periods[m], z1[i][m], ".r")
pl.gca().axvline(period, color="r", lw=3, alpha=0.3)
pl.savefig("dude.png")

pl.xlim(period - 5, period + 5)
pl.savefig("dude-zoom.png")
assert 0

times_1d = np.arange(response.min_time_1d, response.max_time_1d,
                     response.time_spacing)
pl.plot(times_1d, response.dll_1d + 0.5 * np.log(response.depth_ivar_1d), "k")
t = t0 + (period) * np.arange(9)
i = np.round((t - response.min_time_1d) / response.time_spacing).astype(int)
l = response.dll_1d[i] + 0.5 * np.log(response.depth_ivar_1d[i])
print(np.sum(l))

for i in range(10):
    t = t0 + i * period
    pl.gca().axvline(t, color="r", lw=3, alpha=0.3)
pl.ylim(-10, 100)
pl.savefig("dude.png")

assert 0
print(response.gp_light_curves[0])


def model(t):
    t = (t - t0 + 0.5 * period) % period - 0.5 * period
    r = np.zeros_like(t)
    r[np.fabs(t) < 0.5 * 0.3] = -1
    return r


pl.figure(figsize=(20, 4))
for lc in response.gp_light_curves:
    pl.plot(lc.time, lc.flux, ".k", ms=2)
    pl.plot(lc.time, lc.predict(model), "r", lw=1)
pl.savefig("dude.pdf")
