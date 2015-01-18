import turnstile
import numpy as np
import matplotlib.pyplot as pl

pipe = turnstile.K2Data(cache=False)
pipe = turnstile.OneDSearch(pipe)
pipe = turnstile.TwoDSearch(pipe)
pipe = turnstile.PeakDetect(pipe)
pipe = turnstile.FeatureExtract(pipe)
pipe = turnstile.Validate(pipe)

q = dict(
    kicid="EPIC201367065",
    light_curve_file=("../k2/lightcurves/c1/201300000/67000/"
                      "ktwo201367065-c01_lpd-lc.fits"),
    basis_file="../k2/lightcurves/c1-basis.h5",
    durations=[0.2],
    min_period=5.0,
    max_period=100.0,
    validation_path="k2demo",
)

response = pipe.query(**q)

# lc = response.model_light_curves[0]
# mu = lc.predict()
# print(lc.lnlike(lambda x: np.random.randn(len(x))))

pl.plot(response.period_2d, response.phic_same, "k")
pl.gca().axvline(10.05403)
pl.gca().axvline(24.6454)
pl.gca().axvline(44.5631)
for peak in response.peaks[:3]:
    pl.plot(peak["period"], peak["phic_same"], ".r")
pl.xlabel("period")
pl.ylabel("phic")
pl.savefig("blah.png")
