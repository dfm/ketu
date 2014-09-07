import os
import turnstile

pipe = turnstile.Download()
# q = dict(kicid=8644545)
# period, t0 = 295.963, 138.91

# Petigura failures.
q = dict(
    kicid=1724842,
    tarball_root="/export/bbq1/dfm/kplr/data/tarballs",
    data_root="blahface",
)
# q = dict(kicid=1570270)

# q = dict(kicid=8692861)

# pipe = turnstile.Inject(pipe)
# q["injections"] = [dict(radius=0.015, period=125., t0=12.)]

# pipe = turnstile.Prepare(pipe)
# pipe = turnstile.GPLikelihood(pipe)

# pipe = turnstile.OneDSearch(pipe)
# q["durations"] = [0.2, 0.4, 0.6]

# pipe = turnstile.TwoDSearch(pipe)
# q["min_period"] = 50
# q["max_period"] = 400

# pipe = turnstile.PeakDetect(pipe)
# pipe = turnstile.FeatureExtract(pipe)
# pipe = turnstile.Validate(pipe)
# q["validation_path"] = "./results"

response = pipe.query(**q)
