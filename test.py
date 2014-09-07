import os
import turnstile


fn = "blahface/download.pkl"
# archive_root = "/export/bbq1/dfm/kplr/data/lightcurves"
# data_root = "blahface"
# kicid = 1724842
# turnstile.PreparedDownload.prepare(fn, archive_root, data_root, kicid)

pipe = turnstile.PreparedDownload()
q = dict(prepared_file=fn)

result = pipe.query(**q)
print(result.target_datasets[0].cache_exists)

assert 0
pipe = turnstile.Download()
# q = dict(kicid=8644545)
# period, t0 = 295.963, 138.91

# Petigura failures.
q = dict(
    kicid=1724842,
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
