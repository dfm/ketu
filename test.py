import os
import turnstile


kicid = 8644545
fn = "blahface/{0}-download.pkl".format(kicid)
archive_root = "/export/bbq1/dfm/kplr/data/lightcurves"
data_root = "blahface"
turnstile.PreparedDownload.prepare(fn, archive_root, data_root, kicid)

pipe = turnstile.PreparedDownload(basepath="./cache", cache=False)
q = dict(
    kicid=kicid,
    prepared_file=fn,
)
# period, t0 = 295.963, 138.91

# # Petigura failures.
# q = dict(
#     kicid=1724842,
# )
# q = dict(kicid=1570270)

# q = dict(kicid=8692861)

pipe = turnstile.Inject(pipe, cache=False)
q["injections"] = [dict(radius=0.015, period=125., t0=12.)]

pipe = turnstile.Prepare(pipe, cache=False)
pipe = turnstile.GPLikelihood(pipe, cache=False)

pipe = turnstile.OneDSearch(pipe)
q["durations"] = [0.2, 0.4, 0.6]

pipe = turnstile.TwoDSearch(pipe)
q["min_period"] = 50
q["max_period"] = 400

pipe = turnstile.PeakDetect(pipe)
pipe = turnstile.FeatureExtract(pipe)
pipe = turnstile.Validate(pipe)
q["validation_path"] = os.path.join(".", "results",
                                    "{0}-injection".format(kicid))

response = pipe.query(**q)
