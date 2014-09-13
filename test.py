import os
import turnstile


# kicid = 4921578
# kicid = 11244118
# kicid = 5701829
# kicid = 3246460
kicid = 8644545
pipe = turnstile.Download()
q = dict(kicid=kicid)

# pipe = turnstile.Inject(pipe, cache=False)
# q["injections"] = [dict(radius=0.03, period=125., t0=113.)]

pipe = turnstile.Prepare(pipe, cache=False)
pipe = turnstile.GPLikelihood(pipe, cache=False)
q["matern"] = True

pipe = turnstile.OneDSearch(pipe)
q["durations"] = [0.2, 0.4, 0.6]

pipe = turnstile.TwoDSearch(pipe)
q["min_period"] = 50
q["max_period"] = 400

pipe = turnstile.PeakDetect(pipe)
pipe = turnstile.FeatureExtract(pipe)
pipe = turnstile.Validate(pipe)
q["validation_path"] = os.path.join(".", "results",
                                    "{0}-matern32".format(kicid))

response = pipe.query(**q)
