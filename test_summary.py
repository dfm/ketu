import turnstile

pipe = turnstile.K2Data(cache=False)
pipe = turnstile.K2Likelihood(pipe, cache=False)
pipe = turnstile.K2Summary(pipe, cache=False)

r = pipe.query(
    light_curve_file="../k2/lightcurves/c1/201400000/45000/ktwo201445392-c01_lpd-lc.fits",
    tpf_file="../k2/data/c1/201400000/45000/ktwo201445392-c01_lpd-targ.fits.gz",
    basis_file="../k2/lightcurves/c1-basis.h5",
    summary_file="summary.pdf",
    signals=[
        dict(
            period=10.351576376636384,
            t0=3.461119093700571-1975,
            depth=1.202166780640457,
            duration=0.1,
        ),
        dict(
            period=5.064647588071458,
            t0=4.849647683381297-1975.,
            depth=0.8474045504709299,
            duration=0.1,
        ),
    ]
)

# r = pipe.query(
#     light_curve_file="../k2/lightcurves/c1/201100000/46000/ktwo201146489-c01_lpd-lc.fits",
#     tpf_file="../k2/data/c1/201100000/46000/ktwo201146489-c01_lpd-targ.fits.gz",
#     basis_file="../k2/lightcurves/c1-basis.h5",
#     signals=[
#         dict(
#             period=21.424595040563418,
#             t0=18.52482951035313 - 1975.,
#             depth=5.931646710535624,
#             duration=0.2,
#         ),
#     ]
# )
