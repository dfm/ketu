#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import json
import turnstile
import numpy as np
import cPickle as pickle
from scipy.stats import beta


def prepare(kicid, archive_root, results_root, injections=None,
            durations=[0.2, 0.4, 0.6, 0.8], min_period=50, max_period=750):
    archive_root = os.path.abspath(archive_root)
    results_root = os.path.abspath(results_root)

    try:
        os.makedirs(results_root)
    except os.error:
        pass

    # Start by moving the files into place.
    # fn = os.path.join(results_root, "download.pkl")
    # turnstile.PreparedDownload.prepare(fn, archive_root, data_root, kicid)

    # Set up the default query.
    q = dict(
        kicid=kicid,
        # prepared_file=fn,
        durations=sorted(map(float, durations)),
        min_period=float(min_period),
        max_period=float(max_period),
        validation_path=os.path.join(results_root, "results"),
    )

    # And the pipeline.
    cache_root = os.path.join(results_root, "cache")
    pipe = turnstile.Download(basepath=cache_root, cache=False)
    # pipe = turnstile.PreparedDownload(basepath=cache_root, cache=False)

    # Add the injections if given.
    pipe = turnstile.Inject(pipe, cache=False)
    if injections is not None:
        q = dict(q, **injections)

    pipe = turnstile.Prepare(pipe, cache=False)
    pipe = turnstile.GPLikelihood(pipe, cache=False)
    pipe = turnstile.OneDSearch(pipe, clobber=True)
    pipe = turnstile.TwoDSearch(pipe, cache=False)
    pipe = turnstile.PeakDetect(pipe, cache=False)
    pipe = turnstile.FeatureExtract(pipe, cache=False)
    pipe = turnstile.Validate(pipe, cache=False)

    pkl_fn = os.path.join(results_root, "pipeline.pkl")
    with open(pkl_fn, "wb") as f:
        pickle.dump((q, pipe), f, -1)

    query_fn = os.path.join(results_root, "query.json")
    with open(query_fn, "w") as f:
        json.dump(q, f, sort_keys=True, indent=2, separators=(",", ": "))


def generate_system(K, mstar=1.0, rstar=1.0, min_period=50., max_period=750.):
    labels = ["period", "t0", "radius", "b", "e", "pomega", "q1", "q2"]

    periods = np.exp(np.random.uniform(np.log(min_period), np.log(max_period),
                                       size=K))
    t0s = np.array([np.random.uniform(0, p) for p in periods])
    radii = np.exp(np.random.uniform(np.log(0.007), np.log(0.15), K))
    b = np.random.uniform(0, 1, K)
    e = beta.rvs(0.867, 3.03, size=K)
    pomega = np.random.uniform(0, 2*np.pi, K)
    q1 = np.random.uniform(0, 1)
    q2 = np.random.uniform(0, 1)

    return dict(q1=q1, q2=q2, mstar=mstar, rstar=rstar,
                injections=[dict(zip(labels, _))
                            for _ in zip(periods, t0s, radii, b, e, pomega)])


def main(args):
    # Add an injections if requested.
    injections = None
    if args.get("injections", 0) > 0:
        if args.get("seed", None) is not None:
            np.random.seed(args.seed)
        injections = generate_system(args.get("injections", 0),
                                     mstar=args.get("mstar", 1.0),
                                     rstar=args.get("rstar", 1.0),
                                     min_period=args.get("min_period", 50.0),
                                     max_period=args.get("max_period", 750.0))

    # Prepare the system.
    prepare(args["kicid"], args["archive_root"],
            args["results_root"],
            durations=args.get("durations", [0.2, 0.4, 0.6]),
            min_period=args.get("min_period", 50.0),
            max_period=args.get("max_period", 750.0),
            injections=injections)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()

    # Required arguments.
    parser.add_argument("kicid", type=int, help="the KIC ID")
    parser.add_argument("archive_root",
                        help="the current location of the data")
    parser.add_argument("results_root", help="the results location")

    # Search parameters.
    parser.add_argument("--durations", nargs="+", type=float,
                        default=[0.2, 0.4, 0.6],
                        help="the durations to test")
    parser.add_argument("--min-period", type=float, default=50.0,
                        help="minimum period")
    parser.add_argument("--max-period", type=float, default=750.0,
                        help="maximum period")

    # Injection parameters.
    parser.add_argument("-s", "--seed", default=None, type=int,
                        help="the random number seed")
    parser.add_argument("--injections", default=0, type=int,
                        help="the number of injections")
    parser.add_argument("--mstar", default=1.0, type=float,
                        help="the stellar mass")
    parser.add_argument("--rstar", default=1.0, type=float,
                        help="the stellar radius")

    args = parser.parse_args()
    print("Running with the following arguments:")
    print("sys.argv:")
    print(sys.argv)
    print("args:")
    print(args)

    main(vars(args))
