#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import glob
import json
import turnstile
import numpy as np
from scipy.stats import beta

import gc
import time
import cPickle
import traceback
from IPython.parallel import Client, require


@require(os, sys, gc, cPickle, traceback, time)
def search(bp):
    # Insane hackish output capturing context.
    # http://stackoverflow.com/questions/16571150
    #   /how-to-capture-stdout-output-from-a-python-function-call
    class Capturing(object):

        def __init__(self, fn):
            self.fn = fn

        def __enter__(self):
            self._stdout = sys.stdout
            sys.stdout = self._fh = open(self.fn, "a")
            return self

        def __exit__(self, *args):
            self._fh.close()
            sys.stdout = self._stdout

    # Execute the pipeline.
    r, q, pipe = None, None, None
    try:
        with open(os.path.join(bp, "pipeline.pkl"), "rb") as f:
            q, pipe = cPickle.load(f)

        strt = time.time()
        with Capturing(os.path.join(bp, "output.log")):
            r = pipe.query(**q)

        with open(os.path.join(bp, "output.log"), "a") as f:
            f.write("Finished in {0} seconds\n".format(time.time() - strt))

        # Save the output.
        bp = q["results_path"]
        pipe.save_to_cache(os.path.join(bp, "features.h5"), r)
        pipe.parent.save_to_cache(os.path.join(bp, "peaks.h5"),
                                  r.parent_response)

    except:
        with open(os.path.join(bp, "error.log"), "a") as f:
            f.write("Error during execution:\n\n")
            f.write(traceback.format_exc())

    finally:
        # Try to fix memory leaks.
        del r
        del q
        del pipe
        gc.collect()


def generate_system(K, min_period=3., max_period=80.,
                    min_ror=0.02, max_ror=0.2):
    labels = ["period", "t0", "radius", "b", "e", "pomega", "q1", "q2"]

    periods = np.exp(np.random.uniform(np.log(min_period), np.log(max_period),
                                       size=K))
    t0s = np.array([np.random.uniform(0, p) for p in periods])
    radii = np.exp(np.random.uniform(np.log(min_ror), np.log(max_ror), K))
    b = np.random.uniform(0, 1, K)
    e = beta.rvs(0.867, 3.03, size=K)
    pomega = np.random.uniform(0, 2*np.pi, K)
    q1 = np.random.uniform(0, 1)
    q2 = np.random.uniform(0, 1)

    return dict(q1=q1, q2=q2,
                injections=[dict(zip(labels, _))
                            for _ in zip(periods, t0s, radii, b, e, pomega)])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("file_glob", help="pattern for the LC files")
    parser.add_argument("basis_file", help="the archive of PCA comps")
    parser.add_argument("base_dir", help="the directory for output")
    parser.add_argument("--ninj", type=int, default=0,
                        help="number of injections")
    parser.add_argument("--min-period", type=float, default=0.5,
                        help="minimum period")
    parser.add_argument("--max-period", type=float, default=70.0,
                        help="maximum period")
    parser.add_argument("-p", "--profile-dir", default=None,
                        help="the IPython profile dir")

    args = parser.parse_args()
    print("Running with the following arguments:")
    print("sys.argv:")
    print(sys.argv)
    print("args:")
    print(args)

    query = dict(
        basis_file=os.path.abspath(args.basis_file),
        time_spacing=0.02,
        dt=0.02,
        durations=[0.05, 0.1, 0.2],
        min_period=args.min_period,
        max_period=args.max_period,
        nbasis=150,
    )

    # Initialize the pool.
    c = Client(profile_dir=args.profile_dir)
    pool = c.load_balanced_view()
    jobs = []

    # Use the empirical multiplicity distribution for injections.
    multi = np.array([2544., 430., 145., 55., 18., 4.])
    multi /= np.sum(multi)

    # Loop over the files.
    fns = glob.glob(args.file_glob)
    bs = len(fns)
    for batch in range(0, len(fns), bs):
        for fn in fns[batch:batch+bs]:
            for i in range(args.ninj + 1):
                epicid = os.path.split(fn)[-1].split("-")[0][4:]
                outdir = os.path.abspath(os.path.join(args.base_dir, epicid,
                                                      "{0:04d}".format(i)))
                if os.path.exists(os.path.join(outdir, "results", "features.h5")):
                    print("skipping {0}/{1:04d}".format(epicid, i))
                    continue

                # Update the query.
                query["kicid"] = "EPIC {0}".format(epicid)
                query["light_curve_file"] = os.path.abspath(fn)
                query["results_path"] = os.path.join(outdir, "results")
                if i:
                    k = np.argmax(np.random.multinomial(1, multi)) + 1
                    q = dict(query, **(generate_system(
                        k,
                        min_period=args.min_period,
                        max_period=args.max_period,
                    )))
                else:
                    q = dict(query)

                # Build the pipeline first.
                pipe = turnstile.K2Data(cache=False,
                                        basepath=os.path.join(outdir, "cache"))
                if i:
                    pipe = turnstile.K2Inject(pipe, cache=False)
                pipe = turnstile.K2Likelihood(pipe, cache=False)
                pipe = turnstile.OneDSearch(pipe, cache=False)
                pipe = turnstile.IterativeTwoDSearch(pipe, cache=False)
                pipe = turnstile.FeatureExtract(pipe, cache=False)

                # Save the files.
                try:
                    os.makedirs(query["results_path"])
                except os.error:
                    pass
                with open(os.path.join(outdir, "pipeline.pkl"), "w") as f:
                    cPickle.dump((q, pipe), f, -1)
                with open(os.path.join(outdir, "query.json"), "w") as f:
                    json.dump(q, f, sort_keys=True, indent=4)

                # Submit the job.
                jobs.append((outdir, pool.apply(search, outdir)))

        # Monitor the jobs and check for completion and errors.
        retrieved = [False] * len(jobs)
        while not all(retrieved):
            for i, (fn, j) in enumerate(jobs):
                if j.ready() and not retrieved[i]:
                    try:
                        j.get()
                    except Exception as e:
                        with open(os.path.join(fn, "error.log"), "a") as f:
                            f.write("Uncaught error:\n\n")
                            f.write(traceback.format_exc())
                    else:
                        with open(os.path.join(fn, "success.log"), "w") as f:
                            f.write("Finished at: {0}\n".format(time.time()))
                    retrieved[i] = True
            time.sleep(1)

        assert 0
