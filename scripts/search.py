#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import gc
import cPickle
import traceback
from IPython.parallel import Client, require


@require(os, gc, cPickle, traceback)
def search(pkl_fn):
    r, q, pipe = None, None, None
    try:
        with open(pkl_fn, "rb") as f:
            q, pipe = cPickle.load(f)
        if os.path.exists(os.path.join(q["validation_path"], "features.h5")):
            return

        print("Starting {0}".format(q["kicid"]))
        r = pipe.query(**q)
        print("Finished {0}".format(q["kicid"]))

    except Exception as e:
        print(e)

    finally:
        # Try to fix memory leaks.
        del r
        del q
        del pipe
        gc.collect()


if __name__ == "__main__":
    import os
    import sys
    import glob
    import time
    import kplr
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()

    parser.add_argument("base_dir", help="the directory for output")
    parser.add_argument("njobs", type=int, help="the number of jobs to run")
    parser.add_argument("-p", "--profile-dir", default=None,
                        help="the IPython profile dir")

    args = parser.parse_args()
    print("Running with the following arguments:")
    print("sys.argv:")
    print(sys.argv)
    print("args:")
    print(args)

    # Make the output directory.
    try:
        os.makedirs(args.base_dir)
    except os.error:
        pass

    # Load the stellar catalog.
    kic = kplr.huber.get_catalog()

    # Make the magnitude cut.
    m = kic.kic_kepmag < 13.5

    # Make the T_eff cut.
    m &= (4100 < kic.Teff) & (kic.Teff < 6100)

    # Make the log_g cut.
    m &= (4 < kic["log(g)"]) & (kic["log(g)"] < 4.9)

    # Initialize the pool.
    c = Client(profile_dir=args.profile_dir)
    pool = c.load_balanced_view()

    # Generate the jobs.
    count = 0
    kicids = np.array(kic[m].KIC, dtype=int)
    while True:
        i = np.random.randint(len(kicids))
        kid = kicids[i]
        bp = os.path.join(args.base_dir, "{0}".format(i))
        try:
            os.makedirs(args.base_dir)
        except os.error:
            pass

        for
        # Check to see if the directory is locked.
        lockfn = os.path.join(bp, "lock")
        if os.path.exists(lockfn):
            continue

        count += 1

    assert 0

    c = Client(profile_dir=args.profile_dir)
    pool = c.load_balanced_view()

    jobs = [(fn, pool.apply(search, fn))
            for fn in map(os.path.abspath, glob.glob(args.file_pattern))]
    retrieved = [False] * len(jobs)
    while not all(retrieved):
        for i, (fn, j) in enumerate(jobs):
            if j.ready() and not retrieved[i]:
                try:
                    j.get()
                except Exception as e:
                    print("Task failed: {0}".format(fn))
                    print("With error: \n{0}".format(e))
                else:
                    print("Finished: {0}".format(fn))
                retrieved[i] = True
        time.sleep(1)
