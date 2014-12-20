#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
bp = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, bp)
from prepare import main as prep_main

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


if __name__ == "__main__":
    import copy
    import sqlite3
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()

    parser.add_argument("base_dir", help="the directory for output")
    parser.add_argument("njobs", type=int, help="the number of jobs to run")
    parser.add_argument("archive_root",
                        help="the location of the data")
    parser.add_argument("--ninj", type=int, default=2,
                        help="the number of injections per star")

    # Search parameters.
    parser.add_argument("--durations", nargs="+", type=float,
                        default=[0.2, 0.4, 0.6],
                        help="the durations to test")
    parser.add_argument("--min-period", type=float, default=100.0,
                        help="minimum period")
    parser.add_argument("--max-period", type=float, default=725.0,
                        help="maximum period")

    parser.add_argument("-p", "--profile-dir", default=None,
                        help="the IPython profile dir")

    args = parser.parse_args()
    print("Running with the following arguments:")
    print("sys.argv:")
    print(sys.argv)
    print("args:")
    print(args)

    # Check that the stellar database exists first.
    dbfn = os.path.join(args.base_dir, "stars.db")
    if not os.path.exists(dbfn):
        raise RuntimeError("You need to generate the stellar database first")

    # Select the stars.
    with sqlite3.connect(dbfn) as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("select * from stars where started=? or ninj<? limit ?",
                  (False, args.ninj, args.njobs))
        stars = c.fetchall()

    print("Found {0} target stars.".format(len(stars)))

    # Initialize the pool.
    c = Client(profile_dir=args.profile_dir)
    pool = c.load_balanced_view()

    # Use the empirical multiplicity distribution for injections.
    multi = np.array([2544., 430., 145., 55., 18., 4.])
    multi /= np.sum(multi)

    jobs = []
    for row in stars:
        a = copy.copy(args)
        bp = os.path.join(args.base_dir, "{0}".format(row["kic"]))

        # If not already started, fire off the main search.
        print(bp, end="\t")

        a.kicid = row["kic"]
        a.rstar = float(row["radius"])
        a.mstar = float(row["mass"])
        a.results_root = os.path.join(bp, "main")
        a.injections = 0
        prep_main(vars(a))
        jobs.append((a.results_root, pool.apply(search, a.results_root)))

        for j in range(row["ninj"], args.ninj):
            a.results_root = os.path.join(bp, "inj-{0}".format(j))
            k = np.argmax(np.random.multinomial(1, multi)) + 1
            a.injections = k
            print(a.injections, end="\t")
            prep_main(vars(a))
            jobs.append((a.results_root, pool.apply(search, a.results_root)))
        print()

        with sqlite3.connect(dbfn) as conn:
            c = conn.cursor()
            c.execute("update stars set started=?,ninj=? where kic=?",
                      (True, args.ninj, row["kic"]))

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
                    with sqlite3.connect(dbfn) as conn:
                        c = conn.cursor()
                        c.execute("update stars set finished=? where kic=?",
                                  (True, row["kic"]))
                retrieved[i] = True
        time.sleep(1)
