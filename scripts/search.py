#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import gc
import cPickle
from IPython.parallel import Client, require


@require(os, gc, cPickle)
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
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("file_pattern", help="the file pattern")
    parser.add_argument("-p", "--profile-dir", default=None,
                        help="the IPython profile dir")

    args = parser.parse_args()
    print("Running with the following arguments:")
    print("sys.argv:")
    print(sys.argv)
    print("args:")
    print(args)

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
