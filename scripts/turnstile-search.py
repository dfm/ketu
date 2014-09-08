#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import cPickle as pickle


def search(pkl_fn):
    with open(pkl_fn, "rb") as f:
        q, pipe = pickle.load(f)

    print(q)

    print("Starting {0}".format(q["kicid"]))
    pipe.query(**q)
    print("Finished {0}".format(q["kicid"]))


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()

    # Required arguments.
    parser.add_argument("files", nargs="+", help="the prepared files")
    args = parser.parse_args()
    print("Running with the following arguments:")
    print("sys.argv:")
    print(sys.argv)
    print("args:")
    print(args)

    map(search, args.files)
