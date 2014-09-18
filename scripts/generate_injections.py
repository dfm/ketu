#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
bp = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, bp)
from prepare import main

import copy
import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()

# Required arguments.
parser.add_argument("total_number", type=int,
                    help="the number of stars to test")
parser.add_argument("archive_root",
                    help="the current location of the data")
parser.add_argument("data_root", help="the destination for the data")
parser.add_argument("results_root", help="the results location")

parser.add_argument("-t", "--pbs-template",
                    default=os.path.join(os.path.dirname(os.path.abspath(
                        __file__)), "job.pbs"),
                    help="the results location")

parser.add_argument("-n", "--n-injections", default=2, type=int,
                    help="the number of injected searches to do per star")

# Search parameters.
parser.add_argument("--durations", nargs="+", type=float,
                    default=[0.2, 0.4, 0.6],
                    help="the durations to test")
parser.add_argument("--min-period", type=float, default=50.0,
                    help="minimum period")
parser.add_argument("--max-period", type=float, default=400.0,
                    help="maximum period")

args = parser.parse_args()
print("Running with the following arguments:")
print("sys.argv:")
print(sys.argv)
print("args:")
print(args)

try:
    os.makedirs(args.results_root)
except os.error:
    pass

with open(args.pbs_template, "r") as f:
    template = f.read()

with open(os.path.join(args.results_root, "job.pbs"), "w") as f:
    f.write(template.format(
        results_root=os.path.abspath(args.results_root),
        data_root=os.path.abspath(args.data_root),
    ))

# Load the stellar sample.
stars = pd.read_hdf(os.path.join(os.path.dirname(bp), "data", "best42k.h5"),
                    "best42k")
nstars = len(stars)

# Loop over N stars and generate some injections.
for i in np.random.randint(nstars, size=args.total_number):
    a = copy.copy(args)

    star = stars[i:i+1]
    kicid = int(star["kic"])
    print(kicid)
    a.kicid = kicid
    a.rstar = float(star["Rstar"])
    a.mstar = float(star["Mstar"])
    a.results_root = os.path.join(args.results_root, "{0}".format(kicid))
    a.injections = 0
    main(vars(a))

    for j in range(args.n_injections):
        a.results_root = os.path.join(args.results_root,
                                      "{0}-inj-{1}".format(kicid, j))
        k = np.random.poisson(3)
        while k <= 0:
            k = np.random.poisson(3)
        a.injections = k
        print(a.injections)
        main(vars(a))
