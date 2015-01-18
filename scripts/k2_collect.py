#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import json
import glob
import h5py
import numpy as np
import pandas as pd
from collections import defaultdict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # Required arguments.
    parser.add_argument("pattern", help="the directory pattern")
    parser.add_argument("results", help="the results location")

    args = parser.parse_args()
    print("Running with the following arguments:")
    print("sys.argv:")
    print(sys.argv)
    print("args:")
    print(args)

    try:
        os.makedirs(args.results)
    except os.error:
        pass

    # Set up the dictionary that will be used for the pandas DataFrame
    # creation.
    all_features = defaultdict(lambda: [np.nan]*len(all_features["kicid"]))
    all_features["kicid"] = []

    # Loop over the matching directories.
    for ind, d in enumerate(glob.iglob(args.pattern)):
        # Skip if any of the required files don't exist.
        feat_fn = os.path.join(d, "results", "features.h5")
        q_fn = os.path.join(d, "query.json")
        if not os.path.exists(feat_fn) or not os.path.exists(q_fn):
            print("Skipping {0}".format(d))
            continue

        # Get the KIC ID.
        with open(q_fn, "r") as f:
            data = json.load(f)
            kicid = data["kicid"]

        with h5py.File(feat_fn, "r") as f:
            # Parse out the extra information in the header.
            extracols = ["directory"]
            extra = [d]

            # Loop over the peaks and save the features.
            peakid = 0
            for nm in f:
                # Skip non-peak datasets.
                if not nm.startswith("peak_"):
                    continue

                # Extract the peak.
                g = f[nm]
                peak = dict(g.attrs)

                # Include the extra columns.
                for k, v in zip(extracols, extra):
                    all_features[k].append(v)

                # Choose the column names to loop over.
                cols = set(peak.keys() + all_features.keys())
                cols -= set(["kicid", "peakid"] + extracols)
                for k in cols:
                    all_features[str(k)].append(peak.get(k, np.nan))
                all_features["peakid"].append(peakid)
                peakid += 1
                all_features["kicid"].append(kicid)

    # Save the feature DataFrame.
    features = pd.DataFrame(all_features)
    features.to_hdf(os.path.join(args.results, "features.h5"), "features",
                    mode="w")
