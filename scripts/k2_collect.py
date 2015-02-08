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
    all_injections = defaultdict(lambda: [np.nan]*len(all_injections["kicid"]))
    all_injections["kicid"] = []

    # Loop over the matching directories.
    for ind, d in enumerate(glob.iglob(args.pattern)):
        # if ind > 500:
        #     break

        # Skip if any of the required files don't exist.
        feat_fn = os.path.join(d, "results", "features.h5")
        q_fn = os.path.join(d, "query.json")
        if not os.path.exists(feat_fn) or not os.path.exists(q_fn):
            # print("Skipping {0}".format(d))
            continue

        # Get the KIC ID.
        with open(q_fn, "r") as f:
            data = json.load(f)
            kicid = data["kicid"]

        with h5py.File(feat_fn, "r") as f:
            # Get any injection information.
            inj_rec = f["inj_rec"][...]
            if len(inj_rec):
                for inj in inj_rec:
                    for k in inj.dtype.names:
                        all_injections[k].append(inj[k])
                    all_injections["directory"].append(d)
                    all_injections["kepmag"].append(f.attrs["kic_kepmag"])
                    all_injections["kicid"].append(kicid)

            # Parse out the extra information in the header.
            extracols = ["kic_kepmag", "directory", "has_injection"]
            extra = [f.attrs["kic_kepmag"], d, len(inj_rec) > 0]

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

                # Include the binned light curve.
                lc = g["bin_lc"][...]
                to_skip = []
                for i, row in enumerate(lc):
                    k = "lc_{0}".format(i)
                    to_skip.append(k)
                    all_features[k].append(row["flux"])

                # Include stats on the corrected light curve.
                lc = g["corr_lc"][...]
                fl = lc["flux"]
                mean = np.mean(fl)
                median = np.median(fl)
                var = np.var(fl)
                rvar = np.median((fl - median) ** 2)
                mn, mx = np.min(fl), np.max(fl)
                all_features["mean"].append(mean)
                all_features["median"].append(median)
                all_features["var"].append(var)
                all_features["rvar"].append(rvar)
                all_features["min"].append(mn)
                all_features["max"].append(mx)
                to_skip += ["mean", "median", "var", "rvar", "min", "max"]

                # And binned stats.
                be = np.linspace(-1.5, 1.5, 20)
                inds = np.digitize(lc["time"], be) - 1
                meds = np.array([np.median(fl[inds == k])
                                 for k in range(len(be)-1)
                                 if np.any(inds == k)])
                all_features["bvar"].append(np.var(meds))
                mmeds = np.median(meds)
                all_features["rbvar"].append(np.median((meds-mmeds)**2))
                to_skip += ["bvar", "rbvar"]

                # Choose the column names to loop over.
                cols = set(peak.keys() + all_features.keys())
                cols -= set(["kicid", "peakid"] + to_skip + extracols)
                for k in cols:
                    all_features[str(k)].append(peak.get(k, np.nan))
                all_features["peakid"].append(peakid)
                peakid += 1
                all_features["kicid"].append(kicid)

    # Make sure that NaNs become Falses when they should.
    all_features["injected_rec"] = [v if np.isfinite(v) else False
                                    for v in all_features["injected_rec"]]

    # Save the feature DataFrame.
    features = pd.DataFrame(all_features)
    features.to_hdf(os.path.join(args.results, "features.h5"), "features",
                    mode="w")

    # Save the injections DataFrame.
    injs = pd.DataFrame(all_injections)
    injs.to_hdf(os.path.join(args.results, "injections.h5"), "injections",
                mode="w")
