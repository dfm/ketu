#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import json
import glob
import h5py
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as pl
from numpy.lib import recfunctions


with h5py.File("data/completeness.h5", "r") as f:
    bins = [f["ln_period_bin_edges"][...],
            f["ln_radius_bin_edges"][...]]
    lncompleteness = f["ln_completeness"][...]


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

    # Loop over results directories.
    dtype = None
    injections = []
    noinj_features = []
    inj_features = []
    for d in glob.iglob(args.pattern):
        feat_fn = os.path.join(d, "results", "features.h5")
        q_fn = os.path.join(d, "results", "query.json")
        if not os.path.exists(feat_fn) or not os.path.exists(q_fn):
            print("Skipping {0}".format(d))
            continue

        with open(q_fn, "r") as f:
            data = json.load(f)
            kicid = data["kicid"]

        with h5py.File(feat_fn, "r") as f:
            inj_rec = f["inj_rec"][...]
            if len(inj_rec):
                injections.append(
                    recfunctions.append_fields(inj_rec, "kicid",
                                               kicid + np.zeros(len(inj_rec),
                                                                dtype=int)))

            extracols = ["kic_kepmag", "kic_teff", "kic_logg"]
            extra = [kicid] + [f.attrs[k] for k in extracols]
            extracols = ["kicid"] + extracols

            # Loop over the peaks and check if they're injections.
            peaks = []
            for nm in f:
                if not nm.startswith("peak_"):
                    continue
                g = f[nm]
                peak = dict(g.attrs)

                lc = g["bin_lc"][...]

                if dtype is None:
                    dtype = [(str(c), float) for c in sorted(peak.keys())
                             if not (c.startswith("is_") or
                                     c.startswith("injected_"))]
                    dtype = dtype + [("is_injection", bool), ("is_koi", bool)]
                    colnames = [c for c, _ in dtype]
                    dtype = zip(extracols, [int, float, float, float]) + dtype
                    dtype += [("lc_{0}".format(i), float)
                              for i in range(len(lc))]
                    dtype += [("lc_err_{0}".format(i), float)
                              for i in range(len(lc))]
                    dtype = np.dtype(dtype)
                if len(inj_rec):
                    inj_features.append(tuple(extra
                                              + [peak[c] for c in colnames])
                                        + tuple(lc["flux"])
                                        + tuple(lc["flux_err"]))
                else:
                    noinj_features.append(tuple(extra +
                                                [peak[c] for c in colnames])
                                          + tuple(lc["flux"])
                                          + tuple(lc["flux_err"]))

    # Save the features.
    inj_features = np.array(inj_features, dtype=dtype)
    noinj_features = np.array(noinj_features, dtype=dtype)
    print(len(inj_features))
    with h5py.File(os.path.join(args.results, "features.h5"), "w") as f:
        f.create_dataset("inj_features", data=inj_features)
        f.create_dataset("noinj_features", data=noinj_features)

    dtype = injections[0].dtype
    injections = np.array(np.concatenate(injections, axis=0), dtype=dtype)
    with h5py.File(os.path.join(args.results, "injections.h5"), "w") as f:
        f.create_dataset("injections", data=injections)

    assert 0

    m = injections["rec"]
    print(np.sum(m), len(injections))

    samples = np.vstack((np.log(injections["period"]),
                         np.log(injections["radius"] / 0.01))).T
    img_all, b = np.histogramdd(samples, bins=(bins[0][-12:], bins[1][:40]))
    img_yes, _ = np.histogramdd(samples[m], bins=b)
    z = img_yes / img_all
    z[~np.isfinite(z)] = 1.0
    z = scipy.ndimage.filters.gaussian_filter(z, 1.)
    x, y = b
    c = pl.contour(x[:-1]+0.5*np.diff(x), y[:-1]+0.5*np.diff(y), z.T,
                   12, colors="r", linewidths=1, alpha=0.6, vmin=0,
                   vmax=1)
    pl.clabel(c, fontsize=12, inline=1, fmt="%.2f")

    # pl.plot(np.log(injections["period"][~m]),
    #         np.log(injections["radius"][~m] / 0.01),
    #         ".r")
    # pl.plot(np.log(injections["period"][m]),
    #         np.log(injections["radius"][m] / 0.01),
    #         ".k")

    x, y = bins
    z = np.exp(lncompleteness[1:-1, 1:-1])
    z = scipy.ndimage.filters.gaussian_filter(z, 1)
    c = pl.contour(x[:-1]+0.5*np.diff(x), y[:-1]+0.5*np.diff(y), z.T,
                   12, colors="k", linewidths=1, alpha=0.6, vmin=0,
                   vmax=1)
    pl.clabel(c, fontsize=12, inline=1, fmt="%.2f")

    pl.gca().axhline(0.0, color="k", lw=2, alpha=0.3)
    pl.gca().axvline(np.log(365.), color="k", lw=2, alpha=0.3)

    pl.xlim(min(b[0]), max(b[0]))
    pl.ylim(min(b[1]), max(b[1]) + 1)
    pl.xlabel(r"$\ln P / \mathrm{days}$")
    pl.ylabel(r"$\ln R / R_\oplus$")
    pl.savefig(os.path.join(args.results, "completeness.png"))
