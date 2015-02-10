# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["build"]

import os
import glob
import h5py
import fitsio
import logging
import numpy as np


def build(lc_pattern, outfile):
    # Loop over the files and load the aperture photometry.
    print("Loading light curves...")
    lcs = []
    for fn in glob.iglob(lc_pattern):
        if not os.path.exists(fn):
            logging.warn("{0} doesn't exist".format(fn))
            continue
        try:
            aps = fitsio.read(fn, 2)
        except ValueError:
            logging.warn("{0} is corrupted".format(fn))
            continue
        i = np.argmin(np.abs(aps["cdpp6"]))
        data = fitsio.read(fn)
        lcs.append(data["flux"][:, i])

    # Interpolate of missing data.
    print("Interpolating...")
    l_proc = []
    x = np.arange(len(lcs[0]))
    for i, l in enumerate(lcs):
        m = np.isfinite(l)
        l[~m] = np.interp(x[~m], x[m], l[m])
        l_proc.append(l)
    l_proc = np.array(l_proc)

    # Normalize the basis.
    X = np.array(l_proc, dtype=np.float64)
    X -= np.mean(X, axis=1)[:, None]
    X /= np.std(X, axis=1)[:, None]

    # Run the SVD.
    print("Running PCA...")
    u, s, v = np.linalg.svd(X)

    # Save the basis.
    print("Saving to {0}...".format(outfile))
    with h5py.File(outfile, "w") as f:
        f.create_dataset("basis", data=v, compression="gzip")
        f.create_dataset("power", data=s, compression="gzip")

"lightcurves/c1/*/*/*.fits"
"lightcurves/c1-basis.h5"
