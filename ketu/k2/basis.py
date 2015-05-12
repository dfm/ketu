# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["build"]

import os
import glob
import h5py
import fitsio
import logging
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ..pcp import pcp


def build(lc_pattern, outfile, sigma_maxiter=50, sigma_clip=7.0,
          pcp_mu=1e-2, pcp_maxiter=10):
    # Loop over the files and load the aperture photometry.
    print("Loading light curves...")
    lcs = []
    for fn in glob.iglob(lc_pattern):
        hdr = fitsio.read_header(fn)

        # Skip custom targets.
        if hdr["KEPLERID"] < 201000000:
            continue
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
    print("Found {0} light curves...".format(len(lcs)))

    # Sigma clip then normalize the data.
    X = np.empty((len(lcs), len(lcs[0])), dtype=np.float64)
    x = np.arange(len(lcs[0]))
    for i, l in enumerate(lcs):
        m = np.isfinite(l)
        m0 = np.ones_like(m)
        m0[~m] = False
        for j in range(sigma_maxiter):
            mu = np.mean(l[m & m0])
            std = np.std(l[m & m0])
            count = m0.sum()
            m0[m] = np.abs(l[m] - mu) < sigma_clip * std
            if count == m0.sum():
                break
        mu = np.mean(l[m0])
        std = np.std(l[m0])
        X[i] = (l - mu) / std
        X[i, ~m0] = np.nan

    # Run robust PCA.
    print("Running PCA...")
    L, S, (u, s, v) = pcp(X, verbose=True, maxiter=pcp_maxiter, mu=pcp_mu,
                          svd_method="exact")

    # Fit the light curves to build an empirical prior.
    print("Generating empirical prior...")
    factor = cho_factor(np.dot(v, v.T))
    weights = np.empty((len(lcs), len(v)))
    for i, lc in enumerate(lcs):
        m = np.isfinite(lc)
        lc = 1e3 * (lc / np.median(lc[m]) - 1.0)
        lc[~m] = 0.0
        weights[i] = cho_solve(factor, np.dot(v, lc))

    # Normalize the basis so that it has a unit Gaussian prior in PPT.
    basis = np.array(v)
    basis *= np.sqrt(np.median(weights**2, axis=0))[:, None]

    # Save the basis.
    print("Saving to {0}...".format(outfile))
    with h5py.File(outfile, "w") as f:
        f.create_dataset("basis", data=basis, compression="gzip")
        # f.create_dataset("power", data=s, compression="gzip")

"lightcurves/c1/*/*/*.fits"
"lightcurves/c1-basis.h5"
