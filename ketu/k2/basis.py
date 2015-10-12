# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["build"]

import time
import glob
import h5py
import fbpca
import fitsio
import numpy as np
from functools import partial
from multiprocessing import Pool
from scipy.linalg import cho_factor, cho_solve

from ..cdpp import compute_cdpp


def load_data(fn):
    with fitsio.FITS(fn, "r") as f:
        # Skip custom targets.
        hdr = f[1].read_header()
        if hdr["KEPLERID"] < 201000000:
            return None

        # Read the aperture info.
        aps = f[2].read(columns=["cdpp6"])
        c = aps["cdpp6"]
        c[c < 0.0] = np.inf
        i = np.argmin(c)
        data = f[1].read(columns=["flux"])

    # Interpolate over missing data.
    y = data["flux"][:, i]
    x = np.arange(len(y))
    m = np.isfinite(y)
    y[~m] = np.interp(x[~m], x[m], y[m], 1)
    return y


def update_file(K_0, fn):
    with fitsio.FITS(fn, "rw") as f:
        # Skip custom targets.
        hdr = f[1].read_header()
        if hdr["KEPLERID"] < 201000000:
            return None

        # Load the data.
        data = f[1].read(colums=["time", "flux", "quality"])
        t = data["time"]
        y = data["flux"]

        # Choose the "good" data.
        q = data["quality"] == 0
        m = np.all(np.isfinite(y) & q[:, None], axis=1)

        # Mask missing data.
        inds = np.arange(len(t))[m]
        t = t[m]
        y = (y[m, :] / np.median(y[m], axis=0) - 1) * 1e3

        # Predict.
        K = np.array(K_0)
        K[np.diag_indices_from(K)] += np.median(np.diff(y, axis=0)**2)
        pred = np.dot(
            K_0[inds[:, None], inds],
            np.linalg.solve(K[inds[:, None], inds], y)
        )
        resid = y - pred

        # Update the file with the corrected CDPP.
        apinfo = f[2].read()
        for i, r in enumerate(resid.T):
            apinfo["corr_cdpp6"][i] = compute_cdpp(t, 1e-3 * r + 1, 6.,
                                                   robust=True)
        f[2].write(apinfo)

        # Return the best CDPP.
        print(fn)
        return hdr["KEPLERID"], np.min(apinfo["corr_cdpp6"])


def build(lc_pattern, outfile, nbasis=500):
    pool = Pool()

    print("Loading light curves...")
    lcs = []
    fns = glob.glob(lc_pattern)
    lcs = np.array([y for y in pool.map(load_data, fns) if y is not None])
    print("Found {0} light curves...".format(len(lcs)))

    # Normalize the data.
    mu = np.median(lcs, axis=1)
    X = lcs - mu[:, None]

    # Run PCA.
    print("Running PCA...")
    strt = time.time()
    _, _, basis = fbpca.pca(X, k=nbasis, raw=True)
    print("Took {0:.1f} seconds".format(time.time() - strt))

    # Compute the prior.
    print("Computing the empirical 'prior'...")
    factor = cho_factor(np.dot(basis, basis.T))
    Y = 1e3 * (lcs / np.median(lcs, axis=1)[:, None] - 1)
    weights = cho_solve(factor, np.dot(basis, Y.T))
    weights = np.concatenate((weights, -weights), axis=1)
    basis *= np.sqrt(np.median(weights**2, axis=1))[:, None]

    # Update the light curve files with the corrected CDPP values.
    print("Updating light curve files...")
    K_0 = np.dot(basis.T, basis)
    results = np.array(
        [v for v in pool.map(partial(update_file, K_0), fns) if v is not None],
        dtype=[("epicid", int), ("best_cdpp6", float)]
    )

    # Save the basis.
    print("Saving to {0}...".format(outfile))
    with h5py.File(outfile, "w") as f:
        f.create_dataset("basis", data=basis, compression="gzip")
        f.create_dataset("cdpp", data=results, compression="gzip")
