# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["K2Data"]

import os
import kplr
import h5py
import fitsio
import numpy as np
from scipy.linalg import cho_solve, cho_factor

from .pipeline import Pipeline


class K2Data(Pipeline):

    query_parameters = {
        "light_curve_file": (None, True),
        "initial_time": (1975., False)
    }

    def get_result(self, query, parent_response):
        client = kplr.API()
        fn = query["light_curve_file"]
        epicid = os.path.split(fn)[-1].split("-")[0][4:]
        return dict(
            epic=client.k2_star(int(epicid)),
            target_light_curves=[K2LightCurve(fn,
                                              time0=query["initial_time"])],
        )


class K2LightCurve(object):

    def __init__(self, fn, time0=1975.):
        data, hdr = fitsio.read(fn, header=True)
        aps = fitsio.read(fn, 2)

        self.texp = (hdr["INT_TIME"] * hdr["NUM_FRM"]) / 86400.0

        # Choose the photometry with the smallest variance.
        var = aps["cdpp6"]
        var[var < 0.0] = np.inf
        i = np.argmin(var)

        # Load the data.
        self.time = data["time"] - time0
        self.flux = data["flux"][:, i]
        q = data["quality"]

        # Drop the bad data.
        self.m = np.isfinite(self.time) * np.isfinite(self.flux) * (q == 0)
        self.time = np.ascontiguousarray(self.time[self.m], dtype=np.float64)
        self.flux = np.ascontiguousarray(self.flux[self.m], dtype=np.float64)

    def prepare(self, basis_file, nbasis=150, sigma_clip=7.0, max_iter=7):
        # Normalize the data.
        self.flux = self.flux / np.median(self.flux) - 1.0
        self.flux *= 1e3  # Convert to ppt.

        # Estimate the uncertainties.
        self.ivar = 1.0 / np.median(np.diff(self.flux) ** 2)
        self.ferr = np.ones_like(self.flux) / np.sqrt(self.ivar)

        # Load the prediction basis.
        with h5py.File(basis_file, "r") as f:
            basis = f["basis"][:nbasis, :]
        self.basis = np.concatenate((basis[:, self.m],
                                     np.ones((1, self.m.sum()))))
        self.update_matrices()

        # Do a few rounds of sigma clipping.
        m = None
        i = 0
        while i < max_iter and (m is None or m.sum()):
            mu = self.predict()
            std = np.sqrt(np.median((self.flux - mu) ** 2))
            m = self.flux - mu > sigma_clip * std
            self.flux = self.flux[~m]
            self.time = self.time[~m]
            self.ferr = self.ferr[~m]
            self.basis = self.basis[:, ~m]
            self.update_matrices()
            i += 1
            print(i, std, sum(m), len(self.flux))

        # Force contiguity.
        self.time = np.ascontiguousarray(self.time, dtype=np.float64)
        self.flux = np.ascontiguousarray(self.flux, dtype=np.float64)

        # Pre-compute the base likelihood.
        self.ll0 = self.lnlike()

    def update_matrices(self):
        n = self.basis.shape[0] + 1
        self.ATA = np.empty((n, n), dtype=np.float64)
        self.ATA[1:, 1:] = np.dot(self.basis, self.basis.T)
        self.ATA[np.diag_indices_from(self.ATA)] += 1e-10
        self._factor = cho_factor(self.ATA[1:, 1:])

        self.scaled = np.empty(n, dtype=np.float64)
        self.scaled[1:] = np.dot(self.basis, self.flux)

    def lnlike(self, model=None):
        if model is None:
            return -0.5 * np.sum((self.predict() - self.flux)**2) * self.ivar

        # Evaluate the transit model.
        m = model(self.time)
        if m[0] != 0.0 or m[-1] != 0.0 or np.all(m == 0.0):
            return 0.0, 0.0, 0.0

        # Update the matrices.
        v = np.dot(self.basis, m)
        self.ATA[0, 1:] = v
        self.ATA[1:, 0] = v
        self.ATA[0, 0] = np.dot(m, m)
        self.scaled[0] = np.dot(m, self.flux)

        # This is the depth inverse variance.
        s = self.ivar / np.linalg.inv(self.ATA)[0, 0]

        # And the linear weights (depth is the first).
        factor = cho_factor(self.ATA, overwrite_a=False)
        w = cho_solve(factor, self.scaled, overwrite_b=False)

        # Compute the lnlikelihood.
        mu = np.dot(w[1:], self.basis) + w[0] * m
        ll = -0.5 * np.sum((self.flux - mu)**2) * self.ivar
        return ll - self.ll0, w[0], s

    def predict(self, y=None):
        if y is None:
            y = self.flux
        w = cho_solve(self._factor, np.dot(self.basis, y), overwrite_b=True)
        return np.dot(w, self.basis)
