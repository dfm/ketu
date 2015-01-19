# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["K2Data"]

import h5py
import fitsio
import numpy as np
from scipy.linalg import cho_solve, cho_factor

from .pipeline import Pipeline


class K2Data(Pipeline):

    query_parameters = {
        "light_curve_file": (None, True),
        "basis_file": (None, True),
    }

    def get_result(self, query, parent_response):
        return dict(model_light_curves=[K2LightCurve(query["light_curve_file"],
                                                     query["basis_file"])])


class K2LightCurve(object):

    def __init__(self, fn, basis_file):
        data = fitsio.read(fn)
        aps = fitsio.read(fn, 2)

        # Choose the photometry with the smallest variance.
        var = aps["cdpp6"]
        var[var < 0.0] = np.inf
        i = np.argmin(var)

        # Load the data.
        self.time = data["time"]
        self.flux = data["flux"][:, i]
        q = data["quality"]

        # Drop the bad data.
        m = np.isfinite(self.time) * np.isfinite(self.flux) * (q == 0)
        self.time = np.ascontiguousarray(self.time[m], dtype=np.float64)
        self.flux = np.ascontiguousarray(self.flux[m], dtype=np.float64)

        # Normalize the data.
        self.flux = self.flux / np.median(self.flux) - 1.0
        self.flux *= 1e3  # Convert to ppt.

        # Estimate the uncertainties.
        self.ivar = 1.0 / np.median(np.diff(self.flux) ** 2)
        self.ferr = np.ones_like(self.flux) / np.sqrt(self.ivar)
        self.scaled = self.flux * self.ivar

        # Load the prediction basis.
        with h5py.File(basis_file, "r") as f:
            basis = f["basis"][...]
        self.basis = np.concatenate((basis[:, m], np.ones((1, m.sum()))))

        self.ll0 = self.lnlike()

    def lnlike(self, model=None):
        if model is None:
            return -0.5 * np.sum((self.predict() - self.flux)**2) * self.ivar

        # Evaluate the transit model.
        m = model(self.time)
        if m[0] != 0.0 or m[-1] != 0.0 or np.all(m == 0.0):
            return 0.0, 0.0, 0.0

        # Build the design matrix and do the linear fit.
        A = np.concatenate(([m], self.basis))
        ATA = np.dot(A, A.T * self.ivar)
        ATA[np.diag_indices_from(ATA)] += 1e-10

        # This is the depth inverse variance.
        s = 1.0 / np.linalg.inv(ATA)[0, 0]

        # And the linear weights (depth is the first).
        factor = cho_factor(ATA, overwrite_a=True)
        w = cho_solve(factor, np.dot(A, self.scaled), overwrite_b=True)

        # Compute the lnlikelihood.
        ll = -0.5 * np.sum((self.flux - np.dot(w, A))**2) * self.ivar
        return ll - self.ll0, w[0], s

    def predict(self, y=None):
        if y is None:
            y = self.flux
        ATA = np.dot(self.basis, self.basis.T)
        ATA[np.diag_indices_from(ATA)] += 1e-10
        w = np.linalg.solve(ATA, np.dot(self.basis, y))
        return np.dot(w, self.basis)
