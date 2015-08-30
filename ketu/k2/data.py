# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Data"]

import os
import h5py
import copy
import fitsio
import numpy as np
from scipy.linalg import cho_solve, cho_factor

from .epic import Catalog
from ..pipeline import Pipeline
from ..gp_heuristics import estimate_tau, kernel


class Data(Pipeline):

    query_parameters = {
        "light_curve_file": (None, True),
        "catalog_file": (None, True),
        "initial_time": (None, True),
        "skip": (0, False),
        "use_gp": (True, False),
        "invert": (False, False),
    }

    def get_result(self, query, parent_response):
        fn = query["light_curve_file"]
        epicid = os.path.split(fn)[-1].split("-")[0][4:]

        # Query the EPIC.
        cat = Catalog(query["catalog_file"]).df
        _, star = cat[cat.epic_number == int(epicid)].iterrows().next()

        return dict(
            epic=star,
            starid=int(star.epic_number),
            target_light_curves=K2LightCurve(
                fn,
                query["initial_time"],
                gp=query["use_gp"],
                skip=query["skip"],
                invert=query["invert"],
            ).split(),
        )


class K2LightCurve(object):

    def __init__(self, fn, time0, gp=True, skip=0, invert=False):
        self.gp = gp

        data, hdr = fitsio.read(fn, header=True)
        aps = fitsio.read(fn, 2)

        self.texp = (hdr["INT_TIME"] * hdr["NUM_FRM"]) / 86400.0

        # Choose the photometry with the smallest variance.
        var = aps["cdpp6"]
        var[var < 0.0] = np.inf
        i = np.argmin(var)

        # Load the data.
        self.skip = int(skip)
        self.time = data["time"] - time0
        self.flux = data["flux"][:, i]
        if invert:
            mu = np.median(self.flux[np.isfinite(self.flux)])
            self.flux = 2 * mu - self.flux
        q = data["quality"]
        q = ((q == 0) | (q == 16384).astype(bool))
        self.m = (np.isfinite(self.time) &
                  np.isfinite(self.flux) &
                  (np.arange(len(self.time)) > int(skip)) &
                  q)

    def split(self, tol=10):
        # Loop over the time array and break it into "chunks" when there is "a
        # sufficiently long gap" with no data.
        count, current, chunks = 0, [], []
        for i, t in enumerate(self.time):
            if np.isnan(t):
                count += 1
            else:
                if count > tol:
                    chunks.append(list(current))
                    current = []
                    count = 0
                current.append(i)
        if len(current):
            chunks.append(current)

        # Build a list of copies.
        lcs = []
        for chunk in chunks:
            lc = copy.deepcopy(self)
            lc.m = np.zeros(len(self.time), dtype=bool)
            lc.m[chunk] = self.m[chunk]
            lc.time = np.ascontiguousarray(lc.time[lc.m], dtype=np.float64)
            lc.flux = np.ascontiguousarray(lc.flux[lc.m], dtype=np.float64)
            lcs.append(lc)
        return lcs

    def prepare(self, basis_file, nbasis=150, sigma_clip=7.0, max_iter=10,
                tau_frac=0.25, lam=1.0):
        self.lam = lam

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

        # Build the initial kernel matrix.
        self.tau_frac = tau_frac
        self.build_kernels()

        # Do a few rounds of sigma clipping.
        m1 = np.ones_like(self.flux, dtype=bool)
        m2 = np.zeros_like(self.flux, dtype=bool)
        nums = np.arange(len(self.flux))
        count = m1.sum()
        for i in range(max_iter):
            inds = (nums[m1, None], nums[None, m1])
            alpha = np.linalg.solve(self.K[inds], self.flux[m1])
            mu = np.dot(self.K_0[:, m1], alpha)

            # Mask the bad points.
            r = self.flux - mu
            std = np.sqrt(np.median(r ** 2))
            m1 = np.abs(r) < sigma_clip * std
            m2 = r > sigma_clip * std

            if m1.sum() == count:
                break
            count = m1.sum()

        # Force contiguity.
        m2 = ~m2
        self.m[self.m] = m2
        self.time = np.ascontiguousarray(self.time[m2], dtype=np.float64)
        self.flux = np.ascontiguousarray(self.flux[m2], dtype=np.float64)
        self.ferr = np.ascontiguousarray(self.ferr[m2], dtype=np.float64)
        self.basis = np.ascontiguousarray(self.basis[:, m2], dtype=np.float64)

        # Find outliers.
        self.build_kernels()
        mu = np.dot(self.K_0, np.linalg.solve(self.K, self.flux))
        delta = np.diff(self.flux - mu)
        absdel = np.abs(delta)
        mad = np.median(absdel)
        m = np.zeros(self.m.sum(), dtype=bool)
        m[1:-1] = absdel[1:] > sigma_clip * mad
        m[1:-1] &= absdel[:-1] > sigma_clip * mad
        m[1:-1] &= np.sign(delta[1:]) != np.sign(delta[:-1])

        # Remove the outliers and finalize the dataset.
        m = ~m
        self.m[self.m] = m
        self.time = np.ascontiguousarray(self.time[m], dtype=np.float64)
        self.flux = np.ascontiguousarray(self.flux[m], dtype=np.float64)
        self.ferr = np.ascontiguousarray(self.ferr[m], dtype=np.float64)
        self.basis = np.ascontiguousarray(self.basis[:, m], dtype=np.float64)
        self.build_kernels()

        # Precompute some factors.
        self.factor = cho_factor(self.K)
        self.alpha = cho_solve(self.factor, self.flux)

        # Pre-compute the base likelihood.
        self.ll0 = self.lnlike()

    def build_kernels(self):
        self.K_b = np.dot(self.basis.T, self.basis * self.lam)
        if self.gp:
            tau = self.tau_frac * estimate_tau(self.time, self.flux)
            print("tau = {0}".format(tau))
            self.K_t = np.var(self.flux) * kernel(tau, self.time)
            self.K_0 = self.K_b + self.K_t
        else:
            self.K_0 = self.K_b
        self.K = np.array(self.K_0)
        self.K[np.diag_indices_from(self.K)] += self.ferr**2

    def lnlike_eval(self, y):
        return -0.5 * np.dot(y, cho_solve(self.factor, y))

    def grad_lnlike_eval(self, y, dy):
        alpha = cho_solve(self.factor, y)
        return -0.5 * np.dot(y, alpha), np.dot(alpha, dy)

    def lnlike(self, model=None):
        if model is None:
            return -0.5 * np.dot(self.flux, self.alpha)

        # Evaluate the transit model.
        m = model(self.time)
        if np.all(m == 0.0):  # m[0] != 0.0 or m[-1] != 0.0 or
            return 0.0, 0.0, 0.0

        Km = cho_solve(self.factor, m)
        Ky = self.alpha
        ivar = np.dot(m, Km)
        depth = np.dot(m, Ky) / ivar
        r = self.flux - m*depth
        ll = -0.5 * np.dot(r, Ky - depth * Km)
        return ll - self.ll0, depth, ivar

    def predict(self, y=None):
        if y is None:
            y = self.flux
        return np.dot(self.K_0, cho_solve(self.factor, y))

    def predict_t(self, y):
        return np.dot(self.K_t, cho_solve(self.factor, y))

    def predict_b(self, y):
        return np.dot(self.K_b, cho_solve(self.factor, y))
