# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Data"]

import os
import h5py
import fitsio
import numpy as np
from scipy.linalg import cho_solve, cho_factor
from scipy.ndimage.filters import gaussian_filter

from ..pipeline import Pipeline
from .epic import Catalog


class Data(Pipeline):

    query_parameters = {
        "light_curve_file": (None, True),
        "catalog_file": (None, True),
        "initial_time": (1975., False),
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

    def prepare(self, basis_file, nbasis=150, sigma_clip=7.0, max_iter=10,
                tau_frac=0.25):
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

            print(m1.sum(), count)
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

        # Build the final GP kernel.
        self.build_kernels()

        # Precompute some factors.
        self.factor = cho_factor(self.K)
        self.alpha = cho_solve(self.factor, self.flux)

        # Pre-compute the base likelihood.
        self.ll0 = self.lnlike()

    def build_kernels(self):
        self.K_b = np.dot(self.basis.T, self.basis)
        tau = self.tau_frac * estimate_tau(self.time, self.flux)
        print("tau = {0}".format(tau))
        self.K_t = np.var(self.flux) * kernel(tau, self.time)
        self.K_0 = self.K_b + self.K_t
        self.K = np.array(self.K_0)
        self.K[np.diag_indices_from(self.K)] += self.ferr**2

    def lnlike(self, model=None):
        if model is None:
            return -0.5 * np.dot(self.flux, self.alpha)

        # Evaluate the transit model.
        m = model(self.time)
        if m[0] != 0.0 or m[-1] != 0.0 or np.all(m == 0.0):
            return 0.0, 0.0, 0.0

        Km = cho_solve(self.factor, m)
        Ky = cho_solve(self.factor, self.flux)
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


def acor_fn(x):
    """Compute the autocorrelation function of a time series."""
    n = len(x)
    f = np.fft.fft(x-np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    return acf / acf[0]


def estimate_tau(t, y):
    """Estimate the correlation length of a time series."""
    dt = np.min(np.diff(t))
    tt = np.arange(t.min(), t.max(), dt)
    yy = np.interp(tt, t, y, 1)
    f = acor_fn(yy)
    fs = gaussian_filter(f, 50)
    w = dt * np.arange(len(f))
    m = np.arange(1, len(fs)-1)[(fs[1:-1] > fs[2:]) & (fs[1:-1] > fs[:-2])]
    if len(m):
        return w[m[np.argmax(fs[m])]]
    return w[-1]


def kernel(tau, t):
    """Matern-3/2 kernel function"""
    r = np.sqrt(3 * ((t[:, None] - t[None, :]) / tau) ** 2)
    return (1 + r) * np.exp(-r)
