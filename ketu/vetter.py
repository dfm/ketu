# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Vetter"]

import os
import h5py
import transit
import numpy as np
from scipy.optimize import minimize

from .pipeline import Pipeline


def _nll_transit(p, system, lcs):
    system.set_vector(p)

    ll = 0.0
    for lc in lcs:
        mod = 1e3*(system.light_curve(lc.time, texp=lc.texp)-1.0)
        r = mod - lc.flux
        ll += lc.lnlike_eval(r)
    return -ll


def _nll_and_grad_transit(p, system, lcs):
    system.set_vector(p)

    ll = 0.0
    ll_grad = np.zeros_like(p)
    for lc in lcs:
        mod, grad = system.light_curve_gradient(lc.time, texp=lc.texp)
        r = 1e3 * (mod - 1.0) - lc.flux
        if np.any(~np.isfinite(r)):
            assert 0
        grad *= 1e3
        a, b = lc.grad_lnlike_eval(r, grad)
        ll += a
        ll_grad += b
    return -ll, ll_grad


def _ln_evidence_basic(lcs):
    ll = sum(lc.lnlike_eval(lc.flux) for lc in lcs)
    return ll, ll


def _ln_evidence_outlier(lcs, period, duration, t0):
    hp = 0.5 * period
    hd = 0.5 * duration

    lnlike = 0.0
    norm = 0.0
    depths = np.empty(len(lcs))
    ivars = np.empty(len(lcs))
    for i, lc in enumerate(lcs):
        r = lc.flux - lc.predict()
        m = np.fabs((lc.time - t0 + hp) % period - hp) < hd
        if not np.any(m):
            lnlike += lc.ll0
            continue
        tloc = lc.time[m][np.argmin(r[m])]

        def model(t):
            mod = np.zeros_like(t)
            mod[t == tloc] = -1.0
            return mod

        l0, depths[i], ivars[i] = lc.lnlike(model)
        lnlike += lc.ll0
        if ivars[i] > 0.0:
            lnlike += l0
            norm += 0.5 * (np.log(2*np.pi) - np.log(ivars[i]))

    return lnlike, lnlike + norm


def _ln_like_box(lcs, period, duration, t0):
    hp = 0.5 * period
    hd = 0.5 * duration

    def model(t):
        mod = np.zeros_like(t)
        mod[np.fabs((t - t0 + hp) % period - hp) < hd] = -1.0
        return mod

    lnlike = 0.0
    depths = np.empty(len(lcs))
    ivars = np.empty(len(lcs))
    for i, lc in enumerate(lcs):
        l0, depths[i], ivars[i] = lc.lnlike(model)
        lnlike += lc.ll0
        if ivars[i] > 0.0:
            lnlike += l0

    m = ivars > 0.0
    depths = depths[m]
    ivars = ivars[m]

    depth = np.sum(ivars * depths) / np.sum(ivars)
    ivar = np.sum(ivars)

    lnlike -= 0.5 * np.sum(depths**2 * ivars)
    lnlike += 0.5 * depth**2 * ivar
    lnlike += 0.5 * np.sum(np.log(ivars)) - 0.5*len(depths)*np.log(2*np.pi)

    return lnlike, ivar


def _ln_evidence_box(lcs, period, duration, t0):
    lnlike, ivar = _ln_like_box(lcs, period, duration, t0)
    return lnlike, lnlike - 0.5 * np.log(ivar) + 0.5 * np.log(2*np.pi)


def _ln_evidence_transit(p, *args):
    h = 1.1234e-3
    x0 = np.array(p)
    lnd2fdx2 = np.empty_like(x0)
    f0 = _nll_transit(x0, *args)
    for i in range(len(x0)):
        x0[i] += h
        fp = _nll_transit(x0, *args)
        x0[i] -= 2*h
        fm = _nll_transit(x0, *args)
        x0[i] += h
        d = fp - 2*f0 + fm
        if d <= 0.0:
            lnd2fdx2[i] = np.inf
        else:
            lnd2fdx2[i] = np.log(fp - 2*f0 + fm)
    lnd2fdx2 -= 2*np.log(h)
    lnZ = -f0 - 0.5 * np.sum(lnd2fdx2) + 0.5 * len(x0) * np.log(2*np.pi)
    return -f0, lnZ


class Vetter(Pipeline):

    cache_ext = ".h5"
    query_parameters = dict(
        t0_rng=(0.2, False),
        period_rng=(0.1, False),
    )

    def get_result(self, query, parent_response):
        # Get the results from the pipeline so far.
        peaks = parent_response.peaks
        lcs = parent_response.model_light_curves

        # Save the initial flux values.
        flux0 = [np.array(lc.flux) for lc in lcs]

        # Loop over the peaks and compute the evidence for each one.
        for peak in peaks:
            # Compute the evidence for the box model.
            peak["lnlike_none"], peak["lnZ_none"] = _ln_evidence_basic(lcs)
            peak["lnlike_box"], peak["lnZ_box"] = _ln_evidence_box(
                lcs, peak["period"], peak["duration"], peak["t0"])
            peak["lnlike_outlier"], peak["lnZ_outlier"] = _ln_evidence_outlier(
                lcs, peak["period"], peak["duration"], peak["t0"])

            # Set up the Keplerian fit.
            system = transit.SimpleSystem(
                period=peak["period"], t0=peak["t0"],
                ror=np.sqrt(1e-3*peak["depth"]),
                duration=peak["duration"],
                impact=0.5,
            )

            # Fit the transit model.
            p0 = system.get_vector()
            ln_period_rng = np.log((
                peak["period"] - query["period_rng"],
                peak["period"] + query["period_rng"],
            ))
            t0_rng = (
                peak["t0"] - query["t0_rng"],
                peak["t0"] + query["t0_rng"],
            )
            bounds = [(None, None), ln_period_rng, t0_rng,
                      (None, None), (None, None),
                      (-10.0, 10.0), (-10.0, 10.0)]
            result = minimize(_nll_and_grad_transit, p0, method="L-BFGS-B",
                              jac=True,
                              args=(system, lcs),
                              bounds=bounds)
            system.set_vector(result.x)

            # Compute the transit evidence.
            x = result.x
            peak["transit_q1"] = system.q1
            peak["transit_q2"] = system.q2
            peak["transit_period"] = system.period
            peak["transit_ror"] = system.ror
            peak["transit_duration"] = system.duration
            peak["transit_t0"] = system.t0
            peak["transit_b"] = system.impact
            peak["lnlike_transit"], peak["lnZ_transit"] = \
                _ln_evidence_transit(x, system, lcs)

            # Subtract the best fit transit model.
            for lc in lcs:
                mod = 1e3*(system.light_curve(lc.time, texp=lc.texp)-1.0)
                lc.flux -= mod

        # Return the fluxes to their original values.
        for lc, f in zip(lcs, flux0):
            lc.flux[:] = f[:]

        return dict(
            peaks=peaks,
        )

    def save_to_cache(self, fn, response):
        try:
            os.makedirs(os.path.dirname(fn))
        except os.error:
            pass

        # Parse the peaks into a structured array.
        peaks = response["peaks"]
        if len(peaks):
            dtype = [(k, np.float64) for k in sorted(peaks[0].keys())]
            peaks = [tuple(peak[k] for k, _ in dtype) for peak in peaks]
            peaks = np.array(peaks, dtype=dtype)

        with h5py.File(fn, "w") as f:
            f.create_dataset("peaks", data=peaks, compression="gzip")

    def load_from_cache(self, fn):
        if os.path.exists(fn):
            with h5py.File(fn, "r") as f:
                try:
                    peaks = [dict((k, peak[k]) for k in peak.dtype.names)
                             for peak in f["peaks"]]
                    return dict(
                        peaks=peaks,
                    )
                except KeyError:
                    pass
        return None
