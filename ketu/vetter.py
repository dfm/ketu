# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["Vetter"]

import os
import h5py
import transit
import numpy as np
from scipy.optimize import minimize

from .pipeline import Pipeline


def _get_system(p):
    q1, q2 = p[:2]
    period, ror, duration = np.exp(p[2:5])
    t0, b = p[5:]
    q1 = max(min(q1, 1.-1e-4), 1e-4)
    q2 = max(min(q2, 1.-1e-4), 1e-4)
    s = transit.SimpleSystem(
        period=period, t0=t0, ror=ror, duration=duration, impact=b,
        q1=q1, q2=q2)
    return s


def _nll_transit(p, lcs):
    s = _get_system(p)
    ll = 0.0
    for lc in lcs:
        mod = 1e3*(s.light_curve(lc.time, texp=lc.texp)-1.0)
        r = lc.flux - mod
        ll += lc.lnlike_eval(r)
    return -ll


def _ln_evidence_basic(lcs):
    return sum(lc.lnlike_eval(lc.flux) for lc in lcs)


def _ln_evidence_box(lcs, period, duration, t0):
    hp = 0.5 * period
    hd = 0.5 * duration

    def model(t):
        mod = np.zeros_like(t)
        mod[np.fabs((t - t0 + hp) % period - hp) < hd] = -1.0
        return mod

    ll = 0.0
    for lc in lcs:
        l0, d, ivar = lc.lnlike(model)
        ll += l0 + lc.ll0 - 0.5*np.log(ivar)
    return ll


def _ln_evidence_transit(lcs, p):
    h = 1e-2
    x0 = np.array(p)
    lnhess = np.empty_like(x0)
    f0 = _nll_transit(x0, lcs)
    for i in range(len(x0)):
        x0[i] += h
        fp = _nll_transit(x0, lcs)
        x0[i] -= 2*h
        fm = _nll_transit(x0, lcs)
        x0[i] += h
        lnhess[i] = np.log(np.abs(fp - 2*f0 + fm))
    return -f0 + 0.5 * (2*len(x0)*np.log(h) - np.sum(lnhess))


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
            peak["lnZ_none"] = _ln_evidence_basic(lcs)
            peak["lnZ_box"] = _ln_evidence_box(lcs,
                                               peak["period"],
                                               peak["duration"],
                                               peak["t0"])

            # Fit the transit model.
            p0 = np.concatenate(([0.5, 0.5],
                                 np.log([peak["period"],
                                         np.sqrt(1e-3*peak["depth"]),
                                         peak["duration"]]),
                                 [peak["t0"], 0.0]))
            t0rng = peak["t0"]+float(query["t0_rng"])*np.array([-1, 1])
            lnprng = np.log([peak["period"]-query["period_rng"],
                             peak["period"]+query["period_rng"]])
            result = minimize(_nll_transit, p0, method="L-BFGS-B", args=(lcs,),
                              bounds=[(0+1e-4, 1-1e-4), (0+1e-4, 1-1e-4),
                                      lnprng, (None, None),
                                      (None, None), t0rng, (0, 1.5)])

            # Compute the transit evidence.
            x = result.x
            peak["transit_q1"] = x[0]
            peak["transit_q2"] = x[1]
            peak["transit_period"] = np.exp(x[2])
            peak["transit_ror"] = np.exp(x[3])
            peak["transit_duration"] = np.exp(x[4])
            peak["transit_t0"] = x[5]
            peak["transit_b"] = x[6]
            peak["lnZ_transit"] = _ln_evidence_transit(lcs, x)

            # Subtract the best fit transit model.
            s = _get_system(x)
            for lc in lcs:
                mod = 1e3*(s.light_curve(lc.time, texp=lc.texp)-1.0)
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
