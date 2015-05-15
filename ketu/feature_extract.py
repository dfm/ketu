# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["FeatureExtract"]

import os
import h5py
import numpy as np
from functools import partial

from .pipeline import Pipeline


def check_orbits(p1, t1, p2, t2, tmn, tmx, tol):
    n1 = t1 + p1 * np.arange(np.floor((tmn-t1)/p1), np.ceil((tmx-t1)/p1))
    n1 = n1[(tmn <= n1) * (n1 <= tmx)]
    n2 = t2 + p2 * np.arange(np.floor((tmn-t2)/p2), np.ceil((tmx-t2)/p2))
    n2 = n2[(tmn <= n2) * (n2 <= tmx)]
    delta = np.fabs(n1[:, None] - n2[None, :])
    return max(len(n1), len(n2)) == np.sum(delta < tol)


def _time_warp(period, t0, t):
    return (t - t0 + 0.5 * period) % period - 0.5 * period


def _model(duration, depth, period, t0, t):
    r = np.zeros_like(t)
    r[np.fabs(_time_warp(period, t0, t)) < 0.5 * duration] = -depth
    return r


class FeatureExtract(Pipeline):

    cache_ext = ".h5"
    query_parameters = dict(
        lc_window_width=(3.0, False),
        bins=(32, False),
    )

    def get_result(self, query, parent_response):
        bins = int(query["bins"])
        dt = 0.5 * float(query["lc_window_width"])
        durations = parent_response.durations
        peaks = parent_response.peaks

        # Get the list of injections...
        injections = query.get("injections", [])
        dtype = [(k, np.float64)
                 for k in set(n for inj in injections for n in inj)] \
            + [("rec", bool)]

        # Choose the bin edges for the binned light curve.
        bin_edges = dt * np.linspace(-1, 1, bins+1) ** 2
        bin_edges[:len(bin_edges) // 2] *= -1

        # Choose the periods, phases, and durations for the peaks.
        periods = [p["period"] for p in peaks]
        t0s = [p["t0"] for p in peaks]
        durs = [p["duration"] for p in peaks]
        exacts = [-1 for i in range(len(periods))]

        # Get the features for the injections too.
        periods += [inj["period"] for inj in injections
                    for _ in range(len(durations))]
        t0s += [inj["t0"] for inj in injections for _ in range(len(durations))]
        durs += [d for inj in injections for d in durations]
        exacts += [i for i in range(len(injections))
                   for _ in range(len(durations))]

        # Build the feature matrix.
        dtype = [
            ("meta_starid", int), ("meta_isinj", bool), ("meta_isrec", bool),
            ("meta_isexact", bool),
            ("inj_period", float), ("inj_t0", float), ("inj_radius", float),
            ("inj_b", float), ("inj_e", float), ("inj_pomega", float),
            ("period", float), ("t0", float), ("lnlike", float),
            ("depth", float), ("depth_ivar", float), ("duration", float),
            ("mean", float), ("median", float), ("min", float),
            ("max", float), ("var", float), ("rvar", float)
        ] + (
            [("lc_mean_{0}".format(i), float) for i in range(bins)] +
            [("lc_median_{0}".format(i), float) for i in range(bins)] +
            [("lc_var_{0}".format(i), float) for i in range(bins)] +
            [("lc_rvar_{0}".format(i), float) for i in range(bins)]
        )
        features = np.empty(len(periods), dtype=dtype)
        inj_cols = ["period", "t0", "radius", "b", "e", "pomega"]

        # Loop over each peak and compute extract the features.
        for i, (period, t0, duration, exact) in enumerate(zip(periods, t0s,
                                                              durs, exacts)):
            # Loop over the light curves and compute the depth and
            # ln-likelihood.
            lcs = parent_response.model_light_curves
            lnlikes = np.empty(len(lcs))
            depths = np.empty(len(lcs))
            depth_ivars = np.empty(len(lcs))
            model = partial(_model, duration, 1.0, period, t0)
            for j, lc in enumerate(lcs):
                lnlikes[j], depths[j], depth_ivars[j] = lc.lnlike(model)
            depth_ivar = np.sum(depth_ivars)
            depth = np.sum(depths * depth_ivars) / depth_ivar
            lnlike = np.sum(lnlikes)
            lnlike -= 0.5 * np.sum((depth - depths) ** 2 * depth_ivars)
            features[i]["period"] = period
            features[i]["t0"] = t0
            features[i]["duration"] = duration
            features[i]["depth"] = depth
            features[i]["depth_ivar"] = depth_ivar
            features[i]["lnlike"] = lnlike

            # Loop over the light curves and compute the "corrected" fluxes.
            corr_lc = []
            tmn, tmx = np.inf, -np.inf
            for lc in lcs:
                tmn = min(tmn, lc.time.min())
                tmx = max(tmx, lc.time.max())
                t = _time_warp(period, t0, lc.time)
                m = np.abs(t) < dt
                mu = _model(duration, depth, period, t0, lc.time)
                bkg = lc.predict(lc.flux - mu)
                trans_num = np.round((lc.time[m] - t0) / period).astype(int)
                corr_lc += zip(t[m], (lc.flux - bkg)[m], lc.ferr[m],
                               trans_num)
            corr_lc = np.array(corr_lc, dtype=[
                ("time", float), ("flux", float), ("ferr", float),
                ("transit_num", int)
            ])

            # Bin the corrected light curve.
            inds = np.digitize(corr_lc["time"], bin_edges) - 1
            for j in range(len(bin_edges)-1):
                x = corr_lc["flux"][inds == j]
                if not len(x):
                    x = [0.0]
                features[i]["lc_mean_{0}".format(j)] = np.mean(x)
                k = "lc_var_{0}".format(j)
                features[i]["lc_var_{0}".format(j)] = np.var(x)
                mu = np.median(x)
                features[i]["lc_median_{0}".format(j)] = mu
                features[i]["lc_rvar_{0}".format(j)] = np.median((x - mu)**2)

            # Compute the rest of the features.
            x = corr_lc["flux"]
            features[i]["mean"] = np.mean(x)
            features[i]["var"] = np.var(x)
            mu = features[i]["median"] = np.median(x)
            features[i]["rvar"] = np.median((x-mu)**2)
            features[i]["min"] = np.min(x)
            features[i]["max"] = np.max(x)

            # Work out the rest of the meta data.
            features[i]["meta_isexact"] = exact >= 0
            features[i]["meta_starid"] = parent_response.starid
            for k in inj_cols:
                features[i]["inj_" + k] = np.nan
            if exact >= 0:
                isrec = False
                for peak in peaks:
                    features[i]["meta_isrec"] = False
                    if check_orbits(period, t0, peak["period"], peak["t0"],
                                    tmn, tmx, 0.5 * duration):
                        isrec = True
                features[i]["meta_isinj"] = True
                features[i]["meta_isrec"] = isrec

                inj = injections[exact]
                for k in inj_cols:
                    features[i]["inj_" + k] = inj[k]
            else:
                isinj = False
                for inj in injections:
                    if check_orbits(period, t0, inj["period"], inj["t0"], tmn,
                                    tmx, 0.5 * duration):
                        isinj = True
                        for k in inj_cols:
                            features[i]["inj_" + k] = inj[k]
                features[i]["meta_isinj"] = isinj
                features[i]["meta_isrec"] = True

        return dict(features=features)

    def save_to_cache(self, fn, response):
        try:
            os.makedirs(os.path.dirname(fn))
        except os.error:
            pass

        with h5py.File(fn, "w") as f:
            f.create_dataset("features", data=response["features"])

    def load_from_cache(self, fn):
        if os.path.exists(fn):
            with h5py.File(fn, "r") as f:
                return dict(features=f["features"][...])
        return None
