# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["FeatureExtract"]

import os
import h5py
import numpy as np
from functools import partial

from .pipeline import Pipeline


def _time_warp(period, t0, t):
    return (t - t0 + 0.5 * period) % period - 0.5 * period


def _model(duration, depth, t):
    r = np.zeros_like(t)
    r[np.fabs(t) < 0.5 * duration] = -depth
    return r


class FeatureExtract(Pipeline):

    cache_ext = ".h5"
    query_parameters = dict(
        lc_window_width=(3.0, False),
        bins=(32, False),
        period_tol=(0.2, False),
        t0_tol=(None, False),
    )

    def get_result(self, query, parent_response):
        bins = float(query["bins"])
        dt = 0.5 * float(query["lc_window_width"])
        period_tol = float(query["period_tol"])
        t0_tol_0 = query["t0_tol"]

        # Get the list of injections...
        injections = query.get("injections", [])
        dtype = [(k, np.float64)
                 for k in set(n for inj in injections for n in inj)] \
            + [("rec", bool)]
        inj_rec = np.array([tuple([inj.get(k, np.nan) for k, _ in dtype[:-1]]
                                  + [False])
                            for inj in injections], dtype=dtype)

        # ... and the known KOIs.
        try:
            kic = parent_response.star
        except AttributeError:
            kic = None
            koi_rec = np.array([], dtype=dtype)
        else:
            kois = kic.kois
            dtype = [("id", np.float32), ("period", np.float64),
                     ("t0", np.float64), ("depth", np.float64),
                     ("rec", bool)]
            koi_rec = np.array([(float(k.kepoi_name[1:]), k.koi_period,
                                k.koi_time0bk % k.koi_period, k.koi_depth,
                                False)
                                for k in kois], dtype=dtype)

        # Choose the bin edges for the binned light curve.
        bin_edges = np.linspace(-dt, dt, float(bins+1), endpoint=True)

        # Loop over each peak and compute extract the features.
        peaks = []
        dtype = [
            ("transit_number", np.int32), ("time", np.float64),
            ("flux", np.float64), ("flux_err", np.float64),
        ]
        for peak in parent_response.peaks:
            peak = dict(peak)
            period = peak["period"]
            t0 = peak["t0"]
            duration = peak["duration"]
            depth = peak["depth"]
            model = partial(_model, duration, depth)

            # Loop over the light curves and compute the corrected, folded
            # transit.
            corr_lc = []
            for lc in parent_response.model_light_curves:
                # Compute the predicted background at the ML depth.
                t = _time_warp(period, t0, lc.time)
                m = np.fabs(t) < dt
                if not np.any(m):
                    continue
                bkg = lc.predict(y=lc.flux - model(t))

                # Compute the transit number for each point.
                trans_num = np.round((lc.time[m] - t0) / period).astype(int)

                # Update the corrected light curve array.
                corr_lc += zip(trans_num, t[m], lc.flux[m]-bkg[m], lc.ferr[m])

            # Add the corrected light curve the peak object.
            peak["corr_lc"] = corr_lc = np.array(corr_lc, dtype=dtype)

            # Count the number of datasets in/out of transit.
            m = np.fabs(corr_lc["time"]) < 0.25 * duration
            peak["num_in"] = len(set(corr_lc["transit_number"][m]))
            peak["num_out"] = len(set(corr_lc["transit_number"]))

            # Bin the light curve.
            binned_lc = np.zeros(bins, dtype=np.float64)
            binned_lc_err = np.zeros_like(binned_lc)

            i = np.digitize(corr_lc["time"], bin_edges) - 1
            m = (0 <= i) * (i < bins)
            w = 1.0 / (corr_lc["flux_err"][m]) ** 2
            binned_lc[i[m]] += corr_lc["flux"][m] * w
            binned_lc_err[i[m]] += w

            m = binned_lc_err > 0.0
            binned_lc[m] /= binned_lc_err[m]
            binned_lc_err[m] = 1.0 / np.sqrt(binned_lc_err[m])

            peak["bin_lc"] = np.array(zip(
                0.5 * (bin_edges[:-1] + bin_edges[1:]),
                binned_lc, binned_lc_err), dtype=dtype[1:])

            # Check if this is an injection.
            peak["is_injection"] = False
            t0_tol = 1.5 * duration if t0_tol_0 is None else t0_tol_0
            for i, row in enumerate(inj_rec):
                if (np.fabs(row["period"] - period) > period_tol
                        or np.fabs(row["t0"] - t0) > t0_tol):
                    continue
                inj_rec["rec"][i] = True
                peak["is_injection"] = True
                for k in row.dtype.names:
                    peak["injected_{0}".format(k)] = row[k]

            # ... or a KOI.
            peak["is_koi"] = False
            for i, row in enumerate(koi_rec):
                if (np.fabs(row["period"] - period) > period_tol
                        or np.fabs(row["t0"] - t0) > t0_tol):
                    continue
                koi_rec["rec"][i] = True
                peak["is_koi"] = True
                peak["koi_id"] = row["id"]
                for k in row.dtype.names:
                    peak["koi_{0}".format(k)] = row[k]

            peaks.append(peak)

        results = dict(inj_rec=inj_rec, koi_rec=koi_rec, features=peaks)
        if kic is not None:
            results = dict(results,
                           kic_kepmag=kic.kic_kepmag,
                           kic_teff=kic.huber.Teff,
                           kic_logg=kic.huber["log(g)"])
        else:
            try:
                epic = parent_response.epic
                results = dict(results,
                               kic_kepmag=epic.kp,
                               kic_teff=0.0,
                               kic_logg=0.0)
            except AttributeError:
                results = dict(results,
                               kic_kepmag=0.0,
                               kic_teff=0.0,
                               kic_logg=0.0)
        return results

    def save_to_cache(self, fn, response):
        try:
            os.makedirs(os.path.dirname(fn))
        except os.error:
            pass

        with h5py.File(fn, "w") as f:
            f.attrs["kic_kepmag"] = response["kic_kepmag"]
            f.attrs["kic_teff"] = response["kic_teff"]
            f.attrs["kic_logg"] = response["kic_logg"]
            f.create_dataset("inj_rec", data=response["inj_rec"],
                             compression="gzip")
            f.create_dataset("koi_rec", data=response["koi_rec"],
                             compression="gzip")
            for i, peak in enumerate(response["features"]):
                g = f.create_group("peak_{0:04d}".format(i))
                g.create_dataset("corr_lc", data=peak["corr_lc"],
                                 compression="gzip")
                g.create_dataset("bin_lc", data=peak["bin_lc"],
                                 compression="gzip")
                for k, v in peak.iteritems():
                    if k in ["corr_lc", "bin_lc"]:
                        continue
                    g.attrs[k] = v

    def load_from_cache(self, fn):
        if os.path.exists(fn):
            with h5py.File(fn, "r") as f:
                try:
                    inj_rec = f["inj_rec"][...]
                    koi_rec = f["koi_rec"][...]
                    peaks = []
                    for nm in f:
                        if not nm.startswith("peak_"):
                            continue
                        g = f[nm]
                        peak = dict(g.attrs)
                        peak["corr_lc"] = g["corr_lc"][...]
                        peak["bin_lc"] = g["bin_lc"][...]
                        peaks.append(peak)
                    return dict(
                        features=peaks, inj_rec=inj_rec, koi_rec=koi_rec,
                        kic_kepmag=f.attrs["kic_kepmag"],
                        kic_teff=f.attrs["kic_teff"],
                        kic_logg=f.attrs["kic_logg"],
                    )
                except KeyError:
                    pass
        return None
