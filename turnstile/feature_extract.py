# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["FeatureExtract"]

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
        lc_window_width=(2.0, False),
        bins=(20, False),
    )

    def get_result(self, query, parent_response):
        bins = float(query["bins"])
        dt = 0.5 * float(query["lc_window_width"])

        # Choose the bin edges for the binned light curve.
        bin_edges = np.linspace(-dt, dt, float(bins+1), endpoint=True)

        # Loop over each peak and compute extract the features.
        peaks = []
        dtype = np.dtype([
            ("transit_number", np.int32), ("time", np.float64),
            ("flux", np.float64), ("flux_uncert", np.float64),
        ])
        for peak in parent_response.peaks:
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
                mean, bkg = lc.predict(y=lc.flux - model(t))

                # Compute the transit number for each point.
                trans_num = np.round((lc.time[m] - t0) / period).astype(int)

                # Update the
                corr_lc += zip(trans_num, t[m], lc.flux[m]-bkg[m], lc.ferr[m])

            # Add the corrected light curve the peak object.
            peak["corr_lc"] = corr_lc = np.array(corr_lc, dtype=dtype)

            # Bin the light curve.
            binned_lc = np.zeros(bins, dtype=np.float64)
            binned_lc_err = np.zeros_like(binned_lc)

            i = np.digitize(corr_lc["time"], bin_edges) - 1
            m = (0 <= i) * (i < bins)
            w = 1.0 / (corr_lc["flux_uncert"][m]) ** 2
            binned_lc[i[m]] += corr_lc["flux"][m]
            binned_lc[i[m]] += corr_lc["flux"][m] * w
            binned_lc_err[i[m]] += w
            binned_lc /= binned_lc_err
            binned_lc_err = 1.0 / np.sqrt(binned_lc_err)

            peak["bin_edges"] = bin_edges
            peak["bin_lc"] = binned_lc
            peak["bin_lc_uncert"] = binned_lc_err
            peaks.append(peak)

        return dict(features=peaks)

    def save_to_cache(self, fn, response):
        assert 0

    #     try:
    #         os.makedirs(os.path.dirname(fn))
    #     except os.error:
    #         pass

    #     # Parse the peaks into a structured array.
    #     peaks = response["peaks"]
    #     if len(peaks):
    #         dtype = [(k, np.float64) for k in sorted(peaks[0].keys())]
    #         peaks = [tuple(peak[k] for k, _ in dtype) for peak in peaks]
    #         peaks = np.array(peaks, dtype=dtype)

    #     with h5py.File(fn, "w") as f:
    #         f.attrs["rms"] = response["rms"]
    #         f.create_dataset("periods", data=response["periods"],
    #                          compression="gzip")
    #         f.create_dataset("scaled_phic_same",
    #                          data=response["scaled_phic_same"],
    #                          compression="gzip")
    #         f.create_dataset("peaks", data=peaks, compression="gzip")

    # def load_from_cache(self, fn):
    #     if os.path.exists(fn):
    #         with h5py.File(fn, "r") as f:
    #             try:
    #                 peaks = [dict((k, peak[k]) for k in peak.dtype.names)
    #                          for peak in f["peaks"]]
    #                 return dict(
    #                     periods=f["periods"][...],
    #                     scaled_phic_same=f["scaled_phic_same"][...],
    #                     rms=f.attrs["rms"],
    #                     peaks=peaks,
    #                 )
    #             except KeyError:
    #                 pass
    #     return None
