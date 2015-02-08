# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["PeakDetect"]

import os
import h5py
import logging
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from .pipeline import Pipeline


def count_overlapping_transits(p1, t1, p2, t2, tmn, tmx, tol):
    n1 = t1 + p1 * np.arange(np.floor((tmn-t1)/p1), np.ceil((tmx-t1)/p1))
    n1 = n1[(tmn <= n1) * (n1 <= tmx)]
    n2 = t2 + p2 * np.arange(np.floor((tmn-t2)/p2), np.ceil((tmx-t2)/p2))
    n2 = n2[(tmn <= n2) * (n2 <= tmx)]
    delta = np.fabs(n1[:, None] - n2[None, :])
    return np.sum(delta < tol)


def compute_curvature(z, p, i):
    a = np.vander(p[i-1:i+2], 3)
    return np.linalg.solve(a, z[i-1:i+2])[0]


class PeakDetect(Pipeline):

    cache_ext = ".h5"
    query_parameters = dict(
        number_of_peaks=(20, False),
        overlap_tol=(0.1, False),
        max_overlap=(0, False),
        smooth=(None, False),
    )

    def get_result(self, query, parent_response):
        number_of_peaks = int(query["number_of_peaks"])
        overlap_tol = float(query["overlap_tol"])
        max_overlap = int(query["max_overlap"])

        # First profile over duration.
        phic_same = parent_response.phic_same
        dur_ind = np.arange(len(phic_same)), np.argmax(phic_same, axis=1)
        phic_same = phic_same[dur_ind]

        phic_same_2 = parent_response.phic_same_2[dur_ind]
        phic_variable = parent_response.phic_variable[dur_ind]
        t0s = parent_response.t0_2d[dur_ind]
        depth = parent_response.depth_2d[dur_ind]
        depth_ivar = parent_response.depth_ivar_2d[dur_ind]
        duration = np.atleast_1d(parent_response.durations)[dur_ind[1]]

        # Start by fitting out the background level.
        tmx, tmn = parent_response.max_time_1d, parent_response.min_time_1d
        periods = parent_response.period_2d

        # Now we'll fit out the 1/period trend.
        m = np.isfinite(phic_same)
        A = np.vander(1.0 / periods[m], 2)
        ATA = np.dot(A.T, A)
        w = np.linalg.solve(ATA, np.dot(A.T, phic_same[m]))
        z = -np.inf + np.zeros_like(phic_same)
        z[m] = phic_same[m] - np.dot(A, w)
        if query["smooth"] is not None:
            z[m] = gaussian_filter(z[m], query["smooth"])

        # Compute the RMS noise in this object.
        rms = np.sqrt(np.median(z[m] ** 2))

        # Find the peaks.
        peak_inds = (z[1:-1] > z[:-2]) * (z[1:-1] > z[2:])
        peak_inds = np.arange(1, len(z)-1)[peak_inds]

        # Sort them by the PHIC.
        peak_inds = peak_inds[np.argsort(z[peak_inds])][::-1]

        # Loop over the peaks and count the number of overlapping transits.
        accepted_peaks = np.empty(number_of_peaks, dtype=int)
        accepted_peaks[0] = peak_inds[0]
        npeak = 1
        for i in peak_inds[1:]:
            p2, t2 = periods[i], t0s[i]
            n = 0
            for j in accepted_peaks[:npeak]:
                p1, t1 = periods[j], t0s[j]
                n = max(n, count_overlapping_transits(p1, t1, p2, t2, tmn, tmx,
                                                      max(duration[i],
                                                          duration[j])
                                                      + overlap_tol))
            if n <= max_overlap:
                accepted_peaks[npeak] = i
                npeak += 1
                if npeak >= number_of_peaks:
                    break

        if npeak < number_of_peaks:
            logging.warn("Not enough peaks were found")
            accepted_peaks = accepted_peaks[:npeak]

        peaks = [dict(
            period=periods[i], t0=t0s[i], phic_same=phic_same[i],
            delta_phic=phic_same[i] - phic_same_2[i],
            curve_phic=compute_curvature(z, periods, i),
            phic_variable=phic_variable[i], phic_norm=z[i],
            depth=depth[i], depth_ivar=depth_ivar[i],
            depth_s2n=depth[i]*np.sqrt(depth_ivar[i]), rms=rms,
            duration=duration[i],
        ) for i in accepted_peaks]

        return dict(
            periods=periods,
            phic_scale=z,
            rms=rms,
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
            f.attrs["rms"] = response["rms"]
            f.create_dataset("periods", data=response["periods"],
                             compression="gzip")
            f.create_dataset("phic_scale", data=response["phic_scale"],
                             compression="gzip")
            f.create_dataset("peaks", data=peaks, compression="gzip")

    def load_from_cache(self, fn):
        if os.path.exists(fn):
            with h5py.File(fn, "r") as f:
                try:
                    peaks = [dict((k, peak[k]) for k in peak.dtype.names)
                             for peak in f["peaks"]]
                    return dict(
                        periods=f["periods"][...],
                        phic_scale=f["phic_scale"][...],
                        rms=f.attrs["rms"],
                        peaks=peaks,
                    )
                except KeyError:
                    pass
        return None
