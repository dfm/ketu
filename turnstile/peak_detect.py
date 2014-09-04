# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["PeakDetect"]

import os
import h5py
import logging
import numpy as np
from itertools import izip

from .pipeline import Pipeline


def count_overlapping_transits(p1, t1, p2, t2, tmn, tmx, tol):
    n1 = t1 + p1 * np.arange(np.floor((tmn-t1)/p1), np.ceil((tmx-t1)/p1))
    n1 = n1[(tmn <= n1) * (n1 <= tmx)]
    n2 = t2 + p2 * np.arange(np.floor((tmn-t2)/p2), np.ceil((tmx-t2)/p2))
    n2 = n2[(tmn <= n2) * (n2 <= tmx)]
    delta = np.fabs(n1[:, None] - n2[None, :])
    return np.sum(delta < tol)


class PeakDetect(Pipeline):

    cache_ext = ".h5"
    query_parameters = dict(
        number_of_peaks=(10, False),
        overlap_tol=(0.1, False),
        max_overlap=(0, False),
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
        ttot = tmx - tmn
        periods = parent_response.period_2d

        # Do a linear fit of the form: sqrt(N / P)
        bkg = np.vstack((
            np.sqrt(np.ceil(ttot / periods)) / np.sqrt(periods),
            np.ones_like(periods),
        ))
        w = np.linalg.solve(np.dot(bkg, bkg.T), np.dot(bkg, phic_same))
        bkg = np.dot(w, bkg)
        z = phic_same - bkg

        # Compute the RMS noise in this object.
        rms = np.sqrt(np.median(z ** 2))

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
                                                          duration[j]) + 0.1))
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
            phic_variable=phic_variable[i], scaled_phic_same=z[i],
            depth=depth[i], depth_ivar=depth_ivar[i],
            depth_s2n=depth[i]*np.sqrt(depth_ivar[i]), rms=rms,
            duration=duration[i],
        ) for i in accepted_peaks]

        return dict(
            periods=periods,
            scaled_phic_same=z,
            rms=rms,
            peaks=peaks,
        )

    # def save_to_cache(self, fn, response):
    #     try:
    #         os.makedirs(os.path.dirname(fn))
    #     except os.error:
    #         pass
    #     with h5py.File(fn, "w") as f:
    #         f.create_dataset("period_2d", data=response["period_2d"],
    #                          compression="gzip")
    #         f.create_dataset("t0_2d", data=response["t0_2d"],
    #                          compression="gzip")
    #         f.create_dataset("phic_same", data=response["phic_same"],
    #                          compression="gzip")
    #         f.create_dataset("phic_variable", data=response["phic_variable"],
    #                          compression="gzip")
    #         f.create_dataset("depth_2d", data=response["depth_2d"],
    #                          compression="gzip")
    #         f.create_dataset("depth_ivar_2d", data=response["depth_ivar_2d"],
    #                          compression="gzip")

    # def load_from_cache(self, fn):
    #     if os.path.exists(fn):
    #         with h5py.File(fn, "r") as f:
    #             try:
    #                 return dict(
    #                     period_2d=f["period_2d"][...],
    #                     t0_2d=f["t0_2d"][...],
    #                     phic_same=f["phic_same"][...],
    #                     phic_variable=f["phic_variable"][...],
    #                     depth_2d=f["depth_2d"][...],
    #                     depth_ivar_2d=f["depth_ivar_2d"][...],
    #                 )
    #             except KeyError:
    #                 pass
    #     return None
