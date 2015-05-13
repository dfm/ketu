# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["IterativeTwoDSearch"]

import os
import h5py
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from .two_d_search import TwoDSearch
from ._grid_search import grid_search

class IterativeTwoDSearch(TwoDSearch):

    cache_ext = ".h5"
    query_parameters = dict(
        min_period=(None, True),
        max_period=(None, True),
        delta_log_period=(None, False),
        dt=(None, False),
        alpha=(None, False),
        npeaks=(3, False),
        mask_frac=(2.0, False),
        min_points=(500, False),
    )

    def get_alpha(self, query, parent_response):
        a = query.get("alpha", None)
        if a is not None:
            return float(a)
        lc = parent_response.model_light_curves[0]
        n = len(lc.time)
        k = parent_response.nbasis
        return k * np.log(n)

    def get_result(self, query, parent_response):
        periods = self.get_period_grid(query, parent_response)
        dt = self.get_offset_spacing(query, parent_response)
        alpha = self.get_alpha(query, parent_response)

        # Get the parameters of the time grid from the 1-d search.
        time_spacing = parent_response.time_spacing
        mean_time = parent_response.mean_time_1d
        tmin = parent_response.min_time_1d - mean_time
        tmax = parent_response.max_time_1d - mean_time
        time_grid = np.arange(0, tmax-tmin, time_spacing)

        # Get the results of the 1-d search.
        depth_1d = np.array(parent_response.depth_1d)
        depth_ivar_1d = np.array(parent_response.depth_ivar_1d)
        dll_1d = np.array(parent_response.dll_1d)

        # Find the peaks.
        peaks = []
        for _ in range(query["npeaks"]):
            # Run a 2D search.
            results = grid_search(alpha, tmin, tmax, time_spacing, depth_1d,
                                  depth_ivar_1d, dll_1d, periods, dt)
            (t0_2d, phic_same, phic_same_2, phic_variable, depth_2d,
             depth_ivar_2d) = results

            # Profile over duration.
            inds = np.arange(len(phic_same)), np.argmax(phic_same, axis=1)
            t0_2d = t0_2d[inds]
            depth_2d = depth_2d[inds]
            depth_ivar_2d = depth_ivar_2d[inds]

            # Find the top peak.
            s2n = depth_2d * np.sqrt(depth_ivar_2d)
            top_peak = np.argmax(s2n)
            p, t0 = periods[top_peak], t0_2d[top_peak]
            duration = query["durations"][inds[1][top_peak]]

            # Save the peak.
            peaks.append(dict(
                period=p, t0=(t0 + tmin + mean_time) % p,
                duration=duration,
                depth=depth_2d[top_peak],
                depth_ivar=depth_ivar_2d[top_peak],
                s2n=s2n[top_peak],
            ))

            # Mask out these transits.
            m = (np.abs((time_grid-t0+0.5*p) % p-0.5*p)
                 < query["mask_frac"]*duration)
            depth_1d[m] = 0.0
            depth_ivar_1d[m] = 0.0
            dll_1d[m] = 0.0
            if np.sum(np.any(depth_ivar_1d>0.0, axis=1)) < query["min_points"]:
                break

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
