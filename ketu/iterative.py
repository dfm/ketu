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
        k = 150
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

        # The array for peaks.
        peaks = []
        periodograms = []
        depths = []
        depth_ivars = []

        for _ in range(query["npeaks"]):
            # Run a 2D search.
            results = grid_search(alpha, tmin, tmax, time_spacing, depth_1d,
                                  depth_ivar_1d, dll_1d, periods, dt)
            (t0_2d, phic_same, phic_same_2, phic_variable, depth_2d,
             depth_ivar_2d) = results

            # Profile over duration.
            inds = np.arange(len(phic_same)), np.argmax(phic_same, axis=1)
            t0_2d = t0_2d[inds]
            phic_same = phic_same[inds]
            phic_same_2 = phic_same_2[inds]
            phic_variable = phic_variable[inds]
            depth_2d = depth_2d[inds]
            depth_ivar_2d = depth_ivar_2d[inds]

            # Fit out the 1/period trend.
            m = np.isfinite(phic_same)
            A = np.vander(1.0 / periods[m], 2)
            ATA = np.dot(A.T, A)
            w = np.linalg.solve(ATA, np.dot(A.T, phic_same[m]))
            z = -np.inf + np.zeros_like(phic_same)
            z[m] = phic_same[m] - np.dot(A, w)

            # Find the top peak.
            s2n = depth_2d * np.sqrt(depth_ivar_2d)
            mean_s2n = np.mean(s2n)
            med_s2n = np.median(s2n)
            rms_s2n = np.std(s2n)
            rrms_s2n = np.sqrt(np.median((s2n - med_s2n) ** 2))
            top_peak = np.argmax(s2n)
            periodograms.append(s2n)
            depths.append(depth_2d)
            depth_ivars.append(depth_ivar_2d)

            # Extract the information about the peak.
            p, t0 = periods[top_peak], t0_2d[top_peak]
            duration = query["durations"][inds[1][top_peak]]
            rms = np.std(z[m])
            rrms = np.sqrt(np.median((z[m] - np.median(z[m])) ** 2))

            # Save the peak.
            peaks.append(dict(
                period=p, t0=(t0 + tmin + mean_time) % p,
                duration=duration,
                depth=depth_2d[top_peak],
                depth_ivar=depth_ivar_2d[top_peak],
                s2n=s2n[top_peak],
                median_s2n=med_s2n, mean_s2n=mean_s2n,
                rms_s2n=rms_s2n, rrms_s2n=rrms_s2n,
                phic_same=phic_same[top_peak],
                phic_variable=phic_variable[top_peak],
                phic_norm=z[top_peak],
                mean_phic_norm=np.mean(z[m]),
                median_phic_norm=np.median(z[m]),
                rms_phic_norm=rms, rrms_phic_norm=rrms,
                delta_phic=phic_same[top_peak] - phic_same_2[top_peak],
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
            period_2d=periods,
            periodograms=np.array(periodograms),
            depths=np.array(depths),
            depth_ivars=np.array(depth_ivars),
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
            f.create_dataset("period_2d", data=response["period_2d"],
                             compression="gzip")
            f.create_dataset("periodograms", data=response["periodograms"],
                             compression="gzip")
            f.create_dataset("depths", data=response["depths"],
                             compression="gzip")
            f.create_dataset("depth_ivars", data=response["depth_ivars"],
                             compression="gzip")
            f.create_dataset("peaks", data=peaks, compression="gzip")

    def load_from_cache(self, fn):
        if os.path.exists(fn):
            with h5py.File(fn, "r") as f:
                try:
                    peaks = [dict((k, peak[k]) for k in peak.dtype.names)
                             for peak in f["peaks"]]
                    return dict(
                        period_2d=f["period_2d"][...],
                        periodograms=f["periodograms"][...],
                        depths=f["depths"][...],
                        depth_ivars=f["depth_ivars"][...],
                        peaks=peaks,
                    )
                except KeyError:
                    pass
        return None
