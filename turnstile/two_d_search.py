# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["TwoDSearch"]

import os
import h5py
import numpy as np

from .pipeline import Pipeline
from ._grid_search import grid_search


class TwoDSearch(Pipeline):

    cache_ext = ".h5"
    query_parameters = dict(
        min_period=(None, True),
        max_period=(None, True),
        delta_log_period=(None, False),
        dt=(None, False),
        alpha=(None, False)
    )

    def get_period_grid(self, query, parent_response):
        delta_log_period = query.get("delta_log_period", None)
        if delta_log_period is None:
            ttot = parent_response.max_time_1d - parent_response.min_time_1d
            delta_log_period = 0.2*np.min(parent_response.durations)/ttot
        lpmin, lpmax = np.log(query["min_period"]), np.log(query["max_period"])
        return np.exp(np.arange(lpmin, lpmax, delta_log_period))

    def get_offset_spacing(self, query, parent_response):
        dt = query.get("dt", None)
        if dt is None:
            dt = 0.5 * np.min(parent_response.durations)
        return float(dt)

    def get_alpha(self, query, parent_response):
        a = query.get("alpha", None)
        if a is not None:
            return float(a)
        lc = parent_response.model_light_curves[0]
        n = len(lc.time)
        k = len(lc.basis) + 1
        return k * np.log(n) - np.log(2 * np.pi)

    def get_result(self, query, parent_response):
        periods = self.get_period_grid(query, parent_response)
        dt = self.get_offset_spacing(query, parent_response)
        alpha = self.get_alpha(query, parent_response)

        # Get the parameters of the time grid from the 1-d search.
        time_spacing = parent_response.time_spacing
        mean_time = parent_response.mean_time_1d
        tmin = parent_response.min_time_1d - mean_time
        tmax = parent_response.max_time_1d - mean_time

        # Get the results of the 1-d search.
        depth_1d = parent_response.depth_1d
        depth_ivar_1d = parent_response.depth_ivar_1d
        dll_1d = parent_response.dll_1d

        results = grid_search(alpha, tmin, tmax, time_spacing, depth_1d,
                              depth_ivar_1d, dll_1d, periods, dt)
        t0_2d, phic_same, phic_same_2, phic_variable, depth_2d, depth_ivar_2d \
            = results

        return dict(
            period_2d=periods,
            t0_2d=(t0_2d + tmin + mean_time) % periods[:, None],
            phic_same=phic_same, phic_same_2=phic_same_2,
            phic_variable=phic_variable,
            depth_2d=depth_2d, depth_ivar_2d=depth_ivar_2d,
        )

    def save_to_cache(self, fn, response):
        try:
            os.makedirs(os.path.dirname(fn))
        except os.error:
            pass
        with h5py.File(fn, "w") as f:
            f.create_dataset("period_2d", data=response["period_2d"],
                             compression="gzip")
            f.create_dataset("t0_2d", data=response["t0_2d"],
                             compression="gzip")
            f.create_dataset("phic_same", data=response["phic_same"],
                             compression="gzip")
            f.create_dataset("phic_same_2", data=response["phic_same_2"],
                             compression="gzip")
            f.create_dataset("phic_variable", data=response["phic_variable"],
                             compression="gzip")
            f.create_dataset("depth_2d", data=response["depth_2d"],
                             compression="gzip")
            f.create_dataset("depth_ivar_2d", data=response["depth_ivar_2d"],
                             compression="gzip")

    def load_from_cache(self, fn):
        if os.path.exists(fn):
            with h5py.File(fn, "r") as f:
                try:
                    return dict(
                        period_2d=f["period_2d"][...],
                        t0_2d=f["t0_2d"][...],
                        phic_same=f["phic_same"][...],
                        phic_same_2=f["phic_same_2"][...],
                        phic_variable=f["phic_variable"][...],
                        depth_2d=f["depth_2d"][...],
                        depth_ivar_2d=f["depth_ivar_2d"][...],
                    )
                except KeyError:
                    pass
        return None
