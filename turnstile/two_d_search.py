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
        dt=(0.1, False),
        alpha=(np.log(60000)-np.log(2*np.pi), False)
    )

    def get_period_grid(self, query, parent_response):
        delta_log_period = query.get("delta_log_period", None)
        if delta_log_period is None:
            delta_log_period = 0.1*np.min(parent_response.durations)/(4.1*365.)
        lpmin, lpmax = np.log(query["min_period"]), np.log(query["max_period"])
        return np.exp(np.arange(lpmin, lpmax, delta_log_period))

    def get_result(self, query, parent_response):
        periods = self.get_period_grid(query, parent_response)
        dt = float(query["dt"])
        alpha = float(query["alpha"])

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
        phic_same, phic_variable, depth_2d, depth_ivar_2d = results

        return dict(
            period_2d=periods,
            t0_2d=np.arange(0, max(periods), dt) + tmin + mean_time,
            phic_same=phic_same, phic_variable=phic_variable,
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
                        phic_variable=f["phic_variable"][...],
                        depth_2d=f["depth_2d"][...],
                        depth_ivar_2d=f["depth_ivar_2d"][...],
                    )
                except KeyError:
                    pass
        return None
