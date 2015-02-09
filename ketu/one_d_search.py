# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["OneDSearch"]

import os
import h5py
import numpy as np

from .pipeline import Pipeline
from ._compute import compute_hypotheses


class OneDSearch(Pipeline):

    cache_ext = ".h5"
    query_parameters = dict(
        durations=(None, True),
        time_spacing=(0.05, False),
    )

    def get_result(self, query, parent_response):
        # Parse the input parameters.
        durations = np.atleast_1d(query["durations"])
        dt = np.atleast_1d(query["time_spacing"])

        # Get the processed light curves.
        lcs = parent_response.model_light_curves

        # Build the time grid.
        tmin = min(map(lambda l: min(l.time), lcs))
        tmax = max(map(lambda l: max(l.time), lcs))
        time_grid = np.arange(tmin, tmax, dt)

        # Allocate the output arrays.
        dll_grid = np.zeros((len(time_grid), len(durations)))
        depth_grid = np.zeros_like(dll_grid)
        depth_ivar_grid = np.zeros_like(dll_grid)

        # Loop over the light curves and compute the model for each one.
        for lc in lcs:
            # Find the times that are in this light curve.
            m = (lc.time.min() <= time_grid) * (time_grid <= lc.time.max())
            if not np.any(m):
                continue

            # Compute the grid of hypotheses.
            i = np.arange(len(time_grid))[m]
            imn, imx = i.min(), i.max()
            compute_hypotheses(lc.lnlike, time_grid[imn:imx], durations,
                               depth_grid[imn:imx], depth_ivar_grid[imn:imx],
                               dll_grid[imn:imx])

        return dict(
            min_time_1d=tmin,
            max_time_1d=tmax,
            mean_time_1d=0.5 * (tmin + tmax),
            dll_1d=dll_grid,
            depth_1d=depth_grid,
            depth_ivar_1d=depth_ivar_grid,
        )

    def save_to_cache(self, fn, response):
        try:
            os.makedirs(os.path.dirname(fn))
        except os.error:
            pass
        with h5py.File(fn, "w") as f:
            f.attrs["min_time_1d"] = response["min_time_1d"]
            f.attrs["max_time_1d"] = response["max_time_1d"]
            f.attrs["mean_time_1d"] = response["mean_time_1d"]
            f.create_dataset("dll_1d", data=response["dll_1d"],
                             compression="gzip")
            f.create_dataset("depth_1d", data=response["depth_1d"],
                             compression="gzip")
            f.create_dataset("depth_ivar_1d", data=response["depth_ivar_1d"],
                             compression="gzip")

    def load_from_cache(self, fn):
        if os.path.exists(fn):
            with h5py.File(fn, "r") as f:
                try:
                    return dict(
                        min_time_1d=f.attrs["min_time_1d"],
                        max_time_1d=f.attrs["max_time_1d"],
                        mean_time_1d=f.attrs["mean_time_1d"],
                        dll_1d=f["dll_1d"][...],
                        depth_1d=f["depth_1d"][...],
                        depth_ivar_1d=f["depth_ivar_1d"][...],
                    )
                except KeyError:
                    pass
        return None
