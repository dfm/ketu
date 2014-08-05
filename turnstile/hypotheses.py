# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Hypotheses"]

import numpy as np

from .pipeline import Pipeline
from ._compute import compute_hypotheses


class Hypotheses(Pipeline):

    defaults = dict(
        time_spacing=0.02,
    )

    def get_result(self, **kwargs):
        # Parse the input parameters.
        durations = np.atleast_1d(self.get_arg("durations", kwargs))
        depths = np.atleast_1d(self.get_arg("depths", kwargs))
        dt = np.atleast_1d(self.get_arg("time_spacing", kwargs))

        # Get the processed light curves.
        result = self.parent.query(**kwargs)
        lcs = result.pop("data")

        # Pre-allocate the results arrays.
        times = np.concatenate([lc.time for lc in lcs])
        times = np.arange(times.min(), times.max(), dt)
        grid = np.zeros((len(times), len(depths), len(durations)))

        # Loop over the light curves and compute the model for each one.
        for lc in lcs:
            # Compute the "null" hypothesis likelihood (no transit).
            ll0 = lc.lnlike(lambda t: np.ones_like(t))

            # Find the times that are in this light curve.
            m = (lc.time.min() <= times) * (times <= lc.time.max())
            if not np.any(m):
                continue

            # Compute the grid of hypotheses.
            i = np.arange(len(times))[m]
            compute_hypotheses(lc.lnlike, ll0, times[i.min():i.max()],
                               depths, durations, grid[i.min():i.max()])

        # Build a KDTree index.
        result["times"] = times
        result["time_spacing"] = dt
        result["dll"] = grid
        result["rng"] = (times.min(), times.max())
        result["depths"] = depths
        result["durations"] = durations
        return result
