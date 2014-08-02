# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Hypotheses"]

import numpy as np
from scipy.spatial import cKDTree

from .pipeline import Pipeline
from ._compute import compute_hypotheses


class Hypotheses(Pipeline):

    def get_result(self, **kwargs):
        # Parse the input parameters.
        durations = np.atleast_1d(self.get_arg("durations", kwargs))
        depths = np.atleast_1d(self.get_arg("depths", kwargs))

        # Get the processed light curves.
        result = self.parent.query(**kwargs)
        lcs = result.pop("data")

        # Count the total number of hypotheses required.
        ntot = sum((len(lc.time) for lc in lcs))

        # Pre-allocate the results arrays.
        times = np.empty(ntot)
        grid = np.empty((ntot, len(depths), len(durations)))

        # Loop over the light curves and compute the model for each one.
        i = 0
        for lc in lcs:
            # Compute the "null" hypothesis likelihood (no transit).
            ll0 = lc.lnlike(lambda t: np.ones_like(t))
            l = len(lc.time)
            compute_hypotheses(lc.lnlike, ll0, lc.time, depths, durations,
                               grid[i:i+l])
            times[i:i+l] = lc.time
            i += l

        # Build a KDTree index.
        i = np.argsort(times)
        result["times"] = times[i]
        result["dll"] = grid[i]
        result["rng"] = (times.min(), times.max())
        result["depths"] = depths
        result["durations"] = durations
        return result
