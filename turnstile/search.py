# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Search"]

import numpy as np
from .pipeline import Pipeline
from ._compute import grid_search


class Search(Pipeline):

    def get_result(self, **kwargs):
        periods = self.get_arg("periods", kwargs)
        dt = self.get_arg("dt", kwargs)

        # Get the index of hypotheses.
        result = self.parent.query(**kwargs)

        # Run the grid search.
        time_spacing = result.pop("time_spacing")
        times = result.pop("times")
        mu = np.mean(times)
        times -= mu
        grid = grid_search(times.min(), times.max(), time_spacing,
                           result.pop("depths"), result.pop("d_ivars"),
                           result.pop("dll"),
                           np.atleast_1d(periods), dt)

        # Save and return the results.
        result["mean_time"] = mu
        result["periods"] = periods
        result["grid"] = grid
        result["dt"] = dt
        return result
