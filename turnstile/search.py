# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Search"]

import numpy as np
from .pipeline import Pipeline
from ._compute import grid_search


class Search(Pipeline):

    defaults = {"tol": 0.1}

    def get_result(self, **kwargs):
        periods = self.get_arg("periods", kwargs)
        dt = self.get_arg("dt", kwargs)
        tol = self.get_arg("tol", kwargs)

        # Get the index of hypotheses.
        result = self.parent.query(**kwargs)

        # Run the grid search.
        times = result.pop("times")
        grid = grid_search(times, result.pop("dll"), np.atleast_1d(periods),
                           dt, times.min(), times.max(), tol)

        # Save and return the results.
        result["periods"] = periods
        result["grid"] = grid
        result["dt"] = dt
        return result
