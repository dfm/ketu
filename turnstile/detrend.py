# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Detrend"]

from .pipeline import Pipeline


class Detrend(Pipeline):

    defaults = {
        "detrend_window": 4.0,
    }

    def get_result(self, **kwargs):
        detrend_window = self.get_arg("detrend_window", kwargs)
        result = self.parent.query(**kwargs)
        [lc.median_detrend(detrend_window) for lc in result["data"]]
        return result
