# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["BoxModel", "PeriodicBoxModel"]

import numpy as np
from .pipeline import Pipeline


class BoxModel(Pipeline):

    def get_result(self, **kwargs):
        result = self.parent.query(**kwargs)
        result["data"] = map(LCWrapper, result.pop("data"))
        return result
