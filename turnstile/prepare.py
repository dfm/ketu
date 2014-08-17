# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Prepare"]

import numpy as np
from .data import LightCurve
from .pipeline import Pipeline


class Prepare(Pipeline):

    defaults = {
        "normalize": True,
        "quality_flag": 0,
        "split_tol": 0.5,
        "min_chunk_length": 100,
        "max_chunk_length": None,
    }

    def get_result(self, **kwargs):
        normalize = self.get_arg("normalize", kwargs)
        quality_flag = self.get_arg("quality_flag", kwargs)
        split_tol = self.get_arg("split_tol", kwargs)
        min_chunk_length = self.get_arg("min_chunk_length", kwargs)
        max_chunk_length = self.get_arg("max_chunk_length", kwargs)

        # Download the data.
        result = self.parent.query(**kwargs)

        # Loop over the light curves and set them up.
        lcs = result.pop("data")
        result["data"] = []
        for lc in lcs:
            d = lc.read()
            t, f, fe = map(np.array, [d["TIME"], d["SAP_FLUX"],
                                      d["SAP_FLUX_ERR"]])
            q = d["SAP_QUALITY"]
            result["data"] += LightCurve(t, f, fe, q == quality_flag,
                                         normalize=normalize, meta=lc) \
                .autosplit(split_tol, max_chunk_length)

        # Get rid of light curves that are too short.
        result["data"] = [lc for lc in result["data"]
                          if len(lc.time) >= min_chunk_length]
        assert len(result["data"])

        return result
