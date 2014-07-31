# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Prepare"]

from .data import LightCurve
from .pipeline import Pipeline


class Prepare(Pipeline):

    defaults = {
        "normalize": True,
        "quality_flag": 0,
        "split_tol": 0.5,
        "max_chunk_length": None,
    }

    def get_result(self, **kwargs):
        normalize = self.get_arg("normalize", kwargs)
        quality_flag = self.get_arg("quality_flag", kwargs)
        split_tol = self.get_arg("split_tol", kwargs)
        max_chunk_length = self.get_arg("max_chunk_length", kwargs)

        # Download the data.
        result = self.parent.query(**kwargs)

        # Loop over the light curves and set them up.
        lcs = result.pop("data")
        result["data"] = []
        for lc in lcs:
            d = lc.read()
            t, f, fe = d["TIME"], d["SAP_FLUX"], d["SAP_FLUX_ERR"]
            q = d["SAP_QUALITY"]
            result["data"] += LightCurve(t, f, fe, q == quality_flag,
                                         normalize=normalize, meta=lc) \
                .autosplit(split_tol, max_chunk_length)

        return result
