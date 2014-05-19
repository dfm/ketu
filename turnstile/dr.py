#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["DataRetrieval"]

import kplr

from .data import LightCurve
from .pipeline import PipelineElement


class DataRetrieval(PipelineElement):

    element_name = "dr"
    ttol = 0.5
    minpts = 50

    def get_key(self, **kwargs):
        return "{0}_{1}_{2}".format(kwargs["kicid"], self.ttol, self.minpts)

    def get(self, **kwargs):
        client = kplr.API()
        kic = client.star(kwargs["kicid"])

        # Loop over the long cadence light curves from mask and pre-process
        # them all.
        datasets = []
        lcs = kic.get_light_curves(short_cadence=False)
        for lc in lcs:
            data = lc.read()
            datasets += LightCurve(data["TIME"], data["SAP_FLUX"],
                                   data["SAP_FLUX_ERR"],
                                   data["SAP_QUALITY"] == 0) \
                .autosplit(self.ttol)

        # Remove datasets that are too short.
        return filter(lambda d: len(d.time) >= self.minpts, datasets)
