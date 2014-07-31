# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Inject", "InjectedLightCurve"]

import transit
import numpy as np
from .pipeline import Pipeline


class InjectedLightCurve(object):

    def __init__(self, lc):
        d = lc.read()
        self.time = np.array(d["TIME"], dtype=np.float64)
        self.flux = np.array(d["SAP_FLUX"], dtype=np.float64)
        self.ferr = np.array(d["SAP_FLUX_ERR"], dtype=np.float64)
        self.q = np.array(d["SAP_QUALITY"], dtype=np.float64)
        self.m = (np.isfinite(self.time) * np.isfinite(self.flux)
                  * np.isfinite(self.ferr))

    def read(self):
        return dict(TIME=self.time, SAP_FLUX=self.flux, SAP_FLUX_ERR=self.ferr,
                    SAP_QUALITY=self.q)


class Inject(Pipeline):

    defaults = dict(b=0.0, e=0.0, pomega=0.0)

    def get_result(self, **kwargs):
        # Parse the arguments.
        period = self.get_arg("period", kwargs)
        radius = self.get_arg("radius", kwargs)
        t0 = self.get_arg("t0", kwargs)
        assert period is not None
        assert radius is not None
        assert t0 is not None

        b = self.get_arg("b", kwargs)
        e = self.get_arg("e", kwargs)
        pomega = self.get_arg("pomega", kwargs)

        # Get the datasets.
        result = self.parent.query(**kwargs)

        # Build the system.
        s = transit.System(transit.Central())
        body = transit.Body(r=radius, period=period, t0=t0, b=b, e=e,
                            pomega=pomega)
        s.add_body(body)

        # Inject the transit into each dataset.
        lcs = result.pop("data")
        result["data"] = []
        for _ in lcs:
            lc = InjectedLightCurve(_)
            lc.flux[lc.m] *= s.light_curve(lc.time[lc.m])
            result["data"].append(lc)

        # Save the injection system.
        result["injection"] = s

        return result
