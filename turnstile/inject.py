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

    defaults = dict(b=0.0, e=0.0, pomega=0.0, injections=[])

    def get_result(self, **kwargs):
        # Build the system.
        s = transit.System(transit.Central())

        # Parse the arguments.
        injections = self.get_arg("injections", kwargs)
        if not len(injections):
            return self.parent.query(**kwargs)
        for inj in injections:
            body = transit.Body(r=inj["radius"], period=inj["period"],
                                t0=inj["t0"], b=self.get_arg("b", inj),
                                e=self.get_arg("e", inj),
                                pomega=self.get_arg("pomega", inj))
            s.add_body(body)

        # Get the datasets.
        result = self.parent.query(**kwargs)

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
