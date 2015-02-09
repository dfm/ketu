# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Inject", "InjectedLightCurve"]

import logging
import transit
import numpy as np
from .pipeline import Pipeline


class Inject(Pipeline):

    query_parameters = dict(
        q1=(0.5, False),
        q2=(0.5, False),
        mstar=(1.0, False),
        rstar=(1.0, False),
        injections=([], False),
    )

    def get_result(self, query, parent_response):
        # Parse the arguments.
        injections = query["injections"]
        if not len(injections):
            return dict(target_datasets=parent_response.target_datasets)

        # Build the system.
        q1 = query["q1"]
        q2 = query["q2"]
        mstar = query["mstar"]
        rstar = query["rstar"]
        s = transit.System(transit.Central(q1=q1, q2=q2, mass=mstar,
                                           radius=rstar))

        # Loop over injected bodies and add them to the system.
        for inj in injections:
            body = transit.Body(r=inj["radius"], period=inj["period"],
                                t0=inj["t0"], b=inj.get("b", 0.0),
                                e=inj.get("e", 0.0),
                                pomega=inj.get("pomega", 0.0))
            s.add_body(body)
            try:
                body.ix
            except ValueError:
                logging.warn("Removing planet with invalid impact parameter")
                s.bodies.pop()

        # Inject the transit into each dataset.
        results = []
        for _ in parent_response.target_datasets:
            lc = InjectedLightCurve(_)
            lc.flux[lc.m] *= s.light_curve(lc.time[lc.m])
            results.append(lc)

        return dict(target_datasets=results, injected_system=s)


class InjectedLightCurve(object):

    def __init__(self, lc):
        for k, v in lc.params.iteritems():
            setattr(self, k, v)

        d = lc.read()
        self.time = np.array(d["TIME"], dtype=np.float64)
        self.flux = np.array(d["SAP_FLUX"], dtype=np.float64)
        self.ferr = np.array(d["SAP_FLUX_ERR"], dtype=np.float64)
        self.q = np.array(d["SAP_QUALITY"], dtype=int)
        self.m = (np.isfinite(self.time) * np.isfinite(self.flux)
                  * np.isfinite(self.ferr))

    def read(self, **kwargs):
        return dict(TIME=self.time, SAP_FLUX=self.flux, SAP_FLUX_ERR=self.ferr,
                    SAP_QUALITY=self.q)
