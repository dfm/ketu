# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Inject"]

import logging
import transit
from ..pipeline import Pipeline


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
            return dict(
                target_light_curves=parent_response.target_light_curves)

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
        for lc in parent_response.target_light_curves:
            lc.flux *= s.light_curve(lc.time)
            results.append(lc)

        return dict(target_light_curves=results, injected_system=s)
