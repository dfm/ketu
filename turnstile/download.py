# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Download"]

import kplr
from .pipeline import Pipeline


class Download(Pipeline):

    query_parameters = {
        "kicid": (None, True),
        "short_cadence": (False, False),
    }

    def get_result(self, query, parent_response):
        # Connect to the API.
        client = kplr.API()
        kic = client.star(query["kicid"])
        kic.kois

        # Download the light curves.
        short_cadence = query["short_cadence"]
        data = kic.get_light_curves(short_cadence=short_cadence, fetch=True)
        if not len(data):
            raise ValueError("No light curves for KIC {0}"
                             .format(query["kicid"]))

        return dict(star=kic, datasets=data)
