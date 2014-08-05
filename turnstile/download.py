# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Download"]

import kplr
from .pipeline import Pipeline


class Download(Pipeline):

    defaults = {
        "short_cadence": False,
    }

    def get_result(self, kicid=None, **kwargs):
        assert kicid is not None, "You need to give a KIC ID"

        # Connect to the API.
        client = kplr.API()
        kic = client.star(kicid)
        kic.kois

        # Download the light curves.
        short_cadence = self.get_arg("short_cadence", kwargs)
        data = kic.get_light_curves(short_cadence=short_cadence, fetch=True)

        return dict(star=kic, data=data)
