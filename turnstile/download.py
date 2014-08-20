# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Download"]

import sys
import kplr
from .pipeline import Pipeline


class Download(Pipeline):

    query_parameters = {
        "kicid": (None, True),
        "short_cadence": (False, False),
        "npredictor": (50, False),
    }

    def get_result(self, query, parent_response):
        # Connect to the API.
        client = kplr.API()
        kicid = query["kicid"]
        kic = client.star(kicid)
        kic.kois

        # Download the light curves.
        short_cadence = query["short_cadence"]
        data = kic.get_light_curves(short_cadence=short_cadence, fetch=True)
        if not len(data):
            raise ValueError("No light curves for KIC {0}"
                             .format(query["kicid"]))

        # Find predictor stars sorted by distance. TODO: try other sets.
        npredictor = query["npredictor"]
        print("Downloading predictor light curves")
        q = dict(
            ktc_kepler_id="!={0:d}".format(kicid),
            ra=kic.kic_degree_ra, dec=kic.kic_dec, radius=1000,
            ktc_target_type="LC", max_records=npredictor, fetch=True,
        )
        predictor_lcs = []
        for lc in data:
            sys.stdout.write(".")
            sys.stdout.flush()
            q["sci_data_quarter"] = lc.sci_data_quarter
            predictor_lcs += [client.light_curves(**q)]

        return dict(star=kic, target_datasets=data,
                    predictor_datasets=predictor_lcs)
