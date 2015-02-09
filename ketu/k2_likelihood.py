# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["K2Likelihood"]

from .pipeline import Pipeline


class K2Likelihood(Pipeline):

    query_parameters = {
        "basis_file": (None, True),
        "nbasis": (150, False),
    }

    def get_result(self, query, parent_response):
        for lc in parent_response.target_light_curves:
            lc.prepare(query["basis_file"], nbasis=query["nbasis"])
        return dict(model_light_curves=parent_response.target_light_curves)
