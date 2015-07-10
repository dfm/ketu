# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Likelihood"]

from ..pipeline import Pipeline


class Likelihood(Pipeline):

    query_parameters = {
        "basis_file": (None, True),
        "nbasis": (None, True),
        "lambda": (1.0, False),
    }

    def get_result(self, query, parent_response):
        for lc in parent_response.target_light_curves:
            lc.prepare(query["basis_file"], nbasis=query["nbasis"],
                       lam=query["lambda"])
        return dict(model_light_curves=parent_response.target_light_curves)
