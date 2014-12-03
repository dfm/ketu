# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Discontinuity"]

import os
import h5py
import logging
import numpy as np

from .pipeline import Pipeline


class Discontinuity(Pipeline):

    query_parameters = dict(
    )

    def get_result(self, query, parent_response):
        lcs = parent_response.light_curves
        print(lcs)
