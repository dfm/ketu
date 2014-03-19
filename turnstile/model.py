#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["box_model"]

import numpy as np


def box_model(t, t0, duration, depth):
    m = np.ones_like(t)
    m[np.abs(t-t0) < 0.5*duration] -= depth
    return m
