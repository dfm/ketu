#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

import turnstile


def setup_pipeline():
    pipe = turnstile.Download()
    pipe = turnstile.Inject(pipe)
    pipe = turnstile.Prepare(pipe)
    pipe = turnstile.GPLikelihood(pipe)
    pipe = turnstile.Hypotheses(pipe)
    pipe = turnstile.Search(pipe)
    return pipe
