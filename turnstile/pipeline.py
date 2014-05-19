#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["set_basedir", "PipelineElement"]

import os
import cPickle as pickle

VERSION = "0.0"


def set_basedir(path):
    global basedir
    basedir = path
basedir = os.path.expanduser(os.path.join("~", ".turnstile"))


class PipelineElement(object):

    element_name = None

    def __init__(self):
        if self.element_name is None:
            self.element_name = self.__class__.__name__

    @property
    def cachedir(self):
        return os.path.join(basedir, VERSION, self.element_name)

    def get_cache_filename(self, key):
        return os.path.join(self.cachedir, key + ".pkl")

    def get_key(self, **kwargs):
        raise NotImplementedError()

    def get(self, **kwargs):
        raise NotImplementedError()

    def __getitem__(self, key):
        fn = self.get_cache_filename(key)
        if os.path.exists(fn):
            with open(fn, "rb") as f:
                return pickle.load(f)
        return None

    def __setitem__(self, key, value):
        fn = self.get_cache_filename(key)
        try:
            os.makedirs(os.path.split(os.path.abspath(fn))[0])
        except os.error:
            pass
        with open(fn, "wb") as f:
            pickle.dump(value, f, -1)

    def call(self, **kwargs):
        k = self.get_key(**kwargs)
        val = self[k]
        if val is not None:
            return val

        val = self.get(**kwargs)
        self[k] = val
        return val
