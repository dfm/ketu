# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Pipeline"]

import os
import json
import hashlib
import cPickle as pickle

VERSION = "0.0"


basedir = os.path.abspath(os.path.expanduser(os.environ.get("TURNSTILE_PATH",
                                                            "~/.turnstile")))


class Pipeline(object):

    element_name = None
    defaults = {}

    def __init__(self, parent=None, **kwargs):
        self.defaults = dict(self.defaults, **kwargs)
        self.parent = parent
        if self.element_name is None:
            self.element_name = self.__class__.__name__

    def get_arg(self, k, kwargs):
        return kwargs.pop(k, self.defaults.get(k))

    @property
    def cachedir(self):
        return os.path.join(basedir, VERSION, self.element_name)

    def get_cache_filename(self, key):
        return os.path.join(self.cachedir, key + ".pkl")

    def get_key(self, **kwargs):
        k = None if self.parent is None else self.parent.element_name
        return hashlib.sha1(json.dumps([k, dict(self.defaults, **kwargs)],
                                       sort_keys=True)).hexdigest()

    def query(self, **kwargs):
        clobber = kwargs.get("clobber", False)

        # Check if this request is already cached.
        key = self.get_key(**kwargs)
        fn = self.get_cache_filename(key)
        if not clobber and os.path.exists(fn):
            print("Using cached value in {0}".format(self.element_name))
            with open(fn, "rb") as f:
                return pickle.load(f)

        # If we get here then the result isn't yet cached. Let's compute it
        # now.
        result = self.get_result(**kwargs)

        # Save the results to the cache.
        try:
            os.makedirs(os.path.dirname(fn))
        except os.error:
            pass
        with open(fn, "wb") as f:
            pickle.dump(result, f, -1)
        return result

    def get_result(self, **kwargs):
        raise NotImplementedError("Subclasses should implement this method")
