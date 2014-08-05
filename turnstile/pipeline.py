# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Pipeline"]

import os
import gzip
import json
import time
import hashlib
import cPickle as pickle

basedir = os.path.abspath(os.path.expanduser(os.environ.get("TURNSTILE_PATH",
                                                            "~/.turnstile")))


class Pipeline(object):

    element_name = None
    defaults = {}

    def __init__(self, parent=None, clobber=False, **kwargs):
        self.clobber = clobber
        self.defaults = dict(self.defaults, **kwargs)
        self.parent = parent
        if self.element_name is None:
            self.element_name = self.__class__.__name__

    def get_arg(self, k, kwargs):
        if k in kwargs:
            return kwargs.pop(k)
        if k in self.defaults:
            return self.defaults[k]
        raise RuntimeError("Missing required argument {0}".format(k))

    @property
    def cachedir(self):
        from . import __version__
        return os.path.join(basedir, __version__, self.element_name)

    def get_cache_filename(self, key):
        return os.path.join(self.cachedir, key + ".pkl.gz")

    def get_id(self):
        k = self.element_name
        if self.parent is not None:
            k += " < " + self.parent.get_id()
        return k

    def get_key(self, **kwargs):
        return hashlib.sha1(json.dumps([self.get_id(),
                                        dict(self.defaults, **kwargs)],
                                       sort_keys=True)).hexdigest()

    def query(self, **kwargs):
        # Check if this request is already cached.
        key = self.get_key(**kwargs)
        fn = self.get_cache_filename(key)
        if not self.clobber and os.path.exists(fn):
            print("Using cached value in {0}".format(self.element_name))
            with gzip.open(fn, "rb") as f:
                return pickle.load(f)

        # If we get here then the result isn't yet cached. Let's compute it
        # now.
        print("Querying {0}".format(self.element_name))
        strt = time.time()
        result = self.get_result(**kwargs)
        dt = time.time() - strt
        print("Finished querying {0} in {1:.2f}s".format(self.element_name,
                                                         dt))

        # Save the results to the cache.
        try:
            os.makedirs(os.path.dirname(fn))
        except os.error:
            pass
        with gzip.open(fn, "wb") as f:
            pickle.dump(result, f, -1)
        return result

    def get_result(self, **kwargs):
        raise NotImplementedError("Subclasses should implement this method")
