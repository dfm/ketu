# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

__all__ = ["Pipeline"]

import os
import gzip
import json
import time
import hashlib
try:
    import cPickle as pickle
except ImportError:
    import pickle


class Pipeline(object):

    cache_ext = ".pkl.gz"
    element_name = None

    query_parameters = {
        # "param": (defualt_value, required),
    }

    def __init__(self, parent=None, clobber=False, cache=True, basepath=None,
                 **kwargs):
        self.cache = cache
        self.clobber = clobber
        self.parent = parent
        if self.element_name is None:
            self.element_name = self.__class__.__name__

        for k, v in kwargs.items():
            self.query_parameter[k] = (v, False)

        if basepath is None:
            if parent is None:
                from . import __version__
                basepath = os.path.join(os.path.abspath(os.path.expanduser(
                    os.environ.get("KETU_PATH", "~/ketu"))),
                    __version__)
            else:
                basepath = parent.basepath
        self.basepath = basepath

    def get_arg(self, k, kwargs):
        if k in kwargs:
            return kwargs.pop(k)
        if k in self.defaults:
            return self.defaults[k]
        raise RuntimeError("Missing required argument {0}".format(k))

    @property
    def cachedir(self):
        return os.path.join(self.basepath, self.element_name)

    def get_cache_filename(self, key):
        return os.path.join(self.cachedir, key + self.cache_ext)

    def get_id(self):
        k = self.element_name
        if self.parent is not None:
            k += " < " + self.parent.get_id()
        return k

    def get_key(self, **kwargs):
        q = {}
        for k, (default, req) in self.query_parameters.items():
            if k in kwargs:
                q[k] = kwargs[k]
            elif req:
                raise ValueError("Missing required parameter '{0}'".format(k))
            else:
                q[k] = default

        if self.parent is not None:
            q = dict(q, **(self.parent.get_key(**kwargs)[1]))

        key = hashlib.sha1(json.dumps([self.get_id(), q], sort_keys=True)
                           .encode("utf-8")).hexdigest()
        if "kicid" in kwargs:
            key = "{0}-{1}".format(kwargs["kicid"], key)
        return key, q

    def save_to_cache(self, fn, response):
        try:
            os.makedirs(os.path.dirname(fn))
        except os.error:
            pass
        with gzip.open(fn, "wb") as f:
            pickle.dump(response, f, -1)

    def load_from_cache(self, fn):
        if os.path.exists(fn):
            with gzip.open(fn, "rb") as f:
                return pickle.load(f)
        return None

    def query(self, **kwargs):
        key, query = self.get_key(**kwargs)
        fn = self.get_cache_filename(key)

        # Check if this request is already cached.
        if self.cache and not self.clobber:
            v = self.load_from_cache(fn)
            if v is not None:
                print("Using cached value in {0}".format(self.element_name))
                return PipelineResult(self, kwargs, v)

        # Get the response from the parent.
        parent_response = None
        if self.parent is not None:
            parent_response = self.parent.query(**kwargs)

        # If we get here then the result isn't yet cached. Let's compute it
        # now.
        print("Querying {0}".format(self.element_name))
        strt = time.time()
        response = self.get_result(query, parent_response)
        dt = time.time() - strt
        print("Finished querying {0} in {1:.2f}s".format(self.element_name,
                                                         dt))

        # Save the results to the cache.
        if self.cache:
            self.save_to_cache(fn, response)

        return PipelineResult(self, kwargs, response, parent_response)

    def get_result(self, **kwargs):
        raise NotImplementedError("Subclasses should implement this method")


class PipelineResult(object):

    def __init__(self, pipeline_element, query, response,
                 parent_response=None):
        self.pipeline_element = pipeline_element
        self.query = query
        self.response = response
        self._parent_response = parent_response

    @property
    def parent_response(self):
        if self._parent_response is None:
            self._parent_response = \
                self.pipeline_element.parent.query(**(self.query))
        return self._parent_response

    def __getitem__(self, k):
        if k in self.response:
            return self.response[k]
        if k in self.query:
            return self.query[k]
        if k in self.pipeline_element.query_parameters:
            v, r = self.pipeline_element.query_parameters[k]
            if r:
                raise AttributeError("Missing required parameter '{0}'"
                                     .format(k))
            return v
        if self.pipeline_element.parent is None:
            raise AttributeError("Unrecognized parameter '{0}'"
                                 .format(k))
        if k in self.pipeline_element.parent.query_parameters:
            v, r = self.pipeline_element.parent.query_parameters[k]
            if r:
                raise AttributeError("Missing required parameter '{0}'"
                                     .format(k))
            return v

        raise KeyError(k)

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            return getattr(self.parent_response, k)
