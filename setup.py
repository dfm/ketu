#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

try:
    from numpy.distutils.misc_util import get_numpy_include_dirs
except ImportError:
    get_numpy_include_dirs = lambda: []

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

turnstile = Extension("turnstile._turnstile", ["turnstile/_turnstile.c"],
                      include_dirs=get_numpy_include_dirs())

setup(
    name="turnstile",
    url="https://github.com/dfm/untrendy",
    version="0.0.1",
    author="Dan Foreman-Mackey",
    author_email="danfm@nyu.edu",
    description="Style.",
    long_description=open("README.rst").read(),
    packages=["turnstile"],
    # scripts=["bin/untrend"],
    package_data={"": ["README.rst", "LICENSE.rst"], },
    # "untrendy": ["untrendy.h", "test_data/*"]},
    include_package_data=True,
    ext_modules=[turnstile],
    classifiers=[
        # "Development Status :: 2 - Pre-Alpha",
        # "Development Status :: 3 - Alpha",
        "Development Status :: 4 - Beta",
        # "Development Status :: 5 - Production/Stable",
        # "Development Status :: 6 - Mature",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.3",
    ],
)
