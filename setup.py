#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()

includes = [
    "turnstile",
    numpy.get_include(),
    "/usr/include",
    "/usr/local/include",
    "/usr/local/homebrew/include",
]

library_dirs = [
    "/usr/local/lib",
]

libraries = [
    "m",
    "cholmod",
    "amd",
    "camd",
    "colamd",
    "ccolamd",
    "suitesparseconfig",
    "blas",
    "lapack",
    "cxsparse",
    "george_shared",
]

grid = Extension("turnstile._grid", ["turnstile/_grid.c",
                                     "turnstile/turnstile.c"],
                 library_dirs=library_dirs,
                 libraries=libraries,
                 runtime_library_dirs=library_dirs,
                 include_dirs=includes)

setup(
    name="turnstile",
    url="https://github.com/dfm/turnstile",
    version="0.1.0",
    author="Dan Foreman-Mackey",
    author_email="danfm@nyu.edu",
    description="Search.",
    long_description=open("README.rst").read(),
    packages=["turnstile"],
    package_data={"": ["README.rst"]},
    package_dir={"turnstile": "turnstile"},
    include_package_data=True,
    ext_modules=[grid],
    install_requires=[
        "bart",
    ],
    classifiers=[
        # "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
