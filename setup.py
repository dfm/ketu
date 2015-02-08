#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import numpy
from Cython.Build import cythonize

# try:
#     from setuptools import setup, Extension
# except ImportError:
#     from distutils.core import setup, Extension
from distutils.core import setup, Extension

# Publish the library to PyPI.
if "publish" in sys.argv[1:]:
    os.system("python setup.py sdist upload")
    sys.exit()

# Set up the extension.
kwargs = dict(
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-Wno-unused-function", ]  # "-Wno-uninitialized"],
)
exts = [
    Extension("turnstile._compute", sources=["turnstile/_compute.pyx"],
              **kwargs),
    Extension("turnstile._grid_search", sources=["turnstile/_grid_search.pyx"],
              **kwargs),
]

# Hackishly inject a constant into builtins to enable importing of the
# package before the library is built.
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__TURNSTILE_SETUP__ = True
import turnstile

# Execute the setup command.
desc = open("README.rst").read()
setup(
    name="turnstile",
    version=turnstile.__version__,
    author="Daniel Foreman-Mackey",
    author_email="danfm@nyu.edu",
    packages=["turnstile", "turnstile.characterization"],
    ext_modules=cythonize(exts),
    url="http://github.com/dfm/turnstile",
    license="MIT",
    description="MOAR Planets",
    long_description=desc,
    package_data={"": ["README.rst", "LICENSE", ]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    test_suite="nose.collector",
)
