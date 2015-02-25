#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import numpy
from Cython.Build import cythonize

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

# Publish the library to PyPI.
if "publish" in sys.argv[1:]:
    os.system("python setup.py sdist upload")
    sys.exit()

# Set up the extension.
kwargs = dict(
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-Wno-unused-function", ],
)
exts = [
    Extension("ketu._compute", sources=["ketu/_compute.pyx"],
              **kwargs),
    Extension("ketu._grid_search", sources=["ketu/_grid_search.pyx"],
              **kwargs),
    Extension("ketu._traptransit", sources=["ketu/_traptransit.pyx"],
              **kwargs),
]

# Hackishly inject a constant into builtins to enable importing of the
# package before the library is built.
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__KETU_SETUP__ = True
import ketu

# Execute the setup command.
desc = open("README.rst").read()
setup(
    name="ketu",
    version=ketu.__version__,
    author="Daniel Foreman-Mackey",
    author_email="danfm@nyu.edu",
    packages=[
        "ketu",
        "ketu.k2",
        "ketu.kepler",
        "ketu.characterization",
    ],
    ext_modules=cythonize(exts),
    scripts=[
        "scripts/ketu-download",
        "scripts/ketu-photometry",
        "scripts/ketu-basis",
        "scripts/ketu-search",
        "scripts/ketu-collect",
        "scripts/ketu-catalog",
        "scripts/ketu-summary",
        "scripts/ketu-characterization",
        "scripts/ketu-traptransit",
    ],
    url="http://github.com/dfm/ketu",
    license="MIT",
    description="I can haz planetz?",
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
)
