#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from numpy.distutils.core import setup, Extension


if sys.argv[-1] == "publish":
    os.system("git rev-parse --short HEAD > COMMIT")
    os.system("python setup.py sdist upload")
    sys.exit()


# First, make sure that the f2py interfaces exist.
interface_exists = os.path.exists("turnstile/bls.pyf")
if "interface" in sys.argv or not interface_exists:
    # Generate the Fortran signature/interface.
    cmd = "cd src;"
    cmd += "f2py eebls.f -m _bls -h ../turnstile/bls.pyf"
    cmd += " --overwrite-signature"
    os.system(cmd)
    if "interface" in sys.argv:
        sys.exit(0)

# Define the Fortran extension.
bls = Extension("turnstile._bls", ["turnstile/bls.pyf", "src/eebls.f"])

setup(
    name="turnstile",
    url="https://github.com/dfm/turnstile",
    version="0.0.1",
    author="Dan Foreman-Mackey",
    author_email="danfm@nyu.edu",
    description="",
    long_description="",
    packages=["turnstile", ],
    ext_modules=[bls, ],
    classifiers=[
        # "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
