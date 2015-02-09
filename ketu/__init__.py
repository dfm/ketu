# -*- coding: utf-8 -*-

__version__ = "0.3.0"

try:
    __KETU_SETUP__  # NOQA
except NameError:
    __KETU_SETUP__ = False

if not __KETU_SETUP__:
    __all__ = [
        "Pipeline", "kepler", "k2",
        "OneDSearch", "TwoDSearch", "PeakDetect", "FeatureExtract",
        "Validate", "IterativeTwoDSearch",
    ]

    try:
        from . import _compute, _grid_search  # NOQA
    except ImportError:
        raise ImportError("you must first build the Cython extensions and "
                          "leave the source directory (or build them in "
                          "place)")

    from . import kepler, k2

    from .pipeline import Pipeline
    from .one_d_search import OneDSearch
    from .two_d_search import TwoDSearch
    from .peak_detect import PeakDetect
    from .feature_extract import FeatureExtract
    from .dv import Validate
    from .iterative import IterativeTwoDSearch
