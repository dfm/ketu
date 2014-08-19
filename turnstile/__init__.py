# -*- coding: utf-8 -*-

__version__ = "0.1.0"

try:
    __TURNSTILE_SETUP__
except NameError:
    __TURNSTILE_SETUP__ = False

if not __TURNSTILE_SETUP__:
    __all__ = ["Pipeline", "Download", "Inject", "Prepare", "Detrend",
               "GPLikelihood", "OneDSearch", "TwoDSearch"]

    from .pipeline import Pipeline
    from .download import Download
    from .inject import Inject
    from .prepare import Prepare
    from .detrend import Detrend
    from .likelihood import GPLikelihood
    from .one_d_search import OneDSearch
    from .two_d_search import TwoDSearch
