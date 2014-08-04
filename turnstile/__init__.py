# -*- coding: utf-8 -*-

__version__ = "0.0.1"

try:
    __TURNSTILE_SETUP__
except NameError:
    __TURNSTILE_SETUP__ = False

if not __TURNSTILE_SETUP__:
    __all__ = ["Pipeline", "Download", "Inject", "Prepare", "Detrend",
               "BasicLikelihood", "GPLikelihood", "Hypotheses", "Search"]

    from .pipeline import Pipeline
    from .download import Download
    from .inject import Inject
    from .prepare import Prepare
    from .detrend import Detrend
    from .likelihood import BasicLikelihood, GPLikelihood
    from .hypotheses import Hypotheses
    from .search import Search
