import importlib

import numpy as np


def _import_optional(name):
    """Attempt to import the specified module.
    Either returns the module, or None.
    
    See https://github.com/pandas-dev/pandas/blob/master/pandas/compat/_optional.py
    """
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    return module


def import_plt():
    return _import_optional('matplotlib.pyplot')


def plotfun(fn_y, support, ax=None, *args, **kwargs):
    ax = ax or import_plt().gca()
    a, b = support
    xx = np.linspace(a, b, 2001)
    ax.plot(xx, fn_y(xx), *args, **kwargs)
    return ax


def plotfuncoeffs(abscoeffs, ax=None, *args, **kwargs):
    ax = ax or import_plt().gca()
    ax.semilogy(abscoeffs, '.', *args, **kwargs)
    ax.set_ylabel('coefficient magnitude')
    ax.set_xlabel('polynomial degree')
    return ax
