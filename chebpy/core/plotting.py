import importlib

import numpy as np

from chebpy.core.settings import userPrefs as prefs

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


def plotfun(fn_y, support, ax=None, N=None, **kwargs):
    ax = ax or import_plt().gca()
    N = N if N is not None else prefs.N_plot
    a, b = support
    xx = np.linspace(a, b, N)
    ax.plot(xx, fn_y(xx), **kwargs)
    return ax


def plotfuncoeffs(abscoeffs, ax=None, **kwargs):
    ax = ax or import_plt().gca()
    ax.set_ylabel(kwargs.pop('xlabel', 'coefficient magnitude'))
    ax.set_xlabel(kwargs.pop('ylabel', 'polynomial degree'))
    ax.semilogy(abscoeffs, '.', **kwargs)
    return ax
