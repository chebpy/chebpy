import numpy as np

from chebpy.core.settings import userPrefs as prefs
from chebpy.core.importing import import_optional


def import_plt():
    """Import matplotlib.pyplot if available and not skipped.
    No fallback option exists, because the plot* functions
    are not added if module import return None.
    """
    return import_optional('matplotlib.pyplot', 'MPL')


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
