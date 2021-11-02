import numpy as np

from .settings import _preferences as prefs
from .importing import import_optional


def import_plt():
    """Import matplotlib.pyplot if available and not skipped.
    No fallback option exists, because the plot* functions
    are not added if module import return None.
    """
    return import_optional("matplotlib.pyplot", "MPL")


def plotfun(fun, support, ax=None, N=None, **kwds):
    ax = ax or import_plt().gca()
    N = N if N is not None else prefs.N_plot
    xx = np.linspace(*support, N)
    ff = fun(xx)
    if fun.iscomplex:
        ax.plot(np.real(ff), np.imag(ff), **kwds)
        ax.set_xlabel(kwds.pop("ylabel", "real"))
        ax.set_ylabel(kwds.pop("xlabel", "imag"))
    else:
        ax.plot(xx, ff, **kwds)
    return ax


def plotfuncoeffs(abscoeffs, ax=None, **kwds):
    ax = ax or import_plt().gca()
    ax.set_ylabel(kwds.pop("xlabel", "coefficient magnitude"))
    ax.set_xlabel(kwds.pop("ylabel", "polynomial degree"))
    ax.semilogy(abscoeffs, ".", **kwds)
    return ax
