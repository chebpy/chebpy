import numpy as np

from .importing import import_optional
from .settings import _preferences as prefs


def import_plt() -> object:
    """Import matplotlib.pyplot if available and not skipped.

    This function attempts to import matplotlib.pyplot for plotting functionality.
    No fallback option exists, because the plot* functions are not added if
    module import returns None.

    Returns:
        object: The matplotlib.pyplot module if available, None otherwise.
    """
    return import_optional("matplotlib.pyplot", "MPL")


def plotfun(fun: callable, support: tuple, ax=None, n: int = None, **kwds) -> object:
    """Plot a function over a specified support interval.

    This function plots a callable object over a specified interval using
    matplotlib. For complex-valued functions, it plots the real part against
    the imaginary part.

    Args:
        fun (callable): The function to plot. Must be callable and have an
            'iscomplex' attribute.
        support (tuple): A tuple specifying the interval [a, b] over which to plot.
        ax (matplotlib.axes.Axes, optional): The axes on which to plot. If None,
            a new axes will be created. Defaults to None.
        n (int, optional): Number of points to use for plotting. If None, uses
            the value from preferences. Defaults to None.
        **kwds: Additional keyword arguments to pass to matplotlib's plot function.

    Returns:
        matplotlib.axes.Axes: The axes on which the plot was created.
    """
    ax = ax or import_plt().gca()
    n = n if n is not None else prefs.N_plot
    xx = np.linspace(*support, n)
    ff = fun(xx)
    if fun.iscomplex:
        ax.plot(np.real(ff), np.imag(ff), **kwds)
        ax.set_xlabel(kwds.pop("ylabel", "real"))
        ax.set_ylabel(kwds.pop("xlabel", "imag"))
    else:
        ax.plot(xx, ff, **kwds)
    return ax


def plotfuncoeffs(abscoeffs: np.ndarray, ax=None, **kwds) -> object:
    """Plot the absolute values of function coefficients on a semilogy scale.

    This function creates a semilogy plot of the absolute values of function
    coefficients, which is useful for visualizing the decay of coefficients
    in a Chebyshev series.

    Args:
        abscoeffs (numpy.ndarray): Array of absolute coefficient values to plot.
        ax (matplotlib.axes.Axes, optional): The axes on which to plot. If None,
            a new axes will be created. Defaults to None.
        **kwds: Additional keyword arguments to pass to matplotlib's semilogy function.

    Returns:
        matplotlib.axes.Axes: The axes on which the plot was created.
    """
    ax = ax or import_plt().gca()
    ax.set_ylabel(kwds.pop("xlabel", "coefficient magnitude"))
    ax.set_xlabel(kwds.pop("ylabel", "polynomial degree"))
    ax.semilogy(abscoeffs, ".", **kwds)
    return ax
