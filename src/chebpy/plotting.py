"""Plotting utilities for visualizing functions and their properties.

This module provides functions for plotting functions, their coefficients, and
other visualizations useful for understanding function approximations. It uses
matplotlib for the actual plotting, but is designed to gracefully handle cases
where matplotlib is not available.

The main functions are:
- plotfun: Plot a function over a specified interval
- plotfuncoeffs: Plot the coefficients of a function on a semilogy scale

These functions are typically used by higher-level classes like Chebfun and
Chebtech to provide plotting capabilities.
"""

import matplotlib.pyplot as plt
import numpy as np

from .settings import _preferences as prefs


def plotfun(fun: callable, support: tuple, ax=None, n: int | None = None, **kwds) -> object:
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
    ax = ax or plt.gca()
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
    ax = ax or plt.gca()
    ax.set_ylabel(kwds.pop("xlabel", "coefficient magnitude"))
    ax.set_xlabel(kwds.pop("ylabel", "polynomial degree"))
    ax.semilogy(abscoeffs, **kwds)
    return ax
