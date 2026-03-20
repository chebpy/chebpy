"""Tests for the plotting module.

This module tests the plotting utility functions in chebpy.plotting:
- plotfun: plotting a function over a support interval
- plotfuncoeffs: semilogy plot of coefficient magnitudes
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use("Agg")

from chebpy.plotting import plotfun, plotfuncoeffs


class _SimpleFun:
    """Minimal callable object with iscomplex attribute for testing plotfun."""

    def __init__(self, fn, *, iscomplex=False):
        self._fn = fn
        self.iscomplex = iscomplex

    def __call__(self, x):
        return self._fn(x)


class TestPlotfun:
    """Tests for the plotfun function."""

    def test_plotfun_returns_axes(self):
        """Test that plotfun returns a matplotlib Axes object."""
        fun = _SimpleFun(np.sin)
        fig, ax = plt.subplots()
        result = plotfun(fun, (-1, 1), ax=ax)
        assert result is ax
        plt.close(fig)

    def test_plotfun_creates_axes_when_none(self):
        """Test that plotfun creates axes when none is provided."""
        fun = _SimpleFun(np.cos)
        plt.figure()
        result = plotfun(fun, (-1, 1))
        assert isinstance(result, plt.Axes)
        plt.close("all")

    def test_plotfun_complex_function(self):
        """Test that plotfun handles complex-valued functions."""
        fun = _SimpleFun(lambda x: np.exp(1j * np.pi * x), iscomplex=True)
        fig, ax = plt.subplots()
        result = plotfun(fun, (-1, 1), ax=ax)
        assert result is ax
        plt.close(fig)

    def test_plotfun_custom_n(self):
        """Test that plotfun respects the n parameter."""
        call_args = {}

        def tracking_fn(x):
            call_args["n"] = len(x)
            return np.sin(x)

        fun = _SimpleFun(tracking_fn)
        fig, ax = plt.subplots()
        plotfun(fun, (-1, 1), ax=ax, n=37)
        assert call_args["n"] == 37
        plt.close(fig)


class TestPlotfuncoeffs:
    """Tests for the plotfuncoeffs function."""

    def test_plotfuncoeffs_returns_axes(self):
        """Test that plotfuncoeffs returns a matplotlib Axes object."""
        coeffs = np.array([1.0, 0.5, 0.25, 0.125])
        fig, ax = plt.subplots()
        result = plotfuncoeffs(coeffs, ax=ax)
        assert result is ax
        plt.close(fig)

    def test_plotfuncoeffs_creates_axes_when_none(self):
        """Test that plotfuncoeffs creates axes when none is provided."""
        coeffs = np.array([1.0, 0.1, 0.01])
        plt.figure()
        result = plotfuncoeffs(coeffs)
        assert isinstance(result, plt.Axes)
        plt.close("all")
