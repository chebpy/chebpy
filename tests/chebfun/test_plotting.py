"""Unit-tests for Chebfun plotting methods.

This module contains tests for the plotting functionality of Chebfun,
including plot and plotcoeffs methods for both real and complex functions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from chebpy import chebfun
from chebpy.chebfun import Chebfun

from ..generic.plotting import test_plot, test_plot_complex  # noqa: F401
from ..utilities import cos, exp, sin


@pytest.fixture
def plotting_fixtures():
    """Create Chebfun objects for testing plotting methods.

    This fixture creates several Chebfun objects with different characteristics
    for testing various plotting methods.

    Returns:
        dict: Dictionary containing:
            f1: Chebfun representing sin(x)
            f2: Chebfun representing cos(x)
            f3: Chebfun representing exp(x)
            f4: Chebfun representing exp(i*pi*x) (complex)
    """
    f1 = Chebfun.initfun_adaptive(sin, [-1, 1])
    f2 = Chebfun.initfun_adaptive(cos, [-1, 1])
    f3 = Chebfun.initfun_adaptive(exp, [-1, 1])
    f4 = Chebfun.initfun_adaptive(lambda x: np.exp(1j * np.pi * x), [-1, 1])

    return {"f1": f1, "f2": f2, "f3": f3, "f4": f4}


def test_plot_multiple(plotting_fixtures):
    """Test plotting multiple Chebfun objects on the same axes.

    This test verifies that multiple Chebfun objects can be plotted
    on the same axes without errors. It creates a matplotlib figure and
    axes, then calls the plot method for multiple Chebfun objects.

    Args:
        plotting_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = plotting_fixtures["f1"]
    f2 = plotting_fixtures["f2"]
    f3 = plotting_fixtures["f3"]

    fig, ax = plt.subplots()
    f1.plot(ax=ax)
    f2.plot(ax=ax, color="r")
    f3.plot(ax=ax, color="g")
    plt.close(fig)


def test_plotcoeffs(plotting_fixtures):
    """Test the plotcoeffs method of Chebfun objects.

    This test verifies that the plotcoeffs method of Chebfun objects
    can be called without errors. It creates a matplotlib figure and
    axes, then calls the plotcoeffs method with the axes as an argument
    for different Chebfun objects.

    Args:
        plotting_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = plotting_fixtures["f1"]
    f2 = plotting_fixtures["f2"]
    f3 = plotting_fixtures["f3"]

    fig, ax = plt.subplots()
    f1.plotcoeffs(ax=ax)
    plt.close(fig)

    fig, ax = plt.subplots()
    f2.plotcoeffs(ax=ax)
    plt.close(fig)

    fig, ax = plt.subplots()
    f3.plotcoeffs(ax=ax)


def test_plotcoeffs_multiple(plotting_fixtures):
    """Test plotting coefficients of multiple Chebfun objects.

    This test verifies that the coefficients of multiple Chebfun objects
    can be plotted on the same axes without errors. It creates a matplotlib
    figure and axes, then calls the plotcoeffs method for multiple Chebfun objects.

    Args:
        plotting_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = plotting_fixtures["f1"]
    f2 = plotting_fixtures["f2"]

    fig, ax = plt.subplots()
    f1.plotcoeffs(ax=ax)
    f2.plotcoeffs(ax=ax, color="r")


def test_plot_with_options(plotting_fixtures):
    """Test plotting Chebfun objects with various options.

    This test verifies that the plot method of Chebfun objects
    can be called with various matplotlib options without errors.

    Args:
        plotting_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = plotting_fixtures["f1"]

    fig, ax = plt.subplots()
    f1.plot(ax=ax, color="r", linestyle="--", linewidth=2, marker="o", markersize=5)


def test_plotcoeffs_with_options(plotting_fixtures):
    """Test plotting coefficients with various options.

    This test verifies that the plotcoeffs method of Chebfun objects
    can be called with various matplotlib options without errors.

    Args:
        plotting_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = plotting_fixtures["f1"]

    fig, ax = plt.subplots()
    f1.plotcoeffs(ax=ax, color="g", marker="s", markersize=8, linestyle="-.")


def test_plot_multipiece():
    """Test plotting a multi-piece Chebfun.

    This test verifies that a multi-piece Chebfun can be plotted
    without errors. It creates a Chebfun with multiple pieces and
    plots it.
    """
    # Create a multi-piece Chebfun by breaking f1 at multiple points
    domain = np.linspace(-1, 1, 5)
    f_multi = Chebfun.initfun_adaptive(sin, domain)

    fig, ax = plt.subplots()
    f_multi.plot(ax=ax)


class TestChebfunPlottingEdgeCases:
    """Additional edge case tests for Chebfun plotting."""

    def test_plot_returns_axes(self):
        """Test that plot() returns matplotlib axes."""
        f = chebfun(lambda x: x**2, [-1, 1])
        ax = f.plot()
        assert ax is not None
        plt.close("all")

    def test_plotcoeffs_returns_axes(self):
        """Test that plotcoeffs() returns matplotlib axes."""
        f = chebfun(lambda x: np.sin(x), [-1, 1])
        ax = f.plotcoeffs()
        assert ax is not None
        plt.close("all")

    def test_plotcoeffs_multipiece(self):
        """Test plotcoeffs with multiple pieces."""
        f = chebfun(lambda x: np.abs(x), [-1, 0, 1])
        ax = f.plotcoeffs()
        assert ax is not None
        # Should have plotted multiple series
        assert len(ax.lines) >= 2
        plt.close("all")
