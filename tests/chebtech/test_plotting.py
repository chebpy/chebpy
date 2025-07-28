"""Unit-tests for Chebtech2 plotting methods.

This module contains tests for the plotting functionality of Chebtech2,
including plot and plotcoeffs methods for both real and complex functions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from chebpy.core.chebtech import Chebtech2

from .conftest import cos, sin


def _joukowsky(z):
    """Apply the Joukowsky transformation to z.

    The Joukowsky transformation maps the unit circle to an ellipse and is used
    in complex analysis and fluid dynamics. It is defined as f(z) = 0.5 * (z + 1/z).

    Args:
        z (complex or numpy.ndarray): Complex number or array of complex numbers.

    Returns:
        complex or numpy.ndarray: Result of the Joukowsky transformation.
    """
    return 0.5 * (z + 1 / z)


@pytest.fixture
def plotting_fixtures():
    """Create Chebtech2 objects for testing plotting methods.

    This fixture creates three Chebtech2 objects:
    1. A fixed-length Chebtech2 for a real function
    2. An adaptive Chebtech2 for the same real function
    3. An adaptive Chebtech2 for a complex function

    Returns:
        dict: Dictionary containing three Chebtech2 objects:
            f0: Fixed-length Chebtech2 for a real function
            f1: Adaptive Chebtech2 for the same real function
            f2: Adaptive Chebtech2 for a complex function
    """

    def f(x):
        return sin(3 * x) + 5e-1 * cos(30 * x)

    def u(x):
        return np.exp(2 * np.pi * 1j * x)

    f0 = Chebtech2.initfun_fixedlen(f, 100)
    f1 = Chebtech2.initfun_adaptive(f)
    f2 = Chebtech2.initfun_adaptive(u)

    return {"f0": f0, "f1": f1, "f2": f2}


def test_plot(plotting_fixtures):
    """Test the plot method of Chebtech2.

    This test verifies that the plot method of a Chebtech2 object
    can be called without errors. It creates a matplotlib figure and
    axes, then calls the plot method with the axes as an argument.

    Args:
        plotting_fixtures: Fixture providing test Chebtech2 objects.
    """
    fig, ax = plt.subplots()
    plotting_fixtures["f0"].plot(ax=ax)


def test_plot_complex(plotting_fixtures):
    """Test plotting complex Chebtech2 objects.

    This test verifies that complex Chebtech2 objects can be plotted
    without errors. It creates a matplotlib figure and axes, then
    plots Bernstein ellipses by applying the joukowsky function to
    scaled versions of a complex Chebtech2 object.

    Args:
        plotting_fixtures: Fixture providing test Chebtech2 objects.
    """
    fig, ax = plt.subplots()
    # plot Bernstein ellipses
    for rho in np.arange(1.1, 2, 0.1):
        _joukowsky(rho * plotting_fixtures["f2"]).plot(ax=ax)


@pytest.mark.skipif(plt is None, reason="matplotlib not installed")
def test_plotcoeffs(plotting_fixtures):
    """Test the plotcoeffs method of Chebtech2.

    This test verifies that the plotcoeffs method of Chebtech2 objects
    can be called without errors. It creates a matplotlib figure and
    axes, then calls the plotcoeffs method with the axes as an argument
    for two different Chebtech2 objects.

    Args:
        plotting_fixtures: Fixture providing test Chebtech2 objects.
    """
    fig, ax = plt.subplots()
    plotting_fixtures["f0"].plotcoeffs(ax=ax)
    plotting_fixtures["f1"].plotcoeffs(ax=ax, color="r")
