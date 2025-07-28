"""Unit-tests for Chebfun plotting methods.

This module contains tests for the plotting functionality of Chebfun,
including plot and plotcoeffs methods for both real and complex functions.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from chebpy.core.chebfun import Chebfun

from .conftest import sin, cos, exp

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


@pytest.mark.skipif(plt is None, reason="matplotlib not installed")
def test_plot(plotting_fixtures):
    """Test the plot method of Chebfun objects.

    This test verifies that the plot method of a Chebfun object
    can be called without errors. It creates a matplotlib figure and
    axes, then calls the plot method with the axes as an argument.

    Args:
        plotting_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = plotting_fixtures["f1"]
    fig, ax = plt.subplots()
    f1.plot(ax=ax)
    plt.close(fig)


@pytest.mark.skipif(plt is None, reason="matplotlib not installed")
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
    f2.plot(ax=ax, color='r')
    f3.plot(ax=ax, color='g')
    plt.close(fig)


@pytest.mark.skipif(plt is None, reason="matplotlib not installed")
def test_plot_complex(plotting_fixtures):
    """Test plotting complex Chebfun objects.

    This test verifies that complex Chebfun objects can be plotted
    without errors. It creates a matplotlib figure and axes, then
    plots Bernstein ellipses by applying the joukowsky function to
    scaled versions of a complex Chebfun object.

    Args:
        plotting_fixtures: Fixture providing test Chebfun objects.
    """
    f4 = plotting_fixtures["f4"]
    fig, ax = plt.subplots()

    # Plot Bernstein ellipses
    for rho in np.arange(1.1, 2, 0.1):
        (np.exp(1j * 0.5 * np.pi) * _joukowsky(rho * f4)).plot(ax=ax)

    plt.close(fig)


@pytest.mark.skipif(plt is None, reason="matplotlib not installed")
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
    plt.close(fig)


@pytest.mark.skipif(plt is None, reason="matplotlib not installed")
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
    f2.plotcoeffs(ax=ax, color='r')
    plt.close(fig)


@pytest.mark.skipif(plt is None, reason="matplotlib not installed")
def test_plot_with_options(plotting_fixtures):
    """Test plotting Chebfun objects with various options.

    This test verifies that the plot method of Chebfun objects
    can be called with various matplotlib options without errors.

    Args:
        plotting_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = plotting_fixtures["f1"]

    fig, ax = plt.subplots()
    f1.plot(ax=ax, color='r', linestyle='--', linewidth=2, marker='o', markersize=5)
    plt.close(fig)


@pytest.mark.skipif(plt is None, reason="matplotlib not installed")
def test_plotcoeffs_with_options(plotting_fixtures):
    """Test plotting coefficients with various options.

    This test verifies that the plotcoeffs method of Chebfun objects
    can be called with various matplotlib options without errors.

    Args:
        plotting_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = plotting_fixtures["f1"]

    fig, ax = plt.subplots()
    f1.plotcoeffs(ax=ax, color='g', marker='s', markersize=8, linestyle='-.')
    plt.close(fig)


@pytest.mark.skipif(plt is None, reason="matplotlib not installed")
def test_plot_multipiece(plotting_fixtures):
    """Test plotting a multi-piece Chebfun.

    This test verifies that a multi-piece Chebfun can be plotted
    without errors. It creates a Chebfun with multiple pieces and
    plots it.

    Args:
        plotting_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = plotting_fixtures["f1"]
    f2 = plotting_fixtures["f2"]

    # Create a multi-piece Chebfun by breaking f1 at multiple points
    domain = np.linspace(-1, 1, 5)
    f_multi = Chebfun.initfun_adaptive(sin, domain)

    fig, ax = plt.subplots()
    f_multi.plot(ax=ax)
    plt.close(fig)