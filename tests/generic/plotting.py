"""Generic test functions for plotting operations.

This module contains test functions for plotting operations that can be used
with any type of function object (Bndfun, Chebfun, or Chebtech). These tests
focus on basic plotting functionality and complex function visualization.
"""

import numpy as np
from matplotlib import pyplot as plt

from tests.utilities import joukowsky


def test_plot_complex(complexfun):
    """Test plotting complex Chebfun objects.

    This test verifies that complex Chebfun objects can be plotted
    without errors. It creates a matplotlib figure and axes, then
    plots Bernstein ellipses by applying the joukowsky function to
    scaled versions of a complex Chebfun object.

    Args:
        complexfun: Fixture providing a complex function object.
    """
    fig, ax = plt.subplots()

    # Plot Bernstein ellipses
    for rho in np.arange(1.1, 2, 0.1):
        (np.exp(1j * 0.5 * np.pi) * joukowsky(rho * complexfun)).plot(ax=ax)


def test_plot(constfun):
    """Test the plot method of Bndfun."""
    fig, ax = plt.subplots()
    constfun.plot(ax=ax, color="g", marker="o", markersize=2, linestyle="")
