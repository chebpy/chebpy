"""Unit-tests for Chebtech plotting methods.

This module contains tests for the plotting functionality of Chebtech,
including plot and plotcoeffs methods for both real and complex functions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from chebpy.chebtech import Chebtech

from ..generic.plotting import test_plot, test_plot_complex  # noqa: F401
from ..utilities import cos, sin


@pytest.fixture
def plotting_fixtures():
    """Create Chebtech objects for testing plotting methods.

    This fixture creates three Chebtech objects:
    1. A fixed-length Chebtech for a real function
    2. An adaptive Chebtech for the same real function
    3. An adaptive Chebtech for a complex function

    Returns:
        dict: Dictionary containing three Chebtech objects:
            f0: Fixed-length Chebtech for a real function
            f1: Adaptive Chebtech for the same real function
            f2: Adaptive Chebtech for a complex function
    """

    def f(x):
        return sin(3 * x) + 5e-1 * cos(30 * x)

    def u(x):
        return np.exp(2 * np.pi * 1j * x)

    f0 = Chebtech.initfun_fixedlen(f, 100)
    f1 = Chebtech.initfun_adaptive(f)
    f2 = Chebtech.initfun_adaptive(u)

    return {"f0": f0, "f1": f1, "f2": f2}


def test_plotcoeffs(plotting_fixtures):
    """Test the plotcoeffs method of Chebtech.

    This test verifies that the plotcoeffs method of Chebtech objects
    can be called without errors. It creates a matplotlib figure and
    axes, then calls the plotcoeffs method with the axes as an argument
    for two different Chebtech objects.

    Args:
        plotting_fixtures: Fixture providing test Chebtech objects.
    """
    _fig, ax = plt.subplots()
    plotting_fixtures["f0"].plotcoeffs(ax=ax)
    plotting_fixtures["f1"].plotcoeffs(ax=ax, color="r")
