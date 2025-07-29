"""Unit-tests for Bndfun plotting methods."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from chebpy.core.bndfun import Bndfun
from chebpy.core.utilities import Interval

from ..utilities import cos, joukowsky, sin


@pytest.fixture
def plotting_fixtures():
    """Create fixtures for testing Bndfun plotting methods."""

    def f(x):
        return sin(1 * x) + 5e-1 * cos(10 * x) + 5e-3 * sin(100 * x)

    def u(x):
        return np.exp(2 * np.pi * 1j * x)

    subinterval = Interval(-6, 10)
    f0 = Bndfun.initfun_fixedlen(f, subinterval, 1000)
    f1 = Bndfun.initfun_adaptive(f, subinterval)
    f2 = Bndfun.initfun_adaptive(u, Interval(-1, 1))

    return {"f0": f0, "f1": f1, "f2": f2}



def test_plot(plotting_fixtures):
    """Test the plot method of Bndfun."""
    f0 = plotting_fixtures["f0"]
    fig, ax = plt.subplots()
    f0.plot(ax=ax, color="g", marker="o", markersize=2, linestyle="")


def test_plot_complex(plotting_fixtures):
    """Test plotting complex Bndfun objects."""
    f2 = plotting_fixtures["f2"]
    fig, ax = plt.subplots()
    # plot Bernstein ellipses
    for rho in np.arange(1.1, 2, 0.1):
        (np.exp(1j * 0.25 * np.pi) * joukowsky(rho * f2)).plot(ax=ax)


def test_plotcoeffs(plotting_fixtures):
    """Test the plotcoeffs method of Bndfun."""
    f0 = plotting_fixtures["f0"]
    f1 = plotting_fixtures["f1"]
    fig, ax = plt.subplots()
    f0.plotcoeffs(ax=ax)
    f1.plotcoeffs(ax=ax, color="r")
