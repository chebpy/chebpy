"""Unit-tests for Bndfun plotting methods."""

import matplotlib.pyplot as plt
import pytest

from chebpy.bndfun import Bndfun
from chebpy.utilities import Interval

from ..generic.plotting import test_plot, test_plot_complex  # noqa: F401
from ..utilities import cos, sin


@pytest.fixture
def plotting_fixtures():
    """Create fixtures for testing Bndfun plotting methods."""

    def f(x):
        return sin(1 * x) + 5e-1 * cos(10 * x) + 5e-3 * sin(100 * x)

    subinterval = Interval(-6, 10)
    f0 = Bndfun.initfun_fixedlen(f, subinterval, 1000)
    f1 = Bndfun.initfun_adaptive(f, subinterval)

    return {"f0": f0, "f1": f1}


def test_plotcoeffs(plotting_fixtures):
    """Test the plotcoeffs method of Bndfun."""
    f0 = plotting_fixtures["f0"]
    f1 = plotting_fixtures["f1"]
    fig, ax = plt.subplots()
    f0.plotcoeffs(ax=ax)
    f1.plotcoeffs(ax=ax, color="r")
