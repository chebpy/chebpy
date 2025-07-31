"""Unit-tests for Chebfun evaluation.

This module contains tests for evaluating Chebfun objects at points,
including empty Chebfun objects, point evaluation, singleton arrays,
breakpoints, and points outside the interval of definition.
"""

import numpy as np
import pytest

from chebpy.core.chebfun import Chebfun

from ..utilities import cos, eps, exp, sin


@pytest.fixture
def evaluation_fixtures():
    """Create Chebfun objects for testing evaluation.

    This fixture creates several Chebfun objects with different characteristics
    for testing various aspects of evaluation.

    Returns:
        dict: Dictionary containing:
            f0: Empty Chebfun
            f1: Chebfun on [-1, 1]
            f2: Chebfun on a piecewise domain
    """
    f0 = Chebfun.initempty()
    f1 = Chebfun.initfun_adaptive(lambda x: x**2, [-1, 1])
    f2 = Chebfun.initfun_adaptive(lambda x: x**2, [-1, 0, 1, 2])
    return {"f0": f0, "f1": f1, "f2": f2}


def test__call__empty_chebfun(evaluation_fixtures):
    """Test evaluation of an empty Chebfun.

    This test verifies that evaluating an empty Chebfun at any points
    returns an empty array.

    Args:
        evaluation_fixtures: Fixture providing test Chebfun objects.
    """
    f0 = evaluation_fixtures["f0"]
    assert f0(np.linspace(-1, 1, 100)).size == 0


def test__call__empty_array(evaluation_fixtures):
    """Test evaluation at an empty array.

    This test verifies that evaluating any Chebfun at an empty array
    returns an empty array.

    Args:
        evaluation_fixtures: Fixture providing test Chebfun objects.
    """
    f0 = evaluation_fixtures["f0"]
    f1 = evaluation_fixtures["f1"]
    f2 = evaluation_fixtures["f2"]
    assert f0(np.array([])).size == 0
    assert f1(np.array([])).size == 0
    assert f2(np.array([])).size == 0


def test__call__point_evaluation(evaluation_fixtures):
    """Test evaluation at a single point.

    This test verifies that evaluating a Chebfun at a single point
    returns a scalar value.

    Args:
        evaluation_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = evaluation_fixtures["f1"]
    assert np.isscalar(f1(0.1))


def test__call__singleton(evaluation_fixtures):
    """Test evaluation at singleton arrays.

    This test verifies that evaluating a Chebfun at a singleton array
    (an array with a single element) returns an array with a single element,
    and that the result is the same regardless of how the singleton is represented.

    Args:
        evaluation_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = evaluation_fixtures["f1"]
    a = f1(np.array(0.1))
    b = f1(np.array([0.1]))
    c = f1([0.1])
    assert a.size == 1
    assert b.size == 1
    assert c.size == 1
    assert np.equal(a, b).all()
    assert np.equal(b, c).all()
    assert np.equal(a, c).all()


def test__call__breakpoints(evaluation_fixtures):
    """Test evaluation at breakpoints.

    This test verifies that evaluating a Chebfun at its breakpoints
    returns the correct values.

    Args:
        evaluation_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = evaluation_fixtures["f1"]
    f2 = evaluation_fixtures["f2"]
    x1 = f1.breakpoints
    x2 = f2.breakpoints
    assert np.equal(f1(x1), [1, 1]).all()
    assert np.equal(f2(x2), [1, 0, 1, 4]).all()


def test__call__outside_interval(evaluation_fixtures):
    """Test evaluation outside the interval of definition.

    This test verifies that evaluating a Chebfun at points outside its
    interval of definition returns finite values.

    Args:
        evaluation_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = evaluation_fixtures["f1"]
    f2 = evaluation_fixtures["f2"]
    x = np.linspace(-3, 3, 100)
    assert np.isfinite(f1(x)).all()
    assert np.isfinite(f2(x)).all()


def test__call__general_evaluation():
    """Test general evaluation of Chebfun objects.

    This test verifies that Chebfun objects can be evaluated at a large number
    of points and that the results are accurate within a specified tolerance.
    It tests evaluation on continuous and piecewise domains.
    """

    def f(x):
        return sin(4 * x) + exp(cos(14 * x)) - 1.4

    npts = 50000
    dom1 = [-1, 1]
    dom2 = [-1, 0, 1]
    dom3 = [-2, -0.3, 1.2]
    ff1 = Chebfun.initfun_adaptive(f, dom1)
    ff2 = Chebfun.initfun_adaptive(f, dom2)
    ff3 = Chebfun.initfun_adaptive(f, dom3)
    x1 = np.linspace(dom1[0], dom1[-1], npts)
    x2 = np.linspace(dom2[0], dom2[-1], npts)
    x3 = np.linspace(dom3[0], dom3[-1], npts)
    assert np.max(np.abs(f(x1) - ff1(x1))) <= 5e1 * eps
    assert np.max(np.abs(f(x2) - ff2(x2))) <= 5e1 * eps
    assert np.max(np.abs(f(x3) - ff3(x3))) <= 5e1 * eps
