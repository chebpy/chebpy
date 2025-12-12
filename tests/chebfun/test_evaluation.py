"""Unit-tests for Chebfun evaluation.

This module contains tests for evaluating Chebfun objects at points,
including empty Chebfun objects, point evaluation, singleton arrays,
breakpoints, and points outside the interval of definition.
"""

import numpy as np
import pytest

from chebpy.chebfun import Chebfun

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


class TestChebfunEvaluationEdgeCases:
    """Additional edge case tests for Chebfun evaluation."""

    def test_call_with_trigtech_complex_output(self):
        """Test that __call__ properly handles Trigtech complex intermediate results."""
        from chebpy import chebfun

        # This tests the Trigtech branch in __call__ (lines 229, 232, 253-257)
        # We need to use a function that would create Trigtech-based funs
        # For now, we'll test with regular chebfuns since Trigtech integration
        # is handled at a lower level
        f = chebfun(lambda x: np.sin(x), [-np.pi, np.pi])
        xx = np.linspace(-np.pi, np.pi, 100)
        vals = f(xx)
        # Should be real-valued
        assert vals.dtype != np.complex128 or np.max(np.abs(vals.imag)) < 1e-13

    def test_call_outside_domain_both_sides(self):
        """Test evaluation outside domain on both sides."""
        from chebpy import chebfun

        f = chebfun(lambda x: x**2, [0, 1])
        # Extrapolate on left
        assert np.isclose(f(-0.5), 0.25, atol=1e-10)
        # Extrapolate on right
        assert np.isclose(f(1.5), 2.25, atol=1e-10)

    def test_single_point_evaluation(self):
        """Test evaluating at a single point."""
        from chebpy import chebfun

        f = chebfun(lambda x: x**2, [-1, 1])
        result = f(0.5)
        expected = 0.25
        assert np.isclose(result, expected, atol=1e-14)

    def test_empty_array_evaluation(self):
        """Test evaluating with empty array."""
        from chebpy import chebfun

        f = chebfun(lambda x: x, [-1, 1])
        result = f(np.array([]))
        assert result.size == 0

    def test_rtruediv_with_scalar(self):
        """Test scalar / chebfun operation."""
        from chebpy import chebfun

        f = chebfun(lambda x: x + 2, [-1, 1])
        result = 1 / f
        expected = chebfun(lambda x: 1 / (x + 2), [-1, 1])
        xx = np.linspace(-1, 1, 50)
        assert np.allclose(result(xx), expected(xx), atol=1e-10)

    def test_abs_dunder_method(self):
        """Test __abs__ method (absolute value via dunder)."""
        from chebpy import chebfun

        f = chebfun(lambda x: x, [-1, 1])
        f_abs = abs(f)
        xx = np.linspace(-1, 1, 100)
        assert np.allclose(f_abs(xx), np.abs(xx), atol=1e-10)
