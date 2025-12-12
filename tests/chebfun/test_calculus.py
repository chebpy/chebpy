"""Unit-tests for Chebfun calculus operations.

This module contains tests for the calculus operations of Chebfun,
including sum, diff, cumsum, and dot product.
"""

import numpy as np
import pytest

from chebpy.chebfun import Chebfun

from ..utilities import exp, sin


@pytest.fixture
def calculus_fixtures():
    """Create Chebfun objects for testing calculus operations.

    This fixture creates several Chebfun objects with different characteristics
    for testing various calculus operations.

    Returns:
        dict: Dictionary containing:
            f1: Chebfun representing sin(4x - 1.4)
            f2: Chebfun representing exp(x)
            f3: Chebfun representing x^2
            f4: Chebfun representing x^3
    """

    def f(x):
        return sin(4 * x - 1.4)

    def g(x):
        return exp(x)

    f1 = Chebfun.initfun_adaptive(f, [-1, 1])
    f2 = Chebfun.initfun_adaptive(g, [-1, 1])
    f3 = Chebfun.initfun_adaptive(lambda x: x**2, [-1, 1])
    f4 = Chebfun.initfun_adaptive(lambda x: x**3, [-1, 1])

    return {"f1": f1, "f2": f2, "f3": f3, "f4": f4}


def test_sum(calculus_fixtures):
    """Test the sum method of Chebfun objects.

    This test verifies that the sum method correctly computes the definite
    integral of a function over its domain.

    Args:
        calculus_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = calculus_fixtures["f1"]
    f2 = calculus_fixtures["f2"]

    # For sin(4x - 1.4), the integral over [-1, 1] is approximately 0.3729
    assert abs(f1.sum() - 0.372895407327895) < 1e-4

    # For exp(x), the integral over [-1, 1] is approximately e - 1/e
    assert abs(f2.sum() - (np.exp(1) - np.exp(-1))) < 1e-10


def test_diff(calculus_fixtures):
    """Test the diff method of Chebfun objects.

    This test verifies that the diff method correctly computes the derivative
    of a function.

    Args:
        calculus_fixtures: Fixture providing test Chebfun objects.
    """
    f3 = calculus_fixtures["f3"]
    f4 = calculus_fixtures["f4"]

    # Derivative of x^2 is 2x
    df3 = f3.diff()
    xx = np.linspace(-1, 1, 100)
    assert np.max(np.abs(df3(xx) - 2 * xx)) < 1e-10

    # Derivative of x^3 is 3x^2
    df4 = f4.diff()
    assert np.max(np.abs(df4(xx) - 3 * xx**2)) < 1e-10


def test_cumsum(calculus_fixtures):
    """Test the cumsum method of Chebfun objects.

    This test verifies that the cumsum method correctly computes the indefinite
    integral of a function.

    Args:
        calculus_fixtures: Fixture providing test Chebfun objects.
    """
    f3 = calculus_fixtures["f3"]
    f4 = calculus_fixtures["f4"]

    # Indefinite integral of x^2 is x^3/3 + C
    # We need to check that the derivative of the indefinite integral is the original function
    assert (f3.cumsum().diff() - f3).isconst

    # Indefinite integral of x^3 is x^4/4 + C
    assert (f4.cumsum().diff() - f4).isconst


def test_dot(calculus_fixtures):
    """Test the dot method of Chebfun objects.

    This test verifies that the dot method correctly computes the inner product
    of two functions.

    Args:
        calculus_fixtures: Fixture providing test Chebfun objects.
    """
    f3 = calculus_fixtures["f3"]
    f4 = calculus_fixtures["f4"]

    # Inner product of x^2 and x^3 over [-1, 1] is 0 (odd function)
    assert abs(f3.dot(f4)) < 1e-10

    # Inner product of x^2 and x^2 over [-1, 1] is 2/5
    assert abs(f3.dot(f3) - 2 / 5) < 1e-10


def test_dot_commute(calculus_fixtures):
    """Test that the dot method is commutative.

    This test verifies that the dot method is commutative,
    i.e., f.dot(g) = g.dot(f).

    Args:
        calculus_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = calculus_fixtures["f1"]
    f2 = calculus_fixtures["f2"]

    assert abs(f1.dot(f2) - f2.dot(f1)) < 1e-10


def test_dot_empty(emptyfun, calculus_fixtures):
    """Test the dot method with an empty Chebfun.

    This test verifies that the dot method with an empty Chebfun
    returns 0.

    Args:
        emptyfun: Fixture providing an empty Chebfun object.
        calculus_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = calculus_fixtures["f1"]

    assert emptyfun.dot(f1) == 0
    assert f1.dot(emptyfun) == 0


def test_diff_order(calculus_fixtures):
    """Test the diff method with integer order argument.

    This test verifies that diff(n) correctly computes the n-th derivative
    of a function.

    Args:
        calculus_fixtures: Fixture providing test Chebfun objects.
    """
    f3 = calculus_fixtures["f3"]  # x^2
    f4 = calculus_fixtures["f4"]  # x^3

    # Second derivative of x^2 is 2 (constant)
    d2f3 = f3.diff(2)
    xx = np.linspace(-1, 1, 100)
    assert np.max(np.abs(d2f3(xx) - 2)) < 1e-10

    # Second derivative of x^3 is 6x
    d2f4 = f4.diff(2)
    assert np.max(np.abs(d2f4(xx) - 6 * xx)) < 1e-10

    # Third derivative of x^3 is 6 (constant)
    d3f4 = f4.diff(3)
    assert np.max(np.abs(d3f4(xx) - 6)) < 1e-10

    # Fourth derivative of x^3 should be zero
    d4f4 = f4.diff(4)
    assert np.max(np.abs(d4f4(xx))) < 1e-10


def test_diff_successive(calculus_fixtures):
    """Test that successive diff() calls are equivalent to diff(n).

    This test verifies that f.diff().diff() is equivalent to f.diff(2).

    Args:
        calculus_fixtures: Fixture providing test Chebfun objects.
    """
    f4 = calculus_fixtures["f4"]  # x^3

    # Two successive diff() calls
    d2f4_successive = f4.diff().diff()

    # Single diff(2) call
    d2f4_direct = f4.diff(2)

    # Compare results
    xx = np.linspace(-1, 1, 100)
    assert np.max(np.abs(d2f4_successive(xx) - d2f4_direct(xx))) < 1e-10


def test_norm(calculus_fixtures):
    """Test the norm method of Chebfun objects.

    This test verifies that the norm method correctly computes the L2 norm
    of a function, which is defined as sqrt(integral(f^2)).

    Args:
        calculus_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = calculus_fixtures["f1"]
    f2 = calculus_fixtures["f2"]
    f3 = calculus_fixtures["f3"]  # x^2

    # Norm of exp(x) over [-1, 1]
    # integral(exp(2x)) = [exp(2x)/2] from -1 to 1 = (e^2 - e^{-2})/2
    expected_norm_f2 = np.sqrt((np.exp(2) - np.exp(-2)) / 2)
    assert abs(f2.norm() - expected_norm_f2) < 1e-10

    # Norm of x^2 over [-1, 1]
    # integral(x^4) = 2/5 for [-1, 1]
    expected_norm_f3 = np.sqrt(2 / 5)
    assert abs(f3.norm() - expected_norm_f3) < 1e-10

    # Norm should always be non-negative
    assert f1.norm() >= 0
    assert f2.norm() >= 0


def test_norm_empty(emptyfun):
    """Test the norm method with an empty Chebfun.

    This test verifies that the norm of an empty Chebfun is 0.

    Args:
        emptyfun: Fixture providing an empty Chebfun object.
    """
    assert emptyfun.norm() == 0


def test_norm_relation_to_dot(calculus_fixtures):
    """Test the relationship between norm and dot product.

    This test verifies that ||f||^2 = f.dot(f).

    Args:
        calculus_fixtures: Fixture providing test Chebfun objects.
    """
    f2 = calculus_fixtures["f2"]
    f3 = calculus_fixtures["f3"]

    # ||f||^2 should equal f.dot(f)
    assert abs(f2.norm() ** 2 - f2.dot(f2)) < 1e-10
    assert abs(f3.norm() ** 2 - f3.dot(f3)) < 1e-10


class TestChebfunCalculusEdgeCases:
    """Additional edge case tests for Chebfun calculus operations."""

    def test_diff_edge_cases(self):
        """Test diff with n=0 and negative n."""
        from chebpy import chebfun

        f = chebfun(lambda x: x**3, [-1, 1])

        # n=0 should return original function
        f0 = f.diff(0)
        assert f0 == f

        # Negative n should raise ValueError
        with pytest.raises(ValueError, match="non-negative"):
            f.diff(-1)

    def test_diff_higher_order(self):
        """Test higher order derivatives."""
        from chebpy import chebfun

        f = chebfun(lambda x: x**4, [-1, 1])
        # Fourth derivative of x^4 is 24
        f4 = f.diff(4)
        xx = np.linspace(-1, 1, 20)
        assert np.allclose(f4(xx), 24, atol=1e-10)

    def test_multipiece_cumsum_with_many_pieces(self):
        """Test cumsum with more than 2 pieces."""
        from chebpy import chebfun

        # Test that continuity is enforced across multiple pieces
        f = chebfun(lambda x: x * 0 + 1, [-1, -0.5, 0, 0.5, 1])
        f_int = f.cumsum()
        # Check that it's continuous at breakpoints
        xx = np.array([-0.5, 0, 0.5])
        left = xx - 1e-10
        right = xx + 1e-10
        assert np.allclose(f_int(left), f_int(right), atol=1e-5)

    def test_cumsum_multipiece_continuity(self):
        """Test that cumsum maintains continuity across pieces."""
        from chebpy import chebfun

        f = chebfun(lambda x: np.sign(x), [-1, 0, 1])
        f_cumsum = f.cumsum()
        # Check continuity at x=0
        left_val = f_cumsum(-1e-10)
        right_val = f_cumsum(1e-10)
        # Should be continuous (within tolerance)
        assert np.abs(left_val - right_val) < 1e-6

    def test_norm_l1(self):
        """Test L1 norm."""
        from chebpy import chebfun

        f = chebfun(lambda x: x, [-1, 1])
        # L1 norm = integral(|x|) from -1 to 1 = 2 * integral(x) from 0 to 1 = 1
        norm_l1 = f.norm(p=1)
        assert np.isclose(norm_l1, 1.0, atol=1e-10)

    def test_norm_l2(self):
        """Test L2 norm (default)."""
        from chebpy import chebfun

        f = chebfun(lambda x: x, [-1, 1])
        # L2 norm = sqrt(integral(x^2)) from -1 to 1 = sqrt(2/3)
        norm_l2 = f.norm()
        assert np.isclose(norm_l2, np.sqrt(2 / 3), atol=1e-10)

    def test_norm_linf(self):
        """Test L-infinity norm."""
        from chebpy import chebfun

        f = chebfun(lambda x: x**2, [-1, 1])
        # L-inf norm = max|x^2| on [-1, 1] = 1
        norm_linf = f.norm(np.inf)
        assert np.isclose(norm_linf, 1.0, atol=1e-10)

    def test_norm_l3(self):
        """Test L3 norm."""
        from chebpy import chebfun

        f = chebfun(lambda x: x * 0 + 1, [-1, 1])  # Constant function workaround
        # L3 norm of constant 1 = (integral(1) from -1 to 1)^(1/3) = 2^(1/3)
        norm_l3 = f.norm(p=3)
        assert np.isclose(norm_l3, 2 ** (1 / 3), atol=1e-10)

    def test_norm_invalid_p(self):
        """Test norm with invalid p value."""
        from chebpy import chebfun

        f = chebfun(lambda x: x, [-1, 1])
        with pytest.raises(ValueError, match="must be positive"):
            f.norm(p=-1)

        with pytest.raises(ValueError, match="must be positive"):
            f.norm(p=0)

    def test_norm_l2_multipiece(self):
        """Test L2 norm on multipiece function."""
        from chebpy import chebfun

        f = chebfun(lambda x: x, [-1, 0, 1])
        norm = f.norm()  # L2 by default
        # L2 norm of x on [-1,1] = sqrt(2/3)
        assert np.isclose(norm, np.sqrt(2 / 3), atol=1e-10)

    def test_norm_linf_with_multiple_extrema(self):
        """Test L-infinity norm with multiple local maxima."""
        from chebpy import chebfun

        f = chebfun(lambda x: np.sin(3 * x), [-np.pi, np.pi])
        norm_inf = f.norm(np.inf)
        # Should be 1 (max of |sin|)
        assert np.isclose(norm_inf, 1.0, atol=1e-10)
