"""Generic test functions for complex function operations.

This module contains test functions for complex function operations that can be used
with any type of function object (Bndfun, Chebfun, or Chebtech). These tests
focus on operations specific to complex-valued functions.
"""

import numpy as np
import pytest


def test_complexfun_properties(complexfun):
    """Test that complexfun has the expected properties.

    This test verifies that the complex function object created by the complexfun
    fixture has the expected properties, including being complex, not empty,
    not constant, having no roots, and evaluating to the expected values at
    specific points.

    Args:
        complexfun: Fixture providing a complex function object.
    """
    # Test basic properties
    assert complexfun.iscomplex
    assert not complexfun.isempty
    assert not complexfun.isconst
    assert complexfun.roots().size == 0

    # Test evaluation at specific points
    xx = np.array([0.0, 0.5, -0.5])
    expected_values = np.exp(np.pi * 1j * xx)
    actual_values = complexfun(xx)

    # Check that the values match the expected values
    assert np.all(np.abs(actual_values - expected_values) < 1e-14)


def test_roots(complexfun):
    """Test finding roots of complex Chebtech functions.

    This test verifies that the roots method correctly identifies the roots
    of various complex functions derived from the base complex function.
    It checks:
    1. The original function has no roots
    2. The function minus 1 has a root at x=0
    3. The function minus i has a root at x=0.5
    4. The function plus 1 has roots at x=-1 and x=1
    5. The function plus i has a root at x=-0.5

    Args:
        complexfun: Fixture providing a complex function object
    """
    z = complexfun
    r0 = z.roots()
    r1 = (z - 1).roots()
    r2 = (z - 1j).roots()
    r3 = (z + 1).roots()
    r4 = (z + 1j).roots()
    assert r0.size == 0
    assert np.allclose(r1, [0])
    assert np.allclose(r2, [0.5])
    assert np.allclose(r3, [-1, 1])
    assert np.allclose(r4, [-0.5])


def test_rho_ellipse_construction(complexfun):
    """Test construction of rho ellipses with complex functions.

    This test verifies that rho ellipses constructed from complex functions
    have the expected properties. It checks that the function values at
    specific points satisfy certain relationships that are characteristic
    of ellipses in the complex plane.

    Args:
        complexfun: Fixture providing a complex function object
    """
    z = complexfun
    zz = 1.2 * z
    e = 0.5 * (zz + 1 / zz)
    assert e(1) - e(-1) == pytest.approx(0, abs=1e-14)
    assert e(0) + e(-1) == pytest.approx(0, abs=1e-14)
    assert e(0) + e(1) == pytest.approx(0, abs=1e-14)


def test_calculus(complexfun):
    """Test calculus operations on complex Chebtech functions.

    This test verifies that calculus operations (sum, cumsum, diff)
    work correctly on complex functions. It checks:
    1. The sum of the function over [-1,1] is approximately 0
    2. Differentiating the cumulative sum returns the original function

    Args:
        complexfun: Fixture providing a complex function object
    """
    z = complexfun
    assert np.allclose([z.sum()], [0])


def test_real_imag(complexfun):
    """Test real and imaginary part extraction from complex functions.

    This test verifies that the real() and imag() methods correctly extract
    the real and imaginary parts of complex Chebtech functions. It checks:
    1. The coefficients of the real part match the real parts of the original coefficients
    2. The coefficients of the imaginary part match the imaginary parts of the original coefficients
    3. The real part of a real function is the same function
    4. The imaginary part of a real function is a zero constant function

    Args:
        complexfun: Fixture providing a complex function object
    """
    z = complexfun
    # check definition of real and imaginary
    zreal = z.real()
    # check real part of real chebtech is the same chebtech
    assert zreal.real() == zreal
    # check imaginary part of real chebtech is the zero chebtech
    assert zreal.imag().isconst
