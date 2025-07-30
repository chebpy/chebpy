"""Unit-tests for complex Chebfun operations.

This module contains tests for complex Chebfun operations,
including roots, rho ellipse construction, calculus, and real/imag methods.
"""

import numpy as np
import pytest

from chebpy.core.chebfun import Chebfun


@pytest.fixture
def complex_function():
    """Create a complex Chebfun function for testing.

    This fixture creates a Chebfun object representing the complex function
    exp(π·i·x), which is used in multiple tests for complex function operations.

    Returns:
        Chebfun: A Chebfun object representing exp(π·i·x)
    """
    return Chebfun.initfun_adaptive(lambda x: np.exp(np.pi * 1j * x), [-1, 1])


def test_roots(complex_function):
    """Test finding roots of complex Chebfun functions.

    This test verifies that the roots method correctly identifies the roots
    of various complex functions derived from the base complex function.
    It checks:
    1. The original function has no roots
    2. The function minus 1 has a root at x=0
    3. The function minus i has a root at x=0.5
    4. The function plus 1 has roots at x=-1 and x=1
    5. The function plus i has a root at x=-0.5

    Args:
        complex_function: Fixture providing a complex Chebfun function
    """
    z = complex_function
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


def test_rho_ellipse_construction(complex_function):
    """Test construction of rho ellipses with complex functions.

    This test verifies that rho ellipses constructed from complex functions
    have the expected properties. It checks that the function values at
    specific points satisfy certain relationships that are characteristic
    of ellipses in the complex plane.

    Args:
        complex_function: Fixture providing a complex Chebfun function
    """
    z = complex_function
    zz = 1.2 * z
    e = 0.5 * (zz + 1 / zz)
    assert e(1) - e(-1) == pytest.approx(0, abs=1e-14)
    assert e(0) + e(-1) == pytest.approx(0, abs=1e-14)
    assert e(0) + e(1) == pytest.approx(0, abs=1e-14)


def test_calculus(complex_function):
    """Test calculus operations on complex Chebfun functions.

    This test verifies that calculus operations (sum, cumsum, diff)
    work correctly on complex functions. It checks:
    1. The sum of the function over [-1,1] is approximately 0
    2. Differentiating the cumulative sum returns the original function

    Args:
        complex_function: Fixture providing a complex Chebfun function
    """
    z = complex_function
    assert np.allclose([z.sum()], [0])
    assert (z.cumsum().diff() - z).isconst
    assert (z - z.cumsum().diff()).isconst


def test_real_imag(complex_function):
    """Test real and imaginary part extraction from complex functions.

    This test verifies that the real() and imag() methods correctly extract
    the real and imaginary parts of complex Chebfun functions. It checks:
    1. The coefficients of the real part match the real parts of the original coefficients
    2. The coefficients of the imaginary part match the imaginary parts of the original coefficients
    3. The real part of a real function is the same function
    4. The imaginary part of a real function is a zero constant function

    Args:
        complex_function: Fixture providing a complex Chebfun function
    """
    z = complex_function
    # check definition of real and imaginary
    zreal = z.real()
    zimag = z.imag()
    np.testing.assert_equal(zreal.funs[0].coeffs, np.real(z.funs[0].coeffs))
    np.testing.assert_equal(zimag.funs[0].coeffs, np.imag(z.funs[0].coeffs))
    # check real part of real chebtech is the same chebtech
    assert zreal.real() == zreal
    # check imaginary part of real chebtech is the zero chebtech
    assert zreal.imag().isconst
    assert zreal.imag().funs[0].coeffs[0] == 0
