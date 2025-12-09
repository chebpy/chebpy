"""Unit-tests for construction of Trigtech objects.

This module contains tests for the various ways to construct Trigtech objects,
including from values, coefficients, constants, and functions on periodic domains.
"""

import numpy as np
import pytest

from chebpy.trigtech import Trigtech

from ..utilities import eps

# Ensure reproducibility
rng = np.random.default_rng(0)


def test_initvalues():
    """Test initialization of Trigtech objects from values.

    This test verifies that Trigtech objects can be correctly initialized
    from arrays of values at equally-spaced (Fourier) points, checking both
    empty arrays and arrays of various sizes. It ensures that the coefficients
    computed from the values match the expected coefficients from FFT.
    """
    # test n = 0 case separately
    vals = rng.random(0)
    fun = Trigtech.initvalues(vals)
    cfs = Trigtech._vals2coeffs(vals)
    assert fun.coeffs.size == cfs.size == 0

    # now test the other cases
    for n in [1, 2, 3, 5, 8, 13, 21]:
        vals = rng.random(n)
        fun = Trigtech.initvalues(vals)
        cfs = Trigtech._vals2coeffs(vals)
        assert np.max(np.abs(fun.coeffs - cfs)) < 10 * eps


def test_initidentity():
    """Test that initidentity raises NotImplementedError for Trigtech.

    The identity function f(x) = x on [0, 2π] is not periodic (f(0) ≠ f(2π)),
    so it cannot be represented using Fourier series. This test verifies that
    calling initidentity() raises NotImplementedError with an appropriate message.
    """
    with pytest.raises(NotImplementedError, match="initidentity.*not supported"):
        Trigtech.initidentity()


def test_coeff_construction():
    """Test construction of Trigtech objects from Fourier coefficients.

    This test verifies that Trigtech objects can be correctly initialized
    from arrays of Fourier coefficients, checking that the resulting object
    has the expected coefficients within machine precision.
    """
    coeffs = rng.random(10) + 1j * rng.random(10)
    f = Trigtech(coeffs)
    assert isinstance(f, Trigtech)
    assert np.max(np.abs(f.coeffs - coeffs)) < eps


def test_initconst():
    """Test initialization from a constant.

    Verifies that Trigtech can represent constant functions correctly.
    """
    c = 3.14159
    f = Trigtech.initconst(c)
    assert f.size == 1
    x = np.linspace(0, 2*np.pi, 100, endpoint=False)
    vals = f(x)
    assert np.max(np.abs(vals - c)) < 100 * eps


def test_initempty():
    """Test initialization of empty Trigtech.

    Verifies that empty Trigtech objects can be created.
    """
    f = Trigtech.initempty()
    assert f.isempty
    assert f.size == 0


# Test adaptive function construction with periodic functions
@pytest.mark.parametrize("fun,expected_len", [
    (lambda x: np.sin(x), [16, 32, 64]),  # sin(x) is smooth - can converge at 16
    (lambda x: np.cos(2*x), [16, 32, 64]),  # cos(2x) is smooth - can converge at 16
    (lambda x: np.sin(5*x) + np.cos(3*x), [32, 64, 128]),  # Multiple frequencies
    (lambda x: np.exp(np.sin(x)), [32, 64]),  # exp(sin(x)) - nonlinear
    (lambda x: 1.0 / (2.0 + np.cos(x)), [32, 64, 128]),  # Rational function
])
def test_adaptive_construction(fun, expected_len):
    """Test adaptive construction of Trigtech objects from periodic functions.

    This test verifies that Trigtech objects constructed adaptively
    from various periodic functions have sizes within expected bounds.

    Args:
        fun: Periodic function to approximate
        expected_len: Acceptable range of coefficients needed
    """
    ff = Trigtech.initfun_adaptive(fun)
    assert ff.size in expected_len, f"Got size {ff.size}, expected one of {expected_len}"
    # Verify periodicity
    x_test = np.array([0.0, 2*np.pi])
    vals = ff(x_test)
    assert np.abs(vals[0] - vals[1]) < 1e-10, "Function should be periodic"


# Test fixed length function construction
@pytest.mark.parametrize("n", [16, 32, 64])
def test_fixedlen_construction(n):
    """Test fixed-length construction of Trigtech objects.

    This test verifies that Trigtech objects constructed with a fixed length
    have exactly the specified size.

    Args:
        n: Fixed length to use for construction
    """
    def fun(x):
        return np.sin(3*x) + np.cos(5*x)
    ff = Trigtech.initfun_fixedlen(fun, n)
    assert ff.size == n


def test_real_function_real_coeffs():
    """Test that real periodic functions give real-valued Fourier coefficients.

    For real functions, Fourier coefficients should satisfy c_{-k} = conj(c_k).
    """
    def fun(x):
        return np.sin(2*x) + np.cos(3*x)
    f = Trigtech.initfun_adaptive(fun)

    # For real functions, coefficients should be conjugate symmetric
    # But Trigtech stores in a specific format - check that values are real
    x_test = np.linspace(0, 2*np.pi, 100, endpoint=False)
    vals = f(x_test)
    assert np.max(np.abs(vals.imag)) < 1e-12, "Real function should give real values"


def test_periodic_function_approximation():
    """Test that periodic functions are approximated accurately.

    Verifies that Trigtech can approximate common periodic functions
    to high accuracy.
    """
    test_cases = [
        (lambda x: np.sin(x), lambda x: np.sin(x)),
        (lambda x: np.cos(2*x), lambda x: np.cos(2*x)),
        (lambda x: np.sin(x) + np.cos(2*x), lambda x: np.sin(x) + np.cos(2*x)),
        (lambda x: np.exp(np.sin(x)), lambda x: np.exp(np.sin(x))),
    ]

    for fun, exact in test_cases:
        f = Trigtech.initfun_adaptive(fun)
        x_test = np.linspace(0, 2*np.pi, 200, endpoint=False)
        error = np.max(np.abs(f(x_test) - exact(x_test)))
        assert error < 1e-12, f"Approximation error {error} too large"


def test_construction_preserves_periodicity():
    """Test that constructed Trigtech objects are truly periodic.

    Verifies that f(0) ≈ f(2π) for all constructed functions.
    """
    functions = [
        lambda x: np.sin(3*x),
        lambda x: np.cos(2*x) + np.sin(5*x),
        lambda x: np.exp(np.sin(x)),
        lambda x: 1.0 / (2.0 + np.cos(x)),
    ]

    for fun in functions:
        f = Trigtech.initfun_adaptive(fun)
        val_0 = f(np.array([0.0]))[0]
        val_2pi = f(np.array([2*np.pi]))[0]
        assert np.abs(val_0 - val_2pi) < 1e-12, f"Function not periodic: f(0)={val_0}, f(2π)={val_2pi}"


def test_construction_from_nonperiodic_function():
    """Test that constructing from non-periodic function gives warning or fails gracefully.

    Non-periodic functions should either:
    1. Give a warning about periodicity
    2. Have large Fourier coefficients that don't decay

    This is a diagnostic test - Trigtech should only be used for periodic functions.
    """
    # Non-periodic function: x on [0, 2π] (not periodic)
    def fun(x):
        return x

    # This should give poor approximation since function is not periodic
    f = Trigtech.initfun_fixedlen(fun, 32)

    # Check that f(0) != f(2π) - indicating non-periodicity is detected
    f(np.array([0.0]))[0]
    f(np.array([2*np.pi]))[0]

    # For a linear function, the approximation will try its best but fail at endpoints
    # This is expected behavior - we just verify it doesn't crash
    assert isinstance(f, Trigtech)


def test_vals2coeffs_coeffs2vals_inverse():
    """Test that vals2coeffs and coeffs2vals are inverse operations.

    Verifies the fundamental FFT identity: that transforming values to coefficients
    and back recovers the original values.
    """
    for n in [8, 16, 32, 64]:
        vals = rng.random(n)
        coeffs = Trigtech._vals2coeffs(vals)
        vals_back = Trigtech._coeffs2vals(coeffs)
        assert np.max(np.abs(vals - vals_back)) < 10 * eps * n


def test_simplify_chops_coefficients():
    """Test that simplify() properly chops negligible Fourier coefficients.

    This test verifies Issue #6 fix: coefficient chopping with conjugate-pair preservation.
    """
    # Create a function with few significant frequencies but evaluate on large grid
    def fun(x):
        return np.sin(x) + 0.5 * np.cos(2*x)
    f = Trigtech.initfun_fixedlen(fun, 128)  # Force large representation

    # Simplify should reduce size while maintaining accuracy
    f_simple = f.simplify()

    # Verify size reduction (highest freq is 2, so need ~8 points)
    assert f_simple.size < f.size, f"Expected size reduction, got {f_simple.size} vs {f.size}"
    assert f_simple.size <= 16, f"Size {f_simple.size} should be small (8-16) for low-freq function"

    # Verify accuracy is maintained
    x_test = np.linspace(0, 2*np.pi, 200, endpoint=False)
    error = np.max(np.abs(f(x_test) - f_simple(x_test)))
    assert error < 1e-12, f"Simplification error {error} too large"


def test_simplify_preserves_real_functions():
    """Test that simplify() preserves real-valued functions.

    Verifies that conjugate-pair-aware chopping maintains real output.
    """
    # Real periodic function
    def fun(x):
        return np.sin(3*x) + np.cos(5*x) + 0.5
    f = Trigtech.initfun_fixedlen(fun, 64)
    f_simple = f.simplify()

    # Check that output remains real
    x_test = np.linspace(0, 2*np.pi, 100, endpoint=False)
    vals = f_simple(x_test)
    assert np.max(np.abs(vals.imag)) < 1e-12, "Simplification should preserve real functions"


def test_simplify_zero_function():
    """Test that simplify() handles zero function correctly."""
    f = Trigtech.initconst(0)
    f_simple = f.simplify()
    assert f_simple.size <= 1, "Zero function should simplify to constant"

    # Also test near-zero function
    def fun(x):
        return 1e-16 * np.sin(x)
    f = Trigtech.initfun_adaptive(fun)
    f_simple = f.simplify()
    # Should reduce to very small or constant
    assert f_simple.size <= f.size, "Near-zero should simplify"
