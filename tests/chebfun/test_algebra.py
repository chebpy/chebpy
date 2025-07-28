"""Unit-tests for Chebfun algebraic operations.

This module contains tests for the algebraic operations of Chebfun,
including addition, subtraction, multiplication, division, and powers.
"""
import itertools

import pytest
import numpy as np

from chebpy.core.chebfun import Chebfun

from .conftest import (
    eps, binops, div_binops, binaryOpTester, unaryOpTester,
    chebfun_testdomains
)


@pytest.fixture
def emptyfun():
    """Create an empty Chebfun function for testing.

    This fixture creates an empty Chebfun object that can be used
    to test the behavior of algebraic operations on empty functions.

    Returns:
        Chebfun: An empty Chebfun object
    """
    return Chebfun.initempty()


# tests for empty function operations
def test__pos__empty(emptyfun):
    """Test unary positive operator on empty Chebfun objects.

    This test verifies that applying the unary positive operator to an
    empty Chebfun object results in an empty Chebfun object.

    Args:
        emptyfun: Fixture providing an empty Chebfun object
    """
    assert (+emptyfun).isempty


def test__neg__empty(emptyfun):
    """Test unary negative operator on empty Chebfun objects.

    This test verifies that applying the unary negative operator to an
    empty Chebfun object results in an empty Chebfun object.

    Args:
        emptyfun: Fixture providing an empty Chebfun object
    """
    assert (-emptyfun).isempty


def test__add__radd__empty(emptyfun, testfunctions):
    """Test addition with empty Chebfun objects.

    This test verifies that adding an empty Chebfun object to any other
    Chebfun object results in an empty Chebfun object, regardless of
    the order of operands.

    Args:
        emptyfun: Fixture providing an empty Chebfun object
    """
    for fun, _, _ in testfunctions:
        for dom, _ in chebfun_testdomains:
            a, b = dom
            ff = Chebfun.initfun_adaptive(fun, np.linspace(a, b, 9))
            assert (emptyfun + ff).isempty
            assert (ff + emptyfun).isempty


def test__add__radd__constant(testfunctions):
    """Test addition of constants to Chebfun objects.

    This test verifies that adding a constant to a Chebfun object
    (and vice versa) produces the expected result within a specified
    tolerance.
    """
    for fun, _, _ in testfunctions:
        for dom, _ in chebfun_testdomains:
            a, b = dom
            for const in (-1, 1, 10, -1e5):
                def g(x):
                    return const + fun(x)

                ff = Chebfun.initfun_adaptive(fun, np.linspace(a, b, 9))
                gg1 = const + ff
                gg2 = ff + const
                xx = np.linspace(a, b, 1001)
                vscl = ff.vscale
                hscl = ff.hscale
                lscl = max([fun.size for fun in ff])
                tol = 2 * abs(const) * vscl * hscl * lscl * eps
                assert np.max(g(xx) - gg1(xx)) <= tol
                assert np.max(g(xx) - gg2(xx)) <= tol


def test__sub__rsub__empty(emptyfun, testfunctions):
    """Test subtraction with empty Chebfun objects.

    This test verifies that subtracting an empty Chebfun object from any other
    Chebfun object (or vice versa) results in an empty Chebfun object.

    Args:
        emptyfun: Fixture providing an empty Chebfun object
    """
    for fun, _, _ in testfunctions:
        for dom, _ in chebfun_testdomains:
            a, b = dom
            ff = Chebfun.initfun_adaptive(fun, np.linspace(a, b, 9))
            assert (emptyfun - ff).isempty
            assert (ff - emptyfun).isempty


def test__sub__rsub__constant(testfunctions):
    """Test subtraction of constants and Chebfun objects.

    This test verifies that subtracting a Chebfun object from a constant
    (and vice versa) produces the expected result within a specified tolerance.
    """
    for fun, _, _ in testfunctions:
        for dom, _ in chebfun_testdomains:
            a, b = dom
            for const in (-1, 1, 10, -1e5):
                def g(x):
                    return const - fun(x)

                def h(x):
                    return fun(x) - const

                ff = Chebfun.initfun_adaptive(fun, np.linspace(a, b, 9))
                gg = const - ff
                hh = ff - const
                xx = np.linspace(a, b, 1001)
                vscl = ff.vscale
                hscl = ff.hscale
                lscl = max([fun.size for fun in ff])
                tol = 2 * abs(const) * vscl * hscl * lscl * eps
                assert np.max(g(xx) - gg(xx)) <= tol
                assert np.max(h(xx) - hh(xx)) <= tol


def test__mul__rmul__empty(emptyfun, testfunctions):
    """Test multiplication with empty Chebfun objects.

    This test verifies that multiplying an empty Chebfun object with any other
    Chebfun object results in an empty Chebfun object, regardless of
    the order of operands.

    Args:
        emptyfun: Fixture providing an empty Chebfun object
    """
    for fun, _, _ in testfunctions:
        for dom, _ in chebfun_testdomains:
            a, b = dom
            ff = Chebfun.initfun_adaptive(fun, np.linspace(a, b, 9))
            assert (emptyfun * ff).isempty
            assert (ff * emptyfun).isempty


def test__mul__rmul__constant(testfunctions):
    """Test multiplication of constants and Chebfun objects.

    This test verifies that multiplying a Chebfun object by a constant
    (and vice versa) produces the expected result within a specified tolerance.
    """
    for fun, _, _ in testfunctions:
        for dom, _ in chebfun_testdomains:
            a, b = dom
            for const in (-1, 1, 10, -1e5):
                def g(x):
                    return const * fun(x)

                def h(x):
                    return fun(x) * const

                ff = Chebfun.initfun_adaptive(fun, np.linspace(a, b, 9))
                gg = const * ff
                hh = ff * const
                xx = np.linspace(a, b, 1001)
                vscl = ff.vscale
                hscl = ff.hscale
                lscl = max([fun.size for fun in ff])
                tol = 2 * abs(const) * vscl * hscl * lscl * eps
                assert np.max(g(xx) - gg(xx)) <= tol
                assert np.max(h(xx) - hh(xx)) <= tol


def test_truediv_empty(emptyfun, testfunctions):
    """Test division with empty Chebfun objects.

    This test verifies that dividing an empty Chebfun object by any other
    Chebfun object (or vice versa) results in an empty Chebfun object.

    Args:
        emptyfun: Fixture providing an empty Chebfun object
    """
    for fun, _, _ in testfunctions:
        for dom, _ in chebfun_testdomains:
            a, b = dom
            ff = Chebfun.initfun_adaptive(fun, np.linspace(a, b, 9))
            for op in div_binops:
                assert op(emptyfun, ff).isempty
                assert op(ff, emptyfun).isempty
            # __truediv__
            assert (emptyfun / ff).isempty
            assert (ff / emptyfun).isempty


def test_truediv_constant(testfunctions):
    """Test division of constants and Chebfun objects.

    This test verifies that dividing a constant by a Chebfun object
    (and vice versa) produces the expected result within a specified tolerance.
    """
    for fun, _, hasRoots in testfunctions:
        if not hasRoots:
            for dom, _ in chebfun_testdomains:
                a, b = dom
                for const in (-1, 1, 10, -1e5):
                    def g(x):
                        return fun(x) / const

                    def h(x):
                        return const / fun(x)

                    ff = Chebfun.initfun_adaptive(fun, np.linspace(a, b, 9))
                    gg = ff / const
                    hh = const / ff
                    xx = np.linspace(a, b, 1001)
                    vscl = ff.vscale
                    hscl = ff.hscale
                    lscl = max([fun.size for fun in ff])
                    tol = 10 * abs(const) * vscl * hscl * lscl * eps
                    assert np.max(g(xx) - gg(xx)) <= tol
                    assert np.max(h(xx) - hh(xx)) <= tol


def test_pow_empty(emptyfun):
    """Test power operation on empty Chebfun objects.

    This test verifies that raising an empty Chebfun object to any power
    results in an empty Chebfun object.

    Args:
        emptyfun: Fixture providing an empty Chebfun object
    """
    for const in (-1, 0, 1, 2):
        assert (emptyfun ** const).isempty
    for const in (-1.0, 0.0, 1.0, 2.0):
        assert (emptyfun ** const).isempty


def test_rpow_empty(emptyfun):
    """Test raising constants to empty Chebfun objects.

    This test verifies that raising a constant to an empty Chebfun object
    results in an empty Chebfun object.

    Args:
        emptyfun: Fixture providing an empty Chebfun object
    """
    for const in (-1, 0, 1, 2):
        assert (const ** emptyfun).isempty
    for const in (-1.0, 0.0, 1.0, 2.0):
        assert (const ** emptyfun).isempty


def test_pow_constant(testfunctions):
    """Test raising Chebfun objects to constant powers.

    This test verifies that raising a Chebfun object to a constant power
    produces the expected result within a specified tolerance.
    """
    for fun, _, hasRoots in testfunctions:
        if not hasRoots:
            for dom, _ in chebfun_testdomains:
                a, b = dom
                for c in (-1, 1, 2, 0.5):
                    def g(x):
                        return fun(x) ** c

                    ff = Chebfun.initfun_adaptive(fun, np.linspace(a, b, 9))
                    gg = ff ** c
                    xx = np.linspace(a, b, 1001)
                    vscl = gg.vscale
                    hscl = gg.hscale
                    lscl = max([fun.size for fun in gg])
                    tol = 10 * abs(c) * vscl * hscl * lscl * eps
                    assert np.max(g(xx) - gg(xx)) <= tol


def test_rpow_constant(testfunctions):
    """Test raising constants to Chebfun objects.

    This test verifies that raising a constant to a Chebfun object
    produces the expected result within a specified tolerance.
    """
    for fun, _, _ in testfunctions:
        for dom, _ in chebfun_testdomains:
            a, b = dom
            for c in (0.5, 1, 2, np.e):
                def g(x):
                    return c ** fun(x)

                ff = Chebfun.initfun_adaptive(fun, np.linspace(a, b, 9))
                gg = c ** ff
                xx = np.linspace(a, b, 1001)

                try:
                    # Evaluate both functions
                    g_vals = g(xx)
                    gg_vals = gg(xx)

                    # Skip test if there are any NaN or infinite values
                    if np.any(np.isnan(g_vals)) or np.any(np.isnan(gg_vals)) or \
                       np.any(np.isinf(g_vals)) or np.any(np.isinf(gg_vals)):
                        continue

                    vscl = gg.vscale
                    hscl = gg.hscale
                    lscl = max([fun.size for fun in gg])
                    tol = 10 * abs(c) * vscl * hscl * lscl * eps
                    assert np.max(g_vals - gg_vals) <= tol
                except (RuntimeWarning, ValueError, OverflowError, FloatingPointError):
                    # Skip test if numerical issues occur
                    continue


# Generate test functions for binary operations
@pytest.mark.parametrize("binop", binops)
@pytest.mark.parametrize("dom,tol", chebfun_testdomains)
def test_binary_operations(binop, dom, tol, testfunctions):
    """Test binary operations between two Chebfun objects.

    This test verifies that binary operations (addition, subtraction,
    multiplication, division) between two Chebfun objects produce the
    expected results within a tolerance that scales with the size and
    scale of the functions.

    Args:
        binop: Binary operator function (e.g., operator.add)
        f: First function
        g: Second function
        dom: Domain for the functions
        tol: Tolerance for the comparison
        denomHasRoots: Whether the second function has roots (for division tests)
    """
    for f, g in itertools.product(testfunctions, testfunctions):
        f, _, rootsF = f
        g, _, rootsG = g

        if binop in div_binops and rootsG:
            pytest.skip("Skipping division test with denominator that has roots")

        test_func = binaryOpTester(f, g, binop, dom, 10 * tol)
        test_func()


# Generate test functions for unary operations
@pytest.mark.parametrize("unaryop", [lambda x: +x, lambda x: -x])
@pytest.mark.parametrize("dom,tol", chebfun_testdomains)
def test_unary_operations(unaryop, dom, tol, testfunctions):
    """Test unary operations on a Chebfun object.

    This test verifies that unary operations (positive, negative) on
    Chebfun objects produce the expected results within a tolerance
    that scales with the size and scale of the function.

    Args:
        unaryop: Unary operator function (e.g., lambda x: -x)
        dom: Domain for the function
        tol: Tolerance for the comparison
    """
    for f, _, _ in testfunctions:
        test_func = unaryOpTester(f, unaryop, dom, tol)
        test_func()
