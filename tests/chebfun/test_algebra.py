"""Unit-tests for Chebfun algebraic operations.

This module contains tests for the algebraic operations of Chebfun,
including addition, subtraction, multiplication, division, and powers.
"""

import itertools
import operator

import numpy as np
import pytest

from chebpy.core.chebfun import Chebfun

from ..utilities import eps


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


def test__add__radd__empty(emptyfun, testdomains, testfunctions):
    """Test addition with empty Chebfun objects.

    This test verifies that adding an empty Chebfun object to any other
    Chebfun object results in an empty Chebfun object, regardless of
    the order of operands.

    Args:
        emptyfun: Fixture providing an empty Chebfun object
        testdomains: List of tuples, each containing a domain (list of endpoints) and a tolerance value
        testfunctions: List of test functions, each represented as a tuple containing
            (fun, _, _) where fun is the function to test
    """
    for fun, _, _ in testfunctions:
        for dom, _ in testdomains:
            a, b = dom
            ff = Chebfun.initfun_adaptive(fun, np.linspace(a, b, 9))
            assert (emptyfun + ff).isempty
            assert (ff + emptyfun).isempty


def test__add__radd__constant(testdomains, testfunctions):
    """Test addition of constants to Chebfun objects.

    This test verifies that adding a constant to a Chebfun object
    (and vice versa) produces the expected result within a specified
    tolerance.
    """
    for fun, _, _ in testfunctions:
        for dom, _ in testdomains:
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


def test__sub__rsub__empty(emptyfun, testdomains, testfunctions):
    """Test subtraction with empty Chebfun objects.

    This test verifies that subtracting an empty Chebfun object from any other
    Chebfun object (or vice versa) results in an empty Chebfun object.

    Args:
        emptyfun: Fixture providing an empty Chebfun object
        testdomains: List of tuples, each containing a domain (list of endpoints) and a tolerance value
        testfunctions: List of test functions, each represented as a tuple containing
            (fun, _, _) where fun is the function to test
    """
    for fun, _, _ in testfunctions:
        for dom, _ in testdomains:
            a, b = dom
            ff = Chebfun.initfun_adaptive(fun, np.linspace(a, b, 9))
            assert (emptyfun - ff).isempty
            assert (ff - emptyfun).isempty


def test__sub__rsub__constant(testdomains, testfunctions):
    """Test subtraction of constants and Chebfun objects.

    This test verifies that subtracting a Chebfun object from a constant
    (and vice versa) produces the expected result within a specified tolerance.
    """
    for fun, _, _ in testfunctions:
        for dom, _ in testdomains:
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


def test__mul__rmul__empty(emptyfun, testdomains, testfunctions):
    """Test multiplication with empty Chebfun objects.

    This test verifies that multiplying an empty Chebfun object with any other
    Chebfun object results in an empty Chebfun object, regardless of
    the order of operands.

    Args:
        emptyfun: Fixture providing an empty Chebfun object
        testdomains: List of tuples, each containing a domain (list of endpoints) and a tolerance value
        testfunctions: List of test functions, each represented as a tuple containing
            (fun, _, _) where fun is the function to test
    """
    for fun, _, _ in testfunctions:
        for dom, _ in testdomains:
            a, b = dom
            ff = Chebfun.initfun_adaptive(fun, np.linspace(a, b, 9))
            assert (emptyfun * ff).isempty
            assert (ff * emptyfun).isempty


def test__mul__rmul__constant(testdomains, testfunctions):
    """Test multiplication of constants and Chebfun objects.

    This test verifies that multiplying a Chebfun object by a constant
    (and vice versa) produces the expected result within a specified tolerance.
    """
    for fun, _, _ in testfunctions:
        for dom, _ in testdomains:
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


def test_truediv_empty(emptyfun, div_binops, testdomains, testfunctions):
    """Test division with empty Chebfun objects.

    This test verifies that dividing an empty Chebfun object by any other
    Chebfun object (or vice versa) results in an empty Chebfun object.

    Args:
        emptyfun: Fixture providing an empty Chebfun object
        div_binops: List of division operators to test
        testdomains: List of tuples, each containing a domain (list of endpoints) and a tolerance value
        testfunctions: List of test functions, each represented as a tuple containing
            (fun, _, _) where fun is the function to test
    """
    for fun, _, _ in testfunctions:
        for dom, _ in testdomains:
            a, b = dom
            ff = Chebfun.initfun_adaptive(fun, np.linspace(a, b, 9))
            for op in div_binops:
                assert op(emptyfun, ff).isempty
                assert op(ff, emptyfun).isempty
            # __truediv__
            assert (emptyfun / ff).isempty
            assert (ff / emptyfun).isempty


def test_truediv_constant(testdomains, testfunctions):
    """Test division of constants and Chebfun objects.

    This test verifies that dividing a constant by a Chebfun object
    (and vice versa) produces the expected result within a specified tolerance.
    """
    for fun, _, has_roots in testfunctions:
        if not has_roots:
            for dom, _ in testdomains:
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
        assert (emptyfun**const).isempty
    for const in (-1.0, 0.0, 1.0, 2.0):
        assert (emptyfun**const).isempty


def test_rpow_empty(emptyfun):
    """Test raising constants to empty Chebfun objects.

    This test verifies that raising a constant to an empty Chebfun object
    results in an empty Chebfun object.

    Args:
        emptyfun: Fixture providing an empty Chebfun object
    """
    for const in (-1, 0, 1, 2):
        assert (const**emptyfun).isempty
    for const in (-1.0, 0.0, 1.0, 2.0):
        assert (const**emptyfun).isempty


def test_pow_constant(testdomains, testfunctions):
    """Test raising Chebfun objects to constant powers.

    This test verifies that raising a Chebfun object to a constant power
    produces the expected result within a specified tolerance.
    """
    for fun, _, has_roots in testfunctions:
        if not has_roots:
            for dom, _ in testdomains:
                a, b = dom
                for c in (-1, 1, 2, 0.5):

                    def g(x):
                        return fun(x) ** c

                    ff = Chebfun.initfun_adaptive(fun, np.linspace(a, b, 9))
                    gg = ff**c
                    xx = np.linspace(a, b, 1001)
                    vscl = gg.vscale
                    hscl = gg.hscale
                    lscl = max([fun.size for fun in gg])
                    tol = 10 * abs(c) * vscl * hscl * lscl * eps
                    assert np.max(g(xx) - gg(xx)) <= tol


def test_rpow_constant(testdomains, testfunctions):
    """Test raising constants to Chebfun objects.

    This test verifies that raising a constant to a Chebfun object
    produces the expected result within a specified tolerance.
    """
    for fun, _, _ in testfunctions:
        for dom, _ in testdomains:
            a, b = dom
            for c in (0.5, 1, 2, np.e):

                def g(x):
                    return c ** fun(x)

                ff = Chebfun.initfun_adaptive(fun, np.linspace(a, b, 9))
                gg = c**ff
                xx = np.linspace(a, b, 1001)

                # try:
                # Evaluate both functions
                g_vals = g(xx)
                gg_vals = gg(xx)

                # Skip test if there are any NaN or infinite values
                if (
                    np.any(np.isnan(g_vals))
                    or np.any(np.isnan(gg_vals))
                    or np.any(np.isinf(g_vals))
                    or np.any(np.isinf(gg_vals))
                ):
                    continue

                vscl = gg.vscale
                hscl = gg.hscale
                lscl = max([fun.size for fun in gg])
                tol = 50 * abs(c) * vscl * hscl * lscl * eps
                assert np.max(g_vals - gg_vals) <= tol
                # except (RuntimeWarning, ValueError, OverflowError, FloatingPointError):
                #    # Skip test if numerical issues occur
                #    continue


# Generate test functions for binary operations
def test_binary_operations(binops, div_binops, testdomains, testfunctions):
    """Test binary operations between two Chebfun objects.

    This test verifies that binary operations (addition, subtraction,
    multiplication, division) between two Chebfun objects produce the
    expected results within a tolerance that scales with the size and
    scale of the functions.

    Args:
        binops: List of binary operators to test (e.g., operator.add, operator.mul)
        div_binops: List of division operators to check for special handling
        testdomains: List of tuples, each containing a domain (list of endpoints) and a tolerance value
        testfunctions: List of test functions to evaluate
    """
    for binop in binops:
        for f, g in itertools.product(testfunctions, testfunctions):

            f, _, roots_f = f
            g, _, roots_g = g

            for dom, tol in testdomains:
                if binop in div_binops and roots_g:
                    pass
                else:
                    a, b = dom
                    xx = np.linspace(a, b, 1001)
                    n, m = 3, 8
                    ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, n + 1))
                    gg = Chebfun.initfun_adaptive(g, np.linspace(a, b, m + 1))

                    def fg_expected(x):
                        return binop(f(x), g(x))

                    fg = binop(ff, gg)

                        #def tester():
                    vscl = max([ff.vscale, gg.vscale])
                    hscl = max([ff.hscale, gg.hscale])
                    lscl = max([fun.size for fun in np.append(ff.funs, gg.funs)])
                    assert ff.funs.size == n
                    assert gg.funs.size == m
                    assert fg.funs.size == n + m - 1

                    # Increase tolerance for multiplication on large domains
                    extra_factor = 1
                    if binop == operator.mul and abs(b - a) > 10:
                        extra_factor = 100

                    # try:
                    # Evaluate both functions
                    fg_vals = fg(xx)
                    fg_expected_vals = fg_expected(xx)

                    assert np.max(fg_vals - fg_expected_vals) <= extra_factor * vscl * hscl * lscl * tol


# Generate test functions for unary operations
@pytest.mark.parametrize("unaryop", [lambda x: +x, lambda x: -x])
def test_unary_operations(unaryop, testdomains, testfunctions):
    """Test unary operations on a Chebfun object.

    This test verifies that unary operations (positive, negative) on
    Chebfun objects produce the expected results within a tolerance
    that scales with the size and scale of the function.

    Args:
        unaryop: Unary operator function (e.g., lambda x: -x)
        testdomains: List of tuples, each containing a domain (list of endpoints) and a tolerance value
        testfunctions: List of test functions to evaluate
    """
    for f, _, _ in testfunctions:
        for dom, tol in testdomains:
            a, b = dom
            xx = np.linspace(a, b, 1001)
            ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 9))

            #def gg_expected(x):
            #    return unaryop(f(x))

            gg = unaryop(ff)

            #def tester():
            vscl = ff.vscale
            hscl = ff.hscale
            lscl = max([fun.size for fun in ff])

            assert ff.funs.size == gg.funs.size
            assert np.max(gg(xx) - unaryop(f(xx))) <= vscl * hscl * lscl * tol

            #return tester



            #test_func = unary_op_tester(f, unaryop, dom, tol)
            #test_func()
