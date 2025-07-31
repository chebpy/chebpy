"""Unit-tests for Chebfun algebraic operations.

This module contains tests for the algebraic operations of Chebfun,
including addition, subtraction, multiplication, division, and powers.
"""

import itertools

import pytest

from chebpy.core.chebfun import Chebfun

from ..generic.algebra import *


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

                    # Evaluate both functions
                    fg_vals = fg(xx)
                    fg_expected_vals = fg_expected(xx)

                    assert np.max(np.abs(fg_vals - fg_expected_vals)) <= extra_factor * vscl * hscl * lscl * tol


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

            gg = unaryop(ff)

            vscl = ff.vscale
            hscl = ff.hscale
            lscl = max([fun.size for fun in ff])

            assert ff.funs.size == gg.funs.size
            assert np.max(np.abs(gg(xx) - unaryop(f(xx)))) <= vscl * hscl * lscl * tol
