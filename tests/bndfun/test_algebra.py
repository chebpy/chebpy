"""Unit-tests for Bndfun algebraic operations."""

import operator

import numpy as np
import pytest

from chebpy.core.bndfun import Bndfun
from chebpy.core.utilities import Interval

from ..utilities import eps

# Binary operators to test
binops = (operator.add, operator.mul, operator.sub, operator.truediv)


# @pytest.fixture
# def algebra_fixtures(emptyfun):
#     """Create fixtures for testing Bndfun algebraic operations."""
#     yy = np.linspace(-1, 1, 1000)
#
#     return {
#         "yy": yy,
#         "emptyfun": emptyfun
#     }


def test__pos__empty(emptyfun):
    """Test unary positive operator on an empty Bndfun."""
    assert (+emptyfun).isempty


def test__neg__empty(emptyfun):
    """Test unary negative operator on an empty Bndfun."""
    assert (-emptyfun).isempty


def test__add__radd__empty(emptyfun, testfunctions):
    """Test addition with an empty Bndfun."""
    subinterval = Interval(-2, 3)
    for fun, _, _ in testfunctions:
        chebtech = Bndfun.initfun_adaptive(fun, subinterval)
        assert (emptyfun + chebtech).isempty
        assert (chebtech + emptyfun).isempty


def test__add__radd__constant(testfunctions):
    """Test addition of a constant to a Bndfun."""
    yy = np.linspace(-1, 1, 1000)
    subinterval = Interval(-0.5, 0.9)
    xx = subinterval(yy)
    for fun, _, _ in testfunctions:
        for const in (-1, 1, 10, -1e5):

            def f(x):
                return const + fun(x)

            bndfun = Bndfun.initfun_adaptive(fun, subinterval)
            f1 = const + bndfun
            f2 = bndfun + const
            tol = 4e1 * eps * abs(const)
            assert np.max(f(xx) - f1(xx)) <= tol
            assert np.max(f(xx) - f2(xx)) <= tol


def test__sub__rsub__empty(emptyfun, testfunctions):
    """Test subtraction with an empty Bndfun."""
    subinterval = Interval(-2, 3)
    for fun, _, _ in testfunctions:
        chebtech = Bndfun.initfun_adaptive(fun, subinterval)
        assert (emptyfun - chebtech).isempty
        assert (chebtech - emptyfun).isempty


def test__sub__rsub__constant(testfunctions):
    """Test subtraction of a constant from a Bndfun and vice versa."""
    yy = np.linspace(-1, 1, 1000)
    subinterval = Interval(-0.5, 0.9)
    xx = subinterval(yy)
    for fun, _, _ in testfunctions:
        for const in (-1, 1, 10, -1e5):

            def f(x):
                return const - fun(x)

            def g(x):
                return fun(x) - const

            bndfun = Bndfun.initfun_adaptive(fun, subinterval)
            ff = const - bndfun
            gg = bndfun - const
            tol = 5e1 * eps * abs(const)
            assert np.max(f(xx) - ff(xx)) <= tol
            assert np.max(g(xx) - gg(xx)) <= tol


def test__mul__rmul__empty(emptyfun, testfunctions):
    """Test multiplication with an empty Bndfun."""
    subinterval = Interval(-2, 3)
    for fun, _, _ in testfunctions:
        chebtech = Bndfun.initfun_adaptive(fun, subinterval)
        assert (emptyfun * chebtech).isempty
        assert (chebtech * emptyfun).isempty


def test__mul__rmul__constant(testfunctions):
    """Test multiplication of a Bndfun by a constant."""
    yy = np.linspace(-1, 1, 1000)
    subinterval = Interval(-0.5, 0.9)
    xx = subinterval(yy)
    for fun, _, _ in testfunctions:
        for const in (-1, 1, 10, -1e5):

            def f(x):
                return const * fun(x)

            def g(x):
                return fun(x) * const

            bndfun = Bndfun.initfun_adaptive(fun, subinterval)
            ff = const * bndfun
            gg = bndfun * const
            tol = 4e1 * eps * abs(const)
            assert np.max(f(xx) - ff(xx)) <= tol
            assert np.max(g(xx) - gg(xx)) <= tol


def test_truediv_empty(emptyfun, testfunctions):
    """Test division with an empty Bndfun."""
    subinterval = Interval(-2, 3)
    for fun, _, _ in testfunctions:
        bndfun = Bndfun.initfun_adaptive(fun, subinterval)
        assert operator.truediv(emptyfun, bndfun).isempty
        assert operator.truediv(bndfun, emptyfun).isempty
        # __truediv__
        assert (emptyfun / bndfun).isempty
        assert (bndfun / emptyfun).isempty


def test_truediv_constant(testfunctions):
    """Test division of a Bndfun by a constant and vice versa."""
    yy = np.linspace(-1, 1, 1000)
    subinterval = Interval(-0.5, 0.9)
    xx = subinterval(yy)
    for fun, _, has_roots in testfunctions:
        for const in (-1, 1, 10, -1e5):

            def f(x):
                return const / fun(x)

            def g(x):
                return fun(x) / const

            hscl = abs(subinterval).max()
            tol = hscl * eps * abs(const)
            bndfun = Bndfun.initfun_adaptive(fun, subinterval)
            gg = bndfun / const
            assert np.max(g(xx) - gg(xx)) <= 3 * gg.size * tol
            # don't do the following test for functions with roots
            if not has_roots:
                ff = const / bndfun
                assert np.max(f(xx) - ff(xx)) <= 2 * ff.size * tol


def test_pow_empty(emptyfun):
    """Test power operation with an empty Bndfun."""
    for c in range(10):
        assert (emptyfun**c).isempty


def test_rpow_empty(emptyfun):
    """Test raising a constant to an empty Bndfun power."""
    for c in range(10):
        assert (c**emptyfun).isempty


def test_pow_const():
    """Test raising a Bndfun to a constant power."""
    yy = np.linspace(-1, 1, 1000)
    subinterval = Interval(-0.5, 0.9)
    xx = subinterval(yy)
    for func in (np.sin, np.exp, np.cos):
        for c in (1, 2):

            def f(x):
                return func(x) ** c

            ff = Bndfun.initfun_adaptive(func, subinterval) ** c
            tol = 2e1 * eps * abs(c)
            assert np.max(f(xx) - ff(xx)) <= tol


def test_rpow_const():
    """Test raising a constant to a Bndfun power."""
    yy = np.linspace(-1, 1, 1000)
    subinterval = Interval(-0.5, 0.9)
    xx = subinterval(yy)
    for func in (np.sin, np.exp, np.cos):
        for c in (1, 2):

            def f(x):
                return c ** func(x)

            ff = c ** Bndfun.initfun_adaptive(func, subinterval)
            tol = 1e1 * eps * abs(c)
            assert np.max(f(xx) - ff(xx)) <= tol


# Helper function for binary operator tests
def binary_op_test(f, g, subinterval, binop, yy):
    """Test a binary operation between two Bndfun objects."""
    ff = Bndfun.initfun_adaptive(f, subinterval)
    gg = Bndfun.initfun_adaptive(g, subinterval)

    def fg_expected(x):
        return binop(f(x), g(x))

    fg = binop(ff, gg)

    vscl = max([ff.vscale, gg.vscale])
    lscl = max([ff.size, gg.size])
    xx = subinterval(yy)
    assert np.max(fg(xx) - fg_expected(xx)) <= 6 * vscl * lscl * eps


# Test binary operations between Bndfun objects
subintervals = [Interval(-0.5, 0.9), Interval(-1.2, 1.3), Interval(-2.2, -1.9), Interval(0.4, 1.3)]


# # Generate test cases for binary operations
# binary_test_cases = []
# for binop in binops:
#     for (f, _, _), (g, _, denomRoots) in itertools.combinations(testfunctions, 2):
#         for subinterval in subintervals:
#             if binop is operator.truediv and denomRoots:
#                 # skip truediv test if denominator has roots on the real line
#                 continue
#             binary_test_cases.append((f, g, subinterval, binop))
#
@pytest.mark.parametrize("subinterval", subintervals)
@pytest.mark.parametrize("f, g", [(np.exp, np.sin), (np.exp, lambda x: 2 - x), (lambda x: 2 - x, np.exp)])
def test_binary_operations(f, g, subinterval, binop=operator.pow):
    """Test binary operations between Bndfun objects."""
    yy = np.linspace(-1, 1, 1000)
    binary_op_test(f, g, subinterval, binop, yy)


@pytest.mark.parametrize("subinterval", subintervals)
@pytest.mark.parametrize("f, g", [(np.exp, np.sin), (np.exp, lambda x: 2 - x), (lambda x: 2 - x, np.exp)])
def test_pow_operations(f, g, subinterval, binop=operator.pow):
    """Test power operations between Bndfun objects."""
    yy = np.linspace(-1, 1, 1000)
    binary_op_test(f, g, subinterval, binop, yy)


# Helper function for unary operator tests
def unary_op_test(unaryop, f, subinterval, yy):
    """Test a unary operation on a Bndfun object."""
    ff = Bndfun.initfun_adaptive(f, subinterval)

    def gg(x):
        return unaryop(f(x))

    gg_result = unaryop(ff)

    xx = subinterval(yy)
    assert np.max(gg(xx) - gg_result(xx)) <= 4e1 * eps


@pytest.mark.parametrize("unaryop", [operator.pos, operator.neg])
def test_unary_operations(unaryop, testfunctions):
    """Test unary operations on Bndfun objects."""
    yy = np.linspace(-1, 1, 1000)
    subinterval = Interval(-0.5, 0.9)
    for f, _, _ in testfunctions:
        unary_op_test(unaryop, f, subinterval, yy)
