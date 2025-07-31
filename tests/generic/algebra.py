"""Generic test functions for algebraic operations.

This module contains test functions for algebraic operations that can be used
with any type of function object (Bndfun, Chebfun, or Chebtech2). These tests
focus on operations with empty function objects.
"""
import operator

import numpy as np

from tests.utilities import eps


def test__pos__empty(emptyfun):
    """Test unary positive operator on an empty fun."""
    assert (+emptyfun).isempty


def test__neg__empty(emptyfun):
    """Test unary negative operator on an empty fun."""
    assert (-emptyfun).isempty


def test_pow_empty(emptyfun):
    """Test power operation with an empty fun."""
    for c in range(10):
        assert (emptyfun**c).isempty


def test_rpow_empty(emptyfun):
    """Test raising a constant to an empty fun power."""
    for c in range(10):
        assert (c**emptyfun).isempty


def test__add__radd__empty(emptyfun, ttt):
    """Test addition with an empty Bndfun."""
    for test_function in ttt:
        assert (emptyfun + test_function.cheb).isempty
        assert (test_function.cheb + emptyfun).isempty

def test__add__radd__constant(ttt, random_points):
    """Test addition of a constant to a Bndfun."""
    xx = random_points
    for fun in ttt:
        for const in (-1, 1, 10, -1e5):

            def f(x):
                return const + fun.raw(x)

            f1 = const + fun.cheb
            f2 = fun.cheb + const
            tol = 4e1 * eps * abs(const)

            assert np.max(np.abs(f(xx) - f1(xx))) <= tol
            assert np.max(np.abs(f(xx) - f2(xx))) <= tol


def test__sub__rsub__empty(emptyfun, ttt):
    """Test subtraction with an empty Bndfun."""
    for fun in ttt:
        assert (emptyfun - fun.cheb).isempty
        assert (fun.cheb - emptyfun).isempty


def test__sub__rsub__constant(ttt, random_points):
    """Test subtraction of a constant from a Bndfun and vice versa."""
    #yy = np.linspace(-1, 1, 1000)
    #subinterval = Interval(-0.5, 0.9)
    xx = random_points
    for fun in ttt:
        for const in (-1, 1, 10, -1e5):

            def f(x):
                return const - fun.raw(x)

            def g(x):
                return fun.raw(x) - const


            ff = const - fun.cheb
            gg = fun.cheb - const
            tol = 5e1 * eps * abs(const)
            assert np.max(np.abs(f(xx) - ff(xx))) <= tol
            assert np.max(np.abs(g(xx) - gg(xx))) <= tol


def test__mul__rmul__empty(emptyfun, ttt):
    """Test multiplication with an empty Bndfun."""
    #subinterval = Interval(-2, 3)
    for fun in ttt:
        assert (emptyfun * fun.cheb).isempty
        assert (fun.cheb * emptyfun).isempty


def test__mul__rmul__constant(ttt, random_points):
    """Test multiplication of a Bndfun by a constant."""
    #yy = np.linspace(-1, 1, 1000)
    #subinterval = Interval(-0.5, 0.9)
    xx = random_points
    for fun in ttt:
        for const in (-1, 1, 10, -1e5):

            def f(x):
                return const * fun.cheb(x)

            def g(x):
                return fun.cheb(x) * const

            ff = const * fun.cheb
            gg = fun.cheb * const
            tol = 4e1 * eps * abs(const)
            assert np.max(np.abs(f(xx) - ff(xx))) <= tol
            assert np.max(np.abs(g(xx) - gg(xx))) <= tol


def test_truediv_empty(emptyfun, ttt):
    """Test division with an empty Bndfun."""
    for fun in ttt:
        assert operator.truediv(emptyfun, fun.cheb).isempty
        assert operator.truediv(fun.cheb, emptyfun).isempty
        # __truediv__
        assert (emptyfun / fun.cheb).isempty
        assert (fun.cheb / emptyfun).isempty


def test_truediv_constant(ttt, random_points):
    """Test division of a Bndfun by a constant and vice versa."""
    xx = random_points
    for fun in ttt:
        for const in (-1, 1, 10, -1e5):
            def g(x):
                return fun.raw(x) / const

            tol = eps * abs(const)
            gg = fun.cheb / const

            assert np.max(np.abs(g(xx) - gg(xx))) <= 1e2 * tol, "Failed for " + fun.cheb.name + " and c = " + str(const) + "."


            # don't do the following test for functions with roots
            if not fun.has_roots:
                def f(x):
                    return const / fun.raw(x)

                ff = const / fun.cheb
                assert np.max(np.abs(f(xx) - ff(xx))) <= 20 * tol


def test_pow_const(ttt, random_points):
    """Test raising a fun to a constant power."""
    xx = random_points
    for fun in ttt:
        for c in (1, 2):

            def f(x):
                return fun.raw(x) ** c

            tol = 2e1 * eps * abs(c)
            func = fun.cheb ** c

            assert np.max(np.abs(f(xx) - func(xx))) <= 1e2 * tol, (
                f"Failed for {fun.cheb.name} and c = {c}"
            )

def test_rpow_const(ttt, random_points):
    """Test raising a constant to a fun power."""
    xx = random_points
    for fun in ttt:
        for c in (1, 2):

            def f(x):
                return c ** fun.raw(x)

            ff = c ** fun.cheb
            tol = 1e2 * eps * abs(c)
            assert np.max(np.abs(f(xx) - ff(xx))) <= tol, "Failed for " + fun.cheb.name + " and c = " + str(c) + "."


def test__add__negself(random_points, ttt):
    """Test subtraction of a fun object from itself.

    This test verifies that subtracting a Chebtech2 object from itself
    results in a constant zero function.

    Args:
        random_points: Fixture providing random points for evaluation
        testfunctions: List of test functions, each represented as a tuple containing
            (fun, funlen, _) where fun is the function to test, funlen is the expected
            function length, and _ is an unused parameter.
    """
    xx = random_points
    for fun in ttt:
        #chebtech = Chebtech2.initfun_fixedlen(fun, funlen)
        chebzero = fun.cheb - fun.cheb
        assert chebzero.isconst
        assert np.max(chebzero(xx)) == 0
