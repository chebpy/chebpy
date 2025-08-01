"""Generic test functions for algebraic operations.

This module contains test functions for algebraic operations that can be used
with any type of function object (Bndfun, Chebfun, or Chebtech). These tests
focus on operations with empty function objects.
"""
import itertools
import operator

import numpy as np
import pytest

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
    for fun in ttt:
        assert (emptyfun * fun.cheb).isempty
        assert (fun.cheb * emptyfun).isempty


def test__mul__rmul__constant(ttt, random_points):
    """Test multiplication of a Bndfun by a constant."""
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

            error_msg = f"Failed for {fun.cheb.name} and c = {const}."
            assert np.max(np.abs(g(xx) - gg(xx))) <= 1e2 * tol, error_msg


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
            error_msg = f"Failed for {fun.cheb.name} and c = {c}."
            assert np.max(np.abs(f(xx) - ff(xx))) <= tol, error_msg

def test__add__negself(random_points, ttt):
    """Test subtraction of a fun object from itself.

    This test verifies that subtracting a Chebtech object from itself
    results in a constant zero function.

    Args:
        random_points: Fixture providing random points for evaluation
        ttt: List of test function objects for testing
    """
    xx = random_points
    for fun in ttt:
        chebzero = fun.cheb - fun.cheb
        assert chebzero.isconst
        assert np.max(chebzero(xx)) == 0

@pytest.mark.parametrize("unaryop", [operator.pos, operator.neg])
def test_unary_operations(unaryop, ttt, random_points):
    """Test unary operations on Bndfun objects."""
    for f in ttt:
        ff = unaryop(f.cheb)(random_points)
        gg = unaryop(f.raw(random_points))

        assert np.max(np.abs(ff - gg)) <= 4e1 * eps

@pytest.mark.parametrize("binop", [operator.add, operator.sub, operator.mul, operator.truediv])
#@pytest.mark.parametrize("f, g", [(np.exp, np.sin), (np.exp, lambda x: 2 - x), (lambda x: 2 - x, np.exp)])
def test_binary_operations(ttt, random_points, binop):
    """Test binary operations between Bndfun objects."""
    xx = random_points
    for f, g in itertools.product(ttt, ttt):

        if binop == operator.truediv and g.has_roots:
            continue

        def fg_expected(x):
            return binop(f.raw(x), g.raw(x))

        fg = binop(f.cheb, g.cheb)

        a = fg(xx)
        b = fg_expected(xx)
        assert not np.all(np.isnan(a)), f"{binop.__name__} failed for {f.cheb.name} and {g.cheb.name}"
        assert not np.all(np.isnan(b)), f"{binop.__name__} failed for {f.cheb.name} and {g.cheb.name}"
        diff = np.max(np.abs(a-b))
        assert diff <= 1e-10, f"{binop.__name__} failed for {f.cheb.name} and {g.cheb.name}: max diff = {diff:.2e}"
