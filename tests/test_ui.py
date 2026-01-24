"""Unit-tests for chebpy user interface functions.

This module contains tests for the user-facing functions in the chebpy package,
including chebfun() and pwc() constructors. It verifies that these functions
handle various input types correctly and produce expected results.
"""

import pickle

import numpy as np
import pytest

from chebpy import chebfun, pwc
from chebpy.settings import DefaultPreferences


def test_chebfun_null_args() -> None:
    """Test creating an empty chebfun with no arguments."""
    assert chebfun().isempty


def test_chebfun_callable() -> None:
    """Test creating chebfun objects with callable functions."""
    n = 100
    d = np.array([-2, 0, 1])
    f1 = chebfun(np.sin)
    f2 = chebfun(np.sin, d)
    f3 = chebfun(np.sin, n=n)
    f4 = chebfun(np.sin, d, n)

    # check domains
    assert f1.domain == DefaultPreferences.domain
    assert f2.domain == d
    assert f3.domain == DefaultPreferences.domain
    assert f4.domain == d

    # check lengths of f3 and f4
    assert f3.funs[0].size == n
    assert np.all([fun.size == n for fun in f4])


def test_chebfun_alphanum_char() -> None:
    """Test creating chebfun objects with alphanumeric characters."""
    n = 100
    d = np.array([-2, 0, 1])
    f1 = chebfun("x")
    f2 = chebfun("y", d)
    f3 = chebfun("z", n=n)
    f4 = chebfun("a", d, n)

    # check domains
    assert f1.domain == DefaultPreferences.domain
    assert f2.domain == d
    assert f3.domain == DefaultPreferences.domain
    assert f4.domain == d

    # check lengths of f3 and f4
    assert np.sum([fun.size for fun in f3]) == n
    assert np.all([fun.size == n for fun in f4])


def test_chebfun_float_arg() -> None:
    """Test creating chebfun objects with float arguments."""
    d = np.array([-2, 0, 1])
    f1 = chebfun(3.14)
    f2 = chebfun("3.14")
    f3 = chebfun(2.72, d)
    f4 = chebfun("2.72", d)

    # check domains
    assert f1.domain == DefaultPreferences.domain
    assert f2.domain == DefaultPreferences.domain
    assert f3.domain == d
    assert f4.domain == d

    # check all are constant
    assert f1.isconst
    assert f2.isconst
    assert f3.isconst
    assert f4.isconst


def test_chebfun_raises() -> None:
    """Test that invalid inputs raise appropriate exceptions."""
    with pytest.raises(ValueError, match="asdfasdf"):
        chebfun("asdfasdf")


def test_pwc() -> None:
    """Test creating piecewise constant functions."""
    dom = [-1, 0, 1]
    vals = [0, 1]
    f = pwc(dom, vals)
    assert f.funs.size == 2
    for fun, val in zip(f, vals, strict=False):
        assert fun.isconst
        assert fun.coeffs[0] == val


def test_evaluate() -> None:
    """Test that pickled/unpickled chebfun objects evaluate correctly."""
    f0 = chebfun(np.sin, [-2, 0, 1])
    f1 = pickle.loads(pickle.dumps(f0))

    x = -1
    assert f0(x) == f1(x)
