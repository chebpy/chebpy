"""Unit-tests for Chebfun construction methods.

This module contains tests for the various ways to construct Chebfun objects,
including from functions, constants, and identity functions.
"""

import numpy as np
import pytest

from chebpy.bndfun import Bndfun
from chebpy.chebfun import Chebfun
from chebpy.exceptions import BadFunLengthArgument, IntervalGap, IntervalOverlap, InvalidDomain
from chebpy.utilities import Interval

from ..utilities import eps, exp


@pytest.fixture
def construction_fixtures():
    """Create Bndfun objects for testing Chebfun construction.

    This fixture creates several Bndfun objects on different intervals
    and arrays of these objects for testing Chebfun construction.
    The arrays are designed to test various cases:
        - funs_a and funs_b: overlapping intervals (should raise IntervalOverlap)
        - funs_c and funs_d: non-contiguous intervals (should raise IntervalGap)

    Returns:
        dict: Dictionary containing:
            fun0-fun4: Individual Bndfun objects on different intervals
            funs_a: Array of functions with overlapping intervals
            funs_b: Another array with overlapping intervals
            funs_c: Array of functions with a gap between intervals
            funs_d: Another array with a gap between intervals
    """

    def f(x):
        return exp(x)

    fun0 = Bndfun.initfun_adaptive(f, Interval(-1, 0))
    fun1 = Bndfun.initfun_adaptive(f, Interval(0, 1))
    fun2 = Bndfun.initfun_adaptive(f, Interval(-0.5, 0.5))
    fun3 = Bndfun.initfun_adaptive(f, Interval(2, 2.5))
    fun4 = Bndfun.initfun_adaptive(f, Interval(-3, -2))
    funs_a = np.array([fun1, fun0, fun2])
    funs_b = np.array([fun1, fun2])
    funs_c = np.array([fun0, fun3])
    funs_d = np.array([fun1, fun4])

    return {
        "fun0": fun0,
        "fun1": fun1,
        "fun2": fun2,
        "fun3": fun3,
        "fun4": fun4,
        "funs_a": funs_a,
        "funs_b": funs_b,
        "funs_c": funs_c,
        "funs_d": funs_d,
    }


def test__init__pass(construction_fixtures):
    """Test successful initialization of Chebfun objects.

    This test verifies that Chebfun objects can be created from arrays of
    Bndfun objects with valid (non-overlapping, contiguous) intervals.

    Args:
        construction_fixtures: Fixture providing test Bndfun objects.
    """
    fun0 = construction_fixtures["fun0"]
    fun1 = construction_fixtures["fun1"]
    fun2 = construction_fixtures["fun2"]

    Chebfun([fun0])
    Chebfun([fun1])
    Chebfun([fun2])
    Chebfun([fun0, fun1])


def test__init__fail(construction_fixtures):
    """Test failed initialization of Chebfun objects.

    This test verifies that attempting to create Chebfun objects from arrays
    of Bndfun objects with invalid intervals (overlapping or non-contiguous)
    raises the appropriate exceptions.

    Args:
        construction_fixtures: Fixture providing test Bndfun objects.
    """
    funs_a = construction_fixtures["funs_a"]
    funs_b = construction_fixtures["funs_b"]
    funs_c = construction_fixtures["funs_c"]
    funs_d = construction_fixtures["funs_d"]

    with pytest.raises(IntervalOverlap):
        Chebfun(funs_a)
    with pytest.raises(IntervalOverlap):
        Chebfun(funs_b)
    with pytest.raises(IntervalGap):
        Chebfun(funs_c)
    with pytest.raises(IntervalGap):
        Chebfun(funs_d)


def test_initempty():
    """Test initialization of an empty Chebfun.

    This test verifies that an empty Chebfun can be created and has
    the expected properties.
    """
    emptyfun = Chebfun.initempty()
    assert emptyfun.funs.size == 0


def test_initconst(construction_fixtures):
    """Test initialization of constant Chebfun objects.

    This test verifies that Chebfun objects initialized with constants
    have the isconst property set to True, while those initialized with
    non-constant functions have isconst set to False.

    Args:
        construction_fixtures: Fixture providing test Bndfun objects.
    """
    fun0 = construction_fixtures["fun0"]
    fun1 = construction_fixtures["fun1"]
    fun2 = construction_fixtures["fun2"]

    assert Chebfun.initconst(1, [-1, 1]).isconst
    assert Chebfun.initconst(-10, np.linspace(-1, 1, 11)).isconst
    assert Chebfun.initconst(3, [-2, 0, 1]).isconst
    assert Chebfun.initconst(3.14, np.linspace(-100, -90, 11)).isconst
    assert not Chebfun([fun0]).isconst
    assert not Chebfun([fun1]).isconst
    assert not Chebfun([fun2]).isconst
    assert not Chebfun([fun0, fun1]).isconst


def test_initidentity():
    """Test initialization of identity Chebfun objects.

    This test verifies that Chebfun objects initialized as identity functions
    correctly evaluate to their input values within machine precision.
    It tests various domains, including the default domain.
    """
    _doms = (
        np.linspace(-1, 1, 2),
        np.linspace(-1, 1, 11),
        np.linspace(-10, 17, 351),
        np.linspace(-9.3, -3.2, 22),
        np.linspace(2.5, 144.3, 2112),
    )
    for _dom in _doms:
        ff = Chebfun.initidentity(_dom)
        a, b = ff.support
        xx = np.linspace(a, b, 1001)
        tol = eps * ff.hscale
        assert np.max(np.abs(ff(xx) - xx)) <= tol
    # test the default case
    ff = Chebfun.initidentity()
    a, b = ff.support
    xx = np.linspace(a, b, 1001)
    tol = eps * ff.hscale
    assert np.max(np.abs(ff(xx) - xx)) <= tol


def test_initfun_adaptive_continuous_domain():
    """Test adaptive initialization of Chebfun objects on continuous domains.

    This test verifies that Chebfun objects can be adaptively initialized
    from functions on continuous domains (domains with a single interval).
    It checks that the resulting Chebfun has the expected number of pieces
    and evaluates to the correct values.
    """

    def f(x):
        return exp(x)

    ff = Chebfun.initfun_adaptive(f, [-2, -1])
    assert ff.funs.size == 1
    xx = np.linspace(-2, -1, 1001)
    assert np.max(np.abs(f(xx) - ff(xx))) <= 2 * eps


def test_initfun_adaptive_piecewise_domain():
    """Test adaptive initialization of Chebfun objects on piecewise domains.

    This test verifies that Chebfun objects can be adaptively initialized
    from functions on piecewise domains (domains with multiple intervals).
    It checks that the resulting Chebfun has the expected number of pieces
    and evaluates to the correct values.
    """

    def f(x):
        return exp(x)

    ff = Chebfun.initfun_adaptive(f, [-2, -1, 0, 1, 2])
    assert ff.funs.size == 4
    xx = np.linspace(-2, 2, 1001)
    assert np.max(np.abs(f(xx) - ff(xx))) <= 10 * eps


def test_initfun_adaptive_raises():
    """Test that adaptive initialization raises appropriate exceptions.

    This test verifies that attempting to adaptively initialize a Chebfun
    with invalid arguments raises the appropriate exceptions.
    """

    def f(x):
        return exp(x)

    with pytest.raises(InvalidDomain):
        Chebfun.initfun_adaptive(f, [-2])
    with pytest.raises(InvalidDomain):
        Chebfun.initfun_adaptive(f, domain=[-2])
    with pytest.raises(InvalidDomain):
        Chebfun.initfun_adaptive(f, domain=0)


def test_initfun_adaptive_empty_domain():
    """Test adaptive initialization with an empty domain.

    This test verifies that adaptively initializing a Chebfun
    with an empty domain returns an empty Chebfun.
    """

    def f(x):
        return exp(x)

    cheb = Chebfun.initfun_adaptive(f, domain=[])
    assert cheb.isempty


def test_initfun_fixedlen_continuous_domain():
    """Test fixed-length initialization of Chebfun objects on continuous domains.

    This test verifies that Chebfun objects can be initialized with a fixed
    length from functions on continuous domains (domains with a single interval).
    It checks that the resulting Chebfun has the expected number of pieces
    and evaluates to the correct values.
    """

    def f(x):
        return exp(x)

    ff = Chebfun.initfun_fixedlen(f, 20, [-2, -1])
    assert ff.funs.size == 1
    xx = np.linspace(-2, -1, 1001)
    assert np.max(np.abs(f(xx) - ff(xx))) <= 1e1 * eps


def test_initfun_fixedlen_piecewise_domain_0():
    """Test fixed-length initialization on piecewise domains (equal lengths).

    This test verifies that Chebfun objects can be initialized with a fixed
    length from functions on piecewise domains (domains with multiple intervals).
    It checks that the resulting Chebfun has the expected number of pieces
    and each piece has the specified length.
    """

    def f(x):
        return exp(x)

    ff = Chebfun.initfun_fixedlen(f, 30, [-2.0, 0.0, 1.0])
    assert ff.funs.size == 2
    xx = np.linspace(-2, 1, 1001)
    assert np.max(np.abs(f(xx) - ff(xx))) <= 1e1 * eps


def test_initfun_fixedlen_piecewise_domain_1():
    """Test fixed-length initialization on piecewise domains (different lengths).

    This test verifies that Chebfun objects can be initialized with different
    fixed lengths for each piece from functions on piecewise domains.
    It checks that the resulting Chebfun has the expected number of pieces
    and each piece has the specified length.
    """

    def f(x):
        return exp(x)

    ff = Chebfun.initfun_fixedlen(f, [30, 20], [-2, 0, 1])
    assert ff.funs.size == 2
    xx = np.linspace(-2, 1, 1001)
    assert np.max(np.abs(f(xx) - ff(xx))) <= 1e1 * eps


def test_initfun_fixedlen_raises():
    """Test that fixed-length initialization raises appropriate exceptions.

    This test verifies that attempting to initialize a Chebfun with a fixed
    length using invalid arguments raises the appropriate exceptions.
    """

    def f(x):
        return exp(x)

    initfun = Chebfun.initfun_fixedlen
    with pytest.raises(InvalidDomain):
        initfun(f, 10, [-2])
    with pytest.raises(InvalidDomain):
        initfun(f, n=10, domain=[-2])
    with pytest.raises(InvalidDomain):
        initfun(f, n=10, domain=0)
    with pytest.raises(BadFunLengthArgument):
        initfun(f, [30, 40], [-1, 1])
    with pytest.raises(TypeError):
        initfun(f, [], [-2, -1, 0])


def test_initfun_fixedlen_empty_domain():
    """Test fixed-length initialization with an empty domain.

    This test verifies that initializing a Chebfun with a fixed
    length using an empty domain returns an empty Chebfun.
    """

    def f(x):
        return exp(x)

    cheb = Chebfun.initfun_fixedlen(f, n=10, domain=[])
    assert cheb.isempty


def test_initfun_fixedlen_succeeds():
    """Test that fixed-length initialization succeeds with valid arguments.

    This test verifies that Chebfun objects can be initialized with a fixed
    length using various valid arguments, including None values which should
    trigger adaptive construction.
    """

    def f(x):
        return exp(x)

    # check providing a vector with None elements calls the
    # Tech adaptive constructor
    dom = [-2, -1, 0]
    g0 = Chebfun.initfun_adaptive(f, dom)
    g1 = Chebfun.initfun_fixedlen(f, [None, None], dom)
    g2 = Chebfun.initfun_fixedlen(f, [None, 40], dom)
    g3 = Chebfun.initfun_fixedlen(f, None, dom)

    # Check that g1 and g3 have the same coefficients as g0
    for fun_a, fun_b in zip(g1, g0):
        assert np.sum(fun_a.coeffs - fun_b.coeffs) == 0
    for fun_a, fun_b in zip(g3, g0):
        assert np.sum(fun_a.coeffs - fun_b.coeffs) == 0

    # Check that the first piece of g2 has the same coefficients as the first piece of g0
    assert np.sum(g2.funs[0].coeffs - g0.funs[0].coeffs) == 0


class TestChebfunConstructionEdgeCases:
    """Additional edge case tests for Chebfun construction."""

    def test_initfun_with_n_parameter(self):
        """Test initfun with explicit n parameter (delegates to initfun_fixedlen)."""
        f = Chebfun.initfun(lambda x: np.sin(x), domain=[-1, 1], n=20)
        assert f.funs[0].size == 20

        # Test with multiple intervals
        f2 = Chebfun.initfun(lambda x: np.sin(x), domain=[-1, 0, 1], n=15)
        assert len(f2.funs) == 2
        assert all(fun.size == 15 for fun in f2.funs)

    def test_initfun_adaptive_with_empty_domain(self):
        """Test initfun_adaptive with empty domain array."""
        f = Chebfun.initfun_adaptive(lambda x: x, domain=[])
        assert f.isempty

    def test_initfun_adaptive_with_single_point_domain(self):
        """Test initfun_adaptive with single point (should raise error)."""
        with pytest.raises(InvalidDomain, match="at least two points"):
            Chebfun.initfun_adaptive(lambda x: x, domain=[0])

    def test_initfun_adaptive_with_equal_endpoints(self):
        """Test initfun_adaptive with equal endpoints (should raise error)."""
        with pytest.raises(InvalidDomain, match="cannot be equal"):
            Chebfun.initfun_adaptive(lambda x: x, domain=[0, 0])
