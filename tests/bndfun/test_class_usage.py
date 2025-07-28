"""Unit-tests for miscelaneous Bndfun class usage."""

import numpy as np
import pytest

from chebpy.core.algorithms import standard_chop
from chebpy.core.bndfun import Bndfun
from chebpy.core.chebtech import Chebtech2
from chebpy.core.utilities import Interval

from .conftest import cos, eps, exp, pi, sin


@pytest.fixture
def class_usage_fixtures():
    """Create fixtures for testing Bndfun class usage."""
    subinterval = Interval(-2, 3)

    def f(x):
        return sin(30 * x)

    ff = Bndfun.initfun_adaptive(f, subinterval)
    xx = subinterval(np.linspace(-1, 1, 100))
    emptyfun = Bndfun(Chebtech2.initempty(), subinterval)
    constfun = Bndfun(Chebtech2.initconst(1.0), subinterval)

    return {"f": f, "ff": ff, "xx": xx, "emptyfun": emptyfun, "constfun": constfun, "subinterval": subinterval}


def test_isempty_True(class_usage_fixtures):
    """Test that emptyfun.isempty is True."""
    emptyfun = class_usage_fixtures["emptyfun"]
    assert emptyfun.isempty
    assert not (not emptyfun.isempty)


def test_isempty_False(class_usage_fixtures):
    """Test that constfun.isempty is False."""
    constfun = class_usage_fixtures["constfun"]
    assert not constfun.isempty
    assert constfun.isempty is False


def test_isconst_True(class_usage_fixtures):
    """Test that constfun.isconst is True."""
    constfun = class_usage_fixtures["constfun"]
    assert constfun.isconst
    assert not (not constfun.isconst)


def test_isconst_False(class_usage_fixtures):
    """Test that emptyfun.isconst is False."""
    emptyfun = class_usage_fixtures["emptyfun"]
    assert not emptyfun.isconst
    assert emptyfun.isconst is False


def test_size():
    """Test the size method of Bndfun."""
    cfs = np.random.rand(10)
    subinterval = Interval()
    b0 = Bndfun(Chebtech2(np.array([])), subinterval)
    b1 = Bndfun(Chebtech2(np.array([1.0])), subinterval)
    b2 = Bndfun(Chebtech2(cfs), subinterval)
    assert b0.size == 0
    assert b1.size == 1
    assert b2.size == cfs.size


def test_support(class_usage_fixtures):
    """Test the support property of Bndfun."""
    ff = class_usage_fixtures["ff"]
    a, b = ff.support
    assert a == -2
    assert b == 3


def test_endvalues(class_usage_fixtures):
    """Test the endvalues property of Bndfun."""
    ff = class_usage_fixtures["ff"]
    f = class_usage_fixtures["f"]
    a, b = ff.support
    fa, fb = ff.endvalues
    assert abs(fa - f(a)) <= 2e1 * eps
    assert abs(fb - f(b)) <= 2e1 * eps


def test_call(class_usage_fixtures):
    """Test calling a Bndfun object."""
    ff = class_usage_fixtures["ff"]
    xx = class_usage_fixtures["xx"]
    ff(xx)


def test_call_bary(class_usage_fixtures):
    """Test calling a Bndfun object with bary method."""
    ff = class_usage_fixtures["ff"]
    xx = class_usage_fixtures["xx"]
    ff(xx, "bary")
    ff(xx, how="bary")


def test_call_clenshaw(class_usage_fixtures):
    """Test calling a Bndfun object with clenshaw method."""
    ff = class_usage_fixtures["ff"]
    xx = class_usage_fixtures["xx"]
    ff(xx, "clenshaw")
    ff(xx, how="clenshaw")


def test_call_bary_vs_clenshaw(class_usage_fixtures):
    """Test that bary and clenshaw methods give similar results."""
    ff = class_usage_fixtures["ff"]
    xx = class_usage_fixtures["xx"]
    b = ff(xx, "clenshaw")
    c = ff(xx, "bary")
    assert np.max(b - c) <= 2e2 * eps


def test_call_raises(class_usage_fixtures):
    """Test that calling with invalid method raises ValueError."""
    ff = class_usage_fixtures["ff"]
    xx = class_usage_fixtures["xx"]
    with pytest.raises(ValueError):
        ff(xx, "notamethod")
    with pytest.raises(ValueError):
        ff(xx, how="notamethod")


def test_vscale_empty(class_usage_fixtures):
    """Test the vscale property of an empty Bndfun."""
    emptyfun = class_usage_fixtures["emptyfun"]
    assert emptyfun.vscale == 0.0


def test_copy(class_usage_fixtures):
    """Test the copy method of Bndfun."""
    ff = class_usage_fixtures["ff"]
    gg = ff.copy()
    assert ff == ff
    assert gg == gg
    assert ff != gg
    assert np.max(ff.coeffs - gg.coeffs) == 0


def test_restrict(class_usage_fixtures):
    """Test the restrict method of Bndfun."""
    ff = class_usage_fixtures["ff"]
    i1 = Interval(-1, 1)
    gg = ff.restrict(i1)
    yy = np.linspace(-1, 1, 1000)
    assert np.max(ff(yy) - gg(yy)) <= 1e2 * eps


def test_simplify(class_usage_fixtures):
    """Test the simplify method of Bndfun."""
    f = class_usage_fixtures["f"]
    interval = Interval(-2, 1)
    ff = Bndfun.initfun_fixedlen(f, interval, 1000)
    gg = ff.simplify()
    assert gg.size == standard_chop(ff.onefun.coeffs)
    assert np.max(ff.coeffs[: gg.size] - gg.coeffs) == 0
    assert ff.interval == gg.interval


def test_translate(class_usage_fixtures):
    """Test the translate method of Bndfun."""
    ff = class_usage_fixtures["ff"]
    c = -1
    shifted_interval = ff.interval + c
    gg = ff.translate(c)
    hh = Bndfun.initfun_fixedlen(lambda x: ff(x - c), shifted_interval, gg.size)
    yk = shifted_interval(np.linspace(-1, 1, 100))
    assert gg.interval == hh.interval
    assert np.max(gg.coeffs - hh.coeffs) <= 2e1 * eps
    assert np.max(gg(yk) - hh(yk)) <= 1e2 * eps


# --------------------------------------
#          vscale estimates
# --------------------------------------
vscales = [
    # (function, number of points, vscale)
    (lambda x: sin(4 * pi * x), [-2, 2], 1),
    (lambda x: cos(x), [-10, 1], 1),
    (lambda x: cos(4 * pi * x), [-100, 100], 1),
    (lambda x: exp(cos(4 * pi * x)), [-1, 1], exp(1)),
    (lambda x: cos(3244 * x), [-2, 0], 1),
    (lambda x: exp(x), [-1, 2], exp(2)),
    (lambda x: 1e10 * exp(x), [-1, 1], 1e10 * exp(1)),
    (lambda x: 0 * x + 1.0, [-1e5, 1e4], 1),
]


@pytest.mark.parametrize("fun, interval, vscale", vscales)
def test_vscale(fun, interval, vscale):
    """Test the vscale property of Bndfun."""
    subinterval = Interval(*interval)
    ff = Bndfun.initfun_adaptive(fun, subinterval)
    absdiff = abs(ff.vscale - vscale)
    assert absdiff <= 0.1 * vscale
