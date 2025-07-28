"""Unit-tests for chebpy/core/chebfun.py"""

import itertools
import operator
import unittest

import numpy as np

from chebpy import chebfun
from chebpy.core.chebfun import Chebfun
from chebpy.core.settings import DefaultPreferences
from chebpy.core.utilities import Domain, Interval
from chebpy.core.exceptions import (
    IntervalGap,
    IntervalOverlap,
    InvalidDomain,
    BadFunLengthArgument,
)
from chebpy.core.plotting import import_plt

from .utilities import infnorm, testfunctions, joukowsky

# in Python 3, the operator module does not have a 'div' method
binops = [operator.add, operator.mul, operator.sub, operator.truediv]
try:
    # in Python 2 we need to test div separately
    binops.append(operator.div)
    div_binops = (operator.div, operator.truediv)
except AttributeError:
    # Python 3
    div_binops = (operator.truediv,)

# aliases
pi = np.pi
sin = np.sin
cos = np.cos
exp = np.exp
eps = DefaultPreferences.eps

# domain, test_tolerance
chebfun_testdomains = [
    ([-1, 1], 2 * eps),
    ([-2, 1], eps),
    ([-1, 2], eps),
    ([-5, 9], 35 * eps),
]


class Algebra(unittest.TestCase):
    def setUp(self):
        self.emptyfun = Chebfun.initempty()
        self.yy = np.linspace(-1, 1, 2000)

    # check  +(empty Chebfun) = (empty Chebfun)
    def test__pos__empty(self):
        self.assertTrue((+self.emptyfun).isempty)

    # check -(empty Chebfun) = (empty Chebfun)
    def test__neg__empty(self):
        self.assertTrue((-self.emptyfun).isempty)

    # check (empty Chebfun) + (Chebfun) = (empty Chebfun)
    #   and (Chebfun) + (empty Chebfun) = (empty Chebfun)
    def test__add__radd__empty(self):
        for f, _, _ in testfunctions:
            for dom, _ in chebfun_testdomains:
                a, b = dom
                ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 13))
                self.assertTrue((self.emptyfun + ff).isempty)
                self.assertTrue((ff + self.emptyfun).isempty)

    # check the output of (constant + Chebfun)
    #                 and (Chebfun + constant)
    def test__add__radd__constant(self):
        for f, _, _ in testfunctions:
            for c in (-1, 1, 10, -1e5):

                def g(x):
                    return c + f(x)

                for dom, _ in chebfun_testdomains:
                    a, b = dom
                    xx = np.linspace(a, b, 1001)
                    ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 11))
                    gg1 = c + ff
                    gg2 = ff + c
                    vscl = ff.vscale
                    hscl = ff.hscale
                    lscl = max([fun.size for fun in ff])
                    tol = 2 * abs(c) * vscl * hscl * lscl * eps
                    self.assertLessEqual(infnorm(g(xx) - gg1(xx)), tol)
                    self.assertLessEqual(infnorm(g(xx) - gg2(xx)), tol)

    # check (empty Chebfun) - (Chebfun) = (empty Chebfun)
    #   and (Chebfun) - (empty Chebfun) = (empty Chebfun)
    def test__sub__rsub__empty(self):
        for f, _, _ in testfunctions:
            for dom, _ in chebfun_testdomains:
                a, b = dom
                ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 13))
                self.assertTrue((self.emptyfun - ff).isempty)
                self.assertTrue((ff - self.emptyfun).isempty)

    # check the output of (constant - Chebfun)
    #                 and (Chebfun - constant)
    def test__sub__rsub__constant(self):
        for f, _, _ in testfunctions:
            for c in (-1, 1, 10, -1e5):

                def g(x):
                    return c - f(x)

                for dom, _ in chebfun_testdomains:
                    a, b = dom
                    xx = np.linspace(a, b, 1001)
                    ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 11))
                    gg1 = c - ff
                    gg2 = ff - c
                    vscl = ff.vscale
                    hscl = ff.hscale
                    lscl = max([fun.size for fun in ff])
                    tol = 2 * abs(c) * vscl * hscl * lscl * eps
                    self.assertLessEqual(infnorm(g(xx) - gg1(xx)), tol)
                    self.assertLessEqual(infnorm(-g(xx) - gg2(xx)), tol)

    # check (empty Chebfun) * (Chebfun) = (empty Chebfun)
    #   and (Chebfun) * (empty Chebfun) = (empty Chebfun)
    def test__mul__rmul__empty(self):
        for f, _, _ in testfunctions:
            for dom, _ in chebfun_testdomains:
                a, b = dom
                ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 13))
                self.assertTrue((self.emptyfun * ff).isempty)
                self.assertTrue((ff * self.emptyfun).isempty)

    # check the output of (constant * Chebfun)
    #                 and (Chebfun * constant)
    def test__mul__rmul__constant(self):
        for f, _, _ in testfunctions:
            for c in (-1, 1, 10, -1e5):

                def g(x):
                    return c * f(x)

                for dom, _ in chebfun_testdomains:
                    a, b = dom
                    xx = np.linspace(a, b, 1001)
                    ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 11))
                    gg1 = c * ff
                    gg2 = ff * c
                    vscl = ff.vscale
                    hscl = ff.hscale
                    lscl = max([fun.size for fun in ff])
                    tol = 2 * abs(c) * vscl * hscl * lscl * eps
                    self.assertLessEqual(infnorm(g(xx) - gg1(xx)), tol)
                    self.assertLessEqual(infnorm(g(xx) - gg2(xx)), tol)

    # check (empty Chebfun) / (Chebfun) = (empty Chebfun)
    #   and (Chebfun) / (empty Chebfun) = (empty Chebfun)
    def test_truediv_empty(self):
        for f, _, _ in testfunctions:
            for dom, _ in chebfun_testdomains:
                a, b = dom
                ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 13))
                self.assertTrue((self.emptyfun / ff).isempty)
                self.assertTrue((ff / self.emptyfun).isempty)

    # check the output of (constant / Chebfun)
    #                 and (Chebfun / constant)
    def test_truediv_constant(self):
        for f, _, hasRoots in testfunctions:
            for c in (-1, 1, 10, -1e5):

                def g(x):
                    return f(x) / c

                def h(x):
                    return c / f(x)

                for dom, _ in chebfun_testdomains:
                    a, b = dom
                    xx = np.linspace(a, b, 1001)
                    ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 11))
                    gg = ff / c
                    vscl = gg.vscale
                    hscl = gg.hscale
                    lscl = max([fun.size for fun in gg])
                    tol = 2 * abs(c) * vscl * hscl * lscl * eps
                    self.assertLessEqual(infnorm(g(xx) - gg(xx)), tol)
                    # don't do the following test for functions with roots
                    if not hasRoots:
                        hh = c / ff
                        vscl = hh.vscale
                        hscl = hh.hscale
                        lscl = max([fun.size for fun in hh])
                        tol = 2 * abs(c) * vscl * hscl * lscl * eps
                        self.assertLessEqual(infnorm(h(xx) - hh(xx)), tol)

    # check (empty Chebfun) ** (Chebfun) = (empty Chebfun)
    def test_pow_empty(self):
        for f, _, _ in testfunctions:
            for dom, _ in chebfun_testdomains:
                a, b = dom
                ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 13))
                self.assertTrue((self.emptyfun**ff).isempty)

    # chec (Chebfun) ** (empty Chebfun) = (empty Chebfun)
    def test_rpow_empty(self):
        for f, _, _ in testfunctions:
            for dom, _ in chebfun_testdomains:
                a, b = dom
                ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 13))
                self.assertTrue((ff**self.emptyfun).isempty)

    # check the output of (Chebfun) ** (constant)
    def test_pow_constant(self):
        for (_, _), (f, _) in powtestfuns:
            for c in (1, 2, 3):

                def g(x):
                    return f(x) ** c

                for dom, _ in powtestdomains:
                    a, b = dom
                    xx = np.linspace(a, b, 1001)
                    ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 11))
                    gg = ff**c
                    vscl = gg.vscale
                    hscl = gg.hscale
                    lscl = max([fun.size for fun in gg])
                    tol = 2 * abs(c) * vscl * hscl * lscl * eps
                    self.assertLessEqual(infnorm(g(xx) - gg(xx)), tol)

    # check the output of (constant) ** (Chebfun)
    def test_rpow_constant(self):
        for (_, _), (f, _) in powtestfuns:
            for c in (1, 2, 3):

                def g(x):
                    return c ** f(x)

                for dom, _ in powtestdomains:
                    a, b = dom
                    xx = np.linspace(a, b, 1001)
                    ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 11))
                    gg = c**ff
                    vscl = gg.vscale
                    hscl = gg.hscale
                    lscl = max([fun.size for fun in gg])
                    tol = 2 * abs(c) * vscl * hscl * lscl * eps
                    self.assertLessEqual(infnorm(g(xx) - gg(xx)), tol)


# add tests for the binary operators
def binaryOpTester(f, g, binop, dom, tol):
    a, b = dom
    xx = np.linspace(a, b, 1001)
    n, m = 3, 8
    ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, n + 1))
    gg = Chebfun.initfun_adaptive(g, np.linspace(a, b, m + 1))

    def FG(x):
        return binop(f(x), g(x))

    fg = binop(ff, gg)

    def tester(self):
        vscl = max([ff.vscale, gg.vscale])
        hscl = max([ff.hscale, gg.hscale])
        lscl = max([fun.size for fun in np.append(ff.funs, gg.funs)])
        self.assertEqual(ff.funs.size, n)
        self.assertEqual(gg.funs.size, m)
        self.assertEqual(fg.funs.size, n + m - 1)
        self.assertLessEqual(infnorm(fg(xx) - FG(xx)), vscl * hscl * lscl * tol)

    return tester


for binop in binops:
    for (f, _, _), (g, _, denomHasRoots) in itertools.combinations(testfunctions, 2):
        for dom, tol in chebfun_testdomains:
            if binop in div_binops and denomHasRoots:
                # skip truediv test if denominator has roots on the real line
                pass
            else:
                _testfun_ = binaryOpTester(f, g, binop, dom, 2 * tol)
                a, b = dom
                binopname = binop.__name__
                # case of truediv: add leading and trailing underscores
                if binopname[0] != "_":
                    binopname = "_" + binopname
                if binopname[-1] != "_":
                    binopname = binopname + "_"
                _testfun_.__name__ = "test_{}_{}_{}_[{:.0f},..,{:.0f}]".format(
                    binopname, f.__name__, g.__name__, a, b
                )
                setattr(Algebra, _testfun_.__name__, _testfun_)

powtestfuns = (
    [(np.exp, "exp"), (np.sin, "sin")],
    [(np.exp, "exp"), (lambda x: 2 - x, "linear")],
    [(lambda x: 2 - x, "linear"), (np.exp, "exp")],
)

powtestdomains = [
    ([-0.5, 0.9], eps),
    ([-1.2, 1.3], eps),
    ([-2.2, -1.9], eps),
    ([0.4, 1.3], eps),
]

# add operator.pow tests
for (f, namef), (g, nameg) in powtestfuns:
    for dom, tol in powtestdomains:
        _testfun_ = binaryOpTester(f, g, operator.pow, dom, 3 * tol)
        _testfun_.__name__ = "test_{}_{}_{}_[{:.1f},..,{:.1f}]".format(
            "pow", namef, nameg, *dom
        )
        setattr(Algebra, _testfun_.__name__, _testfun_)


# add tests for the unary operators
def unaryOpTester(f, unaryop, dom, tol):
    a, b = dom
    xx = np.linspace(a, b, 1001)
    ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 9))

    def GG(x):
        return unaryop(f(x))

    gg = unaryop(ff)

    def tester(self):
        vscl = ff.vscale
        hscl = ff.hscale
        lscl = max([fun.size for fun in ff])
        self.assertEqual(ff.funs.size, gg.funs.size)
        self.assertLessEqual(infnorm(gg(xx) - GG(xx)), vscl * hscl * lscl * tol)

    return tester


unaryops = (operator.pos, operator.neg)
for unaryop in unaryops:
    for f, _, _ in testfunctions:
        for dom, tol in chebfun_testdomains:
            _testfun_ = unaryOpTester(f, unaryop, dom, tol)
            _testfun_.__name__ = "test_{}_{}_[{:.0f},..,{:.0f}]".format(
                unaryop.__name__, f.__name__, dom[0], dom[1]
            )
            setattr(Algebra, _testfun_.__name__, _testfun_)

ufuncs = (
    np.absolute,
    np.arccos,
    np.arccosh,
    np.arcsin,
    np.arcsinh,
    np.arctan,
    np.arctanh,
    np.cos,
    np.cosh,
    np.exp,
    np.exp2,
    np.expm1,
    np.log,
    np.log2,
    np.log10,
    np.log1p,
    np.sinh,
    np.sin,
    np.tan,
    np.tanh,
    np.sqrt,
)


# empty-case tests
def ufuncEmptyCaseTester(ufunc):
    def tester(self):
        self.assertTrue(getattr(self.emptyfun, ufunc.__name__)().isempty)

    return tester


for ufunc in ufuncs:
    _testfun_ = ufuncEmptyCaseTester(ufunc)
    _testfun_.__name__ = "test_emptycase_{}".format(ufunc.__name__)
    setattr(Ufuncs, _testfun_.__name__, _testfun_)

# TODO: Add more test cases
# add ufunc tests:
#     (ufunc, [([fun1, interval1], tol1), ([fun2, interval2], tol2), ... ])


def uf1(x):
    """uf1.__name__ = "x" """
    return x


def uf2(x):
    """uf2.__name__ = "sin(x-.5)" """
    return sin(x - 0.5)


def uf3(x):
    """uf3.__name__ = "sin(25*x-1)" """
    return sin(25 * x - 1)


ufunc_test_params = [
    (
        np.absolute,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.arccos,
        [
            ([uf1, (-0.8, 0.8)], eps),
        ],
    ),
    (
        np.arccosh,
        [
            ([uf1, (2, 3)], eps),
        ],
    ),
    (
        np.arcsin,
        [
            ([uf1, (-0.8, 0.8)], eps),
        ],
    ),
    (
        np.arcsinh,
        [
            ([uf1, (2, 3)], eps),
        ],
    ),
    (
        np.arctan,
        [
            ([uf1, (-0.8, 0.8)], eps),
        ],
    ),
    (
        np.arctanh,
        [
            ([uf1, (-0.8, 0.8)], eps),
        ],
    ),
    (
        np.cos,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.cosh,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        exp,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.exp2,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.expm1,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.log,
        [
            ([uf1, (2, 3)], eps),
        ],
    ),
    (
        np.log2,
        [
            ([uf1, (2, 3)], eps),
        ],
    ),
    (
        np.log10,
        [
            ([uf1, (2, 3)], eps),
        ],
    ),
    (
        np.log1p,
        [
            ([uf1, (-0.8, 0.8)], eps),
        ],
    ),
    (
        np.sinh,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.sin,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.tan,
        [
            ([uf1, (-0.8, 0.8)], eps),
        ],
    ),
    (
        np.tanh,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.sqrt,
        [
            ([uf1, (2, 3)], eps),
        ],
    ),
    (
        np.absolute,
        [
            ([uf2, (-3, 3)], eps),
        ],
    ),
    (
        np.cos,
        [
            ([uf2, (-3, 3)], eps),
        ],
    ),
    (
        np.cosh,
        [
            ([uf2, (-3, 3)], eps),
        ],
    ),
    (
        np.exp,
        [
            ([uf2, (-3, 3)], eps),
        ],
    ),
    (
        np.expm1,
        [
            ([uf2, (-3, 3)], eps),
        ],
    ),
    (
        np.sinh,
        [
            ([uf2, (-3, 3)], eps),
        ],
    ),
    (
        np.sin,
        [
            ([uf2, (-3, 3)], eps),
        ],
    ),
    (
        np.tan,
        [
            ([uf2, (-0.8, 0.8)], eps),
        ],
    ),
    (
        np.tanh,
        [
            ([uf2, (-3, 3)], eps),
        ],
    ),
    (
        np.absolute,
        [
            ([uf3, (-3, 3)], eps),
        ],
    ),
    (
        np.cos,
        [
            ([uf3, (-3, 3)], eps),
        ],
    ),
    (
        np.cosh,
        [
            ([uf3, (-3, 3)], eps),
        ],
    ),
    (
        np.exp,
        [
            ([uf3, (-3, 3)], eps),
        ],
    ),
    (
        np.expm1,
        [
            ([uf3, (-3, 3)], eps),
        ],
    ),
    (
        np.sinh,
        [
            ([uf3, (-3, 3)], eps),
        ],
    ),
    (
        np.sin,
        [
            ([uf3, (-3, 3)], eps),
        ],
    ),
    (
        np.tan,
        [
            ([uf3, (-0.8, 0.8)], eps),
        ],
    ),
    (
        np.tanh,
        [
            ([uf3, (-3, 3)], eps),
        ],
    ),
]


def ufuncTester(ufunc, f, interval, tol):
    a, b = interval
    ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 13))

    def gg(x):
        return ufunc(f(x))

    GG = getattr(ff, ufunc.__name__)()

    def tester(self):
        xx = interval(self.yy)
        vscl = GG.vscale
        lscl = sum([fun.size for fun in GG])
        self.assertLessEqual(infnorm(gg(xx) - GG(xx)), vscl * lscl * tol)

    return tester


for (
    ufunc,
    [
        ([f, intvl], tol),
    ],
) in ufunc_test_params:
    interval = Interval(*intvl)
    _testfun_ = ufuncTester(ufunc, f, interval, tol)
    _testfun_.__name__ = "test_{}({})_[{:.1f},..,{:.1f}]".format(
        ufunc.__name__, f.__name__, *intvl
    )
    setattr(Ufuncs, _testfun_.__name__, _testfun_)

plt = import_plt()

# domain, test_tolerance
domainBreakOp_args = [
    (lambda x: x, 0, [-1, 1], eps),
    (sin, cos, [-1, 1], eps),
    (cos, np.abs, [-1, 0, 1], eps),
]


# add tests for maximum, minimum


for domainBreakOp in (np.maximum, np.minimum):
    for n, args in enumerate(domainBreakOp_args):
        ff, gg, dom, tol = args
        _testfun_ = domainBreakOpTester(domainBreakOp, ff, gg, dom, tol)
        _testfun_.__name__ = "test_{}_{}".format(domainBreakOp.__name__, n)
        setattr(DomainBreakingOps, _testfun_.__name__, _testfun_)

# reset the testsfun variable so it doesn't get picked up by nose
_testfun_ = None
