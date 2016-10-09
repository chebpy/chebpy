# -*- coding: utf-8 -*-

"""Unit-tests for pyfun/core/chebfun.py"""

from __future__ import division

from operator import __add__
from operator import __mul__
from operator import __neg__
from operator import __pos__
from operator import __sub__
from operator import truediv

binops = [__add__, __mul__, __sub__, truediv]

try:
    # for Python 2 we need to test div separately
    from operator import __div__
    binops.append(__div__)
    div_binops = (__div__, truediv)
except ImportError:
    # Python 3
    div_binops = (truediv,)

from unittest import TestCase
from itertools import combinations

from numpy import arccos
from numpy import arccosh
from numpy import arcsin
from numpy import arcsinh
from numpy import arctan
from numpy import arctanh
from numpy import cos
from numpy import cosh
from numpy import exp
from numpy import exp2
from numpy import expm1
from numpy import sin
from numpy import sinh
from numpy import tan
from numpy import tanh
from numpy import log
from numpy import log2
from numpy import log10
from numpy import log1p
from numpy import sqrt
from numpy import ndarray
from numpy import array
from numpy import arange
from numpy import append
from numpy import sum
from numpy import abs
from numpy import pi
from numpy import linspace
from numpy import maximum
from numpy import minimum
from numpy import equal
from numpy import isscalar
from numpy import isfinite
from numpy.random import rand

from matplotlib.pyplot import subplots

from chebpy.core.bndfun import Bndfun
from chebpy.core.chebfun import Chebfun
from chebpy.core.settings import DefaultPrefs
from chebpy.core.utilities import Domain
from chebpy.core.utilities import Interval
from chebpy.core.exceptions import IntervalGap
from chebpy.core.exceptions import IntervalOverlap
from chebpy.core.exceptions import BadDomainArgument
from chebpy.core.exceptions import BadFunLengthArgument

from chebpy import chebfun

from tests.utilities import infnorm
from tests.utilities import testfunctions

eps = DefaultPrefs.eps


class Construction(TestCase):

    def setUp(self):
        f = lambda x: exp(x)
        self.f = f
        self.fun0 = Bndfun.initfun_adaptive(f, Interval(-1,0))
        self.fun1 = Bndfun.initfun_adaptive(f, Interval(0,1))
        self.fun2 = Bndfun.initfun_adaptive(f, Interval(-.5,0.5))
        self.fun3 = Bndfun.initfun_adaptive(f, Interval(2,2.5))
        self.fun4 = Bndfun.initfun_adaptive(f, Interval(-3,-2))
        self.funs_a = array([self.fun1, self.fun0, self.fun2])
        self.funs_b = array([self.fun1, self.fun2])
        self.funs_c = array([self.fun0, self.fun3])
        self.funs_d = array([self.fun1, self.fun4])

    def test__init__pass(self):
        Chebfun([self.fun0])
        Chebfun([self.fun1])
        Chebfun([self.fun2])
        Chebfun([self.fun0, self.fun1])

    def test__init__fail(self):
        self.assertRaises(IntervalOverlap, Chebfun, self.funs_a)
        self.assertRaises(IntervalOverlap, Chebfun, self.funs_b)
        self.assertRaises(IntervalGap, Chebfun, self.funs_c)
        self.assertRaises(IntervalGap, Chebfun, self.funs_d)

    def test_initempty(self):
        emptyfun = Chebfun.initempty()
        self.assertEqual(emptyfun.funs.size, 0)

    def test_initconst(self):
        self.assertTrue(Chebfun.initconst(1, [-1,1]).isconst)
        self.assertTrue(Chebfun.initconst(-10, linspace(-1,1,11)).isconst)
        self.assertTrue(Chebfun.initconst(3, [-2,0,1]).isconst)
        self.assertTrue(Chebfun.initconst(3.14, linspace(-100,-90,11)).isconst)
        self.assertFalse(Chebfun([self.fun0]).isconst)
        self.assertFalse(Chebfun([self.fun1]).isconst)
        self.assertFalse(Chebfun([self.fun2]).isconst)
        self.assertFalse(Chebfun([self.fun0, self.fun1]).isconst)

    def test_initidentity(self):
        _doms = (
            linspace(-1,1,2),
            linspace(-1,1,11),
            linspace(-10,17,351),
            linspace(-9.3,-3.2,22),  
            linspace(2.5,144.3,2112),
        )
        for _dom in _doms:
            ff = Chebfun.initidentity(_dom)
            a, b = ff.support
            xx = linspace(a, b, 1001)
            tol = eps * ff.hscale
            self.assertLessEqual(infnorm(ff(xx)-xx), tol)
        # test the default case
        ff = Chebfun.initidentity()
        a, b = ff.support
        xx = linspace(a, b, 1001)
        tol = eps * ff.hscale
        self.assertLessEqual(infnorm(ff(xx)-xx), tol)     

    def test_initfun_adaptive_continuous_domain(self):
        ff = Chebfun.initfun_adaptive(self.f, [-2,-1])
        self.assertEqual(ff.funs.size, 1)
        a, b = ff.breakdata.keys()
        fa, fb, = ff.breakdata.values()
        self.assertEqual(a,-2)
        self.assertEqual(b,-1)
        self.assertLessEqual(abs(fa-self.f(-2)), eps)
        self.assertLessEqual(abs(fb-self.f(-1)), eps)

    def test_initfun_adaptive_piecewise_domain(self):
        ff = Chebfun.initfun_adaptive(self.f, [-2,0,1])
        self.assertEqual(ff.funs.size, 2)
        a, b, c = ff.breakdata.keys()
        fa, fb, fc = ff.breakdata.values()
        self.assertEqual(a,-2)
        self.assertEqual(b, 0)
        self.assertEqual(c, 1)
        self.assertLessEqual(abs(fa-self.f(-2)), eps)
        self.assertLessEqual(abs(fb-self.f( 0)), eps)
        self.assertLessEqual(abs(fc-self.f( 1)), 2*eps)

    def test_initfun_adaptive_raises(self):
        initfun = Chebfun.initfun_adaptive
        self.assertRaises(BadDomainArgument, initfun, self.f, [-2])
        self.assertRaises(BadDomainArgument, initfun, self.f, domain=[-2])
        self.assertRaises(BadDomainArgument, initfun, self.f, domain=None)

    def test_initfun_fixedlen_continuous_domain(self):
        ff = Chebfun.initfun_fixedlen(self.f, 20, [-2,-1])
        self.assertEqual(ff.funs.size, 1)
        a, b = ff.breakdata.keys()
        fa, fb, = ff.breakdata.values()
        self.assertEqual(a,-2)
        self.assertEqual(b,-1)
        self.assertLessEqual(abs(fa-self.f(-2)), eps)
        self.assertLessEqual(abs(fb-self.f(-1)), eps)

    def test_initfun_fixedlen_piecewise_domain_0(self):
        ff = Chebfun.initfun_fixedlen(self.f, 30, [-2,0,1])
        self.assertEqual(ff.funs.size, 2)
        a, b, c = ff.breakdata.keys()
        fa, fb, fc = ff.breakdata.values()
        self.assertEqual(a,-2)
        self.assertEqual(b, 0)
        self.assertEqual(c, 1)
        self.assertLessEqual(abs(fa-self.f(-2)), 3*eps)
        self.assertLessEqual(abs(fb-self.f( 0)), 3*eps)
        self.assertLessEqual(abs(fc-self.f( 1)), 3*eps)

    def test_initfun_fixedlen_piecewise_domain_1(self):
        ff = Chebfun.initfun_fixedlen(self.f, [30,20], [-2,0,1])
        self.assertEqual(ff.funs.size, 2)
        a, b, c = ff.breakdata.keys()
        fa, fb, fc = ff.breakdata.values()
        self.assertEqual(a,-2)
        self.assertEqual(b, 0)
        self.assertEqual(c, 1)
        self.assertLessEqual(abs(fa-self.f(-2)), 3*eps)
        self.assertLessEqual(abs(fb-self.f( 0)), 3*eps)
        self.assertLessEqual(abs(fc-self.f( 1)), 6*eps)

    def test_initfun_fixedlen_raises(self):
        initfun = Chebfun.initfun_fixedlen
        self.assertRaises(BadDomainArgument, initfun, self.f, 10, [-2])
        self.assertRaises(BadDomainArgument, initfun, self.f, n=10, domain=[-2])
        self.assertRaises(BadDomainArgument, initfun, self.f, n=10, domain=None)
        self.assertRaises(BadFunLengthArgument, initfun, self.f, [30,40], [-1,1])
        self.assertRaises(TypeError, initfun, self.f, None, [-1,1])

    def test_initfun_fixedlen_succeeds(self):
        self.assertTrue(Chebfun.initfun_fixedlen(self.f, [], [-2,-1,0]).isempty)
        # check that providing a vector with None elements calls the
        # Tech adaptive constructor
        g0 = Chebfun.initfun_adaptive(self.f, [-2,-1,0])
        g1 = Chebfun.initfun_fixedlen(self.f, [None,None], [-2,-1,0])
        g2 = Chebfun.initfun_fixedlen(self.f, [None,40], [-2,-1,0])
        for fun1, fun2 in zip(g1,g0):
            self.assertEqual(sum(fun1.coeffs-fun2.coeffs), 0)
        self.assertEqual(sum(g2.funs[0].coeffs-g0.funs[0].coeffs), 0)


class Properties(TestCase):

    def setUp(self):
        self.f0 = Chebfun.initempty()
        self.f1 = Chebfun.initfun_adaptive(lambda x: x**2, [-1,1])
        self.f2 = Chebfun.initfun_adaptive(lambda x: x**2, [-1,0,1,2])

    def test_breakpoints(self):
        self.assertEqual(self.f0.breakpoints.size, 0)
        self.assertTrue(equal(self.f1.breakpoints,[-1,1]).all())
        self.assertTrue(equal(self.f2.breakpoints,[-1,0,1,2]).all())

    def test_domain(self):
        d1 = Domain([-1,1])
        d2 = Domain([-1,0,1,2])
        self.assertIsInstance(self.f0.domain, ndarray)
        self.assertIsInstance(self.f1.domain, Domain)
        self.assertIsInstance(self.f2.domain, Domain)
        self.assertEqual(self.f0.domain.size, 0)
        self.assertEqual(self.f1.domain, d1)
        self.assertEqual(self.f2.domain, d2)

    def test_hscale(self):
        self.assertEqual(self.f0.hscale, 0)
        self.assertEqual(self.f1.hscale, 1)
        self.assertEqual(self.f2.hscale, 2)

    def test_isempty(self):
        self.assertTrue(self.f0.isempty)
        self.assertFalse(self.f1.isempty)
        self.assertFalse(self.f2.isempty)

    def test_isconst(self):
        self.assertFalse(self.f0.isconst)
        self.assertFalse(self.f1.isconst)
        self.assertFalse(self.f2.isconst)
        c1 = Chebfun.initfun_fixedlen(lambda x: 0*x+3, 1, [-2,-1,0,1,2,3])
        c2 = Chebfun.initfun_fixedlen(lambda x: 0*x-1, 1, [-2,3])
        self.assertTrue(c1.isconst)
        self.assertTrue(c2.isconst)

    def test_support(self):
        self.assertIsInstance(self.f0.support, ndarray)
        self.assertIsInstance(self.f1.support, ndarray)
        self.assertIsInstance(self.f2.support, ndarray)
        self.assertEqual(self.f0.support.size, 0)
        self.assertTrue(equal(self.f1.support,[-1,1]).all())
        self.assertTrue(equal(self.f2.support,[-1,2]).all())

    def test_vscale(self):
        self.assertEqual(self.f0.vscale, 0)
        self.assertEqual(self.f1.vscale, 1)
        self.assertEqual(self.f2.vscale, 4)


class ClassUsage(TestCase):

    def setUp(self):
        self.f0 = Chebfun.initempty()
        self.f1 = Chebfun.initfun_adaptive(lambda x: x**2, [-1,1])
        self.f2 = Chebfun.initfun_adaptive(lambda x: x**2, [-1,0,1,2])

    def test_copy(self):
        f0_copy = self.f0.copy()
        f1_copy = self.f1.copy()
        f2_copy = self.f2.copy()
        self.assertTrue(f0_copy.isempty)
        self.assertEquals(f1_copy.funs.size, 1)
        for k in range(self.f1.funs.size):
            fun = self.f1.funs[k]
            funcopy = f1_copy.funs[k]
            self.assertNotEqual(fun, funcopy)
            self.assertEquals(sum(fun.coeffs-funcopy.coeffs), 0)
        for k in range(self.f2.funs.size):
            fun = self.f2.funs[k]
            funcopy = f2_copy.funs[k]
            self.assertNotEqual(fun, funcopy)
            self.assertEquals(sum(fun.coeffs-funcopy.coeffs), 0)

    def test__iter__(self):
        for f in [self.f0, self.f1, self.f2]:
            a1 = [x for x in f]
            a2 = [x for x in f.funs]
            self.assertTrue(equal(a1,a2).all())

    def test_x_property(self):
        _doms = (
            linspace(-1,1,2),
            linspace(-1,1,11),
            linspace(-9.3,-3.2,22),
        )
        for _dom in _doms:
            f = Chebfun.initfun_fixedlen(sin, 1000, _dom)
            x = f.x
            a, b = x.support
            pts = linspace(a, b, 1001)
            tol = eps * f.hscale
            self.assertLessEqual(infnorm(x(pts)-pts), tol)

    def test_restrict_(self):
        # test a variety of domains with breaks
        doms = [(-4,4), (-4,0,4), (-2,-1, 0.3, 1, 2.5)]
        for dom in doms:
            ff = Chebfun.initfun_fixedlen(cos, 25, domain=dom)
            # define some arbitrary subdomains
            yy = linspace(dom[0], dom[-1], 11)
            subdoms = [yy, yy[2:7], yy[::2]]
            for subdom in subdoms:
                xx = linspace(subdom[0], subdom[-1], 1001)
                gg = ff._restrict(subdom)
                vscl = ff.vscale
                hscl = ff.hscale
                lscl = max([fun.size for fun in ff])
                tol = vscl*hscl*lscl*eps
                # sample the restricted function and comapre with original
                self.assertLessEqual(infnorm(ff(xx)-gg(xx)), tol)
                # check there are at least as many funs as subdom elements
                self.assertGreaterEqual(len(gg.funs), len(subdom)-1)
                for fun in gg:
                    # chec each fun has length 25
                    self.assertEqual(fun.size, 25)

    def test_restrict__empty(self):
        self.assertTrue(self.f0._restrict([-1,1]).isempty)

    def test_simplify(self):
        dom = linspace(-2,1.5,13)
        f = chebfun(cos, dom, 70).simplify()
        g = chebfun(cos, dom)
        self.assertEquals(f.domain, g.domain)
        for n, fun in enumerate(f):
            # we allow one degree of freedom difference
            # TODO: check this
            self.assertLessEqual(fun.size-g.funs[n].size, 1)

    def test_simplify_empty(self):
        self.assertTrue(self.f0.simplify().isempty)

    def test_restrict(self):
        dom1 = Domain(linspace(-2,1.5,13))
        dom2 = Domain(linspace(-1.7,0.93,17))
        dom3 = dom1.merge(dom2).restrict(dom2)
        f = chebfun(cos, dom1).restrict(dom2)
        g = chebfun(cos, dom3)
        self.assertEquals(f.domain, g.domain)
        for n, fun in enumerate(f):
            # we allow two degrees of freedom difference either way
            # TODO: once standard chop is fixed, may be able to reduce 4 to 0
            self.assertLessEqual(fun.size-g.funs[n].size, 4)

    def test_restrict_empty(self):
        self.assertTrue(self.f0.restrict([-1,1]).isempty)

class Algebra(TestCase):

    def setUp(self):
        self.emptyfun = Chebfun.initempty()
        self.yy = -1 + 2*rand(1000)

    # check  +(empty Chebfun) = (empty Chebfun)
    def test__pos__empty(self):
        self.assertTrue((+self.emptyfun).isempty)

    # check -(empty Chebfun) = (empty Chebfun)
    def test__neg__empty(self):
        self.assertTrue((-self.emptyfun).isempty)

    # check (empty Chebfun) + (Chebfun) = (empty Chebfun)
    #   and (Chebfun) + (empty Chebfun) = (empty Chebfun)
    def test__add__radd__empty(self):
        for (f, _, _) in testfunctions:
            for dom, _ in chebfun_testdomains:
                a, b = dom
                ff = Chebfun.initfun_adaptive(f, linspace(a, b, 13))
                self.assertTrue((self.emptyfun+ff).isempty)
                self.assertTrue((ff+self.emptyfun).isempty)

    # check the output of (constant + Chebfun)
    #                 and (Chebfun + constant)
    def test__add__radd__constant(self):
        for (f, _, _) in testfunctions:
            for c in (-1, 1, 10, -1e5):
                for dom, _ in chebfun_testdomains:
                    a, b = dom
                    xx = linspace(a, b, 1001)
                    ff = Chebfun.initfun_adaptive(f, linspace(a, b, 11))
                    g = lambda x: c + f(x)
                    gg1 = c + ff
                    gg2 = ff + c
                    vscl = ff.vscale
                    hscl = ff.hscale
                    lscl = max([fun.size for fun in ff])
                    tol = 2*abs(c)*vscl*hscl*lscl*eps
                    self.assertLessEqual(infnorm(g(xx)-gg1(xx)), tol)
                    self.assertLessEqual(infnorm(g(xx)-gg2(xx)), tol)

    # check (empty Chebfun) - (Chebfun) = (empty Chebfun)
    #   and (Chebfun) - (empty Chebfun) = (empty Chebfun)
    def test__sub__rsub__empty(self):
        for (f, _, _) in testfunctions:
            for dom, _ in chebfun_testdomains:
                a, b = dom
                ff = Chebfun.initfun_adaptive(f, linspace(a, b, 13))
                self.assertTrue((self.emptyfun-ff).isempty)
                self.assertTrue((ff-self.emptyfun).isempty)

    # check the output of (constant - Chebfun)
    #                 and (Chebfun - constant)
    def test__sub__rsub__constant(self):
        for (f, _, _) in testfunctions:
            for c in (-1, 1, 10, -1e5):
                for dom, _ in chebfun_testdomains:
                    a, b = dom
                    xx = linspace(a, b, 1001)
                    ff = Chebfun.initfun_adaptive(f, linspace(a, b, 11))
                    g = lambda x: c - f(x)
                    gg1 = c - ff
                    gg2 = ff - c
                    vscl = ff.vscale
                    hscl = ff.hscale
                    lscl = max([fun.size for fun in ff])
                    tol = 2*abs(c)*vscl*hscl*lscl*eps
                    self.assertLessEqual(infnorm(g(xx)-gg1(xx)), tol)
                    self.assertLessEqual(infnorm(-g(xx)-gg2(xx)), tol)

    # check (empty Chebfun) * (Chebfun) = (empty Chebfun)
    #   and (Chebfun) * (empty Chebfun) = (empty Chebfun)
    def test__mul__rmul__empty(self):
        for (f, _, _) in testfunctions:
            for dom, _ in chebfun_testdomains:
                a, b = dom
                ff = Chebfun.initfun_adaptive(f, linspace(a, b, 13))
                self.assertTrue((self.emptyfun*ff).isempty)
                self.assertTrue((ff*self.emptyfun).isempty)

    # check the output of (constant * Chebfun)
    #                 and (Chebfun * constant)
    def test__mul__rmul__constant(self):
        for (f, _, _) in testfunctions:
            for c in (-1, 1, 10, -1e5):
                for dom, _ in chebfun_testdomains:
                    a,b = dom
                    xx = linspace(a, b, 1001)
                    ff = Chebfun.initfun_adaptive(f, linspace(a, b, 11))
                    g = lambda x: c * f(x)
                    gg1 = c * ff
                    gg2 = ff * c
                    vscl = ff.vscale
                    hscl = ff.hscale
                    lscl = max([fun.size for fun in ff])
                    tol = 2*abs(c)*vscl*hscl*lscl*eps
                    self.assertLessEqual(infnorm(g(xx)-gg1(xx)), tol)
                    self.assertLessEqual(infnorm(g(xx)-gg2(xx)), tol)

    # check (empty Chebfun) / (Chebfun) = (empty Chebfun)
    #   and (Chebfun) / (empty Chebfun) = (empty Chebfun)
    def test_truediv_empty(self):
        for (f, _, _) in testfunctions:
            for dom, _ in chebfun_testdomains:
                a, b = dom
                ff = Chebfun.initfun_adaptive(f, linspace(a, b, 13))
                self.assertTrue((self.emptyfun/ff).isempty)
                self.assertTrue((ff/self.emptyfun).isempty)

    # check the output of (constant / Chebfun)
    #                 and (Chebfun / constant)
    def test_truediv_constant(self):
        for (f, _, hasRoots) in testfunctions:
            for c in (-1, 1, 10, -1e5):
                for dom, _ in chebfun_testdomains:
                    a,b = dom
                    xx = linspace(a, b, 1001)
                    ff = Chebfun.initfun_adaptive(f, linspace(a, b, 11))
                    g = lambda x: f(x) / c
                    gg = ff / c
                    vscl = gg.vscale
                    hscl = gg.hscale
                    lscl = max([fun.size for fun in gg])
                    tol = 2*abs(c)*vscl*hscl*lscl*eps
                    self.assertLessEqual(infnorm(g(xx)-gg(xx)), tol)
                    # don't do the following test for functions with roots
                    if not hasRoots:
                        h = lambda x: c / f(x)
                        hh = c / ff
                        vscl = hh.vscale
                        hscl = hh.hscale
                        lscl = max([fun.size for fun in hh])
                        tol = 2*abs(c)*vscl*hscl*lscl*eps
                        self.assertLessEqual(infnorm(h(xx)-hh(xx)), tol)


# domain, test_tolerance
chebfun_testdomains = [
    ([-1,1], 2*eps),
    ([-2,1], eps),
    ([-1,2], eps),
    ([-5,9], 35*eps),
]

# add tests for the binary operators
def binaryOpTester(f, g, binop, dom, tol):
    a, b = dom
    xx = linspace(a,b,1001)
    n, m = 3, 8
    ff = Chebfun.initfun_adaptive(f, linspace(a,b,n+1))
    gg = Chebfun.initfun_adaptive(g, linspace(a,b,m+1))
    FG = lambda x: binop(f(x), g(x))
    fg = binop(ff, gg)
    def tester(self):
        vscl = max([ff.vscale, gg.vscale])
        hscl = max([ff.hscale, gg.hscale])
        lscl = max([fun.size for fun in append(ff.funs, gg.funs)])
        self.assertEqual(ff.funs.size, n)
        self.assertEqual(gg.funs.size, m)
        self.assertEqual(fg.funs.size, n+m-1)
        self.assertLessEqual(infnorm(fg(xx)-FG(xx)), vscl*hscl*lscl*tol)
    return tester


for binop in binops:
    for (f, _, _), (g, _, denomHasRoots) in combinations(testfunctions, 2):
        for dom, tol in chebfun_testdomains:
            if binop in div_binops and denomHasRoots:
                # skip truediv test if denominator has roots on the real line
                pass
            else:
                _testfun_ = binaryOpTester(f, g, binop, dom, 2*tol)
                a, b = dom
                binopname = binop.__name__
                # case of truediv: add leading and trailing underscores
                if binopname[0] != '_':
                    binopname = '_' + binopname
                if binopname[-1] != '_':
                    binopname = binopname + '_'
                _testfun_.__name__ = \
                    "test{}{}_{}_[{:.0f},..,{:.0f}]".format(
                        binopname, f.__name__,  g.__name__, a, b)
                setattr(Algebra, _testfun_.__name__, _testfun_)


# add tests for the unary operators
def unaryOpTester(f, unaryop, dom, tol):
    a, b = dom
    xx = linspace(a,b,1001)
    ff = Chebfun.initfun_adaptive(f, linspace(a,b,9))
    GG = lambda x: unaryop(f(x))
    gg = unaryop(ff)
    def tester(self):
        vscl = ff.vscale
        hscl = ff.hscale
        lscl = max([fun.size for fun in ff])
        self.assertEqual(ff.funs.size, gg.funs.size)
        self.assertLessEqual(infnorm(gg(xx)-GG(xx)), vscl*hscl*lscl*tol)
    return tester

unaryops = (
    __pos__,
    __neg__,
    )

for unaryop in unaryops:
    for (f, _, _) in testfunctions:
        for dom, tol in chebfun_testdomains:
            _testfun_ = unaryOpTester(f, unaryop, dom, tol)
            _testfun_.__name__ = \
                "test{}{}_[{:.0f},..,{:.0f}]".format(
                    unaryop.__name__, f.__name__, dom[0], dom[1])
            setattr(Algebra, _testfun_.__name__, _testfun_)



class Ufuncs(TestCase):

    def setUp(self):
        self.emptyfun = Chebfun.initempty()
        self.yy = -1 + 2*rand(1000)

ufuncs = (
    arccos, arccosh, arcsin, arcsinh, arctan, arctanh, cos, cosh, exp, exp2,
    expm1, log, log2, log10, log1p, sinh, sin, tan, tanh, sqrt,
)

# empty-case tests
def ufuncEmptyCaseTester(ufunc):
    def tester(self):
        self.assertTrue(ufunc(self.emptyfun).isempty)
    return tester

for ufunc in ufuncs:
    _testfun_ = ufuncEmptyCaseTester(ufunc)
    _testfun_.__name__ = "test_emptycase_{}".format(ufunc.__name__)
    setattr(Ufuncs, _testfun_.__name__, _testfun_)

# TODO: Add more test cases
# add ufunc tests:
#     (ufunc, [([fun1, interval1], tol1), ([fun2, interval2], tol2), ... ])

uf1 = lambda x: x
uf1.__name__ = "x"

ufunc_test_params = [
        (arccos,  [([uf1, (-.8,.8)],  eps), ]),
        (arccosh, [([uf1, (2,3)    ], eps), ]),
        (arcsin,  [([uf1, (-.8,.8)],  eps), ]),
        (arcsinh, [([uf1, (2,3)    ], eps), ]),
        (arctan,  [([uf1, (-.8,.8)],  eps), ]),
        (arctanh, [([uf1, (-.8,.8)],  eps), ]),
        (cos,     [([uf1, (-3,3)   ], eps), ]),
        (cosh,    [([uf1, (-3,3)   ], eps), ]),
        (exp,     [([uf1, (-3,3)   ], eps), ]),
        (exp2,    [([uf1, (-3,3)   ], eps), ]),
        (expm1,   [([uf1, (-3,3)   ], eps), ]),
        (log,     [([uf1, (2,3)    ], eps), ]),
        (log2,    [([uf1, (2,3)    ], eps), ]),
        (log10,   [([uf1, (2,3)    ], eps), ]),
        (log1p,   [([uf1, (-.8,.8)],  eps), ]),
        (sinh,    [([uf1, (-3,3)   ], eps), ]),
        (sin,     [([uf1, (-3,3)   ], eps), ]),
        (tan,     [([uf1, (-.8,.8)],  eps), ]),
        (tanh,    [([uf1, (-3,3)   ], eps), ]),
        (sqrt,    [([uf1, (2,3)    ], eps), ]),
]


def ufuncTester(ufunc, f, interval, tol):
    a,b = interval.values
    ff = Chebfun.initfun_adaptive(f, linspace(a,b,13))
    gg = lambda x: ufunc(f(x))
    GG = ufunc(ff)
    def tester(self):
        xx = interval(self.yy)
        vscl = GG.vscale
        lscl = sum([fun.size for fun in GG])
        self.assertLessEqual(infnorm(gg(xx)-GG(xx)), vscl*lscl*tol)
    return tester

for (ufunc,  [([f, intvl], tol), ]) in ufunc_test_params:
    interval = Interval(*intvl)
    _testfun_ = ufuncTester(ufunc, f, interval, tol)
    _testfun_.__name__ = \
        "test_{}_{}_[{:.1f},..,{:.1f}]".format(
        ufunc.__name__, f.__name__, *intvl)
    setattr(Ufuncs, _testfun_.__name__, _testfun_)


class Evaluation(TestCase):

    def setUp(self):
        self.f0 = Chebfun.initempty()
        self.f1 = Chebfun.initfun_adaptive(lambda x: x**2, [-1,1])
        self.f2 = Chebfun.initfun_adaptive(lambda x: x**2, [-1,0,1,2])

    def test__call__empty_chebfun(self):
        self.assertEqual(self.f0(linspace(-1,1,100)).size, 0)

    def test__call__empty_array(self):
        self.assertEqual(self.f0(array([])).size, 0)
        self.assertEqual(self.f1(array([])).size, 0)
        self.assertEqual(self.f2(array([])).size, 0)

    def test__call__point_evaluation(self):
        # check we get back a scalar for scalar input
        self.assertTrue(isscalar(self.f1(0.1)))

    def test__call__singleton(self):
        # check that the output is the same for the following inputs:
        # array(x), array([x]), [x]
        a = self.f1(array(0.1))
        b = self.f1(array([0.1]))
        c = self.f1([0.1])
        self.assertEqual(a.size, 1)
        self.assertEqual(b.size, 1)
        self.assertEqual(c.size, 1)
        self.assertTrue(equal(a,b).all())
        self.assertTrue(equal(b,c).all())
        self.assertTrue(equal(a,c).all())

    def test__call__breakpoints(self):
        # check we get the values at the breakpoints back
        x1 = self.f1.breakpoints
        x2 = self.f2.breakpoints
        self.assertTrue(equal(self.f1(x1), [1,1]).all())
        self.assertTrue(equal(self.f2(x2), [1,0,1,4]).all())

    def test__call__outside_interval(self):
        # check we are able to evaluate the Chebfun outside the
        # interval of definition
        x = linspace(-3,3,100)
        self.assertTrue(isfinite(self.f1(x)).all())
        self.assertTrue(isfinite(self.f2(x)).all())

    def test__call__general_evaluation(self):
        f = lambda x: sin(4*x) + exp(cos(14*x)) - 1.4
        npts = 50000
        dom1 = [-1,1]
        dom2 = [-1,0,1]
        dom3 = [-2,-0.3,1.2]
        ff1 = Chebfun.initfun_adaptive(f, dom1)
        ff2 = Chebfun.initfun_adaptive(f, dom2)
        ff3 = Chebfun.initfun_adaptive(f, dom3)
        x1 = linspace(dom1[0], dom1[-1], npts)
        x2 = linspace(dom2[0], dom2[-1], npts)
        x3 = linspace(dom3[0], dom3[-1], npts)
        self.assertLessEqual(infnorm(f(x1)-ff1(x1)), 5e1*eps)
        self.assertLessEqual(infnorm(f(x2)-ff2(x2)), 2e1*eps)
        self.assertLessEqual(infnorm(f(x3)-ff3(x3)), 5e1*eps)


class Calculus(TestCase):

    def setUp(self):
        f = lambda x: sin(4*x-1.4)
        self.df = lambda x: 4*cos(4*x-1.4)
        self.If = lambda x: -.25*cos(4*x-1.4)
        self.f1 = Chebfun.initfun_adaptive(f, [-1,1])
        self.f2 = Chebfun.initfun_adaptive(f, [-3,0,1])
        self.f3 = Chebfun.initfun_adaptive(f, [-2,-0.3,1.2])
        self.f4 = Chebfun.initfun_adaptive(f, linspace(-1,1,11))

    def test_sum(self):
        self.assertLessEqual(abs(self.f1.sum()-0.372895407327895),2*eps)
        self.assertLessEqual(abs(self.f2.sum()-0.382270459230604),2*eps)
        self.assertLessEqual(abs(self.f3.sum()-(-0.008223712363936)),2*eps)
        self.assertLessEqual(abs(self.f4.sum()-0.372895407327895),2*eps)

    def test_diff(self):
        xx = linspace(-5,5,10000)
        for f in [self.f1, self.f2, self.f3, self.f4]:
            a, b = f.support
            x = xx[(xx>a)&(xx<b)]
            self.assertLessEqual(infnorm(f.diff()(x)-self.df(x)), 2e3*eps)

    def test_cumsum(self):
        xx = linspace(-5,5,10000)
        for f in [self.f1, self.f2, self.f3, self.f4]:
            a, b = f.support
            x = xx[(xx>a)&(xx<b)]
            fa = self.If(a)
            self.assertLessEqual(infnorm(f.cumsum()(x)-self.If(x)+fa), 3*eps)

    def test_sum_empty(self):
        f = Chebfun.initempty()
        self.assertEqual(f.sum(), .0)

    def test_cumsum_empty(self):
        If = Chebfun.initempty().cumsum()
        self.assertIsInstance(If, Chebfun)
        self.assertTrue(If.isempty)

    def test_diff_empty(self):
        df = Chebfun.initempty().diff()
        self.assertIsInstance(df, Chebfun)
        self.assertTrue(df.isempty)


class Roots(TestCase):

    def setUp(self):
        self.f1 = Chebfun.initfun_adaptive(lambda x: cos(4*pi*x), linspace(-10,10,101))        
        self.f2 = Chebfun.initfun_adaptive(lambda x: sin(2*pi*x), linspace(-1,1,5))
        self.f3 = Chebfun.initfun_adaptive(lambda x: sin(4*pi*x), linspace(-10,10,101))        

    def test_empty(self):
        rts = Chebfun.initempty().roots()
        self.assertIsInstance(rts, ndarray)
        self.assertEqual(rts.size, 0)

    def test_multiple_pieces(self):
        rts = self.f1.roots()
        self.assertEqual(rts.size, 80)
        self.assertLessEqual(infnorm(rts-arange(-9.875,10,.25)), 10*eps)

    # check we don't get repeated roots at breakpoints
    def test_breakpoint_roots_1(self):
        rts = self.f2.roots()
        self.assertEqual(rts.size, 5)
        self.assertLessEqual(infnorm(rts-self.f2.breakpoints), eps)

    # check we don't get repeated roots at breakpoints
    def test_breakpoint_roots_2(self):
        rts = self.f3.roots()
        self.assertEqual(rts.size, 81)
        self.assertLessEqual(infnorm(rts-arange(-10,10.25,.25)), 1e1*eps)



class Plotting(TestCase):

    def setUp(self):
        f = lambda x: sin(4*x) + exp(cos(14*x)) - 1.4
        self.f1 = Chebfun.initfun_adaptive(f, [-1,1])
        self.f2 = Chebfun.initfun_adaptive(f, [-3,0,1])
        self.f3 = Chebfun.initfun_adaptive(f, [-2,-0.3,1.2])

    def test_plot(self):
        for fun in [self.f1, self.f2, self.f3]:
            fig, ax = subplots()
            fun.plot(ax=ax)

    def test_plotcoeffs(self):
        for fun in [self.f1, self.f2, self.f3]:
            fig, ax = subplots()
            fun.plotcoeffs(ax=ax)


class PrivateMethods(TestCase):

    def setUp(self):
        f = lambda x: sin(x-.1)
        self.f1 = Chebfun.initfun_adaptive(f, [-2,0,3])
        self.f2 = Chebfun.initfun_adaptive(f, linspace(-2,3,5))

    # in the test_break_x methods, we check that (1) the newly computed domain
    # is what it should be, and (2) the new chebfun still provides an accurate
    # approximation
    def test__break_1(self):
        altdom = Domain([-2,-1,1,2,3])
        newdom = self.f1.domain.union(altdom)
        f1_new = self.f1._break(newdom)
        self.assertEqual(f1_new.domain, newdom)
        self.assertNotEqual(f1_new.domain, altdom)
        self.assertNotEqual(f1_new.domain, self.f1.domain)
        xx = linspace(-2,3,1000)
        error = infnorm(self.f1(xx)-f1_new(xx))
        self.assertLessEqual(error, 3*eps)

    def test__break_2(self):
        altdom = Domain([-2,3])
        newdom = self.f1.domain.union(altdom)
        f1_new = self.f1._break(newdom)
        self.assertEqual(f1_new.domain, newdom)
        self.assertNotEqual(f1_new.domain, altdom)
        xx = linspace(-2,3,1000)
        error = infnorm(self.f1(xx)-f1_new(xx))
        self.assertLessEqual(error, 3*eps)

    def test__break_3(self):
        altdom = Domain(linspace(-2,3,1000))
        newdom = self.f2.domain.union(altdom)
        f2_new = self.f2._break(newdom)
        self.assertEqual(f2_new.domain, newdom)
        self.assertNotEqual(f2_new.domain, altdom)
        self.assertNotEqual(f2_new.domain, self.f2.domain)
        xx = linspace(-2,3,1000)
        error = infnorm(self.f2(xx)-f2_new(xx))
        self.assertLessEqual(error, 3*eps)

class DomainBreakingOps(TestCase):
    pass

# domain, test_tolerance
domainBreakOp_args = [
    (lambda x: x, 0, [-1,1], eps),
    (sin, cos, [-1,1], eps),
#    ([-2,1], eps),
#    ([-1,2], eps),
#    ([-5,9], 35*eps),
]

# add tests for maximu, minimum
def domainBreakOpTester(domainBreakOp, f, g, dom, tol):
    a, b = dom
    xx = linspace(a,b,1001)
    ff = chebfun(f, dom)
    gg = chebfun(g, dom)
    # convert constant g to to callable
    if isinstance(g, (int, float)):
        ffgg = domainBreakOp(f(xx), g)
    else:
        ffgg = domainBreakOp(f(xx), g(xx))
    fg = getattr(ff, domainBreakOp.__name__)(gg)
    def tester(self):
        vscl = max([ff.vscale, gg.vscale])
        hscl = max([ff.hscale, gg.hscale])
        lscl = max([fun.size for fun in append(ff.funs, gg.funs)])
        self.assertLessEqual(infnorm(fg(xx)-ffgg), vscl*hscl*lscl*tol)
    return tester

for domainBreakOp in (maximum, ):
    for n, args in enumerate(domainBreakOp_args):
        ff, gg, dom, tol = args
        _testfun_ = domainBreakOpTester(domainBreakOp, ff, gg, dom, tol)
        _testfun_.__name__ = "test_{}_{}".format(domainBreakOp.__name__, n)
        setattr(DomainBreakingOps, _testfun_.__name__, _testfun_)

# reset the testsfun variable so it doesn't get picked up by nose
_testfun_ = None
