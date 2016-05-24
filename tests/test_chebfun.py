# -*- coding: utf-8 -*-
"""
Unit-tests for pyfun/core/chebfun.py
"""
from __future__ import division

from operator import __add__
from operator import __pos__
from operator import __neg__

from itertools import combinations
from unittest import TestCase
from unittest import skip

from numpy import ndarray
from numpy import array
from numpy import arange
from numpy import exp
from numpy import sin
from numpy import log
from numpy import cos
from numpy import sum
from numpy import abs
from numpy import pi
from numpy import linspace
from numpy import equal
from numpy import isscalar
from numpy import isfinite

from matplotlib.pyplot import subplots

from chebpy.core.bndfun import Bndfun
from chebpy.core.settings import DefaultPrefs
from chebpy.core.utilities import Domain
from chebpy.core.utilities import Interval
from chebpy.core.chebfun import Chebfun
from chebpy.core.exceptions import IntervalGap
from chebpy.core.exceptions import IntervalOverlap
from chebpy.core.exceptions import BadDomainArgument
from chebpy.core.exceptions import BadFunLengthArgument
from chebpy.core.exceptions import DomainBreakpoints

#from utilities import testfunctions
from utilities import infnorm
#from utilities import scaled_tol
#from utilities import infNormLessThanTol

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

    def test__init__empty(self):
        emptyfun = Chebfun.initempty()
        self.assertEqual(emptyfun.funs.size, 0)

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
        ff = Chebfun.initfun_fixedlen(self.f, [-2,-1], 20)
        self.assertEqual(ff.funs.size, 1)
        a, b = ff.breakdata.keys()
        fa, fb, = ff.breakdata.values()
        self.assertEqual(a,-2)
        self.assertEqual(b,-1)
        self.assertLessEqual(abs(fa-self.f(-2)), eps)
        self.assertLessEqual(abs(fb-self.f(-1)), eps)

    def test_initfun_fixedlen_piecewise_domain_0(self):
        ff = Chebfun.initfun_fixedlen(self.f, [-2,0,1], 30)
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
        ff = Chebfun.initfun_fixedlen(self.f, [-2,0,1], [30,20])
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
        self.assertRaises(BadDomainArgument, initfun, self.f, [-2], 10)
        self.assertRaises(BadDomainArgument, initfun, self.f, domain=[-2], n=10)
        self.assertRaises(BadDomainArgument, initfun, self.f, domain=None, n=10)
        self.assertRaises(BadFunLengthArgument, initfun, self.f, [-1,1], [30,40])
        self.assertRaises(TypeError, initfun, self.f, [-1,1], None)

    def test_initfun_fixedlen_succeeds(self):
        self.assertTrue(Chebfun.initfun_fixedlen(self.f, [-2,-1,0], []).isempty)
        # check that providing a vector with None elements calls the
        # Tech adaptive constructor
        g0 = Chebfun.initfun_adaptive(self.f, [-2,-1,0])
        g1 = Chebfun.initfun_fixedlen(self.f, [-2,-1,0], [None,None])
        g2 = Chebfun.initfun_fixedlen(self.f, [-2,-1,0], [None,40])
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


class Algebra(TestCase):

    def setUp(self):
        self.emptyfun = Chebfun.initempty()

    # check  +(empty Chebfun) = (empty Chebfun)
    def test__pos__empty(self):
        self.assertTrue((+self.emptyfun).isempty)

    # check -(empty Chebfun) = (empty Chebfun)
    def test__neg__empty(self):
        self.assertTrue((-self.emptyfun).isempty)

    # check (empty Chebfun) + (Chebfun) = (empty Chebfun)
    #   and (Chebfun) + (empty Chebfun) = (empty Chebfun)
    def test__add__radd__empty(self):
        for f in chebfun_testfunctions:
            for dom, _ in chebfun_testdomains:
                a, b = dom
                ff = Chebfun.initfun_adaptive(f, linspace(a,b,13))
                self.assertTrue((self.emptyfun+ff).isempty)
                self.assertTrue((ff+self.emptyfun).isempty)

# fun, periodic break conditions
testfuns = [
    (lambda x: sin(5*x-1), "sin(5*x-1)"),
    (lambda x: cos(2*pi*x), "cos(2*pi*x)"),
    (lambda x: log(x+10), "log(x+10)"),
]

chebfun_testfunctions = []
for (f, name) in testfuns:
    f.__name__ = name
    chebfun_testfunctions.append(f)

# domain, test_tolerance
chebfun_testdomains = [
    ([-1,1], 8e0*eps),
    ([-2,1], 2e1*eps),
    ([-1,2], 2e1*eps),
    ([-5,9], 6e1*eps),
]


# add tests for the binary operators
def binaryOpTester(f, g, binop, dom, tol):
    a, b = dom
    xx = linspace(a,b,1001)
    n, m = 3, 8
    ff = Chebfun.initfun_adaptive(f, linspace(a,b,n+1))
    gg = Chebfun.initfun_adaptive(g, linspace(a,b,m+1))
    FG = lambda x: binop(f(x),g(x))
    fg = binop(ff, gg)
    def tester(self):
        self.assertEqual(ff.funs.size, n)
        self.assertEqual(gg.funs.size, m)
        self.assertEqual(fg.funs.size, n+m-1)
        self.assertLessEqual(infnorm(fg(xx)-FG(xx)), tol)
    return tester


binops = (
    __add__,
#    __sub__,
#    __mul__,
    )

for binop in binops:
    for f, g in combinations(chebfun_testfunctions, 2):
        for dom, tol in chebfun_testdomains:
            _testfun_ = binaryOpTester(f, g, binop, dom, tol)
            _testfun_.__name__ = \
                "test{}{}_{}_[{:.0f},..,{:.0f}]".format(
                    binop.__name__, f.__name__,  g.__name__, dom[0], dom[1])
            setattr(Algebra, _testfun_.__name__, _testfun_)


# add tests for the unary operators
def unaryOpTester(f, unaryop, dom, tol):
    a, b = dom
    xx = linspace(a,b,1001)
    ff = Chebfun.initfun_adaptive(f, linspace(a,b,9))
    GG = lambda x: unaryop(f(x))
    gg = unaryop(ff)
    def tester(self):
        self.assertEqual(ff.funs.size, ff.funs.size)
        self.assertLessEqual(infnorm(gg(xx)-GG(xx)), tol)
    return tester

unaryops = (
    __pos__,
    __neg__,
    )

for unaryop in unaryops:
    for f in chebfun_testfunctions:
        for dom, tol in chebfun_testdomains:
            _testfun_ = unaryOpTester(f, unaryop, dom, tol)
            _testfun_.__name__ = \
                "test{}{}_[{:.0f},..,{:.0f}]".format(
                    unaryop.__name__, f.__name__, dom[0], dom[1])
            setattr(Algebra, _testfun_.__name__, _testfun_)

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
        self.assertLessEqual(infnorm(f(x1)-ff1(x1)), 4e1*eps)
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
        self.assertLessEqual(abs(self.f3.sum()--0.008223712363936),2*eps)
        self.assertLessEqual(abs(self.f4.sum()-0.372895407327895),2*eps)

    def test_diff(self):
        xx = linspace(-5,5,10000)
        for f in [self.f1, self.f2, self.f3, self.f4]:
            a, b = f.support
            x = xx[(xx>a)&(xx<b)]
            self.assertLessEqual(infnorm(f.diff()(x)-self.df(x)), 1e3*eps)

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
        f1_new = self.f1._Chebfun__break(newdom)
        self.assertEqual(f1_new.domain, newdom)
        self.assertNotEqual(f1_new.domain, altdom)
        self.assertNotEqual(f1_new.domain, self.f1.domain)
        xx = linspace(-2,3,1000)
        error = infnorm(self.f1(xx)-f1_new(xx))
        self.assertLessEqual(error, 3*eps)

    def test__break_2(self):
        altdom = Domain([-2,3])
        newdom = self.f1.domain.union(altdom)
        f1_new = self.f1._Chebfun__break(newdom)
        self.assertEqual(f1_new.domain, newdom)
        self.assertNotEqual(f1_new.domain, altdom)
        xx = linspace(-2,3,1000)
        error = infnorm(self.f1(xx)-f1_new(xx))
        self.assertLessEqual(error, 3*eps)

    def test__break_3(self):
        altdom = Domain(linspace(-2,3,1000))
        newdom = self.f2.domain.union(altdom)
        f2_new = self.f2._Chebfun__break(newdom)
        self.assertEqual(f2_new.domain, newdom)
        self.assertNotEqual(f2_new.domain, altdom)
        self.assertNotEqual(f2_new.domain, self.f2.domain)
        xx = linspace(-2,3,1000)
        error = infnorm(self.f2(xx)-f2_new(xx))
        self.assertLessEqual(error, 3*eps)

    def test__break_raises(self):
        dom1 = Domain([-1,1])
        dom2 = Domain(self.f2.domain.breakpoints[:-1])
        self.assertRaises(DomainBreakpoints, self.f1._Chebfun__break, dom1)
        self.assertRaises(DomainBreakpoints, self.f2._Chebfun__break, dom2)

# ------------------------------------------------------------------------
# Tests to verify the mutually inverse nature of vals2coeffs and coeffs2vals
# ------------------------------------------------------------------------
#def vals2coeffs2valsTester(n):
#    def asserter(self):
#        values = rand(n)
#        coeffs = _vals2coeffs(values)
#        _values_ = _coeffs2vals(coeffs)
#        self.assertLessEqual( infnorm(values-_values_), scaled_tol(n) )
#    return asserter
#
#def coeffs2vals2coeffsTester(n):
#    def asserter(self):
#        coeffs = rand(n)
#        values = _coeffs2vals(coeffs)
#        _coeffs_ = _vals2coeffs(values)
#        self.assertLessEqual( infnorm(coeffs-_coeffs_), scaled_tol(n) )
#    return asserter
#
#for k, n in enumerate(2**arange(2,18,2)+1):
#
#    # vals2coeffs2vals
#    _testfun_ = vals2coeffs2valsTester(n)
#    _testfun_.__name__ = "test_vals2coeffs2vals_{:02}".format(k)
#    setattr(ChebyshevPoints, _testfun_.__name__, _testfun_)
#
#    # coeffs2vals2coeffs
#    _testfun_ = coeffs2vals2coeffsTester(n)
#    _testfun_.__name__ = "test_coeffs2vals2coeffs_{:02}".format(k)
#    setattr(ChebyshevPoints, _testfun_.__name__, _testfun_)
# ------------------------------------------------------------------------

# reset the testsfun variable so it doesn't get picked up by nose
_testfun_ = None
