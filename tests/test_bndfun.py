# -*- coding: utf-8 -*-
"""
Unit-tests for pyfun/chebtech.py
"""
from __future__ import division

from itertools import combinations
from unittest import TestCase

from operator import __add__
from operator import __sub__
from operator import __pos__
from operator import __neg__
from operator import __mul__

from numpy import linspace
from numpy import array
from numpy import sin
from numpy import cos
from numpy import exp
from numpy import pi
from numpy.random import rand
from numpy.random import seed

from matplotlib.pyplot import subplots

from pyfun.settings import DefaultPrefs
from pyfun.chebtech import Chebtech2
from pyfun.algorithms import standard_chop

from pyfun.bndfun import Bndfun
from pyfun.utilities import Interval

from utilities import testfunctions
from utilities import infnorm

eps = DefaultPrefs.eps

seed(0)


# NOTE: since (Fun/ClassicFun/)Bndfun is not a user-facing class
# (although it is not abstract) we will test the interface in the way
# Chebfun will interact with it, which means working explcitly with
# Interval objects.

# Furthermore, since we have already tested the adaptive constructor
# in the Chebtech-level tests, we just use the adaptive constructor in
# these tests.

class ClassUsage(TestCase):
    """Unit-tests for miscelaneous Bndfun class usage"""

    def setUp(self):
        f = lambda x: sin(30*x)
        subdomain = Interval(-2,3)
        self.f = f
        self.ff = Bndfun.initfun_adaptive(f, subdomain)
        self.xx = subdomain(-1 + 2*rand(100))
        self.emptyfun = Bndfun(Chebtech2.initempty(), subdomain)
        self.constfun = Bndfun(Chebtech2.initconst(1.), subdomain)

    # tests for emptiness of Bndfun objects
    def test_isempty_True(self):
        self.assertTrue(self.emptyfun.isempty())
        self.assertFalse(not self.emptyfun.isempty())

    def test_isempty_False(self):
        self.assertFalse(self.constfun.isempty())
        self.assertTrue(not self.constfun.isempty())

    # tests for constantness of Bndfun objects
    def test_isconst_True(self):
        self.assertTrue(self.constfun.isconst())
        self.assertFalse(not self.constfun.isconst())

    def test_isconst_False(self):
        self.assertFalse(self.emptyfun.isconst())
        self.assertTrue(not self.emptyfun.isconst())

    # check the size() method is working properly
    def test_size(self):
        cfs = rand(10)
        subdomain = Interval()
        b0 = Bndfun(Chebtech2(array([])), subdomain)
        b1 = Bndfun(Chebtech2(array([1.])), subdomain)
        b2 = Bndfun(Chebtech2(cfs), subdomain)
        self.assertEquals(b0.size(), 0)
        self.assertEquals(b1.size(), 1)
        self.assertEquals(b2.size(), cfs.size)

    def test_endpoints(self):
        a, b = self.ff.endpoints()
        self.assertEqual(a, -2)
        self.assertEqual(b, 3)

    def test_endvalues(self):
        a, b = self.ff.endpoints()
        fa, fb = self.ff.endvalues()
        self.assertLessEqual(abs(fa-self.f(a)), 2e1*eps)
        self.assertLessEqual(abs(fb-self.f(b)), 2e1*eps)

    # test the different permutations of self(xx, ..)
    def test_call(self):
        self.ff(self.xx)

    def test_call_bary(self):
        self.ff(self.xx, "bary")
        self.ff(self.xx, how="bary")

    def test_call_clenshaw(self):
        self.ff(self.xx, "clenshaw")
        self.ff(self.xx, how="clenshaw")

    def test_call_bary_vs_clenshaw(self):
        b = self.ff(self.xx, "clenshaw")
        c = self.ff(self.xx, "bary")
        self.assertLessEqual(infnorm(b-c), 2e2*eps)

    def test_call_raises(self):
        self.assertRaises(ValueError, self.ff, self.xx, "notamethod")
        self.assertRaises(ValueError, self.ff, self.xx, how="notamethod")

    def test_vscale_empty(self):
        self.assertEquals(self.emptyfun.vscale(), 0.)

    def test_copy(self):
        ff = self.ff
        gg = self.ff.copy()
        self.assertEquals(ff, ff)
        self.assertEquals(gg, gg)
        self.assertNotEquals(ff, gg)
        self.assertEquals(infnorm(ff.coeffs()-gg.coeffs()), 0)

    # check that the restricted fun matches self on the subinterval
    def test_restrict(self):
        i1 = Interval(-1,1)
        gg = self.ff.restrict(i1)
        yy = -1 + 2*rand(1000)
        self.assertLessEqual(infnorm(self.ff(yy)-gg(yy)), 1e2*eps)

    # check that the restricted fun matches self on the subinterval
    def test_simplify(self):
        interval = Interval(-2,1)
        ff = Bndfun.initfun_fixedlen(self.f, interval, 1000)
        gg = ff.simplify()
        self.assertEqual(gg.size(), standard_chop(ff.onefun.coeffs()))
        self.assertEqual(infnorm(ff.coeffs()[:gg.size()]-gg.coeffs()), 0)
        self.assertEqual(ff.interval, gg.interval)
# --------------------------------------
#          vscale estimates
# --------------------------------------
vscales = [
    # (function, number of points, vscale)
    (lambda x: sin(4*pi*x),      [-2, 2],     1),
    (lambda x: cos(x),           [-10, 1],    1),
    (lambda x: cos(4*pi*x),      [-100, 100], 1),
    (lambda x: exp(cos(4*pi*x)), [-1,1],      exp(1)),
    (lambda x: cos(3244*x),      [-2,0],      1),
    (lambda x: exp(x),           [-1,2],      exp(2)),
    (lambda x: 1e10*exp(x),      [-1,1],      1e10*exp(1)),
    (lambda x: 0*x+1.,           [-1e5,1e4],  1),
]

def definiteIntegralTester(fun, interval, vscale):
    subdomain = Interval(*interval)
    ff = Bndfun.initfun_adaptive(fun, subdomain)
    def tester(self):
        absdiff = abs(ff.vscale()-vscale)
        self.assertLessEqual(absdiff, .1*vscale)
    return tester

for k, args in enumerate(vscales):
    _testfun_ = definiteIntegralTester(*args)
    _testfun_.__name__ = "test_vscale_{:02}".format(k)
    setattr(ClassUsage, _testfun_.__name__, _testfun_)


class Plotting(TestCase):
    """Unit-tests for Bndfun plotting methods"""

    def setUp(self):
        f = lambda x: sin(1*x) + 5e-1*cos(10*x) + 5e-3*sin(100*x)
        subdomain = Interval(-6, 10)
        self.f0 = Bndfun.initfun_fixedlen(f, subdomain, 1000)
        self.f1 = Bndfun.initfun_adaptive(f, subdomain)

    def test_plot(self):
        fig, ax = subplots()
        self.f0.plot(ax=ax, color="g", marker="o", markersize=2, linestyle="")

    def test_plotcoeffs(self):
        fig, ax = subplots()
        self.f0.plotcoeffs(ax=ax)
        self.f1.plotcoeffs(ax=ax, color="r")



class Calculus(TestCase):
    """Unit-tests for Bndfun calculus operations"""

    def setUp(self):
        self.emptyfun = Bndfun(Chebtech2.initempty(), Interval())
        self.yy = -1 + 2*rand(2000)
#        self.constfun = Bndfun(Chebtech2.initconst(1.), subdomain)

    # tests for the correct results in the empty cases
    def test_sum_empty(self):
        self.assertEqual(self.emptyfun.sum(), 0)

    def test_cumsum_empty(self):
        self.assertTrue(self.emptyfun.cumsum().isempty())

    def test_diff_empty(self):
        self.assertTrue(self.emptyfun.diff().isempty())

# --------------------------------------
#           definite integrals
# --------------------------------------
def_integrals = [
    # (function, interval, integral, tolerance)
    (lambda x: sin(x),           [-2,2],                        .0,    2*eps),
    (lambda x: sin(4*pi*x),      [-.1, .7],      0.088970317927147,  1e1*eps),
    (lambda x: cos(x),           [-100,203],     0.426944059057085,  4e2*eps),
    (lambda x: cos(4*pi*x),      [-1e-1,-1e-3],  0.074682699182803,    2*eps),
    (lambda x: exp(cos(4*pi*x)), [-3,1],         5.064263511008033,    4*eps),
    (lambda x: cos(3244*x),      [0,0.4],   -3.758628487169980e-05,  5e2*eps),
    (lambda x: exp(x),           [-2,-1],          exp(-1)-exp(-2),    2*eps),
    (lambda x: 1e10*exp(x),      [-1,2],     1e10*(exp(2)-exp(-1)), 2e10*eps),
    (lambda x: 0*x+1.,           [-100,300],                   400,      eps),
]

def definiteIntegralTester(fun, interval, integral, tol):
    subdomain = Interval(*interval)
    ff = Bndfun.initfun_adaptive(fun, subdomain)
    def tester(self):
        absdiff = abs(ff.sum()-integral)
        self.assertLessEqual(absdiff, tol)
    return tester

for k, (fun, n, integral, tol) in enumerate(def_integrals):
    _testfun_ = definiteIntegralTester(fun, n, integral, tol)
    _testfun_.__name__ = "test_sum_{:02}".format(k)
    setattr(Calculus, _testfun_.__name__, _testfun_)

# --------------------------------------
#          indefinite integrals
# --------------------------------------
indef_integrals = [
    # (function, indefinite integral, interval, tolerance)
    (lambda x: 0*x+1.,      lambda x: x,             [-2,3],         eps),
    (lambda x: x,           lambda x: 1/2*x**2,      [-5,0],       4*eps),
    (lambda x: x**2,        lambda x: 1/3*x**3,      [1,10],     2e2*eps),
    (lambda x: x**3,        lambda x: 1/4*x**4,      [-1e-2,4e-1], 2*eps),
    (lambda x: x**4,        lambda x: 1/5*x**5,      [-3,-2],    3e2*eps),
    (lambda x: x**5,        lambda x: 1/6*x**6,      [-1e-10,1],   4*eps),
    (lambda x: sin(x),      lambda x: -cos(x),       [-10,22],   3e1*eps),
    (lambda x: cos(3*x),    lambda x: 1./3*sin(3*x), [-3,4],       2*eps),
    (lambda x: exp(x),      lambda x: exp(x),        [-60,1],    1e1*eps),
    (lambda x: 1e10*exp(x), lambda x: 1e10*exp(x),   [-1,1], 1e10*(3*eps)),
]

def indefiniteIntegralTester(fun, ifn, interval, tol):
    subdomain = Interval(*interval)
    ff = Bndfun.initfun_adaptive(fun, subdomain)
    gg = Bndfun.initfun_fixedlen(ifn, subdomain, ff.size()+1)
    coeffs = gg.coeffs()
    coeffs[0] = coeffs[0] - ifn(array([interval[0]]))
    def tester(self):
        absdiff = infnorm(ff.cumsum().coeffs() - coeffs)
        self.assertLessEqual(absdiff, tol)
    return tester

for k, (fun, dfn, n, tol) in enumerate(indef_integrals):
    _testfun_ = indefiniteIntegralTester(fun, dfn, n, tol)
    _testfun_.__name__ = "test_cumsum_{:02}".format(k)
    setattr(Calculus, _testfun_.__name__, _testfun_)

# --------------------------------------
#            derivatives
# --------------------------------------
derivatives = [
#     (function, derivative, number of points, tolerance)
    (lambda x: 0*x+1.,      lambda x: 0*x+0,       [-2,3],           eps),
    (lambda x: x,           lambda x: 0*x+1,       [-5,0],       2e1*eps),
    (lambda x: x**2,        lambda x: 2*x,         [1,10],       2e2*eps),
    (lambda x: x**3,        lambda x: 3*x**2,      [-1e-2,4e-1],   3*eps),
    (lambda x: x**4,        lambda x: 4*x**3,      [-3,-2],      1e3*eps),
    (lambda x: x**5,        lambda x: 5*x**4,      [-1e-10,1],   4e1*eps),
    (lambda x: sin(x),      lambda x: cos(x),      [-10,22],     5e2*eps),
    (lambda x: cos(3*x),    lambda x: -3*sin(3*x), [-3,4],       5e2*eps),
    (lambda x: exp(x),      lambda x: exp(x),      [-60,1],      2e2*eps),
    (lambda x: 1e10*exp(x), lambda x: 1e10*exp(x), [-1,1],  1e10*2e2*eps),
]

def derivativeTester(fun, ifn, interval, tol):
    subdomain = Interval(*interval)
    ff = Bndfun.initfun_adaptive(fun, subdomain)
    gg = Bndfun.initfun_fixedlen(ifn, subdomain, max(ff.size()-1,1))
    def tester(self):
        absdiff = infnorm(ff.diff().coeffs() - gg.coeffs())
        self.assertLessEqual(absdiff, tol)
    return tester

for k, (fun, der, n, tol) in enumerate(derivatives):
    _testfun_ = derivativeTester(fun, der, n, tol)
    _testfun_.__name__ = "test_diff_{:02}".format(k)
    setattr(Calculus, _testfun_.__name__, _testfun_)


class Construction(TestCase):
    """Unit-tests for construction of Bndfun objects"""

    def test_onefun_construction(self):
        coeffs = rand(10)
        subdomain = Interval()
        onefun = Chebtech2(coeffs)
        f = Bndfun(onefun, subdomain)
        self.assertIsInstance(f, Bndfun)
        self.assertLess(infnorm(f.coeffs()-coeffs), eps)

    def test_const_construction(self):
        subdomain = Interval()
        ff = Bndfun.initconst(1., subdomain)
        self.assertEquals(ff.size(), 1)
        self.assertTrue(ff.isconst())
        self.assertFalse(ff.isempty())
        self.assertRaises(ValueError, Bndfun.initconst, [1.], subdomain)

    def test_empty_construction(self):
        ff = Bndfun.initempty()
        self.assertEquals(ff.size(), 0)
        self.assertFalse(ff.isconst())
        self.assertTrue(ff.isempty())
        self.assertRaises(TypeError, Bndfun.initempty, [1.])


def adaptiveTester(fun, subdomain, funlen):
    ff = Bndfun.initfun_adaptive(fun, subdomain)
    def tester(self):
        self.assertEquals(ff.size(), funlen)
    return tester

def fixedlenTester(fun, subdomain, n):
    ff = Bndfun.initfun_fixedlen(fun, subdomain, n)
    def tester(self):
        self.assertEquals(ff.size(), n)
    return tester

funs = []
fun_details = [
    # (function, name for the test printouts,
    #  Matlab chebfun adaptive degree on [-2,3])
    (lambda x: x**3 + x**2 + x + 1, "poly3(x)", [-2,3],  4),
    (lambda x: exp(x),              "exp(x)",   [-2,3], 20),
    (lambda x: sin(x),              "sin(x)",   [-2,3], 20),
    (lambda x: cos(20*x),           "cos(20x)", [-2,3], 90),
    (lambda x: 0.*x+1.,             "constfun", [-2,3],  1),
    (lambda x: 0.*x,                "zerofun",  [-2,3],  1),
]

for k, (fun, name, interval, funlen) in enumerate(fun_details):

    fun.__name__ = name
    subdomain = Interval(*interval)

    # add the adaptive tests
    _testfun_ = adaptiveTester(fun, subdomain, funlen)
    _testfun_.__name__ = "test_adaptive_{}".format(fun.__name__)
    setattr(Construction, _testfun_.__name__, _testfun_)

    # add the fixedlen tests
    for n in array([100]):
        _testfun_ = fixedlenTester(fun, subdomain, n)
        _testfun_.__name__ = \
            "test_fixedlen_{}_{:003}pts".format(fun.__name__, n)
        setattr(Construction, _testfun_.__name__, _testfun_)


class Algebra(TestCase):
    """Unit-tests for Bndfun algebraic operations"""
    def setUp(self):
        self.yy = -1 + 2 * rand(1000)
        self.emptyfun = Bndfun.initempty()

    # check (empty Bndfun) + (Bndfun) = (empty Bndfun)
    #   and (Bndfun) + (empty Bndfun) = (empty Bndfun)
    def test__add__radd__empty(self):
        subdomain = Interval(-2,3)
        for (fun, funlen) in testfunctions:
            chebtech = Bndfun.initfun_fixedlen(fun, subdomain, funlen)
            self.assertTrue((self.emptyfun+chebtech).isempty())
            self.assertTrue((chebtech+self.emptyfun).isempty())

    # check the output of (constant + Bndfun)
    #                 and (Bndfun + constant)
    def test__add__radd__constant(self):
        subdomain = Interval(-.5,.9)
        xx = subdomain(self.yy)
        for (fun, funlen) in testfunctions:
            for const in (-1, 1, 10, -1e5):
                f = lambda x: const + fun(x)
                boundedfun = Bndfun.initfun_fixedlen(fun, subdomain, funlen)
                f1 = const + boundedfun
                f2 = boundedfun + const
                tol = 3e1 * eps * abs(const)
                self.assertLessEqual( infnorm(f(xx)-f1(xx)), tol )
                self.assertLessEqual( infnorm(f(xx)-f2(xx)), tol )

    # check (empty Bndfun) - (Bndfun) = (empty Bndfun)
    #   and (Bndfun) - (empty Bndfun) = (empty Bndfun)
    def test__sub__rsub__empty(self):
        subdomain = Interval(-2,3)
        for (fun, funlen) in testfunctions:
            chebtech = Bndfun.initfun_fixedlen(fun, subdomain, funlen)
            self.assertTrue((self.emptyfun-chebtech).isempty())
            self.assertTrue((chebtech-self.emptyfun).isempty())

    # check the output of constant - Bndfun
    #                 and Bndfun - constant
    def test__sub__rsub__constant(self):
        subdomain = Interval(-.5,.9)
        xx = subdomain(self.yy)
        for (fun, funlen) in testfunctions:
            for const in (-1, 1, 10, -1e5):
                boundedfun = Bndfun.initfun_fixedlen(fun, subdomain, funlen)
                f = lambda x: const - fun(x)
                g = lambda x: fun(x) - const
                ff = const - boundedfun
                gg = boundedfun - const
                tol = 5e1 * eps * abs(const)
                self.assertLessEqual( infnorm(f(xx)-ff(xx)), tol )
                self.assertLessEqual( infnorm(g(xx)-gg(xx)), tol )

    # check (empty Bndfun) * (Bndfun) = (empty Bndfun)
    #   and (Bndfun) * (empty Bndfun) = (empty Bndfun)
    def test__mul__rmul__empty(self):
        subdomain = Interval(-2,3)
        for (fun, funlen) in testfunctions:
            chebtech = Bndfun.initfun_fixedlen(fun, subdomain, funlen)
            self.assertTrue((self.emptyfun*chebtech).isempty())
            self.assertTrue((chebtech*self.emptyfun).isempty())

    # check the output of constant * Bndfun
    #                 and Bndfun * constant
    def test__rmul__constant(self):
        subdomain = Interval(-.5,.9)
        xx = subdomain(self.yy)
        for (fun, funlen) in testfunctions:
            for const in (-1, 1, 10, -1e5):
                boundedfun = Bndfun.initfun_fixedlen(fun, subdomain, funlen)
                f = lambda x: const * fun(x)
                g = lambda x: fun(x) * const
                ff = const * boundedfun
                gg = boundedfun * const
                tol = 4e1 * eps * abs(const)
                self.assertLessEqual( infnorm(f(xx)-ff(xx)), tol )
                self.assertLessEqual( infnorm(g(xx)-gg(xx)), tol )

    # check    +(empty Bndfun) = (empty Bndfun)
    def test__pos__empty(self):
        self.assertTrue( (+self.emptyfun).isempty() )

    # check -(empty Bndfun) = (empty Bndfun)
    def test__neg__empty(self):
        self.assertTrue( (-self.emptyfun).isempty() )


binops = (
    __add__,
    __sub__,
    __mul__,
    )

# add tests for the binary operators
def binaryOpTester(f, g, subdomain, binop):
    ff = Bndfun.initfun_adaptive(f, subdomain)
    gg = Bndfun.initfun_adaptive(g, subdomain)
    FG = Bndfun.initfun_adaptive(lambda x: binop(f(x),g(x)), subdomain)
    fg = binop(ff, gg)
    def tester(self):
        xx = subdomain(self.yy)
        self.assertLessEqual( infnorm(fg(xx)-FG(xx)), 2e2*eps)
    return tester

# Note: defining __radd__(a,b) = __add__(b,a) and feeding this into the
# test will not in fact test the __radd__ functionality of the class.
# These test will need to be added manually.

for binop in binops:
    # add the generic binary operator tests
    for (f, _), (g, _) in combinations(testfunctions, 2):
        subdomain = Interval(-.5,.9)
        _testfun_ = binaryOpTester(f, g, subdomain, binop)
        _testfun_.__name__ = \
            "test{}{}_{}".format(binop.__name__, f.__name__,  g.__name__)
        setattr(Algebra, _testfun_.__name__, _testfun_)

unaryops = (
    __pos__,
    __neg__,
    )

# add tests for the unary operators
def unaryOpTester(unaryop, f, subdomain):
    ff = Bndfun.initfun_adaptive(f, subdomain)
    gg = Bndfun.initfun_adaptive(lambda x: unaryop(f(x)), subdomain)
    GG = unaryop(ff)
    def tester(self):
        xx = subdomain(self.yy)
        self.assertLessEqual( infnorm(gg(xx)-GG(xx)), 2e2*eps)
    return tester

for unaryop in unaryops:
    for (f, _) in testfunctions:
        subdomain = Interval(-.5,.9)
        _testfun_ = unaryOpTester(unaryop, f, subdomain)
        _testfun_.__name__ = \
            "test{}{}".format(unaryop.__name__, f.__name__)
        setattr(Algebra, _testfun_.__name__, _testfun_)

class Roots(TestCase):

    def test_empty(self):
        ff = Bndfun.initempty()
        self.assertEquals(ff.roots().size, 0)

    def test_const(self):
        ff = Bndfun.initconst(0., Interval(-2,3))
        gg = Bndfun.initconst(2., Interval(-2,3))
        self.assertEquals(ff.roots().size, 0)
        self.assertEquals(gg.roots().size, 0)

# add tests for roots
def rootsTester(f, interval, roots, tol):
    subdomain = Interval(*interval)
    ff = Bndfun.initfun_adaptive(f, subdomain)
    rts = ff.roots()
    def tester(self):
        self.assertLessEqual( infnorm(rts-roots), tol)
    return tester

rootstestfuns = (
    (lambda x: 3*x+2.,        [-2,3],     array([-2/3]),                  eps),
    (lambda x: x**2+.2*x-.08, [-2,5],     array([-.4, .2]),           3e1*eps),
    (lambda x: sin(x),        [-7,7],     pi*linspace(-2,2,5),        1e1*eps),
    (lambda x: cos(2*pi*x),   [-20,10],   linspace(-19.75, 9.75, 60), 3e1*eps),
    (lambda x: sin(100*pi*x), [-0.5,0.5], linspace(-.5,.5,101),           eps),
    (lambda x: sin(5*pi/2*x), [-1,1],     array([-.8, -.4, 0, .4, .8]),   eps)
    )
for k, args in enumerate(rootstestfuns):
    _testfun_ = rootsTester(*args)
    _testfun_.__name__ = "test_roots_{}".format(k)
    setattr(Roots, _testfun_.__name__, _testfun_)

# reset the testsfun variable so it doesn't get picked up by nose
_testfun_ = None
