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
from pyfun.chebtech import ChebTech2

from pyfun.bndfun import BndFun
from pyfun.utilities import Domain

from utilities import testfunctions
from utilities import infnorm

eps = DefaultPrefs.eps

seed(0)


# NOTE: since (Fun/ClassicFun/)BndFun is not a user-facing class
# (although it is not abstract) we will test the interface in the way
# Chebfun will interact with it, which means working explcitly with
# Domain objects.

# Furthermore, since we have already tested the adaptive constructor
# in the ChebTech-level tests, we jsut use the adaptive constructor in
# these tests.

class ClassUsage(TestCase):
    """Unit-tests for miscelaneous BndFun class usage"""

    def setUp(self):
        f = lambda x: sin(30*x)
        domain = Domain(-2,3)
        self.ff = BndFun.initfun_adaptive(f, domain)
        self.xx = domain(-1 + 2*rand(100))

        self.emptyfun = BndFun(ChebTech2.initempty(), domain)
        self.constfun = BndFun(ChebTech2.initconst(1.), domain)

    # tests for emptiness of BndFun objects
    def test_isempty_True(self):
        self.assertTrue(self.emptyfun.isempty())
        self.assertFalse(not self.emptyfun.isempty())

    def test_isempty_False(self):
        self.assertFalse(self.constfun.isempty())
        self.assertTrue(not self.constfun.isempty())

    # tests for constantness of BndFun objects
    def test_isconst_True(self):
        self.assertTrue(self.constfun.isconst())
        self.assertFalse(not self.constfun.isconst())

    def test_isconst_False(self):
        self.assertFalse(self.emptyfun.isconst())
        self.assertTrue(not self.emptyfun.isconst())

    # check the size() method is working properly
    def test_size(self):
        cfs = rand(10)
        domain = Domain()
        b0 = BndFun(ChebTech2(array([])), domain)
        b1 = BndFun(ChebTech2(array([1.])), domain)
        b2 = BndFun(ChebTech2(cfs), domain)
        self.assertEquals(b0.size(), 0)
        self.assertEquals(b1.size(), 1)
        self.assertEquals(b2.size(), cfs.size)

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
        self.assertEquals( infnorm(ff.coeffs() - gg.coeffs()) , 0)

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
    domain = Domain(*interval)
    ff = BndFun.initfun_adaptive(fun, domain)
    def tester(self):
        absdiff = abs(ff.vscale()-vscale)
        self.assertLessEqual(absdiff, .1*vscale)
    return tester

for k, args in enumerate(vscales):
    _testfun_ = definiteIntegralTester(*args)
    _testfun_.__name__ = "test_vscale_{:02}".format(k)
    setattr(ClassUsage, _testfun_.__name__, _testfun_)


class Plotting(TestCase):
    """Unit-tests for BndFun plotting methods"""

    def setUp(self):
        f = lambda x: sin(1*x) + 5e-1*cos(10*x) + 5e-3*sin(100*x)
        domain = Domain(-6, 10)
        self.f0 = BndFun.initfun_fixedlen(f, domain, 1000)
        self.f1 = BndFun.initfun_adaptive(f, domain)

    def test_plot(self):
        fig, ax = subplots()
        self.f0.plot(ax=ax, color="g", marker="o", markersize=2, linestyle="")

    def test_plotcoeffs(self):
        fig, ax = subplots()
        self.f0.plotcoeffs(ax=ax)
        self.f1.plotcoeffs(ax=ax, color="r")



class Calculus(TestCase):
    """Unit-tests for BndFun calculus operations"""

    def setUp(self):
        self.emptyfun = BndFun(ChebTech2.initempty(), Domain())
        self.yy = -1 + 2*rand(2000)
#        self.constfun = BndFun(ChebTech2.initconst(1.), domain)

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
    domain = Domain(*interval)
    ff = BndFun.initfun_adaptive(fun, domain)
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
    domain = Domain(*interval)
    ff = BndFun.initfun_adaptive(fun, domain)
    gg = BndFun.initfun_fixedlen(ifn, domain, ff.size()+1)
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
    domain = Domain(*interval)
    ff = BndFun.initfun_adaptive(fun, domain)
    gg = BndFun.initfun_fixedlen(ifn, domain, max(ff.size()-1,1))
    def tester(self):
        absdiff = infnorm(ff.diff().coeffs() - gg.coeffs())
        self.assertLessEqual(absdiff, tol)
    return tester

for k, (fun, der, n, tol) in enumerate(derivatives):
    _testfun_ = derivativeTester(fun, der, n, tol)
    _testfun_.__name__ = "test_diff_{:02}".format(k)
    setattr(Calculus, _testfun_.__name__, _testfun_)


class Construction(TestCase):
    """Unit-tests for construction of BndFun objects"""

    def test_onefun_construction(self):
        coeffs = rand(10)
        domain = Domain()
        onefun = ChebTech2(coeffs)
        f = BndFun(onefun, domain)
        self.assertIsInstance(f, BndFun)
        self.assertLess(infnorm(f.coeffs()-coeffs), eps)

    def test_const_construction(self):
        domain = Domain()
        ff = BndFun.initconst(1., domain)
        self.assertEquals(ff.size(), 1)
        self.assertTrue(ff.isconst())
        self.assertFalse(ff.isempty())
        self.assertRaises(ValueError, BndFun.initconst, [1.], domain)

    def test_empty_construction(self):
        ff = BndFun.initempty()
        self.assertEquals(ff.size(), 0)
        self.assertFalse(ff.isconst())
        self.assertTrue(ff.isempty())
        self.assertRaises(TypeError, BndFun.initempty, [1.])


def adaptiveTester(fun, domain, funlen):
    ff = BndFun.initfun_adaptive(fun, domain)
    def tester(self):
        self.assertEquals(ff.size(), funlen)
    return tester

def fixedlenTester(fun, domain, n):
    ff = BndFun.initfun_fixedlen(fun, domain, n)
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
    domain = Domain(*interval)

    # add the adaptive tests
    _testfun_ = adaptiveTester(fun, domain, funlen)
    _testfun_.__name__ = "test_adaptive_{}".format(fun.__name__)
    setattr(Construction, _testfun_.__name__, _testfun_)

    # add the fixedlen tests
    for n in array([100]):
        _testfun_ = fixedlenTester(fun, domain, n)
        _testfun_.__name__ = \
            "test_fixedlen_{}_{:003}pts".format(fun.__name__, n)
        setattr(Construction, _testfun_.__name__, _testfun_)


class Algebra(TestCase):
    """Unit-tests for BndFun algebraic operations"""
    def setUp(self):
        self.yy = -1 + 2 * rand(1000)
        self.emptyfun = BndFun.initempty()

    # check (empty BndFun) + (BndFun) = (empty BndFun)
    #   and (BndFun) + (empty BndFun) = (empty BndFun)
    def test__add__radd__empty(self):
        domain = Domain(-2,3)
        for (fun, funlen) in testfunctions:
            chebtech = BndFun.initfun_fixedlen(fun, domain, funlen)
            self.assertTrue((self.emptyfun+chebtech).isempty())
            self.assertTrue((chebtech+self.emptyfun).isempty())

    # check the output of (constant + BndFun)
    #                 and (BndFun + constant)
    def test__add__radd__constant(self):
        domain = Domain(-.5,.9)
        xx = domain(self.yy)
        for (fun, funlen) in testfunctions:
            for const in (-1, 1, 10, -1e5):
                f = lambda x: const + fun(x)
                boundedfun = BndFun.initfun_fixedlen(fun, domain, funlen)
                f1 = const + boundedfun
                f2 = boundedfun + const
                tol = 3e1 * eps * abs(const)
                self.assertLessEqual( infnorm(f(xx)-f1(xx)), tol )
                self.assertLessEqual( infnorm(f(xx)-f2(xx)), tol )

    # check (empty BndFun) - (BndFun) = (empty BndFun)
    #   and (BndFun) - (empty BndFun) = (empty BndFun)
    def test__sub__rsub__empty(self):
        domain = Domain(-2,3)
        for (fun, funlen) in testfunctions:
            chebtech = BndFun.initfun_fixedlen(fun, domain, funlen)
            self.assertTrue((self.emptyfun-chebtech).isempty())
            self.assertTrue((chebtech-self.emptyfun).isempty())

    # check the output of constant - BndFun
    #                 and BndFun - constant
    def test__sub__rsub__constant(self):
        domain = Domain(-.5,.9)
        xx = domain(self.yy)
        for (fun, funlen) in testfunctions:
            for const in (-1, 1, 10, -1e5):
                boundedfun = BndFun.initfun_fixedlen(fun, domain, funlen)
                f = lambda x: const - fun(x)
                g = lambda x: fun(x) - const
                ff = const - boundedfun
                gg = boundedfun - const
                tol = 5e1 * eps * abs(const)
                self.assertLessEqual( infnorm(f(xx)-ff(xx)), tol )
                self.assertLessEqual( infnorm(g(xx)-gg(xx)), tol )

    # check (empty BndFun) * (BndFun) = (empty BndFun)
    #   and (BndFun) * (empty BndFun) = (empty BndFun)
    def test__mul__rmul__empty(self):
        domain = Domain(-2,3)
        for (fun, funlen) in testfunctions:
            chebtech = BndFun.initfun_fixedlen(fun, domain, funlen)
            self.assertTrue((self.emptyfun*chebtech).isempty())
            self.assertTrue((chebtech*self.emptyfun).isempty())

    # check the output of constant * BndFun
    #                 and BndFun * constant
    def test__rmul__constant(self):
        domain = Domain(-.5,.9)
        xx = domain(self.yy)
        for (fun, funlen) in testfunctions:
            for const in (-1, 1, 10, -1e5):
                boundedfun = BndFun.initfun_fixedlen(fun, domain, funlen)
                f = lambda x: const * fun(x)
                g = lambda x: fun(x) * const
                ff = const * boundedfun
                gg = boundedfun * const
                tol = 4e1 * eps * abs(const)
                self.assertLessEqual( infnorm(f(xx)-ff(xx)), tol )
                self.assertLessEqual( infnorm(g(xx)-gg(xx)), tol )

    # check    +(empty BndFun) = (empty BndFun)
    def test__pos__empty(self):
        self.assertTrue( (+self.emptyfun).isempty() )

    # check -(empty BndFun) = (empty BndFun)
    def test__neg__empty(self):
        self.assertTrue( (-self.emptyfun).isempty() )


binops = (
    __add__,
    __sub__,
    __mul__,
    )

# add tests for the binary operators
def binaryOpTester(f, g, domain, binop):
    ff = BndFun.initfun_adaptive(f, domain)
    gg = BndFun.initfun_adaptive(g, domain)
    FG = BndFun.initfun_adaptive(lambda x: binop(f(x),g(x)), domain)
    fg = binop(ff, gg)
    def tester(self):
        xx = domain(self.yy)
        self.assertLessEqual( infnorm(fg(xx)-FG(xx)), 2e2*eps)
    return tester

# Note: defining __radd__(a,b) = __add__(b,a) and feeding this into the
# test will not in fact test the __radd__ functionality of the class.
# These test will need to be added manually.

for binop in binops:
    # add the generic binary operator tests
    for (f, _), (g, _) in combinations(testfunctions, 2):
        domain = Domain(-.5,.9)
        _testfun_ = binaryOpTester(f, g, domain, binop)
        _testfun_.__name__ = \
            "test{}{}_{}".format(binop.__name__, f.__name__,  g.__name__)
        setattr(Algebra, _testfun_.__name__, _testfun_)

unaryops = (
    __pos__,
    __neg__,
    )

# add tests for the unary operators
def unaryOpTester(unaryop, f, domain):
    ff = BndFun.initfun_adaptive(f, domain)
    gg = BndFun.initfun_adaptive(lambda x: unaryop(f(x)), domain)
    GG = unaryop(ff)
    def tester(self):
        xx = domain(self.yy)
        self.assertLessEqual( infnorm(gg(xx)-GG(xx)), 2e2*eps)
    return tester

for unaryop in unaryops:
    for (f, _) in testfunctions:
        domain = Domain(-.5,.9)
        _testfun_ = unaryOpTester(unaryop, f, domain)
        _testfun_.__name__ = \
            "test{}{}".format(unaryop.__name__, f.__name__)
        setattr(Algebra, _testfun_.__name__, _testfun_)

class Roots(TestCase):

    def test_empty(self):
        ff = BndFun.initempty()
        self.assertEquals(ff.roots().size, 0)

    def test_const(self):
        ff = BndFun.initconst(0., Domain(-2,3))
        gg = BndFun.initconst(2., Domain(-2,3))
        self.assertEquals(ff.roots().size, 0)
        self.assertEquals(gg.roots().size, 0)

# add tests for roots
def rootsTester(f, interval, roots, tol):
    domain = Domain(*interval)
    ff = BndFun.initfun_adaptive(f, domain)
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
