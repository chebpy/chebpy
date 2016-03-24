# -*- coding: utf-8 -*-
"""
Unit-tests for pyfun/chebtech.py 
"""
from __future__ import division

from unittest import TestCase

from numpy import arange
from numpy import array
from numpy import sin
from numpy import cos
from numpy import exp
from numpy import pi
from numpy import all as all_
from numpy import diff
from numpy.random import rand
from numpy.random import seed

from matplotlib.pyplot import subplots

from pyfun.chebtech import ChebTech2
from pyfun.chebtech import eps

from utilities import funs
from utilities import infnorm
from utilities import scaled_tol
from utilities import infNormLessThanTol

seed(0)

# staticmethod aliases
chebpts      = ChebTech2.chebpts
_vals2coeffs = ChebTech2._vals2coeffs
_coeffs2vals = ChebTech2._coeffs2vals

# ------------------------
class ChebyshevPoints(TestCase):
    """Unit-tests for ChebTech2"""

    def test_chebpts_0(self):
        self.assertEquals(ChebTech2.chebpts(0).size, 0)
            
    def test_vals2coeffs_empty(self):
        self.assertEquals(_vals2coeffs(array([])).size, 0)

    def test_coeffs2vals_empty(self):
        self.assertEquals(_coeffs2vals(array([])).size, 0)

    # check we are returned the array for an array of size 1
    def test_vals2coeffs_size1(self):
        for k in arange(10):
            fk = array([k])
            self.assertLessEqual(infnorm(_vals2coeffs(fk)-fk), eps)

    # check we are returned the array for an array of size 1
    def test_coeffs2vals_size1(self):
        for k in arange(10):
            ak = array([k])
            self.assertLessEqual(infnorm(_coeffs2vals(ak)-ak), eps)

    # TODO: further checks for chepbts

# ------------------------------------------------------------------------
# Tests to verify the mutually inverse nature of vals2coeffs and coeffs2vals
# ------------------------------------------------------------------------
def vals2coeffs2valsTester(n):
    def asserter(self):
        values = rand(n)
        coeffs = _vals2coeffs(values)
        _values_ = _coeffs2vals(coeffs)
        self.assertLessEqual( infnorm(values-_values_), scaled_tol(n) )
    return asserter

def coeffs2vals2coeffsTester(n):
    def asserter(self):
        coeffs = rand(n)
        values = _coeffs2vals(coeffs)
        _coeffs_ = _vals2coeffs(values)
        self.assertLessEqual( infnorm(coeffs-_coeffs_), scaled_tol(n) )
    return asserter

for k, n in enumerate(2**arange(2,18,2)):

    # vals2coeffs2vals
    testfun = vals2coeffs2valsTester(n)
    testfun.__name__ = "test_vals2coeffs2vals_{:02}".format(k)
    setattr(ChebyshevPoints, testfun.__name__, testfun)

    # coeffs2vals2coeffs
    testfun = coeffs2vals2coeffsTester(n)
    testfun.__name__ = "test_coeffs2vals2coeffs_{:02}".format(k)
    setattr(ChebyshevPoints, testfun.__name__, testfun)
# ------------------------------------------------------------------------
   
# ------------------------------------------------------------------------
# Add second-kind Chebyshev points test cases to ChebyshevPoints
# ------------------------------------------------------------------------
chebpts2_testlist = (
    (ChebTech2.chebpts(1), array([0.]), eps),
    (ChebTech2.chebpts(2), array([-1., 1.]), eps),
    (ChebTech2.chebpts(3), array([-1., 0., 1.]), eps),
    (ChebTech2.chebpts(4), array([-1., -.5, .5, 1.]), 2*eps),
    (ChebTech2.chebpts(5), array([-1., -2.**(-.5), 0., 2.**(-.5), 1.]), eps),
)
for k, (a,b,tol) in enumerate(chebpts2_testlist):
    testfun = infNormLessThanTol(a,b,tol)
    testfun.__name__ = "test_chebpts_{:02}".format(k+1)
    setattr(ChebyshevPoints, testfun.__name__, testfun)

# check the output is of the correct length, the endpoint values are -1
# and 1, respectively, and that the sequence is monotonically increasing
def chebptsLenTester(k):
    def asserter(self):
        pts = ChebTech2.chebpts(k)
        self.assertEquals(pts.size, k)
        self.assertEquals(pts[0], -1.)
        self.assertEquals(pts[-1], 1.)
        self.assertTrue( all_(diff(pts)) > 0 )
    return asserter
    
for k, n in enumerate(2**arange(2,18,2)):
    testfun = chebptsLenTester(n+3)
    testfun.__name__ = "test_chebpts_len_{:02}".format(k)
    setattr(ChebyshevPoints, testfun.__name__, testfun)
# ------------------------------------------------------------------------

class ClassUsage(TestCase):
    """Unit-tests for miscelaneous ChebTech2 class usage"""

    def setUp(self):
        self.ff = ChebTech2.initfun_fixedlen(lambda x: sin(30*x), 100)
        self.xx = -1 + 2*rand(100)

    # tests for emptiness of ChebTech2 objects
    def test_isempty_True(self):
        f = ChebTech2(array([]))
        self.assertTrue(f.isempty())
        self.assertFalse(not f.isempty())

    def test_isempty_False(self):
        f = ChebTech2(array([1.]))
        self.assertFalse(f.isempty())
        self.assertTrue(not f.isempty())

    # check the size() method is working properly
    def test_size(self):
        cfs = rand(10)
        self.assertEquals(ChebTech2(array([])).size(), 0)
        self.assertEquals(ChebTech2(array([1.])).size(), 1)
        self.assertEquals(ChebTech2(cfs).size(), cfs.size)

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
        self.assertLessEqual(infnorm(b-c), 5e1*eps)

    def test_call_raises(self):
        self.assertRaises(ValueError, self.ff, self.xx, "notamethod")
        self.assertRaises(ValueError, self.ff, self.xx, how="notamethod")


class Plotting(TestCase):
    """Unit-tests for ChebTech2 plotting methods"""

    def setUp(self):
        f = lambda x: sin(3*x) + 5e-1*cos(30*x)
        self.f0 = ChebTech2.initfun_fixedlen(f, 100)
        self.f1 = ChebTech2.initfun_adaptive(f)

    def test_plot(self):
        self.f0.plot()

    def test_plotcoeffs(self):
        fig, ax = subplots()
        self.f0.plotcoeffs(ax=ax)
        self.f1.plotcoeffs(ax=ax, color="r")



class Calculus(TestCase):
    """Unit-tests for ChebTech2 calculus operations"""

    def test_sum_empty(self):
        ff = ChebTech2(array([]))
        self.assertEquals(ff.sum(), None)

    def test_cumsum_empty(self):
        ff = ChebTech2(array([]))
        self.assertTrue(ff.cumsum().isempty())

    def test_diff_empty(self):
        ff = ChebTech2(array([]))
        self.assertTrue(ff.diff().isempty())

# --------------------------------------
#           definite integrals
# --------------------------------------
def_integrals = [
    # (function, number of points, integral, tolerance)
    (lambda x: sin(x),             14,                    .0,      eps),
    (lambda x: sin(4*pi*x),        40,                    .0,  1e1*eps),
    (lambda x: cos(x),             15,     1.682941969615793,    2*eps),
    (lambda x: cos(4*pi*x),        39,                    .0,    2*eps),
    (lambda x: exp(cos(4*pi*x)),  181,     2.532131755504016,    2*eps),
    (lambda x: cos(3244*x),      3389, 5.879599674161602e-04,  5e2*eps),
    (lambda x: exp(x),             15,        exp(1)-exp(-1),    2*eps),
    (lambda x: 1e10*exp(x),        15, 1e10*(exp(1)-exp(-1)), 2e10*eps),
    (lambda x: 0*x+1.,              1,                     2,      eps),
]

def definiteIntegralTester(fun, n, integral, tol):
    ff = ChebTech2.initfun_fixedlen(fun, n)
    def tester(self):
        absdiff = abs(ff.sum()-integral)
        return self.assertLessEqual(absdiff, tol)
    return tester

for k, (fun, n, integral, tol) in enumerate(def_integrals):
    testfun = definiteIntegralTester(fun, n, integral, tol)
    testfun.__name__ = "test_sum_{:02}".format(k)
    setattr(Calculus, testfun.__name__, testfun)

# --------------------------------------
#          indefinite integrals
# --------------------------------------
indef_integrals = [
    # (function, indefinite integral, number of points, tolerance)
    (lambda x: 0*x+1.,      lambda x: x,              1,         eps),
    (lambda x: x,           lambda x: 1/2*x**2,       2,       2*eps),
    (lambda x: x**2,        lambda x: 1/3*x**3,       3,       2*eps),
    (lambda x: x**3,        lambda x: 1/4*x**4,       4,       2*eps),
    (lambda x: x**4,        lambda x: 1/5*x**5,       5,       2*eps),
    (lambda x: x**5,        lambda x: 1/6*x**6,       6,       4*eps),
    (lambda x: sin(x),      lambda x: -cos(x),       16,       2*eps),
    (lambda x: cos(3*x),    lambda x: 1./3*sin(3*x), 23,       2*eps),
    (lambda x: exp(x),      lambda x: exp(x),        16,       3*eps),
    (lambda x: 1e10*exp(x), lambda x: 1e10*exp(x),   16, 1e10*(3*eps)),
]

def indefiniteIntegralTester(fun, dfn, n, tol):
    ff = ChebTech2.initfun_fixedlen(fun, n)
    gg = ChebTech2.initfun_fixedlen(dfn, n+1)
    coeffs = gg.coeffs()
    coeffs[0] = coeffs[0] - dfn(array([-1]))
    def tester(self):
        absdiff = infnorm(ff.cumsum().coeffs() - coeffs)
        return self.assertLessEqual(absdiff, tol)
    return tester

for k, (fun, dfn, n, tol) in enumerate(indef_integrals):
    testfun = indefiniteIntegralTester(fun, dfn, n, tol)
    testfun.__name__ = "test_cumsum_{:02}".format(k)
    setattr(Calculus, testfun.__name__, testfun)

# --------------------------------------
#            derivatives
# --------------------------------------
derivatives = [
    # (function, derivative, number of points, tolerance)
    (lambda x: 0*x+1.,      lambda x: 0*x+0,        1,         eps),
    (lambda x: x,           lambda x: 0*x+1,        2,       2*eps),
    (lambda x: x**2,        lambda x: 2*x,          3,       2*eps),
    (lambda x: x**3,        lambda x: 3*x**2,       4,       2*eps),
    (lambda x: x**4,        lambda x: 4*x**3,       5,       2*eps),
    (lambda x: x**5,        lambda x: 5*x**4,       6,       4*eps),
    (lambda x: sin(x),      lambda x: cos(x),      16,     5e1*eps),
    (lambda x: cos(3*x),    lambda x: -3*sin(3*x), 23,     5e2*eps),
    (lambda x: exp(x),      lambda x: exp(x),      16,     2e2*eps),
    (lambda x: 1e10*exp(x), lambda x: 1e10*exp(x), 16, 1e12*(2*eps)),
]

def derivativeTester(fun, der, n, tol):
    ff = ChebTech2.initfun_fixedlen(fun, n)
    gg = ChebTech2.initfun_fixedlen(der, max(n-1,1))
    def tester(self):
        absdiff = infnorm(ff.diff().coeffs() - gg.coeffs())
        return self.assertLessEqual(absdiff, tol)
    return tester

for k, (fun, der, n, tol) in enumerate(derivatives):
    testfun = derivativeTester(fun, der, n, tol)
    testfun.__name__ = "test_diff_{:02}".format(k)
    setattr(Calculus, testfun.__name__, testfun)


class Construction(TestCase):
    """Unit-tests for construction of ChebTech2 objects"""

    def test_coeff_construction(self):
        coeffs = rand(10)
        f = ChebTech2(coeffs)
        self.assertIsInstance(f, ChebTech2)
        self.assertTrue(infNormLessThanTol(f.coeffs(), coeffs, eps))

# TODO: check these lengths against Chebfun
# TODO: more examples
fun_lens = {
    "cos(20x)": 53,
    "exp(x)": 18,
    "poly3(x)": 4,
    "sin(x)": 16,
}

def adaptiveTester(fun):
    ff = ChebTech2.initfun_adaptive(fun)
    def tester(self):
        return self.assertEquals(ff.size(), fun_lens[fun.__name__])
    return tester

for fun in funs:
    testfun = adaptiveTester(fun)
    testfun.__name__ = "test_adaptive_{}".format(fun.__name__)
    setattr(Construction, testfun.__name__, testfun)

def fixedlenTester(fun, n):
    ff = ChebTech2.initfun_fixedlen(fun, n)
    def tester(self):
        return self.assertEquals(ff.size(), n)
    return tester

for fun in funs:
    for n in array([50, 100, 300, 500]):
        testfun = fixedlenTester(fun, n)
        testfun.__name__ = \
            "test_fixedlen_{}_{:003}pts".format(fun.__name__, n)
        setattr(Construction, testfun.__name__, testfun)

# reset the testsfun variable so it doesn't get picked up by nose
testfun = None
