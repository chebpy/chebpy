# -*- coding: utf-8 -*-
"""
Unit-tests for pyfun/chebtech.py 
"""
from __future__ import division

#from functools import wraps
from unittest import TestCase

from numpy import arange
from numpy import array
from numpy import inf
from numpy import sin
from numpy import cos
from numpy import exp
from numpy import log
from numpy import linspace
from numpy import all as all_
from numpy import max as max_
from numpy import diff
from numpy.linalg import norm
from numpy.random import rand
from numpy.random import seed

from pyfun.chebtech import ChebTech2
from pyfun.chebtech import bary
from pyfun.chebtech import clenshaw
from pyfun.chebtech import eps

# staticmethod aliases
chebpts      = ChebTech2.chebpts
_vals2coeffs = ChebTech2._vals2coeffs
_coeffs2vals = ChebTech2._coeffs2vals

#def epsclose(arr0, arr1, eps=1e2*eps):
#    """Return True if all elements are within eps"""
#    return allclose(arr0, arr1, rtol=eps, atol=eps)

seed(0)

def infnorm(x):
    return norm(x, inf)


def scaled_tol(n):
    tol = 5e1*eps if n < 20 else log(n)**2.5*eps
    return tol

# ------------------------
# Dynamic Test Generators
# ------------------------

def infNormLessThanTol(a, b, tol):
    def asserter(self):
        self.assertLessEqual(infnorm(a-b), tol)
    return asserter


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

# check the output is of the right length, that the endpoint values are -1 
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

class Evaluation(TestCase):
    """Tests for the Barycentric formula and Clenshaw algorithm"""

    def setUp(self):
        funs_and_names = [
            (lambda x: x**3 + x**2 + x + 1, "poly3(x)"),
            (lambda x: exp(x), "exp(x)"),
            (lambda x: sin(x), "sin(x)"),
            (lambda x: cos(20*x), "cos(20x)"),
        ]
        funs = []
        for k, item in enumerate(funs_and_names):
            fun = item[0]
            fun.__name__ = item[1]
            funs.append(fun)

        self.funs = funs
        self.evalpts = [linspace(-1,1,10**n) for n in arange(6)]

    def test_barycentric_empty(self):
        self.assertEquals(bary(array([]), array([])).size, 0)
        
    def test_clenshaw_empty(self):
        self.assertEquals(clenshaw(array([]), array([1.])).size, 0)

funs = []
funs_and_names = [
    (lambda x: x**3 + x**2 + x + 1, "poly3(x)"),
    (lambda x: exp(x), "exp(x)"),
    (lambda x: sin(x), "sin(x)"),
    (lambda x: cos(20*x), "cos(20x)"),
]
for k, item in enumerate(funs_and_names):
    fun = item[0]
    fun.__name__ = item[1]
    funs.append(fun)

evalpts = [linspace(-1,1,n) for n in 10**array([2,3,4,5])]
ptsarry = [ChebTech2.chebpts(n) for n in array([100, 200])]
methods = [bary, clenshaw]

def evalTester(method, fun, evalpts, chebpts):
    
    x = evalpts
    xk = chebpts
    fvals = fun(xk)
    
    if method is bary:
        vk = ChebTech2.barywts(fvals.size)
        a = bary(x, fvals, xk, vk)
        tol_multiplier = 1e0

    elif method is clenshaw:
        ak = _vals2coeffs(fvals)
        a = clenshaw(x, ak)
        tol_multiplier = 2e1
        
    b = fun(evalpts)
    n = evalpts.size
    tol = tol_multiplier * scaled_tol(n)

    return infNormLessThanTol(a, b, tol)

for method in methods:
    for fun in funs:
        for j, chebpts in enumerate(ptsarry):
            for k, xx in enumerate(evalpts):
                testfun = evalTester(method, fun, xx, chebpts)
                testfun.__name__ = "test_{}_{}_{:02}_{:02}".format(
                    method.__name__, fun.__name__, j, k)
                setattr(Evaluation, testfun.__name__, testfun)


# reset the testsfun variable so it doesn't get picked up by nose
testfun = None