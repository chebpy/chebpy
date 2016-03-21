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
from numpy import log
from numpy import linspace
from numpy import all as all_
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

# ------------------------
# Dynamic Test Generators
# ------------------------

def infNormLessThanTol(a, b, tol):
    def asserter(self):
        msg = "Not all components were less than the required tolerance"
        self.assertLessEqual(infnorm(a-b), tol, msg)
    return asserter



class TestChebTech2(TestCase):
    """Unit-tests for ChebTech2"""

    def test_chebpts_0(self):
        self.assertEquals(chebpts(0).size, 0)
            
    def test_vals2coeffs_0(self):
        assert _vals2coeffs(array([])).size == 0
        
    def test_vals2coeffs_1(self):
        for x in arange(3):
            aa = array([x])
            coeffs = _vals2coeffs(aa)
            assert infnorm( coeffs - aa ) < eps
        
    def test_coeffs2vals_0(self):
        assert _coeffs2vals(array([])).size == 0


# ------------------------------------------------------------------------
# Tests to verify the mutually inverse nature of vals2coeffs and coeffs2vals
# ------------------------------------------------------------------------
def vals2coeffs2valsTester(n, tol=eps):
    def asserter(self):
        values = rand(n)
        coeffs = _vals2coeffs(values)
        _values_ = _coeffs2vals(coeffs)
        self.assertLessEqual( infnorm(values-_values_), log(n)**2.5*tol )
    return asserter

def coeffs2vals2coeffsTester(n, tol=eps):
    def asserter(self):
        coeffs = rand(n)
        values = _coeffs2vals(coeffs)
        _coeffs_ = _vals2coeffs(values)
        self.assertLessEqual( infnorm(coeffs-_coeffs_), log(n)**2.5*tol )
    return asserter

for k, n in enumerate(2**arange(2,18,2)):

    # vals2coeffs2vals
    testfun = vals2coeffs2valsTester(n)
    testfun.__name__ = "test_vals2coeffs2vals_{:02}".format(k)
    setattr(TestChebTech2, testfun.__name__, testfun)

    # coeffs2vals2coeffs
    testfun = coeffs2vals2coeffsTester(n)
    testfun.__name__ = "test_coeffs2vals2coeffs_{:02}".format(k)
    setattr(TestChebTech2, testfun.__name__, testfun)

# ------------------------------------------------------------------------
   
# ------------------------------------------------------------------------
# Add second-kind Chebyshev points test cases to TestChebTech2 
# ------------------------------------------------------------------------
chebpts2_testlist = (
    (chebpts(1), array([0.]), eps),
    (chebpts(2), array([-1., 1.]), eps),
    (chebpts(3), array([-1., 0., 1.]), eps),   
    (chebpts(4), array([-1., -.5, .5, 1.]), 2*eps),
    (chebpts(5), array([-1., -2.**(-.5), 0., 2.**(-.5), 1.]), eps),
)
for k, (a,b,tol) in enumerate(chebpts2_testlist):
    testfun = infNormLessThanTol(a,b,tol)
    testfun.__name__ = "test_chebpts_{:02}".format(k+1)
    setattr(TestChebTech2, testfun.__name__, testfun)

# check the output is of the right length, that the endpoint values are -1 
# and 1, respectively, and that the sequence is monotonically increasing
def chebptsLenTester(k):
    def asserter(self):
        pts = chebpts(k)
        self.assertEquals(pts.size, k)
        self.assertEquals(pts[0], -1.)
        self.assertEquals(pts[-1], 1.)
        self.assertTrue( all_(diff(pts)) > 0 )
    return asserter
    
for k, n in enumerate(2**arange(2,18,2)):
    testfun = chebptsLenTester(n+3)
    testfun.__name__ = "test_chebpts_len_{:02}".format(k)
    setattr(TestChebTech2, testfun.__name__, testfun)    
# ------------------------------------------------------------------------
   
# reset the testsfun variable so it doesn't get picked up by nose
testfun = None

class TestBary(TestCase):
    """Unit-tests for the barycentric formula"""
    
    def setUp(self):
        self.f1 = lambda x: x**3 + x**2 + x + 1
        self.f2 = sin
        self.xx = linspace(-1, 1, 1e3)
        self.yy = linspace(-1, 1, 1e5)
    
    # check that we get an empty arrya back when we look at two empty arrays
    def test_empty(self):
        assert bary(array([]), array([])).size == 0 
    
    # Evaluate 1e3 points at a function defined by 11 chebpts   
    def test_evaluation_1(self):
        fun, xx = self.f1, self.xx
        ff = fun(chebpts(11))
        assert infnorm( bary(xx, ff) - fun(xx) ) < 1e1*eps

    # Evaluate 1e3 points at a function defined by 1e5 chebpts
    def test_evaluation_2(self):
        fun, xx = self.f1, self.xx
        ff = fun(chebpts(1e5))
        assert infnorm( bary(xx, ff) - fun(xx) ) < 1e2*eps

    # Evaluate 1e5 points at a function defined by 1e3 chebpts
    def test_evaluation_3(self):
        fun, yy = self.f1, self.yy
        ff = fun(chebpts(1e3))
        assert infnorm( bary(yy, ff) - fun(yy) ) < 1e2*eps

    # Evaluate 11 chebpts at a function defined by those chebpts
    def test_evaluation_4(self):
        fun, xx = self.f1, chebpts(11)
        ff = fun(xx)
        assert infnorm( bary(xx, ff) - fun(xx) ) < eps
        
        
class TestClenshaw(TestCase):
    """Unit-tests for the Clenshaw Algorithm"""
    
    def setUp(self):
        self.f1 = lambda x: x**3 + x**2 + x + 1
        self.f2 = sin
        self.xx = linspace(-1, 1, 1e3)
        self.yy = linspace(-1, 1, 1e5)
    
    # check that we get an empty array back when we look at two empty arrays
    def test_empty(self):
        assert bary(array([]), array([1])).size == 0 
    
    # Evaluate 1e3 points at a function defined by 11 chebpts   
    def test_evaluation_1(self):
        fun, xx = self.f1, self.xx
        ak = _vals2coeffs(fun(chebpts(11)))
        assert infnorm( clenshaw(xx, ak) - fun(xx) ) < 1e1*eps

    # Evaluate 1e3 points at a function defined by 1e5 chebpts
    def test_evaluation_2(self):
        fun, xx = self.f1, self.xx
        ak = _vals2coeffs( fun(chebpts(1e5)) )
        assert infnorm( clenshaw(xx, ak) - fun(xx) ) < 2.5e3*eps

    # Evaluate 1e5 points at a function defined by 1e3 chebpts
    def test_evaluation_3(self):
        fun, yy = self.f1, self.yy
        ak = _vals2coeffs( fun(chebpts(1e3)) )
        assert infnorm( clenshaw(yy, ak) - fun(yy) )  < 3e2*eps

    # Evaluate 11 chebpts at a function defined by those chebpts
    def test_evaluation_4(self):
        fun, xx = self.f1, chebpts(11)
        ff = _vals2coeffs( fun(xx) )
        assert infnorm( clenshaw(xx, ff) - fun(xx) ) < 5*eps
        
