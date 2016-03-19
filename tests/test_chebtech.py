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
from numpy import linspace
from numpy.linalg import norm

from pyfun.chebtech import ChebTech2
from pyfun.chebtech import bary
from pyfun.chebtech import clenshaw
from pyfun.chebtech import eps

# staticmethod aliases
chebpts     = ChebTech2.chebpts
_vals2coeffs = ChebTech2._vals2coeffs
_coeffs2vals = ChebTech2._coeffs2vals

#def epsclose(arr0, arr1, eps=1e2*eps):
#    """Return True if all elements are within eps"""
#    return allclose(arr0, arr1, rtol=eps, atol=eps)

def infnorm(x):
    return norm(x, inf)

class TestChebTech2(TestCase):
    """Unit-tests for Chebtech2"""

    def setUp(self):
        self.pts0 = chebpts(0)
        self.pts1 = chebpts(1)
        self.pts2 = chebpts(2)
        self.pts3 = chebpts(3)
        self.pts4 = chebpts(4)
        self.pts5 = chebpts(5)
      
    def test_chebpts_0(self):
        assert self.pts0.size == 0
        
    def test_chebpts_1(self):
        assert infnorm( self.pts1 - array([0.]) ) < eps

    def test_chebpts_2(self):
        assert infnorm( self.pts2 -array([-1., 1.]) ) < eps

    def test_chebpts_3(self):
        assert infnorm( self.pts3 - array([-1., 0., 1.]) ) < eps

    def test_chebpts_4(self):
        assert infnorm( self.pts4 - array([-1., -.5, .5, 1.]) ) < 2*eps
        
    def test_chebpts_5(self):
        r2 = 2.**(-.5)
        assert infnorm( self.pts5 - array([-1., -r2, 0., r2, 1.]) ) < eps
               
    def test_chebpts_len(self):
        for k in 100 * arange(10):
            assert chebpts(k).size == k
            
    def test_vals2coeffs_0(self):
        assert _vals2coeffs(array([])).size == 0
        
    def test_vals2coeffs_1(self):
        for x in arange(3):
            aa = array([x])
            coeffs = _vals2coeffs(aa)
            assert infnorm( coeffs - aa ) < eps
        
    def test_coeffs2vals_0(self):
        assert _coeffs2vals(array([])).size == 0


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
        
