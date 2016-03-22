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
from numpy import finfo
from numpy import linspace
from numpy.linalg import norm

from pyfun.chebtech import ChebTech2
from pyfun.utilities import bary
from pyfun.utilities import clenshaw

eps = finfo(float).eps

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
        self.assertEquals(
            bary(array([]), array([]), 
                 array([]), array([])).size, 0)
        
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
        ak = ChebTech2._vals2coeffs(fvals)
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