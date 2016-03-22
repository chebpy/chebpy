# -*- coding: utf-8 -*-
"""
Unit-tests for pyfun/chebtech.py 
"""
from __future__ import division

#from functools import wraps
from unittest import TestCase

from numpy import array
from numpy import linspace

from pyfun.chebtech import ChebTech2
from pyfun.utilities import bary
from pyfun.utilities import clenshaw

from utilities import funs
from utilities import scaled_tol
from utilities import infNormLessThanTol

class Evaluation(TestCase):
    """Tests for the Barycentric formula and Clenshaw algorithm"""

    def test_barycentric_empty(self):
        self.assertEquals(
            bary(array([]), array([]), 
                 array([]), array([])).size, 0)
        
    def test_clenshaw_empty(self):
        self.assertEquals(clenshaw(array([]), array([1.])).size, 0)

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