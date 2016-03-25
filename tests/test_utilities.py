# -*- coding: utf-8 -*-
"""
Unit-tests for pyfun/utilities.py
"""
from __future__ import division

#from functools import wraps
from unittest import TestCase

from numpy import array
from numpy import linspace
from numpy import isscalar
from numpy.random import rand
from numpy.random import seed

from pyfun.chebtech import ChebTech2
from pyfun.utilities import bary
from pyfun.utilities import clenshaw

from utilities import funs
from utilities import scaled_tol
from utilities import infNormLessThanTol

seed(0)

class Evaluation(TestCase):
    """Tests for the Barycentric formula and Clenshaw algorithm"""

    def setUp(self):
        npts = 15
        self.xk = ChebTech2.chebpts(npts)
        self.vk = ChebTech2.barywts(npts)
        self.fk = rand(npts)
        self.ak = rand(11)
        self.xx = -1 + 2*rand(9)
        self.pts = -1 + 2*rand(1001)

    # check an empty array is returned whenever either or both of the first
    # two arguments are themselves empty arrays
    def test_bary__empty(self):
        null = (None, None)
        self.assertEquals(bary(array([]), array([]), *null).size, 0)
        self.assertEquals(bary(array([.1]), array([]), *null).size, 0)
        self.assertEquals(bary(array([]), array([.1]), *null).size, 0)
        self.assertEquals(bary(self.pts, array([]), *null).size, 0)
        self.assertEquals(bary(array([]), self.pts, *null).size, 0)
        self.assertNotEquals(bary(array([.1]), array([.1]), *null).size, 0)

    def test_clenshaw__empty(self):
        self.assertEquals(clenshaw(array([]), array([])).size, 0)
        self.assertEquals(clenshaw(array([]), array([1.])).size, 0)
        self.assertEquals(clenshaw(array([1.]), array([])).size, 0)
        self.assertEquals(clenshaw(self.pts, array([])).size, 0)
        self.assertEquals(clenshaw(array([]), self.pts).size, 0)
        self.assertNotEquals(clenshaw(array([.1]), array([.1])).size, 0)

    # check that scalars get evaluated to scalars (not arrays)
    def test_clenshaw__scalar_input(self):
        for x in self.xx:
            self.assertTrue( isscalar(clenshaw(x, self.ak)) )
        self.assertFalse( isscalar(clenshaw(xx, self.ak)) )

    def test_bary__scalar_input(self):
        for x in self.xx:
            self.assertTrue( isscalar(bary(x, self.fk, self.xk, self.vk)) )
        self.assertFalse( isscalar(bary(xx, self.fk, self.xk, self.vk)) )

    # TODO: check that we always get out a float even if we pass in an int
#    def test_bary_float_output(self):
#        pass
#
#    def test_clenshaw_float_output(self):
#        pass

evalpts = [linspace(-1,1,n) for n in array([1e2, 1e3, 1e4, 1e5])]
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
