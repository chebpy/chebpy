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
from numpy import exp
from numpy import cos
from numpy.random import rand
from numpy.random import seed

from pyfun.settings import DefaultPrefs
from pyfun.chebtech import Chebtech2
from pyfun.chebfun import chebfun
from pyfun.algorithms import bary
from pyfun.algorithms import clenshaw
from pyfun.algorithms import coeffmult
from pyfun.utilities import Subdomain
from pyfun.utilities import Domain

from pyfun.exceptions import SubdomainValues
from pyfun.exceptions import SubdomainGap
from pyfun.exceptions import SubdomainOverlap

from utilities import testfunctions
from utilities import scaled_tol
from utilities import infNormLessThanTol
from utilities import infnorm

seed(0)

eps = DefaultPrefs.eps

class Evaluation(TestCase):
    """Tests for the Barycentric formula and Clenshaw algorithm"""

    def setUp(self):
        npts = 15
        self.xk = Chebtech2._chebpts(npts)
        self.vk = Chebtech2._barywts(npts)
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

    # Check that we always get float output for constant Chebtechs, even 
    # when passing in an integer input.
    # TODO: Move these tests elsewhere?
    def test_bary__float_output(self):
        ff = Chebtech2.initconst(1)
        gg = Chebtech2.initconst(1.)
        self.assertTrue(isinstance(ff(0, "bary"), float))
        self.assertTrue(isinstance(gg(0, "bary"), float))

    def test_clenshaw__float_output(self):
        ff = Chebtech2.initconst(1)
        gg = Chebtech2.initconst(1.)
        self.assertTrue(isinstance(ff(0, "clenshaw"), float))
        self.assertTrue(isinstance(gg(0, "clenshaw"), float))

    # Check that we get consistent output from bary and clenshaw
    # TODO: Move these tests elsewhere?
    def test_bary_clenshaw_consistency(self):
        coeffs = rand(3)
        evalpts = (0.5, array([]), array([.5]), array([.5, .6]))
        for n in range(len(coeffs)):
            ff = Chebtech2(coeffs[:n])
            for xx in evalpts:
                fb = ff(xx, "bary")
                fc = ff(xx, "clenshaw")
                self.assertEquals(type(fb), type(fc))

evalpts = [linspace(-1,1,n) for n in array([1e2, 1e3, 1e4, 1e5])]
ptsarry = [Chebtech2._chebpts(n) for n in array([100, 200])]
methods = [bary, clenshaw]

def evalTester(method, fun, evalpts, chebpts):

    x = evalpts
    xk = chebpts
    fvals = fun(xk)

    if method is bary:
        vk = Chebtech2._barywts(fvals.size)
        a = bary(x, fvals, xk, vk)
        tol_multiplier = 1e0

    elif method is clenshaw:
        ak = Chebtech2._vals2coeffs(fvals)
        a = clenshaw(x, ak)
        tol_multiplier = 2e1

    b = fun(evalpts)
    n = evalpts.size
    tol = tol_multiplier * scaled_tol(n)

    return infNormLessThanTol(a, b, tol)

for method in methods:
    for (fun, _) in testfunctions:
        for j, chebpts in enumerate(ptsarry):
            for k, xx in enumerate(evalpts):
                testfun = evalTester(method, fun, xx, chebpts)
                testfun.__name__ = "test_{}_{}_{:02}_{:02}".format(
                    method.__name__, fun.__name__, j, k)
                setattr(Evaluation, testfun.__name__, testfun)

# tests for Miscelaneous functionality
class Misc(TestCase):

    def setUp(self):
        self.f = lambda x: exp(x)
        self.g = lambda x: cos(x)
        self.fn = 15
        self.gn = 15

    def test_coeffmult(self):
        f, g = self.f, self.g
        fn, gn = self.fn, self.gn
        hn = fn + gn - 1
        h  = lambda x: self.f(x) * self.g(x)
        fc = Chebtech2.initfun(f, fn).prolong(hn).coeffs()
        gc = Chebtech2.initfun(g, gn).prolong(hn).coeffs()
        hc = coeffmult(fc, gc)
        HC = Chebtech2.initfun(h, hn).coeffs()
        self.assertLessEqual( infnorm(hc-HC), 2e1*eps)

# tests for usage of the Subdomain class
class TestSubdomain(TestCase):

    def test_init(self):
        Subdomain(-1,1)
        self.assertTrue((Subdomain().values==array([-1,1])).all())

    def test_init_disallow(self):
        self.assertRaises(SubdomainValues, Subdomain, 2, 0)
        self.assertRaises(SubdomainValues, Subdomain, 0, 0)

    def test__eq__(self):
        d1 = Subdomain(-2,3)
        d2 = Subdomain(-2,3)
        d3 = Subdomain(-1,1)
        self.assertTrue(Subdomain()==Subdomain())
        self.assertTrue(d1==d2)
        self.assertTrue(d2==d1)
        self.assertFalse(d3==d1)
        self.assertFalse(d2==d3)

    def test__ne__(self):
        d1 = Subdomain(-2,3)
        d2 = Subdomain(-2,3)
        d3 = Subdomain(-1,1)
        self.assertFalse(Subdomain()!=Subdomain())
        self.assertFalse(d1!=d2)
        self.assertFalse(d2!=d1)
        self.assertTrue(d3!=d1)
        self.assertTrue(d2!=d3)

    def test_maps(self):
        yy = -1 + 2 * rand(1000)
        subdomain = Subdomain(-2,3)
        vals = subdomain.invmap( subdomain(yy) ) - yy
        self.assertLessEqual( infnorm(vals), eps)

    def test_isinterior(self):
        npts = 1000
        x1 = linspace(-2, 3,npts)
        x2 = linspace(-3,-2,npts)
        x3 = linspace(3,4,npts)
        x4 = linspace(5,6,npts)
        subdomain = Subdomain(-2,3)
        self.assertEquals(subdomain.isinterior(x1).sum(), npts-2)
        self.assertEquals(subdomain.isinterior(x2).sum(), 0)
        self.assertEquals(subdomain.isinterior(x3).sum(), 0)
        self.assertEquals(subdomain.isinterior(x4).sum(), 0)


# tests for usage of the Subdomain class
class TestDomain(TestCase):

    def setUp(self):
        self.sd1 = Subdomain(-2,-1)
        self.sd2 = Subdomain(-1,1)
        self.sd3 = Subdomain(1,2)
        self.sd4 = Subdomain(2,3)
        self.sd5 = Subdomain(-2,1)

    def test_init(self):
        Domain([self.sd1])
        Domain([self.sd1, self.sd2])

    def test_init_disallow(self):
        self.assertRaises(SubdomainGap, Domain, [self.sd2,self.sd4])
        self.assertRaises(SubdomainOverlap, Domain, [self.sd1,self.sd5])

    def test_init_from_funs(self):
        ff = chebfun(lambda x: cos(x), linspace(-10,10,11))
        Domain.init_from_funs(ff.funs)
        self.assertRaises(SubdomainGap, Domain.init_from_funs, ff.funs[::2])

    def test__eq__(self):
        d1 = Domain([self.sd1, self.sd2, self.sd3, self.sd4])
        d2 = Domain([self.sd1, self.sd2, self.sd3, self.sd4])
        d3 = Domain([self.sd1])
        self.assertTrue(d1==d2)
        self.assertFalse(d1==d3)

    def test__ne__(self):
        d1 = Domain([self.sd1, self.sd2, self.sd3, self.sd4])
        d2 = Domain([self.sd1, self.sd2, self.sd3, self.sd4])
        d3 = Domain([self.sd1])
        self.assertFalse(d1!=d2)
        self.assertTrue(d1!=d3)

# reset the testsfun variable so it doesn't get picked up by nose
testfun = None
