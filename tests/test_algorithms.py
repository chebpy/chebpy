"""Unit-tests for pyfun/utilities.py"""

import unittest

import numpy as np

from chebpy.core.settings import DefaultPreferences
from chebpy.core.chebtech import Chebtech2
from chebpy.core.algorithms import bary, clenshaw, coeffmult

from .utilities import testfunctions, scaled_tol, infNormLessThanTol, infnorm

# aliases
pi = np.pi
sin = np.sin
cos = np.cos
exp = np.exp
eps = DefaultPreferences.eps

np.random.seed(0)

# turn off 'divide' and 'invalid' Runtimewarnings: these are invoked in the
# barycentric formula and the warned-of behaviour is actually required
np.seterr(divide="ignore", invalid="ignore")


class Evaluation(unittest.TestCase):
    """Tests for the Barycentric formula and Clenshaw algorithm"""

    def setUp(self):
        npts = 15
        self.xk = Chebtech2._chebpts(npts)
        self.vk = Chebtech2._barywts(npts)
        self.fk = np.random.rand(npts)
        self.ak = np.random.rand(11)
        self.xx = -1 + 2 * np.random.rand(9)
        self.pts = -1 + 2 * np.random.rand(1001)

    # check an empty array is returned whenever either or both of the first
    # two arguments are themselves empty arrays
    def test_bary__empty(self):
        null = (None, None)
        self.assertEquals(bary(np.array([]), np.array([]), *null).size, 0)
        self.assertEquals(bary(np.array([0.1]), np.array([]), *null).size, 0)
        self.assertEquals(bary(np.array([]), np.array([0.1]), *null).size, 0)
        self.assertEquals(bary(self.pts, np.array([]), *null).size, 0)
        self.assertEquals(bary(np.array([]), self.pts, *null).size, 0)
        self.assertNotEquals(bary(np.array([0.1]), np.array([0.1]), *null).size, 0)

    def test_clenshaw__empty(self):
        self.assertEquals(clenshaw(np.array([]), np.array([])).size, 0)
        self.assertEquals(clenshaw(np.array([]), np.array([1.0])).size, 0)
        self.assertEquals(clenshaw(np.array([1.0]), np.array([])).size, 0)
        self.assertEquals(clenshaw(self.pts, np.array([])).size, 0)
        self.assertEquals(clenshaw(np.array([]), self.pts).size, 0)
        self.assertNotEquals(clenshaw(np.array([0.1]), np.array([0.1])).size, 0)

    # check that scalars get evaluated to scalars (not arrays)
    def test_clenshaw__scalar_input(self):
        for x in self.xx:
            self.assertTrue(np.isscalar(clenshaw(x, self.ak)))
        self.assertFalse(np.isscalar(clenshaw(xx, self.ak)))

    def test_bary__scalar_input(self):
        for x in self.xx:
            self.assertTrue(np.isscalar(bary(x, self.fk, self.xk, self.vk)))
        self.assertFalse(np.isscalar(bary(xx, self.fk, self.xk, self.vk)))

    # Check that we always get float output for constant Chebtechs, even
    # when passing in an integer input.
    # TODO: Move these tests elsewhere?
    def test_bary__float_output(self):
        ff = Chebtech2.initconst(1)
        gg = Chebtech2.initconst(1.0)
        self.assertTrue(isinstance(ff(0, "bary"), float))
        self.assertTrue(isinstance(gg(0, "bary"), float))

    def test_clenshaw__float_output(self):
        ff = Chebtech2.initconst(1)
        gg = Chebtech2.initconst(1.0)
        self.assertTrue(isinstance(ff(0, "clenshaw"), float))
        self.assertTrue(isinstance(gg(0, "clenshaw"), float))

    # Check that we get consistent output from bary and clenshaw
    # TODO: Move these tests elsewhere?
    def test_bary_clenshaw_consistency(self):
        coeffs = np.random.rand(3)
        evalpts = (0.5, np.array([]), np.array([0.5]), np.array([0.5, 0.6]))
        for n in range(len(coeffs)):
            ff = Chebtech2(coeffs[:n])
            for xx in evalpts:
                fb = ff(xx, "bary")
                fc = ff(xx, "clenshaw")
                self.assertEquals(type(fb), type(fc))


evalpts = [np.linspace(-1, 1, int(n)) for n in np.array([1e2, 1e3, 1e4, 1e5])]
ptsarry = [Chebtech2._chebpts(n) for n in np.array([100, 200])]
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
    for (fun, _, _) in testfunctions:
        for j, chebpts in enumerate(ptsarry):
            for k, xx in enumerate(evalpts):
                testfun = evalTester(method, fun, xx, chebpts)
                testfun.__name__ = "test_{}_{}_{:02}_{:02}".format(
                    method.__name__, fun.__name__, j, k
                )
                setattr(Evaluation, testfun.__name__, testfun)


class CoeffMult(unittest.TestCase):
    def setUp(self):
        self.f = lambda x: exp(x)
        self.g = lambda x: cos(x)
        self.fn = 15
        self.gn = 15

    def test_coeffmult(self):
        def h(x):
            return self.f(x) * self.g(x)

        f, g = self.f, self.g
        fn, gn = self.fn, self.gn
        hn = fn + gn - 1
        fc = Chebtech2.initfun(f, fn).prolong(hn).coeffs
        gc = Chebtech2.initfun(g, gn).prolong(hn).coeffs
        hc = coeffmult(fc, gc)
        HC = Chebtech2.initfun(h, hn).coeffs
        self.assertLessEqual(infnorm(hc - HC), 2e1 * eps)


# reset the testsfun variable so it doesn't get picked up by nose
testfun = None
