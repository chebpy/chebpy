"""Unit-tests for chebpy/core/chebtech.py"""

import itertools
import operator
import unittest

import numpy as np

from chebpy.core.settings import DefaultPreferences
from chebpy.core.chebtech import Chebtech2
from chebpy.core.algorithms import standard_chop
from chebpy.core.plotting import import_plt

from .utilities import testfunctions, infnorm, scaled_tol, infNormLessThanTol, joukowsky

np.random.seed(0)

# aliases
pi = np.pi
sin = np.sin
cos = np.cos
exp = np.exp
eps = DefaultPreferences.eps
_vals2coeffs = Chebtech2._vals2coeffs
_coeffs2vals = Chebtech2._coeffs2vals


# ------------------------
class ChebyshevPoints(unittest.TestCase):
    """Unit-tests for Chebtech2"""

    def test_chebpts_0(self):
        self.assertEquals(Chebtech2._chebpts(0).size, 0)

    def test_vals2coeffs_empty(self):
        self.assertEquals(_vals2coeffs(np.array([])).size, 0)

    def test_coeffs2vals_empty(self):
        self.assertEquals(_coeffs2vals(np.array([])).size, 0)

    # check we are returned the array for an array of size 1
    def test_vals2coeffs_size1(self):
        for k in np.arange(10):
            fk = np.array([k])
            self.assertLessEqual(infnorm(_vals2coeffs(fk) - fk), eps)

    # check we are returned the array for an array of size 1
    def test_coeffs2vals_size1(self):
        for k in np.arange(10):
            ak = np.array([k])
            self.assertLessEqual(infnorm(_coeffs2vals(ak) - ak), eps)

    # TODO: further checks for chepbts


# ------------------------------------------------------------------------
# Tests to verify the mutually inverse nature of vals2coeffs and coeffs2vals
# ------------------------------------------------------------------------
def vals2coeffs2valsTester(n):
    def asserter(self):
        values = np.random.rand(n)
        coeffs = _vals2coeffs(values)
        _values_ = _coeffs2vals(coeffs)
        self.assertLessEqual(infnorm(values - _values_), scaled_tol(n))

    return asserter


def coeffs2vals2coeffsTester(n):
    def asserter(self):
        coeffs = np.random.rand(n)
        values = _coeffs2vals(coeffs)
        _coeffs_ = _vals2coeffs(values)
        self.assertLessEqual(infnorm(coeffs - _coeffs_), scaled_tol(n))

    return asserter


for k, n in enumerate(2 ** np.arange(2, 18, 2) + 1):

    # vals2coeffs2vals
    _testfun_ = vals2coeffs2valsTester(n)
    _testfun_.__name__ = "test_vals2coeffs2vals_{:02}".format(k)
    setattr(ChebyshevPoints, _testfun_.__name__, _testfun_)

    # coeffs2vals2coeffs
    _testfun_ = coeffs2vals2coeffsTester(n)
    _testfun_.__name__ = "test_coeffs2vals2coeffs_{:02}".format(k)
    setattr(ChebyshevPoints, _testfun_.__name__, _testfun_)
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
# Add second-kind Chebyshev points test cases to ChebyshevPoints
# ------------------------------------------------------------------------
chebpts2_testlist = (
    (Chebtech2._chebpts(1), np.array([0.0]), eps),
    (Chebtech2._chebpts(2), np.array([-1.0, 1.0]), eps),
    (Chebtech2._chebpts(3), np.array([-1.0, 0.0, 1.0]), eps),
    (Chebtech2._chebpts(4), np.array([-1.0, -0.5, 0.5, 1.0]), 2 * eps),
    (
        Chebtech2._chebpts(5),
        np.array([-1.0, -(2.0 ** (-0.5)), 0.0, 2.0 ** (-0.5), 1.0]),
        eps,
    ),
)
for k, (a, b, tol) in enumerate(chebpts2_testlist):
    _testfun_ = infNormLessThanTol(a, b, tol)
    _testfun_.__name__ = "test_chebpts_{:02}".format(k + 1)
    setattr(ChebyshevPoints, _testfun_.__name__, _testfun_)


# check the output is of the correct length, the endpoint values are -1
# and 1, respectively, and that the sequence is monotonically increasing
def chebptsLenTester(k):
    def asserter(self):
        pts = Chebtech2._chebpts(k)
        self.assertEquals(pts.size, k)
        self.assertEquals(pts[0], -1.0)
        self.assertEquals(pts[-1], 1.0)
        self.assertTrue(np.all(np.diff(pts)) > 0)

    return asserter


for k, n in enumerate(2 ** np.arange(2, 18, 2)):
    _testfun_ = chebptsLenTester(n + 3)
    _testfun_.__name__ = "test_chebpts_len_{:02}".format(k)
    setattr(ChebyshevPoints, _testfun_.__name__, _testfun_)
# ------------------------------------------------------------------------


class ClassUsage(unittest.TestCase):
    """Unit-tests for miscelaneous Chebtech2 class usage"""

    def setUp(self):
        self.ff = Chebtech2.initfun_fixedlen(lambda x: np.sin(30 * x), 100)
        self.xx = -1 + 2 * np.random.rand(100)

    # tests for emptiness of Chebtech2 objects
    def test_isempty_True(self):
        f = Chebtech2(np.array([]))
        self.assertTrue(f.isempty)
        self.assertFalse(not f.isempty)

    def test_isempty_False(self):
        f = Chebtech2(np.array([1.0]))
        self.assertFalse(f.isempty)
        self.assertTrue(not f.isempty)

    # tests for constantness of Chebtech2 objects
    def test_isconst_True(self):
        f = Chebtech2(np.array([1.0]))
        self.assertTrue(f.isconst)
        self.assertFalse(not f.isconst)

    def test_isconst_False(self):
        f = Chebtech2(np.array([]))
        self.assertFalse(f.isconst)
        self.assertTrue(not f.isconst)

    # check the size() method is working properly
    def test_size(self):
        cfs = np.random.rand(10)
        self.assertEquals(Chebtech2(np.array([])).size, 0)
        self.assertEquals(Chebtech2(np.array([1.0])).size, 1)
        self.assertEquals(Chebtech2(cfs).size, cfs.size)

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
        self.assertLessEqual(infnorm(b - c), 5e1 * eps)

    def test_call_raises(self):
        self.assertRaises(ValueError, self.ff, self.xx, "notamethod")
        self.assertRaises(ValueError, self.ff, self.xx, how="notamethod")

    def test_prolong(self):
        for k in [0, 1, 20, self.ff.size, 200]:
            self.assertEquals(self.ff.prolong(k).size, k)

    def test_vscale_empty(self):
        gg = Chebtech2(np.array([]))
        self.assertEquals(gg.vscale, 0.0)

    def test_copy(self):
        ff = self.ff
        gg = self.ff.copy()
        self.assertEquals(ff, ff)
        self.assertEquals(gg, gg)
        self.assertNotEquals(ff, gg)
        self.assertEquals(infnorm(ff.coeffs - gg.coeffs), 0)

    def test_simplify(self):
        gg = self.ff.simplify()
        # check that simplify is calling standard_chop underneath
        self.assertEqual(gg.size, standard_chop(self.ff.coeffs))
        self.assertEqual(infnorm(self.ff.coeffs[: gg.size] - gg.coeffs), 0)
        # check we are returned a copy of self's coeffcients by changing
        # one entry of gg
        fcfs = self.ff.coeffs
        gcfs = gg.coeffs
        self.assertEqual((fcfs[: gg.size] - gcfs).sum(), 0)
        gg.coeffs[0] = 1
        self.assertNotEqual((fcfs[: gg.size] - gcfs).sum(), 0)


# --------------------------------------
#          vscale estimates
# --------------------------------------
vscales = [
    # (function, number of points, vscale)
    (lambda x: sin(4 * pi * x), 40, 1),
    (lambda x: cos(x), 15, 1),
    (lambda x: cos(4 * pi * x), 39, 1),
    (lambda x: exp(cos(4 * pi * x)), 181, exp(1)),
    (lambda x: cos(3244 * x), 3389, 1),
    (lambda x: exp(x), 15, exp(1)),
    (lambda x: 1e10 * exp(x), 15, 1e10 * exp(1)),
    (lambda x: 0 * x + 1.0, 1, 1),
]


def vscaleTester(fun, n, vscale):
    ff = Chebtech2.initfun_fixedlen(fun, n)

    def tester(self):
        absdiff = abs(ff.vscale - vscale)
        self.assertLessEqual(absdiff, 0.1 * vscale)

    return tester


for k, args in enumerate(vscales):
    _testfun_ = vscaleTester(*args)
    _testfun_.__name__ = "test_vscale_{:02}".format(k)
    setattr(ClassUsage, _testfun_.__name__, _testfun_)


plt = import_plt()


class Plotting(unittest.TestCase):
    """Unit-tests for Chebtech2 plotting methods"""

    def setUp(self):
        def f(x):
            return sin(3 * x) + 5e-1 * cos(30 * x)

        def u(x):
            return np.exp(2 * np.pi * 1j * x)

        self.f0 = Chebtech2.initfun_fixedlen(f, 100)
        self.f1 = Chebtech2.initfun_adaptive(f)
        self.f2 = Chebtech2.initfun_adaptive(u)

    @unittest.skipIf(plt is None, "matplotlib not installed")
    def test_plot(self):
        fig, ax = plt.subplots()
        self.f0.plot(ax=ax)

    @unittest.skipIf(plt is None, "matplotlib not installed")
    def test_plot_complex(self):
        fig, ax = plt.subplots()
        # plot Bernstein ellipses
        for rho in np.arange(1.1, 2, 0.1):
            joukowsky(rho * self.f2).plot(ax=ax)

    @unittest.skipIf(plt is None, "matplotlib not installed")
    def test_plotcoeffs(self):
        fig, ax = plt.subplots()
        self.f0.plotcoeffs(ax=ax)
        self.f1.plotcoeffs(ax=ax, color="r")


class Calculus(unittest.TestCase):
    """Unit-tests for Chebtech2 calculus operations"""

    def setUp(self):
        self.emptyfun = Chebtech2(np.array([]))

    # tests for the correct results in the empty cases
    def test_sum_empty(self):
        self.assertEqual(self.emptyfun.sum(), 0)

    def test_cumsum_empty(self):
        self.assertTrue(self.emptyfun.cumsum().isempty)

    def test_diff_empty(self):
        self.assertTrue(self.emptyfun.diff().isempty)


class Complex(unittest.TestCase):
    def setUp(self):
        self.z = Chebtech2.initfun_adaptive(lambda x: np.exp(np.pi * 1j * x))

    def test_init_empty(self):
        Chebtech2.initempty()

    def test_roots(self):
        r0 = self.z.roots()
        r1 = (self.z - 1).roots()
        r2 = (self.z - 1j).roots()
        r3 = (self.z + 1).roots()
        r4 = (self.z + 1j).roots()
        self.assertEqual(r0.size, 0)
        self.assertTrue(np.allclose(r1, [0]))
        self.assertTrue(np.allclose(r2, [0.5]))
        self.assertTrue(np.allclose(r3, [-1, 1]))
        self.assertTrue(np.allclose(r4, [-0.5]))

    def test_rho_ellipse_construction(self):
        zz = 1.2 * self.z
        e = 0.5 * (zz + 1 / zz)
        self.assertAlmostEqual(e(1) - e(-1), 0, places=14)
        self.assertAlmostEqual(e(0) + e(-1), 0, places=14)
        self.assertAlmostEqual(e(0) + e(1), 0, places=14)

    def test_calculus(self):
        self.assertTrue(np.allclose([self.z.sum()], [0]))
        self.assertTrue((self.z.cumsum().diff() - self.z).size, 1)
        self.assertTrue((self.z - self.z.cumsum().diff()).size, 1)

    def test_real_imag(self):
        # ceck definition of real and imaginary
        zreal = self.z.real()
        zimag = self.z.imag()
        np.testing.assert_equal(zreal.coeffs, np.real(self.z.coeffs))
        np.testing.assert_equal(zimag.coeffs, np.imag(self.z.coeffs))
        # check real part of real chebtech is the same chebtech
        self.assertTrue(zreal.real() == zreal)
        # check imaginary part of real chebtech is the zero chebtech
        self.assertTrue(zreal.imag().isconst)
        self.assertTrue(zreal.imag().coeffs[0] == 0)


# --------------------------------------
#           definite integrals
# --------------------------------------
def_integrals = [
    # (function, number of points, integral, tolerance)
    (lambda x: sin(x), 14, 0.0, eps),
    (lambda x: sin(4 * pi * x), 40, 0.0, 1e1 * eps),
    (lambda x: cos(x), 15, 1.682941969615793, 2 * eps),
    (lambda x: cos(4 * pi * x), 39, 0.0, 2 * eps),
    (lambda x: exp(cos(4 * pi * x)), 182, 2.532131755504016, 4 * eps),
    (lambda x: cos(3244 * x), 3389, 5.879599674161602e-04, 5e2 * eps),
    (lambda x: exp(x), 15, exp(1) - exp(-1), 2 * eps),
    (lambda x: 1e10 * exp(x), 15, 1e10 * (exp(1) - exp(-1)), 4e10 * eps),
    (lambda x: 0 * x + 1.0, 1, 2, eps),
]


def definiteIntegralTester(fun, n, integral, tol):
    ff = Chebtech2.initfun_fixedlen(fun, n)

    def tester(self):
        absdiff = abs(ff.sum() - integral)
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
    # (function, indefinite integral, number of points, tolerance)
    (lambda x: 0 * x + 1.0, lambda x: x, 1, eps),
    (lambda x: x, lambda x: 1 / 2 * x ** 2, 2, 2 * eps),
    (lambda x: x ** 2, lambda x: 1 / 3 * x ** 3, 3, 2 * eps),
    (lambda x: x ** 3, lambda x: 1 / 4 * x ** 4, 4, 2 * eps),
    (lambda x: x ** 4, lambda x: 1 / 5 * x ** 5, 5, 2 * eps),
    (lambda x: x ** 5, lambda x: 1 / 6 * x ** 6, 6, 4 * eps),
    (lambda x: sin(x), lambda x: -cos(x), 16, 2 * eps),
    (lambda x: cos(3 * x), lambda x: 1.0 / 3 * sin(3 * x), 23, 2 * eps),
    (lambda x: exp(x), lambda x: exp(x), 16, 3 * eps),
    (lambda x: 1e10 * exp(x), lambda x: 1e10 * exp(x), 16, 1e10 * (3 * eps)),
]


def indefiniteIntegralTester(fun, dfn, n, tol):
    ff = Chebtech2.initfun_fixedlen(fun, n)
    gg = Chebtech2.initfun_fixedlen(dfn, n + 1)
    coeffs = gg.coeffs
    coeffs[0] = coeffs[0] - dfn(np.array([-1]))

    def tester(self):
        absdiff = infnorm(ff.cumsum().coeffs - coeffs)
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
    # (function, derivative, number of points, tolerance)
    (lambda x: 0 * x + 1.0, lambda x: 0 * x + 0, 1, eps),
    (lambda x: x, lambda x: 0 * x + 1, 2, 2 * eps),
    (lambda x: x ** 2, lambda x: 2 * x, 3, 2 * eps),
    (lambda x: x ** 3, lambda x: 3 * x ** 2, 4, 2 * eps),
    (lambda x: x ** 4, lambda x: 4 * x ** 3, 5, 3 * eps),
    (lambda x: x ** 5, lambda x: 5 * x ** 4, 6, 4 * eps),
    (lambda x: sin(x), lambda x: cos(x), 16, 5e1 * eps),
    (lambda x: cos(3 * x), lambda x: -3 * sin(3 * x), 23, 5e2 * eps),
    (lambda x: exp(x), lambda x: exp(x), 16, 2e2 * eps),
    (lambda x: 1e10 * exp(x), lambda x: 1e10 * exp(x), 16, 1e10 * 2e2 * eps),
]


def derivativeTester(fun, der, n, tol):
    ff = Chebtech2.initfun_fixedlen(fun, n)
    gg = Chebtech2.initfun_fixedlen(der, max(n - 1, 1))

    def tester(self):
        absdiff = infnorm(ff.diff().coeffs - gg.coeffs)
        self.assertLessEqual(absdiff, tol)

    return tester


for k, (fun, der, n, tol) in enumerate(derivatives):
    _testfun_ = derivativeTester(fun, der, n, tol)
    _testfun_.__name__ = "test_diff_{:02}".format(k)
    setattr(Calculus, _testfun_.__name__, _testfun_)


class Construction(unittest.TestCase):
    """Unit-tests for construction of Chebtech2 objects"""

    # TODO: expand to all the constructor variants
    def test_initvalues(self):
        # test n = 0 case separately
        vals = np.random.rand(0)
        fun = Chebtech2.initvalues(vals)
        cfs = Chebtech2._vals2coeffs(vals)
        self.assertTrue(fun.coeffs.size == cfs.size == 0)
        # now test the other cases
        for n in range(1, 10):
            vals = np.random.rand(n)
            fun = Chebtech2.initvalues(vals)
            cfs = Chebtech2._vals2coeffs(vals)
            self.assertEqual(infnorm(fun.coeffs - cfs), 0.0)

    def test_initidentity(self):
        x = Chebtech2.initidentity()
        s = -1 + 2 * np.random.rand(10000)
        self.assertEqual(infnorm(s - x(s)), 0.0)

    def test_coeff_construction(self):
        coeffs = np.random.rand(10)
        f = Chebtech2(coeffs)
        self.assertIsInstance(f, Chebtech2)
        self.assertLess(infnorm(f.coeffs - coeffs), eps)

    def test_const_construction(self):
        ff = Chebtech2.initconst(1.0)
        self.assertEquals(ff.size, 1)
        self.assertTrue(ff.isconst)
        self.assertFalse(ff.isempty)
        self.assertRaises(ValueError, Chebtech2.initconst, [1.0])

    def test_empty_construction(self):
        ff = Chebtech2.initempty()
        self.assertEquals(ff.size, 0)
        self.assertFalse(ff.isconst)
        self.assertTrue(ff.isempty)
        self.assertRaises(TypeError, Chebtech2.initempty, [1.0])


def adaptiveTester(fun, funlen):
    ff = Chebtech2.initfun_adaptive(fun)

    def tester(self):
        self.assertLessEqual(ff.size - funlen, 2)
        self.assertGreater(ff.size - funlen, -1)

    return tester


def fixedlenTester(fun, n):
    ff = Chebtech2.initfun_fixedlen(fun, n)

    def tester(self):
        self.assertEquals(ff.size, n)

    return tester


for (fun, funlen, _) in testfunctions:

    # add the adaptive tests
    _testfun_ = adaptiveTester(fun, funlen)
    _testfun_.__name__ = "test_adaptive_{}".format(fun.__name__)
    setattr(Construction, _testfun_.__name__, _testfun_)

    # add the fixedlen tests
    for n in np.array([50, 500]):
        _testfun_ = fixedlenTester(fun, n)
        _testfun_.__name__ = "test_fixedlen_{}_{:003}pts".format(fun.__name__, n)
        setattr(Construction, _testfun_.__name__, _testfun_)


class Algebra(unittest.TestCase):
    """Unit-tests for Chebtech2 algebraic operations"""

    def setUp(self):
        self.xx = -1 + 2 * np.random.rand(1000)
        self.emptyfun = Chebtech2.initempty()

    # check (empty Chebtech) + (Chebtech) = (empty Chebtech)
    #   and (Chebtech) + (empty Chebtech) = (empty Chebtech)
    def test__add__radd__empty(self):
        for (fun, funlen, _) in testfunctions:
            chebtech = Chebtech2.initfun_fixedlen(fun, funlen)
            self.assertTrue((self.emptyfun + chebtech).isempty)
            self.assertTrue((chebtech + self.emptyfun).isempty)

    # check the output of (constant + Chebtech)
    #                 and (Chebtech + constant)
    def test__add__radd__constant(self):
        xx = self.xx
        for (fun, funlen, _) in testfunctions:
            for const in (-1, 1, 10, -1e5):

                def f(x):
                    return const + fun(x)

                techfun = Chebtech2.initfun_fixedlen(fun, funlen)
                f1 = const + techfun
                f2 = techfun + const
                tol = 5e1 * eps * abs(const)
                self.assertLessEqual(infnorm(f(xx) - f1(xx)), tol)
                self.assertLessEqual(infnorm(f(xx) - f2(xx)), tol)

    # check the output of (Chebtech - Chebtech)
    def test__add__negself(self):
        xx = self.xx
        for (fun, funlen, _) in testfunctions:
            chebtech = Chebtech2.initfun_fixedlen(fun, funlen)
            chebzero = chebtech - chebtech
            self.assertTrue(chebzero.isconst)
            self.assertEqual(infnorm(chebzero(xx)), 0)

    # check (empty Chebtech) - (Chebtech) = (empty Chebtech)
    #   and (Chebtech) - (empty Chebtech) = (empty Chebtech)
    def test__sub__rsub__empty(self):
        for (fun, funlen, _) in testfunctions:
            chebtech = Chebtech2.initfun_fixedlen(fun, funlen)
            self.assertTrue((self.emptyfun - chebtech).isempty)
            self.assertTrue((chebtech - self.emptyfun).isempty)

    # check the output of constant - Chebtech
    #                 and Chebtech - constant
    def test__sub__rsub__constant(self):
        xx = self.xx
        for (fun, funlen, _) in testfunctions:
            for const in (-1, 1, 10, -1e5):

                def f(x):
                    return const - fun(x)

                def g(x):
                    return fun(x) - const

                techfun = Chebtech2.initfun_fixedlen(fun, funlen)
                ff = const - techfun
                gg = techfun - const
                tol = 5e1 * eps * abs(const)
                self.assertLessEqual(infnorm(f(xx) - ff(xx)), tol)
                self.assertLessEqual(infnorm(g(xx) - gg(xx)), tol)

    # check (empty Chebtech) * (Chebtech) = (empty Chebtech)
    #   and (Chebtech) * (empty Chebtech) = (empty Chebtech)
    def test__mul__rmul__empty(self):
        for (fun, funlen, _) in testfunctions:
            chebtech = Chebtech2.initfun_fixedlen(fun, funlen)
            self.assertTrue((self.emptyfun * chebtech).isempty)
            self.assertTrue((chebtech * self.emptyfun).isempty)

    # check the output of constant * Chebtech
    #                 and Chebtech * constant
    def test__mul__rmul__constant(self):
        xx = self.xx
        for (fun, funlen, _) in testfunctions:
            for const in (-1, 1, 10, -1e5):

                def f(x):
                    return const * fun(x)

                def g(x):
                    return fun(x) * const

                techfun = Chebtech2.initfun_fixedlen(fun, funlen)
                ff = const * techfun
                gg = techfun * const
                tol = 5e1 * eps * abs(const)
                self.assertLessEqual(infnorm(f(xx) - ff(xx)), tol)
                self.assertLessEqual(infnorm(g(xx) - gg(xx)), tol)

    # check (empty Chebtech) / (Chebtech) = (empty Chebtech)
    #   and (Chebtech) / (empty Chebtech) = (empty Chebtech)
    def test_truediv_empty(self):
        for (fun, funlen, _) in testfunctions:
            chebtech = Chebtech2.initfun_fixedlen(fun, funlen)
            self.assertTrue(operator.truediv(self.emptyfun, chebtech).isempty)
            self.assertTrue(operator.truediv(chebtech, self.emptyfun).isempty)
            # __truediv__
            self.assertTrue((self.emptyfun / chebtech).isempty)
            self.assertTrue((chebtech / self.emptyfun).isempty)

    # check the output of constant / Chebtech
    #                 and Chebtech / constant
    # this tests truediv, __rdiv__, __truediv__, __rtruediv__, since
    # from __future__ import division is executed at the top of the file
    # TODO: find a way to test truediv and  __truediv__ genuinely separately
    def test_truediv_constant(self):
        xx = self.xx
        for (fun, funlen, hasRoots) in testfunctions:
            for const in (-1, 1, 10, -1e5):

                def f(x):
                    return const / fun(x)

                def g(x):
                    return fun(x) / const

                tol = eps * abs(const)
                techfun = Chebtech2.initfun_fixedlen(fun, funlen)
                gg = techfun / const
                self.assertLessEqual(infnorm(g(xx) - gg(xx)), 2 * gg.size * tol)
                # don't do the following test for functions with roots
                if not hasRoots:
                    ff = const / techfun
                    self.assertLessEqual(infnorm(f(xx) - ff(xx)), 3 * ff.size * tol)

    # check    +(empty Chebtech) = (empty Chebtech)
    def test__pos__empty(self):
        self.assertTrue((+self.emptyfun).isempty)

    # check -(empty Chebtech) = (empty Chebtech)
    def test__neg__empty(self):
        self.assertTrue((-self.emptyfun).isempty)

    def test_pow_empty(self):
        for c in range(10):
            self.assertTrue((self.emptyfun ** c).isempty)

    def test_rpow_empty(self):
        for c in range(10):
            self.assertTrue((c ** self.emptyfun).isempty)

    # check the output of Chebtech ** constant
    #                 and constant ** Chebtech
    def test_pow_const(self):
        xx = self.xx
        for (fun, funlen) in [(np.sin, 15), (np.exp, 15)]:
            for c in (1, 2):

                def f(x):
                    return fun(x) ** c

                techfun = Chebtech2.initfun_fixedlen(fun, funlen)
                ff = techfun ** c
                tol = 2e1 * eps * abs(c)
                self.assertLessEqual(infnorm(f(xx) - ff(xx)), tol)

    def test_rpow_const(self):
        xx = self.xx
        for (fun, funlen) in [(np.sin, 15), (np.exp, 15)]:
            for c in (1, 2):

                def g(x):
                    return c ** fun(x)

                techfun = Chebtech2.initfun_fixedlen(fun, funlen)
                gg = c ** techfun
                tol = 2e1 * eps * abs(c)
                self.assertLessEqual(infnorm(g(xx) - gg(xx)), tol)


# add tests for the binary operators
def binaryOpTester(f, g, binop, nf, ng):
    ff = Chebtech2.initfun_fixedlen(f, nf)
    gg = Chebtech2.initfun_fixedlen(g, ng)

    def FG(x):
        return binop(f(x), g(x))

    fg = binop(ff, gg)

    def tester(self):
        vscl = max([ff.vscale, gg.vscale])
        lscl = max([ff.size, gg.size])
        self.assertLessEqual(infnorm(fg(self.xx) - FG(self.xx)), 3 * vscl * lscl * eps)
        if binop is operator.mul:
            # check simplify is not being called in __mul__
            self.assertEqual(fg.size, ff.size + gg.size - 1)

    return tester


# note: defining __radd__(a,b) = operator.add(b,a) and feeding this into the
# test will not in fact test the __radd__ functionality of the class. These
# test need to be added manually to the class.
binops = (operator.add, operator.mul, operator.sub, operator.truediv)
for binop in binops:
    # add generic binary operator tests
    for (f, nf, _), (g, ng, denomRoots) in itertools.combinations(testfunctions, 2):
        if binop is operator.truediv and denomRoots:
            # skip truediv test if the denominator has roots
            pass
        else:
            _testfun_ = binaryOpTester(f, g, binop, nf, ng)
            _testfun_.__name__ = "test_{}_{}_{}".format(
                binop.__name__, f.__name__, g.__name__
            )
            setattr(Algebra, _testfun_.__name__, _testfun_)

powtestfuns = (
    [(np.exp, 15, "exp"), (np.sin, 15, "sin")],
    [(np.exp, 15, "exp"), (lambda x: 2 - x, 2, "linear")],
    [(lambda x: 2 - x, 2, "linear"), (np.exp, 15, "exp")],
)
# add operator.power tests
for (f, nf, namef), (g, ng, nameg) in powtestfuns:
    _testfun_ = binaryOpTester(f, g, operator.pow, nf, ng)
    _testfun_.__name__ = "test_{}_{}_{}".format("pow", namef, nameg)
    setattr(Algebra, _testfun_.__name__, _testfun_)


# add tests for the unary operators
def unaryOpTester(unaryop, f, nf):
    ff = Chebtech2.initfun_fixedlen(f, nf)

    def gg(x):
        return unaryop(f(x))

    GG = unaryop(ff)

    def tester(self):
        self.assertLessEqual(infnorm(gg(self.xx) - GG(self.xx)), 4e1 * eps)

    return tester


unaryops = (operator.pos, operator.neg)
for unaryop in unaryops:
    for (f, nf, _) in testfunctions:
        _testfun_ = unaryOpTester(unaryop, f, nf)
        _testfun_.__name__ = "test_{}_{}".format(unaryop.__name__, f.__name__)
        setattr(Algebra, _testfun_.__name__, _testfun_)


class Roots(unittest.TestCase):
    def test_empty(self):
        ff = Chebtech2.initempty()
        self.assertEquals(ff.roots().size, 0)

    def test_const(self):
        ff = Chebtech2.initconst(0.0)
        gg = Chebtech2.initconst(2.0)
        self.assertEquals(ff.roots().size, 0)
        self.assertEquals(gg.roots().size, 0)


# add tests for roots
def rootsTester(f, roots, tol):
    ff = Chebtech2.initfun_adaptive(f)
    rts = ff.roots()

    def tester(self):
        self.assertLessEqual(infnorm(rts - roots), tol)

    return tester


rootstestfuns = (
    (lambda x: 3 * x + 2.0, np.array([-2 / 3]), 1 * eps),
    (lambda x: x ** 2, np.array([0.0, 0.0]), 1 * eps),
    (lambda x: x ** 2 + 0.2 * x - 0.08, np.array([-0.4, 0.2]), 1 * eps),
    (lambda x: sin(x), np.array([0]), 1 * eps),
    (lambda x: cos(2 * pi * x), np.array([-0.75, -0.25, 0.25, 0.75]), 1 * eps),
    (lambda x: sin(100 * pi * x), np.linspace(-1, 1, 201), 1 * eps),
    (lambda x: sin(5 * pi / 2 * x), np.array([-0.8, -0.4, 0, 0.4, 0.8]), 1 * eps),
)

for k, args in enumerate(rootstestfuns):
    _testfun_ = rootsTester(*args)
    _testfun_.__name__ = "test_roots_{}".format(k)
    setattr(Roots, _testfun_.__name__, _testfun_)

# reset the testsfun variable so it doesn't get picked up by nose
_testfun_ = None
