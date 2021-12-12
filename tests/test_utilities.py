"""Unit-tests for pyfun/core/utilities.py"""

import unittest

import numpy as np

from chebpy import chebfun
from chebpy.core.bndfun import Bndfun
from chebpy.core.settings import DefaultPreferences
from chebpy.core.utilities import HTOL, Interval, Domain, compute_breakdata, check_funs
from chebpy.core.exceptions import (
    IntervalGap,
    IntervalOverlap,
    IntervalValues,
    InvalidDomain,
    SupportMismatch,
    NotSubdomain,
)

from .utilities import infnorm


np.random.seed(0)
eps = DefaultPreferences.eps
HTOL = HTOL()


# tests for usage of the Interval class
class TestInterval(unittest.TestCase):
    def setUp(self):
        self.i1 = Interval(-2, 3)
        self.i2 = Interval(-2, 3)
        self.i3 = Interval(-1, 1)
        self.i4 = Interval(-1, 2)

    def test_init(self):
        Interval(-1, 1)
        self.assertTrue((np.asarray(Interval()) == np.array([-1, 1])).all())

    def test_init_disallow(self):
        self.assertRaises(IntervalValues, Interval, 2, 0)
        self.assertRaises(IntervalValues, Interval, 0, 0)

    def test__eq__(self):
        self.assertTrue(Interval() == Interval())
        self.assertTrue(self.i1 == self.i2)
        self.assertTrue(self.i2 == self.i1)
        self.assertFalse(self.i3 == self.i1)
        self.assertFalse(self.i2 == self.i3)

    def test__ne__(self):
        self.assertFalse(Interval() != Interval())
        self.assertFalse(self.i1 != self.i2)
        self.assertFalse(self.i2 != self.i1)
        self.assertTrue(self.i3 != self.i1)
        self.assertTrue(self.i2 != self.i3)

    def test__contains__(self):
        self.assertTrue(self.i1 in self.i2)
        self.assertTrue(self.i3 in self.i1)
        self.assertTrue(self.i4 in self.i1)
        self.assertFalse(self.i1 in self.i3)
        self.assertFalse(self.i1 in self.i4)
        self.assertFalse(self.i1 not in self.i2)
        self.assertFalse(self.i3 not in self.i1)
        self.assertFalse(self.i4 not in self.i1)
        self.assertTrue(self.i1 not in self.i3)
        self.assertTrue(self.i1 not in self.i4)

    # Interval objects used to have tolerance-sensitive definitions of __eq__ and
    # __contains__, though these were removed in the commit following
    # 9eaf1c5e0674dab1a676d04a02ceda329beec2ea.
    #    def test__eq__close(self):
    #        tol = .8*HTOL
    #        i4 = Interval(-2,5)
    #        i5 = Interval(-2*(1+tol),5*(1-tol))
    #        i6 = Interval(-2*(1+2*tol),5*(1-2*tol))
    #        self.assertEqual(i4,i5)
    #        self.assertNotEqual(i4,i6)

    #    def test__contains__close(self):
    #        tol = .8*HTOL
    #        i1 = Interval(-1,2)
    #        i2 = Interval(-1-tol,2+2*tol)
    #        i3 = Interval(-1-2*tol,2+4*tol)
    #        self.assertTrue(i1 in i2)
    #        self.assertTrue(i2 in i1)
    #        self.assertFalse(i3 in i1)

    def test_maps(self):
        yy = -1 + 2 * np.random.rand(1000)
        interval = Interval(-2, 3)
        vals = interval.invmap(interval(yy)) - yy
        self.assertLessEqual(infnorm(vals), eps)

    def test_isinterior(self):
        npts = 1000
        x1 = np.linspace(-2, 3, npts)
        x2 = np.linspace(-3, -2, npts)
        x3 = np.linspace(3, 4, npts)
        x4 = np.linspace(5, 6, npts)
        interval = Interval(-2, 3)
        self.assertEquals(interval.isinterior(x1).sum(), npts - 2)
        self.assertEquals(interval.isinterior(x2).sum(), 0)
        self.assertEquals(interval.isinterior(x3).sum(), 0)
        self.assertEquals(interval.isinterior(x4).sum(), 0)


# tests for usage of the Domain class
class TestDomain(unittest.TestCase):
    def test__init__(self):
        Domain([-2, 1])
        Domain([-2, 0, 1])
        Domain(np.array([-2, 1]))
        Domain(np.array([-2, 0, 1]))
        Domain(np.linspace(-10, 10, 51))

    def test__init__disallow(self):
        self.assertRaises(InvalidDomain, Domain, [1])
        self.assertRaises(InvalidDomain, Domain, [1, -1])
        self.assertRaises(InvalidDomain, Domain, [-1, 0, 0])
        self.assertRaises(ValueError, Domain, ["a", "b"])

    def test__iter__(self):
        dom_a = Domain([-2, 1])
        dom_b = Domain([-2, 0, 1])
        dom_c = Domain([-1, 0, 1, 2])
        res_a = (-2, 1)
        res_b = (-2, 0, 1)
        res_c = (-1, 0, 1, 2)
        self.assertTrue(all([x == y for x, y in zip(dom_a, res_a)]))
        self.assertTrue(all([x == y for x, y in zip(dom_b, res_b)]))
        self.assertTrue(all([x == y for x, y in zip(dom_c, res_c)]))

    def test_intervals(self):
        dom_a = Domain([-2, 1])
        dom_b = Domain([-2, 0, 1])
        dom_c = Domain([-1, 0, 1, 2])
        res_a = [(-2, 1)]
        res_b = [(-2, 0), (0, 1)]
        res_c = [(-1, 0), (0, 1), (1, 2)]
        self.assertTrue(
            all(
                [itvl == Interval(a, b) for itvl, (a, b) in zip(dom_a.intervals, res_a)]
            )
        )
        self.assertTrue(
            all(
                [itvl == Interval(a, b) for itvl, (a, b) in zip(dom_b.intervals, res_b)]
            )
        )
        self.assertTrue(
            all(
                [itvl == Interval(a, b) for itvl, (a, b) in zip(dom_c.intervals, res_c)]
            )
        )

    def test__contains__(self):
        d1 = Domain([-2, 0, 1, 3, 5])
        d2 = Domain([-1, 2])
        d3 = Domain(np.linspace(-10, 10, 1000))
        d4 = Domain([-1, 0, 1, 2])
        self.assertTrue(d2 in d1)
        self.assertTrue(d1 in d3)
        self.assertTrue(d2 in d3)
        self.assertTrue(d2 in d3)
        self.assertTrue(d2 in d4)
        self.assertTrue(d4 in d2)
        self.assertFalse(d1 in d2)
        self.assertFalse(d3 in d1)
        self.assertFalse(d3 in d2)

    def test__contains__close(self):
        tol = 0.8 * HTOL
        d1 = Domain([-1, 2])
        d2 = Domain([-1 - tol, 2 + 2 * tol])
        d3 = Domain([-1 - 2 * tol, 2 + 4 * tol])
        self.assertTrue(d1 in d2)
        self.assertTrue(d2 in d1)
        self.assertFalse(d3 in d1)

    def test__eq__(self):
        d1 = Domain([-2, 0, 1, 3, 5])
        d2 = Domain([-2, 0, 1, 3, 5])
        d3 = Domain([-1, 1])
        self.assertEqual(d1, d2)
        self.assertNotEqual(d1, d3)

    def test__eq___result_type(self):
        d1 = Domain([-2, 0, 1, 3, 5])
        d2 = Domain([-2, 0, 1, 3, 5])
        d3 = Domain([-1, 1])
        self.assertIsInstance(d1 == d2, bool)
        self.assertIsInstance(d1 == d3, bool)

    def test__eq__close(self):
        tol = 0.8 * HTOL
        d4 = Domain([-2, 0, 1, 3, 5])
        d5 = Domain([-2 * (1 + tol), 0 - tol, 1 + tol, 3 * (1 + tol), 5 * (1 - tol)])
        d6 = Domain(
            [
                -2 * (1 + 2 * tol),
                0 - 2 * tol,
                1 + 2 * tol,
                3 * (1 + 2 * tol),
                5 * (1 - 2 * tol),
            ]
        )
        self.assertEqual(d4, d5)
        self.assertNotEqual(d4, d6)

    def test__ne__(self):
        d1 = Domain([-2, 0, 1, 3, 5])
        d2 = Domain([-2, 0, 1, 3, 5])
        d3 = Domain([-1, 1])
        self.assertFalse(d1 != d2)
        self.assertTrue(d1 != d3)

    def test__ne___result_type(self):
        d1 = Domain([-2, 0, 1, 3, 5])
        d2 = Domain([-2, 0, 1, 3, 5])
        d3 = Domain([-1, 1])
        self.assertIsInstance(d1 != d2, bool)
        self.assertIsInstance(d1 != d3, bool)

    def test_from_chebfun(self):
        ff = chebfun(lambda x: np.cos(x), np.linspace(-10, 10, 11))
        Domain.from_chebfun(ff)

    def test_breakpoints_in(self):
        d1 = Domain([-1, 0, 1])
        d2 = Domain([-2, 0.5, 1, 3])

        result1 = d1.breakpoints_in(d2)
        self.assertIsInstance(result1, np.ndarray)
        self.assertTrue(result1.size, 3)
        self.assertFalse(result1[0])
        self.assertFalse(result1[1])
        self.assertTrue(result1[2])

        result2 = d2.breakpoints_in(d1)
        self.assertIsInstance(result2, np.ndarray)
        self.assertTrue(result2.size, 4)
        self.assertFalse(result2[0])
        self.assertFalse(result2[1])
        self.assertTrue(result2[2])
        self.assertFalse(result2[3])

        self.assertTrue(d1.breakpoints_in(d1).all())
        self.assertTrue(d2.breakpoints_in(d2).all())
        self.assertFalse(d1.breakpoints_in(Domain([-5, 5])).any())
        self.assertFalse(d2.breakpoints_in(Domain([-5, 5])).any())

    def test_breakpoints_in_close(self):
        tol = 0.8 * HTOL
        d1 = Domain([-1, 0, 1])
        d2 = Domain([-2, 0 - tol, 1 + tol, 3])
        result = d1.breakpoints_in(d2)
        self.assertFalse(result[0])
        self.assertTrue(result[1])
        self.assertTrue(result[2])

    def test_support(self):
        dom_a = Domain([-2, 1])
        dom_b = Domain([-2, 0, 1])
        dom_c = Domain(np.linspace(-10, 10, 51))
        self.assertTrue(np.all(dom_a.support.view(np.ndarray) == [-2, 1]))
        self.assertTrue(np.all(dom_b.support.view(np.ndarray) == [-2, 1]))
        self.assertTrue(np.all(dom_c.support.view(np.ndarray) == [-10, 10]))

    def test_size(self):
        dom_a = Domain([-2, 1])
        dom_b = Domain([-2, 0, 1])
        dom_c = Domain(np.linspace(-10, 10, 51))
        self.assertEqual(dom_a.size, 2)
        self.assertEqual(dom_b.size, 3)
        self.assertEqual(dom_c.size, 51)

    def test_restrict(self):
        dom_a = Domain([-2, -1, 0, 1])
        dom_b = Domain([-1.5, -0.5, 0.5])
        dom_c = Domain(np.linspace(-2, 1, 16))
        self.assertEqual(dom_a.restrict(dom_b), Domain([-1.5, -1, -0.5, 0, 0.5]))
        self.assertEqual(dom_a.restrict(dom_c), dom_c)
        self.assertEqual(dom_a.restrict(dom_a), dom_a)
        self.assertEqual(dom_b.restrict(dom_b), dom_b)
        self.assertEqual(dom_c.restrict(dom_c), dom_c)
        # tests to check if catch breakpoints that are different by eps
        # (linspace introduces these effects)
        dom_d = Domain(np.linspace(-0.4, 0.4, 2))
        self.assertEqual(dom_c.restrict(dom_d), Domain([-0.4, -0.2, 0, 0.2, 0.4]))

    def test_restrict_raises(self):
        dom_a = Domain([-2, -1, 0, 1])
        dom_b = Domain([-1.5, -0.5, 0.5])
        dom_c = Domain(np.linspace(-2, 1, 16))
        self.assertRaises(NotSubdomain, dom_b.restrict, dom_a)
        self.assertRaises(NotSubdomain, dom_b.restrict, dom_c)

    def test_merge(self):
        dom_a = Domain([-2, -1, 0, 1])
        dom_b = Domain([-1.5, -0.5, 0.5])
        self.assertEqual(dom_b.merge(dom_a), Domain([-2, -1.5, -1, -0.5, 0, 0.5, 1]))

    def test_union(self):
        dom_a = Domain([-2, 0, 2])
        dom_b = Domain([-2, -1, 1, 2])
        self.assertNotEqual(dom_a.union(dom_b), dom_a)
        self.assertNotEqual(dom_a.union(dom_b), dom_b)
        self.assertEqual(dom_a.union(dom_b), Domain([-2, -1, 0, 1, 2]))
        self.assertEqual(dom_b.union(dom_a), Domain([-2, -1, 0, 1, 2]))

    def test_union_close(self):
        tol = 0.8 * HTOL
        dom_a = Domain([-2, 0, 2])
        dom_c = Domain([-2 - 2 * tol, -1 + tol, 1 + tol, 2 + 2 * tol])
        self.assertEqual(dom_a.union(dom_c), Domain([-2, -1, 0, 1, 2]))
        self.assertEqual(dom_c.union(dom_a), Domain([-2, -1, 0, 1, 2]))

    def test_union_raises(self):
        dom_a = Domain([-2, 0])
        dom_b = Domain([-2, 3])
        self.assertRaises(SupportMismatch, dom_a.union, dom_b)
        self.assertRaises(SupportMismatch, dom_b.union, dom_a)


class CheckFuns(unittest.TestCase):
    """Tests for the chebpy.core.utilities check_funs method"""

    def setUp(self):
        def f(x):
            return np.exp(x)

        self.fun0 = Bndfun.initfun_adaptive(f, Interval(-1, 0))
        self.fun1 = Bndfun.initfun_adaptive(f, Interval(0, 1))
        self.fun2 = Bndfun.initfun_adaptive(f, Interval(-0.5, 0.5))
        self.fun3 = Bndfun.initfun_adaptive(f, Interval(2, 2.5))
        self.fun4 = Bndfun.initfun_adaptive(f, Interval(-3, -2))
        self.funs_a = np.array([self.fun1, self.fun0, self.fun2])
        self.funs_b = np.array([self.fun1, self.fun2])
        self.funs_c = np.array([self.fun0, self.fun3])
        self.funs_d = np.array([self.fun1, self.fun4])

    def test_verify_empty(self):
        funs = check_funs(np.array([]))
        self.assertTrue(funs.size == 0)

    def test_verify_contiguous(self):
        funs = check_funs(np.array([self.fun0, self.fun1]))
        self.assertTrue(funs[0] == self.fun0)
        self.assertTrue(funs[1] == self.fun1)

    def test_verify_sort(self):
        funs = check_funs(np.array([self.fun1, self.fun0]))
        self.assertTrue(funs[0] == self.fun0)
        self.assertTrue(funs[1] == self.fun1)

    def test_verify_overlapping(self):
        self.assertRaises(IntervalOverlap, check_funs, self.funs_a)
        self.assertRaises(IntervalOverlap, check_funs, self.funs_b)

    def test_verify_gap(self):
        self.assertRaises(IntervalGap, check_funs, self.funs_c)
        self.assertRaises(IntervalGap, check_funs, self.funs_d)


# tests for the chebpy.core.utilities compute_breakdata function
class ComputeBreakdata(unittest.TestCase):
    def setUp(self):
        def f(x):
            return np.exp(x)

        self.fun0 = Bndfun.initfun_adaptive(f, Interval(-1, 0))
        self.fun1 = Bndfun.initfun_adaptive(f, Interval(0, 1))

    def test_compute_breakdata_empty(self):
        breaks = compute_breakdata(np.array([]))
        # list(...) for Python 2/3 compatibility
        self.assertTrue(np.array(list(breaks.items())).size == 0)

    def test_compute_breakdata_1(self):
        funs = np.array([self.fun0])
        breaks = compute_breakdata(funs)
        x, y = list(breaks.keys()), list(breaks.values())
        self.assertLessEqual(infnorm(x - np.array([-1, 0])), eps)
        self.assertLessEqual(infnorm(y - np.array([np.exp(-1), np.exp(0)])), 2 * eps)

    def test_compute_breakdata_2(self):
        funs = np.array([self.fun0, self.fun1])
        breaks = compute_breakdata(funs)
        x, y = list(breaks.keys()), list(breaks.values())
        self.assertLessEqual(infnorm(x - np.array([-1, 0, 1])), eps)
        self.assertLessEqual(
            infnorm(y - np.array([np.exp(-1), np.exp(0), np.exp(1)])), 2 * eps
        )


# reset the testsfun variable so it doesn't get picked up by nose
testfun = None
