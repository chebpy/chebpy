# -*- coding: utf-8 -*-
"""
Unit-tests for pyfun/utilities.py
"""
from __future__ import division

from unittest import TestCase

from numpy import array
from numpy import cos
from numpy import exp
from numpy import linspace
from numpy import ndarray
from numpy.random import rand
from numpy.random import seed

from chebpy.core.bndfun import Bndfun
from chebpy.core.settings import DefaultPrefs
from chebpy.core.utilities import Interval
from chebpy.core.utilities import Domain
from chebpy.core.utilities import compute_breakdata
from chebpy.core.utilities import check_funs
from chebpy.core.exceptions import IntervalGap
from chebpy.core.exceptions import IntervalOverlap
from chebpy.core.exceptions import IntervalValues
from chebpy.core.exceptions import InvalidDomain
from chebpy.core.exceptions import SupportMismatch

from chebpy import chebfun

from utilities import infnorm

seed(0)

eps = DefaultPrefs.eps


# tests for usage of the Interval class
class TestInterval(TestCase):

    def setUp(self):
        self.i1 = Interval(-2,3)
        self.i2 = Interval(-2,3)
        self.i3 = Interval(-1,1)
        self.i4 = Interval(-1,2)

    def test_init(self):
        Interval(-1,1)
        self.assertTrue((Interval().values==array([-1,1])).all())

    def test_init_disallow(self):
        self.assertRaises(IntervalValues, Interval, 2, 0)
        self.assertRaises(IntervalValues, Interval, 0, 0)

    def test__eq__(self):
        self.assertTrue(Interval()==Interval())
        self.assertTrue(self.i1==self.i2)
        self.assertTrue(self.i2==self.i1)
        self.assertFalse(self.i3==self.i1)
        self.assertFalse(self.i2==self.i3)

    def test__ne__(self):
        self.assertFalse(Interval()!=Interval())
        self.assertFalse(self.i1!=self.i2)
        self.assertFalse(self.i2!=self.i1)
        self.assertTrue(self.i3!=self.i1)
        self.assertTrue(self.i2!=self.i3)

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

    def test_maps(self):
        yy = -1 + 2 * rand(1000)
        interval = Interval(-2,3)
        vals = interval.invmap(interval(yy)) - yy
        self.assertLessEqual(infnorm(vals), eps)

    def test_isinterior(self):
        npts = 1000
        x1 = linspace(-2, 3,npts)
        x2 = linspace(-3,-2,npts)
        x3 = linspace(3,4,npts)
        x4 = linspace(5,6,npts)
        interval = Interval(-2,3)
        self.assertEquals(interval.isinterior(x1).sum(), npts-2)
        self.assertEquals(interval.isinterior(x2).sum(), 0)
        self.assertEquals(interval.isinterior(x3).sum(), 0)
        self.assertEquals(interval.isinterior(x4).sum(), 0)


# tests for usage of the Domain class
class TestDomain(TestCase):

    def test__init__(self):
        Domain([-2,1])
        Domain([-2,0,1])
        Domain(array([-2,1]))
        Domain(array([-2,0,1]))
        Domain(linspace(-10,10,51))

    def test__init__disallow(self):
        self.assertRaises(InvalidDomain, Domain, [1])
        self.assertRaises(InvalidDomain, Domain, [1, -1])
        self.assertRaises(InvalidDomain, Domain, [-1, 0, 0])
        self.assertRaises(InvalidDomain, Domain, ["a", "b"])

    def test__iter__(self):
        dom_a = Domain([-2,1])
        dom_b = Domain([-2,0,1])
        dom_c = Domain([-1,0,1,2])
        res_a = [(-2,1)]
        res_b = [(-2,0), (0,1)]
        res_c = [(-1,0), (0,1), (1,2)]
        self.assertTrue(all([x==y for x,y in zip(dom_a, res_a)]))
        self.assertTrue(all([x==y for x,y in zip(dom_b, res_b)]))
        self.assertTrue(all([x==y for x,y in zip(dom_c, res_c)]))

    def test__eq__(self):
        d1 = Domain([-2,0,1,3,5])
        d2 = Domain([-2,0,1,3,5])
        d3 = Domain([-1,1])
        self.assertTrue(d1==d2)
        self.assertFalse(d1==d3)

    def test__ne__(self):
        d1 = Domain([-2,0,1,3,5])
        d2 = Domain([-2,0,1,3,5])
        d3 = Domain([-1,1])
        self.assertFalse(d1!=d2)
        self.assertTrue(d1!=d3)

    def test_from_chebfun(self):
        ff = chebfun(lambda x: cos(x), linspace(-10,10,11))
        Domain.from_chebfun(ff)

    def test_breakpoints_in(self):
        d1 = Domain([-1,0,1])
        d2 = Domain([-2,0.5,1,3])

        result1 = d1.breakpoints_in(d2)
        self.assertIsInstance(result1, ndarray)
        self.assertTrue(result1.size, 3)
        self.assertFalse(result1[0])
        self.assertFalse(result1[1])
        self.assertTrue(result1[2])

        result2 = d2.breakpoints_in(d1)
        self.assertIsInstance(result2, ndarray)
        self.assertTrue(result2.size, 4)
        self.assertFalse(result2[0])
        self.assertFalse(result2[1])
        self.assertTrue(result2[2])
        self.assertFalse(result2[3])

        self.assertTrue(d1.breakpoints_in(d1).all())
        self.assertTrue(d2.breakpoints_in(d2).all())
        self.assertFalse(d1.breakpoints_in(Domain([-5,5])).any())
        self.assertFalse(d2.breakpoints_in(Domain([-5,5])).any())

    def test_support(self):
        dom_a = Domain([-2,1])
        dom_b = Domain([-2,0,1])
        dom_c = Domain(linspace(-10,10,51))
        self.assertTrue(all(dom_a.support==[-2,1]))
        self.assertTrue(all(dom_b.support==[-2,1]))
        self.assertTrue(all(dom_c.support==[-10,10]))

    def test_size(self):
        dom_a = Domain([-2,1])
        dom_b = Domain([-2,0,1])
        dom_c = Domain(linspace(-10,10,51))
        self.assertEqual(dom_a.size, 2)
        self.assertEqual(dom_b.size, 3)
        self.assertEqual(dom_c.size, 51)

    def test_union(self):
        dom_a = Domain([-2,0,2])
        dom_b = Domain([-2,-1,1,2])
        self.assertNotEqual(dom_a.union(dom_b), dom_a)
        self.assertNotEqual(dom_a.union(dom_b), dom_b)
        self.assertEqual(dom_a.union(dom_b), Domain([-2,-1,0,1,2]))

    def test_union_convert_type(self):
        dom_a = Domain([-2,0,2])
        dom_r = Domain([-2,-1,0,1,2])
        self.assertEqual(dom_a.union([-2,-1,0,1,2]), dom_r)

    def test_union_raises(self):
        dom_a = Domain([-2,0])
        dom_b = Domain([-2,3])
        self.assertRaises(SupportMismatch, dom_a.union, dom_b)


class CheckFuns(TestCase):
    """Tests for the chebpy.core.utilities check_funs method"""

    def setUp(self):
        f = lambda x: exp(x)
        self.fun0 = Bndfun.initfun_adaptive(f, Interval(-1,0))
        self.fun1 = Bndfun.initfun_adaptive(f, Interval(0,1))
        self.fun2 = Bndfun.initfun_adaptive(f, Interval(-.5,0.5))
        self.fun3 = Bndfun.initfun_adaptive(f, Interval(2,2.5))
        self.fun4 = Bndfun.initfun_adaptive(f, Interval(-3,-2))
        self.funs_a = array([self.fun1, self.fun0, self.fun2])
        self.funs_b = array([self.fun1, self.fun2])
        self.funs_c = array([self.fun0, self.fun3])
        self.funs_d = array([self.fun1, self.fun4])

    def test_verify_empty(self):
        funs = check_funs(array([]))
        self.assertTrue(funs.size==0)

    def test_verify_contiguous(self):
        funs = check_funs(array([self.fun0, self.fun1]))
        self.assertTrue(funs[0]==self.fun0)
        self.assertTrue(funs[1]==self.fun1)

    def test_verify_sort(self):
        funs = check_funs(array([self.fun1, self.fun0]))
        self.assertTrue(funs[0]==self.fun0)
        self.assertTrue(funs[1]==self.fun1)

    def test_verify_overlapping(self):
        self.assertRaises(IntervalOverlap, check_funs, self.funs_a)
        self.assertRaises(IntervalOverlap, check_funs, self.funs_b)

    def test_verify_gap(self):
        self.assertRaises(IntervalGap, check_funs, self.funs_c)
        self.assertRaises(IntervalGap, check_funs, self.funs_d)


# tests for the chebpy.core.utilities compute_breakdata function
class ComputeBreakdata(TestCase):

    def setUp(self):
        f = lambda x: exp(x)
        self.fun0 = Bndfun.initfun_adaptive(f, Interval(-1,0) )
        self.fun1 = Bndfun.initfun_adaptive(f, Interval(0,1) )

    def test_compute_breakdata_empty(self):
        breaks = compute_breakdata(array([]))
        self.assertTrue(array(breaks.items()).size==0)

    def test_compute_breakdata_1(self):
        funs = array([self.fun0])
        breaks = compute_breakdata(funs)
        x, y = breaks.keys(), breaks.values()
        self.assertLessEqual(infnorm(x-array([-1,0])), eps)
        self.assertLessEqual(infnorm(y-array([exp(-1),exp(0)])), 2*eps)

    def test_compute_breakdata_2(self):
        funs = array([self.fun0, self.fun1])
        breaks = compute_breakdata(funs)
        x, y = breaks.keys(), breaks.values()
        self.assertLessEqual(infnorm(x-array([-1,0,1])), eps)
        self.assertLessEqual(infnorm(y-array([exp(-1),exp(0),exp(1)])), 2*eps)

# reset the testsfun variable so it doesn't get picked up by nose
testfun = None
