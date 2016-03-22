# -*- coding: utf-8 -*-
"""
Unit-tests for pyfun/chebtech.py 
"""
from __future__ import division

from unittest import TestCase

from numpy import arange
from numpy import array
from numpy import all as all_
from numpy import diff
from numpy.random import rand
from numpy.random import seed

from pyfun.chebtech import ChebTech2
from pyfun.chebtech import eps

from utilities import infnorm
from utilities import scaled_tol
from utilities import infNormLessThanTol

seed(0)

# staticmethod aliases
chebpts      = ChebTech2.chebpts
_vals2coeffs = ChebTech2._vals2coeffs
_coeffs2vals = ChebTech2._coeffs2vals

# ------------------------
class ChebyshevPoints(TestCase):
    """Unit-tests for ChebTech2"""

    def test_chebpts_0(self):
        self.assertEquals(ChebTech2.chebpts(0).size, 0)
            
    def test_vals2coeffs_empty(self):
        self.assertEquals(_vals2coeffs(array([])).size, 0)

    def test_coeffs2vals_empty(self):
        self.assertEquals(_coeffs2vals(array([])).size, 0)

    # check we are returned the array for an array of size 1
    def test_vals2coeffs_size1(self):
        for k in arange(10):
            fk = array([k])
            self.assertLessEqual(infnorm(_vals2coeffs(fk)-fk), eps)

    # check we are returned the array for an array of size 1
    def test_coeffs2vals_size1(self):
        for k in arange(10):
            ak = array([k])
            self.assertLessEqual(infnorm(_coeffs2vals(ak)-ak), eps)

    # TODO: further checks for chepbts

# ------------------------------------------------------------------------
# Tests to verify the mutually inverse nature of vals2coeffs and coeffs2vals
# ------------------------------------------------------------------------
def vals2coeffs2valsTester(n):
    def asserter(self):
        values = rand(n)
        coeffs = _vals2coeffs(values)
        _values_ = _coeffs2vals(coeffs)
        self.assertLessEqual( infnorm(values-_values_), scaled_tol(n) )
    return asserter

def coeffs2vals2coeffsTester(n):
    def asserter(self):
        coeffs = rand(n)
        values = _coeffs2vals(coeffs)
        _coeffs_ = _vals2coeffs(values)
        self.assertLessEqual( infnorm(coeffs-_coeffs_), scaled_tol(n) )
    return asserter

for k, n in enumerate(2**arange(2,18,2)):

    # vals2coeffs2vals
    testfun = vals2coeffs2valsTester(n)
    testfun.__name__ = "test_vals2coeffs2vals_{:02}".format(k)
    setattr(ChebyshevPoints, testfun.__name__, testfun)

    # coeffs2vals2coeffs
    testfun = coeffs2vals2coeffsTester(n)
    testfun.__name__ = "test_coeffs2vals2coeffs_{:02}".format(k)
    setattr(ChebyshevPoints, testfun.__name__, testfun)
# ------------------------------------------------------------------------
   
# ------------------------------------------------------------------------
# Add second-kind Chebyshev points test cases to ChebyshevPoints
# ------------------------------------------------------------------------
chebpts2_testlist = (
    (ChebTech2.chebpts(1), array([0.]), eps),
    (ChebTech2.chebpts(2), array([-1., 1.]), eps),
    (ChebTech2.chebpts(3), array([-1., 0., 1.]), eps),
    (ChebTech2.chebpts(4), array([-1., -.5, .5, 1.]), 2*eps),
    (ChebTech2.chebpts(5), array([-1., -2.**(-.5), 0., 2.**(-.5), 1.]), eps),
)
for k, (a,b,tol) in enumerate(chebpts2_testlist):
    testfun = infNormLessThanTol(a,b,tol)
    testfun.__name__ = "test_chebpts_{:02}".format(k+1)
    setattr(ChebyshevPoints, testfun.__name__, testfun)

# check the output is of the right length, that the endpoint values are -1 
# and 1, respectively, and that the sequence is monotonically increasing
def chebptsLenTester(k):
    def asserter(self):
        pts = ChebTech2.chebpts(k)
        self.assertEquals(pts.size, k)
        self.assertEquals(pts[0], -1.)
        self.assertEquals(pts[-1], 1.)
        self.assertTrue( all_(diff(pts)) > 0 )
    return asserter
    
for k, n in enumerate(2**arange(2,18,2)):
    testfun = chebptsLenTester(n+3)
    testfun.__name__ = "test_chebpts_len_{:02}".format(k)
    setattr(ChebyshevPoints, testfun.__name__, testfun)
# ------------------------------------------------------------------------

# reset the testsfun variable so it doesn't get picked up by nose
testfun = None