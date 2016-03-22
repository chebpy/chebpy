# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:12:43 2016

@author: mark.richardson
"""
from numpy import inf
from numpy import log
from numpy import finfo

from numpy.linalg import norm
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