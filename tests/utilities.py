# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:12:43 2016

@author: mark.richardson
"""
from numpy import inf
from numpy import sin
from numpy import cos
from numpy import exp
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

# ------------------------

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
