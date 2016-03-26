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

# bespoke test generators
def infNormLessThanTol(a, b, tol):
    def asserter(self):
        self.assertLessEqual(infnorm(a-b), tol)
    return asserter

# test functions
testfunctions = []
fun_details = [
    # (function, name for the test printouts, Matlab chebfun adaptive degree)
    (lambda x: x**3 + x**2 + x + 1, "poly3(x)",  4),
    (lambda x: exp(x),              "exp(x)",   18),
    (lambda x: sin(x),              "sin(x)",   16),
    (lambda x: cos(20*x),           "cos(20x)", 53),
    (lambda x: 0.*x+1.,             "constfun",  1),
    (lambda x: 0.*x,                "zerofun",   1),
]
for k, item in enumerate(fun_details):
    fun = item[0]
    fun.__name__ = item[1]
    testfunctions.append((fun, item[2]))

# TODO: check these lengths against Chebfun
# TODO: more examples

