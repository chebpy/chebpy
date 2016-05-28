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
    # (
    #  function,
    #  name for the test printouts,
    #  Matlab chebfun adaptive degree on [-1,1],
    #  Any roots on the real line?
    # )
    (lambda x: x**3 + x**2 + x + 1.1, "poly3(x)",        4, False),
    (lambda x: exp(x),                "exp(x)",         15, False),
    (lambda x: sin(x),                "sin(x)",         14, True),
    (lambda x: .2+.1*sin(x),          "(.2+.1*sin(x))", 14, False),
    (lambda x: cos(20*x),             "cos(20x)",       51, True),
    (lambda x: 0.*x+1.,               "constfun",        1, False),
    (lambda x: 0.*x,                  "zerofun",         1, True),
]
for k, items in enumerate(fun_details):
    fun = items[0]
    fun.__name__ = items[1]
    testfunctions.append((fun, items[2], items[3]))

# TODO: check these lengths against Chebfun
# TODO: more examples

