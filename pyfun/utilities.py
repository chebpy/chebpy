
from __future__ import division

from functools import wraps
from warnings import warn

from numpy import finfo
from numpy import ones
from numpy import array
from numpy import asarray
from numpy import isscalar
from numpy import arange
from numpy import log
from numpy import log10
from numpy import linspace
from numpy import argmin
from numpy import any
from numpy import isnan
from numpy import NaN
from numpy import zeros
from numpy import dot
from numpy import where
from numpy import mod

eps = finfo(float).eps

# local helpers
def find(x):
    return where(x)[0]

# -------------------------------------
#              decorators
# -------------------------------------

# Factory method to produce a decorator that checks whether the object
# whose classmethod is being wrapped is empty, returning the object if
# so, but returning the supplied resultif if not. (Used in chebtech.py)
# TODO: add unit test for this
def checkempty(resultif=None):
    def decorator(f):
        @wraps(f)
        def wrapper(self, *args, **kwargs):
            if self.isempty():
                    if resultif is not None:
                        return resultif
                    else:
                        return self
            else:
                return f(self, *args, **kwargs)
        return wrapper
    return decorator


# check first if either of the first two arguments of the wrapped function
# are empty arrays, and if so then return an empty array. Although we use
# numpy.asarray to check the length of the obeject, we pass the input as is
# to the subsequent evaluation routines. (Wraps bary and clenshaw.)
def returnemptyarray(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if ( asarray(args[0]).size==0) | (asarray(args[1]).size==0 ):
            return array([])
        else:
            return f(*args, **kwargs)
    return wrapper

# -------------------------------------

@returnemptyarray
def bary(xx, fk, xk, vk):
    """Barycentric interpolation formula. See:

    J.P. Berrut, L.N. Trefethen, Barycentric Lagrange Interpolation, SIAM
    Review (2004)

    Inputs
    ------
    xx : numpy ndarray
        array of evaluation points
    fk : numpy ndarray
        array of function values at the interpolation nodes xk
    xk: numpy ndarray
        array of interpolation nodes
    vk: numpy ndarray
        barycentric weights corresponding to the interpolation nodes xk
    """

    # bookkeeping for non-array inputs
    xx_is_scalar = isscalar(xx)
    if xx_is_scalar:
        xx = array([xx])

    # deal with a constant function
    if fk.size == 1:
        return fk[0] if xx_is_scalar else fk * ones(xx.size)

    # NaNs in function values
    if any(isnan(fk)):
        return NaN * ones(xx.size)

    # either iterate over evaluation points, or ...
    if xx.size < 4*xk.size:
        out = zeros(xx.size)
        for i in xrange(xx.size):
            tt = vk / (xx[i] - xk)
            out[i] = dot(tt, fk) / tt.sum()

    # ... iterate over barycenters
    else:
        numer = zeros(xx.size)
        denom = zeros(xx.size)
        for j in xrange(xk.size):
            temp = vk[j] / (xx - xk[j])
            numer = numer + temp * fk[j]
            denom = denom + temp
        out = numer / denom

    # replace NaNs
    for k in find( isnan(out) ):
        idx = find( xx[k] == xk )
        if idx.size > 0:
            out[k] = fk[idx[0]]

    return out[0] if xx_is_scalar else out


@returnemptyarray
def clenshaw(xx, ak):
    """Clenshaw's algorithm for the evaluation of a first-kind Chebyshev 
    series expansion at some array of points x"""
    bk1 = 0*xx
    bk2 = 0*xx
    xx = 2*xx
    idx = range(ak.size)
    for k in idx[ak.size:1:-2]:
        bk2 = ak[k] + xx*bk1 - bk2
        bk1 = ak[k-1] + xx*bk2 - bk1
    if mod(ak.size-1, 2) == 1:
        bk1, bk2 = ak[1] + xx*bk1 - bk2, bk1
    out = ak[0] + .5*xx*bk1 - bk2
    return out


def standard_chop(coeffs, tol=eps):
    """Chop a Chebyshev series to a given tolerance. This is a Python
    transcription of the algorithm described in:

    J. Aurentz and L.N. Trefethen, Chopping a Chebyshev series (2015)
    (http://arxiv.org/pdf/1512.01803v1.pdf)
    """

    # ensure length at least 17:
    n = coeffs.size
    cutoff = n
    if n < 17:
        return cutoff

    # Step 1
    b = abs(coeffs)
    m = b[-1] * ones(n)
    for j in arange(n-2, -1, -1):   # n-2, ... , 2, 1, 0
        m[j] = max( (b[j], m[j+1]) )
    if m[0] == 0.:
        cutoff = 0
        return cutoff
    envelope = m / m[0]

    # Step 2
    for j in arange(1, n):
        j2 = round(1.25*j+5)
        if j2 > n-1:
            # there is no plateau: exit
            return cutoff
        e1 = envelope[j]
        e2 = envelope[int(j2)]
        r = 3 * (1 - log(e1) / log(tol))
        plateau = (e1==0.) | (e2/e1>r)
        if plateau:
            # a plateau has been found: go to Step 3
            plateauPoint = j
            break

    # Step 3
    if envelope[int(plateauPoint)] == 0.:
        cutoff = plateauPoint
    else:
        j3 = sum(envelope >= tol**(7./6.))
        if j3 < j2:
            j2 = j3 + 1
            envelope[j2] = tol**(7./6.)
        cc = log10(envelope[:int(j2)])
        cc = cc + linspace(0, (-1./3.)*log10(tol), j2)
        d = argmin(cc)
        cutoff = d + 2
    return min( (cutoff, n-1) )


def ctor_adaptive(cls, fun, maxpow2=16):
    """Adaptive constructor: cycle over powers of two, calling
    standard_chop each time, the output of which determines whether or not
    we are happy."""
    for k in xrange(4, maxpow2+1):
        n = 2**k + 1
        points = cls.chebpts(n)
        values = fun(points)
        coeffs = cls._vals2coeffs(values)
        chplen = standard_chop(coeffs)
        if chplen < coeffs.size:
            coeffs = coeffs[:chplen]
            break
        if k == maxpow2:
            warn("The {} constructor did not converge: "\
                 "using {} points".format(cls.__name__, n))
            break
    return coeffs
