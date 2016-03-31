
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
from numpy import append
from numpy import real
from numpy import isreal

from numpy.fft import fft
from numpy.fft import ifft

eps = finfo(float).eps

# local helpers
def find(x):
    return where(x)[0]


class Domain(object):
    """
    Utility class to implement domain logic. The purpose of this class
    is to both enforce certain properties of domain components such as
    having exactly two monotonically increasing elements which are
    monotonically, and also to implment mapping to and from the unit
    interval.

        formap: y in [-1,1] -> x in [a,b]
        invmap: x in  [a,b] -> y in [-1,1]
        drvmap: y in [-1,1] -> x in [a,b]

    We also provide a convenience __eq__ method amd set the __call__
    method to evaluate self.formap since this will be used the most
    frequently.

    Currently only implemented for finite a and b.
    """
    def __init__(self, a=-1, b=1):
        if a >= b:
            raise ValueError("Domain values must be strictly increasing")
        self.values = array([a, b])
        self.formap = lambda y: .5*b*(y+1.) + .5*a*(1.-y)
        self.invmap = lambda x: (2.*x-a-b) / (b-a)
        self.dermap = lambda y: 0.*y + .5*(b-a)
        
    def __eq__(self, other):
        return (self.values == other.values).all() 

    def __call__(self, y):
        return self.formap(y)

    def __str__(self):
        cls = self.__class__
        out = "{0}([{1}, {2}])".format(cls.__name__, *self.values)
        return out

    def __repr__(self):
        return self.__str__()

# ------------------------------------
#              decorators
# -------------------------------------

# define an abstract class method decorator:
# http://stackoverflow.com/questions/11217878/python-2-7-combine-abc-abstractmethod-and-classmethod
class abstractclassmethod(classmethod):
    __isabstractmethod__ = True
    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(callable)

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
                    return self.copy()
            else:
                return f(self, *args, **kwargs)
        return wrapper
    return decorator


# pre- and post-processing tasks common to bary and clenshaw
def preandpostprocess(f):

    @wraps(f)
    def thewrapper(*args, **kwargs):
        xx, akfk = args[:2]

        # are any of the first two arguments empty arrays?
        if ( asarray(xx).size==0) | (asarray(akfk).size==0 ):
            return array([])

        # is the function constant?
        elif akfk.size == 1:
            if isscalar(xx):
                return akfk[0]
            else:
                return akfk * ones(xx.size)

        # are there any NaNs in the second argument?
        elif any(isnan(akfk)):
            return NaN * ones(xx.size)

        # convert first argument to an array if it is a scalar and then
        # return the first (and only) element of the result if so
        else:
            args = list(args)
            args[0] = array([xx]) if isscalar(xx) else args[0]
            out = f(*args, **kwargs)
            return out[0] if isscalar(xx) else out

    return thewrapper

# -------------------------------------


@preandpostprocess
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

    # either iterate over the evaluation points, or ...
    if xx.size < 4*xk.size:
        out = zeros(xx.size)
        for i in xrange(xx.size):
            tt = vk / (xx[i] - xk)
            out[i] = dot(tt, fk) / tt.sum()

    # ... iterate over the barycenters
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

    return out


@preandpostprocess
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
        # TODO: check this
        cutoff = 1 # cutoff = 0
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
        # TODO: check this
        cutoff = d # + 2
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


def coeffmult(fc, gc):
    """Coefficient-Space multiplication of equal-length first-kind 
    Chebyshev series."""
    Fc = append( 2.*fc[:1], (fc[1:], fc[:0:-1]) )
    Gc = append( 2.*gc[:1], (gc[1:], gc[:0:-1]) )
    ak = ifft( fft(Fc) * fft(Gc) )
    ak = append( ak[:1], ak[1:] + ak[:0:-1] ) * .25
    ak = ak[:fc.size]
    inputcfs = append(fc, gc)
    out = real(ak) if isreal(inputcfs).all() else ak
    return out
