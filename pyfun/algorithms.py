# -*- coding: utf-8 -*-

from __future__ import division

from warnings import warn

from numpy import finfo
from numpy import ones
from numpy import arange
from numpy import log
from numpy import log10
from numpy import linspace
from numpy import argmin
from numpy import isnan
from numpy import zeros
from numpy import dot
from numpy import where
from numpy import mod
from numpy import append
from numpy import real
from numpy import imag
from numpy import isreal
from numpy import array
from numpy import cos
from numpy import pi

from numpy.fft import fft
from numpy.fft import ifft

from pyfun.decorators import preandpostprocess

eps = finfo(float).eps

# local helpers
def find(x):
    return where(x)[0]

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


def adaptive(cls, fun, maxpow2=16):
    """Adaptive constructor: cycle over powers of two, calling
    standard_chop each time, the output of which determines whether or not
    we are happy."""
    for k in xrange(4, maxpow2+1):
        n = 2**k + 1
        points = cls._chebpts(n)
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


def barywts2(n):
    """Barycentric weights for Chebyshev points of 2nd kind"""
    if n == 0:
        wts = array([])
    elif n == 1:
        wts = array([1])
    else:
        wts = append( ones(n-1), .5 )
        wts[n-2::-2] = -1
        wts[0] = .5 * wts[0]
    return wts


def chebpts2(n):
    """Return n Chebyshev points of the second-kind"""
    if n == 1:
        pts = array([0.])
    else:
        nn = arange(n)
        pts = cos( nn[::-1] * pi/(n-1) )
    return pts


def vals2coeffs2(vals):
    """Map function values at Chebyshev points of 2nd kind to 
    first-kind Chebyshev polynomial coefficients"""        
    n = vals.size
    if n <= 1:
        coeffs = vals
        return coeffs
    tmp = append( vals[::-1], vals[1:-1] )    
    if isreal(vals).all():
        coeffs = ifft(tmp)
        coeffs = real(coeffs)
    elif isreal( 1j*vals ).all():
        coeffs = ifft( imag(tmp) )
        coeffs = 1j * real(coeffs)
    else:
        coeffs = ifft(tmp)
    coeffs = coeffs[:n]        
    coeffs[1:n-1] = 2*coeffs[1:n-1]
    return coeffs


def coeffs2vals2(coeffs):
    """Map first-kind Chebyshev polynomial coefficients to 
    function values at Chebyshev points of 2nd kind"""
    n = coeffs.size
    if n <= 1:
        vals = coeffs
        return vals
    coeffs = coeffs.copy()
    coeffs[1:n-1] = .5 * coeffs[1:n-1]
    tmp = append( coeffs, coeffs[n-2:0:-1] )
    if isreal(coeffs).all():
        vals = fft(tmp)
        vals = real(vals)
    elif isreal( 1j*coeffs ).all():
        vals = fft( imag(tmp) )
        vals = 1j * real(vals)
    else:
        vals = fft(tmp) 
    vals = vals[n-1::-1]
    return vals
