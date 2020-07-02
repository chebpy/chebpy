# -*- coding: utf-8 -*-

from __future__ import division

import warnings
import numpy as np

from chebpy.core.ffts import fft, ifft
from chebpy.core.utilities import Interval
from chebpy.core.settings import DefaultPrefs
from chebpy.core.decorators import preandpostprocess

# supress numpy division and multiply warnings
np.seterr(divide='ignore', invalid='ignore')

# constants
eps = DefaultPrefs.eps
SPLITPOINT = -0.004849834917525

# local helpers
def find(x):
    return np.where(x)[0]

def rootsunit(ak, htol=1e2*eps):
    """Compute the roots of a funciton on [-1,1] using the coefficeints
    in the associated Chebyshev series representation.

    References
    ----------
    .. [1] I. J. Good, "The colleague matrix, a Chebyshev analogue of the
        companion matrix", Quarterly Journal of Mathematics 12 (1961).

    .. [2] J. A. Boyd, "Computing zeros on a real interval through
        Chebyshev expansion and polynomial rootfinding", SIAM Journal on
        Numerical Analysis 40 (2002).

    .. [3] L. N. Trefethen, Approximation Theory and Approximation
        Practice, SIAM, 2013, chapter 18.
    """
    n = standard_chop(ak)
    ak = ak[:n]

    # if n > 50, we split and recurse
    if n > 50:
        chebpts = chebpts2(ak.size)
        lmap = Interval(-1, SPLITPOINT)
        rmap = Interval(SPLITPOINT, 1)
        lpts = lmap(chebpts)
        rpts = rmap(chebpts)
        lval = clenshaw(lpts, ak)
        rval = clenshaw(rpts, ak)
        lcfs = vals2coeffs2(lval)
        rcfs = vals2coeffs2(rval)
        lrts = rootsunit(lcfs, 2*htol)
        rrts = rootsunit(rcfs, 2*htol)
        return np.append(lmap(lrts), rmap(rrts))

    # trivial base case
    if n <= 1:
        return np.array([])

    # nontrivial base case: either compute directly or solve
    # a Colleague Matrix eigenvalue problem
    if n == 2:
        rts = np.array([-ak[0]/ak[1]])
    elif n <= 50:
        v = .5 * np.ones(n-2)
        C = np.diag(v,-1) + np.diag(v, 1)
        C[0,1] = 1
        D = np.zeros(C.shape)
        D[-1,:] = ak[:-1]
        E = C - .5 * 1./ak[-1] * D
        rts = np.linalg.eigvals(E)

    # discard values with large imaginary part and treat the remaining
    # ones as real; then sort and retain only the roots inside [-1,1]
    mask = abs(np.imag(rts)) < htol
    rts = np.real(rts[mask])
    rts = rts[abs(rts)<=1.+htol]
    rts = np.sort(rts)
    if rts.size >= 2:
        rts[ 0] = max([rts[ 0],-1])
        rts[-1] = min([rts[-1], 1])
    return rts

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
        out = np.zeros(xx.size)
        for i in range(xx.size):
            tt = vk / (xx[i] - xk)
            out[i] = np.dot(tt, fk) / tt.sum()

    # ... iterate over the barycenters
    else:
        numer = np.zeros(xx.size)
        denom = np.zeros(xx.size)
        for j in range(xk.size):
            temp = vk[j] / (xx - xk[j])
            numer = numer + temp * fk[j]
            denom = denom + temp
        out = numer / denom

    # replace NaNs
    for k in find(np.isnan(out)):
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
    if np.mod(ak.size-1, 2) == 1:
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

    # Step 1: Convert coeffs input to a new monotonically nonincreasing
    # vector (envelope) normalized to begin with the value 1.
    b = np.flipud(np.abs(coeffs))
    m = np.flipud(np.maximum.accumulate(b))
    if m[0] == 0.:
        # TODO: check this
        cutoff = 1 # cutoff = 0
        return cutoff
    envelope = m / m[0]

    # Step 2: Scan envelope for a value plateauPoint, the first point, if any,
    # that is followed by a plateau
    for j in np.arange(1, n):
        j2 = round(1.25*j+5)
        if j2 > n-1:
            # there is no plateau: exit
            return cutoff
        e1 = envelope[j]
        e2 = envelope[int(j2)]
        r = 3 * (1 - np.log(e1) / np.log(tol))
        plateau = (e1==0.) | (e2/e1>r)
        if plateau:
            # a plateau has been found: go to Step 3
            plateauPoint = j
            break

    # Step 3: Fix cutoff at a point where envelope, plus a linear function
    # included to bias the result towards the left end, is minimal.
    if envelope[int(plateauPoint)] == 0.:
        cutoff = plateauPoint
    else:
        j3 = sum(envelope >= tol**(7./6.))
        if j3 < j2:
            j2 = j3 + 1
            envelope[j2] = tol**(7./6.)
        cc = np.log10(envelope[:int(j2)])
        cc = cc + np.linspace(0, (-1./3.)*np.log10(tol), int(j2))
        d = np.argmin(cc)
        # TODO: check this
        cutoff = d # + 2
    return min( (cutoff, n-1) )


def adaptive(cls, fun, maxpow2=16):
    """Adaptive constructor: cycle over powers of two, calling
    standard_chop each time, the output of which determines whether or not
    we are happy."""
    for k in range(4, maxpow2+1):
        n = 2**k + 1
        points = cls._chebpts(n)
        values = fun(points)
        coeffs = cls._vals2coeffs(values)
        chplen = standard_chop(coeffs)
        if chplen < coeffs.size:
            coeffs = coeffs[:chplen]
            break
        if k == maxpow2:
            warnings.warn('The {} constructor did not converge: '\
                          'using {} points'.format(cls.__name__, n))
            break
    return coeffs


def coeffmult(fc, gc):
    """Coefficient-Space multiplication of equal-length first-kind
    Chebyshev series."""
    Fc = np.append( 2.*fc[:1], (fc[1:], fc[:0:-1]) )
    Gc = np.append( 2.*gc[:1], (gc[1:], gc[:0:-1]) )
    ak = ifft( fft(Fc) * fft(Gc) )
    ak = np.append( ak[:1], ak[1:] + ak[:0:-1] ) * .25
    ak = ak[:fc.size]
    inputcfs = np.append(fc, gc)
    out = np.real(ak) if np.isreal(inputcfs).all() else ak
    return out


def barywts2(n):
    """Barycentric weights for Chebyshev points of 2nd kind"""
    if n == 0:
        wts = np.array([])
    elif n == 1:
        wts = np.array([1])
    else:
        wts = np.append(np.ones(n-1), .5)
        wts[n-2::-2] = -1
        wts[0] = .5 * wts[0]
    return wts


def chebpts2(n):
    """Return n Chebyshev points of the second-kind"""
    if n == 1:
        pts = np.array([0.])
    else:
        nn = np.arange(n)
        pts = np.cos(nn[::-1]*np.pi/(n-1))
    return pts


def vals2coeffs2(vals):
    """Map function values at Chebyshev points of 2nd kind to
    first-kind Chebyshev polynomial coefficients"""
    n = vals.size
    if n <= 1:
        coeffs = vals
        return coeffs
    tmp = np.append( vals[::-1], vals[1:-1] )
    if np.isreal(vals).all():
        coeffs = ifft(tmp)
        coeffs = np.real(coeffs)
    elif np.isreal( 1j*vals ).all():
        coeffs = ifft(np.imag(tmp))
        coeffs = 1j * np.real(coeffs)
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
    tmp = np.append( coeffs, coeffs[n-2:0:-1] )
    if np.isreal(coeffs).all():
        vals = fft(tmp)
        vals = np.real(vals)
    elif np.isreal(1j*coeffs).all():
        vals = fft(np.imag(tmp))
        vals = 1j * np.real(vals)
    else:
        vals = fft(tmp)
    vals = vals[n-1::-1]
    return vals


def newtonroots(fun, rts, tol=2*eps, maxiter=10):
    """Rootfinding for a callable and differentiable fun, typically used to
    polish already computed roots."""
    if rts.size > 0:
        dfun = fun.diff()
        prv = np.inf*rts
        count = 0
        while (np.linalg.norm(rts-prv, np.inf)>tol) & (count<=maxiter):
            count += 1
            prv = rts
            rts = rts - fun(rts) / dfun(rts)
    return rts
