# -*- coding: utf-8 -*-
"""
"""
from __future__ import division

from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty

from numpy import finfo
from numpy import array
from numpy import zeros
from numpy import ones
from numpy import ones_like
from numpy import arange
from numpy import append
from numpy import pi
from numpy import NaN
from numpy import cos
from numpy import real
from numpy import imag
from numpy import isreal
from numpy import isnan
from numpy import any as any_
from numpy import dot
from numpy import where
from numpy import linspace
from numpy import log10
from numpy import mod

from numpy.fft import fft
from numpy.fft import ifft

from matplotlib.pyplot import gca

def find(x):
    return where(x)[0]

# machine epsilon
eps = finfo(float).eps

class ChebTech(object):
    """Abstract base class serving as the template for ChebTech1 and 
    ChebTech2 subclasses. 
    
    ChebTech objects always work with first-kind coefficients, so much 
    of the core operational functionality is defined this level.
    
    The user will rarely work with these classes directly so we make
    several assumptions regarding input data types.
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, coeffs, vscale=None, hscale=None):
        self._coeffs = coeffs
        self._vscale = vscale
        self._hscale = hscale

    @classmethod
    def initfun(cls, fun, n=None, maxdof=None):
        if n is None:
            coeffs = _adaptive(cls, fun)
        else:
            points = cls.chebpts(n)
            values = fun(points)
            coeffs = cls._vals2coeffs(values)
        return cls(coeffs)

    def coeffs(self):
        """Chebyshev expansion coefficients in the T_k (first-kind) basis"""
        return self._coeffs
        
    def values(self):
        """Function values at Chebyshev points"""
        return self._coeffs2vals(self._coeffs)
        
    def __call__(self, x, how="clenshaw"):
        method = {
            "clenshaw": self.__call__clenshaw,
            "bary": self.__call__bary,
            }
        try:
            return method[how](x)
        except:
            raise ValueError("\'how' must be either \'clenshaw\' or \'bary\'")

    def __call__clenshaw(self, x):
        return clenshaw(x, self._coeffs)
        
    def __call__bary(self, x):
        fk = self.values()
        xk = self.chebpts(fk.size)
        vk = self.barywts(fk.size)
        return bary(x, fk, xk, vk)

    def plot(self, ax=None, *args, **kwargs):
        ax = ax if ax else gca()
        xx = linspace(-1, 1, 2001)
        yy = self(xx)
        ax.plot(xx, yy, *args, **kwargs)
        return ax

    def chebpolyplot(self, ax=None, *args, **kwargs):
        ax = ax if ax else gca()
        logcoeffs = log10(abs(self._coeffs))
        ax.plot(logcoeffs, ".", *args, **kwargs)
        return ax

    def __repr__(self):
        out = "{} <{}>".format(self.__class__.__name__, self._coeffs.size)
        return out
    

    # --------------------------------------------------
    #          abstract method declarations
    # --------------------------------------------------
    @abstractmethod
    def chebpts():
        pass

    @abstractmethod
    def barywts():
        pass
    
    @abstractmethod
    def _vals2coeffs():
        pass

    @abstractmethod
    def _coeffs2vals():
        pass
    # --------------------------------------------------


class ChebTech2(ChebTech):
    """Second-Kind Chebyshev technology"""
    
    @staticmethod
    def chebpts(n):
        """Return n Chebyshev points of the second-kind"""
        if n == 1:
            pts = array([0.])
        else:
            nn = arange(n)
            pts = cos( nn[::-1] * pi/(n-1) )
        return pts

    @staticmethod
    def barywts(n):
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
    
    @staticmethod
    def _vals2coeffs(vals):
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

    @staticmethod
    def _coeffs2vals(coeffs):
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


def bary(x, fvals, xk=None, vk=None):
    """Barycentric interpolation formula"""
    
    # default to 2nd kind nodes and weights
    if xk is None:
        xk = ChebTech2.chebpts(fvals.size)
    if vk is None:
        vk = ChebTech2.barywts(fvals.size)

    # function is constant
    if fvals.size == 1:
        fx = fvals * ones_like(x)
        return fx

    # function contains NaN
    if any_(isnan(fvals)):
        fx = NaN * ones_like(x)
        return fx
    
    # ether iterate over evaluation points, or ... 
    if x.size < 4*xk.size:
        fx = zeros(x.size)
        for i in xrange(x.size):
            xx = vk / (x[i] - xk)
            fx[i] = dot(xx, fvals) / xx.sum()
            
    # ... iterate over interpolation nodes
    else:
        numer = zeros(x.size)
        denom = zeros(x.size)
        for j in xrange(xk.size):
            temp = vk[j] / (x - xk[j])
            numer = numer + temp * fvals[j]
            denom = denom + temp
        fx = numer / denom
    
    # replace NaNs
    for k in find( isnan(fx) ):
        idx = find( x[k] == xk )
        if idx.size > 0:
            fx[k] = fvals[idx[0]] 
            
    return fx


def clenshaw(x, a):
    """Clenshaw's Algorithm"""
    bk1 = 0*x 
    bk2 = 0*x
    x = 2*x
    idx = range(a.size)
    for k in idx[a.size:1:-2]:
        bk2 = a[k] + x*bk1 - bk2
        bk1 = a[k-1] + x*bk2 - bk1
    if mod(a.size-1, 2) != 0:
        bk1, bk2 = a[1] + x*bk1 - bk2, bk1
    out = a[0] + .5*x*bk1 - bk2
    return out


def _adaptive(cls, fun, vscale=None, hscale=None, maxdof=None):
    """Adaptive constructor"""
    for k in xrange(2, 17):
        n = 2 ** k
        points = cls.chebpts(n)            
        values = fun(points)
        coeffs = cls._vals2coeffs(values)
        vscl = vscale if vscale else abs(values).max()
        tol = 1 * k * eps * vscl
        tail = abs(coeffs)[-2:]
        if (tail < tol).all():
            break
    return coeffs


def _simplify(coeffs):
    pass

if __name__ == "__main__":
    
    from numpy import sin, cos, exp
    import matplotlib.pyplot as plt
    xk = ChebTech2.chebpts(15)
    fk = exp(xk)
    ak = ChebTech2._vals2coeffs(fk)
    c1 = ChebTech2(ak)
    c1.plot()

    c2 = ChebTech2.initfun(sin, 15)
    
    c2.plot()    
    
    fix, ax = plt.subplots()
    c3 = ChebTech2.initfun(sin)
    c3.chebpolyplot(ax=ax)