# -*- coding: utf-8 -*-
"""
"""
from __future__ import division

from abc import ABCMeta
from abc import abstractmethod

from numpy import finfo
from numpy import array
from numpy import ones
from numpy import arange
from numpy import append
from numpy import pi
from numpy import cos
from numpy import real
from numpy import imag
from numpy import isreal
from numpy import linspace
from numpy import zeros
from numpy import isscalar
from numpy import max

from numpy.fft import fft
from numpy.fft import ifft

from matplotlib.pyplot import gca

from pyfun.utilities import bary
from pyfun.utilities import clenshaw
from pyfun.utilities import checkempty
from pyfun.utilities import ctor_adaptive
from pyfun.utilities import coeffmult


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
    
    def __init__(self, coeffs):
        self._coeffs = coeffs

    @classmethod
    def initconst(cls, c):
        """Initialise a ChebTech from a constant c"""
        if not isscalar(c):
            raise ValueError(c)
        return cls(array([float(c)]))

    @classmethod
    def initempty(cls):
        """Initialise an empty ChebTech"""
        return cls(array([]))

    @classmethod
    def initfun(cls, fun, n=None):
        if n is None:
            return cls.initfun_adaptive(fun)
        else:
            return cls.initfun_fixedlen(fun, n)

    @classmethod
    def initfun_fixedlen(cls, fun, n):
        points = cls.chebpts(n)
        values = fun(points)
        coeffs = cls._vals2coeffs(values)
        return cls(coeffs)

    @classmethod
    def initfun_adaptive(cls, fun):
        coeffs = ctor_adaptive(cls, fun)
        return cls(coeffs)

    def __call__(self, x, how="clenshaw"):
        method = {
            "clenshaw": self.__call__clenshaw,
            "bary": self.__call__bary,
            }
        try:
            return method[how](x)
        except KeyError:
            raise ValueError(how)

    def __call__clenshaw(self, x):
        return clenshaw(x, self._coeffs)
        
    def __call__bary(self, x):
        fk = self.values()
        xk = self.chebpts(fk.size)
        vk = self.barywts(fk.size)
        return bary(x, fk, xk, vk)

    def __str__(self):
        out = "<{0}{{{1}}}>".format(self.__class__.__name__, self.size())
        return out

    def __repr__(self):
        return self.__str__()

    # ---------------------------------
    #     "public" utility methods
    # ---------------------------------
    def prolong(self, n):
        """Return a ChebTech of length n, obtained either by truncating
        if n < self.size or zero-padding if n > self.size. In all cases a
        deep copy is returned.
        """
        m = self.size()
        ak = self.coeffs()
        cls = self.__class__
        if n - m < 0:
            out = cls(ak[:n].copy())
        elif n - m > 0:
            out = cls(append(ak, zeros(n-m)))
        else:
            out = self.copy()
        return out

    def copy(self):
        """Return a deep copy of the ChebTech"""
        return self.__class__(self.coeffs().copy())

    def coeffs(self):
        """Chebyshev expansion coefficients in the T_k basis"""
        return self._coeffs

    def values(self):
        """Function values at Chebyshev points"""
        return self._coeffs2vals(self._coeffs)

    def size(self):
        """Return the size of the object"""
        return self.coeffs().size

    def isempty(self):
        """Return True if the ChebTech is empty"""
        return self.size() == 0

    def isconst(self):
        """Return True if the ChebTech represents a constant"""
        return self.size() == 1

    def simplify(self):
        """Placeholder: Implement This"""
        return self

    @checkempty(resultif=0.)
    def vscale(self):
        """Estimate the vertical scale of a ChebTech"""
        if self.isconst():
            values = self.coeffs()
        else:
            values = self.values()
        vscale = abs(values).max()
        return vscale

    # ---------------------------------
    #        ChebTech algebra
    # ---------------------------------
    @checkempty()
    def __add__(self, f):
        cls = self.__class__
        if isscalar(f):
            cfs = self.coeffs().copy()
            cfs[0] += f
            return cls(cfs)
        else:
            # TODO: is a more general decorator approach better here?
            # TODO: for constant chebtech, convert to constant and call __add__ again 
            if f.isempty():
                return f.copy()
            g = self
            n, m = g.size(), f.size()
            if n < m:
                g = g.prolong(m)
            elif m < n:
                f = f.prolong(n)
            cfs = f.coeffs() + g.coeffs()

            # check for zero output (merge this into simplify?)
            tol = .2 * eps * max([f.vscale(), g.vscale()])
            if all( abs(cfs)<tol ):
                return cls.initconst(0.)
            else:
                return cls(cfs)

    def __sub__(self, f):
        return self + (-f)

    def __pos__(self):
        return self.copy()

    def __neg__(self):
        coeffs = -self.coeffs()
        return self.__class__(coeffs)

    __radd__ = __add__

    def __rsub__(self, f):
        return -(self-f)

    @checkempty()
    def __mul__(self, g):
        cls = self.__class__
        if isscalar(g):
            cfs = self.coeffs().copy()
            cfs *= g
            return cls(cfs)
        else:
            # TODO: review with reference to __add__
            if g.isempty():
                return g.copy()
            f = self
            n = f.size() + g.size() - 1
            f = f.prolong(n)
            g = g.prolong(n)
            cfs = coeffmult(f.coeffs(), g.coeffs())
            out = cls(cfs).simplify()
            return out

    __rmul__ = __mul__


    # ---------------------------------
    #            calculus
    # ---------------------------------
    @checkempty(resultif=0.)
    def sum(self):
        """Definite integral of a ChebTech on the interval [-1,1]"""
        if self.isconst():
            out = 2.*self(0.)
        else:
            ak = self.coeffs().copy()
            ak[1::2] = 0
            kk = arange(2, ak.size)
            ii = append([2,0], 2/(1-kk**2))
            out = (ak*ii).sum()
        return out

    @checkempty()
    def cumsum(self):
        """Return a ChebTech object representing the indefinite integral
        of a ChebTech on the interval [-1,1]. The constant term is chosen
        such that F(-1) = 0."""
        n = self.size()
        ak = append(self.coeffs(), [0, 0])
        bk = zeros(n+1)
        rk = arange(2,n+1)
        bk[2:] = .5*(ak[1:n] - ak[3:]) / rk
        bk[1] = ak[0] - .5*ak[2]
        vk = ones(n)
        vk[1::2] = -1
        bk[0] = (vk*bk[1:]).sum()
        out = self.__class__(bk)
        return out

    @checkempty()
    def diff(self):
        """Return a ChebTech object representing the derivative of a
        ChebTech on the interval [-1,1]."""
        if self.isconst():
            out = self.__class__([0])
        else:
            n = self.size()
            ak = self.coeffs()
            zk = zeros(n-1)
            wk = 2 * arange(1, n)
            vk = wk * ak[1:]
            zk[-1::-2] = vk[-1::-2].cumsum()
            zk[-2::-2] = vk[-2::-2].cumsum()
            zk[0] = .5 * zk[0]
            out = self.__class__(zk)
        return out

    # ---------------------------------
    #            plotting
    # ---------------------------------
    def plot(self, ax=None, *args, **kwargs):
        ax = ax if ax else gca()
        xx = linspace(-1, 1, 2001)
        yy = self(xx)
        ax.plot(xx, yy, *args, **kwargs)
        return ax

    def plotcoeffs(self, ax=None, *args, **kwargs):
        ax = ax if ax else gca()
        abscoeffs = abs(self._coeffs)
        ax.semilogy(abscoeffs, ".", *args, **kwargs)
        ax.set_ylabel("coefficient magnitude")
        ax.set_xlabel("polynomial degree")
        return ax

    # ---------------------------------
    #  subclasses must implement these
    # ---------------------------------
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
