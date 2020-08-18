# -*- coding: utf-8 -*-

from __future__ import division

import abc
import numpy as np

from chebpy.core.smoothfun import Smoothfun
from chebpy.core.settings import userPrefs as prefs
from chebpy.core.decorators import self_empty
from chebpy.core.algorithms import (bary, clenshaw, adaptive, coeffmult,
                                    vals2coeffs2, coeffs2vals2, chebpts2,
                                    barywts2, rootsunit, newtonroots,
                                    standard_chop)
from chebpy.core.plotting import import_plt, plotfun, plotfuncoeffs


class Chebtech(Smoothfun):
    '''Abstract base class serving as the template for Chebtech1 and
    Chebtech2 subclasses. 

    Chebtech objects always work with first-kind coefficients, so much 
    of the core operational functionality is defined this level.

    The user will rarely work with these classes directly so we make
    several assumptions regarding input data types.
    '''
    __metaclass__ = abc.ABCMeta
    
    @classmethod
    def initconst(cls, c):
        '''Initialise a Chebtech from a constant c'''
        if not np.isscalar(c):
            raise ValueError(c)
        return cls(np.array([c]))

    @classmethod
    def initempty(cls):
        '''Initialise an empty Chebtech'''
        return cls(np.array([]))

    @classmethod
    def initidentity(cls):
        '''Chebtech representation of f(x) = x on [-1,1]'''
        return cls(np.array([0,1]))

    @classmethod
    def initfun(cls, fun, n=None):
        '''Convenience constructor to automatically select the adaptive or
        fixedlen constructor from the input arguments passed.'''
        if n is None:
            return cls.initfun_adaptive(fun)
        else:
            return cls.initfun_fixedlen(fun, n)

    @classmethod
    def initfun_fixedlen(cls, fun, n):
        '''Initialise a Chebtech from the callable fun using n degrees of
        freedom.'''
        points = cls._chebpts(n)
        values = fun(points)
        coeffs = vals2coeffs2(values)
        return cls(coeffs)

    @classmethod
    def initfun_adaptive(cls, fun, interval=None):
        '''Initialise a Chebtech from the callable fun utilising the adaptive
        constructor to determine the number of degrees of freedom parameter.'''
        interval = interval if interval is not None else prefs.domain
        dom = (interval[0], interval[-1])
        hscale = max(np.linalg.norm(dom, np.inf), 1)
        hscaleF = dom[-1]-dom[0]  # this scales hscale back to 1 if interval=domain
        hscale = max(hscale/hscaleF, 1)  # otherwise, hscale < 1 in default case
        coeffs = adaptive(cls, fun, hscale=hscale)
        return cls(coeffs)

    @classmethod
    def initvalues(cls, values):
        '''Initialise a Chebtech from an array of values at Chebyshev points'''
        return cls(cls._vals2coeffs(values))

    def __init__(self, coeffs):
        self._coeffs = np.array(coeffs, dtype=float)

    def __call__(self, x, how='clenshaw'):
        method = {
            'clenshaw': self.__call__clenshaw,
            'bary': self.__call__bary,
            }
        try:
            return method[how](x)
        except KeyError:
            raise ValueError(how)

    def __call__clenshaw(self, x):
        return clenshaw(x, self.coeffs)
        
    def __call__bary(self, x):
        fk = self.values()
        xk = self._chebpts(fk.size)
        vk = self._barywts(fk.size)
        return bary(x, fk, xk, vk)

    def __str__(self):
        out = '<{0}{{{1}}}>'.format(self.__class__.__name__, self.size)
        return out

    def __repr__(self):
        return self.__str__()

    # ------------
    #  properties
    # ------------
    @property
    def coeffs(self):
        '''Chebyshev expansion coefficients in the T_k basis'''
        return self._coeffs

    @property
    def size(self):
        '''Return the size of the object'''
        return self.coeffs.size

    @property
    def isempty(self):
        '''Return True if the Chebtech is empty'''
        return self.size == 0

    @property
    def isconst(self):
        '''Return True if the Chebtech represents a constant'''
        return self.size == 1

    @property
    @self_empty(0.)
    def vscale(self):
        '''Estimate the vertical scale of a Chebtech'''
        if self.isconst:
            values = self.coeffs
        else:
            values = self.values()
        vscale = abs(values).max()
        return vscale

    # -----------
    #  utilities
    # -----------
    def prolong(self, n):
        '''Return a Chebtech of length n, obtained either by truncating
        if n < self.size or zero-padding if n > self.size. In all cases a
        deep copy is returned.
        '''
        m = self.size
        ak = self.coeffs
        cls = self.__class__
        if n - m < 0:
            out = cls(ak[:n].copy())
        elif n - m > 0:
            out = cls(np.append(ak, np.zeros(n-m)))
        else:
            out = self.copy()
        return out

    def copy(self):
        '''Return a deep copy of the Chebtech'''
        return self.__class__(self.coeffs.copy())

    def values(self):
        '''Function values at Chebyshev points'''
        return coeffs2vals2(self.coeffs)

    def simplify(self):
        '''Call standard_chop on the coefficients of self, returning a
        Chebtech comprised of a copy of the truncated coefficients.'''
        cfs = self.coeffs
        npts = standard_chop(cfs)
        return self.__class__(cfs[:npts].copy())

    # ---------
    #  algebra
    # ---------
    @self_empty()
    def __add__(self, f):
        cls = self.__class__
        if np.isscalar(f):
            cfs = self.coeffs.copy()
            cfs[0] += f
            return cls(cfs)
        else:
            # TODO: is a more general decorator approach better here?
            # TODO: for constant Chebtech, convert to constant and call __add__ again 
            if f.isempty:
                return f.copy()
            g = self
            n, m = g.size, f.size
            if n < m:
                g = g.prolong(m)
            elif m < n:
                f = f.prolong(n)
            cfs = f.coeffs + g.coeffs

            # check for zero output
            eps = prefs.eps
            tol = .2 * eps * max([f.vscale, g.vscale])
            if all(abs(cfs)<tol):
                return cls.initconst(0.)
            else:
                return cls(cfs)

    @self_empty()
    def __div__(self, f):
        cls = self.__class__
        if np.isscalar(f):
            cfs = 1./f * self.coeffs
            return cls(cfs)
        else:
            # TODO: review with reference to __add__
            if f.isempty:
                return f.copy()
            divfun = lambda x: self(x) / f(x)
            return cls.initfun_adaptive(divfun)

    __truediv__ = __div__

    @self_empty()
    def __mul__(self, g):
        cls = self.__class__
        if np.isscalar(g):
            cfs = g * self.coeffs
            return cls(cfs)
        else:
            # TODO: review with reference to __add__
            if g.isempty:
                return g.copy()
            f = self
            n = f.size + g.size - 1
            f = f.prolong(n)
            g = g.prolong(n)
            cfs = coeffmult(f.coeffs, g.coeffs)
            out = cls(cfs)
            return out

    def __neg__(self):
        coeffs = -self.coeffs
        return self.__class__(coeffs)

    def __pos__(self):
        return self

    @self_empty()
    def __pow__(self, f):
        if np.isscalar(f):
            powfun = lambda x: np.power(self(x), f)
        else:
            powfun = lambda x: np.power(self(x), f(x))
        return self.__class__.initfun_adaptive(powfun)

    def __rdiv__(self, f):
        # Executed when __div__(f, self) fails, which is to say whenever f
        # is not a Chebtech. We proceeed on the assumption f is a scalar.
        constfun = lambda x: .0*x + f
        quotient = lambda x: constfun(x) / self(x)
        return self.__class__.initfun_adaptive(quotient)

    __radd__ = __add__

    def __rsub__(self, f):
        return -(self-f)

    @self_empty()
    def __rpow__(self, f):
        powfun = lambda x: np.power(f, self(x))
        return self.__class__.initfun_adaptive(powfun)

    __rtruediv__ = __rdiv__
    __rmul__ = __mul__

    def __sub__(self, f):
        return self + (-f)

    # -------
    #  roots
    # -------
    def roots(self, sort=None):
        '''Compute the roots of the Chebtech on [-1,1] using the
        coefficients in the associated Chebyshev series approximation'''
        sort = sort if sort is not None else prefs.sortroots
        rts = rootsunit(self.coeffs)
        rts = newtonroots(self, rts)
        # fix problems with newton for roots that are numerically very close
        rts = np.clip(rts, -1, 1)  # if newton roots are just outside [-1,1]
        rts = rts if not sort else np.sort(finalrts)
        return rts

    # ----------
    #  calculus
    # ----------
    # Note that function returns 0 for an empty Chebtech object; this is
    # consistent with numpy, which returns zero for the sum of an empty array
    @self_empty(resultif=0.)
    def sum(self):
        '''Definite integral of a Chebtech on the interval [-1,1]'''
        if self.isconst:
            out = 2.*self(0.)
        else:
            ak = self.coeffs.copy()
            ak[1::2] = 0
            kk = np.arange(2, ak.size)
            ii = np.append([2,0], 2/(1-kk**2))
            out = (ak*ii).sum()
        return out

    @self_empty()
    def cumsum(self):
        '''Return a Chebtech object representing the indefinite integral
        of a Chebtech on the interval [-1,1]. The constant term is chosen
        such that F(-1) = 0.'''
        n = self.size
        ak = np.append(self.coeffs, [0, 0])
        bk = np.zeros(n+1)
        rk = np.arange(2,n+1)
        bk[2:] = .5*(ak[1:n] - ak[3:]) / rk
        bk[1] = ak[0] - .5*ak[2]
        vk = np.ones(n)
        vk[1::2] = -1
        bk[0] = (vk*bk[1:]).sum()
        out = self.__class__(bk)
        return out

    @self_empty()
    def diff(self):
        '''Return a Chebtech object representing the derivative of a
        Chebtech on the interval [-1,1].'''
        if self.isconst:
            out = self.__class__(np.array([0.]))
        else:
            n = self.size
            ak = self.coeffs
            zk = np.zeros(n-1)
            wk = 2*np.arange(1, n)
            vk = wk * ak[1:]
            zk[-1::-2] = vk[-1::-2].cumsum()
            zk[-2::-2] = vk[-2::-2].cumsum()
            zk[0] = .5 * zk[0]
            out = self.__class__(zk)
        return out

    # ---------------------------------
    #  subclasses must implement these
    # ---------------------------------
    @abc.abstractmethod
    def _chebpts():
        pass

    @abc.abstractmethod
    def _barywts():
        pass
    
    @abc.abstractmethod
    def _vals2coeffs():
        pass

    @abc.abstractmethod
    def _coeffs2vals():
        pass

# ----------
#  plotting
# ----------

plt = import_plt()
if plt:
    def plot(self, ax=None, **kwargs):
        return plotfun(self, (-1, 1), ax=ax, **kwargs)
    setattr(Chebtech, 'plot', plot)

    def plotcoeffs(self, ax=None, **kwargs):
        ax = ax or plt.gca()
        return plotfuncoeffs(abs(self.coeffs), ax=ax, **kwargs)
    setattr(Chebtech, 'plotcoeffs', plotcoeffs)


class Chebtech2(Chebtech):
    '''Second-Kind Chebyshev technology'''
    
    @staticmethod
    def _chebpts(n):
        '''Return n Chebyshev points of the second-kind'''
        return chebpts2(n)

    @staticmethod
    def _barywts(n):
        '''Barycentric weights for Chebyshev points of 2nd kind'''
        return barywts2(n)
    
    @staticmethod
    def _vals2coeffs(vals):
        '''Map function values at Chebyshev points of 2nd kind to
        first-kind Chebyshev polynomial coefficients'''
        return vals2coeffs2(vals)

    @staticmethod
    def _coeffs2vals(coeffs):
        '''Map first-kind Chebyshev polynomial coefficients to
        function values at Chebyshev points of 2nd kind'''
        return coeffs2vals2(coeffs)
