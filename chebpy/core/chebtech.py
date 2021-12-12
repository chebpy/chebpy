from abc import ABC, abstractmethod

import numpy as np

from .smoothfun import Smoothfun
from .settings import _preferences as prefs
from .decorators import self_empty
from .algorithms import (
    bary,
    clenshaw,
    adaptive,
    coeffmult,
    vals2coeffs2,
    coeffs2vals2,
    chebpts2,
    barywts2,
    rootsunit,
    newtonroots,
    standard_chop,
)
from .plotting import import_plt, plotfun, plotfuncoeffs
from .utilities import Interval, coerce_list


class Chebtech(Smoothfun, ABC):
    """Abstract base class serving as the template for Chebtech1 and
    Chebtech2 subclasses.

    Chebtech objects always work with first-kind coefficients, so much
    of the core operational functionality is defined this level.

    The user will rarely work with these classes directly so we make
    several assumptions regarding input data types.
    """

    @classmethod
    def initconst(cls, c, *, interval=None):
        """Initialise a Chebtech from a constant c"""
        if not np.isscalar(c):
            raise ValueError(c)
        if isinstance(c, int):
            c = float(c)
        return cls(np.array([c]), interval=interval)

    @classmethod
    def initempty(cls, *, interval=None):
        """Initialise an empty Chebtech"""
        return cls(np.array([]), interval=interval)

    @classmethod
    def initidentity(cls, *, interval=None):
        """Chebtech representation of f(x) = x on [-1,1]"""
        return cls(np.array([0, 1]), interval=interval)

    @classmethod
    def initfun(cls, fun, n=None, *, interval=None):
        """Convenience constructor to automatically select the adaptive or
        fixedlen constructor from the input arguments passed."""
        if n is None:
            return cls.initfun_adaptive(fun, interval=interval)
        else:
            return cls.initfun_fixedlen(fun, n, interval=interval)

    @classmethod
    def initfun_fixedlen(cls, fun, n, *, interval=None):
        """Initialise a Chebtech from the callable fun using n degrees of
        freedom."""
        points = cls._chebpts(n)
        values = fun(points)
        coeffs = vals2coeffs2(values)
        return cls(coeffs, interval=interval)

    @classmethod
    def initfun_adaptive(cls, fun, *, interval=None):
        """Initialise a Chebtech from the callable fun utilising the adaptive
        constructor to determine the number of degrees of freedom parameter."""
        interval = interval if interval is not None else prefs.domain
        interval = Interval(*interval)
        coeffs = adaptive(cls, fun, hscale=interval.hscale)
        return cls(coeffs, interval=interval)

    @classmethod
    def initvalues(cls, values, *, interval=None):
        """Initialise a Chebtech from an array of values at Chebyshev points"""
        return cls(cls._vals2coeffs(values), interval=interval)

    def __init__(self, coeffs, interval=None):
        interval = interval if interval is not None else prefs.domain
        self._coeffs = np.array(coeffs)
        self._interval = Interval(*interval)

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
        return clenshaw(x, self.coeffs)

    def __call__bary(self, x):
        fk = self.values()
        xk = self._chebpts(fk.size)
        vk = self._barywts(fk.size)
        return bary(x, fk, xk, vk)

    def __repr__(self):
        out = "<{0}{{{1}}}>".format(self.__class__.__name__, self.size)
        return out

    # ------------
    #  properties
    # ------------
    @property
    def coeffs(self):
        """Chebyshev expansion coefficients in the T_k basis"""
        return self._coeffs

    @property
    def interval(self):
        """Interval that Chebtech is mapped to"""
        return self._interval

    @property
    def size(self):
        """Return the size of the object"""
        return self.coeffs.size

    @property
    def isempty(self):
        """Return True if the Chebtech is empty"""
        return self.size == 0

    @property
    def iscomplex(self):
        """Determine whether the underlying onefun is complex or real valued"""
        return self._coeffs.dtype == complex

    @property
    def isconst(self):
        """Return True if the Chebtech represents a constant"""
        return self.size == 1

    @property
    @self_empty(0.0)
    def vscale(self):
        """Estimate the vertical scale of a Chebtech"""
        return np.abs(coerce_list((self.values()))).max()

    # -----------
    #  utilities
    # -----------
    def copy(self):
        """Return a deep copy of the Chebtech"""
        return self.__class__(self.coeffs.copy(), interval=self.interval.copy())

    def imag(self):
        if self.iscomplex:
            return self.__class__(np.imag(self.coeffs), self.interval)
        else:
            return self.initconst(0, interval=self.interval)

    def prolong(self, n):
        """Return a Chebtech of length n, obtained either by truncating
        if n < self.size or zero-padding if n > self.size. In all cases a
        deep copy is returned.
        """
        m = self.size
        ak = self.coeffs
        cls = self.__class__
        if n - m < 0:
            out = cls(ak[:n].copy(), interval=self.interval)
        elif n - m > 0:
            out = cls(np.append(ak, np.zeros(n - m)), interval=self.interval)
        else:
            out = self.copy()
        return out

    def real(self):
        if self.iscomplex:
            return self.__class__(np.real(self.coeffs), self.interval)
        else:
            return self

    def simplify(self):
        """Call standard_chop on the coefficients of self, returning a
        Chebtech comprised of a copy of the truncated coefficients."""
        # coefficients
        oldlen = len(self.coeffs)
        longself = self.prolong(max(17, oldlen))
        cfs = longself.coeffs
        # scale (decrease) tolerance by hscale
        tol = prefs.eps * max(self.interval.hscale, 1)
        # chop
        npts = standard_chop(cfs, tol=tol)
        npts = min(oldlen, npts)
        # construct
        return self.__class__(cfs[:npts].copy(), interval=self.interval)

    def values(self):
        """Function values at Chebyshev points"""
        return coeffs2vals2(self.coeffs)

    # ---------
    #  algebra
    # ---------
    @self_empty()
    def __add__(self, f):
        cls = self.__class__
        if np.isscalar(f):
            if np.iscomplexobj(f):
                dtype = complex
            else:
                dtype = self.coeffs.dtype
            cfs = np.array(self.coeffs, dtype=dtype)
            cfs[0] += f
            return cls(cfs, interval=self.interval)
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
            tol = 0.5 * eps * max([f.vscale, g.vscale])
            if all(abs(cfs) < tol):
                return cls.initconst(0.0, interval=self.interval)
            else:
                return cls(cfs, interval=self.interval)

    @self_empty()
    def __div__(self, f):
        cls = self.__class__
        if np.isscalar(f):
            cfs = 1.0 / f * self.coeffs
            return cls(cfs, interval=self.interval)
        else:
            # TODO: review with reference to __add__
            if f.isempty:
                return f.copy()
            return cls.initfun_adaptive(
                lambda x: self(x) / f(x), interval=self.interval
            )

    __truediv__ = __div__

    @self_empty()
    def __mul__(self, g):
        cls = self.__class__
        if np.isscalar(g):
            cfs = g * self.coeffs
            return cls(cfs, interval=self.interval)
        else:
            # TODO: review with reference to __add__
            if g.isempty:
                return g.copy()
            f = self
            n = f.size + g.size - 1
            f = f.prolong(n)
            g = g.prolong(n)
            cfs = coeffmult(f.coeffs, g.coeffs)
            out = cls(cfs, interval=self.interval)
            return out

    def __neg__(self):
        coeffs = -self.coeffs
        return self.__class__(coeffs, interval=self.interval)

    def __pos__(self):
        return self

    @self_empty()
    def __pow__(self, f):
        def powfun(fn, x):
            if np.isscalar(fn):
                return fn
            else:
                return fn(x)

        return self.__class__.initfun_adaptive(
            lambda x: np.power(self(x), powfun(f, x)), interval=self.interval
        )

    def __rdiv__(self, f):
        # Executed when __div__(f, self) fails, which is to say whenever f
        # is not a Chebtech. We proceeed on the assumption f is a scalar.
        def constfun(x):
            return 0.0 * x + f

        return self.__class__.initfun_adaptive(
            lambda x: constfun(x) / self(x), interval=self.interval
        )

    __radd__ = __add__

    def __rsub__(self, f):
        return -(self - f)

    @self_empty()
    def __rpow__(self, f):
        return self.__class__.initfun_adaptive(
            lambda x: np.power(f, self(x)), interval=self.interval
        )

    __rtruediv__ = __rdiv__
    __rmul__ = __mul__

    def __sub__(self, f):
        return self + (-f)

    # -------
    #  roots
    # -------
    def roots(self, sort=None):
        """Compute the roots of the Chebtech on [-1,1] using the
        coefficients in the associated Chebyshev series approximation"""
        sort = sort if sort is not None else prefs.sortroots
        rts = rootsunit(self.coeffs)
        rts = newtonroots(self, rts)
        # fix problems with newton for roots that are numerically very close
        rts = np.clip(rts, -1, 1)  # if newton roots are just outside [-1,1]
        rts = rts if not sort else np.sort(rts)
        return rts

    # ----------
    #  calculus
    # ----------
    # Note that function returns 0 for an empty Chebtech object; this is
    # consistent with numpy, which returns zero for the sum of an empty array
    @self_empty(resultif=0.0)
    def sum(self):
        """Definite integral of a Chebtech on the interval [-1,1]"""
        if self.isconst:
            out = 2.0 * self(0.0)
        else:
            ak = self.coeffs.copy()
            ak[1::2] = 0
            kk = np.arange(2, ak.size)
            ii = np.append([2, 0], 2 / (1 - kk ** 2))
            out = (ak * ii).sum()
        return out

    @self_empty()
    def cumsum(self):
        """Return a Chebtech object representing the indefinite integral
        of a Chebtech on the interval [-1,1]. The constant term is chosen
        such that F(-1) = 0."""
        n = self.size
        ak = np.append(self.coeffs, [0, 0])
        bk = np.zeros(n + 1, dtype=self.coeffs.dtype)
        rk = np.arange(2, n + 1)
        bk[2:] = 0.5 * (ak[1:n] - ak[3:]) / rk
        bk[1] = ak[0] - 0.5 * ak[2]
        vk = np.ones(n)
        vk[1::2] = -1
        bk[0] = (vk * bk[1:]).sum()
        out = self.__class__(bk, interval=self.interval)
        return out

    @self_empty()
    def diff(self):
        """Return a Chebtech object representing the derivative of a
        Chebtech on the interval [-1,1]."""
        if self.isconst:
            out = self.__class__(np.array([0.0]), interval=self.interval)
        else:
            n = self.size
            ak = self.coeffs
            zk = np.zeros(n - 1, dtype=self.coeffs.dtype)
            wk = 2 * np.arange(1, n)
            vk = wk * ak[1:]
            zk[-1::-2] = vk[-1::-2].cumsum()
            zk[-2::-2] = vk[-2::-2].cumsum()
            zk[0] = 0.5 * zk[0]
            out = self.__class__(zk, interval=self.interval)
        return out

    # ---------------------------------
    #  subclasses must implement these
    # ---------------------------------
    @abstractmethod
    def _chebpts():
        raise NotImplementedError

    @abstractmethod
    def _barywts():
        raise NotImplementedError

    @abstractmethod
    def _vals2coeffs():
        raise NotImplementedError

    @abstractmethod
    def _coeffs2vals():
        raise NotImplementedError


# ----------
#  plotting
# ----------

plt = import_plt()
if plt:

    def plot(self, ax=None, **kwargs):
        return plotfun(self, (-1, 1), ax=ax, **kwargs)

    setattr(Chebtech, "plot", plot)

    def plotcoeffs(self, ax=None, **kwargs):
        ax = ax or plt.gca()
        return plotfuncoeffs(abs(self.coeffs), ax=ax, **kwargs)

    setattr(Chebtech, "plotcoeffs", plotcoeffs)


class Chebtech2(Chebtech):
    """Second-Kind Chebyshev technology"""

    @staticmethod
    def _chebpts(n):
        """Return n Chebyshev points of the second-kind"""
        return chebpts2(n)

    @staticmethod
    def _barywts(n):
        """Barycentric weights for Chebyshev points of 2nd kind"""
        return barywts2(n)

    @staticmethod
    def _vals2coeffs(vals):
        """Map function values at Chebyshev points of 2nd kind to
        first-kind Chebyshev polynomial coefficients"""
        return vals2coeffs2(vals)

    @staticmethod
    def _coeffs2vals(coeffs):
        """Map first-kind Chebyshev polynomial coefficients to
        function values at Chebyshev points of 2nd kind"""
        return coeffs2vals2(coeffs)
