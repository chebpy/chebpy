import operator

import numpy as np

from .bndfun import Bndfun
from .settings import _preferences as prefs
from .utilities import Domain, check_funs, generate_funs, compute_breakdata
from .decorators import self_empty, float_argument, cast_arg_to_chebfun, cache
from .exceptions import BadFunLengthArgument
from .plotting import import_plt, plotfun


class Chebfun:
    def __init__(self, funs):
        self.funs = check_funs(funs)
        self.breakdata = compute_breakdata(self.funs)
        self.transposed = False

    @classmethod
    def initempty(cls):
        return cls([])

    @classmethod
    def initidentity(cls, domain=None):
        return cls(generate_funs(domain, Bndfun.initidentity))

    @classmethod
    def initconst(cls, c, domain=None):
        return cls(generate_funs(domain, Bndfun.initconst, {"c": c}))

    @classmethod
    def initfun_adaptive(cls, f, domain=None):
        return cls(generate_funs(domain, Bndfun.initfun_adaptive, {"f": f}))

    @classmethod
    def initfun_fixedlen(cls, f, n, domain=None):
        nn = np.array(n)
        if nn.size < 2:
            funs = generate_funs(domain, Bndfun.initfun_fixedlen, {"f": f, "n": n})
        else:
            domain = Domain(domain if domain is not None else prefs.domain)
            if not nn.size == domain.size - 1:
                raise BadFunLengthArgument
            funs = []
            for interval, length in zip(domain.intervals, nn):
                funs.append(Bndfun.initfun_fixedlen(f, interval, length))
        return cls(funs)

    @classmethod
    def initfun(cls, f, domain=None, n=None):
        if n is None:
            return cls.initfun_adaptive(f, domain)
        else:
            return cls.initfun_fixedlen(f, n, domain)

    # --------------------
    #  operator overloads
    # --------------------
    def __add__(self, f):
        return self._apply_binop(f, operator.add)

    @self_empty(np.array([]))
    @float_argument
    def __call__(self, x):

        # initialise output
        dtype = complex if self.iscomplex else float
        out = np.full(x.size, np.nan, dtype=dtype)

        # evaluate a fun when x is an interior point
        for fun in self:
            idx = fun.interval.isinterior(x)
            out[idx] = fun(x[idx])

        # evaluate the breakpoint data for x at a breakpoint
        breakpoints = self.breakpoints
        for break_point in breakpoints:
            out[x == break_point] = self.breakdata[break_point]

        # first and last funs used to evaluate outside of the chebfun domain
        lpts, rpts = x < breakpoints[0], x > breakpoints[-1]
        out[lpts] = self.funs[0](x[lpts])
        out[rpts] = self.funs[-1](x[rpts])
        return out

    def __iter__(self):
        return self.funs.__iter__()

    def __mul__(self, f):
        return self._apply_binop(f, operator.mul)

    def __neg__(self):
        return self.__class__(-self.funs)

    def __pos__(self):
        return self

    def __pow__(self, f):
        return self._apply_binop(f, operator.pow)

    def __rtruediv__(self, c):
        # Executed when truediv(f, self) fails, which is to say whenever c
        # is not a Chebfun. We proceeed on the assumption f is a scalar.
        def constfun(cheb, const):
            return 0.0 * cheb + const

        newfuns = [
            fun.initfun_adaptive(lambda x: constfun(x, c) / fun(x), fun.interval)
            for fun in self
        ]
        return self.__class__(newfuns)

    @self_empty("chebfun<empty>")
    def __repr__(self):
        rowcol = "row" if self.transposed else "column"
        numpcs = self.funs.size
        plural = "" if numpcs == 1 else "s"
        header = "chebfun {} ({} smooth piece{})\n".format(rowcol, numpcs, plural)
        toprow = "       interval       length     endpoint values\n"
        tmplat = "[{:8.2g},{:8.2g}]   {:6}  {:8.2g} {:8.2g}\n"
        rowdta = ""
        for fun in self:
            endpts = fun.support
            xl, xr = endpts
            fl, fr = fun(endpts)
            row = tmplat.format(xl, xr, fun.size, fl, fr)
            rowdta += row
        btmrow = "vertical scale = {:3.2g}".format(self.vscale)
        btmxtr = (
            ""
            if numpcs == 1
            else "    total length = {}".format(sum([f.size for f in self]))
        )
        return header + toprow + rowdta + btmrow + btmxtr

    def __rsub__(self, f):
        return -(self - f)

    @cast_arg_to_chebfun
    def __rpow__(self, f):
        return f ** self

    def __truediv__(self, f):
        return self._apply_binop(f, operator.truediv)

    __rmul__ = __mul__
    __div__ = __truediv__
    __rdiv__ = __rtruediv__
    __radd__ = __add__

    def __str__(self):
        rowcol = "row" if self.transposed else "col"
        out = "<chebfun-{},{},{}>\n".format(
            rowcol, self.funs.size, sum([f.size for f in self])
        )
        return out

    def __sub__(self, f):
        return self._apply_binop(f, operator.sub)

    # ------------------
    #  internal helpers
    # ------------------
    @self_empty()
    def _apply_binop(self, f, op):
        """Funnel method used in the implementation of Chebfun binary
        operators. The high-level idea is to first break each chebfun into a
        series of pieces corresponding to the union of the domains of each
        before applying the supplied binary operator and simplifying. In the
        case of the second argument being a scalar we don't need to do the
        simplify step, since at the Tech-level these operations are are defined
        such that there is no change in the number of coefficients.
        """
        if hasattr(f, "isempty") and f.isempty:
            return f
        if np.isscalar(f):
            chbfn1 = self
            chbfn2 = f * np.ones(self.funs.size)
            simplify = False
        else:
            newdom = self.domain.union(f.domain)
            chbfn1 = self._break(newdom)
            chbfn2 = f._break(newdom)
            simplify = True
        newfuns = []
        for fun1, fun2 in zip(chbfn1, chbfn2):
            newfun = op(fun1, fun2)
            if simplify:
                newfun = newfun.simplify()
            newfuns.append(newfun)
        return self.__class__(newfuns)

    def _break(self, targetdomain):
        """Resamples self to the supplied Domain object, targetdomain. This
        method is intended as private since one will typically need to have
        called either Domain.union(f), or Domain.merge(f) prior to call."""
        newfuns = []
        subintervals = targetdomain.intervals
        interval = next(subintervals)  # next(..) for Python2/3 compatibility
        for fun in self:
            while interval in fun.interval:
                newfun = fun.restrict(interval)
                newfuns.append(newfun)
                try:
                    interval = next(subintervals)
                except StopIteration:
                    break
        return self.__class__(newfuns)

    # ------------
    #  properties
    # ------------
    @property
    def breakpoints(self):
        return np.array([x for x in self.breakdata.keys()])

    @property
    @self_empty(np.array([]))
    def domain(self):
        """Construct and return a Domain object corresponding to self."""
        return Domain.from_chebfun(self)

    @property
    @self_empty(Domain([]))
    def support(self):
        """Return an array containing the first and last breakpoints."""
        return self.domain.support

    @property
    @self_empty(0.0)
    def hscale(self):
        return np.float(np.abs(self.support).max())

    @property
    @self_empty(False)
    def iscomplex(self):
        return any(fun.iscomplex for fun in self)

    @property
    @self_empty(False)
    def isconst(self):
        # TODO: find an abstract way of referencing funs[0].coeffs[0]
        c = self.funs[0].coeffs[0]
        return all(fun.isconst and fun.coeffs[0] == c for fun in self)

    @property
    def isempty(self):
        return self.funs.size == 0

    @property
    @self_empty(0.0)
    def vscale(self):
        return np.max([fun.vscale for fun in self])

    @property
    @self_empty()
    def x(self):
        """Identity function on the support of self."""
        return self.__class__.initidentity(self.support)

    # -----------
    #  utilities
    # ----------

    def imag(self):
        if self.iscomplex:
            return self.__class__([fun.imag() for fun in self])
        else:
            return self.initconst(0, domain=self.domain)

    def real(self):
        if self.iscomplex:
            return self.__class__([fun.real() for fun in self])
        else:
            return self

    def copy(self):
        return self.__class__([fun.copy() for fun in self])

    @self_empty()
    def _restrict(self, subinterval):
        """Restrict a chebfun to a subinterval, without simplifying."""
        newdom = self.domain.restrict(Domain(subinterval))
        return self._break(newdom)

    def restrict(self, subinterval):
        """Restrict a chebfun to a subinterval."""
        return self._restrict(subinterval).simplify()

    @cache
    @self_empty(np.array([]))
    def roots(self, merge=None):
        """Compute the roots of a Chebfun, i.e., the set of values x for which
        f(x) = 0.
        """
        merge = merge if merge is not None else prefs.mergeroots
        allrts = []
        prvrts = np.array([])
        htol = 1e2 * self.hscale * prefs.eps
        for fun in self:
            rts = fun.roots()
            # ignore first root if equal to the last root of previous fun
            # TODO: there could be multiple roots at breakpoints
            if prvrts.size > 0 and rts.size > 0:
                if merge and abs(prvrts[-1] - rts[0]) <= htol:
                    rts = rts[1:]
            allrts.append(rts)
            prvrts = rts
        return np.concatenate([x for x in allrts])

    @self_empty()
    def simplify(self):
        """Simplify each fun in the chebfun"""
        return self.__class__([fun.simplify() for fun in self])

    def translate(self, c):
        """Translate a chebfun by c, i.e., return f(x-c)"""
        return self.__class__([x.translate(c) for x in self])

    # ----------
    #  calculus
    # ----------
    def cumsum(self):
        newfuns = []
        prevfun = None
        for fun in self:
            integral = fun.cumsum()
            if prevfun:
                # enforce continuity by adding the function value
                # at the right endpoint of the previous fun
                _, fb = prevfun.endvalues
                integral = integral + fb
            newfuns.append(integral)
            prevfun = integral
        return self.__class__(newfuns)

    def diff(self):
        dfuns = np.array([fun.diff() for fun in self])
        return self.__class__(dfuns)

    def sum(self):
        return np.sum([fun.sum() for fun in self])

    def dot(self, f):
        return (self * f).sum()

    # ----------
    #  utilities
    # ----------
    @self_empty()
    def absolute(self):
        """Absolute value of a Chebfun"""
        newdom = self.domain.merge(self.roots())
        funs = [x.absolute() for x in self._break(newdom)]
        return self.__class__(funs)

    abs = absolute

    @self_empty()
    @cast_arg_to_chebfun
    def maximum(self, other):
        """Pointwise maximum of self and another chebfun"""
        return self._maximum_minimum(other, operator.ge)

    @self_empty()
    @cast_arg_to_chebfun
    def minimum(self, other):
        """Pointwise mimimum of self and another chebfun"""
        return self._maximum_minimum(other, operator.lt)

    def _maximum_minimum(self, other, comparator):
        """Method for computing the pointwise maximum/minimum of two
        Chebfuns"""
        roots = (self - other).roots()
        newdom = self.domain.union(other.domain).merge(roots)
        switch = newdom.support.merge(roots)
        keys = 0.5 * ((-1) ** np.arange(switch.size - 1) + 1)
        if comparator(other(switch[0]), self(switch[0])):
            keys = 1 - keys
        funs = np.array([])
        for interval, use_self in zip(switch.intervals, keys):
            subdom = newdom.restrict(interval)
            if use_self:
                subfun = self.restrict(subdom)
            else:
                subfun = other.restrict(subdom)
            funs = np.append(funs, subfun.funs)
        return self.__class__(funs)


# ----------
#  plotting
# ----------

plt = import_plt()
if plt:

    def plot(self, ax=None, **kwds):
        return plotfun(self, self.support, ax=ax, **kwds)

    setattr(Chebfun, "plot", plot)

    def plotcoeffs(self, ax=None, **kwds):
        ax = ax or plt.gca()
        for fun in self:
            fun.plotcoeffs(ax=ax, **kwds)
        return ax

    setattr(Chebfun, "plotcoeffs", plotcoeffs)


# ---------
#  ufuncs
# ---------
def addUfunc(op):
    @self_empty()
    def method(self):
        return self.__class__([op(fun) for fun in self])

    name = op.__name__
    method.__name__ = name
    method.__doc__ = "TODO: CHANGE THIS TO SOMETHING MEANINGFUL"
    setattr(Chebfun, name, method)


ufuncs = (
    np.arccos,
    np.arccosh,
    np.arcsin,
    np.arcsinh,
    np.arctan,
    np.arctanh,
    np.cos,
    np.cosh,
    np.exp,
    np.exp2,
    np.expm1,
    np.log,
    np.log2,
    np.log10,
    np.log1p,
    np.sinh,
    np.sin,
    np.tan,
    np.tanh,
    np.sqrt,
)

for op in ufuncs:
    addUfunc(op)
