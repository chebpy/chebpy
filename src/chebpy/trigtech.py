"""Trigonometric (Fourier) technology for periodic function approximation.

This module provides the Trigtech class, which represents smooth periodic functions
on [-1, 1] using truncated Fourier series.  It is the trigonometric analogue of
Chebtech and sits in the same class hierarchy:

    Onefun → Smoothfun → Trigtech

Coefficient storage convention (NumPy-native / FFT order)
----------------------------------------------------------
Given n equispaced sample points x_j = -1 + 2j/n (j = 0, …, n-1), the stored
coefficients are

    coeffs[k]  =  (1/n) * sum_j  f(x_j) * exp(-2*pi*i*j*k/n)
               =  (numpy.fft.fft(values) / n)[k]

This is exactly the output of ``numpy.fft.fft(values) / n``, i.e. NumPy-native
(FFT) ordering: DC at index 0, positive frequencies 1 … n//2, then negative
frequencies -(n//2)+1 … -1.

Use ``_coeffs_to_plotorder()`` to obtain the human-readable DC-centred ordering
(equivalent to ``numpy.fft.fftshift``).

Evaluation
----------
Any point x ∈ [-1, 1] is evaluated via the DFT summation formula:

    f(x) = Σ_k  coeffs[k] * exp(i*π*ω_k*(x+1))

where ω_k = numpy.fft.fftfreq(n)*n  gives the integer frequencies in FFT order.

References:
----------
* Trefethen, "Spectral Methods in MATLAB" (SIAM 2000)
* Chebfun @trigtech (github.com/chebfun/chebfun)
"""

import warnings
from abc import ABC
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .decorators import self_empty
from .plotting import plotfun, plotfuncoeffs
from .settings import _preferences as prefs
from .smoothfun import Smoothfun
from .utilities import Interval, coerce_list


def _trig_adaptive(
    cls: Any,
    fun: Any,
    hscale: float = 1,
    maxpow2: int | None = None,
) -> np.ndarray:
    """Adaptively determine the Fourier coefficients needed to represent *fun*.

    Uses successively finer equispaced grids (sizes 2**k) until the
    high-frequency Fourier modes decay below tolerance.  Convergence is
    assessed via the one-sided symmetric maximum of the DC-centred coefficient
    magnitudes: ``abs_sym[k] = max(|c_k|, |c_{-k}|) / vscale``.  The series
    is considered converged when the Nyquist/highest-frequency mode
    ``abs_sym[-1]`` falls below *tol*.

    Args:
        cls: Trigtech class (provides ``_trigpts`` and ``_vals2coeffs``).
        fun: Callable to approximate.
        hscale: Horizontal scale for tolerance adjustment.
        maxpow2: Maximum power of 2 to try (defaults to ``prefs.maxpow2``).

    Returns:
        numpy.ndarray: Fourier coefficients in NumPy FFT order.
    """
    minpow2 = 3  # start at n = 8
    maxpow2 = maxpow2 if maxpow2 is not None else prefs.maxpow2
    tol = prefs.eps * max(hscale, 1)
    coeffs: np.ndarray = np.array([])
    for k in range(minpow2, max(minpow2, maxpow2) + 1):
        n = 2**k
        points = cls._trigpts(n)
        values = fun(points)
        coeffs = cls._vals2coeffs(values)
        vscale = float(np.max(np.abs(values)))
        if vscale <= tol:
            return np.array([0.0])

        # Build one-sided symmetric maximum:
        # abs_sym[ki] = max(|c_{ki}|, |c_{-ki}|) / vscale  for ki = 0…n//2
        centered = np.fft.fftshift(coeffs)
        dc_idx = n // 2
        abs_sym = np.zeros(dc_idx + 1)
        for ki in range(dc_idx + 1):
            p = centered[dc_idx + ki] if dc_idx + ki < n else 0.0
            q = centered[dc_idx - ki]
            abs_sym[ki] = max(abs(p), abs(q)) / vscale

        # Convergence: the Nyquist/highest-frequency mode is negligible.
        if abs_sym[-1] <= tol:
            above = np.where(abs_sym > tol)[0]
            if len(above) == 0:
                return np.array([0.0])
            max_k = int(above[-1])  # highest significant frequency index
            start = dc_idx - max_k
            end = dc_idx + max_k + 1
            return np.fft.ifftshift(centered[start:end])

        if k == maxpow2:
            warnings.warn(
                f"The {cls.__name__} constructor did not converge: using {n} points",
                stacklevel=3,
            )
            break
    return coeffs


class Trigtech(Smoothfun, ABC):
    """Trigonometric (Fourier) function approximation on [-1, 1].

    Represents a smooth periodic function f: [-1, 1] -> R (or C) as a
    truncated Fourier series.  Coefficients are stored in NumPy FFT order;
    see module docstring for the precise convention.

    This class is ``ABC`` so that it cannot be instantiated directly—exactly
    mirroring Chebtech, which is also abstract (concrete only through the
    ``Chebtech`` name used everywhere). In practice ``Trigtech`` is both the
    abstract base and the concrete class: it is not further subclassed, but
    the ABC marker prevents accidental bare construction without going through
    a named constructor.
    """

    # ------------------------------------------------------------------
    #  alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def initconst(cls, c: Any = None, *, interval: Any = None) -> "Trigtech":
        """Initialise a Trigtech from a constant *c*."""
        if not np.isscalar(c):
            raise ValueError(c)
        if isinstance(c, int):
            c = float(c)
        return cls(np.array([c]), interval=interval)

    @classmethod
    def initempty(cls, *, interval: Any = None) -> "Trigtech":
        """Initialise an empty Trigtech."""
        return cls(np.array([]), interval=interval)

    @classmethod
    def initidentity(cls, *, interval: Any = None) -> "Trigtech":
        """Trigtech approximation of the identity f(x) = x on [-1, 1].

        Note: f(x) = x is *not* periodic on [-1, 1], so this will not converge
        to machine precision.  It is provided for interface compatibility with
        Chebtech; in practice ``Classicfun.initidentity`` is used instead.
        """
        interval = interval if interval is not None else prefs.domain
        return cls.initfun_adaptive(lambda x: x, interval=interval)

    @classmethod
    def initfun(cls, fun: Any = None, n: Any = None, *, interval: Any = None) -> "Trigtech":
        """Convenience constructor: adaptive if *n* is None, fixed-length otherwise."""
        if n is None:
            return cls.initfun_adaptive(fun, interval=interval)
        return cls.initfun_fixedlen(fun, n, interval=interval)

    @classmethod
    def initfun_fixedlen(cls, fun: Any = None, n: Any = None, *, interval: Any = None) -> "Trigtech":
        """Initialise a Trigtech from callable *fun* using *n* equispaced points."""
        if n is None:
            raise ValueError("initfun_fixedlen requires the n parameter to be specified")  # noqa: TRY003
        points = cls._trigpts(int(n))
        values = fun(points)
        coeffs = cls._vals2coeffs(values)
        return cls(coeffs, interval=interval)

    @classmethod
    def initfun_adaptive(cls, fun: Any = None, *, interval: Any = None) -> "Trigtech":
        """Initialise a Trigtech from callable *fun* using the adaptive constructor."""
        interval = interval if interval is not None else prefs.domain
        interval = Interval(*interval)
        coeffs = _trig_adaptive(cls, fun, hscale=interval.hscale)
        return cls(coeffs, interval=interval)

    @classmethod
    def initvalues(cls, values: Any = None, *, interval: Any = None) -> "Trigtech":
        """Initialise a Trigtech from function values at equispaced points."""
        return cls(cls._vals2coeffs(np.asarray(values)), interval=interval)

    # ------------------------------------------------------------------
    #  core dunder methods
    # ------------------------------------------------------------------

    def __init__(self, coeffs: Any, interval: Any = None) -> None:
        """Initialise a Trigtech with FFT-order *coeffs* on *interval*.

        Coefficients are always stored as complex128.  The :attr:`iscomplex`
        property returns True only when the function *values* are complex
        (i.e., the coefficients do **not** satisfy the conjugate-symmetry
        condition C_{n-k} ≈ conj(C_k)).

        Args:
            coeffs: 1-D array of Fourier coefficients in NumPy FFT order.
            interval: Two-element interval [a, b].  Defaults to ``prefs.domain``.
        """
        interval = interval if interval is not None else prefs.domain
        self._coeffs = np.array(coeffs, dtype=complex)
        self._interval = Interval(*interval)

    def __call__(self, x: Any, how: str = "fft") -> Any:
        """Evaluate the Trigtech at points *x* via the DFT summation formula.

        f(x) = Σ_k  coeffs[k] * exp(i*π*ω_k*(x+1))

        where ω_k = fftfreq(n)*n gives integer frequencies in FFT order.
        For real-valued functions the imaginary part of the result is discarded.

        Args:
            x: Evaluation points in [-1, 1].
            how: Ignored; present for interface compatibility with Chebtech.
        """
        if self.isempty:
            return np.array([])
        scalar = np.isscalar(x)
        x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()

        if self.isconst:
            c0 = self._coeffs[0].real if not self.iscomplex else self._coeffs[0]
            out = c0 * np.ones(x.size)
            return float(out[0]) if scalar else out

        n = self.size
        freqs = np.fft.fftfreq(n) * n  # [0, 1, …, n//2, -(n//2)+1, …, -1]
        # shape: (len(x), n) @ (n,) → (len(x),)
        phases = np.exp(1j * np.pi * np.outer(x + 1.0, freqs))
        result = phases @ self._coeffs
        if not self.iscomplex:
            result = result.real
        return float(result[0]) if scalar else result

    def __repr__(self) -> str:  # pragma: no cover
        """Return a concise string representation."""
        return f"<{self.__class__.__name__}{{{self.size}}}>"

    # ------------------------------------------------------------------
    #  properties
    # ------------------------------------------------------------------

    @property
    def coeffs(self) -> np.ndarray:
        """Fourier coefficients in NumPy FFT order (always complex128)."""
        return self._coeffs

    @property
    def interval(self) -> Interval:
        """Interval that the Trigtech is mapped to."""
        return self._interval

    @property
    def size(self) -> int:
        """Number of stored Fourier coefficients."""
        return self._coeffs.size

    @property
    def isempty(self) -> bool:
        """True if the Trigtech has no coefficients."""
        return self.size == 0

    @property
    def iscomplex(self) -> bool:
        """True if the function is complex-valued (values have a non-negligible imaginary part).

        This is determined by checking whether the Fourier coefficients violate
        the conjugate-symmetry condition C_{n-k} ≈ conj(C_k) that holds for
        every real-valued periodic function.
        """
        n = self.size
        if n <= 1:
            return bool(np.any(np.abs(np.imag(self._coeffs)) > 0))
        abs_max = float(np.max(np.abs(self._coeffs)))
        if abs_max == 0.0:
            return False
        tol = 1e-8 * abs_max
        # mirror[k-1] = conj(C_{n-k}) for k = 1,...,n-1
        mirror = np.conj(self._coeffs[-1:0:-1])
        return bool(np.any(np.abs(self._coeffs[1:] - mirror) > tol))

    @property
    def isconst(self) -> bool:
        """True if the Trigtech represents a constant (single coefficient)."""
        return self.size == 1

    @property
    def isperiodic(self) -> bool:
        """Always True: Trigtech always represents a periodic function."""
        return True

    @property
    @self_empty(0.0)
    def vscale(self) -> float:
        """Estimate the vertical scale (max |f|)."""
        return float(np.abs(np.asarray(coerce_list(self.values()))).max())

    # ------------------------------------------------------------------
    #  utilities
    # ------------------------------------------------------------------

    def copy(self) -> "Trigtech":
        """Return a deep copy."""
        return self.__class__(self._coeffs.copy(), interval=self._interval.copy())

    def imag(self) -> "Trigtech":
        """Return the imaginary part of the function as a real-valued Trigtech.

        For a complex function f(x) = g(x) + i·h(x), the Fourier coefficients
        of h(x) are H[k] = (D[k] - conj(D[n-k])) / (2i) for k ≥ 1,
        and H[0] = Im(D[0]).
        """
        if not self.iscomplex:
            return self.initconst(0.0, interval=self._interval)
        n = self.size
        c = self._coeffs
        imag_c = np.zeros(n, dtype=complex)
        imag_c[0] = np.imag(c[0])
        if n > 1:
            mirror = np.conj(c[-1:0:-1])  # conj(c[n-1]), ..., conj(c[1])
            imag_c[1:] = (c[1:] - mirror) / (2j)
        return self.__class__(imag_c, self._interval)

    def prolong(self, n: int) -> "Trigtech":
        """Return a Trigtech of length *n* (truncate or zero-pad in frequency space).

        The operation aligns DC components of the source and target DC-centred
        representations, then either pads with zeros (n > m) or slices (n < m).
        This correctly handles the asymmetry between even- and odd-length arrays.
        """
        m = self.size
        if n == m:
            return self.copy()

        centered = np.fft.fftshift(self._coeffs)
        dc_src = m // 2
        dc_tgt = n // 2

        if n > m:
            padded = np.zeros(n, dtype=centered.dtype)
            start = dc_tgt - dc_src
            padded[start : start + m] = centered
            return self.__class__(np.fft.ifftshift(padded), interval=self._interval)
        else:
            start = dc_src - dc_tgt
            truncated = centered[start : start + n]
            return self.__class__(np.fft.ifftshift(truncated), interval=self._interval)

    def real(self) -> "Trigtech":
        """Return the real part of the function as a real-valued Trigtech.

        For a complex function f(x) = g(x) + i·h(x), the Fourier coefficients
        of g(x) are G[k] = (D[k] + conj(D[n-k])) / 2 for k ≥ 1,
        and G[0] = Re(D[0]).
        """
        if not self.iscomplex:
            return self
        n = self.size
        c = self._coeffs
        real_c = np.zeros(n, dtype=complex)
        real_c[0] = np.real(c[0])
        if n > 1:
            mirror = np.conj(c[-1:0:-1])  # conj(c[n-1]), ..., conj(c[1])
            real_c[1:] = (c[1:] + mirror) / 2
        return self.__class__(real_c, self._interval)

    def simplify(self) -> "Trigtech":
        """Truncate high-frequency Fourier coefficients that are below tolerance.

        Uses the same one-sided symmetric-maximum criterion as the adaptive
        constructor: the highest-frequency mode retained is the one where
        ``max(|c_k|, |c_{-k}|) / vscale > tol``.
        """
        oldlen = len(self._coeffs)
        longself = self.prolong(max(17, oldlen))
        n = longself.size
        tol = prefs.eps * max(self._interval.hscale, 1)

        centered = np.fft.fftshift(longself._coeffs)
        dc_idx = n // 2
        abs_max = float(np.max(np.abs(centered)))
        if abs_max == 0.0:
            return self.initconst(0.0, interval=self._interval)

        abs_sym = np.zeros(dc_idx + 1)
        for ki in range(dc_idx + 1):
            p = centered[dc_idx + ki] if dc_idx + ki < n else 0.0
            q = centered[dc_idx - ki]
            abs_sym[ki] = max(abs(p), abs(q)) / abs_max

        above = np.where(abs_sym > tol)[0]
        if len(above) == 0:
            return self.initconst(0.0, interval=self._interval)
        max_k = int(above[-1])
        max_k = min(max_k, oldlen // 2)  # don't exceed original size

        start = dc_idx - max_k
        end = dc_idx + max_k + 1
        return self.__class__(np.fft.ifftshift(centered[start:end]), interval=self._interval)

    def values(self) -> np.ndarray:
        """Function values at the n equispaced points x_j = -1 + 2j/n."""
        return self._coeffs2vals(self._coeffs)

    def _coeffs_to_plotorder(self) -> np.ndarray:
        """Return coefficients in DC-centred (human-readable) order.

        Equivalent to ``numpy.fft.fftshift(self.coeffs)``:
        ordering is [c_{-n//2}, …, c_{-1}, c_0, c_1, …, c_{n//2-1}].
        """
        return np.fft.fftshift(self._coeffs)

    # ------------------------------------------------------------------
    #  algebra
    # ------------------------------------------------------------------

    @self_empty()
    def __add__(self, f: Any) -> "Trigtech":
        """Add a scalar or another Trigtech."""
        cls = self.__class__
        if np.isscalar(f):
            dtype: Any = complex if np.iscomplexobj(f) else self._coeffs.dtype
            cfs = np.array(self._coeffs, dtype=dtype)
            cfs[0] += f  # add to DC component
            return cls(cfs, interval=self._interval)
        if f.isempty:
            return f.copy()
        g = self
        n, m = g.size, f.size
        if n < m:
            g = g.prolong(m)
        elif m < n:
            f = f.prolong(n)
        cfs = f.coeffs + g.coeffs
        eps = prefs.eps
        tol = 0.5 * eps * max(f.vscale, g.vscale)
        if np.all(np.abs(cfs) < tol):
            return cls.initconst(0.0, interval=self._interval)
        return cls(cfs, interval=self._interval)

    @self_empty()
    def __div__(self, f: Any) -> "Trigtech":
        """Divide this Trigtech by a scalar or another Trigtech."""
        cls = self.__class__
        if np.isscalar(f):
            return cls(self._coeffs / np.asarray(f), interval=self._interval)
        if f.isempty:
            return f.copy()
        return cls.initfun_adaptive(lambda x: self(x) / f(x), interval=self._interval)

    __truediv__ = __div__

    @self_empty()
    def __mul__(self, g: Any) -> "Trigtech":
        """Multiply this Trigtech by a scalar or another Trigtech.

        Trig-polynomial multiplication is circular convolution in frequency
        space.  We implement this cleanly by evaluating both on a grid of
        size n1 + n2 (sufficient to avoid aliasing), multiplying pointwise,
        and taking the FFT.
        """
        cls = self.__class__
        if np.isscalar(g):
            return cls(g * self._coeffs, interval=self._interval)
        if g.isempty:
            return g.copy()
        n = self.size + g.size
        f_vals = self.prolong(n).values()
        g_vals = g.prolong(n).values()
        return cls(cls._vals2coeffs(f_vals * g_vals), interval=self._interval)

    def __neg__(self) -> "Trigtech":
        """Return the negation."""
        return self.__class__(-self._coeffs, interval=self._interval)

    def __pos__(self) -> "Trigtech":
        """Return self (unary plus)."""
        return self

    @self_empty()
    def __pow__(self, f: Any) -> "Trigtech":
        """Raise this Trigtech to a power *f* (scalar or Trigtech)."""

        def powfun(fn: Any, x: Any) -> Any:
            return fn if np.isscalar(fn) else fn(x)

        return self.__class__.initfun_adaptive(
            lambda x: np.power(self(x), powfun(f, x)),
            interval=self._interval,
        )

    def __rdiv__(self, f: Any) -> "Trigtech":
        """Compute f / self where *f* is a scalar."""
        return self.__class__.initfun_adaptive(
            lambda x: (0.0 * x + f) / self(x),
            interval=self._interval,
        )

    __radd__ = __add__
    __rmul__ = __mul__
    __rtruediv__ = __rdiv__

    def __rsub__(self, f: Any) -> "Trigtech":
        """Compute f - self."""
        return -(self - f)

    @self_empty()
    def __rpow__(self, f: Any) -> "Trigtech":
        """Compute f ** self."""
        return self.__class__.initfun_adaptive(
            lambda x: np.power(f, self(x)),
            interval=self._interval,
        )

    def __sub__(self, f: Any) -> "Trigtech":
        """Subtract *f* (scalar or Trigtech) from this Trigtech."""
        return self + (-f)

    # ------------------------------------------------------------------
    #  rootfinding
    # ------------------------------------------------------------------

    def roots(self, sort: bool | None = None) -> np.ndarray:
        """Find the roots of this Trigtech on [-1, 1].

        Converts to a Chebyshev representation via re-sampling on Chebyshev
        points and delegates to the Chebtech colleague-matrix root-finder.

        Args:
            sort: If True, sort the roots in ascending order.  Defaults to
                ``prefs.sortroots``.
        """
        from .algorithms import newtonroots, rootsunit
        from .chebtech import Chebtech

        sort = sort if sort is not None else prefs.sortroots

        if self.isempty:
            return np.array([])

        # Sample on a Chebyshev grid and fit a Chebtech of the same resolution
        n = max(2 * self.size + 1, 33)
        cheb_pts = Chebtech._chebpts(n)
        vals = self(cheb_pts)
        ct = Chebtech(Chebtech._vals2coeffs(vals))
        rts = rootsunit(ct.coeffs)
        rts = newtonroots(ct, rts)
        rts = np.clip(rts, -1.0, 1.0)
        return np.sort(rts) if sort else rts

    # ------------------------------------------------------------------
    #  calculus
    # ------------------------------------------------------------------

    @self_empty(resultif=0.0)
    def sum(self) -> Any:
        """Definite integral of the Trigtech over [-1, 1].

        Only the DC coefficient contributes:
            ∫_{-1}^{1} exp(i*π*k*(x+1)) dx = 0  for k ≠ 0
            ∫_{-1}^{1} 1 dx = 2               for k = 0
        """
        return 2.0 * float(np.real(self._coeffs[0]))

    @self_empty()
    def cumsum(self) -> "Trigtech":
        """Indefinite integral, zero at x = -1, in Fourier coefficient space.

        For mode k ≠ 0:  antiderivative coefficient = c_k / (i*π*ω_k)
        For mode k = 0:  set to the constant needed so that F(-1) = 0.

        Note: if the DC component (self.coeffs[0]) is non-zero the true
        antiderivative contains a linear trend and is not periodic.  We still
        return a Trigtech representing the *periodic* part, adjusted so that
        the result evaluates to 0 at x = -1.
        """
        n = self.size
        c = self._coeffs.copy()
        freqs = np.fft.fftfreq(n) * n  # FFT-order integer frequencies

        int_c = np.zeros(n, dtype=complex)
        mask = freqs != 0
        int_c[mask] = c[mask] / (1j * np.pi * freqs[mask])

        # Enforce F(-1) = 0.
        # F(x) = Σ_k int_c[k] * exp(i*π*ω_k*(x+1))
        # At x = -1:  exp(i*π*ω_k*0) = 1  for all k,  so F(-1) = Σ int_c
        # Set int_c[0] so that sum(int_c) = 0.
        int_c[0] = -np.sum(int_c[1:])
        return self.__class__(int_c, interval=self._interval)

    @self_empty()
    def diff(self) -> "Trigtech":
        """Derivative via the Fourier multiplier i*π*ω_k.

        d/dx [c_k * exp(i*π*ω_k*(x+1))] = i*π*ω_k * c_k * exp(i*π*ω_k*(x+1))
        """
        if self.isconst:
            return self.__class__(np.array([0.0 + 0.0j]), interval=self._interval)
        n = self.size
        freqs = np.fft.fftfreq(n) * n
        d_coeffs = (1j * np.pi * freqs) * self._coeffs
        return self.__class__(d_coeffs, interval=self._interval)

    # ------------------------------------------------------------------
    #  static helpers (FFT ↔ values)
    # ------------------------------------------------------------------

    @staticmethod
    def _trigpts(n: int) -> np.ndarray:
        """Return *n* equispaced points on [-1, 1)."""
        if n == 0:
            return np.array([])
        return -1.0 + 2.0 * np.arange(n) / n

    @staticmethod
    def _vals2coeffs(vals: Any) -> np.ndarray:
        """Convert values at equispaced points to FFT coefficients (divided by n).

        Always returns complex128, even for real-valued inputs, because Fourier
        coefficients for functions such as sin are purely imaginary and would be
        discarded if forced to real.

        Inverse of ``_coeffs2vals``.
        """
        vals = np.asarray(vals)
        n = vals.size
        if n == 0:
            return np.array([], dtype=complex)
        return np.fft.fft(vals) / n

    @staticmethod
    def _coeffs2vals(coeffs: Any) -> np.ndarray:
        """Convert FFT coefficients (divided by n) to values at equispaced points.

        Inverse of ``_vals2coeffs``.
        """
        coeffs = np.asarray(coeffs, dtype=complex)
        n = coeffs.size
        if n == 0:
            return np.array([], dtype=float)
        vals = n * np.fft.ifft(coeffs)
        # Discard negligible imaginary parts for conjugate-symmetric coefficients
        max_real = float(np.max(np.abs(np.real(vals))))
        if float(np.max(np.abs(np.imag(vals)))) < 1e-10 * max(max_real, 1.0):
            return np.real(vals)
        return vals

    # ------------------------------------------------------------------
    #  plotting
    # ------------------------------------------------------------------

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Plot the Trigtech over [-1, 1].

        Args:
            ax: Matplotlib axes.  If None, uses the current axes.
            **kwargs: Forwarded to matplotlib.

        Returns:
            The axes on which the plot was drawn.
        """
        return plotfun(self, (-1, 1), ax=ax, **kwargs)

    def plotcoeffs(self, ax: Any = None, **kwargs: Any) -> Any:
        """Plot the absolute Fourier coefficient magnitudes in DC-centred order.

        Uses ``_coeffs_to_plotorder()`` so the horizontal axis runs from
        the most-negative frequency on the left to the most-positive on
        the right, with DC in the centre.

        Args:
            ax: Matplotlib axes.  If None, uses the current axes.
            **kwargs: Forwarded to matplotlib.

        Returns:
            The axes on which the plot was drawn.
        """
        ax = ax or plt.gca()
        return plotfuncoeffs(np.abs(self._coeffs_to_plotorder()), ax=ax, **kwargs)
