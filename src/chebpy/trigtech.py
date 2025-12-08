"""Implementation of Fourier (trigonometric) technology for periodic function approximation.

This module provides the Trigtech class, which represents periodic functions using
Fourier series expansions on the interval [0, 2π]. It serves as the periodic analog
to Chebtech for smooth, periodic functions.

The Trigtech class implements core functionality for working with Fourier expansions, including:
- Function evaluation using FFT or trigonometric sum
- Algebraic operations (addition, multiplication, etc.)
- Calculus operations (differentiation, integration, etc.)
- Orthogonality properties of Fourier basis

These classes are primarily used internally by higher-level classes like Bndfun
and Chebfun when periodic boundary conditions are specified, rather than being
used directly by end users.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

from .algorithms import standard_chop
from .decorators import self_empty
from .plotting import plotfun, plotfuncoeffs
from .settings import _preferences as prefs
from .smoothfun import Smoothfun
from .utilities import Interval, coerce_list


class Trigtech(Smoothfun):
    """Class for representing periodic functions using Fourier series.

    Trigtech objects use FFT-based Fourier series representation:
        f(x) = Σ c_k exp(ikx)

    The domain is the periodic interval [0, 2π], and functions are represented
    using complex Fourier coefficients that enable fast operations via FFT.

    The user will rarely work with these classes directly so we make
    several assumptions regarding input data types.
    """

    @classmethod
    def initconst(cls, c, *, interval=None):
        """Initialise a Trigtech from a constant c."""
        if not np.isscalar(c):
            raise ValueError(c)
        if isinstance(c, int):
            c = float(c)
        return cls(np.array([c], dtype=complex), interval=interval)

    @classmethod
    def initempty(cls, *, interval=None):
        """Initialise an empty Trigtech."""
        return cls(np.array([], dtype=complex), interval=interval)

    @classmethod
    def initidentity(cls, *, interval=None):
        """Trigtech representation of f(x) = x is not supported.

        The identity function f(x) = x on [0, 2π] is not periodic
        (f(0) = 0 ≠ 2π = f(2π)), so it cannot be accurately represented
        using Fourier series. For periodic functions, there is no natural
        identity function analogous to Chebtech's representation of x on [-1, 1].

        Raises:
            NotImplementedError: Always, as this operation is not mathematically valid.
        """
        raise NotImplementedError(
            "initidentity() is not supported for Trigtech. "
            "The identity function f(x) = x on [0, 2π] is not periodic "
            "(f(0) = 0 ≠ 2π = f(2π)) and cannot be accurately represented "
            "using Fourier series. Consider using Chebtech for non-periodic functions."
        )

    @classmethod
    def initfun(cls, fun, n=None, *, interval=None):
        """Convenience constructor to automatically select the adaptive or fixedlen constructor.

        This constructor automatically selects between the adaptive or fixed-length
        constructor based on the input arguments passed.
        """
        if n is None:
            return cls.initfun_adaptive(fun, interval=interval)
        else:
            return cls.initfun_fixedlen(fun, n, interval=interval)

    @classmethod
    def initfun_fixedlen(cls, fun, n, *, interval=None):
        """Initialise a Trigtech from the callable fun using n degrees of freedom.

        This constructor creates a Trigtech representation of the function using
        a fixed number of degrees of freedom specified by n.

        Args:
            fun (callable): Function to approximate.
            n (int): Number of degrees of freedom (points).
            interval (list, optional): Interval [a, b]. Defaults to [0, 2π].
        """
        interval = interval if interval is not None else [0, 2 * np.pi]
        interval = Interval(*interval)
        # Get equally-spaced points in [0, 1]
        t = np.arange(n) / n if n > 0 else np.array([])
        # Map to interval [a, b]: equally spaced points starting at a
        a, b = interval
        points_mapped = a + (b - a) * t
        values = fun(points_mapped)
        # Handle scalar returns from constant functions like lambda x: 1.0
        values = np.atleast_1d(values)
        if values.size == 1:
            values = np.broadcast_to(values, points_mapped.shape)
        coeffs = cls._vals2coeffs(values)
        return cls(coeffs, interval=interval)

    @classmethod
    def initfun_adaptive(cls, fun, *, interval=None, minpow2=None):
        """Initialise a Trigtech from the callable fun utilising the adaptive constructor.

        This constructor uses an adaptive algorithm to determine the appropriate
        number of degrees of freedom needed to represent the function.

        Args:
            fun (callable): Function to approximate
            interval (list, optional): Interval [a, b]. Defaults to [0, 2π].
            minpow2 (int, optional): Minimum power of 2 for number of points.
        """
        interval = interval if interval is not None else [0, 2 * np.pi]
        interval = Interval(*interval)
        coeffs = cls._adaptive_trig(fun, hscale=interval.hscale, minpow2=minpow2, interval=interval)
        return cls(coeffs, interval=interval)

    @classmethod
    def _pair_fourier_coeffs(cls, coeffs):
        """Pair Fourier coefficients for use with standard_chop.

        Following MATLAB Chebfun's standardCheck for trigtech, this pairs
        up k and -k frequency coefficients and prepares them for standardchop.

        Args:
            coeffs (np.ndarray): Fourier coefficients in FFT order

        Returns:
            np.ndarray: Paired coefficients suitable for standard_chop
        """
        n = len(coeffs)
        # MATLAB: coeffs = abs(f.coeffs(end:-1:1,:))
        abs_coeffs = np.abs(coeffs[::-1])

        # Pair up k and -k coefficients following MATLAB logic exactly
        is_even = n % 2 == 0
        if is_even:
            n2 = n // 2
            # MATLAB (1-indexed): [coeffs(n) ; coeffs(n-1:-1:n/2+1) + coeffs(1:n/2-1) ; coeffs(n/2)]
            # Python (0-indexed):
            part1 = abs_coeffs[n - 1 : n]  # coeffs(n) -> coeffs[n-1]
            part2 = (
                abs_coeffs[n - 2 : n2 - 1 : -1] + abs_coeffs[0 : n2 - 1]
            )  # Pair coeffs[n-1:-1:n/2+1] with coeffs[1:n/2-1]
            part3 = abs_coeffs[n2 - 1 : n2]  # coeffs(n/2) -> coeffs[n2-1]
            paired = np.concatenate([part1, part2, part3])
        else:
            n2 = (n + 1) // 2
            # MATLAB (1-indexed): [coeffs(n:-1:(n+1)/2+1) + coeffs(1:(n+1)/2-1) ; coeffs((n+1)/2)]
            # Python (0-indexed):
            part1 = (
                abs_coeffs[n - 1 : n2 - 1 : -1] + abs_coeffs[0 : n2 - 1]
            )  # Pair coeffs[n:-1:(n+1)/2+1] with coeffs[1:(n+1)/2-1]
            part2 = abs_coeffs[n2 - 1 : n2]  # coeffs((n+1)/2) -> coeffs[n2-1]
            paired = np.concatenate([part1, part2])

        # NOTE: MATLAB does flipud, but that's because MATLAB's trigtech stores coefficients
        # in a different order than Python's FFT. After pairing, we already have:
        # paired = [DC, k=±1, k=±2, ..., k=±(n/2-1), Nyquist]
        # which is already in the right order (low freq first = large coeffs first for most functions).
        # So we DON'T flip in Python.

        # MATLAB: [coeffs(1,:) ; kron(coeffs(2:end,:), [1 ; 1])]
        # Double up all except first element
        if len(paired) > 1:
            doubled = np.concatenate([paired[0:1], np.repeat(paired[1:], 2)])
        else:
            doubled = paired

        return doubled

    @classmethod
    def _adaptive_trig(cls, fun, hscale=1, maxpow2=None, minpow2=None, interval=None):
        """Adaptive constructor for Trigtech using Fourier points.

        Implements MATLAB's standardCheck happiness criterion using standardChop
        on paired and doubled Fourier coefficients.

        Args:
            fun (callable): Function to approximate
            hscale (float): Scale factor for tolerance
            maxpow2 (int): Maximum power of 2 to try
            minpow2 (int): Minimum power of 2 to try (default 4 for n=16)
            interval (Interval): Interval [a, b] for mapping points

        Returns:
            np.ndarray: Fourier coefficients
        """
        minpow2 = minpow2 if minpow2 is not None else 4  # Start with 2^4 = 16 points
        # Use smaller maxpow2 for trigtech to prevent hangs (2^12 = 4096 points max)
        # Fourier methods converge faster than Chebyshev for smooth periodic functions
        maxpow2 = maxpow2 if maxpow2 is not None else min(prefs.maxpow2, 12)
        epslevel = prefs.eps * max(hscale, 1)

        # Set up interval mapping
        if interval is None:
            interval = Interval(0, 2 * np.pi)
        a, b = interval

        for k in range(minpow2, max(minpow2, maxpow2) + 1):
            n = 2**k
            # Get equally-spaced points in [0, 1]
            t = np.arange(n) / n
            # Map to interval [a, b]: equally spaced points starting at a
            points_mapped = a + (b - a) * t
            values = fun(points_mapped)
            # Handle scalar returns from constant functions like lambda x: 1.0
            values = np.atleast_1d(values)
            if values.size == 1:
                values = np.broadcast_to(values, points_mapped.shape)

            # Check for zero function
            vscale = np.max(np.abs(values))
            if vscale == 0:
                return np.array([0.0 + 0j])

            coeffs = cls._vals2coeffs(values)

            # MATLAB standardCheck: pair, flip, and double coefficients for standardChop
            paired_doubled = cls._pair_fourier_coeffs(coeffs)

            # Normalize tolerance by vscale
            tol = epslevel

            # Call standardChop to find cutoff
            cutoff = standard_chop(paired_doubled, tol=tol)

            # Happy if cutoff < n (MATLAB line 85: ishappy = cutoff < n)
            # Note: cutoff is compared to original n, not to len(paired_doubled)
            # Allow n=16 to be happy (changed from n >= 17 to n >= 16)
            is_happy = (n >= 16) and (cutoff < n)

            if is_happy:
                return coeffs

        # Did not converge
        warnings.warn(
            f"Trigtech adaptive constructor did not converge after {maxpow2 - minpow2 + 1} "
            f"iterations: using {n} points. Function may be too oscillatory or non-periodic.",
            stacklevel=2,
        )
        return coeffs

    @classmethod
    def initvalues(cls, values, *, interval=None):
        """Initialise a Trigtech from an array of values at equally-spaced (Fourier) points."""
        return cls(cls._vals2coeffs(values), interval=interval)

    def __init__(self, coeffs, interval=None):
        """Initialize a Trigtech object.

        This method initializes a new Trigtech object with the given Fourier coefficients
        and interval. If no interval is provided, the default periodic interval [0, 2π]
        is used.

        Args:
            coeffs (array-like): The Fourier coefficients (complex exponential basis).
            interval (array-like, optional): The interval on which the function
                is defined. Defaults to [0, 2π].
        """
        interval = interval if interval is not None else [0, 2 * np.pi]
        self._coeffs = np.array(coeffs, dtype=complex)
        self._interval = Interval(*interval)

    def __call__(self, x, how="direct"):
        """Evaluate the Trigtech at the given points.

        Args:
            x: Points at which to evaluate the Trigtech.
            how (str, optional): Method to use for evaluation. Either "direct" or "fft".
                Defaults to "direct".

        Returns:
            The values of the Trigtech at the given points.

        Raises:
            ValueError: If the specified method is not supported.
        """
        method = {
            "direct": self.__call__direct,
            "fft": self.__call__fft,
        }
        try:
            return method[how](x)
        except KeyError:
            raise ValueError(how)

    def __call__direct(self, x):
        """Evaluate using direct summation of Fourier series."""
        x = np.asarray(x)
        n = self.size
        coeffs = self.coeffs

        # Handle empty case
        if n == 0:
            return np.zeros_like(x, dtype=complex)

        # Get frequency indices (integer mode numbers)
        # fftfreq with d=1/n gives [0, 1, 2, ..., n//2, -(n//2-1), ..., -1]
        k = np.fft.fftfreq(n, d=1.0 / n)

        # Scale frequencies for the interval [a, b]
        # For a function on [a,b], Fourier basis is exp(2πik(x-a)/(b-a))
        # This is because FFT assumes periodicity starting at 0, but our interval starts at a
        a, b = self.interval
        L = b - a
        omega = 2.0 * np.pi / L  # Angular frequency scaling

        # Evaluate: f(x) = Σ c_k exp(i*omega*k*(x-a))
        result = np.zeros_like(x, dtype=complex)
        for i, coeff in enumerate(coeffs):
            result += coeff * np.exp(1j * omega * k[i] * (x - a))

        # Always return complex result - let higher-level code decide whether to take real part
        # This is critical for differentiation operations where intermediate results may have
        # significant imaginary components that must be preserved
        return result

    def __call__fft(self, x):
        """Evaluate using FFT (only works for points on the standard grid)."""
        # FFT evaluation is only efficient for grid points; always use direct
        return self.__call__direct(x)

    def __repr__(self):  # pragma: no cover
        """Return a string representation of the Trigtech.

        Returns:
            str: A string representation of the Trigtech.
        """
        out = f"<{self.__class__.__name__}{{{self.size}}}>"
        return out

    # ------------
    #  properties
    # ------------
    @property
    def coeffs(self):
        """Fourier expansion coefficients in the exp(ikx) basis."""
        return self._coeffs

    @property
    def interval(self):
        """Interval that Trigtech is mapped to."""
        return self._interval

    @property
    def size(self):
        """Return the size of the object."""
        return self.coeffs.size

    @property
    def isempty(self):
        """Return True if the Trigtech is empty."""
        return self.size == 0

    @property
    def iscomplex(self):
        """Determine whether the underlying function is complex or real valued."""
        # Check if imaginary part is negligible
        vals = self.values()
        if vals.size == 0:
            return False
        max_imag = np.max(np.abs(vals.imag))
        max_real = np.max(np.abs(vals.real))
        return max_imag > 1e-13 * (max_real + 1e-14)

    @property
    def isconst(self):
        """Return True if the Trigtech represents a constant."""
        return self.size == 1

    @property
    @self_empty(0.0)
    def vscale(self):
        """Estimate the vertical scale of a Trigtech."""
        return np.abs(coerce_list(self.values())).max()

    # -----------
    #  utilities
    # -----------
    def copy(self):
        """Return a deep copy of the Trigtech."""
        return self.__class__(self.coeffs.copy(), interval=self.interval.copy())

    def imag(self):
        """Return the imaginary part of the Trigtech.

        Returns:
            Trigtech: A new Trigtech representing the imaginary part of this Trigtech.
                If this Trigtech is real-valued, returns a zero Trigtech.
        """
        if self.iscomplex:
            return self.__class__(np.imag(self.coeffs), self.interval)
        else:
            return self.initconst(0, interval=self.interval)

    def prolong(self, n):
        """Return a Trigtech of length n.

        Obtained either by truncating if n < self.size or zero-padding if n > self.size.
        In all cases a deep copy is returned.

        For Trigtech, we cannot simply append/truncate coefficients because FFT
        coefficients have a specific ordering. Instead, we resample the function.
        """
        m = self.size
        cls = self.__class__

        if n == m:
            return self.copy()

        # For Trigtech, we need to resample rather than just padding coefficients
        # because FFT coefficients have positive and negative frequencies interleaved
        # Resampling ensures the function is preserved correctly

        # Use the function values approach: evaluate at old points, resample at new points
        # But for efficiency, we can use initfun_fixedlen which will call the function
        # at the new points
        return cls.initfun_fixedlen(lambda x: self(x), n, interval=self.interval)

    def real(self):
        """Return the real part of the Trigtech.

        Returns:
            Trigtech: A new Trigtech representing the real part of this Trigtech.
                If this Trigtech is already real-valued, returns self.
        """
        if self.iscomplex:
            return self.__class__(np.real(self.coeffs), self.interval)
        else:
            return self

    def simplify(self):
        """Simplify the Trigtech by truncating negligible coefficients.

        Returns a Trigtech comprised of a copy of the truncated coefficients.
        Uses conjugate-pair-aware chopping to preserve real-valued functions.

        Note: For Fourier coefficients, we find the highest significant frequency
        and reconstruct using fewer points to represent that bandwidth.
        """
        n = self.size
        if n == 0:
            return self.copy()

        coeffs = self.coeffs
        vscale = self.vscale
        if vscale == 0:
            return self.initconst(0, interval=self.interval)

        tol = prefs.eps * max(self.interval.hscale, 1) * vscale

        # Find the highest significant frequency
        # FFT ordering: [0, 1, 2, ..., n//2, -(n//2-1 or n//2), ..., -2, -1]
        abs_coeffs = np.abs(coeffs)
        freq = np.fft.fftfreq(n, d=1.0 / n)
        abs_freq = np.abs(freq)

        # Find indices where coefficients are significant
        significant = abs_coeffs > tol

        if not np.any(significant):
            return self.initconst(0, interval=self.interval)

        # Find maximum significant frequency
        sig_indices = np.where(significant)[0]
        max_sig_freq = np.max(abs_freq[sig_indices])

        # To represent frequency k, we need at least 2*k+1 points (Nyquist)
        # For safety and to avoid aliasing, use 2*(k+1)
        n_min = int(2 * (max_sig_freq + 1))

        # Round up to power of 2 for FFT efficiency
        n_keep = 2 ** int(np.ceil(np.log2(n_min))) if n_min > 1 else 1

        # Don't increase size
        n_keep = min(n_keep, n)

        if n_keep >= n:
            return self.copy()

        # Reconstruct with fewer points rather than simple truncation
        # This properly handles the fact that negative frequencies are at the end
        vals = self.values()  # Get function values
        points_old = self._trigpts(n)
        points_new = self._trigpts(n_keep)

        # Resample at new grid
        vals_new = np.interp(points_new, points_old, vals.real)
        if np.iscomplexobj(vals):
            vals_new = vals_new + 1j * np.interp(points_new, points_old, vals.imag)

        # Convert back to coefficients
        coeffs_new = self._vals2coeffs(vals_new)
        return self.__class__(coeffs_new, interval=self.interval)

    def values(self):
        """Function values at equally-spaced (Fourier) points."""
        return self._coeffs2vals(self.coeffs)

    # ---------
    #  algebra
    # ---------
    def __add__(self, f):
        """Add a scalar or another Trigtech to this Trigtech.

        Args:
            f: A scalar or another Trigtech to add to this Trigtech.

        Returns:
            Trigtech: A new Trigtech representing the sum.
        """
        cls = self.__class__
        if np.isscalar(f):
            cfs = self.coeffs.copy()
            if cfs.size > 0:
                cfs[0] += f
            else:
                cfs = np.array([f], dtype=complex)
            return cls(cfs, interval=self.interval)
        else:
            if f.isempty:
                return self.copy()
            if self.isempty:
                return f.copy()

            # Handle mixing with other Smoothfun types (e.g., Chebtech)
            # This should ideally not happen - periodic problems should use Trigtech throughout
            # But if it does, convert the other operand to Trigtech by resampling
            if not isinstance(f, cls):
                # Import Chebtech to check type
                from .chebtech import Chebtech

                # Convert f to Trigtech by sampling at enough points
                # Use fixed-length to avoid adaptive convergence issues
                target_size = 2 * max(self.size, getattr(f, "size", 16))
                target_size = min(target_size, 512)  # Cap to avoid excessive computation

                # If f is a Chebtech, we need to map points from [a,b] to [-1,1]
                # because Chebtech always works on [-1,1] interval
                if isinstance(f, Chebtech):
                    a, b = self.interval

                    def wrapped_f(x):
                        # Map x from [a, b] to [-1, 1] for Chebtech evaluation
                        x_mapped = 2 * (x - a) / (b - a) - 1
                        return f(x_mapped)

                    f = cls.initfun(wrapped_f, n=target_size, interval=self.interval)
                else:
                    # For other Smoothfun types, assume they handle intervals correctly
                    f = cls.initfun(lambda x: f(x), n=target_size, interval=self.interval)

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
        """Divide this Trigtech by a scalar or another Trigtech.

        Args:
            f: A scalar or another Trigtech to divide this Trigtech by.

        Returns:
            Trigtech: A new Trigtech representing the quotient.
        """
        cls = self.__class__
        if np.isscalar(f):
            cfs = 1.0 / f * self.coeffs
            return cls(cfs, interval=self.interval)
        else:
            if f.isempty:
                return f.copy()
            return cls.initfun_adaptive(lambda x: self(x) / f(x), interval=self.interval)

    __truediv__ = __div__

    @self_empty()
    def __mul__(self, g):
        """Multiply this Trigtech by a scalar or another Trigtech.

        Args:
            g: A scalar or another Trigtech to multiply this Trigtech by.

        Returns:
            Trigtech: A new Trigtech representing the product.
        """
        cls = self.__class__
        if np.isscalar(g):
            cfs = g * self.coeffs
            return cls(cfs, interval=self.interval)
        else:
            if g.isempty:
                return g.copy()
            # Multiplication in Fourier space is convolution
            # For simplicity, use pointwise multiplication and recompute
            return cls.initfun_adaptive(lambda x: self(x) * g(x), interval=self.interval)

    def __neg__(self):
        """Return the negative of this Trigtech.

        Returns:
            Trigtech: A new Trigtech representing the negative of this Trigtech.
        """
        coeffs = -self.coeffs
        return self.__class__(coeffs, interval=self.interval)

    def __pos__(self):
        """Return this Trigtech (unary positive).

        Returns:
            Trigtech: This Trigtech (self).
        """
        return self

    @self_empty()
    def __pow__(self, f):
        """Raise this Trigtech to a power.

        Args:
            f: The exponent, which can be a scalar or another Trigtech.

        Returns:
            Trigtech: A new Trigtech representing this Trigtech raised to the power f.
        """
        return self.__class__.initfun_adaptive(
            lambda x: np.power(self(x), f(x) if callable(f) else f), interval=self.interval
        )

    def __radd__(self, f):
        """Right addition (commutative with __add__)."""
        return self + f

    @self_empty()
    def __rdiv__(self, f):
        """Right division: f / self."""
        return self.__class__.initfun_adaptive(lambda x: f / self(x), interval=self.interval)

    @self_empty()
    def __rpow__(self, f):
        """Right power: f ** self."""
        return self.__class__.initfun_adaptive(lambda x: np.power(f, self(x)), interval=self.interval)

    __rtruediv__ = __rdiv__
    __rmul__ = __mul__

    def __sub__(self, f):
        """Subtract a scalar or another Trigtech from this Trigtech.

        Args:
            f: A scalar or another Trigtech to subtract from this Trigtech.

        Returns:
            Trigtech: A new Trigtech representing the difference.
        """
        return self + (-f)

    def __rsub__(self, f):
        """Right subtraction: f - self."""
        return -(self - f)

    # NumPy ufunc support
    __array_priority__ = 1000

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handle numpy universal functions.

        This allows numpy functions like np.exp, np.sin, np.cos, etc. to work
        directly on Trigtech objects.
        """
        if method == "__call__":
            # Get the Trigtech object from inputs and maximum size
            trigtech_obj = None
            max_size = 0
            for inp in inputs:
                if isinstance(inp, Trigtech):
                    if trigtech_obj is None:
                        trigtech_obj = inp
                    max_size = max(max_size, inp.size)

            if trigtech_obj is None:
                return NotImplemented

            # Apply the ufunc by creating a new function adaptively
            def newfun(x):
                # Evaluate all inputs at x
                vals = []
                for inp in inputs:
                    if isinstance(inp, Trigtech):
                        vals.append(inp(x))
                    else:
                        vals.append(inp)
                # Apply the ufunc
                return ufunc(*vals, **kwargs)

            # Following MATLAB's compose(): pref.min_samples = max(pref.min_samples, length(f))
            # Ensure result has at least as many points as inputs
            minpow2 = int(np.ceil(np.log2(max_size))) if max_size > 0 else None
            return self.__class__.initfun_adaptive(newfun, interval=trigtech_obj.interval, minpow2=minpow2)
        else:
            return NotImplemented

    # -------
    #  roots
    # -------
    def roots(self, sort=None):
        """Compute the roots of the Trigtech on [0, 2π].

        For periodic functions, finding roots is more complex than for Chebyshev.
        We use a simple approach: evaluate on a fine grid and look for sign changes,
        then refine with Newton's method.

        Args:
            sort (bool, optional): Whether to sort the roots. Defaults to prefs.sortroots.

        Returns:
            np.ndarray: Array of roots in [0, 2π].
        """
        sort = sort if sort is not None else prefs.sortroots

        if self.isempty or self.size == 0:
            return np.array([])

        # Evaluate on a fine grid
        n_grid = max(1000, 10 * self.size)
        x_grid = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
        vals = self(x_grid)

        # Handle complex values - take real part if imaginary is negligible
        if np.iscomplexobj(vals):
            if np.max(np.abs(vals.imag)) < 1e-13 * (np.max(np.abs(vals.real)) + 1e-14):
                vals = vals.real
            else:
                # For complex functions, return empty (root finding is more complex)
                return np.array([])

        # Find sign changes
        signs = np.sign(vals)
        signs[signs == 0] = 1  # Treat zeros as positive
        sign_changes = np.where(np.diff(signs) != 0)[0]

        roots = []
        for idx in sign_changes:
            # Refine with bisection
            a, b = x_grid[idx], x_grid[idx + 1]
            fa = vals[idx]

            # Bisection
            for _ in range(50):  # Max iterations
                mid = (a + b) / 2
                fmid = self(np.array([mid]))[0]
                if np.iscomplexobj(fmid):
                    fmid = fmid.real
                if np.abs(fmid) < 1e-14:
                    break
                if np.sign(fmid) == np.sign(fa):
                    a, fa = mid, fmid
                else:
                    b = mid

            roots.append((a + b) / 2)

        roots = np.array(roots)

        # Remove duplicates (roots near 0 and 2π might be the same)
        if len(roots) > 0:
            # Wrap to [0, 2π)
            roots = roots % (2 * np.pi)
            # Remove duplicates
            roots = np.unique(np.round(roots / 1e-10) * 1e-10)

        roots = roots if not sort else np.sort(roots)
        return roots

    # ----------
    #  calculus
    # ----------
    def mean(self):
        """Compute the mean (average) value of the function over [0, 2π].

        Returns:
            float or complex: The mean value.
        """
        return self.sum() / (2 * np.pi)

    def norm(self, p=2):
        """Compute the Lp norm of the Trigtech over [0, 2π].

        Args:
            p (int or float): The norm type. Supported values are 1, 2, or np.inf.

        Returns:
            float: The p-norm.
        """
        if p == 1:
            # L1 norm: integral of |f|
            abs_self = self.__class__.initfun_adaptive(lambda x: np.abs(self(x)), interval=self.interval)
            return abs_self.sum()
        elif p == 2:
            # L2 norm: sqrt(integral of f^2)
            result = (self * self).sum()
            if np.iscomplexobj(result):
                result = result.real
            return np.sqrt(result)
        elif p == np.inf:
            # L-infinity norm: max|f|
            n_eval = max(1000, 10 * self.size)
            x = np.linspace(0, 2 * np.pi, n_eval, endpoint=False)
            vals = self(x)
            return np.max(np.abs(vals))
        else:
            raise ValueError(f"Unsupported norm type: {p}")

    @self_empty(resultif=0.0)
    def sum(self):
        """Definite integral of a Trigtech over the interval [0, 2π].

        For a periodic function represented by Fourier series:
            f(x) = Σ c_k exp(ikx)

        The integral over one period [0, 2π] is:
            ∫₀^{2π} f(x) dx = 2π * c_0

        where c_0 is the zeroth Fourier coefficient (DC component).
        """
        if self.size == 0:
            return 0.0

        # The integral is 2π times the DC component (k=0)
        # In FFT ordering, c_0 is at index 0
        out = 2 * np.pi * self.coeffs[0]

        # Return real part if negligible imaginary part
        if np.abs(out.imag) < 1e-13 * (np.abs(out.real) + 1e-14):
            out = out.real

        return out

    @self_empty()
    def cumsum(self):
        """Return a Trigtech object representing the indefinite integral.

        Computes the indefinite integral of a Trigtech on the interval [0, 2π].
        The constant term is chosen such that F(0) = 0.

        For Fourier series: if f(x) = Σ c_k exp(ikx), then
            F(x) = Σ (c_k / (ik)) exp(ikx)  for k ≠ 0

        The k=0 term requires special handling (constant of integration).
        """
        n = self.size
        if n == 0:
            return self.copy()

        coeffs = self.coeffs.copy()

        # Get frequency indices
        k = np.fft.fftfreq(n, d=1.0 / n)

        # Compute integrated coefficients: c_k / (ik)
        int_coeffs = np.zeros(n, dtype=complex)

        for i in range(n):
            if np.abs(k[i]) < 1e-14:  # k = 0 term
                # For k=0, integral of constant is linear term
                # We set this to zero initially (will adjust below)
                int_coeffs[i] = 0.0
            else:
                int_coeffs[i] = coeffs[i] / (1j * k[i])

        # Adjust constant to ensure F(0) = 0
        # At x=0, all exp(ikx) = 1, so F(0) = sum of all coefficients
        f0 = np.sum(int_coeffs)
        int_coeffs[0] -= f0  # Subtract constant from DC component

        return self.__class__(int_coeffs, interval=self.interval)

    @self_empty()
    def diff(self, n=1):
        """Return a Trigtech object representing the nth derivative.

        Computes the nth derivative of a Trigtech on its interval [a, b].

        For Fourier series on [a,b]: if f(x) = Σ c_k exp(2πikx/(b-a)), then
            f^(n)(x) = Σ (2πik/(b-a))^n c_k exp(2πikx/(b-a))

        Args:
            n (int): Order of differentiation. Defaults to 1.

        Returns:
            Trigtech: The nth derivative.
        """
        if n == 0:
            return self.copy()

        if self.isconst and n > 0:
            return self.__class__(np.array([0.0 + 0j], dtype=complex), interval=self.interval)

        m = self.size
        if m == 0:
            return self.copy()

        coeffs = self.coeffs.copy()

        # Get frequency indices (integer mode numbers)
        # fftfreq with d=1/m gives [0, 1, 2, ..., m//2, -(m//2-1), ..., -1]
        k = np.fft.fftfreq(m, d=1.0 / m)

        # Scale frequencies for the interval [a, b]
        # For a function on [a,b], Fourier basis is exp(2πikx/(b-a))
        # Derivative: d/dx[exp(2πikx/(b-a))] = (2πik/(b-a)) * exp(2πikx/(b-a))
        a, b = self.interval
        L = b - a
        omega = 2.0 * np.pi / L  # Angular frequency scaling

        # Multiply by (i*omega*k)^n for n-th derivative
        diff_coeffs = coeffs * (1j * omega * k) ** n

        return self.__class__(diff_coeffs, interval=self.interval)

    @staticmethod
    def _chop_coeffs(coeffs, tol):
        """Remove trailing insignificant Fourier coefficients.

        For Fourier series in FFT ordering [0, 1, ..., n/2, -n/2+1, ..., -1],
        we find the highest significant frequency and keep only what's needed.
        IMPORTANT: Must preserve both positive and negative frequencies for real functions.

        Args:
            coeffs (np.ndarray): Fourier coefficients in FFT ordering
            tol (float): Tolerance for chopping

        Returns:
            np.ndarray: Chopped coefficients
        """
        if len(coeffs) == 0:
            return coeffs

        n = len(coeffs)
        abs_coeffs = np.abs(coeffs)

        # Check which coefficients are significant
        significant = abs_coeffs > tol

        if not np.any(significant):
            return np.array([coeffs[0]])  # Return just DC component

        # Find the maximum significant frequency
        # For real functions, we need both +k and -k frequencies
        # FFT frequency ordering for n points:
        # [0, 1, 2, ..., n//2, -(n//2-1) or -n//2, ..., -2, -1]
        freq = np.fft.fftfreq(n, d=1.0 / n)
        sig_indices = np.where(significant)[0]
        sig_freqs = np.abs(freq[sig_indices])

        if len(sig_freqs) == 0:
            return np.array([coeffs[0]])

        max_sig_freq = np.max(sig_freqs)

        # We need enough points to represent max_sig_freq in both positive and negative
        # For frequency k, we need n such that k < n/2 (Nyquist)
        # So n > 2*k, and we use n >= 2*k + 1 for safety
        n_min = int(2 * max_sig_freq + 2)  # Add 2 for safety margin

        # Round up to a power of 2 (efficient for FFT)
        if n_min <= 4:
            n_keep = 4  # Minimum practical size
        else:
            n_keep = 2 ** int(np.ceil(np.log2(n_min)))

        n_keep = min(n_keep, n)

        # For the chopped array, we must ensure we keep corresponding negative frequencies
        # If we're keeping n_keep points, positive freqs go up to n_keep//2
        # and negative freqs from n_keep//2+1 to n_keep-1
        # But we're taking from a larger array, so we need to be careful

        # Actually, simpler approach: if we decided n_keep, just return first n_keep coeffs
        # This works IF the original array had all the structure. The issue is when
        # we're truncating an already small array. Let's just not truncate too aggressively.

        # For safety, if n_keep would cut off important negative frequencies, keep more
        # Check: for n_keep points, frequencies range from 0 to n_keep//2 and -(n_keep//2-1) to -1
        # Maximum representable frequency magnitude is n_keep//2
        if max_sig_freq > n_keep // 2:
            # Need more points
            n_keep = int(2 * max_sig_freq + 2)
            n_keep = min(2 ** int(np.ceil(np.log2(n_keep))), n)

        return coeffs[:n_keep]

    @staticmethod
    def _trigpts(n):
        """Return n equally-spaced points on [0, 2π).

        These are the standard Fourier points:
            x_j = 2πj/n  for j = 0, 1, ..., n-1

        Args:
            n (int): Number of points.

        Returns:
            np.ndarray: Array of n equally-spaced points.
        """
        if n == 0:
            return np.array([])
        return 2 * np.pi * np.arange(n) / n

    @staticmethod
    def _trigwts(n):
        """Return trapezoidal quadrature weights for integration on [0, 2π).

        For equally-spaced points, all weights are equal: 2π/n

        Args:
            n (int): Number of points.

        Returns:
            np.ndarray: Array of quadrature weights.
        """
        if n == 0:
            return np.array([])
        return (2 * np.pi / n) * np.ones(n)

    @staticmethod
    def _vals2coeffs(vals):
        """Map function values at Fourier points to Fourier coefficients.

        Converts values at equally-spaced points to Fourier coefficients using FFT.

        Args:
            vals (np.ndarray): Function values at Fourier points.

        Returns:
            np.ndarray: Fourier coefficients.
        """
        if len(vals) == 0:
            return np.array([], dtype=complex)

        # Use FFT to get Fourier coefficients
        # numpy's FFT: F[k] = Σ f[j] exp(-2πijk/n)
        # We want: c[k] = (1/n) Σ f[j] exp(-2πijk/n) = F[k]/n
        coeffs = np.fft.fft(vals) / len(vals)

        return coeffs

    @staticmethod
    def _coeffs2vals(coeffs):
        """Map Fourier coefficients to function values at Fourier points.

        Converts Fourier coefficients to values at equally-spaced points using iFFT.

        Args:
            coeffs (np.ndarray): Fourier coefficients.

        Returns:
            np.ndarray: Function values at Fourier points.
        """
        if len(coeffs) == 0:
            return np.array([], dtype=complex)

        # Use inverse FFT to get function values
        # f[j] = Σ c[k] exp(2πijk/n) = ifft(c * n)
        vals = np.fft.ifft(coeffs * len(coeffs))

        return vals

    # ----------
    #  plotting
    # ----------
    def plot(self, ax=None, **kwargs):
        """Plot the Trigtech on the interval [0, 2π].

        Args:
            ax (matplotlib.axes.Axes, optional): The axes on which to plot. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the plot function.

        Returns:
            matplotlib.lines.Line2D: The line object created by the plot.
        """
        return plotfun(self, (0, 2 * np.pi), ax=ax, **kwargs)

    def plotcoeffs(self, ax=None, **kwargs):
        """Plot the absolute values of the Fourier coefficients.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes on which to plot. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the plot function.

        Returns:
            matplotlib.lines.Line2D: The line object created by the plot.
        """
        ax = ax or plt.gca()
        return plotfuncoeffs(abs(self.coeffs), ax=ax, **kwargs)
