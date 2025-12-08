"""Numerical algorithms for Chebyshev approximation and manipulation.

This module provides core numerical algorithms used throughout the ChebPy package,
including rootfinding, barycentric interpolation, Chebyshev coefficient manipulation,
and adaptive approximation techniques.

The algorithms implemented here are based on established numerical methods for
working with Chebyshev polynomials and approximations, many of which are described
in Trefethen's "Approximation Theory and Approximation Practice".
"""

import warnings

import numpy as np
from numpy.fft import fft, ifft

from .decorators import preandpostprocess
from .settings import _preferences as prefs
from .utilities import Interval, infnorm

# supress numpy division and multiply warnings
np.seterr(divide="ignore", invalid="ignore")

# constants
SPLITPOINT = -0.004849834917525


# local helpers
def find(x: np.ndarray) -> np.ndarray:
    """Find the indices of non-zero elements in an array.

    A simple wrapper around numpy.where that returns only the indices.

    Args:
        x (array-like): Input array.

    Returns:
        numpy.ndarray: Indices of non-zero elements in the input array.
    """
    return np.where(x)[0]


def rootsunit(ak: np.ndarray, htol: float = None) -> np.ndarray:
    """Compute the roots of a function on [-1,1] using Chebyshev coefficients.

    This function finds the real roots of a function on the interval [-1,1]
    using the coefficients in its Chebyshev series representation. For large
    degree polynomials, it uses a recursive subdivision approach.

    Args:
        ak (numpy.ndarray): Coefficients of the Chebyshev series.
        htol (float, optional): Tolerance for determining which roots to keep.
            Defaults to 100 * machine epsilon.

    Returns:
        numpy.ndarray: Array of roots in the interval [-1,1], sorted in
            ascending order.

    References:
        I. J. Good, "The colleague matrix, a Chebyshev analogue of the
        companion matrix", Quarterly Journal of Mathematics 12 (1961).

        J. A. Boyd, "Computing zeros on a real interval through
        Chebyshev expansion and polynomial rootfinding", SIAM Journal on
        Numerical Analysis 40 (2002).

        L. N. Trefethen, Approximation Theory and Approximation
        Practice, SIAM, 2013, chapter 18.
    """
    htol = htol if htol is not None else 1e2 * prefs.eps
    n = standard_chop(ak, tol=htol)
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
        lrts = rootsunit(lcfs, 2 * htol)
        rrts = rootsunit(rcfs, 2 * htol)
        return np.append(lmap(lrts), rmap(rrts))

    # trivial base case
    if n <= 1:
        return np.array([])

    # nontrivial base case: either compute directly or solve
    # a Colleague Matrix eigenvalue problem
    if n == 2:
        rts = np.array([-ak[0] / ak[1]])
    elif n <= 50:
        v = 0.5 * np.ones(n - 2)
        colleague_matrix = np.diag(v, -1) + np.diag(v, 1)
        colleague_matrix[0, 1] = 1
        coeffs_matrix = np.zeros(colleague_matrix.shape, dtype=ak.dtype)
        coeffs_matrix[-1, :] = ak[:-1]
        eigenvalue_matrix = colleague_matrix - 0.5 * 1.0 / ak[-1] * coeffs_matrix
        rts = np.linalg.eigvals(eigenvalue_matrix)

    # discard values with large imaginary part and treat the remaining
    # ones as real; then sort and retain only the roots inside [-1,1]
    mask = abs(np.imag(rts)) < htol
    rts = np.real(rts[mask])
    rts = rts[abs(rts) <= 1.0 + htol]
    rts = np.sort(rts)
    if rts.size >= 2:
        rts[0] = max([rts[0], -1])
        rts[-1] = min([rts[-1], 1])
    return rts


@preandpostprocess
def bary(xx: np.ndarray, fk: np.ndarray, xk: np.ndarray, vk: np.ndarray) -> np.ndarray:
    """Evaluate a function using the barycentric interpolation formula.

    This function implements the barycentric interpolation formula for evaluating
    a function at arbitrary points given its values at a set of nodes. It uses
    an efficient algorithm that switches between two implementations based on
    the number of evaluation points.

    Args:
        xx (numpy.ndarray): Array of evaluation points.
        fk (numpy.ndarray): Array of function values at the interpolation nodes xk.
        xk (numpy.ndarray): Array of interpolation nodes.
        vk (numpy.ndarray): Barycentric weights corresponding to the interpolation nodes xk.

    Returns:
        numpy.ndarray: Function values at the evaluation points xx.

    References:
        J.P. Berrut, L.N. Trefethen, Barycentric Lagrange Interpolation, SIAM
        Review (2004)
    """
    # either iterate over the evaluation points, or ...
    if xx.size < 4 * xk.size:
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
        idx = find(xx[k] == xk)
        if idx.size > 0:
            out[k] = fk[idx[0]]

    return out


@preandpostprocess
def clenshaw(xx: np.ndarray, ak: np.ndarray) -> np.ndarray:
    """Evaluate a Chebyshev series using Clenshaw's algorithm.

    This function implements Clenshaw's algorithm for the evaluation of a
    first-kind Chebyshev series expansion at an array of points.

    Args:
        xx (numpy.ndarray): Array of points at which to evaluate the series.
        ak (numpy.ndarray): Coefficients of the Chebyshev series.

    Returns:
        numpy.ndarray: Values of the Chebyshev series at the points xx.

    References:
        C. W. Clenshaw, "A note on the summation of Chebyshev series",
        Mathematics of Computation, Vol. 9, No. 51, 1955, pp. 118-120.
    """
    bk1 = 0 * xx
    bk2 = 0 * xx
    xx = 2 * xx
    idx = range(ak.size)
    for k in idx[ak.size : 1 : -2]:
        bk2 = ak[k] + xx * bk1 - bk2
        bk1 = ak[k - 1] + xx * bk2 - bk1
    if np.mod(ak.size - 1, 2) == 1:
        bk1, bk2 = ak[1] + xx * bk1 - bk2, bk1
    out = ak[0] + 0.5 * xx * bk1 - bk2
    return out


def standard_chop(coeffs: np.ndarray, tol: float = None) -> int:
    """Determine where to truncate a Chebyshev series based on coefficient decay.

    This function determines an appropriate cutoff point for a Chebyshev series
    by analyzing the decay of its coefficients. It implements the algorithm
    described by Aurentz and Trefethen.

    Args:
        coeffs (numpy.ndarray): Coefficients of the Chebyshev series.
        tol (float, optional): Tolerance for determining the cutoff point.
            Defaults to machine epsilon from preferences.

    Returns:
        int: Index at which to truncate the series.

    References:
        J. Aurentz and L.N. Trefethen, "Chopping a Chebyshev series" (2015)
        (http://arxiv.org/pdf/1512.01803v1.pdf)
    """
    # check magnitude of tol:
    tol = tol if tol is not None else prefs.eps
    if tol >= 1:  # pragma: no cover
        cutoff = 1
        return cutoff

    # ensure length at least 17:
    n = coeffs.size
    cutoff = n
    if n < 17:
        return cutoff

    # Step 1: Convert coeffs input to a new monotonically nonincreasing
    # vector (envelope) normalized to begin with the value 1.
    b = np.flipud(np.abs(coeffs))
    m = np.flipud(np.maximum.accumulate(b))
    if m[0] == 0.0:
        # TODO: check this
        cutoff = 1  # cutoff = 0
        return cutoff
    envelope = m / m[0]

    # Step 2: Scan envelope for a value plateauPoint, the first point, if any,
    # that is followed by a plateau
    for j in np.arange(1, n):
        j2 = round(1.25 * j + 5)
        if j2 > n - 1:
            # there is no plateau: exit
            return cutoff
        e1 = envelope[j]
        e2 = envelope[int(j2)]
        r = 3 * (1 - np.log(e1) / np.log(tol))
        plateau = (e1 == 0.0) | (e2 / e1 > r)
        if plateau:
            # a plateau has been found: go to Step 3
            plateau_point = j
            break

    # Step 3: Fix cutoff at a point where envelope, plus a linear function
    # included to bias the result towards the left end, is minimal.
    if envelope[int(plateau_point)] == 0.0:
        cutoff = plateau_point
    else:
        j3 = sum(envelope >= tol ** (7.0 / 6.0))
        if j3 < j2:
            j2 = j3 + 1
            envelope[j2] = tol ** (7.0 / 6.0)
        cc = np.log10(envelope[: int(j2)])
        cc = cc + np.linspace(0, (-1.0 / 3.0) * np.log10(tol), int(j2))
        d = np.argmin(cc)
        # TODO: check this
        cutoff = d  # + 2
    return min((cutoff, n - 1))


def adaptive(cls: type, fun: callable, hscale: float = 1, maxpow2: int = None, min_samples: int = None) -> np.ndarray:
    """Adaptively determine the number of points needed to represent a function.

    This function implements an adaptive algorithm to determine the appropriate
    number of points needed to represent a function to a specified tolerance.
    It cycles over powers of two, evaluating the function at Chebyshev points
    and checking if the resulting coefficients can be truncated.

    Args:
        cls: The class that provides the _chebpts and _vals2coeffs methods.
        fun (callable): The function to be approximated.
        hscale (float, optional): Scale factor for the tolerance. Defaults to 1.
        maxpow2 (int, optional): Maximum power of 2 to try. If None, uses the
            value from preferences.
        min_samples (int, optional): Minimum number of sample points to use. This
            ensures the result has at least as many points as the input when
            composing operations.

    Returns:
        numpy.ndarray: Coefficients of the Chebyshev series representing the function.

    Warns:
        UserWarning: If the constructor does not converge within the maximum
            number of iterations.
    """
    minpow2 = 4  # 17 points
    if min_samples is not None and min_samples > 17:
        # Find the power of 2 that gives at least min_samples points
        minpow2 = max(minpow2, int(np.ceil(np.log2(min_samples - 1))))

    maxpow2 = maxpow2 if maxpow2 is not None else prefs.maxpow2
    for k in range(minpow2, max(minpow2, maxpow2) + 1):
        n = 2**k + 1
        points = cls._chebpts(n)
        values = fun(points)
        # Handle scalar returns from constant functions like lambda x: 1.0
        values = np.atleast_1d(values)
        if values.size == 1:
            values = np.broadcast_to(values, points.shape)
        coeffs = cls._vals2coeffs(values)
        eps = prefs.eps
        tol = eps * max(hscale, 1)  # scale (decrease) tolerance by hscale
        chplen = standard_chop(coeffs, tol=tol)
        if chplen < coeffs.size:
            coeffs = coeffs[:chplen]
            break
        if k == maxpow2:
            warnings.warn(
                f"The {cls.__name__} constructor did not converge: using {n} points. "
                f"Function may be too oscillatory or have discontinuities. "
                f"Tolerance: {tol:.2e}. Consider increasing prefs.maxpow2 if needed."
            )
            break
    return coeffs


def coeffmult(fc: np.ndarray, gc: np.ndarray) -> np.ndarray:
    """Multiply two Chebyshev series in coefficient space.

    This function performs multiplication of two Chebyshev series represented by
    their coefficients. It uses FFT-based convolution for efficiency.

    Args:
        fc (numpy.ndarray): Coefficients of the first Chebyshev series.
        gc (numpy.ndarray): Coefficients of the second Chebyshev series.

    Returns:
        numpy.ndarray: Coefficients of the product series.

    Note:
        The input series must have the same length.
    """
    fc_extended = np.append(2.0 * fc[:1], (fc[1:], fc[:0:-1]))
    gc_extended = np.append(2.0 * gc[:1], (gc[1:], gc[:0:-1]))
    ak = ifft(fft(fc_extended) * fft(gc_extended))
    ak = np.append(ak[:1], ak[1:] + ak[:0:-1]) * 0.25
    ak = ak[: fc.size]
    inputcfs = np.append(fc, gc)
    out = np.real(ak) if np.isreal(inputcfs).all() else ak
    return out


def barywts2(n: int) -> np.ndarray:
    """Compute barycentric weights for Chebyshev points of the second kind.

    This function calculates the barycentric weights used in the barycentric
    interpolation formula for Chebyshev points of the second kind.

    Args:
        n (int): Number of points (n+1 weights will be computed).

    Returns:
        numpy.ndarray: Array of barycentric weights.

    Note:
        For Chebyshev points of the second kind, the weights have a simple
        explicit formula with alternating signs.
    """
    if n == 0:
        wts = np.array([])
    elif n == 1:
        wts = np.array([1])
    else:
        wts = np.append(np.ones(n - 1), 0.5)
        wts[n - 2 :: -2] = -1
        wts[0] = 0.5 * wts[0]
    return wts


def clencurt_weights(n: int) -> np.ndarray:
    """Compute Clenshaw-Curtis quadrature weights for n+1 Chebyshev points.

    Returns the weights for integrating on [-1, 1] with Chebyshev points
    x_j = cos(j*pi/n), j=0..n. The rule is exact for polynomials up to degree n.

    This implementation uses an FFT-based O(n log n) algorithm via DCT-I,
    matching MATLAB Chebfun's Waldvogel method in spirit but adapted for
    the n+1 points convention.

    Args:
        n (int): Discretization size (returns n+1 weights for n+1 points).

    Returns:
        numpy.ndarray: Array of n+1 quadrature weights.

    References:
        - Waldvogel, "Fast Construction of the Fejér and Clenshaw-Curtis
          Quadrature Rules", BIT Numerical Mathematics 46 (2006), pp 195-202.
        - MATLAB Chebfun @chebtech2/quadwts.m

    Examples:
        >>> w = clencurt_weights(1)
        >>> w
        array([1., 1.])
        >>> w = clencurt_weights(2)
        >>> np.allclose(w, [1/3, 4/3, 1/3])
        True
        >>> np.allclose(np.sum(clencurt_weights(4)), 2.0)  # Should equal 2
        True
    """
    if n == 0:
        return np.array([2.0])
    elif n == 1:
        return np.array([1.0, 1.0])

    n_pts = n + 1  # Number of points

    # Moments of Chebyshev polynomials: int_{-1}^1 T_m(x) dx
    # = 2/(1-m^2) for even m, 0 for odd m
    c = np.zeros(n_pts)
    c[0] = 2.0
    for k in range(2, n_pts, 2):
        c[k] = 2.0 / (1.0 - k**2)

    # DCT-I via FFT: mirror as [c[0], c[1], ..., c[n_pts-1], c[n_pts-2], ..., c[1]]
    c_mirror = np.concatenate([c, c[-2:0:-1]])
    w = np.real(np.fft.ifft(c_mirror))[:n_pts]

    # Adjust boundary weights (DCT-I endpoints are weighted by 1/2)
    w[0] /= 2.0
    w[-1] /= 2.0

    # Scale by 2
    w *= 2.0

    return w


def chebpts2(n: int) -> np.ndarray:
    """Compute Chebyshev points of the second kind.

    This function calculates the n Chebyshev points of the second kind in the
    interval [-1, 1], which are the extrema of the Chebyshev polynomial T_{n-1}
    together with the endpoints ±1.

    Args:
        n (int): Number of points to compute.

    Returns:
        numpy.ndarray: Array of n Chebyshev points of the second kind.

    Note:
        The points are ordered from left to right on the interval [-1, 1].
    """
    if n == 1:
        pts = np.array([0.0])
    else:
        nn = np.arange(n)
        pts = np.cos(nn[::-1] * np.pi / (n - 1))
    return pts


def vals2coeffs2(vals: np.ndarray) -> np.ndarray:
    """Convert function values to Chebyshev coefficients.

    This function maps function values at Chebyshev points of the second kind
    to coefficients of the corresponding first-kind Chebyshev polynomial expansion.
    It uses an FFT-based algorithm for efficiency.

    Args:
        vals (numpy.ndarray): Function values at Chebyshev points of the second kind.

    Returns:
        numpy.ndarray: Coefficients of the first-kind Chebyshev polynomial expansion.

    Note:
        This transformation is the discrete cosine transform of type I (DCT-I),
        which is implemented here using FFT for efficiency.
    """
    n = vals.size
    if n <= 1:
        coeffs = vals
        return coeffs
    tmp = np.append(vals[::-1], vals[1:-1])
    if np.isreal(vals).all():
        coeffs = ifft(tmp)
        coeffs = np.real(coeffs)
    elif np.isreal(1j * vals).all():  # pragma: no cover
        coeffs = ifft(np.imag(tmp))
        coeffs = 1j * np.real(coeffs)
    else:
        coeffs = ifft(tmp)
    coeffs = coeffs[:n]
    coeffs[1 : n - 1] = 2 * coeffs[1 : n - 1]
    return coeffs


def coeffs2vals2(coeffs: np.ndarray) -> np.ndarray:
    """Convert Chebyshev coefficients to function values.

    This function maps coefficients of a first-kind Chebyshev polynomial expansion
    to function values at Chebyshev points of the second kind. It uses an FFT-based
    algorithm for efficiency.

    Args:
        coeffs (numpy.ndarray): Coefficients of the first-kind Chebyshev polynomial expansion.

    Returns:
        numpy.ndarray: Function values at Chebyshev points of the second kind.

    Note:
        This transformation is the inverse discrete cosine transform of type I (IDCT-I),
        which is implemented here using FFT for efficiency. It is the inverse of vals2coeffs2.
    """
    n = coeffs.size
    if n <= 1:
        vals = coeffs
        return vals
    coeffs = coeffs.copy()
    coeffs[1 : n - 1] = 0.5 * coeffs[1 : n - 1]
    tmp = np.append(coeffs, coeffs[n - 2 : 0 : -1])
    if np.isreal(coeffs).all():
        vals = fft(tmp)
        vals = np.real(vals)
    elif np.isreal(1j * coeffs).all():  # pragma: no cover
        vals = fft(np.imag(tmp))
        vals = 1j * np.real(vals)
    else:
        vals = fft(tmp)
    vals = vals[n - 1 :: -1]
    return vals


def newtonroots(fun: callable, rts: np.ndarray, tol: float = None, maxiter: int = None) -> np.ndarray:
    """Refine root approximations using Newton's method.

    This function applies Newton's method to refine the approximations of roots
    for a callable and differentiable function. It is typically used to polish
    already computed roots to higher accuracy.

    Args:
        fun (callable): A callable and differentiable function.
        rts (numpy.ndarray): Initial approximations of the roots.
        tol (float, optional): Tolerance for convergence. Defaults to 2 * machine epsilon.
        maxiter (int, optional): Maximum number of iterations. Defaults to value from preferences.

    Returns:
        numpy.ndarray: Refined approximations of the roots.

    Note:
        The function must support differentiation via a .diff() method that returns
        the derivative function.
    """
    tol = tol if tol is not None else 2 * prefs.eps
    maxiter = maxiter if maxiter is not None else prefs.maxiter
    if rts.size > 0:
        dfun = fun.diff()
        prv = np.inf * rts
        count = 0
        while (infnorm(rts - prv) > tol) & (count <= maxiter):
            count += 1
            prv = rts
            rts = rts - fun(rts) / dfun(rts)
    return rts
