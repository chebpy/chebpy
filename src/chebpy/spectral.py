"""Spectral discretization matrices for differential operators.

This module provides functions to construct spectral discretization matrices
for Chebyshev and Fourier collocation methods, including differentiation matrices
and multiplication operators.

These matrices are used by the Chebop system to convert differential operators
into linear algebra problems that can be solved numerically.
"""

import warnings

import numpy as np
from scipy import sparse
from scipy.linalg import toeplitz

from .algorithms import barywts2, chebpts2
from .sparse_utils import sparse_to_dense
from .utilities import Interval, ensure_interval


def cheb_points_scaled(n, interval: Interval):
    """Compute Chebyshev points of the second kind on an arbitrary interval.

    pts[0] = a (left endpoint), pts[-1] = b (right endpoint)

    Args:
        n (int): Number of interior degrees of freedom (returns n+1 points).
        interval (Interval): Interval [a, b] defining the domain.

    Returns:
        numpy.ndarray: Array of n+1 Chebyshev points on [a, b] in ascending order.

    Examples:
        >>> pts = cheb_points_scaled(4, Interval(0, 1))  # n=4 gives 5 points
        >>> len(pts)
        5
        >>> pts[0]  # Left endpoint  # doctest: +SKIP
        0.0
        >>> pts[-1]  # Right endpoint  # doctest: +SKIP
        1.0
    """
    # Get standard Chebyshev points on [-1, 1]
    y = chebpts2(n + 1)

    # Map from [-1, 1] to [a, b]
    return interval(y)


def diff_matrix_rectangular(n, m, interval, order=1):
    """Construct rectangular Chebyshev differentiation matrix.

    Maps from n+1 Chebyshev coefficients to m+1 collocation points (m >= n).
    This creates an overdetermined system that improves eigenvalue accuracy.

    The rectangular matrix is constructed using barycentric interpolation:
    values at n+1 Chebyshev points are first differentiated, then interpolated
    to m+1 evaluation points.

    Args:
        n (int): Number of coefficient DOFs (yields n+1 Chebyshev coefficients).
        m (int): Number of collocation points (yields m+1 points), must satisfy m >= n.
        interval (array-like): Two-element array [a, b] defining the physical interval.
        order (int, optional): Order of differentiation (1, 2, 3, ...). Defaults to 1.

    Returns:
        scipy.sparse matrix: Rectangular differentiation matrix of size (m+1) x (n+1).

    Raises:
        ValueError: If m < n (underdetermined system).

    Notes:
        - For square case (m = n), this reduces to standard diff_matrix().
        - Typical choices: m = 2*n or m = n + 50 for overdetermination.
        - See Driscoll & Hale (2016), "Rectangular spectral collocation".

    Examples:
        >>> # Overdetermined system: 17 coefficients, 33 collocation points
        >>> D = diff_matrix_rectangular(16, 32, [-1, 1], order=1)
        >>> D.shape
        (33, 17)

        >>> # Square case (equivalent to diff_matrix)
        >>> D_square = diff_matrix_rectangular(16, 16, [-1, 1], order=1)
        >>> D_square.shape
        (17, 17)
    """
    if m < n:
        raise ValueError(f"Rectangular differentiation requires m >= n, got m={m}, n={n}")

    # For square case, use standard differentiation matrix
    if m == n:
        return diff_matrix(n, interval, order=order)

    # Get standard Chebyshev differentiation matrix on n+1 points
    # This maps values at n+1 Chebyshev points to derivatives at same points
    d_standard = diff_matrix(n, interval, order=order)

    # Get m+1 evaluation points (also Chebyshev points, but different resolution)
    interval_obj = Interval(*interval) if not isinstance(interval, Interval) else interval
    x_eval = cheb_points_scaled(m, interval_obj)

    # Build barycentric interpolation matrix: maps from n+1 points to m+1 points
    # eval_matrix is (m+1) x (n+1)
    eval_matrix = barycentric_matrix(x_eval, n, interval_obj)

    # Compose: eval_matrix @ d maps from n+1 coefficient values to m+1 derivative values
    # Result is (m+1) x (n+1) rectangular differentiation matrix
    d_rect = eval_matrix @ d_standard

    return sparse.csr_matrix(d_rect)


def diff_matrix_driscoll_hale(n, interval, order=1):
    """Construct Driscoll-Hale rectangular differentiation matrix.

    This implements the rectangular spectral collocation approach from
    Driscoll & Hale (2016). The key insight is that differentiating a
    polynomial of degree n gives a polynomial of degree n-1. So the k-th
    derivative of a degree-n polynomial can be represented by n-k+1 values.

    The matrix maps from n+1 Chebyshev coefficients to (n-order+1) collocation
    points, creating a rectangular matrix that preserves polynomial accuracy
    without row deletion.

    For a k-th order ODE with k BCs, this gives:
        - Operator matrix: (n-k+1) × (n+1)
        - Add k BC rows: (n+1) × (n+1) square system

    Args:
        n: Number of coefficient DOFs (yields n+1 Chebyshev coefficients)
        interval: Two-element array [a, b] defining the physical interval
        order: Order of differentiation (1, 2, 3, ...). Defaults to 1.

    Returns:
        scipy.sparse matrix: Rectangular differentiation matrix of size
        (n-order+1) × (n+1).

    References:
        Driscoll & Hale (2016), "Rectangular spectral collocation",
        IMA Journal of Numerical Analysis 36(1):108-132.

    Examples:
        >>> D = diff_matrix_driscoll_hale(10, [-1, 1], order=2)
        >>> D.shape  # 2nd derivative: maps 11 coeffs to 9 values
        (9, 11)

        >>> # For u'' = f with 2 BCs:
        >>> # Operator: (n-1) × (n+1), add 2 BC rows → (n+1) × (n+1)
    """
    if n < order:
        raise ValueError(f"Polynomial degree n={n} must be >= derivative order={order}")

    # Number of output points (one fewer for each derivative order)
    m = n - order

    if m < 0:  # pragma: no cover  # Unreachable - caught by n < order check above
        raise ValueError(f"Cannot take order-{order} derivative of degree-{n} polynomial")

    # For the Driscoll-Hale approach, we need to:
    # 1. Differentiate using standard n+1 point differentiation matrix
    # 2. Resample from n+1 points to m+1 points (polynomial subspace)

    # Get standard differentiation matrix (n+1) × (n+1)
    D_full = diff_matrix(n, interval, order=order)

    # Get the sampling points for input (n+1 points) and output (m+1 points)
    interval_obj = Interval(*interval) if not isinstance(interval, Interval) else interval
    cheb_points_scaled(n, interval_obj)  # n+1 points
    x_output = cheb_points_scaled(m, interval_obj)  # m+1 points

    # Build resampling matrix: interpolate from n+1 to m+1 points
    # This is (m+1) × (n+1)
    P = barycentric_matrix(x_output, n, interval_obj)

    # The Driscoll-Hale matrix is P @ D_full
    # This maps: n+1 values → (D_full) → n+1 derivative values → (P) → m+1 resampled values
    D_rect = P @ sparse_to_dense(D_full)

    return sparse.csr_matrix(D_rect)


def diff_matrix(n, interval, order=1):
    """Construct Chebyshev differentiation matrix of given order.

    This function constructs the spectral differentiation matrix that
    approximates the derivative of a function represented by values at
    Chebyshev points. The matrix is scaled appropriately for the given interval.

    For high-order derivatives (order >= 2), uses the barycentric differentiation
    formula which is much more stable than computing matrix powers D^order.

    Args:
        n (int): Number of interior degrees of freedom (matrix will be (n+1) x (n+1)).
        interval (array-like): Two-element array [a, b] defining the interval.
        order (int, optional): Order of differentiation (1, 2, 3, ...). Defaults to 1.

    Returns:
        scipy.sparse matrix: Differentiation matrix of size (n+1) x (n+1).

    Notes:
        - For order >= 2, uses barycentric formula for numerical stability
        - Avoids ill-conditioning from matrix powers D^order

    References:
        - Schneider & Werner (1986), "Some new aspects of rational interpolation"
        - Welfert (1997), "Generation of pseudospectral matrices I"
        - Baltensperger & Trummer (2003), "Spectral Differencing with a Twist"

    Examples:
        >>> D = diff_matrix(4, [-1, 1])  # n=4 gives 5x5 matrix
        >>> D.shape
        (5, 5)
    """
    if n < 1:
        return sparse.csr_matrix((1, 1))

    # Warn about high-order derivatives (numerically unstable)
    if order > 6:
        warnings.warn(
            f"Computing derivative of order {order} is numerically unstable. "
            "Consider reformulating the problem with lower-order derivatives.",
            UserWarning,
            stacklevel=2,
        )

    # Get Chebyshev points and weights on [-1, 1]
    x = chebpts2(n + 1)
    w = barywts2(n + 1)

    # For higher orders, use barycentric differentiation formula
    if order >= 2:
        # Compute angles for stability (Baltensperger & Trummer 2003)
        t = np.arccos(x)
        D = _barydiff_matrix(x, w, order, t)
    else:
        # For first order, use standard formula (faster)
        c = np.concatenate(([2.0], np.ones(n - 1), [2.0])) * ((-1.0) ** np.arange(n + 1))
        X = np.tile(x.reshape(-1, 1), (1, n + 1))
        dX = X - X.T

        # Avoid division by zero on diagonal
        D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))

        # Fix diagonal
        D = D - np.diag(np.sum(D, axis=1))

    # Scale for interval [a, b]
    a, b = interval
    scale_factor = 2.0 / (b - a)
    D = (scale_factor**order) * D

    # Convert to sparse for efficiency
    return sparse.csr_matrix(D)


def mult_matrix(chebfun, n, interval=None):
    """Construct multiplication matrix for a given chebfun.

    This function constructs a diagonal matrix whose diagonal entries are
    the values of the given chebfun at Chebyshev points. This represents
    the operation of multiplying by the function.

    Args:
        chebfun: Chebfun object representing the function to multiply by.
        n (int): Number of points (matrix will be (n+1) x (n+1)).
        interval: Optional interval [a, b] to evaluate on. If None, uses chebfun.support.
            Note: For multi-block problems, must pass the specific block interval.

    Returns:
        scipy.sparse matrix: Diagonal multiplication matrix of size (n+1) x (n+1).

    Examples:
        Multiplication matrices are used in spectral discretization to represent
        the action of multiplying by a function on the space of polynomials.
    """
    # Get Chebyshev points on the specified or chebfun's domain
    if interval is None:
        interval = ensure_interval(chebfun.support)

    x = cheb_points_scaled(n, interval)

    # Evaluate chebfun at these points
    values = chebfun(x)

    # Ensure values is 1D
    values = np.atleast_1d(values).ravel()

    # Create diagonal matrix
    return sparse.diags(values, 0, format="csr")


def identity_matrix(n):
    """Construct sparse identity matrix.

    Args:
        n (int): Size of matrix (will be (n+1) x (n+1)).

    Returns:
        scipy.sparse matrix: Identity matrix of size (n+1) x (n+1).

    Examples:
        >>> I = identity_matrix(4)
        >>> I.shape
        (5, 5)
        >>> I.toarray()[2, 2]  # doctest: +SKIP
        1.0
    """
    return sparse.eye(n + 1, format="csr")


def _barydiff_matrix(x, w, order, t=None):
    """Compute barycentric differentiation matrix for high-order derivatives.

    This function implements the stable barycentric differentiation formula
    from Schneider & Werner (1986), Welfert (1997), and the 'twist' from
    Baltensperger & Trummer (2003) for improved accuracy.

    Args:
        x: Chebyshev points (on [-1, 1])
        w: Barycentric weights for these points
        order: Order of differentiation (should be >= 2 for this function)
        t: Optional angles (arccos(x)) for improved accuracy

    Returns:
        numpy.ndarray: Differentiation matrix D of size (N, N) where N = len(x)

    References:
        - Schneider & Werner (1986), "Some new aspects of rational interpolation"
        - Welfert (1997), "Generation of pseudospectral matrices I"
        - Baltensperger & Trummer (2003), "Spectral Differencing with a Twist"
    """
    N = len(x)

    if N == 0:
        return np.array([])
    if N == 1:
        return np.array([[0.0]])

    if order == 0:
        return np.eye(N)

    # Construct Dx and Dw
    ii = np.arange(N)

    if t is not None:
        # Trig identity for improved accuracy (Baltensperger & Trummer 2003)
        t = np.flipud(t)

        # Compute pairwise differences using trig identity: 2*sin((t+tp)/2)*sin((t-tp)/2)
        t_half = t / 2
        t_sum = t_half[:, None] + t_half[None, :]  # t/2 + tp/2
        t_diff = t_half[:, None] - t_half[None, :]  # t/2 - tp/2
        Dx = 2 * np.sin(t_sum) * np.sin(t_diff)
    else:
        # Standard pairwise differences
        Dx = x[:, None] - x[None, :]

    # Flipping trick from Baltensperger & Trummer (2003)
    DxRot = np.rot90(Dx, 2)
    idxTo = np.rot90(~np.triu(np.ones((N, N), dtype=bool)))
    Dx[idxTo] = -DxRot[idxTo]

    # Avoid division by zero on diagonal
    Dx[ii, ii] = 1.0

    # Reciprocal
    Dxi = 1.0 / Dx

    # Pairwise weight divisions
    Dw = w[None, :] / w[:, None]
    Dw[ii, ii] = 0.0

    # k = 1 (first derivative)
    D = Dw * Dxi
    D[ii, ii] = 0.0
    D[ii, ii] = -np.sum(D, axis=1)  # Negative sum trick

    # Force symmetry for even N (from MATLAB baryDiffMat.m line 82-83)
    # This forces the diagonal to be symmetric about the center
    if N >= 2:
        half_N = N // 2
        # Set bottom-right corner diagonals equal to negative of top-left
        for k in range(half_N):
            idx = N - 1 - k
            D[idx, idx] = -D[k, k]

    if order == 1:
        return D

    # k = 2 (second derivative)
    D_diag = np.diag(D).copy()
    D = 2 * D * (D_diag[:, None] - Dxi)
    D[ii, ii] = 0.0
    D[ii, ii] = -np.sum(D, axis=1)  # Negative sum trick

    if order == 2:
        return D

    # k = 3, 4, 5, ... (higher derivatives using recursion)
    for k in range(3, order + 1):
        D_diag = np.diag(D).copy()
        D = k * Dxi * (Dw * D_diag[:, None] - D)
        D[ii, ii] = 0.0
        D[ii, ii] = -np.sum(D, axis=1)  # Negative sum trick

    return D


def barycentric_matrix(x_eval, n, interval):
    """Construct barycentric interpolation matrix.

    Creates a matrix that maps values at Chebyshev collocation points
    to values at arbitrary evaluation points via barycentric interpolation.

    For collocation values v = [v_0, ..., v_n] at Chebyshev points,
    E @ v gives interpolated values at x_eval points.

    Args:
        x_eval: Points to evaluate at (scalar or array)
        n: Discretization size (collocation points - 1)
        interval: Interval [a, b] to evaluate on

    Returns:
        scipy.sparse matrix: Evaluation matrix of size (len(x_eval), n+1)

    Examples:
        >>> E = barycentric_matrix(np.array([0.5]), 4, Interval(0, 1))
        >>> E.shape
        (1, 5)
    """
    x_eval = np.atleast_1d(x_eval)
    n_eval = x_eval.size
    n_pts = n + 1

    # Get Chebyshev collocation points
    x_cheb = cheb_points_scaled(n, interval)

    # Barycentric weights for Chebyshev points (second kind)
    weights = barywts2(n_pts)

    # Broadcast differences: shape = (n_eval, n_pts)
    diffs = x_eval[:, None] - x_cheb[None, :]

    # Detect exact node matches
    tol = 1e-14
    mask = np.abs(diffs) < tol  # shape (n_eval, n_pts)

    # Initialize matrix
    E = np.zeros((n_eval, n_pts))

    # Rows with exact node matches (should be 0 or 1)
    rows, cols = np.where(mask)
    if rows.size > 0:
        E[rows, cols] = 1.0

    # For rows without exact matches, use barycentric formula
    no_match = ~np.any(mask, axis=1)
    if np.any(no_match):
        diffs_nm = diffs[no_match]
        numer = weights / diffs_nm  # broadcast divide
        denom = np.sum(numer, axis=1, keepdims=True)
        E[no_match, :] = numer / denom

    return sparse.csr_matrix(E)


def projection_matrix_rectangular(n, m, interval):
    """Construct projection matrix for rectangular spectral collocation.

    This matrix projects from m+1 collocation points down to n+1 coefficient points
    using barycentric interpolation. This is the "PS" matrix in MATLAB Chebfun's
    rectangular spectral collocation for eigenvalue problems.

    Following Driscoll & Hale (2016), the projection matrix maps values at
    m+1 > n+1 collocation points to values at n+1 coefficient points, enabling
    overdetermined discretizations that significantly improve eigenvalue accuracy.

    Args:
        n: Number of coefficient degrees of freedom (yields n+1 points)
        m: Number of collocation points (yields m+1 points), must satisfy m >= n
        interval: Interval [a, b] defining the domain

    Returns:
        scipy.sparse matrix: Projection matrix of size (n+1, m+1)

    Raises:
        ValueError: If m < n (underdetermined system)

    References:
        Driscoll & Hale (2016), "Rectangular spectral collocation"

    Examples:
        >>> PS = projection_matrix_rectangular(8, 16, Interval(-1, 1))
        >>> PS.shape
        (9, 17)
    """
    if m < n:
        raise ValueError(f"Rectangular projection requires m >= n, got m={m}, n={n}")

    # Coefficient grid: n+1 Chebyshev points (where we want to evaluate)
    x_coeff = cheb_points_scaled(n, interval)

    # Collocation grid: m+1 Chebyshev points (where we have values)
    x_colloc = cheb_points_scaled(m, interval)

    # Barycentric weights for collocation grid
    weights_colloc = barywts2(m + 1)

    # Build projection matrix: PS[i, j] interpolates from x_colloc[j] to x_coeff[i]
    # PS @ v_colloc = v_coeff
    diffs = x_coeff[:, None] - x_colloc[None, :]  # shape (n+1, m+1)

    # Detect exact node matches
    tol = 1e-14
    mask = np.abs(diffs) < tol

    # Initialize matrix
    PS = np.zeros((n + 1, m + 1))

    # Rows with exact node matches
    rows, cols = np.where(mask)
    if rows.size > 0:
        PS[rows, cols] = 1.0

    # For rows without exact matches, use barycentric formula
    no_match = ~np.any(mask, axis=1)
    if np.any(no_match):
        diffs_nm = diffs[no_match]
        numer = weights_colloc / diffs_nm
        denom = np.sum(numer, axis=1, keepdims=True)
        PS[no_match, :] = numer / denom

    return sparse.csr_matrix(PS)


def fourier_points_scaled(n, interval: Interval):
    """Compute equally-spaced Fourier collocation points on an arbitrary interval.

    Returns n equally-spaced points on [a, b), excluding the right endpoint
    (which is equivalent to the left endpoint for periodic functions).

    Args:
        n (int): Number of collocation points.
        interval (Interval): Interval [a, b] defining the domain.

    Returns:
        numpy.ndarray: Array of n equally-spaced points on [a, b).
    """
    a, b = interval
    h = (b - a) / n
    return np.arange(n) * h + a


def fourier_diff_matrix(n, interval: Interval, order=1):
    """Compute Fourier spectral differentiation matrix.

    Args:
        n (int): Number of collocation points.
        interval (Interval): Interval [a, b] for the periodic domain.
        order (int): Differentiation order (default 1).

    Returns:
        numpy.ndarray: Dense differentiation matrix of size (n, n).
    """
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([[0.0]])

    # Grid spacing on reference interval [-pi, pi)
    h = 2 * np.pi / n

    # Identity for zeroth order
    if order == 0:
        return np.eye(n)

    # First-order
    elif order == 1:
        if n % 2 == 1:  # odd
            column = np.concatenate([[0.0], 0.5 / np.sin(np.arange(1, n) * h / 2)])
        else:  # even
            column = np.concatenate([[0.0], 0.5 / np.tan(np.arange(1, n) * h / 2)])
        column[1::2] = -column[1::2]
        row = column[[0, *range(n - 1, 0, -1)]]
        D = toeplitz(column, row)

    # Second-order
    elif order == 2:
        if n % 2 == 1:  # odd
            cscc = 1.0 / np.sin(np.arange(1, n) * h / 2)
            cott = 1.0 / np.tan(np.arange(1, n) * h / 2)
            column = np.concatenate([[np.pi**2 / 3 / h**2 - 1 / 12], 0.5 * cscc * cott])
        else:  # even
            cscc_sq = (1.0 / np.sin(np.arange(1, n) * h / 2)) ** 2
            column = np.concatenate([[np.pi**2 / 3 / h**2 + 1 / 6], 0.5 * cscc_sq])
        column[::2] = -column[::2]
        D = toeplitz(column)

    # Higher orders use FFT
    else:
        if n % 2 == 1:  # odd
            column = (1j * np.concatenate([np.arange((n + 1) // 2), np.arange(-(n - 1) // 2, 0)])) ** order
        else:  # even
            column = (1j * np.concatenate([np.arange(n // 2), [0], np.arange(-n // 2 + 1, 0)])) ** order
        D = np.real(np.fft.ifft(column[:, np.newaxis] * np.fft.fft(np.eye(n), axis=0), axis=0))

    # Scale from [-pi, pi) to [a, b)
    a, b = interval
    scale_factor = (2 * np.pi) / (b - a)
    return (scale_factor**order) * D


# ============================================================================
# ULTRASPHERICAL SPECTRAL METHOD (Olver-Townsend)
# ============================================================================
#
# The ultraspherical spectral method works in coefficient space rather than
# collocation (value) space. Key advantages:
# - Banded matrices (O(n) solve vs O(n³) for dense)
# - Better conditioning than collocation
# - Stable for high-order derivatives
#
# Reference: S. Olver and A. Townsend, "A fast and well-conditioned spectral
# method," SIAM Review, 55 (2013), 462-489.


def ultraspherical_diff(n, lmbda=0):
    """Compute differentiation operator in ultraspherical basis.

    Maps Chebyshev coefficients (C^(λ) basis) to C^(λ+1) basis.

    Following MATLAB's diffmat convention: returns n×n matrix operating on
    n coefficients (not n+1).

    The derivative of a Chebyshev T_k polynomial is:
        d/dx T_k(x) = k * U_{k-1}(x) = k * C^(1)_{k-1}(x)

    More generally, for C^(λ) polynomials:
        d/dx C^(λ)_k(x) = 2λ * C^(λ+1)_{k-1}(x)

    Args:
        n (int): Number of coefficients (returns n×n matrix).
        lmbda (int): Current ultraspherical parameter (0 for Chebyshev T).

    Returns:
        scipy.sparse matrix: Differentiation matrix of size n×n.
        Maps C^(λ) coefficients to C^(λ+1) coefficients.
    """
    if n < 1:
        return sparse.csr_matrix((0, 0))

    # Differentiation formula: d/dx C^(λ)_k = 2λ * C^(λ+1)_{k-1}
    # For Chebyshev (λ=0), special case: d/dx T_k = k * U_{k-1}

    if lmbda == 0:
        # Chebyshev case: d/dx T_k = k * U_{k-1} = k * C^(1)_{k-1}
        # MATLAB diffmat: spdiags((0 : n - 1)', 1, n, n)
        diag = np.arange(0, n)
    else:
        # General case: d/dx C^(λ)_k = 2λ * C^(λ+1)_{k-1}
        # MATLAB diffmat: for m>0, multiplies by spdiags(2*m*ones(n,1), 1, n, n)
        diag = 2 * lmbda * np.ones(n)

    # Build sparse matrix: D[i, i+1] = diag[i] (n×n matrix with superdiagonal)
    D = sparse.diags([diag], [1], shape=(n, n), format="csr")
    return D


def ultraspherical_conversion(n, lmbda):
    """Compute conversion matrix between ultraspherical bases.

    Maps C^(λ) coefficients to C^(λ+1) coefficients.

    Following MATLAB's convertmat convention: returns n×n matrix operating on
    n coefficients.

    The conversion uses the recurrence relation:
        C^(λ)_k(x) = (λ/(λ+k)) * [C^(λ+1)_k(x) - C^(λ+1)_{k-2}(x)]

    Inverting this gives the forward conversion (C^λ → C^(λ+1)):
        C^(λ+1)_k = C^(λ)_k + (λ/(λ+k-1)) * C^(λ+1)_{k-2}

    Which leads to a bidiagonal matrix.

    Args:
        n (int): Number of coefficients (returns n×n matrix).
        lmbda (int): Current ultraspherical parameter.

    Returns:
        scipy.sparse matrix: Conversion matrix of size n×n.
    """
    if n < 1:
        return sparse.eye(1, format="csr")

    if lmbda == 0:
        # Special case: Chebyshev T → C^(1)
        # MATLAB spconvert: relation is C_n^(0) = 0.5*(C_n^(1) - C_{n-2}^(1))
        # Inverting: C_n^(1) = 2*C_n^(0) + C_{n-2}^(1)
        # In matrix form with superdiagonal structure:
        # S[0,0] = 1, S[1,1] = 0.5, S[k,k] = 0.5 for k >= 2
        # S[k, k+2] = -0.5 for k >= 0

        # Build main diagonal
        main_diag = np.ones(n)
        if n > 1:
            main_diag[1:] = 0.5

        # Superdiagonal at offset +2 (only if n >= 3)
        if n >= 3:
            super_diag = -0.5 * np.ones(n - 2)
            S = sparse.diags([main_diag, super_diag], [0, 2], shape=(n, n), format="csr")
        else:
            S = sparse.diags([main_diag], [0], shape=(n, n), format="csr")
        return S

    else:
        # General case: C^(λ) → C^(λ+1)
        # MATLAB implementation uses superdiagonal structure
        # S[0,0] = 1, S[1,1] = lam/(lam+1)
        # S[k,k] = lam/(lam+k) for k >= 2
        # S[k, k+2] = -lam/(lam+k) for k >= 0

        # Build main diagonal
        main_diag = np.ones(n)
        if n > 1:
            main_diag[1] = lmbda / (lmbda + 1)
        if n >= 3:
            k_vals = np.arange(2, n)
            main_diag[2:] = lmbda / (lmbda + k_vals)

        # Superdiagonal at offset +2 (only if n >= 3)
        if n >= 3:
            k_super = np.arange(0, n - 2)
            super_diag = -lmbda / (lmbda + k_super + 2)
            S = sparse.diags([main_diag, super_diag], [0, 2], shape=(n, n), format="csr")
        else:
            S = sparse.diags([main_diag], [0], shape=(n, n), format="csr")
        return S


def ultraspherical_multiplication(coeffs_f, n):
    """Build multiplication operator in Chebyshev coefficient space.

    Given a function f(x) represented by Chebyshev coefficients,
    construct the matrix M such that (f*u) has coefficients M @ u_coeffs.

    The multiplication of two Chebyshev series uses the product formula:
        T_m * T_n = (1/2) * [T_{m+n} + T_{|m-n|}]

    This gives a sparse banded matrix when f has limited Chebyshev degree.

    Args:
        coeffs_f (array): Chebyshev coefficients of the multiplier function f.
        n (int): Size of the output coefficient vector.

    Returns:
        scipy.sparse matrix: Multiplication matrix of size (n+1) × (n+1).
    """
    m = len(coeffs_f) - 1  # Degree of f
    nn = n + 1

    # Build the multiplication matrix
    # M[i, j] = coefficient of T_i in (f * T_j)

    # For T_j * sum_k f_k T_k = sum_k f_k * (T_j * T_k)
    # = sum_k f_k * (1/2) * [T_{j+k} + T_{|j-k|}]

    # Build M as dense first for simplicity, then convert to sparse
    M = np.zeros((nn, nn))

    for j in range(nn):
        for k in range(m + 1):
            fk = coeffs_f[k]
            if fk == 0:
                continue

            # T_j * T_k = 0.5 * [T_{j+k} + T_{|j-k|}]
            # We want coefficient in row i

            # Contribution from T_{j+k}
            if j + k < nn:
                if k == 0:
                    M[j + k, j] += fk
                else:
                    M[j + k, j] += 0.5 * fk

            # Contribution from T_{|j-k|}
            abs_jk = abs(j - k)
            if abs_jk < nn:
                if k == 0:
                    # T_j * T_0 = T_j (no factor of 0.5)
                    pass  # Already handled above
                else:
                    M[abs_jk, j] += 0.5 * fk

    return sparse.csr_matrix(M)


def ultraspherical_bc_row(n, interval, bc_order, bc_side):
    """Construct boundary condition row for ultraspherical method.

    The BC row enforces u^(bc_order)(x_bc) = value, where x_bc is the
    left (a) or right (b) endpoint.

    For Chebyshev polynomials:
        T_k(1) = 1 for all k
        T_k(-1) = (-1)^k

    For derivatives at endpoints, we use the formula:
        T'_k(±1) = ±k² (at ±1)
        T''_k(±1) = k²(k²-1)/3 (at ±1, sign alternates)

    Args:
        n (int): Number of Chebyshev coefficients (n+1 total).
        interval (Interval): Physical interval [a, b].
        bc_order (int): Order of derivative (0 for Dirichlet, 1 for Neumann, etc.)
        bc_side (str): 'left' or 'right'.

    Returns:
        numpy.ndarray: Row vector of length n+1.
    """
    nn = n + 1
    a, b = interval if hasattr(interval, "__iter__") else (interval.a, interval.b)
    L = (b - a) / 2  # Half-width of interval

    row = np.zeros(nn)

    if bc_side == "left":
        # Evaluate at x = a, which corresponds to ξ = -1 in [-1, 1]
        xi = -1

        def sign(k):
            return (-1) ** k
    else:
        # Evaluate at x = b, which corresponds to ξ = +1 in [-1, 1]
        xi = 1

        def sign(k):
            return 1

    if bc_order == 0:
        # Dirichlet: u(x_bc) = sum_k c_k * T_k(xi)
        for k in range(nn):
            row[k] = sign(k) if xi == -1 else 1
    elif bc_order == 1:
        # Neumann: u'(x_bc) = sum_k c_k * T'_k(xi) / L
        # T'_k(1) = k², T'_k(-1) = (-1)^{k+1} * k²
        for k in range(nn):
            if xi == 1:
                row[k] = k**2 / L
            else:
                row[k] = ((-1) ** (k + 1)) * (k**2) / L
    elif bc_order == 2:
        # Second derivative: u''(x_bc)
        # T''_k(1) = k²(k²-1)/3, T''_k(-1) = (-1)^k * k²(k²-1)/3
        for k in range(nn):
            val = k**2 * (k**2 - 1) / 3 / (L**2)
            row[k] = sign(k) * val
    else:
        # General case using explicit formula for T^(m)_k at ±1
        # This gets complicated - raise error for order > 2
        raise NotImplementedError(f"BC order {bc_order} not implemented in ultraspherical method")

    return row


def ultraspherical_solve(coeffs, rhs_coeffs, n, interval, lbc, rbc):
    """Solve a linear ODE using the ultraspherical method.

    Follows MATLAB's ultraspherical implementation approach.

    For an ODE of the form:
        a_2(x) * u''(x) + a_1(x) * u'(x) + a_0(x) * u(x) = f(x)

    with boundary conditions at the endpoints.

    Args:
        coeffs (list): Coefficient functions [a_0, a_1, a_2] as constants.
        rhs_coeffs (array): Chebyshev coefficients of RHS f(x).
        n (int): Number of Chebyshev coefficients for solution.
        interval (Interval): Physical domain [a, b].
        lbc (list): Left BCs as list of values or single value.
        rbc (list): Right BCs as list of values or single value.

    Returns:
        numpy.ndarray: Chebyshev coefficients of the solution.
    """
    diff_order = len(coeffs) - 1  # Highest derivative order

    # Scale factor for interval mapping
    a, b = interval if hasattr(interval, "__iter__") else (interval.a, interval.b)
    L = (b - a) / 2

    # Only support 2nd order
    if diff_order != 2:
        raise NotImplementedError(f"Ultraspherical method supports 2nd order ODEs only, got order {diff_order}")

    # Following MATLAB's approach:
    # 1. Build matrices: D0 (n×n), D1 ((n-1)×(n-1)), S0 (n×n), S1 (n×n)
    # 2. Build operator: L = D1 @ D0 + a1 * S1_reduced @ D0 + a0 * S1 @ S0
    # 3. Project: remove last m=2 rows
    # 4. Add BC rows
    # 5. Build RHS and solve

    # Get conversion and differentiation matrices (all n×n now)
    D0 = ultraspherical_diff(n, 0) / L  # d/dx: T → C^(1), scaled for interval
    D1 = ultraspherical_diff(n, 1) / L  # d/dx: C^(1) → C^(2), scaled
    S0 = ultraspherical_conversion(n, 0)  # T → C^(1)
    S1 = ultraspherical_conversion(n, 1)  # C^(1) → C^(2)

    # Build operator matrix in C^(2) basis
    # L_op will be (n)×(n) initially, then we project to (n-2)×(n)

    # u'' term: D1 @ D0
    # Note: D0 is n×n, D1 is n×n, but D1 @ D0 loses a row due to differentiation
    # So we need to be careful about dimensions
    L_op = (D1 @ D0).tocsr()

    # a_1 * u' term
    if len(coeffs) > 1 and coeffs[1] is not None:
        a1_val = float(coeffs[1]) if not isinstance(coeffs[1], (int, float)) else coeffs[1]
        # S1 @ D0: convert C^(1) to C^(2) after differentiation
        L_op = L_op + a1_val * (S1 @ D0).tocsr()

    # a_0 * u term
    if len(coeffs) > 0 and coeffs[0] is not None:
        a0_val = float(coeffs[0]) if not isinstance(coeffs[0], (int, float)) else coeffs[0]
        # S1 @ S0: convert T → C^(1) → C^(2)
        L_op = L_op + a0_val * (S1 @ S0).tocsr()

    # PROJECTION STEP: Remove last m=2 rows (MATLAB's reduce() operation)
    m = diff_order  # For 2nd order ODE, m = 2
    if L_op.shape[0] > m:
        L_op_reduced = L_op[:-m, :]
    else:
        raise ValueError(f"Matrix too small for projection: {L_op.shape[0]} rows, need to remove {m}")

    # Build BC rows (these operate on Chebyshev coefficients directly)
    bc_rows = []
    bc_values = []

    # Process left BCs
    # Note: ultraspherical_bc_row(n, ...) returns n+1 elements,
    # but our matrices are n x n. So we pass n-1 to get n elements.
    if lbc is not None:
        if isinstance(lbc, (int, float)):
            # Dirichlet: u(a) = lbc
            row = ultraspherical_bc_row(n - 1, interval, 0, "left")
            bc_rows.append(row)
            bc_values.append(lbc)
        elif isinstance(lbc, list):
            for i, val in enumerate(lbc):
                if val is not None:
                    row = ultraspherical_bc_row(n - 1, interval, i, "left")
                    bc_rows.append(row)
                    bc_values.append(val)

    # Process right BCs
    if rbc is not None:
        if isinstance(rbc, (int, float)):
            # Dirichlet: u(b) = rbc
            row = ultraspherical_bc_row(n - 1, interval, 0, "right")
            bc_rows.append(row)
            bc_values.append(rbc)
        elif isinstance(rbc, list):
            for i, val in enumerate(rbc):
                if val is not None:
                    row = ultraspherical_bc_row(n - 1, interval, i, "right")
                    bc_rows.append(row)
                    bc_values.append(val)

    # Assemble full system: [BC_rows; L_op_reduced]
    if bc_rows:
        BC_matrix = np.vstack(bc_rows)
        A = sparse.vstack([sparse.csr_matrix(BC_matrix), L_op_reduced])
    else:
        A = L_op_reduced

    # Build RHS vector
    # Convert RHS from T to C^(2) basis, then project

    # Pad or truncate RHS coefficients to size n
    rhs_cheb = np.zeros(n)
    if rhs_coeffs is not None and len(rhs_coeffs) > 0:
        rhs_cheb[: min(len(rhs_coeffs), n)] = rhs_coeffs[:n]

    # Convert T → C^(1) → C^(2)
    rhs_c1 = S0 @ rhs_cheb
    rhs_c2 = S1 @ rhs_c1

    # Project: remove last m=2 entries
    rhs_c2_projected = rhs_c2[:-m] if len(rhs_c2) > m else rhs_c2

    # Full RHS: BC values + projected interior RHS
    b = np.concatenate([bc_values, rhs_c2_projected])

    # Solve the system
    A_dense = sparse_to_dense(A) if sparse.issparse(A) else A
    coeffs_solution = np.linalg.lstsq(A_dense, b, rcond=None)[0]

    return coeffs_solution
