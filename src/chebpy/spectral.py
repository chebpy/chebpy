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
from .utilities import Interval


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
        - Typical choices: m = 2*n or m = n + 50 (MATLAB Chebfun heuristic).
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
        interval = chebfun.support

    x = cheb_points_scaled(n, interval)

    # Evaluate chebfun at these points
    values = chebfun(x)

    # Ensure values is 1D (MATLAB Chebfun returns column vectors)
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
