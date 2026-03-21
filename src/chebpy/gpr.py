"""Gaussian process regression with Chebfun representations.

Implements Gaussian process regression (GPR) following the algorithm described
in Rasmussen & Williams, *Gaussian Processes for Machine Learning*, MIT Press,
2006, and the MATLAB Chebfun ``gpr.m`` by The University of Oxford and The
Chebfun Developers.

The posterior mean, variance, and (optionally) random samples from the
posterior are all returned as Chebfun / Quasimatrix objects so that they can
be manipulated with the full ChebPy toolkit (differentiation, integration,
rootfinding, etc.).

Reference:
    C. E. Rasmussen & C. K. I. Williams, "Gaussian Processes for Machine
    Learning", MIT Press, 2006.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from .algorithms import chebpts2
from .chebfun import Chebfun
from .quasimatrix import Quasimatrix


# ---------------------------------------------------------------------------
#  Options container
# ---------------------------------------------------------------------------
@dataclass
class _GPROptions:
    """Parsed options for a GPR call."""

    sigma: float = 1.0
    sigma_given: bool = False
    length_scale: float = 0.0
    noise: float = 0.0
    domain: np.ndarray = field(default_factory=lambda: np.array([-1.0, 1.0]))
    trig: bool = False
    n_samples: int = 0


# ---------------------------------------------------------------------------
#  Kernel helpers
# ---------------------------------------------------------------------------


def _kernel_matrix(
    x1: np.ndarray,
    x2: np.ndarray,
    opts: _GPROptions,
) -> np.ndarray:
    """Evaluate the covariance kernel k(x1_i, x2_j) for all pairs."""
    r = x1[:, None] - x2[None, :]
    if opts.trig:
        period = opts.domain[1] - opts.domain[0]
        return opts.sigma**2 * np.exp(-2.0 / opts.length_scale**2 * np.sin(np.pi / period * r) ** 2)
    return opts.sigma**2 * np.exp(-0.5 / opts.length_scale**2 * r**2)


def _log_marginal_likelihood(
    length_scale: float | np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    opts: _GPROptions,
) -> float | np.ndarray:
    """Negative log marginal likelihood (eq. 2.30 in Rasmussen & Williams).

    Accepts scalar or array *length_scale* so that it can be wrapped as a
    Chebfun for optimisation.
    """
    scalar_input = np.ndim(length_scale) == 0
    ls = np.atleast_1d(np.asarray(length_scale, dtype=float))
    n = len(x)
    rx = x[:, None] - x[None, :]
    result = np.empty_like(ls)

    for idx in np.ndindex(ls.shape):
        l_val = ls[idx]
        if opts.trig:
            period = opts.domain[1] - opts.domain[0]
            cov_mat = opts.sigma**2 * np.exp(-2.0 / l_val**2 * np.sin(np.pi / period * rx) ** 2)
        else:
            cov_mat = opts.sigma**2 * np.exp(-0.5 / l_val**2 * rx**2)

        if opts.noise != 0:
            cov_mat += opts.noise**2 * np.eye(n)
        else:
            cov_mat += 1e-15 * n * opts.sigma**2 * np.eye(n)

        chol_l = np.linalg.cholesky(cov_mat)
        alpha = np.linalg.solve(chol_l.T, np.linalg.solve(chol_l, y))
        lml = -0.5 * y @ alpha - np.sum(np.log(np.diag(chol_l))) - 0.5 * n * np.log(2 * np.pi)
        result[idx] = lml

    return float(result.item()) if scalar_input else result.ravel()


# ---------------------------------------------------------------------------
#  Length-scale selection via max log marginal likelihood
# ---------------------------------------------------------------------------


def _select_length_scale(x: np.ndarray, y: np.ndarray, opts: _GPROptions) -> float:
    """Choose the length-scale that maximises the log marginal likelihood."""
    n = len(x)
    dom_size = opts.domain[1] - opts.domain[0]

    if opts.trig:
        lo, hi = 1.0 / (2 * n), 10.0
    else:
        lo, hi = dom_size / (2 * np.pi * n), 10.0 / np.pi * dom_size

    # Heuristic: shrink the right end of the search domain if the lml is
    # monotonically decreasing (mirrors the MATLAB implementation).
    f1 = float(_log_marginal_likelihood(lo, x, y, opts))
    f2 = float(_log_marginal_likelihood(hi, x, y, opts))
    while f1 > f2 and hi / lo > 1 + 1e-4:
        new_bound = lo + (hi - lo) / 10.0
        f_new = float(_log_marginal_likelihood(new_bound, x, y, opts))
        if f_new > f1:
            break
        hi = new_bound
        f2 = f_new

    # Maximise using golden-section search (negated to find the max).
    return _golden_section_max(lambda ls: float(_log_marginal_likelihood(ls, x, y, opts)), lo, hi)


def _golden_section_max(f: Callable[[float], float], a: float, b: float, tol: float = 1e-6) -> float:
    """Golden-section search for the scalar argmax of *f* on [a, b]."""
    gr = (np.sqrt(5.0) + 1.0) / 2.0
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(b - a) > tol * (abs(a) + abs(b)):
        if f(c) > f(d):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    return 0.5 * (a + b)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------


def _parse_inputs(
    x: ArrayLike,
    y: ArrayLike,
    *,
    sigma: float | None,
    noise: float,
    trig: bool,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, _GPROptions, float]:
    """Validate inputs and build the initial options container.

    Returns ``(x_arr, y_arr, opts, scaling_factor)``.
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if x_arr.shape != y_arr.shape:
        msg = "x and y must have the same length."
        raise ValueError(msg)

    opts = _GPROptions(trig=trig, noise=noise, n_samples=n_samples)

    scaling_factor = 1.0
    if sigma is not None:
        opts.sigma = sigma
        opts.sigma_given = True
    else:
        if len(y_arr) > 0:
            scaling_factor = float(np.max(np.abs(y_arr)))
        opts.sigma_given = False
        opts.sigma = scaling_factor

    return x_arr, y_arr, opts, scaling_factor


def _infer_domain(
    x_arr: np.ndarray,
    opts: _GPROptions,
    domain: tuple[float, float] | list[float] | np.ndarray | None,
) -> None:
    """Set ``opts.domain`` from *domain* or from the observation locations."""
    if domain is not None:
        opts.domain = np.asarray(domain, dtype=float)
    elif len(x_arr) == 0:
        opts.domain = np.array([-1.0, 1.0])
    elif len(x_arr) == 1:
        opts.domain = np.array([x_arr[0] - 1, x_arr[0] + 1])
    elif opts.trig:
        span = float(np.max(x_arr) - np.min(x_arr))
        opts.domain = np.array([float(np.min(x_arr)), float(np.max(x_arr)) + 0.1 * span])
    else:
        opts.domain = np.array([float(np.min(x_arr)), float(np.max(x_arr))])


def _infer_length_scale(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    opts: _GPROptions,
    scaling_factor: float,
    length_scale: float | None,
) -> None:
    """Set ``opts.length_scale`` — user-supplied or auto-selected."""
    if length_scale is not None:
        opts.length_scale = length_scale
        return

    if len(x_arr) == 0:
        opts.length_scale = 1.0
        return

    y_n = y_arr / scaling_factor if scaling_factor != 0 else y_arr

    if not opts.sigma_given:
        tmp = _GPROptions(
            sigma=1.0,
            sigma_given=True,
            noise=opts.noise / scaling_factor if scaling_factor != 0 else opts.noise,
            domain=opts.domain,
            trig=opts.trig,
        )
        y_opt = y_n
    else:
        tmp = _GPROptions(
            sigma=opts.sigma,
            sigma_given=True,
            noise=opts.noise,
            domain=opts.domain,
            trig=opts.trig,
        )
        y_opt = y_arr

    opts.length_scale = _select_length_scale(x_arr, y_opt, tmp)


def _posterior_chebfuns(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    opts: _GPROptions,
    scaling_factor: float,
    n_samples: int,
) -> tuple[Chebfun, Chebfun] | tuple[Chebfun, Chebfun, Quasimatrix]:
    """Compute posterior mean, variance, and optional samples as Chebfuns."""
    n = len(x_arr)
    cov_mat = _kernel_matrix(x_arr, x_arr, opts)
    if opts.noise == 0:
        cov_mat += 1e-15 * scaling_factor**2 * n * np.eye(n)
    else:
        cov_mat += opts.noise**2 * np.eye(n)

    chol_l = np.linalg.cholesky(cov_mat)
    alpha = np.linalg.solve(chol_l.T, np.linalg.solve(chol_l, y_arr))

    # Shared Chebyshev grid
    sample_size = min(20 * n, 2000)
    t = chebpts2(sample_size)
    x_sample = 0.5 * (opts.domain[1] - opts.domain[0]) * t + 0.5 * (opts.domain[0] + opts.domain[1])

    in_x = np.isin(x_sample, x_arr)

    k_star = _kernel_matrix(x_sample, x_arr, opts)
    if opts.noise:
        k_star += opts.noise**2 * (np.abs(x_sample[:, None] - x_arr[None, :]) == 0)

    # Posterior mean
    mean_vals = k_star @ alpha
    f_mean = Chebfun.initfun_fixedlen(lambda _z: mean_vals, sample_size, opts.domain)

    # Posterior variance
    k_ss = _kernel_matrix(x_sample, x_sample, opts)
    if opts.noise:
        k_ss += opts.noise**2 * np.diag(in_x.astype(float))

    v = np.linalg.solve(chol_l, k_star.T)
    var_diag = np.diag(k_ss) - np.sum(v**2, axis=0)
    var_diag = np.maximum(var_diag, 0.0)
    f_var = Chebfun.initfun_fixedlen(lambda _z: var_diag, sample_size, opts.domain)

    if n_samples <= 0:
        return f_mean, f_var

    # Posterior samples
    cov_post = k_ss - v.T @ v
    cov_post = 0.5 * (cov_post + cov_post.T)
    cov_post += 1e-12 * scaling_factor**2 * n * np.eye(sample_size)
    chol_s = np.linalg.cholesky(cov_post)

    draws = mean_vals[:, None] + chol_s @ np.random.randn(sample_size, n_samples)
    cols: list[Chebfun] = []
    for j in range(n_samples):
        cols.append(
            Chebfun.initfun_fixedlen(
                lambda _z, _j=j: draws[:, _j],
                sample_size,
                opts.domain,
            )
        )
    return f_mean, f_var, Quasimatrix(cols)


def gpr(
    x: ArrayLike,
    y: ArrayLike,
    *,
    domain: tuple[float, float] | list[float] | np.ndarray | None = None,
    sigma: float | None = None,
    length_scale: float | None = None,
    noise: float = 0.0,
    trig: bool = False,
    n_samples: int = 0,
) -> tuple[Chebfun, Chebfun] | tuple[Chebfun, Chebfun, Quasimatrix]:
    """Gaussian process regression returning Chebfun objects.

    Given observations ``(x, y)`` of a latent function, compute the posterior
    mean and variance of a Gaussian process with zero prior mean and a squared
    exponential kernel::

        k(x, x') = sigma**2 * exp(-0.5 / L**2 * (x - x')**2)

    When ``trig=True`` a periodic variant is used instead::

        k(x, x') = sigma**2 * exp(-2 / L**2 * sin(pi * (x - x') / P)**2)

    where *P* is the period (length of the domain).

    Args:
        x: Observation locations (1-D array-like).
        y: Observation values (same length as *x*).
        domain: Domain ``[a, b]`` for the output Chebfuns.  Defaults to
            ``[min(x), max(x)]`` (or slightly extended for ``trig``).
        sigma: Signal variance of the kernel.  Defaults to ``max(|y|)``.
        length_scale: Length-scale *L* of the kernel.  If ``None``, it is
            chosen to maximise the log marginal likelihood.
        noise: Standard deviation of i.i.d. Gaussian observation noise.
            The kernel diagonal is augmented by ``noise**2``.
        trig: If ``True``, use a periodic squared-exponential kernel.
        n_samples: Number of independent posterior samples to draw.  When
            positive, a :class:`Quasimatrix` with *n_samples* columns is
            returned as the third element of the output tuple.

    Returns:
        ``(f_mean, f_var)`` — posterior mean and variance as Chebfun objects.
        If ``n_samples > 0``, returns ``(f_mean, f_var, samples)`` where
        *samples* is a Quasimatrix whose columns are independent draws from
        the posterior.

    Raises:
        ValueError: If *x* and *y* have different lengths or are empty.

    Examples:
        >>> import numpy as np
        >>> from chebpy.gpr import gpr
        >>> rng = np.random.default_rng(1)
        >>> x = -2 + 4 * rng.random(10)
        >>> y = np.sin(np.exp(x))
        >>> f_mean, f_var = gpr(x, y, domain=[-2, 2])

    Reference:
        C. E. Rasmussen & C. K. I. Williams, "Gaussian Processes for Machine
        Learning", MIT Press, 2006.
    """
    x_arr, y_arr, opts, scaling_factor = _parse_inputs(
        x,
        y,
        sigma=sigma,
        noise=noise,
        trig=trig,
        n_samples=n_samples,
    )
    _infer_domain(x_arr, opts, domain)
    _infer_length_scale(x_arr, y_arr, opts, scaling_factor, length_scale)

    # No data → return prior
    if len(x_arr) == 0:
        f_mean = Chebfun.initconst(0.0, opts.domain)
        f_var = Chebfun.initconst(opts.sigma**2, opts.domain)
        if n_samples > 0:
            return f_mean, f_var, _prior_samples(opts, scaling_factor, n_samples)
        return f_mean, f_var

    return _posterior_chebfuns(x_arr, y_arr, opts, scaling_factor, n_samples)


def _prior_samples(
    opts: _GPROptions,
    scaling_factor: float,
    n_samples: int,
) -> Quasimatrix:
    """Draw samples from the GP prior (no observations)."""
    sample_size = 1000
    if opts.trig:
        x_sample = np.linspace(opts.domain[0], opts.domain[1], sample_size)
    else:
        t = chebpts2(sample_size)
        x_sample = 0.5 * (opts.domain[1] - opts.domain[0]) * t + 0.5 * (opts.domain[0] + opts.domain[1])

    k_ss = _kernel_matrix(x_sample, x_sample, opts)
    k_ss += 1e-12 * scaling_factor**2 * np.eye(sample_size)
    chol_s = np.linalg.cholesky(k_ss)

    f_mean_vals = np.zeros(sample_size)
    draws = f_mean_vals[:, None] + chol_s @ np.random.randn(sample_size, n_samples)

    cols: list[Chebfun] = []
    for j in range(n_samples):
        cols.append(
            Chebfun.initfun_fixedlen(
                lambda _z, _j=j: draws[:, _j],
                sample_size,
                opts.domain,
            )
        )
    return Quasimatrix(cols)
