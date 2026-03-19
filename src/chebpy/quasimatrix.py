"""Quasimatrix: a matrix with one continuous dimension.

A quasimatrix is an inf x n matrix whose columns are Chebfun objects defined on
the same domain. This enables continuous analogues of linear algebra operations
such as QR factorization, SVD, least-squares, and more.

Reference: Trefethen, "Householder triangularization of a quasimatrix,"
IMA Journal of Numerical Analysis, 30 (2010), 887-897.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .chebfun import Chebfun


class Quasimatrix:
    """An inf x n column quasimatrix whose columns are Chebfun objects.

    A quasimatrix generalises the idea of a matrix so that one of its
    dimensions is continuous.  Here the rows are indexed by points in an
    interval and the columns are Chebfun objects.

    Attributes:
        columns: list of Chebfun objects forming the columns.
    """

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------
    def __init__(self, columns: list[Any]) -> None:
        """Initialise from a list of Chebfun objects, callables, or scalars."""
        cols: list[Chebfun] = []
        for c in columns:
            if isinstance(c, Chebfun):
                cols.append(c)
            elif callable(c):
                cols.append(Chebfun.initfun_adaptive(c, cols[0].domain if cols else None))
            else:
                # scalar → constant chebfun on the domain of the first column
                if cols:
                    cols.append(Chebfun.initconst(float(c), cols[0].domain))
                else:
                    from .settings import _preferences as prefs

                    cols.append(Chebfun.initconst(float(c), prefs.domain))
        if len(cols) > 1:
            # Verify all columns share the same support
            ref = cols[0].support
            for k, col in enumerate(cols[1:], 1):
                if col.support != ref:
                    msg = f"Column {k} support {col.support} does not match column 0 support {ref}"
                    raise ValueError(msg)
        self.columns: list[Chebfun] = cols

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple[float, int]:
        """Return (∞, n) where n is the number of columns."""
        return (np.inf, len(self.columns))

    @property
    def T(self) -> _TransposedQuasimatrix:
        """Return the transpose (an n x inf row quasimatrix)."""
        return _TransposedQuasimatrix(self)

    @property
    def domain(self) -> Any:
        """Domain of the quasimatrix columns."""
        if not self.columns:
            return None
        return self.columns[0].domain

    @property
    def support(self) -> tuple[float, float]:
        """Support interval of the quasimatrix."""
        if not self.columns:
            return (0.0, 0.0)
        return self.columns[0].support

    @property
    def isempty(self) -> bool:
        """Return True if the quasimatrix has no columns."""
        return len(self.columns) == 0

    # ------------------------------------------------------------------
    #  Indexing  A[:, k],  A(x, k)
    # ------------------------------------------------------------------
    def __getitem__(self, key: Any) -> Any:
        """Column indexing: ``A[:, k]`` returns column k as a Chebfun."""
        if isinstance(key, tuple):
            row, col = key
            if isinstance(col, slice):
                return Quasimatrix(self.columns[col])
            # A[x, k] - evaluate column k at point x
            if isinstance(row, slice) and row == slice(None):
                return self.columns[col]
            return self.columns[col](row)
        # A[k] - return column k
        if isinstance(key, (int, np.integer)):
            return self.columns[key]
        if isinstance(key, slice):
            return Quasimatrix(self.columns[key])
        raise TypeError(key)

    def __len__(self) -> int:
        """Return the number of columns."""
        return len(self.columns)

    def __iter__(self):
        """Iterate over the columns."""
        return iter(self.columns)

    # ------------------------------------------------------------------
    #  Calling  A(x) - evaluate all columns at x, return array
    # ------------------------------------------------------------------
    def __call__(self, x: Any) -> np.ndarray:
        """Evaluate every column at *x* and return the results as an array.

        If *x* is a scalar the result has shape ``(n,)``.
        If *x* is an array of length *m* the result has shape ``(m, n)``.
        """
        vals = [col(x) for col in self.columns]
        return np.column_stack(vals) if np.ndim(x) else np.array(vals)

    # ------------------------------------------------------------------
    #  Arithmetic
    # ------------------------------------------------------------------
    def __matmul__(self, other: Any) -> Any:
        """Matrix-vector product: ``A @ c`` returns a Chebfun.

        *other* must be a 1-D array-like of length n.
        """
        c = np.asarray(other, dtype=float)
        if c.ndim != 1 or len(c) != len(self.columns):
            msg = f"Cannot multiply {self.shape} quasimatrix by vector of length {len(c)}"
            raise ValueError(msg)
        result = c[0] * self.columns[0]
        for coeff, col in zip(c[1:], self.columns[1:], strict=True):
            result = result + coeff * col
        return result

    def __mul__(self, other: Any) -> Quasimatrix:
        """Element-wise scalar multiplication."""
        return Quasimatrix([c * other for c in self.columns])

    def __rmul__(self, other: Any) -> Quasimatrix:
        """Right scalar multiplication."""
        return self.__mul__(other)

    # ------------------------------------------------------------------
    #  Integrals and inner products
    # ------------------------------------------------------------------
    def sum(self) -> np.ndarray:
        """Definite integral of each column (column sums)."""
        return np.array([col.sum() for col in self.columns])

    def inner(self, other: Quasimatrix | None = None) -> np.ndarray:
        """Gram matrix ``self.T @ other`` (or ``self.T @ self``)."""
        other = other if other is not None else self
        m = len(self.columns)
        n = len(other.columns)
        G = np.empty((m, n))
        for i in range(m):
            for j in range(n):
                G[i, j] = self.columns[i].dot(other.columns[j])
        return G

    # ------------------------------------------------------------------
    #  QR factorization  (modified Gram-Schmidt)
    # ------------------------------------------------------------------
    def qr(self) -> tuple[Quasimatrix, np.ndarray]:
        """Compute the reduced QR factorization ``A = Q R``.

        Uses modified Gram-Schmidt orthogonalisation in function space.

        Returns:
            Q: Quasimatrix with orthonormal columns.
            R: Upper-triangular n x n NumPy array.
        """
        n = len(self.columns)
        Q = [col.copy() for col in self.columns]
        R = np.zeros((n, n))
        for k in range(n):
            for j in range(k):
                R[j, k] = Q[j].dot(Q[k])
                Q[k] = Q[k] - R[j, k] * Q[j]
            R[k, k] = Q[k].norm(2)
            if R[k, k] == 0:
                msg = "Rank-deficient quasimatrix: QR factorization failed"
                raise np.linalg.LinAlgError(msg)
            Q[k] = (1.0 / R[k, k]) * Q[k]
        return Quasimatrix(Q), R

    # ------------------------------------------------------------------
    #  SVD
    # ------------------------------------------------------------------
    def svd(self) -> tuple[Quasimatrix, np.ndarray, np.ndarray]:
        """Compute the reduced SVD ``A = U S V^T``.

        Returns:
            U: inf x n quasimatrix with orthonormal columns.
            S: 1-D array of singular values (length n).
            V: n x n orthogonal NumPy matrix.
        """
        Q, R = self.qr()
        # Economy SVD of the n x n matrix R
        U_r, S, Vt = np.linalg.svd(R, full_matrices=False)
        # U = Q @ U_r  (linear combinations of orthonormal columns)
        U_cols = []
        for j in range(U_r.shape[1]):
            U_cols.append(Q @ U_r[:, j])
        return Quasimatrix(U_cols), S, Vt.T  # V = Vt.T

    # ------------------------------------------------------------------
    #  Least-squares  (backslash)
    # ------------------------------------------------------------------
    def solve(self, f: Chebfun) -> np.ndarray:
        r"""Least-squares solution ``c`` to ``A c ~ f``.

        Equivalent to MATLAB ``A\f``.  Computed via QR factorisation.
        """
        Q, R = self.qr()
        # b = Q' * f  (inner products)
        b = np.array([col.dot(f) for col in Q.columns])
        # Solve R c = b  (back-substitution)
        return np.linalg.solve(R, b)

    # ------------------------------------------------------------------
    #  Norms
    # ------------------------------------------------------------------
    def norm(self, p: Any = "fro") -> float:
        """Compute the norm of the quasimatrix.

        Args:
            p: Norm type.
                - 2: the 2-norm (largest singular value).
                - 1: max column 1-norm.
                - np.inf: max row-sum := max_x sum_j |A_j(x)|.
                - 'fro': Frobenius norm (default).
        """
        if p == 2:
            _, S, _ = self.svd()
            return float(S[0])
        if p == 1:
            return float(max(col.norm(1) for col in self.columns))
        if p == np.inf:
            abssum = self.columns[0].absolute()
            for col in self.columns[1:]:
                abssum = abssum + col.absolute()
            return float(abssum.norm(np.inf))
        if p == "fro":
            _, S, _ = self.svd()
            return float(np.sqrt(np.sum(S**2)))
        raise ValueError(f"Unsupported norm type: {p}")  # noqa: TRY003

    # ------------------------------------------------------------------
    #  Condition number
    # ------------------------------------------------------------------
    def cond(self) -> float:
        """2-norm condition number (ratio of largest to smallest singular value)."""
        _, S, _ = self.svd()
        return float(S[0] / S[-1])

    # ------------------------------------------------------------------
    #  Rank
    # ------------------------------------------------------------------
    def rank(self, tol: float | None = None) -> int:
        """Numerical rank (number of significant singular values)."""
        _, S, _ = self.svd()
        if tol is None:
            tol = max(self.shape[1], 20) * np.finfo(float).eps * S[0]
        return int(np.sum(tol < S))

    # ------------------------------------------------------------------
    #  Null space
    # ------------------------------------------------------------------
    def null(self, tol: float | None = None) -> np.ndarray:
        """Orthonormal basis for the null space of the quasimatrix.

        Returns an n x k NumPy array whose columns span ``null(A)``.
        """
        _, S, V = self.svd()
        if tol is None:
            tol = max(self.shape[1], 20) * np.finfo(float).eps * S[0]
        mask = tol >= S
        return V[:, mask]

    # ------------------------------------------------------------------
    #  Orth  (orthonormal basis for range)
    # ------------------------------------------------------------------
    def orth(self, tol: float | None = None) -> Quasimatrix:
        """Orthonormal basis for the column space (range) of the quasimatrix."""
        U, S, _ = self.svd()
        if tol is None:
            tol = max(self.shape[1], 20) * np.finfo(float).eps * S[0]
        mask = tol < S
        return Quasimatrix([U.columns[j] for j in range(len(S)) if mask[j]])

    # ------------------------------------------------------------------
    #  Pseudoinverse
    # ------------------------------------------------------------------
    def pinv(self) -> _TransposedQuasimatrix:
        """Moore-Penrose pseudoinverse (returned as an n x inf row quasimatrix).

        ``pinv(A) @ f`` gives the same result as ``A.solve(f)``.
        """
        U, S, V = self.svd()
        # pinv(A) = V S^{-1} U^T
        # The rows of pinv(A) are: sum_k V[i,k] / S[k] * U_k
        n = len(S)
        pinv_cols: list[Chebfun] = []
        for i in range(n):
            col = (V[i, 0] / S[0]) * U.columns[0]
            for k in range(1, n):
                col = col + (V[i, k] / S[k]) * U.columns[k]
            pinv_cols.append(col)
        return _TransposedQuasimatrix(Quasimatrix(pinv_cols))

    # ------------------------------------------------------------------
    #  Plotting
    # ------------------------------------------------------------------
    def plot(self, ax: Axes | None = None, **kwds: Any) -> Axes:
        """Plot all columns on the same axes."""
        ax = ax or plt.gca()
        for col in self.columns:
            col.plot(ax=ax, **kwds)
        return ax

    def spy(self, ax: Axes | None = None, **kwds: Any) -> Axes:
        """Visualise the shape of the quasimatrix.

        Draws a rectangle representing the inf x n structure, with a dot for
        each column to indicate nonzero content.
        """
        ax = ax or plt.gca()
        n = len(self.columns)
        # Draw the bounding rectangle
        rect = plt.Rectangle((0.5, 0.5), n, 10, fill=False, edgecolor="black", linewidth=1.5)
        ax.add_patch(rect)
        # A dot for each column
        for j in range(n):
            ax.plot(j + 1, 5.5, "bs", markersize=8, **kwds)
        ax.set_xlim(0, n + 1)
        ax.set_ylim(0, 11)
        ax.set_aspect("equal")
        ax.set_xlabel(f"n = {n}")
        ax.set_ylabel("∞")
        ax.set_xticks(range(1, n + 1))
        ax.set_yticks([])
        return ax

    # ------------------------------------------------------------------
    #  Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        """Return a string representation."""
        n = len(self.columns)
        if n == 0:
            return "Quasimatrix(empty)"
        sup = self.support
        return f"Quasimatrix(inf x {n} on [{sup[0]}, {sup[1]}])"

    def __str__(self) -> str:
        """Return a string representation."""
        return self.__repr__()


class _TransposedQuasimatrix:
    """An n x inf row quasimatrix (transpose of a column quasimatrix).

    This is a thin wrapper that enables ``A.T @ f`` and ``A.T @ B``
    with the correct semantics.
    """

    def __init__(self, qm: Quasimatrix) -> None:
        """Wrap a column quasimatrix as its transpose."""
        self._qm = qm

    @property
    def shape(self) -> tuple[int, float]:
        """Return (n, inf)."""
        return (len(self._qm.columns), np.inf)

    @property
    def T(self) -> Quasimatrix:
        """Return the original column quasimatrix."""
        return self._qm

    def __matmul__(self, other: Any) -> Any:
        """Compute inner products: ``A.T @ f`` or ``A.T @ B``."""
        if isinstance(other, Quasimatrix):
            return self._qm.inner(other)
        if isinstance(other, Chebfun):
            return np.array([col.dot(other) for col in self._qm.columns])
        raise TypeError(f"Cannot multiply _TransposedQuasimatrix by {type(other)}")  # noqa: TRY003

    def spy(self, ax: Axes | None = None, **kwds: Any) -> Axes:
        """Visualise the shape of the transposed quasimatrix."""
        ax = ax or plt.gca()
        n = len(self._qm.columns)
        rect = plt.Rectangle((0.5, 0.5), 10, n, fill=False, edgecolor="black", linewidth=1.5)
        ax.add_patch(rect)
        for j in range(n):
            ax.plot(5.5, j + 1, "bs", markersize=8, **kwds)
        ax.set_xlim(0, 11)
        ax.set_ylim(0, n + 1)
        ax.set_aspect("equal")
        ax.set_ylabel(f"n = {n}")
        ax.set_xlabel("∞")
        ax.set_yticks(range(1, n + 1))
        ax.set_xticks([])
        return ax

    def __repr__(self) -> str:
        """Return a string representation."""
        n = len(self._qm.columns)
        if n == 0:
            return "_TransposedQuasimatrix(empty)"
        sup = self._qm.support
        return f"_TransposedQuasimatrix({n}x inf on [{sup[0]}, {sup[1]}])"


# ------------------------------------------------------------------
#  Module-level convenience functions
# ------------------------------------------------------------------
def polyfit(f: Chebfun, n: int) -> Chebfun:
    """Least-squares polynomial fit of degree *n* to a Chebfun *f*.

    Returns a Chebfun representing the best degree-*n* polynomial
    approximation to *f* in the L²-norm.
    """
    x = Chebfun.initidentity(f.domain)
    cols: list[Chebfun] = [Chebfun.initconst(1.0, f.domain)]
    xk = cols[0]
    for _ in range(n):
        xk = xk * x
        cols.append(xk)
    A = Quasimatrix(cols)
    c = A.solve(f)
    return A @ c
