"""Tests for the Quasimatrix class and polyfit function.

These tests verify the quasimatrix linear algebra operations against the
examples from Chebfun Guide Chapter 6 (Quasimatrices and Least-Squares).
"""

import numpy as np
import pytest

from chebpy import Quasimatrix, chebfun, polyfit


@pytest.fixture
def x():
    """Chebfun of x on [-1, 1]."""
    return chebfun("x")


@pytest.fixture
def A(x):  # noqa: N802
    """Quasimatrix of monomials 1, x, x^2, ..., x^5 on [-1, 1]."""
    return Quasimatrix([1, x, x**2, x**3, x**4, x**5])


# ------------------------------------------------------------------
#  Section 6.1 - Construction, shape, indexing, inner products
# ------------------------------------------------------------------
class TestQuasimatrixBasics:
    def test_shape(self, A):
        assert A.shape == (np.inf, 6)

    def test_len(self, A):
        assert len(A) == 6

    def test_column_indexing(self, A, x):
        col2 = A[:, 2]
        assert abs(float(col2(0.5)) - 0.25) < 1e-14

    def test_scalar_indexing(self, A):
        assert abs(A[0.5, 2] - 0.25) < 1e-14

    def test_integer_indexing(self, A, x):
        col = A[0]
        assert abs(float(col(0.3)) - 1.0) < 1e-14

    def test_slice_indexing(self, A):
        sub = A[:, 1:3]
        assert sub.shape == (np.inf, 2)

    def test_call_scalar(self, A):
        vals = A(0.5)
        expected = np.array([1, 0.5, 0.25, 0.125, 0.0625, 0.03125])
        np.testing.assert_allclose(vals, expected, atol=1e-14)

    def test_call_array(self, A):
        pts = np.array([0.0, 0.5])
        vals = A(pts)
        assert vals.shape == (2, 6)
        np.testing.assert_allclose(vals[0], [1, 0, 0, 0, 0, 0], atol=1e-14)

    def test_column_sums(self, A):
        s = A.sum()
        expected = np.array([2.0, 0.0, 2.0 / 3, 0.0, 2.0 / 5, 0.0])
        np.testing.assert_allclose(s, expected, atol=1e-14)

    def test_inner_product(self, A):
        # <x^2, x^4> = integral of x^6 from -1 to 1 = 2/7
        ip = A[:, 2].dot(A[:, 4])
        assert abs(ip - 2.0 / 7) < 1e-14

    def test_gram_matrix(self, A):
        G = A.T @ A
        assert G.shape == (6, 6)
        assert abs(G[0, 0] - 2.0) < 1e-14
        assert abs(G[0, 2] - 2.0 / 3) < 1e-14

    def test_transpose_chebfun_matmul(self, A, x):
        # A.T @ x should be [0, 2/3, 0, 2/5, 0, 2/7]
        v = A.T @ x
        expected = np.array([0.0, 2.0 / 3, 0.0, 2.0 / 5, 0.0, 2.0 / 7])
        np.testing.assert_allclose(v, expected, atol=1e-14)


class TestQuasimatrixArithmetic:
    def test_matmul_vector(self, A, x):
        # A @ [0, 1, 0, 0, 0, 0] should give x
        c = np.array([0, 1, 0, 0, 0, 0])
        result = A @ c
        assert abs(float(result(0.5)) - 0.5) < 1e-14

    def test_scalar_multiply(self, A):
        B = 2 * A
        assert abs(float(B[0](0.0)) - 2.0) < 1e-14

    def test_matmul_wrong_size(self, A):
        with pytest.raises(ValueError, match="Cannot multiply"):
            A @ np.array([1, 2, 3])


# ------------------------------------------------------------------
#  Section 6.2 - Least-squares and polyfit
# ------------------------------------------------------------------
class TestLeastSquares:
    def test_solve(self, A, x):
        f = np.exp(x) * np.sin(6 * x)
        c = A.solve(f)
        assert len(c) == 6
        np.testing.assert_allclose(c[0], 0.309654988398406, atol=1e-10)

    def test_solve_error(self, A, x):
        f = np.exp(x) * np.sin(6 * x)
        c = A.solve(f)
        ffit = A @ c
        error = (f - ffit).norm(2)
        np.testing.assert_allclose(error, 0.356073976001434, atol=1e-10)

    def test_polyfit(self, x):
        f = np.exp(x) * np.sin(6 * x)
        ffit = polyfit(f, 5)
        # Compare with manual quasimatrix approach
        A = Quasimatrix([1, x, x**2, x**3, x**4, x**5])
        c = A.solve(f)
        ffit_manual = A @ c
        assert (ffit - ffit_manual).norm(2) < 1e-13


# ------------------------------------------------------------------
#  Section 6.3 - QR factorization
# ------------------------------------------------------------------
class TestQR:
    def test_qr_shapes(self, A):
        Q, R = A.qr()
        assert Q.shape == (np.inf, 6)
        assert R.shape == (6, 6)

    def test_qr_orthonormality(self, A):
        Q, _R = A.qr()
        QTQ = Q.T @ Q
        np.testing.assert_allclose(QTQ, np.eye(6), atol=1e-10)

    def test_qr_upper_triangular(self, A):
        _, R = A.qr()
        assert np.allclose(R, np.triu(R), atol=1e-14)

    def test_qr_reconstruction(self, A):
        Q, R = A.qr()
        # A = Q R, so A @ e_k should equal Q @ R[:, k]
        for k in range(6):
            ek = np.zeros(6)
            ek[k] = 1.0
            col_A = A @ ek
            col_QR = Q @ R[:, k]
            assert (col_A - col_QR).norm(2) < 1e-10

    def test_legendre_normalisation(self, A):
        """QR of monomials gives Legendre polynomials after P(1)=1 renorm."""
        Q, R = A.qr()
        # Renormalize so each Q column satisfies Q_k(1) = 1  (Legendre convention)
        R_new = R.copy()
        Q_cols = [col.copy() for col in Q.columns]
        for j in range(len(Q_cols)):
            val_at_1 = float(Q_cols[j](1.0))
            R_new[j, :] *= val_at_1
            Q_cols[j] = (1.0 / val_at_1) * Q_cols[j]
        inv_R_new = np.linalg.inv(R_new)
        # P_3(x) = 2.5 x^3 - 1.5 x
        np.testing.assert_allclose(inv_R_new[0, 3], 0.0, atol=1e-10)
        np.testing.assert_allclose(inv_R_new[1, 3], -1.5, atol=1e-10)
        np.testing.assert_allclose(inv_R_new[3, 3], 2.5, atol=1e-10)


# ------------------------------------------------------------------
#  Section 6.4 - SVD, norm, cond
# ------------------------------------------------------------------
class TestSVD:
    def test_singular_values(self, A):
        _, S, _ = A.svd()
        expected = [
            1.532062889375341,
            1.032551897396700,
            0.518125864967969,
            0.258419769500035,
            0.080938947808205,
            0.035425077461572,
        ]
        np.testing.assert_allclose(S, expected, atol=1e-8)

    def test_svd_shapes(self, A):
        U, S, V = A.svd()
        assert U.shape == (np.inf, 6)
        assert S.shape == (6,)
        assert V.shape == (6, 6)

    def test_svd_orthonormality(self, A):
        U, _, V = A.svd()
        UTU = U.T @ U
        np.testing.assert_allclose(UTU, np.eye(6), atol=1e-10)
        np.testing.assert_allclose(V.T @ V, np.eye(6), atol=1e-10)

    def test_norm_2(self, A):
        np.testing.assert_allclose(A.norm(2), 1.532062889375341, atol=1e-8)

    def test_cond(self, A):
        np.testing.assert_allclose(A.cond(), 43.247975704139819, atol=1e-6)


# ------------------------------------------------------------------
#  Section 6.5 - Other norms
# ------------------------------------------------------------------
class TestNorms:
    def test_norm_1(self, A):
        assert abs(A.norm(1) - 2.0) < 1e-14

    def test_norm_inf(self, A):
        # 1 + 1 + 1 + 1 + 1 + 1 = 6 at x = ±1
        np.testing.assert_allclose(A.norm(np.inf), 6.0, atol=1e-10)

    def test_norm_fro(self, A):
        np.testing.assert_allclose(A.norm("fro"), 1.938148951041007, atol=1e-8)


# ------------------------------------------------------------------
#  Section 6.6 - rank, null, orth, pinv
# ------------------------------------------------------------------
class TestRankNullOrthPinv:
    def test_rank_deficient(self, x):
        B = Quasimatrix([1, np.sin(x) ** 2, np.cos(x) ** 2])
        assert B.rank() == 2

    def test_null_space(self, x):
        B = Quasimatrix([1, np.sin(x) ** 2, np.cos(x) ** 2])
        N = B.null()
        assert N.shape == (3, 1)
        # Check it's actually in the null space: B @ N[:,0] ≈ 0
        nullvec = N[:, 0]
        zero_fn = B @ nullvec
        assert zero_fn.norm(2) < 1e-10

    def test_orth(self, x):
        B = Quasimatrix([1, np.sin(x) ** 2, np.cos(x) ** 2])
        orth_basis = B.orth()
        assert orth_basis.shape == (np.inf, 2)
        # Columns should be orthonormal
        OTO = orth_basis.T @ orth_basis
        np.testing.assert_allclose(OTO, np.eye(2), atol=1e-10)

    def test_pinv(self, A, x):
        f = np.exp(x) * np.sin(6 * x)
        c_solve = A.solve(f)
        P = A.pinv()
        c_pinv = P @ f
        np.testing.assert_allclose(c_pinv, c_solve, atol=1e-10)

    def test_full_rank(self, A):
        assert A.rank() == 6


# ------------------------------------------------------------------
#  Miscellaneous
# ------------------------------------------------------------------
class TestMisc:
    def test_repr(self, A):
        assert "inf x 6" in repr(A)

    def test_transpose_repr(self, A):
        r = repr(A.T)
        assert "6x" in r or "6 x" in r

    def test_empty_quasimatrix(self):
        q = Quasimatrix([])
        assert q.isempty
        assert q.shape == (np.inf, 0)

    def test_iter(self, A):
        cols = list(A)
        assert len(cols) == 6

    def test_domain_property(self, A):
        assert A.domain is not None

    def test_transpose_t_roundtrip(self, A):
        assert A.T.T is A

    def test_matmul_type_error(self, A):
        with pytest.raises(TypeError):
            A.T @ "invalid"

    def test_bare_slice_indexing(self, A):
        """A[0:2] without tuple returns a Quasimatrix subset."""
        sub = A[0:2]
        assert isinstance(sub, Quasimatrix)
        assert sub.shape == (np.inf, 2)

    def test_getitem_type_error(self, A):
        """Indexing with an unsupported key type raises TypeError."""
        with pytest.raises(TypeError):
            A["bad_key"]

    def test_qr_rank_deficient(self, x):
        """QR on a rank-deficient quasimatrix raises LinAlgError."""
        B = Quasimatrix([x, 2 * x])  # linearly dependent columns
        with pytest.raises(np.linalg.LinAlgError, match="Rank-deficient"):
            B.qr()

    def test_support_mismatch(self, x):
        y = chebfun("x", [0, 2])
        with pytest.raises(ValueError, match="support"):
            Quasimatrix([x, y])

    def test_unsupported_norm(self, A):
        with pytest.raises(ValueError):
            A.norm(3)


# ------------------------------------------------------------------
#  Additional coverage
# ------------------------------------------------------------------
class TestConstruction:
    def test_from_callables(self):
        """Construct a quasimatrix from plain callables."""
        x = chebfun("x")
        A = Quasimatrix([x, lambda t: t**2])
        assert A.shape == (np.inf, 2)
        np.testing.assert_allclose(float(A[1](0.5)), 0.25, atol=1e-14)

    def test_scalar_first_column(self):
        """First column is a bare scalar (uses default domain)."""
        A = Quasimatrix([3.0])
        np.testing.assert_allclose(float(A[0](0.0)), 3.0, atol=1e-14)

    def test_scalar_after_chebfun(self):
        """Scalar column after a Chebfun column inherits the domain."""
        x = chebfun("x")
        A = Quasimatrix([x, 5.0])
        assert A.shape == (np.inf, 2)
        np.testing.assert_allclose(float(A[1](0.3)), 5.0, atol=1e-14)

    def test_non_default_domain(self):
        """Quasimatrix on a non-default domain [0, 2]."""
        y = chebfun("x", [0, 2])
        one = chebfun(1.0, [0, 2])
        B = Quasimatrix([one, y, y**2])
        np.testing.assert_allclose(B.sum(), [2.0, 2.0, 8.0 / 3], atol=1e-12)


class TestCrossGram:
    def test_inner_cross(self, A, x):
        """inner(A, B) where B ≠ A."""
        B = Quasimatrix([1, x])
        G = A.inner(B)
        assert G.shape == (6, 2)
        # G[k,0] = int(x^k, -1, 1)
        np.testing.assert_allclose(G[0, 0], 2.0, atol=1e-14)
        np.testing.assert_allclose(G[1, 0], 0.0, atol=1e-14)


class TestSVDReconstruction:
    def test_svd_reconstruction(self, A):
        """A v_k = sigma_k u_k for each k (Guide §6.4)."""
        U, S, V = A.svd()
        for k in range(len(S)):
            vk = V[:, k]
            Avk = A @ vk
            sk_uk = S[k] * U[k]
            assert (Avk - sk_uk).norm(2) < 1e-10

    def test_frobenius_matches_singular_values(self, A):
        _, S, _ = A.svd()
        np.testing.assert_allclose(A.norm("fro"), np.sqrt(np.sum(S**2)), atol=1e-12)


class TestPlotAndSpy:
    def test_plot(self, A):
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ret = A.plot(ax=ax)
        assert ret is ax
        plt.close(fig)

    def test_spy(self, A):
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ret = A.spy(ax=ax)
        assert ret is ax
        plt.close(fig)

    def test_spy_transposed(self, A):
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ret = A.T.spy(ax=ax)
        assert ret is ax
        plt.close(fig)


class TestEmptyEdgeCases:
    def test_empty_support(self):
        q = Quasimatrix([])
        assert q.support == (0.0, 0.0)

    def test_empty_domain(self):
        q = Quasimatrix([])
        assert q.domain is None

    def test_empty_repr(self):
        assert "empty" in repr(Quasimatrix([]))

    def test_transposed_empty_repr(self):
        assert "empty" in repr(Quasimatrix([]).T)

    def test_str(self, A):
        """str() delegates to __repr__."""
        assert str(A) == repr(A)

    def test_transposed_shape(self, A):
        """Transposed quasimatrix reports correct shape."""
        assert A.T.shape == (6, np.inf)


class TestPolyfitExtended:
    def test_polyfit_low_degree(self, x):
        """polyfit(f, 0) returns the mean value as a constant."""
        f = chebfun(lambda t: t**2, [-1, 1])
        p0 = polyfit(f, 0)
        # Best constant approx to x^2 on [-1,1] is its mean = 1/3
        np.testing.assert_allclose(float(p0(0.0)), 2.0 / 3 / 2, atol=1e-10)

    def test_polyfit_exact(self, x):
        """polyfit(f, n) is exact when f is degree ≤ n."""
        f = x**3 - 2 * x + 1
        p = polyfit(f, 3)
        assert (f - p).norm(2) < 1e-12

    def test_polyfit_nondefault_domain(self):
        y = chebfun("x", [0, 1])
        f = np.exp(y)
        p = polyfit(f, 4)
        assert (f - p).norm(2) < 0.005


class TestConditionNumberGrowth:
    def test_monomial_conditioning_increases(self, x):
        """cond([1, x, ..., x^n]) grows with n."""
        cond_prev = 1.0
        for n in [3, 6, 10]:
            cols = [x**k for k in range(n + 1)]
            cols[0] = chebfun(1.0)
            cond_n = Quasimatrix(cols).cond()
            assert cond_n > cond_prev
            cond_prev = cond_n
