"""Test sparse_utils module for matrix and numerical utilities."""

import numpy as np
from scipy import sparse

from chebpy.sparse_utils import extract_scalar, jacobian_to_row, is_nearly_zero, prune_sparse, sparse_to_dense


class TestExtractScalar:
    """Test extract_scalar function."""

    def test_scalar_float(self):
        """Extract from Python float."""
        assert extract_scalar(5.0) == 5.0

    def test_scalar_int(self):
        """Extract from Python int."""
        assert extract_scalar(3) == 3.0

    def test_array_single_element(self):
        """Extract from single-element array."""
        arr = np.array([2.5])
        assert extract_scalar(arr) == 2.5

    def test_array_multiple_elements(self):
        """Extract first element from array."""
        arr = np.array([2.5, 3.5])
        assert extract_scalar(arr) == 2.5

    def test_list_single_element(self):
        """Extract from single-element list."""
        assert extract_scalar([7.0]) == 7.0

    def test_list_multiple_elements(self):
        """Extract first element from list."""
        lst = [7.0, 8.0]
        assert extract_scalar(lst) == 7.0

    def test_negate_flag(self):
        """Extract and negate scalar value."""
        assert extract_scalar(5.0, negate=True) == -5.0
        assert extract_scalar([3.0], negate=True) == -3.0

    def test_empty_list(self):
        """Extract from empty list returns 0.0."""
        assert extract_scalar([]) == 0.0

    def test_zero_value(self):
        """Extract zero value."""
        assert extract_scalar(0.0) == 0.0

    def test_negative_value(self):
        """Extract negative value."""
        assert extract_scalar(-2.5) == -2.5

    def test_adchebfun_scalar(self):
        """Extract from AdChebfunScalar wrapper."""
        # Create a mock object that mimics AdChebfunScalar interface
        # Must be named exactly "AdChebfunScalar" to match the type check in extract_scalar
        class AdChebfunScalar:
            def __init__(self, value):
                self.value = value

        mock_scalar = AdChebfunScalar(np.array([2.5]))
        result = extract_scalar(mock_scalar)
        assert isinstance(result, float)
        np.testing.assert_almost_equal(result, 2.5)

    def test_unknown_type(self):
        """Extract from unknown object type returns 0.0."""
        class UnknownType:
            pass
        unknown_obj = UnknownType()
        assert extract_scalar(unknown_obj) == 0.0


class TestJacobianToRow:
    """Test jacobian_to_row function."""

    def test_dense_1d_array(self):
        """Convert 1D dense array to row."""
        arr = np.array([1, 2, 3])
        row = jacobian_to_row(arr)
        assert row.shape == (3,)
        np.testing.assert_array_equal(row, [1, 2, 3])

    def test_dense_2d_array(self):
        """Convert 2D dense array to row."""
        mat = np.array([[1, 2, 3]])
        row = jacobian_to_row(mat)
        assert row.shape == (3,)
        np.testing.assert_array_equal(row, [1, 2, 3])

    def test_sparse_csr_matrix(self):
        """Convert sparse CSR matrix to row."""
        mat = sparse.csr_matrix([[1, 0, 2, 0]])
        row = jacobian_to_row(mat)
        assert row.shape == (4,)
        np.testing.assert_array_equal(row, [1, 0, 2, 0])

    def test_sparse_csc_matrix(self):
        """Convert sparse CSC matrix to row."""
        mat = sparse.csc_matrix([[0, 1, 0, 3]])
        row = jacobian_to_row(mat)
        assert row.shape == (4,)
        np.testing.assert_array_equal(row, [0, 1, 0, 3])

    def test_sparse_lil_matrix(self):
        """Convert sparse LIL matrix to row."""
        mat = sparse.lil_matrix([[1, 2, 3]])
        row = jacobian_to_row(mat)
        assert row.shape == (3,)
        np.testing.assert_array_equal(row, [1, 2, 3])

    def test_scalar_value(self):
        """Convert scalar value to row."""
        row = jacobian_to_row(5.0)
        assert row.shape == (1,)
        np.testing.assert_array_equal(row, [5.0])


class TestIsNearlyZero:
    """Test is_nearly_zero function."""

    def test_dense_below_threshold(self):
        """Detect dense matrix below threshold."""
        mat = np.array([[1e-13, 1e-14]])
        assert is_nearly_zero(mat, threshold=1e-12)

    def test_dense_above_threshold(self):
        """Detect dense matrix above threshold."""
        mat = np.array([[1e-11, 1e-12]])
        assert not is_nearly_zero(mat, threshold=1e-12)

    def test_sparse_below_threshold(self):
        """Detect sparse matrix below threshold."""
        mat = sparse.csr_matrix([[1e-14, 0, 1e-15]])
        assert is_nearly_zero(mat, threshold=1e-12)

    def test_sparse_above_threshold(self):
        """Detect sparse matrix above threshold."""
        mat = sparse.csr_matrix([[1e-10, 0, 0]])
        assert not is_nearly_zero(mat, threshold=1e-12)

    def test_zero_array(self):
        """Detect zero array."""
        mat = np.zeros((2, 3))
        assert is_nearly_zero(mat, threshold=1e-12)

    def test_sparse_zero_matrix(self):
        """Detect sparse zero matrix."""
        mat = sparse.csr_matrix((2, 3))
        assert is_nearly_zero(mat, threshold=1e-12)

    def test_default_threshold(self):
        """Use default threshold."""
        mat = np.array([[1e-13]])
        assert is_nearly_zero(mat)  # Default threshold is 1e-12


class TestPruneSparse:
    """Test prune_sparse function."""

    def test_removes_tiny_coefficients(self):
        """Prune removes tiny coefficients."""
        mat = sparse.lil_matrix([[1e-15, 1.0, 1e-16, 2.5]])
        pruned = prune_sparse(mat, threshold=1e-14)
        assert pruned.nnz == 2
        assert pruned[0, 1] == 1.0
        assert pruned[0, 3] == 2.5

    def test_returns_csr_format(self):
        """Prune always returns CSR format."""
        mat = sparse.csc_matrix([[1.0, 2.0]])
        pruned = prune_sparse(mat)
        assert isinstance(pruned, sparse.csr_matrix)

    def test_preserves_significant_values(self):
        """Prune preserves values above threshold."""
        mat = sparse.lil_matrix([[1.0, 1e-15, 0.5]])
        pruned = prune_sparse(mat, threshold=1e-14)
        assert pruned[0, 0] == 1.0
        assert pruned[0, 2] == 0.5

    def test_empty_matrix(self):
        """Prune empty matrix."""
        mat = sparse.csr_matrix((3, 3))
        pruned = prune_sparse(mat)
        assert pruned.nnz == 0

    def test_default_threshold(self):
        """Use default threshold."""
        mat = sparse.lil_matrix([[1e-15, 1.0]])
        pruned = prune_sparse(mat)  # Default threshold is 1e-14
        assert pruned.nnz == 1
        assert pruned[0, 1] == 1.0


class TestSparseToDense:
    """Test sparse_to_dense function."""

    def test_sparse_csr_to_dense(self):
        """Convert sparse CSR to dense array."""
        sparse_mat = sparse.csr_matrix([[1, 0, 2], [0, 3, 0]])
        dense = sparse_to_dense(sparse_mat)
        expected = np.array([[1, 0, 2], [0, 3, 0]])
        np.testing.assert_array_equal(dense, expected)

    def test_sparse_csc_to_dense(self):
        """Convert sparse CSC to dense array."""
        sparse_mat = sparse.csc_matrix([[1, 2], [0, 3]])
        dense = sparse_to_dense(sparse_mat)
        expected = np.array([[1, 2], [0, 3]])
        np.testing.assert_array_equal(dense, expected)

    def test_dense_passthrough(self):
        """Convert dense to dense (passthrough)."""
        mat = np.array([[1, 2], [3, 4]])
        result = sparse_to_dense(mat)
        np.testing.assert_array_equal(result, mat)

    def test_empty_sparse_matrix(self):
        """Convert empty sparse matrix."""
        sparse_mat = sparse.csr_matrix((2, 3))
        dense = sparse_to_dense(sparse_mat)
        expected = np.zeros((2, 3))
        np.testing.assert_array_equal(dense, expected)

    def test_preserves_values(self):
        """Verify values are preserved."""
        vals = [[1.5, 0, 2.3], [0, -1.1, 0]]
        sparse_mat = sparse.csr_matrix(vals)
        dense = sparse_to_dense(sparse_mat)
        np.testing.assert_array_almost_equal(dense, vals)
