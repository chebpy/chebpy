"""Unit tests for the importing utilities in chebpy.core.importing.

This module contains tests that verify the behavior of the import_optional function
in the chebpy.core.importing module, including successful imports, fallback behavior,
and environment variable control.
"""

import os

from chebpy.core.importing import import_optional


class TestImporting:
    """Test suite for importing utilities in chebpy.core.importing."""

    def test_import_optional_success(self):
        """Test that import_optional successfully imports an existing module."""
        # Test importing a standard library module
        os_module = import_optional("os", "OS")
        assert os_module is not None
        assert os_module.__name__ == "os"

    def test_import_optional_with_fallback(self):
        """Test that import_optional uses the fallback when the primary import fails."""
        # Test with a non-existent module and a fallback
        module = import_optional("non_existent_module", "NON_EXISTENT", fallback="os")
        assert module is not None
        assert module.__name__ == "os"

    def test_import_optional_no_fallback(self):
        """Test that import_optional returns None when both the import fails and there's no fallback."""
        # Test with a non-existent module and no fallback
        module = import_optional("non_existent_module", "NON_EXISTENT")
        assert module is None

    def test_import_optional_disabled(self):
        """Test that import_optional respects the environment variable to disable imports."""
        # Set the environment variable to disable the import
        os.environ["CHEBPY_USE_OS"] = "0"
        try:
            # This should use the fallback even though os exists
            module = import_optional("os", "OS", fallback="sys")
            assert module is not None
            assert module.__name__ == "sys"

            # This should return None when disabled and no fallback
            module = import_optional("os", "OS")
            assert module is None
        finally:
            # Clean up the environment variable
            os.environ.pop("CHEBPY_USE_OS", None)
