"""Unit tests for the exception classes in chebpy.core.exceptions.

This module contains tests that verify the behavior of all exception classes
defined in the chebpy.core.exceptions module, including initialization with
custom messages, default messages, and the abstract base class behavior.
"""

import pytest

from chebpy.core.exceptions import (
    BadFunLengthArgument,
    ChebpyBaseException,
    IntervalGap,
    IntervalMismatch,
    IntervalOverlap,
    IntervalValues,
    InvalidDomain,
    NotSubdomain,
    NotSubinterval,
    SupportMismatch,
)


class TestExceptions:
    """Test suite for exception classes in chebpy.core.exceptions."""

    def test_exception_with_custom_message(self):
        """Test that exceptions can be initialized with a custom message."""
        custom_message = "This is a custom error message"
        exc = IntervalOverlap(custom_message)
        assert exc.message == custom_message
        assert str(exc) == custom_message

    def test_exception_with_default_message(self):
        """Test that exceptions use their default message when no message is provided."""
        exc = IntervalOverlap()
        assert exc.message == "The supplied Interval objects overlap"
        assert str(exc) == "The supplied Interval objects overlap"

    def test_all_exception_types(self):
        """Test that all exception types can be instantiated."""
        exceptions = [
            IntervalOverlap,
            IntervalGap,
            IntervalMismatch,
            NotSubinterval,
            IntervalValues,
            InvalidDomain,
            NotSubdomain,
            SupportMismatch,
            BadFunLengthArgument,
        ]

        for exc_class in exceptions:
            # Test with default message
            exc = exc_class()
            assert isinstance(exc, ChebpyBaseException)
            assert isinstance(exc, Exception)
            assert exc.message == exc.default_message
            assert str(exc) == exc.default_message

            # Test with custom message
            custom_msg = f"Custom message for {exc_class.__name__}"
            exc_custom = exc_class(custom_msg)
            assert exc_custom.message == custom_msg
            assert str(exc_custom) == custom_msg

    def test_abstract_default_message(self):
        """Test that ChebpyBaseException is abstract and requires default_message."""
        # We can't instantiate ChebpyBaseException directly because it's abstract,
        # but we can create a concrete subclass and test that it raises NotImplementedError
        # when default_message is accessed

        class TestException(ChebpyBaseException):
            pass

        # This should raise NotImplementedError when default_message is accessed
        with pytest.raises(NotImplementedError):
            TestException()
