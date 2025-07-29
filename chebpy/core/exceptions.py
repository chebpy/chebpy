"""Exception classes for the ChebPy package.

This module defines the exception hierarchy used throughout the ChebPy package.
It includes a base exception class and various specific exception types for
different error conditions related to intervals, domains, and function operations.
"""

from abc import ABC, abstractmethod


class ChebpyBaseError(Exception, ABC):
    """Abstract base class for all ChebPy exceptions.

    This class serves as the base for all exception types in the ChebPy package.
    It provides a common interface for exception handling and requires subclasses
    to define a default error message.

    Attributes:
        message (str): The error message to be displayed.
    """

    def __init__(self, *args):
        """Initialize the exception with an optional custom message.

        Args:
            *args: Variable length argument list. If provided, the first argument
                is used as the error message. Otherwise, the default message is used.
        """
        if args:
            self.message = args[0]
        else:
            self.message = self.default_message

    def __str__(self):
        """Return the string representation of the exception.

        Returns:
            str: The error message.
        """
        return self.message

    @property
    @abstractmethod
    def default_message(self):
        """Default error message for the exception.

        This property must be implemented by all concrete subclasses.

        Returns:
            str: The default error message.

        Raises:
            NotImplementedError: If the subclass does not implement this property.
        """
        raise NotImplementedError


# ===============================================
#    chebpy.core.utilities.Interval exceptions
# ===============================================

# Exception raised when two intervals overlap but should be disjoint
IntervalOverlap = type(
    "IntervalOverlap",
    (ChebpyBaseError,),
    {
        "default_message": "The supplied Interval objects overlap",
        "__doc__": """Exception raised when intervals overlap.

        This exception is raised when two or more intervals overlap
        but are required to be disjoint for the operation.
        """
    },
)

# Exception raised when intervals have gaps between them
IntervalGap = type(
    "IntervalGap",
    (ChebpyBaseError,),
    {
        "default_message": "The supplied Interval objects do not form a complete partition "
        "of the approximation interval",
        "__doc__": """Exception raised when intervals have gaps.

        This exception is raised when a collection of intervals does not
        form a complete partition of the approximation interval.
        """
    },
)

# Exception raised when intervals don't match for an operation
IntervalMismatch = type(
    "IntervalMismatch",
    (ChebpyBaseError,),
    {
        "default_message": "This operation can only be performed for Fun objects defined on identical intervals",
        "__doc__": """Exception raised when intervals don't match.

        This exception is raised when an operation requires Fun objects
        to be defined on identical intervals, but they are not.
        """
    },
)

# Exception raised when an interval is not a subinterval of another
NotSubinterval = type(
    "NotSubinterval",
    (ChebpyBaseError,),
    {
        "default_message": "Not a subinterval",
        "__doc__": """Exception raised when an interval is not a subinterval.

        This exception is raised when an interval is expected to be
        a subinterval of another interval, but it is not.
        """
    },
)

# Exception raised when interval values are not strictly increasing
IntervalValues = type(
    "IntervalValues",
    (ChebpyBaseError,),
    {
        "default_message": "The defining values of a Interval object must be strictly increasing",
        "__doc__": """Exception raised when interval values are invalid.

        This exception is raised when the defining values of an Interval
        object are not strictly increasing.
        """
    },
)


# ===============================================
#    chebpy.core.utilities.Domain exceptions
# ===============================================

# Exception raised when a domain is invalid
InvalidDomain = type(
    "InvalidDomain",
    (ChebpyBaseError,),
    {
        "default_message": "Domain objects must be initialised from an iterable "
        "collection of at least two monotonically increasing "
        "scalars",
        "__doc__": """Exception raised when a domain is invalid.

        This exception is raised when attempting to create a Domain object
        with invalid parameters, such as non-monotonic values or too few points.
        """
    },
)

# Exception raised when a domain is not a subdomain of another
NotSubdomain = type(
    "NotSubdomain",
    (ChebpyBaseError,),
    {
        "default_message": "The support of the target Domain object is required "
        "to define a subinterval of the support of the "
        "original",
        "__doc__": """Exception raised when a domain is not a subdomain.

        This exception is raised when a domain is expected to be
        a subdomain of another domain, but it is not.
        """
    },
)

# Exception raised when supports don't match for an operation
SupportMismatch = type(
    "SupportMismatch",
    (ChebpyBaseError,),
    {
        "default_message": "Both objects are required to be supported on the same interval",
        "__doc__": """Exception raised when supports don't match.

        This exception is raised when an operation requires objects
        to be supported on the same interval, but they are not.
        """
    },
)

# Exception raised when the length argument for a function is invalid
BadFunLengthArgument = type(
    "BadFunLengthArgument",
    (ChebpyBaseError,),
    {
        "default_message": "The 'n' argument must be either a single numeric "
        "value, or iterable thereof posessing one fewer "
        "elements than the size of the domain",
        "__doc__": """Exception raised when a function length argument is invalid.

        This exception is raised when the 'n' argument for a function does not
        meet the requirements: it must be either a single numeric value or an
        iterable with one fewer elements than the size of the domain.
        """
    },
)
