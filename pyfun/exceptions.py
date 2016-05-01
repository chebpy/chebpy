# -*- coding: utf-8 -*-

class IntervalOverlap(Exception):
    pass

class IntervalGap(Exception):
    pass

class IntervalMismatch(Exception):
    """Raised when two Classicfun intervals do not match"""

class IntervalValues(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "The defining values of a Interval object must "\
            "be strictly increasing"
        super(self.__class__, self).__init__(message)

class BadDomainArgument(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "The \'domain\' argument must be an iterable "\
                "containing two or more elements"
        super(self.__class__, self).__init__(message)

class BadFunLengthArgument(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "The \'n\' argument must be either a single "\
                "numeric value, or an iterable thereof containing one "\
                "fewer elements than the size of the domain"
        super(self.__class__, self).__init__(message)
