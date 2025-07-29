"""Utilities for optional imports in the ChebPy package.

This module provides functionality for handling optional dependencies,
allowing the package to gracefully degrade when certain dependencies
are not available.
"""

import os
from importlib import import_module


def import_optional(name: str, envvarname: str, fallback: str = None) -> object:
    """Attempt to import an optional module with fallback and environment control.

    This function tries to import a module that may or may not be installed.
    It provides a way to control the import behavior using environment variables
    and to specify a fallback module if the primary import fails.

    Args:
        name (str): The name of the module to import.
        envvarname (str): The name to use in the environment variable that controls
            whether to attempt the import. The actual environment variable will be
            "CHEBPY_USE_" followed by the uppercase version of this name.
        fallback (str, optional): The name of a fallback module to import if the
            primary import fails or is disabled. Defaults to None.

    Returns:
        module or None: The imported module if successful, or None if the import
            failed and no fallback was provided.

    Note:
        The environment variable "CHEBPY_USE_<ENVVARNAME>" can be set to "0" or
        "false" to disable the import attempt for the primary module.

    Example:
        >>> # Try to import matplotlib, fall back to None if not available
        >>> mpl = import_optional("matplotlib", "MPL")
        >>>
        >>> # Try to import numpy, fall back to math if not available
        >>> np = import_optional("numpy", "NUMPY", fallback="math")

    See Also:
        https://github.com/pandas-dev/pandas/blob/master/pandas/compat/_optional.py
    """
    use_module = os.environ.get("CHEBPY_USE_" + envvarname.upper(), "1")
    if use_module.lower() in ["true", "1"]:
        try:
            return import_module(name)
        except ImportError:  # pragma: no cover
            pass
    if fallback is not None:
        return import_module(fallback)
    else:
        return None
