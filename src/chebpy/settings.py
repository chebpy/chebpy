"""Configuration and preferences management for the ChebPy package.

This module provides classes for managing preferences and settings in ChebPy.
It implements a singleton pattern for the preferences object, ensuring that
all parts of the package use the same settings. It also provides a context
manager interface for temporarily changing preferences.

The main classes are:
- DefaultPreferences: Defines the default values for all preferences
- ChebPreferences: The actual preferences object used throughout the package

The module creates a singleton instance of ChebPreferences called _preferences,
which is imported by other modules to access the current settings.
"""

from types import TracebackType
from typing import Any, ClassVar

import numpy as np


class DefaultPreferences:
    """Default preferences for chebpy."""

    eps = np.finfo(float).eps
    tech = "Chebtech"
    domain = np.array([-1.0, 1.0])  # TODO: should this be .utilities.Domain?
    N_plot = 2001
    maxpow2 = 16
    maxiter = 10
    sortroots = False
    mergeroots = True

    @classmethod
    def _defaults(cls) -> dict[str, Any]:
        """Returns all defined class attributes."""
        return {k: v for k, v in cls.__dict__.items() if k[0] != "_"}


class ChebPreferences(DefaultPreferences):
    """Preferences object used in chebpy."""

    def reset(self, *names: str) -> None:
        """Reset default preferences.

        `.reset()` resets all preferences to the DefaultPrefs state
        `.reset(*names)` resets only the selected ones.
        This leaves additional user-added prefs untouched.
        """
        reset_names: Any = names if len(names) > 0 else DefaultPreferences._defaults()
        for name in reset_names:
            if hasattr(DefaultPreferences, name):
                setattr(self, name, getattr(DefaultPreferences, name))

    # Singleton
    _instance: ClassVar["ChebPreferences | None"] = None  # persistent reference for the singleton object

    def __new__(cls) -> "ChebPreferences":
        """Create or return the singleton instance of ChebPreferences.

        This method implements the singleton pattern, ensuring that only one
        instance of ChebPreferences exists. If an instance already exists,
        it returns that instance; otherwise, it creates a new one.

        Args:
            cls (type): The class being instantiated (ChebPreferences).

        Returns:
            ChebPreferences: The singleton instance of ChebPreferences.
        """
        if cls._instance is None:
            cls._instance = super(DefaultPreferences, cls).__new__(cls)
        return cls._instance

    # Context manager
    _stash: ClassVar[list[dict[str, Any]]] = []  # persistent stash for old prefs when entering context(s)

    def __enter__(self) -> "ChebPreferences":
        """Save current preferences when entering a context.

        This method is called when entering a context manager block. It saves
        the current preferences to a stack so they can be restored when exiting
        the context.

        Args:
            self (ChebPreferences): The preferences object.

        Returns:
            ChebPreferences: The preferences object (self._instance).
        """
        self._stash.append({k: getattr(self, k) for k in DefaultPreferences._defaults().keys()})
        return self._instance  # type: ignore[return-value]

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore previous preferences when exiting a context.

        This method is called when exiting a context manager block. It restores
        the preferences to their previous values by popping the stashed values
        from the stack and setting them back on the object.

        Args:
            self (ChebPreferences): The preferences object.
            exc_type: The exception type, if an exception was raised in the context.
            exc_value: The exception value, if an exception was raised in the context.
            traceback: The traceback, if an exception was raised in the context.
        """
        for k, v in self._stash.pop().items():
            setattr(self, k, v)


# create the singleton object for easy import in sister modules
_preferences = ChebPreferences()
