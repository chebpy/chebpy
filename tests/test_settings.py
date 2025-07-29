"""Unit-tests for chebpy/core/settings.py.

This module contains tests for the UserPreferences class and its functionality,
including updating preferences with a context manager and resetting preferences
to their default values.
"""

from chebpy import UserPreferences
from chebpy.core.settings import _preferences


def _user_pref(name: str) -> float:
    """Get the current preference value for name.

    This is how chebpy core modules access these.

    Args:
        name: Name of the preference to retrieve.

    Returns:
        Current value of the preference.
    """
    return getattr(_preferences, name)


def test_update_pref() -> None:
    """Test updating preferences with context manager.

    This test verifies that:
    1. Preferences can be updated within a context manager
    2. The updated values are accessible during the context
    3. The original values are restored after the context exits
    """
    eps_new = 1e-3
    eps_old = _user_pref("eps")
    with UserPreferences() as prefs:
        prefs.eps = eps_new
        assert eps_new == _user_pref("eps")
    assert eps_old == _user_pref("eps")


def test_reset() -> None:
    """Test resetting preferences.

    This test verifies that:
    1. Preferences can be reset to their default values
    2. Both global reset and named reset functionality work correctly
    3. After reset, preferences return to their original values
    """

    def change(reset_named: bool = False) -> None:
        """Helper function to change and reset preferences.

        Args:
            reset_named: Whether to reset a specific preference or all preferences.
                Defaults to False.
        """
        prefs = UserPreferences()
        prefs.eps = 99
        if reset_named:
            prefs.reset("eps")
        else:
            prefs.reset()

    eps_bak = _user_pref("eps")
    change(False)
    assert eps_bak == _user_pref("eps")
    change(True)
    assert eps_bak == _user_pref("eps")
