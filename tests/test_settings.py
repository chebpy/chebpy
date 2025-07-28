import pytest

from chebpy import UserPreferences
from chebpy.core.settings import _preferences


def userPref(name: str) -> float:
    """Get the current preference value for name.

    This is how chebpy core modules access these.

    Parameters
    ----------
    name : str
        Name of the preference to retrieve

    Returns
    -------
    float
        Current value of the preference
    """
    return getattr(_preferences, name)


def test_update_pref() -> None:
    """Test updating preferences with context manager."""
    eps_new = 1e-3
    eps_old = userPref("eps")
    with UserPreferences() as prefs:
        prefs.eps = eps_new
        assert eps_new == userPref("eps")
    assert eps_old == userPref("eps")


def test_reset() -> None:
    """Test resetting preferences."""
    def change(reset_named: bool = False) -> None:
        """Helper function to change and reset preferences.

        Parameters
        ----------
        reset_named : bool, optional
            Whether to reset a specific preference or all preferences, by default False
        """
        prefs = UserPreferences()
        prefs.eps = 99
        if reset_named:
            prefs.reset("eps")
        else:
            prefs.reset()

    eps_bak = userPref("eps")
    change(False)
    assert eps_bak == userPref("eps")
    change(True)
    assert eps_bak == userPref("eps")
