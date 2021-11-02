import unittest

from chebpy import UserPreferences
from chebpy.core.settings import _preferences


def userPref(name):
    """Get the current preference value for name.
    This is how chebpy core modules access these."""
    return getattr(_preferences, name)


class Settings(unittest.TestCase):
    def test_update_pref(self):
        eps_new = 1e-3
        eps_old = userPref("eps")
        with UserPreferences() as prefs:
            prefs.eps = eps_new
            self.assertEqual(eps_new, userPref("eps"))
        self.assertEqual(eps_old, userPref("eps"))

    def test_reset(self):
        def change(reset_named=False):
            prefs = UserPreferences()
            prefs.eps = 99
            if reset_named:
                prefs.reset("eps")
            else:
                prefs.reset()

        eps_bak = userPref("eps")
        change(False)
        self.assertEqual(eps_bak, userPref("eps"))
        change(True)
        self.assertEqual(eps_bak, userPref("eps"))
