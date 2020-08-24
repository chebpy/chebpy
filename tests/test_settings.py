# -*- coding: utf-8 -*-

import unittest
import chebpy

def userPref(name):
    from chebpy.core.settings import userPrefs
    return getattr(userPrefs, name)

class Settings(unittest.TestCase):

    def test_update_pref(self):
        eps_new = 1e-3
        eps_old = chebpy.core.settings.userPrefs.eps
        chebpy.core.settings.userPrefs.eps = eps_new
        self.assertEqual(eps_new, userPref('eps'))
        chebpy.core.settings.userPrefs.eps = eps_old  # we *must* change it back

    def test_reset(self):
        def change(reset_named=False):
            chebpy.core.settings.userPrefs.eps = 99
            if reset_named:
                chebpy.core.settings.userPrefs.reset('eps')
            else:
                chebpy.core.settings.userPrefs.reset()
        eps_bak = chebpy.core.settings.defaultPrefs.eps
        change(False)
        self.assertEqual(eps_bak, userPref('eps'))
        change(True)
        self.assertEqual(eps_bak, userPref('eps'))
