# -*- coding: utf-8 -*-
"""
Entry-points for pyfun unit-tests
"""

import os
import nose

core_args = [
    "DUMMY",                   # dummy entry (the first)
    '--verbosity', '2',        # verbosity level
    '-s',                      # print stdout for each test
]

# user-specfic absolute location of this file
thisfile = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
pyfunloc = "/".join(thisfile.split("/")[:-1])

def test_chebtech():
    """Unit-tests for pyfun/chebtech"""
    args = core_args[:] + [pyfunloc + u"/tests/test_chebtech.py"]
    nose.run(argv=args)

def test_utilities():
    """Unit-tests for pyfun/utilities"""
    args = core_args[:] + [pyfunloc + u"/tests/test_utilities.py"]
    nose.run(argv=args)

def test_all():
    """Run all unit-tests"""
    args = core_args[:] + [pyfunloc + u"/tests"]
    nose.run(argv=args)

if __name__ == "__main__":
    test_chebtech()
#    test_utilities()
#    test_all()