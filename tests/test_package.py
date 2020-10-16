from __future__ import division, print_function, unicode_literals, absolute_import
import unittest
import sys
sys.path.append("../pyNSID/")

class TestImport(unittest.TestCase):

    def test_basic(self):
        import pyNSID as nsid
        print(nsid.__version__)
        self.assertTrue(True)

