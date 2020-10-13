'''
Unit testing for timingconfiguration.py

These tests are performed on YAML configuration files
'''


import sys
sys.path.append("../utils/")
import unittest

from timingconfiguration import TimingConfiguration


class TimingConfigurationTests(unittest.TestCase):
    """ timingconfiguration.py testing class """

    def setUp(self):
        """ Load a TimingConfiguration object during setup """
        self.tc = TimingConfiguration("config/goodconfig.yaml")
        self.badtc = TimingConfiguration("config/badconfig.yaml")


    def test_get_source(self):
        """ Check the reading of the source entry """
        self.assertEqual(self.tc.get_source(), "B1855+09")


    def test_get_model(self):
        """ Check the return of a PINT model object """
        self.assertEqual(self.tc.get_model().PSR.value, "B1855+09")
        with self.assertRaises(ValueError):
            self.badtc.get_model()


    def test_get_TOAs(self):
        """ Check the return of a PINT toa object, with various filters """
        pass


try:
    unittest.main(argv=[''], verbosity=2)
except SystemExit: #cleaner output below
    print
