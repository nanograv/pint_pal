'''
Unit testing for par_checker.py

These tests are performed on modified versions of the
NANOGrav 12.5-year data set (version 3) parameter files
'''


import unittest
from astropy import log
import pint.models as model
from timing_analysis.par_checker import *

log.setLevel("ERROR") # do not show PINT warnings here to avoid clutter




class ParCheckerTests(unittest.TestCase):
    """ par_checker.py testing class """

    def test_check_if_fit(self):
        parfile = "par/bad_values.par"
        psr_model = model.get_model(parfile)
        self.assertIsNone(check_if_fit(psr_model, "DMX_0002"))
        with self.assertRaises(ValueError):
            check_if_fit(psr_model, "DMX_0001")

    def test_check_name(self):
        """
        Check to see if the pulsar name is in the proper format,
        either J####[+/-]####
        or B####[+/-]##
        """
        parfile = "par/J1024-0719_NANOGrav_12yv4.gls.par"
        psr_model = model.get_model(parfile)
        self.assertIsNone(check_name(psr_model))

        parfile = "par/bad_values.par"
        psr_model = model.get_model(parfile)
        with self.assertWarns(Warning):
            check_name(psr_model)

    def test_check_spin(self):
        """
        Check to see if the spin and spindown parameters
        are fit for, and also cases for F2
        """

        # Check F2 for J1024-0719
        parfile = "par/J1024-0719_NANOGrav_12yv4.gls.par"
        psr_model = model.get_model(parfile)
        self.assertIsNone(check_spin(psr_model))

        # F2 has been inserted here, which should throw a warning
        parfile = "par/bad_values.par"
        psr_model = model.get_model(parfile)
        with self.assertWarns(Warning):
            check_spin(psr_model)

    def test_check_astrometry(self):
        """
        Check to see if astrometric parameters are in ecliptic coordinates
        and parallax is fit for
        """

        # Parallax has been removed
        parfile = "par/bad_values.par"
        psr_model = model.get_model(parfile)
        with self.assertRaises(ValueError):
            check_astrometry(psr_model)


    def test_check_binary(self):
        pass

    def test_check_jumps(self):
        pass



try:
    unittest.main(argv=[''], verbosity=2)
except SystemExit: #cleaner output below
    print
