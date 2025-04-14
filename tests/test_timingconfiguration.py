'''
Unit testing for timingconfiguration.py

These tests are performed on YAML configuration files
'''


import unittest
from pint_pal.timingconfiguration import TimingConfiguration


class TimingConfigurationTests(unittest.TestCase):
    """ timingconfiguration.py testing class """

    def setUp(self):
        """ Load a TimingConfiguration object during setup """
        self.tc = TimingConfiguration("configs/tcconfig.yaml")
        self.mo,self.to = self.tc.get_model_and_toas(apply_initial_cuts=False,usepickle=False)


    def test_get_source(self):
        """ Check the reading of the source entry """
        self.assertEqual(self.tc.get_source(), "B1855+09")


    def test_get_model(self):
        """ Check the return of a PINT model object """
        self.assertEqual(self.mo.PSR.value, "B1855+09")


    def test_get_TOAs(self):
        """ Check the return of a PINT toa object """
        summary = "arecibo TOAs (6464):\n  Min freq:      422.187 MHz\n  Max freq:      1762.048 MHz\n  Min error:     0.05 us\n  Max error:     25 us\n  Median error:  1.11 us\n"
        self.assertEqual(self.to.get_summary()[192:], summary)

    def test_get_summary(self):
        """ Check the retun of the timing configuration summary method """
        summary = "Pulsar: B1855+09  \nMode: NB  \nPar file: `par/B1855+09_NANOGrav_12yv4.gls.par`  \nTim files:\n\n- `B1855+09_NANOGrav_12yv4.tim`\n\n"
        self.assertEqual(self.tc.get_summary(), summary)
    
    def test_manual_cuts(self):
        """ Check the return of a PINT toa object, with toa exclusion """
        # Floating Point Error possiblity?
        summary = "Number of TOAs:  6441\nNumber of commands:  2\nNumber of observatories: 1 ['arecibo']\nMJD span:  53358.727 to 57915.275\nDate span: 2004-12-19 17:27:32.961413065 to 2017-06-11 06:35:52.512117675\narecibo TOAs (6441):\n  Min freq:      422.187 MHz\n  Max freq:      1762.048 MHz\n  Min error:     0.05 us\n  Max error:     25 us\n  Median error:  1.1 us\n"
        self.tc.manual_cuts(self.to)
        self.assertEqual(self.to.get_summary(), summary)
    
    def test_construct_fitter(self):
        """ Check the return of the construct fitter method """
        fo = self.tc.construct_fitter(self.to,self.mo)
        summary = "Chisq = 7215.997 for 6320 d.o.f. for reduced Chisq of 1.142"
        self.assertEqual(repr(fo.get_summary())[174:233], summary)
    
    def test_get_free_params(self):
        """ Check the accuracy of the get_free_params method """
        fo = self.tc.construct_fitter(self.to,self.mo)
        fo.model.free_params = self.tc.get_free_params(fo)
        params = "['PX', 'ELONG', 'ELAT', 'PMELONG', 'PMELAT', 'F0', 'F1']"
        self.assertEqual(str(fo.model.free_params), params)
    
    def test_get_niter(self):
        """ Check to accuracy of the get_niter method """
        self.assertEqual(self.tc.get_niter(),20)



try:
    unittest.main(argv=[''], verbosity=2)
except SystemExit: #cleaner output below
    print()
