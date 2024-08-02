import sys
sys.path.append("../utils/")

import unittest
import numpy as np
import astropy.units as u
import pint.toa as toa
import pint.models as models
from pint.models.parameter import maskParameter
from pint.models.timing_model import Component
from pint_pal.timingconfiguration import TimingConfiguration
from pint_pal.lite_utils import *

class LiteUtilsTests(unittest.TestCase):
    """lite_utils.py testing class"""
    
    def test_add_feJumps(self):
        """Check N-1 jumps added if N receivers present"""
        m = models.get_model("par/J2022+2534.basic.par")
        t = toa.get_TOAs("tim/J2022+2534_15y_L-S_nb.tim")
    
        # Assert model starts with no jumps
        assert not any('JUMP' in p for p in m.params)
    
        receivers = set(t.get_flag_value('fe')[0])
        add_feJumps(m,list(receivers))
        
        # Assert proper number of fe jumps have been added (Nrec-1)
        all_jumps = m.components['PhaseJump'].get_jump_param_objects()
        jump_rcvrs = [x.key_value[0] for x in all_jumps if x.key == '-fe'] 
        assert len(jump_rcvrs) == len(receivers)-1
    
    def test_convert_pint_to_tempo_timfile(self):
        """ Check the output file of the convert_pint_to_tempo_timfile method """
        with open("tim/B1855+09_tempo.tim", "r") as text_file:
            ground_truth = repr(text_file.read())
        convert_pint_to_tempo_timfile(tim_path="tim/B1855+09_NANOGrav_12yv4.tim",op_path="tim",psr_name="B1855+09")
        with open("tim/B1855+09.tim", "r") as text_file:
            current_output = repr(text_file.read())
        self.assertEqual(current_output,ground_truth)
    
    def test_convert_pint_to_tempo_parfile(self):
        """ Check the output file of the convert_pint_to_tempo_parfile method """
        with open("par/B1855+09_tempo.par", "r") as text_file:
            ground_truth = repr(text_file.read())
        convert_pint_to_tempo_parfile(path_to_par="par/B1855+09_NANOGrav_12yv4.gls.par",op_path="par")
        with open("par/B1855+09.par", "r") as text_file:
            current_output = repr(text_file.read())
        self.assertEqual(current_output,ground_truth)

    def test_write_tim(self):
        """ Check the output file of the write_tim method """
        config = "configs/luconfig.yaml"  
        par_directory = None  
        tim_directory = None  
        tc = TimingConfiguration(config, par_directory=par_directory, tim_directory=tim_directory)
        mo,to = tc.get_model_and_toas(apply_initial_cuts=False,usepickle=False)
        fo = tc.construct_fitter(to,mo)
        fo.model.free_params = tc.get_free_params(fo)
        with open("tim/J2022+2534_PINT.tim", "r") as text_file:
            ground_truth = repr('\n'.join(text_file.readlines()[8:]))
        write_tim(outfile="tim/J2022+2534_test.tim",fitter=fo)
        with open("tim/J2022+2534_test.tim", "r") as text_file:
            current_output = repr('\n'.join(text_file.readlines()[8:]))
        self.assertEqual(current_output,ground_truth)

    def test_write_par(self):
        """ Check the output file of the write_par method """
        # Issue with par files not generating the same way each time
        config = "configs/luconfig.yaml"  
        par_directory = None  
        tim_directory = None  
        tc = TimingConfiguration(config, par_directory=par_directory, tim_directory=tim_directory)
        mo,to = tc.get_model_and_toas(apply_initial_cuts=False,usepickle=False)
        fo = tc.construct_fitter(to,mo)
        fo.model.free_params = tc.get_free_params(fo)
        with open("par/J2022+2534_PINT.par", "r") as text_file:
            ground_truth = repr('\n'.join(text_file.readlines()[8:27]))
        write_par(outfile="par/J2022+2534_test.par",fitter=fo)
        with open("par/J2022+2534_test.par", "r") as text_file:
            current_output = repr('\n'.join(text_file.readlines()[8:27]))
        self.assertEqual(current_output,ground_truth)


    def test_center_epochs(self):
        """ Checks the accuracy of the center_epochs method """
        config = "configs/luconfig.yaml"  
        par_directory = None  
        tim_directory = None  
        tc = TimingConfiguration(config, par_directory=par_directory, tim_directory=tim_directory)
        mo,to = tc.get_model_and_toas(apply_initial_cuts=False,usepickle=False)
        center_epochs(mo,to)
        epochs = str(mo.DMEPOCH.value) + " " + str(mo.POSEPOCH.value)
        self.assertEqual(epochs, "58439.0 58439.0")

    def test_large_residuals(self):
        """  """
        config = "configs/luconfig.yaml"  
        par_directory = None  
        tim_directory = None  
        tc = TimingConfiguration(config, par_directory=par_directory, tim_directory=tim_directory)
        mo,to = tc.get_model_and_toas(apply_initial_cuts=False,usepickle=False)
        fo = tc.construct_fitter(to,mo)
        with open("results/lite_utils_large_residuals.txt", "r") as text_file:
            ground_truth = text_file.read()
        good_toas = ' '.join(large_residuals(fo,100).get_flag_value('name')[:][0])
        self.assertEqual(good_toas, ground_truth)

    def test_compare_model(self):
        """ Check the output of the compare_model method """
        with open("results/lite_utils_compare_model.txt", "r") as text_file:
            ground_truth = text_file.read()
        config = "configs/luconfig.yaml"  
        par_directory = None  
        tim_directory = None  
        tc = TimingConfiguration(config, par_directory=par_directory, tim_directory=tim_directory)
        mo,to = tc.get_model_and_toas(apply_initial_cuts=False,usepickle=False)
        fo = tc.construct_fitter(to,mo)
        self.assertEqual(repr(compare_models(fo,"par/J2022+2534.compare.par",verbosity="min")), ground_truth)

    # def test_check_fit(self):
    #     """  """
    #     self.assertEqual()

    # def test_remove_noise(self):
    #     """  """
    #     self.assertEqual()

    # def test_get_config_info(self):
    #     """  """
    #     self.assertEqual()

    # def test_cut_summary(self):
    #     """  """
    #     self.assertEqual()

    # def test_check_toa_version(self):
    #     """  """
    #     self.assertEqual()

    # def test_check_convergence(self):
    #     """ Checks the return of the check_convergence method """
    #     # no return values to check, perphaps propose a change to original method
    #     ground_truth = repr(['chi-squared decreased during fit by 0.0', 'Fitter has probably converged'])
    #     config = "configs/luconfig.yaml"  
    #     tc = TimingConfiguration(config, par_directory=None, tim_directory=None)
    #     mo,to = tc.get_model_and_toas(apply_initial_cuts=False,usepickle=False)
    #     fo = tc.construct_fitter(to,mo)
    #     self.assertEqual(repr(check_convergence(fo)),ground_truth)

try:
    unittest.main(argv=[''], verbosity=2)
except SystemExit: #cleaner output below
    print()
