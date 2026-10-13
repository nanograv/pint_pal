'''
Unit testing for timingconfiguration.py

These tests are performed on YAML configuration files
'''
import pytest
from pathlib import Path
from pint.fitter import DownhillGLSFitter
from pint_pal.timingconfiguration import TimingConfiguration



@pytest.fixture
def tc():
    """
    Load a working TimingConfiguration object for testing

    To allow for running tests outside of tests/, the
    par and tim directories are overwritten
    """
    parent = Path(__file__).parent
    par_directory = parent / "results/"
    tim_directory = parent / "tim/" 
    configfile = parent / "configs/tctest.nb.yaml"
    print(par_directory, configfile)
    return TimingConfiguration(configfile, tim_directory=tim_directory, par_directory=par_directory)
@pytest.fixture
def PSR():
    return "J0605+3757"


def test_get_source(tc, PSR):
    print(PSR)
    assert tc.get_source() == PSR

def test_get_free_params(tc):
    mo, to = tc.get_model_and_toas()
    fo = DownhillGLSFitter(to, mo)
    assert "ELONG" in tc.get_free_params(fo)
    assert "JUMP1" in tc.get_free_params(fo)
    assert tc.get_free_jumps()[0] == "JUMP -fe Rcvr_800"
    assert tc.get_free_jumps(fitter=fo, convert_to_indices=True)[0] == "JUMP1"

def test_get_model_and_toas(tc, PSR):
    mo, to = tc.get_model_and_toas()
    assert mo.PSR.value == PSR

def test_count_bad_files(tc):
    assert tc.count_bad_files() is None

def test_count_bad_toas(tc):
    assert tc.count_bad_toas() == 1

def test_get_bipm(tc):
    assert tc.get_bipm() == "BIPM2019"

def test_get_ephem(tc):
    assert tc.get_ephem() == "DE440"
    
def test_get_notebook_run_Ftest(tc):
    assert tc.get_notebook_run_Ftest() == True

def test_get_notebook_run_noise_analysis(tc):
    assert tc.get_notebook_run_noise_analysis() == True

def test_get_notebook_check_excision(tc):
    assert tc.get_notebook_check_excision() == True

def test_get_notebook_use_existing_noise_dir(tc):
    assert tc.get_notebook_use_existing_noise_dir() == False

def test_get_notebook_use_toa_pickle(tc):
    assert tc.get_notebook_use_toa_pickle() == False

def test_get_notebook_log_level(tc):
    assert tc.get_notebook_log_level() == "INFO"

def test_get_notebook_log_to_file(tc):
    assert tc.get_notebook_log_to_file() == False

def test_get_fitter(tc):
    assert tc.get_fitter() == "DownhillGLSFitter"

def test_get_toa_type(tc):
    assert tc.get_toa_type() == "NB"

def test_get_niter(tc):
    assert tc.get_niter() == 20

def test_get_mjd_start(tc):
    assert tc.get_mjd_start() is None

def test_get_mjd_end(tc):
    assert tc.get_mjd_end() == 59072.0

def test_get_orphaned_rec(tc):
    assert tc.get_orphaned_rec() is None

def test_get_bad_group(tc):
    assert tc.get_bad_group() is None

def test_get_poor_febes(tc):
    assert tc.get_poor_febes() is None

def test_get_snr_cut(tc):
    assert tc.get_snr_cut() == 8

def test_get_bad_files(tc):
    assert tc.get_bad_files() is None

def test_get_bad_ranges(tc):
    assert tc.get_bad_ranges() == [[0, 60000, "fake"]]

def test_get_bad_toas(tc):
    assert tc.get_bad_toas() == [["fake.ff", 1, 1]]

def test_get_bad_toas_averaged(tc):
    assert tc.get_bad_toas_averaged() is None

def test_get_prob_outlier(tc):
    assert tc.get_prob_outlier() == 0.1

def test_get_noise_dir(tc):
    assert tc.get_noise_dir() is None

def test_get_compare_noise_dir(tc):
    assert tc.get_compare_noise_dir() is None

def test_get_no_corner(tc):
    assert tc.get_no_corner() == True

def test_get_ignore_dmx(tc):
    assert tc.get_ignore_dmx() == False

def test_get_fratio(tc):
    assert tc.get_fratio() == 1.1

def test_get_sw_delay(tc):
    assert tc.get_sw_delay() == 0.1

def test_get_custom_dmx(tc):
    assert tc.get_custom_dmx() == []

def test_get_outlier_burn(tc):
    assert tc.get_outlier_burn() == 1000

def test_get_outlier_samples(tc):
    assert tc.get_outlier_samples() == 20000

def test_get_outlier_method(tc):
    assert tc.get_outlier_method() == "gibbs"

def test_get_orb_phase_range(tc):
    assert tc.get_orb_phase_range() is None

def test_get_check_cleared(tc):
    assert tc.get_check_cleared() == True
