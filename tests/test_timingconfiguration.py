'''
Unit testing for timingconfiguration.py

These tests are performed on YAML configuration files
'''

import pytest
from pathlib import Path
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
    configfile = parent / "configs/J0605+3757.nb.yaml"
    print(par_directory, configfile)
    return TimingConfiguration(configfile, tim_directory=tim_directory, par_directory=par_directory)
@pytest.fixture
def PSR():
    return "J0605+3757"


def test_get_source(tc, PSR):
    print(PSR)
    assert tc.get_source() == PSR


def test_get_model_and_toas(tc, PSR):
    mo, to = tc.get_model_and_toas()
    assert mo.PSR.value == PSR
