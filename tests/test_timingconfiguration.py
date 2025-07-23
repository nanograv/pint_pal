'''
Unit testing for timingconfiguration.py

These tests are performed on YAML configuration files
'''

import pytest
from pint_pal.timingconfiguration import TimingConfiguration


@pytest.fixture
def tc(scope="class"):
    """ Load a working TimingConfiguration object for testing """
    return TimingConfiguration("configs/J0605+3757.nb.yaml")
@pytest.fixture
def PSR():
    return "J0605+3757"


def test_get_source(tc, PSR):
    print(PSR)
    assert tc.get_source() == PSR


def test_get_model_and_toas(tc, PSR):
    mo, to = tc.get_model_and_toas()
    assert mo.PSR.value == PSR
