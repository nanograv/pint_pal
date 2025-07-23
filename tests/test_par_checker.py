'''
Unit testing for par_checker.py

These tests are performed on modified versions of the
NANOGrav 12.5-year data set (version 3) parameter files
'''


import pytest
from astropy import log
import pint.models as models
import pint_pal.par_checker as pc

log.setLevel("ERROR") # do not show PINT warnings here to avoid clutter

@pytest.fixture
def modelB1855():
    parfile = "par/B1855+09_NANOGrav_12yv4.gls.par"
    return models.get_model(parfile)
@pytest.fixture
def modelJ1024():
    parfile = "par/J1024-0719_NANOGrav_12yv4.gls.par"
    return models.get_model(parfile)
@pytest.fixture(params=['modelB1855', 'modelJ1024'])
def model(request):
    return request.getfixturevalue(request.param)


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_check_name(model):
    """
    Check to see if the pulsar name is in the proper format,
    either J####[+/-]####
    or B####[+/-]##
    """
    assert pc.check_name(model) is None


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_check_spin(model):
    """
    Check to see if the spin and spindown parameters
    are fit for, and also cases for F2
    """
    assert pc.check_spin(model) is None

    
@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_astrometry(model):
    """
    Check to see if astrometric parameters are in ecliptic coordinates
    and parallax is fit for
    """
    assert pc.check_astrometry(model) is None



