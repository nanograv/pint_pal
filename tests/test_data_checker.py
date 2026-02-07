'''
Unit testing for par_checker.py

These tests are performed on modified versions of the
NANOGrav 12.5-year data set (version 3) parameter files
'''


import pytest
from pathlib import Path
from astropy import log
import pint.models as models
import pint.toa as toa
import data_checker as dc

log.setLevel("ERROR") # do not show PINT warnings here to avoid clutter

@pytest.fixture
def modelB1855():
    parent = Path(__file__).parent
    parfile = parent / "par/B1855+09_NANOGrav_12yv4.gls.par"
    return models.get_model(parfile)
@pytest.fixture
def modelJ1024():
    parent = Path(__file__).parent
    parfile = parent / "par/J1024-0719_NANOGrav_12yv4.gls.par"
    return models.get_model(parfile)
@pytest.fixture(params=['modelB1855', 'modelJ1024'])
def model(request):
    return request.getfixturevalue(request.param)

@pytest.fixture
def toasB1855():
    parent = Path(__file__).parent
    timfile = parent / "tim/B1855+09_NANOGrav_12yv4.tim"
    return toa.get_TOAs(timfile)
@pytest.fixture
def toasJ1024():
    parent = Path(__file__).parent
    timfile = parent / "tim/J1024-0719_NANOGrav_12yv4.tim"
    return toa.get_TOAs(timfile)
@pytest.fixture(params=['toasB1855', 'toasJ1024'])
def toas(request):
    return request.getfixturevalue(request.param)



@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_check_name(model):
    """
    Check to see if the pulsar name is in the proper format,
    either J####[+/-]####
    or B####[+/-]##
    """
    namechecker = dc.NameChecker(model)
    assert namechecker.check()
    # this is circular, but we know the names of the par files above
    filename = f"{model.PSR.value}_NANOGrav_12yv4.gls.par"
    assert namechecker.check_against_filename(filename)

@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_check_parameters(model):
    """
    Check to see if the spin and spindown, astrometric, and binary parameters
    are fit for, and also case for F2

    As these are older par files, we write the defaults supplied to the check
    """

    required = ["F0", "F1", "PX", "ELONG", "ELAT", "PMELONG", "PMELAT"]
    excluded = []
    if model.PSR.value == "J1024-0719":
        required.append("F2")
        excluded = []

    parchecker = dc.ParChecker(model)
    # Rewrite this for the older data sets checked against
    required_value = {
        "PLANET_SHAPIRO": False,
        "EPHEM": "DE436",
        "CLOCK": "TT(BIPM2017)",
        "CORRECT_TROPOSPHERE": False
    }
    assert parchecker.check(required=required, excluded=excluded, required_value=required_value)


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_check_epoch_centering(model):
    """
    Check to see if the EPOCH parameters are appropriately centered.
    """
    epochchecker = dc.EpochChecker(model)
    assert epochchecker.check()

@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_check_jumps(model, toas):
    """
    Check to see if the proper number of JUMPs are written in.
    """
    if model.PSR.value in str(toas.filename):
        jumpchecker = dc.JumpChecker(model, toas)
        assert jumpchecker.check()
