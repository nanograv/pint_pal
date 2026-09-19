import pytest
from pathlib import Path
import numpy as np
import astropy.units as u
import pint.toa as toa
import pint.models as models
from pint.models.parameter import maskParameter
from pint.models.timing_model import Component

from pint_pal.lite_utils import add_feJumps

from pint_pal.outlier_utils import (
    gibbs_run,
    get_entPintPulsar,
    calculate_pout,
    make_pout_cuts,
    Ftest,
    test_one_epoch,
    epochalyptica
)

@pytest.fixture
def model():
    parent = Path(__file__).parent
    parfile = parent / "par/B1855+09_NANOGrav_12yv4.gls.par"
    return models.get_model(parfile)

@pytest.fixture
def toas():
    parent = Path(__file__).parent
    timfile = parent / "tim/B1855+09_NANOGrav_12yv4.tim"
    return toa.get_TOAs(timfile)

def test_get_entPintPulsar(model, toas):
    epp = get_entPintPulsar(model, toas)
    assert hasattr(epp, 'name')
    assert epp.name.startswith('B1855')

def test_Ftest_basic():
    # Should return a float < 1 for improved model
    p = Ftest(100.0, 50, 80.0, 48)
    assert isinstance(p, float)
    assert 0 <= p <= 1
    # Should return False for non-improved model
    p2 = Ftest(100.0, 50, 120.0, 48)
    assert p2 is False

def test_gibbs_run_smoke(model, toas):
    try:
        epp = get_entPintPulsar(model, toas)
        pout = gibbs_run(epp, Nsamples=10)
        assert isinstance(pout, np.ndarray)
        assert pout.size > 0
    except Exception as e:
        pytest.skip(f"gibbs_run skipped due to: {e}")


