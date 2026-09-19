import pytest
from pathlib import Path
import numpy as np
import astropy.units as u
import pint.toa as toa
import pint.models as models

from pint_pal.dmx_utils import get_dmx_ranges, check_frequency_ratio, apply_frequency_ratio_cuts

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

def test_get_dmx_ranges(toas):
    ranges = get_dmx_ranges(toas, bin_width=1.0)
    assert isinstance(ranges, list)
    assert all(isinstance(r, tuple) and len(r) == 2 for r in ranges)
    assert all(r[0] < r[1] for r in ranges)

def test_check_frequency_ratio(toas):
    ranges = get_dmx_ranges(toas, bin_width=1.0)
    toamask, rangemask = check_frequency_ratio(toas, ranges, frequency_ratio=1.1)
    assert isinstance(toamask, np.ndarray)
    assert isinstance(rangemask, np.ndarray)
    assert toamask.dtype == int
    assert rangemask.dtype == bool

def test_apply_frequency_ratio_cuts(toas):
    ranges = get_dmx_ranges(toas, bin_width=1.0)
    toas2 = apply_frequency_ratio_cuts(toas, ranges, frequency_ratio=1.1)
    # Should return a TOAs object with possible cuts applied
    assert hasattr(toas2, 'table')








    
