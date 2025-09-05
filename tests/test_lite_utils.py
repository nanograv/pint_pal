import pytest
from pathlib import Path
import numpy as np
import astropy.units as u
import pint.toa as toa
import pint.models as models
from pint.models.parameter import maskParameter
from pint.models.timing_model import Component

from pint_pal.lite_utils import add_feJumps

@pytest.fixture
def model():
    parent = Path(__file__).parent
    parfile = parent / "par/J2022+2534.basic.par"
    return models.get_model(parfile)
@pytest.fixture
def toas():
    parent = Path(__file__).parent
    timfile = parent / "tim/J2022+2534_15y_L-S_nb.tim"
    return toa.get_TOAs(timfile)


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_add_feJump(model, toas):
    """Check N-1 jumps added if N receivers present"""
    assert not any('JUMP' in p for p in model.params)

    receivers = set(toas.get_flag_value('fe')[0])
    add_feJumps(model, list(receivers))
        
    # Assert proper number of fe jumps have been added (Nrec-1)
    all_jumps = model.components['PhaseJump'].get_jump_param_objects()
    jump_rcvrs = [x.key_value[0] for x in all_jumps if x.key == '-fe'] 
    assert len(jump_rcvrs) == len(receivers)-1



    
