import sys
sys.path.append("../utils/")

import numpy as np
import astropy.units as u
import pint.toa as toa
import pint.models as models
from pint.models.parameter import maskParameter
from pint.models.timing_model import Component

from lite_utils import add_feJumps

def test_add_feJumps():
    """Check N-1 jumps added if N receivers present"""
    m = models.get_model("par/J2022+2534.basic.par")
    t = toa.get_TOAs("tim/J2022+2534_15y_L-S_nb.tim")
    
    # Assert model starts with no jumps
    
    receivers = set(t.get_flag_value('fe')[0])
    add_feJumps(m,list(receivers))
    print(m)
    
test_add_feJumps()
    
    
    