import sys
sys.path.append("../utils/")

import unittest
import numpy as np
import astropy.units as u
import pint.toa as toa
import pint.models as models
from pint.models.parameter import maskParameter
from pint.models.timing_model import Component

from timing_analysis.lite_utils import add_feJumps

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
        jump_rcvrs = [x.key_value[0] for x in all_jumps if x.key == 'fe'] 
        assert len(jump_rcvrs) == len(receivers)-1
    
if __name__ == '__main__':
    unittest.main()
    
    
    
