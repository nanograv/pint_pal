'''
Unit testing for noise_utils.py
'''



import pytest
from pathlib import Path
import pint.models as models
# temporary
import sys
sys.path.insert(0, "/home/michael/Research/Programs/pint_pal/src/pint_pal/")
sys.path.append("/home/michael/Research/Programs/pint_pal/src/pint_pal/")
import pint_pal.noise_utils as nu



@pytest.fixture
def model():
    parfile = Path(__file__).parent / "par/B1855+09_NANOGrav_12yv4.gls.par"
    return models.get_model(parfile)



def test_find_chain_dir(model):
    root_dir = "/path/to/chains/B1855+09_nb/"
    
    assert nu.find_chain_dir(root_dir, model)
