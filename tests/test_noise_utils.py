import pytest
from pathlib import Path
import pint.models as models
import pint_pal.noise_utils as nu


@pytest.fixture
def model():
    parent = Path(__file__).parent
    parfile = parent / "par/B1855+09_NANOGrav_12yv4.gls.par"
    return models.get_model(parfile)



@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_format_chain_dir(model):
    assert nu.format_chain_dir("/path/to/chains/B1855+09_nb/", model) == "/path/to/chains/B1855+09_nb/"
    assert nu.format_chain_dir("/path/to/chains/", model) == "/path/to/chains/B1855+09_nb/"
    assert nu.format_chain_dir("/path/to/chains/", model, using_wideband=True) == "/path/to/chains/B1855+09_wb/"
