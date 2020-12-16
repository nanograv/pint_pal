from timing_analysis.lite_utils import check_fit
from timing_analysis.timingconfiguration import TimingConfiguration
from astropy import log
import pint.fitter
from os.path import dirname, join

def test_run_notebook(config_file_path):
    log.setLevel("INFO")

    tc = TimingConfiguration(config_file_path)
    to = tc.get_TOAs()
    mo = tc.get_model()
    
    fo = getattr(pint.fitter,tc.get_fitter())(to,mo)
    
    fo.model.free_params = tc.get_free_params()
    check_fit(fo)

    fo.fit_toas()
    fo.print_summary()

if __name__ == '__main__':
    from glob import glob
    
    base_dir = dirname(dirname(__file__))
    config_files = (glob(join(base_dir, 'configs/B*.nb.yaml'))
                     + glob(join(base_dir, 'configs/J*.nb.yaml')))
    for config_file in sorted(config_files):
        test_run_notebook(config_file)