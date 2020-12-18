from timing_analysis.lite_utils import check_fit
from timing_analysis.timingconfiguration import TimingConfiguration
from astropy import log
import pint.fitter

from os.path import dirname, join
from multiprocessing import Pool
import traceback
from glob import glob
import pytest

global_log = 'test-run-notebooks.log'

def config_files():
    base_dir = dirname(dirname(__file__))
    config_files = (glob(join(base_dir, 'configs/B*.nb.yaml'))
                     + glob(join(base_dir, 'configs/J*.nb.yaml')))
    config_file = sorted(config_files)
    return config_files

@pytest.mark.parametrize('config_file', config_files())
def test_run_notebook(config_file, suppress_errors=False):
    """
    Run through the basic set of functions that will be called in the notebook for each pulsar.
    For now, this must be run from the top-level `timing-analysis` directory.
    It will create a global log called test-run-notebooks.log, and a log file for each pulsar.
    """
    log.setLevel("DEBUG")
    cfg_name = '.'.join(config_file.split('/')[-1].split('.')[:-1])
    log_file = f'{cfg_name}.log'

    # clear log file
    with open(log_file, 'w') as f:
        pass

    with log.log_to_file(log_file):
        try:
            tc = TimingConfiguration(config_file)
            mo, to = tc.get_model_and_toas()

            fo = getattr(pint.fitter,tc.get_fitter())(to,mo)

            fo.model.free_params = tc.get_free_params(fo)
            check_fit(fo)

            fo.fit_toas()
            fo.print_summary()

            with open(global_log, 'a') as f:
                print(f"{config_file}: success!", file=f)
        except Exception as e:
            with open(log_file, 'a') as f:
                print(f"Processing config file {config_file} failed with the following error:", file=f)
                print(traceback.format_exc(), file=f)

            with open(global_log, 'a') as f:
                print(f"{config_file}: failure - {repr(e)}", file=f)
            if not suppress_errors:
                raise e

if __name__ == '__main__':
    # clear global log
    with open(global_log, 'w') as f:
        pass

    with Pool(processes=4) as pool:
        results = []
        for config_file in config_files():
            results.append(pool.apply_async(test_run_notebook, (config_file,), {'suppress_errors': True}))
        for result in results:
            result.get()