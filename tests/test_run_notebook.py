from timing_analysis.lite_utils import check_fit
from timing_analysis.timingconfiguration import TimingConfiguration
from astropy import log
import pint.fitter

from os.path import dirname, join
from multiprocessing import Pool
import traceback

def test_run_notebook(config_file, log_file):
    log.setLevel("DEBUG")

    with log.log_to_file(log_file):
        try:
            tc = TimingConfiguration(config_file)
            to = tc.get_TOAs()
            mo = tc.get_model()

            fo = getattr(pint.fitter,tc.get_fitter())(to,mo)

            fo.model.free_params = tc.get_free_params()
            check_fit(fo)

            fo.fit_toas()
            fo.print_summary()
        except:
            with open(log_file, 'a') as f:
                print(f"Processing config file {config_file} failed with the following error:", file=f)
                print(traceback.format_exc(), file=f)

if __name__ == '__main__':
    from glob import glob
    
    base_dir = dirname(dirname(__file__))
    config_files = (glob(join(base_dir, 'configs/B*.nb.yaml'))
                     + glob(join(base_dir, 'configs/J*.nb.yaml')))
    with Pool(processes=4) as pool:
        results = []
        for config_file in sorted(config_files):
            psrname = config_file.split('/')[-1].split('.')[0]
            log_file = f'{psrname}.log'
            results.append(pool.apply_async(test_run_notebook, (config_file, log_file)))
        for result in results:
            result.get()