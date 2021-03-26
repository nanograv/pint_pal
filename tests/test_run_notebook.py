from astropy import log
from os import makedirs, chdir
from os.path import dirname, join, split, splitext
from datetime import datetime
from multiprocessing import Pool
import traceback
from glob import glob
import pytest
import nbformat
from timing_analysis.notebook_runner import run_notebook

base_dir = dirname(dirname(__file__))

def config_files():
    config_files = (glob(join(base_dir, 'configs/B*.nb.yaml'))
                     + glob(join(base_dir, 'configs/J*.nb.yaml'))
                     + glob(join(base_dir, 'configs/B*.wb.yaml'))
                     + glob(join(base_dir, 'configs/J*.wb.yaml')))
    config_files = sorted(config_files)
    basenames = [splitext(split(filename)[1])[0] for filename in config_files]
    return [pytest.param(filename, id=basename) for filename, basename in zip(config_files, basenames)]

@pytest.fixture(scope='session', autouse=True)
def output_dir():
    output_dir = datetime.strftime(datetime.now(), 'tmp-%Y-%m-%dT%H-%M-%S')
    output_dir = join(base_dir, output_dir)
    makedirs(output_dir, exist_ok=True)
    return output_dir

@pytest.mark.parametrize('config_file', config_files())
def test_run_notebook(config_file, output_dir, suppress_errors=False):
    """
    Run through the functions called in the notebook for each pulsar (excluding plotting).
    This will create a global log called test-run-notebooks.log, and a log file for each pulsar.

    To run for only one pulsar (using J1713+0747 as an example):
        `pytest tests/test_run_notebook.py::test_run_notebook[J1713+0747.nb]`
        or `pytest -k J1713+0747` (selects tests whose name contains "J1713+0747")
    To run for all pulsars in parallel (requires `pytest-xdist`):
        `pytest -n <workers> tests/test_run_notebook.py`
        <workers> is the number of worker processes to launch (e.g. 4 to use 4 CPU threads)
    """
    log.setLevel("INFO")
    global_log = join(output_dir, f'test-run-notebook.log')
    cfg_name = splitext(split(config_file)[1])[0]
    cfg_dir = join(output_dir, cfg_name)
    makedirs(cfg_dir)
    log_file = join(cfg_dir, f'{cfg_name}.log')
    err_file = join(cfg_dir, f'{cfg_name}.traceback')
    output_nb = join(cfg_dir, f'{cfg_name}.ipynb')
    
    transformations = {
        'config': f'"{config_file}"',
        'par_directory': f'"{join(base_dir, "results")}"',
        'use_existing_noise_dir': 'True',
        'log_to_file': 'True',
    }

    try:
        run_notebook(
            join(base_dir, 'nb_templates/process_v0.9.ipynb'),
            output_nb,
            err_file = err_file,
            workdir = cfg_dir,
            transformations = transformations
        )
        with open(global_log, 'a') as f:
            print(f"{cfg_name}: success!", file=f)
    except Exception as err:
        with open(global_log, 'a') as f:
            if hasattr(err, 'ename'):
                print(f"{cfg_name}: failure - {err.ename}", file=f)
            else:
                print(f"{cfg_name}: failure - {err}", file=f)
        raise err
