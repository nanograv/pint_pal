from os.path import dirname, join, split, splitext, basename
from os import makedirs
import shutil
from datetime import datetime
import traceback
from glob import glob
import pytest
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

@pytest.fixture(scope='session')
def output_dir():
    output_dir = datetime.strftime(datetime.now(), 'tmp-%Y-%m-%dT%H-%M-%S')
    output_dir = join(base_dir, output_dir)
    makedirs(output_dir, exist_ok=True)
    return output_dir

@pytest.mark.parametrize('config_file', config_files())
def test_run_notebook_quick(config_file, tmp_path, output_dir):
    """
    Run through the functions called in the notebook for each pulsar (excluding plotting).

    To run for only one pulsar (using J1713+0747 as an example):
        `pytest tests/test_run_notebook_quick.py::test_run_notebook_quick[J1713+0747.nb]`
        or `pytest -k J1713+0747` (selects tests whose name contains "J1713+0747")
    To run for all pulsars in parallel (requires `pytest-xdist`):
        `pytest -n <workers> tests/test_run_notebook_quick.py`
        <workers> is the number of worker processes to launch (e.g. 4 to use 4 CPU threads)
    """
    transformations = {
        'config': f'"{config_file}"',
        'par_directory': f'"{join(base_dir, "results")}"',
        'use_existing_noise_dir': 'True',
        'log_to_file': 'False',
        'run_Ftest': 'False',
    }

    try:
        run_notebook(
            join(base_dir, 'nb_templates/process_v0.9.ipynb'),
            output_nb=None,
            workdir=tmp_path,
            transformations=transformations,
        )
    finally:
        for f in glob(join(tmp_path, "*summary*.pdf")):
            shutil.copy(f, join(output_dir, basename(f)))
