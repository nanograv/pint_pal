from astropy import log
from os import mkdir, chdir
from os.path import dirname, join, split, splitext
from multiprocessing import Pool
import traceback
from glob import glob
import pytest
import nbformat

base_dir = dirname(dirname(__file__))
global_log = join(base_dir, 'test-run-notebooks.log')

def config_files():
    config_files = (glob(join(base_dir, 'configs/B*.nb.yaml'))
                     + glob(join(base_dir, 'configs/J*.nb.yaml'))
                     + glob(join(base_dir, 'configs/B*.wb.yaml'))
                     + glob(join(base_dir, 'configs/J*.wb.yaml')))
    config_files = sorted(config_files)
    basenames = [splitext(split(filename)[1])[0] for filename in config_files]
    return [pytest.param(filename, id=basename) for filename, basename in zip(config_files, basenames)]

@pytest.fixture
def notebook_code():
    notebook_location = join(base_dir, 'nb_templates/process_v0.9.ipynb')
    template_notebook = nbformat.read(notebook_location, as_version=4)
    
    code_blocks = []
    for cell in template_notebook['cells']:
        if cell['cell_type'] == 'code':
            lines = cell['source'].split('\n')
            code_lines = []
            for line in lines:
                # Skip full-line comments and IPython magics
                if line.startswith('#') or line.startswith('%'):
                    continue
                code_lines.append(line)
            code_blocks.append('\n'.join(code_lines))
    return code_blocks

@pytest.fixture(scope='session', autouse=True)
def startup():
    try:
        mkdir(join(base_dir, 'logs'))
    except FileExistsError:
        pass
    # clear global log
    with open(global_log, 'w') as f:
        pass

@pytest.mark.parametrize('config_file', config_files())
def test_run_notebook(notebook_code, config_file, tmpdir, suppress_errors=False):
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
    cfg_name = splitext(split(config_file)[1])[0]
    log_file = join(join(base_dir, 'logs'), f'{cfg_name}.log')
    err_file = join(join(base_dir, 'logs'), f'{cfg_name}.traceback')

    # clear log file
    with open(log_file, 'w') as f:
        pass

    chdir(tmpdir)
    with log.log_to_file(log_file):
        try:
            # Execute notebook contents
            for code_block in notebook_code:
                # Fill in the name of the config file
                code_block = code_block.replace(
                    '"configs/[filename].yaml"',
                    f'"{config_file}", par_directory="{join(base_dir, "results")}"'
                )
                exec(code_block)

            with open(global_log, 'a') as f:
                print(f"{cfg_name}: success!", file=f)
        except Exception as e:
            with open(err_file, 'w') as f:
                print(f"Processing config file {config_file} failed with the following error:\n", file=f)
                print(traceback.format_exc(), file=f)
                print(f"While processing the following code block:\n\n{code_block}", file=f)

            with open(global_log, 'a') as f:
                print(f"{cfg_name}: failure - {repr(e)}", file=f)
            if not suppress_errors:
                raise e

if __name__ == '__main__':
    # clear global log
    with open(global_log, 'w') as f:
        pass

    with Pool(processes=4) as pool:
        code = notebook_code.__wrapped__()
        results = []
        for config_file in config_files():
            results.append(pool.apply_async(test_run_notebook, (code, config_file), {'suppress_errors': True}))
        for result in results:
            result.get()
