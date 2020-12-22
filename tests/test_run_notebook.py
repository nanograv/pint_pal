from astropy import log
from os.path import dirname, join
from multiprocessing import Pool
import traceback
from glob import glob
import pytest
import nbformat

global_log = 'test-run-notebooks.log'

def config_files():
    base_dir = dirname(dirname(__file__))
    config_files = (glob(join(base_dir, 'configs/B*.nb.yaml'))
                     + glob(join(base_dir, 'configs/J*.nb.yaml')))
    config_file = sorted(config_files)
    return config_files

@pytest.fixture
def notebook_code():
    template_notebook = nbformat.read('nb_templates/draft_process.ipynb', as_version=4)
    
    code_blocks = []
    for cell in template_notebook['cells']:
        if cell['cell_type'] == 'code':
            lines = cell['source'].split('\n')
            code_lines = []
            for line in lines:
                # Skip full-line comments and IPython magics
                if line.startswith('#') or line.startswith('%'):
                    continue
                # Skip certain kinds of lines that aren't useful here
                if ('log.setLevel' in line
                    or 'quantity_support' in line
                    or 'plot_residuals' in line):
                    continue
                code_lines.append(line)
            code_blocks.append('\n'.join(code_lines))
    return code_blocks

@pytest.mark.parametrize('config_file', config_files())
def test_run_notebook(notebook_code, config_file, suppress_errors=False):
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
            # Execute notebook contents
            for code_block in notebook_code:
                # Fill in the name of the config file
                code_block = code_block.replace('[filename]', cfg_name)
                print(code_block)
                exec(code_block)

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
        code = notebook_code.__wrapped__()
        results = []
        for config_file in config_files():
            results.append(pool.apply_async(test_run_notebook, (code, config_file), {'suppress_errors': True}))
        for result in results:
            result.get()