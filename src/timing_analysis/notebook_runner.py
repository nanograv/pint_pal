import os
import nbformat
import textwrap
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

import timing_analysis
from timing_analysis.notebook_templater import transform_notebook

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def run_notebook(template_nb, output_nb, log_file, err_file, workdir=base_dir, transformations=None):
    with open(template_nb) as f:
        nb = nbformat.read(f, as_version=4)
    if transformations is not None:
        n_subs = transform_notebook(nb, transformations)
    
    for cell in nb['cells']:
        if 'tags' in cell['metadata'] and 'logging' in cell['metadata']['tags']:
            cell['source'] = f'''
            log.setLevel("INFO")
            log.log_to_file("{log_file}")
            '''
            cell['source'] = textwrap.dedent(cell['source']).strip()
    
    ep = ExecutePreprocessor()
    try:
        ep.preprocess(nb, {'metadata': {'path': workdir}})
    except CellExecutionError as err:
        with open(err_file, 'w') as f:
            print(err.traceback, file=f)
        raise err
    finally:
        with open(output_nb, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
