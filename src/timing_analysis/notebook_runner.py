import os
import nbformat
import textwrap
import re
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

import timing_analysis
from timing_analysis.notebook_templater import transform_notebook

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
ansi_color = re.compile(r'\x1b\[([0-9]{1,3};)*[0-9]{1,3}m')

def run_notebook(template_nb, output_nb, err_file=None, workdir=base_dir, color_err=False, transformations=None):
    with open(template_nb) as f:
        nb = nbformat.read(f, as_version=4)
    if transformations is not None:
        n_subs = transform_notebook(nb, transformations)
    
    ep = ExecutePreprocessor(timeout=0)
    try:
        ep.preprocess(nb, {'metadata': {'path': workdir}})
    except CellExecutionError as err:
        if err_file is not None:
            with open(err_file, 'w') as f:
                if not color_err:
                    traceback = re.sub(ansi_color, '', err.traceback)
                print(traceback, file=f)
        raise err
    finally:
        if output_nb is not None:
            with open(output_nb, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
