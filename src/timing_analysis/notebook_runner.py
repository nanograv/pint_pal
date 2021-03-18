import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

import timing_analysis
from timing_analysis.notebook_templater import transform_notebook

base_dir = os.path.dirname(os.path.dirname(timing_analysis.__path__))

def run_notebook(template_nb, output_nb, workdir=base_dir, transformations=None):
    with open(template_nb) as f:
        nb = nbformat.read(f, as_version=4)
    if transformations is not None:
        n_subs = transform_notebook(nb, transformations)
    
    ep = ExecutePreprocessor()
    try:
        ep.preprocess(nb, {'metadata': {'path': workdir}})
    except CellExecutionError as err:
        print(err.traceback)
    finally:
        with open(output_nb, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
