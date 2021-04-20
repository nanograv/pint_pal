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
        with open(output_nb, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a Jupyter notebook with a set of "
                                     "substitutions and save the completed notebook, log, "
                                     "and error traceback (if any).")
    def parse(s):
        key, value = s.split("=")
        return key.strip(), value.strip()
    parser.add_argument("-s", "--set-variable", type=parse, action="append",
                        help="The variable to replace and its value, separated by an equal sign. "
                        "May require quotes to protect it from your shell; the value is "
                        "python source code (so strings need quotes in addition to whatever "
                        "the shell requires. Can be specified multiple times to have multiple "
                        "substitutions.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Describe all substitutions.")
    parser.add_argument("template-nb", help="Input file to transform")
    parser.add_argument("output-nb", help="Output file to write")
    parser.add_argument("-e", "--err-file", default=None, help="Error traceback log file")
    parser.add_argument("-w", "--workdir", default=os.getcwd(), help="Directory in which to work")
    args = parser.parse_args()

    transformations = {k: v for (k,v) in args.set_variable}
    run_notebook(
        args.template_nb,
        args.output_nb,
        err_file = args.err_file,
        wordir = args.workdir,
        transformations = transformations,
    )
