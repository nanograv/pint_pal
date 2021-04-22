import os
import sys
import nbformat
import textwrap
import re
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
import multiprocessing
from glob import glob

import timing_analysis
from timing_analysis.notebook_templater import transform_notebook

ansi_color = re.compile(r'\x1b\[([0-9]{1,3};)*[0-9]{1,3}m')

def run_notebook(template_nb, config_file, output_nb=None, err_file=None, workdir=os.getcwd(), log_status_to=sys.stdout, color_err=False, verbose=False, transformations=None):
    """
    Run a template notebook with a set of transformations and save the completed notebook,
    log, and error traceback (if any).
    
    Parameters
    ----------
    template_nb:     Template notebook to use.
    config_file:     Configuration file (YAML).
    output_nb:       Location to write the completed notebook.
    err_file:        Location to write the error traceback log (if necessary).
    workdir:         Directory in which to work.
    log_status_to:   File-like object (stream) to write status (success/failure) to.
    color_err:       Whether to keep ANSI color codes in the error traceback.
    verbose:         Print a description of replacements made in the template notebook.
    transformations: Transformations to apply to the notebook.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(config_file)))
    default_transformations = {
        'config': f'"{config_file}"',
        'par_directory': f'"{os.path.join(base_dir, "results")}"',
        'write_prenoise': "True",
        'write_results': "True",
        'use_existing_noise_dir': "True",
        'log_to_file': "True",
    }
    if transformations is None:
        transformations = default_transformations
    else:
        transformations = {**default_transformations, **transformations}

    with open(template_nb) as f:
        nb = nbformat.read(f, as_version=4)

    if transformations is not None:
        n_subs = transform_notebook(nb, transformations, verbose=verbose)
    cfg_name = os.path.splitext(os.path.split(config_file)[1])[0]
    
    ep = ExecutePreprocessor(timeout=0)
    try:
        ep.preprocess(nb, {'metadata': {'path': workdir}})
    except CellExecutionError as err:
        if err_file is not None:
            with open(err_file, 'w') as f:
                if not color_err:
                    traceback = re.sub(ansi_color, '', err.traceback)
                print(traceback, file=f)
        if log_status_to is not None:
            if hasattr(err, 'ename'):
                print(f"{cfg_name}: failure - {err.ename}", file=log_status_to)
            else:
                print(f"{cfg_name}: failure - {err}", file=log_status_to)
        raise err
    finally:
        if output_nb is not None:
            with open(output_nb, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
    if log_status_to is not None:
        print(f"{cfg_name}: success!", file=log_status_to)

def run_in_subdir(template_nb, config_file, output_dir=os.getcwd(), global_log='/dev/stdout', verbose=False, transformations=None):
    """
    Given a template notebook and configuration file, create a subdirectory with a name
    based on the configuration file and run the notebook inside it. Used by the run_batch()
    function and test_run_notebook.py.
    
    Parameters
    ----------
    template_nb:     Template notebook to use.
    config_file:     Configuration file to use.
    output_dir:      Location where the subdirectory will be created.
    global_log:      File where a line indicating success or failure will be written.
    verbose:         Print a description of replacements made in the template notebook.
    transformations: Transformations to apply to the notebook.
    """
    cfg_name = os.path.splitext(os.path.split(config_file)[1])[0]
    cfg_dir = os.path.join(output_dir, cfg_name)
    os.makedirs(cfg_dir)
    err_file = os.path.join(cfg_dir, f'{cfg_name}.traceback')
    output_nb = os.path.join(cfg_dir, f'{cfg_name}.ipynb')

    with open(global_log, 'a') as f:
        run_notebook(
            template_nb,
            config_file,
            output_nb,
            err_file = err_file,
            workdir = cfg_dir,
            verbose = verbose,
            log_status_to = f,
        )

def run_batch(template_nb, config_glob=None, config_files=None, processes=4, output_dir=os.getcwd(), verbose=False, transformations=None):
    """
    Run a template notebook for each of several configuration files, storing the results of
    each in a separate subdirectory.
    
    Parameters
    ----------
    template_nb:     Template notebook to use.
    config_glob:     Glob expression matching YAML configuration files to use.
    config_files:    List of configuration files to use. Either this or config_glob
                     must be specified. If both are, the former will be preferred.
    processes:       Number of worker processes to launch.
    output_dir:      Directory for output files. A subdirectory will be created
                     for each configuration file run.
    verbose:         Print a description of replacements made in the template notebook.
    transformations: Transformations to apply to the notebook.
    """
    if config_glob is not None:
        config_files = sorted(glob(config_glob))
    if config_files is None:
        raise ValueError("Please specify at least one configuration file.")

    os.makedirs(output_dir, exist_ok=True)
    global_log = os.path.join(output_dir, 'batch-status.log')

    results = []
    with multiprocessing.Pool(processes) as p:
        for config_file in config_files:
            args = (template_nb, config_file, output_dir, global_log)
            kwargs = {'verbose': verbose, 'transformations': transformations}
            results.append(p.apply_async(run_in_subdir, args, kwargs))
        for result in results:
            result.get()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a template notebook with a set of "
                                     "transformations and save the completed notebook, log, "
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
    parser.add_argument("config-file", help="Configuration file (YAML)")
    parser.add_argument("output-nb", help="Output file to write")
    parser.add_argument("-e", "--err-file", default=None, help="Error traceback log file")
    parser.add_argument("-w", "--workdir", default=os.getcwd(), help="Directory in which to work")
    args = parser.parse_args()

    transformations = {k: v for (k,v) in args.set_variable}
    run_notebook(
        args.template_nb,
        args.config_file,
        args.output_nb,
        err_file = args.err_file,
        workdir = args.workdir,
        verbose = args.verbose,
        transformations = transformations,
    )
