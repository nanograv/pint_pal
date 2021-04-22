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

def run_notebook(template_nb, config_file, output_nb=None, err_file=None, workdir=None, log_status_to=None, color_err=False, verbose=False, transformations=None):
    """
    Run a template notebook with a set of transformations and save the completed notebook,
    log, and error traceback (if any).
    
    Parameters
    ----------
    template_nb:     Template notebook to use.
    config_file:     Configuration file (YAML).
    output_nb:       Location to write the completed notebook.
    err_file:        Location to write the error traceback log (if necessary).
    workdir:         Directory in which to work (default: current working directory).
    log_status_to:   File-like object (stream) to write status (success/failure) to
                     (default: stdout).
    color_err:       Whether to keep ANSI color codes in the error traceback.
    verbose:         Print a description of replacements made in the template notebook.
    transformations: Transformations to apply to the notebook.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(config_file)))
    if workdir is None:
        workdir = os.getcwd()
    if log_status_to is None:
        log_status_to = sys.stdout
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

def run_in_subdir(template_nb, config_file, output_dir=None, log_status_to=None, verbose=False, transformations=None):
    """
    Given a template notebook and configuration file, create a subdirectory with a name
    based on the configuration file and run the notebook inside it. This function is
    called directly by test_run_notebook.py.
    
    Parameters
    ----------
    template_nb:     Template notebook to use.
    config_file:     Configuration file to use.
    output_dir:      Location where the subdirectory will be created
                     (default: current working directory).
    log_status_to:   File-like object (stream) to write status (success/failure) to
                     (default: stdout).
    verbose:         Print a description of replacements made in the template notebook.
    transformations: Transformations to apply to the notebook.
    """
    if output_dir is None:
        output_dir = os.getcwd()
    if log_status_to is None:
        log_status_to = sys.stdout
    
    cfg_name = os.path.splitext(os.path.split(config_file)[1])[0]
    cfg_dir = os.path.join(output_dir, cfg_name)
    os.makedirs(cfg_dir)
    err_file = os.path.join(cfg_dir, f'{cfg_name}.traceback')
    output_nb = os.path.join(cfg_dir, f'{cfg_name}.ipynb')

    run_notebook(
        template_nb,
        config_file,
        output_nb,
        err_file = err_file,
        workdir = cfg_dir,
        verbose = verbose,
        log_status_to = log_status_to,
    )
