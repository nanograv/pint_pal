import os
import sys
import nbformat
import textwrap
import re
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
import multiprocessing
from glob import glob
from ruamel.yaml import YAML
yaml = YAML(typ='safe')

import pint_pal
import pint_pal.config
from pint_pal.notebook_templater import transform_notebook

ansi_color = re.compile(r'\x1b\[([0-9]{1,3};)*[0-9]{1,3}m')

def run_template_notebook(template_nb, config_file, output_nb=None, err_file=None, output_dir=None,
                          log_status_to=None, color_err=False, verbose=False, transformations=None):
    """
    Run a template notebook with a set of transformations and save the completed notebook,
    log, and error traceback (if any).
    
    Parameters
    ----------
    template_nb:     Template notebook to use.
    config_file:     Configuration file (YAML).
    output_nb:       Location to write the completed notebook
                     (if `None`, use template notebook filename).
    err_file:        Location to write the error traceback log
                     (if `None`, create based on template notebook filename).
    output_dir:      Location where output will be written (default: current working directory).
                     A new subdirectory will be created with a name based on the config file name.
    log_status_to:   File-like object (stream) to write status (success/failure) to
                     (default: stdout).
    color_err:       Whether to keep ANSI color codes in the error traceback.
    verbose:         Print a description of replacements made in the template notebook.
    transformations: Transformations to apply to the notebook.
    """
    # base_dir = root of data repository
    base_dir = pint_pal.config.DATA_ROOT

    nb_name = os.path.splitext(os.path.split(template_nb)[1])[0]
    cfg_name = os.path.splitext(os.path.split(config_file)[1])[0]
    if output_dir is None:
        output_dir = os.getcwd()

    # Create working directory within output_dir
    workdir = os.path.join(output_dir, cfg_name)
    os.makedirs(workdir)

    # Fill in output filenames, if unspecified
    if output_nb is None:
        output_nb = os.path.join(workdir, f'{nb_name}.ipynb')
    if err_file is None:
        err_file = os.path.join(workdir, f'{nb_name}.traceback')
    if log_status_to is None:
        log_status_to = sys.stdout

    # Find absolute path to config_file (look in current working directory)
    # os.path.abspath() and os.path.normpath() will leave absolute paths alone
    config_file = os.path.abspath(config_file)

    # Find absolute paths to par_directory and tim_directory (look in base_dir)
    with open(config_file) as f:
        config = yaml.load(f)
    par_directory = config['par-directory']
    tim_directory = config['tim-directory']
    par_directory = os.path.normpath(os.path.join(base_dir, par_directory))
    tim_directory = os.path.normpath(os.path.join(base_dir, tim_directory))

    # Find absolute path to template_nb (look in "pint_pal/nb_templates")
    pint_pal_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    template_dir = os.path.join(pint_pal_dir, 'nb_templates')
    template_nb = os.path.normpath(os.path.join(template_dir, template_nb))

    default_transformations = {
        'config': f'"{config_file}"',
        'par_directory': f'"{par_directory}"',
        'tim_directory': f'"{tim_directory}"',
    }
    if transformations is None:
        transformations = default_transformations
    else:
        transformations = {**default_transformations, **transformations}

    with open(template_nb) as f:
        nb = nbformat.read(f, as_version=4)

    n_subs = transform_notebook(nb, transformations, verbose=verbose)
    cfg_name = os.path.splitext(os.path.split(config_file)[1])[0]
    
    ep = ExecutePreprocessor(timeout=0)
    try:
        ep.preprocess(nb, {'metadata': {'path': workdir}})
    except CellExecutionError as err:
        with open(err_file, 'w') as f:
            if not color_err:
                traceback = re.sub(ansi_color, '', err.traceback)
            print(traceback, file=f)
        if hasattr(err, 'ename'):
            print(f"{cfg_name}: failure - {err.ename}", file=log_status_to)
        else:
            print(f"{cfg_name}: failure - {err}", file=log_status_to)
        raise err
    finally:
        if output_nb is not None:
            with open(output_nb, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
    print(f"{cfg_name}: success!", file=log_status_to)
