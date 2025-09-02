"""
This module provides tools for automatically updating results (yamls/pars)
based on output from autoruns on Thorny Flats (or potentially elsewhere).
"""

from pint_pal.yamlio import *
from pint_pal.timingconfiguration import TimingConfiguration
from loguru import logger as log
from datetime import datetime
import subprocess
import argparse
import os

# accessible to functions here, apparently
PAL_PATH = "/home/jovyan/work/pint_pal/" # assume running from here?
INTERMED_PATH = "/nanograv/share/15yr/timing/intermediate/"
PAL_RESULTS = os.path.join(PAL_PATH,"results")
PAL_ARCHIVE = os.path.join(PAL_RESULTS,"archive")

def new_noise_results(input_pars,logs_only=False):
    """ Check in new par file, point to new noise results, edit yaml appropriately

    Parameters
    ==========
    input_pars: list
        one or more par files from latest noise run
        e.g. /nanograv/share/15yr/timing/intermediate/20211028.Noise.nb.d92afa6/results/J1713+0747gbt_PINT_20211029.nb.par 
    logs_only: bool, optional
        show logs describing file manipulations without making any changes (default: False)

    Example usage (from TA base dir):
    >>> python src/pint_pal/update_results.py -p [par file(s)]

    ...or to do a test run first without making any changes:
    >>> python src/pint_pal/update_results.py --logsonly -p [par file(s)]
    """
    for par_path in input_pars:
        # Compose yaml file corresponding to input par
        p = os.path.basename(par_path) # get par file only
        src = p.split('_')[0]
        mode = p.split('.')[-2]
        y = f"{src}.{mode}.yaml" # yaml file only
        yaml_path = os.path.join(PAL_PATH,"configs",y)

        # Determine noise-dir from par_path unless told to do otherwise (needs dev if so)
        noise_path = os.path.dirname(os.path.dirname(par_path)) # assumes etc./noise/results/*.par
        noise_path = os.path.join(noise_path,'') # adds / if not there

        log.info(src)
        if os.path.exists(yaml_path):
            log.info(f"Associating input par {p} with {y}")

            # Copy par file(s) and git add
            log.info(f"Copying {p} to {PAL_RESULTS}")
            # Could also add a check here for whether results are already up to date
            if not logs_only:
                process_cpnew = subprocess.Popen(["cp",par_path,PAL_RESULTS],
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE)
                # Ensures next process waits for this one to finish...
                _stdout, _stderr = process_cpnew.communicate()
                process_gitadd = subprocess.Popen(["git","add",os.path.join(PAL_RESULTS,p)],
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE)
                _stdout, _stderr = process_gitadd.communicate()

            # Read yaml file and use timing-model to identify existing par
            tc = TimingConfiguration(yaml_path)
            prev_par = tc.get_model_path()
            pp = os.path.basename(prev_par) # par only
            log.info(f"Previous par file: {prev_par}")

            # git mv to archive
            log.info(f"Moving {pp} to {PAL_ARCHIVE}")
            if not logs_only:
                process_gitmv = subprocess.Popen(["git","mv",prev_par,PAL_ARCHIVE],
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE)
                _stdout, _stderr = process_gitmv.communicate()

            # Modify yaml file accordingly (timing-model, compare-model, noise-dir, etc.)
            log.info(f"Setting {y} timing-model to: {p}")
            log.info(f"Setting {y} compare-model to: {pp}")
            if not logs_only:
                set_field(yaml_path,'timing-model',p)
                set_field(yaml_path,'compare-model',os.path.join("archive",pp))
            prev_noisedir = tc.get_noise_dir()
            if prev_noisedir:
                log.info(f"Setting compare-noise-dir to: {prev_noisedir}")
                if not logs_only: set_field(yaml_path,'compare-noise-dir',prev_noisedir)
            else:
                log.warning(f"{y} noise-dir is unset or does not exist.")

            log.info(f"Setting noise-dir to: {noise_path}")
            if not logs_only: set_field(yaml_path,'noise-dir',noise_path)

        else:
            log.warning(f"Corresponding yaml file {y} not found.")

def new_outlier_results(input_tims,logs_only=False):
    """ Find appropriate yaml(s), update excised-tim field

    Parameters
    ==========
    input_tims: list
        one or more excise.tim files from latest outlier run
        e.g. /nanograv/share/15yr/timing/intermediate/20211028.Outlier.nb.d92afa6/J1600-3053gbt.nb/J1600-3053gbt.nb_excise.tim 
    logs_only: bool, optional
        show logs describing file manipulations without making any changes (default: False)

    Example usage (from TA base dir):
    >>> python src/pint_pal/update_results.py -t [tim file(s)]

    ...or to do a test run first without making any changes:
    >>> python src/pint_pal/update_results.py --logsonly -t [tim file(s)]
    """
    for tim_path in input_tims:
        # Compose yaml file corresponding to input par
        t = os.path.basename(tim_path) # get tim file only
        src = t.split('.')[0]
        mode = t.split('.')[1].split('_')[0]
        y = f"{src}.{mode}.yaml" # yaml file only
        yaml_path = os.path.join(PAL_PATH,"configs",y)

        log.info(src)
        if os.path.exists(yaml_path):
            log.info(f"Associating input tim {t} with {y}")
            log.info(f"Setting {y} excised-tim to: {tim_path}")
            if not logs_only:
                set_field(yaml_path,'excised-tim',tim_path)

        else:
            log.warning(f"Corresponding yaml file {y} not found.")

def add_ready_for(input_yamls,version):
    """Add READY_FOR [version] changelogs to yaml(s)

    Parameters
    ==========
    input_yamls: list
        one or more yamls to modify
    version:
        data set version string (e.g. v1.1)

    Example usage (from TA base dir):
    >>> python src/pint_pal/update_results.py -y [yaml file(s)] --readyfor [version]
    """
    time = datetime.now()
    date_string = time.strftime("%Y-%m-%d")
    gitemail = os.popen('git config --get user.email').read().rstrip()
    user_string = gitemail.split('@')[0]

    for y in input_yamls:
        log.info(f"Adding READY_FOR changelog to {y}")
        ready_for_log = f"{date_string} {user_string} READY_FOR: {version}"
        set_field(y,'changelog',ready_for_log) # can set overwrite=False to test

def main():

    parser = argparse.ArgumentParser(
        description="Update/check in new TA results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--pars",
        type=str,
        nargs="+",
        help="New par file(s); noise run",
    )
    parser.add_argument(
        "-t",
        "--tims",
        type=str,
        nargs="+",
        help="New tim file(s); outlier run",
    )
    parser.add_argument(
        "-y",
        "--yamls",
        type=str,
        nargs="+",
        help="Work with yaml(s)",
    )
    parser.add_argument(
        "--readyfor",
        type=str,
        nargs=1,
        help="Add changelog READY_FOR with input version",
    )
    parser.add_argument(
        "--logsonly",
        action="store_true",
        default=False,
        help="Show logs without modifying files",
    )
    args = parser.parse_args()

    if args.pars:
        log.info('Working with par files from a recent noise run...')
        print(args.pars)
        new_noise_results(args.pars,logs_only=args.logsonly)
    if args.tims:
        log.info('Working with tim files from a recent outlier run...')
        print(args.tims)
        new_outlier_results(args.tims,logs_only=args.logsonly)
    if args.yamls:
        log.info('Working with input yamls...')
        print(args.yamls)
        if args.readyfor:
            add_ready_for(args.yamls,args.readyfor[0])
    if (not args.pars) and (not args.tims) and (not args.yamls):
        log.warning('No pars/tims/yamls specified.')


if __name__ == "__main__":
    main()
