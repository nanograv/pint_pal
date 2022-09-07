"""
This module provides tools for checking the finality of yamls
and gathering results for hand-off to DWG
"""

from timing_analysis.yamlio import *
from timing_analysis.timingconfiguration import TimingConfiguration
from astropy import log
from datetime import datetime
import subprocess
import shutil
import argparse
import glob
import os

# accessible to functions here, apparently
TA_PATH = "/home/jovyan/work/timing_analysis/" # assume running from here?
INTERMED_PATH = "/nanograv/share/15yr/timing/intermediate/"
TA_RESULTS = os.path.join(TA_PATH,"results")
TA_CONFIGS = os.path.join(TA_PATH,"configs")

def make_release_dir(type, overwrite=False):
    """
    Make new release directory to contain latest results

    Parameters
    ==========
    type: str
        narrowband (nb) or wideband (wb)
    overwrite: bool, optional
        overwrite existing files if release directory already exists (default: False)
    """
    now = datetime.now()
    Ymd = now.strftime("%Y%m%d")

    cmd = subprocess.Popen(["git","rev-parse","--short","HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
    githash, stderr = cmd.communicate()
    githash = githash.strip().decode() # decode() since a bytestring is returned initially (b'[hash]')
    release_dir = f"{INTERMED_PATH}{Ymd}.Release.{type}.{githash}"
    if not os.path.isdir(release_dir):
        log.info(f"Making new release directory: {release_dir}")
        os.mkdir(release_dir)
    elif os.path.isdir(release_dir) and overwrite:
        log.warning(f"Overwriting files in release directory: {release_dir}")
    else:
        log.warning(f"Release directory already exists: {release_dir}")

    return release_dir

def check_cleared(type):
    """
    Check that all yamls of the specified type have been cleared

    Parameters
    ==========
    type: str
        narrowband (nb), wideband (wb), or both (nbwb)
    """
    if type == "nbwb":
        yamls = glob.glob(f"{TA_CONFIGS}/*.yaml")
    else:
        yamls = glob.glob(f"{TA_CONFIGS}/*.{type}.yaml")
    for y in yamls:
        tc = TimingConfiguration(y)
        if not tc.get_check_cleared():
            log.warning(f"{tc.get_source()} has not been cleared.")
            
    return yamls

def check_dupes_copy(results, release_dir, add_base=None):
    """
    Check for duplicate results (copy if no duplicates)

    Parameters
    ==========
    results: list
        list (should be one element) of results file(s) to copy
    release_dir: str
        path to release directory
    add_base: str, optional
        optional basename added to file being copied
    """
    if len(results) != 1:
        log.warning("Multiple/no matching results files found.")
        print(results)
    else:
        file2copy = os.path.basename(results[0])
        if add_base:
            file2copy = f"{add_base}.{file2copy}"

        dest_file = f"{release_dir}/{file2copy}"
        shutil.copyfile(results[0], dest_file)

def locate_copy_results(yamls,type,destination=None):
    """
    Get latest results from yamls, copy to release directory

    Parameters
    ==========
    yamls: list
        yamls to use for locating latest results
    type: str
        narrowband (nb) or wideband (wb)
    destination: str
        path to release directory
    """
    for y in yamls:
        tc = TimingConfiguration(y)
        source = tc.get_source()
        noise_dir = tc.get_noise_dir()
        latest_yaml = [y]
        latest_par = [f"{TA_PATH}{tc.get_model_path()}"]
        latest_tim = glob.glob(f"{noise_dir}results/{source}_*.tim") # underscore to avoid duplicating split-tel results
        noise_chains = glob.glob(f"{noise_dir}{source}_{type}/chain_1.txt")
        noise_pars = glob.glob(f"{noise_dir}{source}_{type}/pars.txt")

        log.info(f"Locating/copying files for {source}...")
        check_dupes_copy(latest_tim, destination)
        check_dupes_copy(latest_par, destination)
        check_dupes_copy(latest_yaml, destination)
        check_dupes_copy(noise_chains, destination, add_base=f"{source}.{type}")
        check_dupes_copy(noise_pars, destination, add_base=f"{source}.{type}")

def main():

    parser = argparse.ArgumentParser(
        description="Make release directory; copy final nb/wb TA data products there",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        nargs=1,
        help="Release type: nb, wb, or both (nbwb)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite files in release directory",
    )
    args = parser.parse_args()

    if not args.type:
        print("Please provide a release type.")
    elif args.type[0] not in ['nb','wb','nbwb']:
        print(args.type)
        print("Unrecognized release type.")
    else:
        if args.type[0] != "nbwb":
            # make directory
            rel_dir = make_release_dir(args.type[0], overwrite=args.overwrite)

            # get yamls
            yamls = check_cleared(args.type[0])

            # locate results and copy them to release directory
            locate_copy_results(yamls,args.type[0],rel_dir)

        else: # nbwb
            rel_dir = make_release_dir(args.type[0], overwrite=args.overwrite)
            nb_yamls = check_cleared('nb')
            locate_copy_results(nb_yamls,'nb',rel_dir) # works for nb/wb separately
            wb_yamls = check_cleared('wb')
            locate_copy_results(yamls,'wb',rel_dir)

if __name__ == "__main__":
    main()
