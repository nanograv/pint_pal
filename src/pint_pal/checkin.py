from pint_pal.yamlio import read_yaml, write_yaml
from astropy import log
import os
import glob
import shutil
import git

def shuffle_pars(tc):
    """
    Copy the current timing-model to the archive directory and set it as compare-model;
    then copy the most recent par file to the par directory and set it as timing-model.
    Automatically perform a `git add` in each case.

    Parameters
    ----------
    tc: TimingConfiguration object derived from the YAML configuration file.
    """
    # Get file and repo locations
    par_dir = tc.par_directory
    archive_dir = os.path.join(par_dir, "archive")
    repo = git.Repo(par_dir, search_parent_directories=True)

    timing_model = tc.get_model_path()
    timing_model_basename = os.path.split(timing_model)[1]
    new_compare_model = os.path.join(archive_dir, timing_model_basename)

    # Get path to new par file
    # Assummes the newest parfile in the working directory is the correct one
    new_par = max(glob.iglob("*.par"), key=os.path.getctime)
    new_par_basename = os.path.split(new_par)[1]
    new_timing_model = os.path.join(par_dir, new_par_basename)

    # Make sure there are no uncommitted changes to the current timing-model
    head = repo.commit("HEAD")
    tracked_files = set(item.path for item in head.tree.traverse())
    diff = head.diff(None) # diff between HEAD and current working directory state

    if timing_model not in tracked_files:
        log.info(f"Par file to be archived ({timing_model}) is not tracked. "
                 f"Copying it to {archive_dir} and leaving the original.")
        shutil.copy(timing_model, new_compare_model)
        repo.index.add(os.path.join(os.getcwd(), new_compare_model))
    else:
        log.info(f"Moving {timing_model} to {archive_dir}")
        shutil.move(timing_model, new_compare_model, copy_function=shutil.copy)
        repo.index.add(os.path.abspath(new_compare_model))

    log.info(f"Copying {new_par} to {par_dir}")             
    shutil.copy(new_par, new_timing_model)
    repo.index.add(os.path.abspath(new_timing_model))
    
    # Update YAML configuration accordingly
    config_path = os.path.abspath(tc.filename)
    config = read_yaml(config_path)
    config['timing-model'] = os.path.relpath(new_timing_model, start=par_dir)
    config['compare-model'] = os.path.relpath(new_compare_model, start=par_dir)
    write_yaml(config, config_path)
    repo.index.add(config_path)
 
