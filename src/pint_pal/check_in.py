from yamlio import read_yaml, write_yaml
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
    yaml_file = tc.filename
    par_dir = tc.par_directory
    archive_dir = os.path.join(par_dir, "archive")
    repo = git.Repo(par_dir, search_parent_directories=True)

    timing_model = tc.get_model_path()
    timing_model_basename = os.path.split(timing_model)[1]

    # Assummes the newest parfile in the working directory is the correct one
    new_par = max(glob.iglob("*.par"), key=os.path.getctime)

    new_compare_model = os.path.join(archive_dir, timing_model_basename)
    shutil.copy(timing_model, new_compare_model)
    repo.index.add(new_compare_model)

    new_timing_model = os.path.join(par_dir, new_par_basename)
    shutil.copy(new_par, new_timing_model)
    repo.index.add(new_timing_model)
