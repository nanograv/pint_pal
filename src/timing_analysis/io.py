import numpy as np
import os
import glob

################################################################################

def find_git_directories(startDir, excludeDirs=["shared"]):
    """
    Function find all git directories within a starting directory (stardir).

    Inputs:
    ---------
    startDir ['string']: Absolute path for the desired directory to start walking from.
    excludeDirs ['list']: A list of strings of folder names to exlude from the walk function.

    Returns:
    ---------
    dirpath [generator]: Generator that can be interated over to find the desired git base directory.
    """
    exclude = set(excludeDirs)
    for dirpath, dirnames, _ in os.walk(startDir, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in exclude]
        if set(['info', 'objects', 'refs']).issubset(set(dirnames)):
            yield dirpath.split("/.git")[0]

################################################################################

class NotebookIO():
    def __init__(self):
        self.share_directory = "/nanograv/share"
        self.intermidate_data_directory = self.share_directory + "/15yr/timing/intermediate"
        self.working_directory = os.getcwd()
        for git_directory in find_git_directories("/home/jovyan/work/"):
            if set(['timing_analysis']).issubset(set(git_directory.split("/"))):
                self.git_directory = git_directory

    def share_intermidate_data(self,):
        return True

    def finalize_for_merge(self,):
        return True
