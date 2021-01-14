import numpy as np
import os
import glob

class NotebookIO(self, ):
    def __init__(self):
        self.share_directory = "/nanograv/share"
        self.intermidate_data_directory = self.share_directory + "/15yr/timing/intermediate"
        self.working_directory = ""
        self.git_directory = ""

    def share_intermidate_data(self,):
        return True

    def finalize_for_merge(self,):
        return True
