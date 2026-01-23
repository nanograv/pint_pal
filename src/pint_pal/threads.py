import os
from typing import Optional

def set_thread_limit(n_threads: Optional[int] = None) -> None:
    """
    Set the maximum number of threads to be used by Numpy and other
    multithreaded code, including MCMC samplers, in this process.
    To take effect correctly, this must be run before Numpy is imported.

    Parameters
    ----------
    n_threads: int (optional)
        Maximum number of threads to use. If unspecified, look for an
        environment variable set by the HPC job manager. If none is found,
        the default is to use 2 less than the number of CPUs (minimum 1).
    """
    if n_threads is None:
        if "SLURM_TASKS_PER_NODE" in os.environ:
            # Assume this is being run under Slurm
            n_threads = int(os.environ["SLURM_TASKS_PER_NODE"])
        elif "NCPUS" in os.environ:
            # Assume this is being run under the PBS job manager
            n_threads = int(os.environ["NCPUS"])
        else:
            # Use all but 2 CPUs (minimum 1)
            n_threads = max(os.cpu_count() - 2, 1)

    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
