import os
from typing import Optional

import threadpoolctl
from loguru import logger

def set_thread_limit(n_threads: Optional[int] = None, reserve: int = 2) -> None:
    """
    Set the maximum number of threads to be used by Numpy and other
    multithreaded code, including MCMC samplers, in this process.
    To take effect correctly, this must be run before Numpy is imported.

    Parameters
    ----------
    n_threads : int (optional)
        Maximum number of threads to use. If unspecified, look for an
        environment variable set by the HPC job manager. If none is found,
        the default is to use a limit based on the number of CPUs detected,
        subtracting `reserve` (by default, 2).
    reserve : int, default 2
        If the `n_threads` is unspecified, this will be subtracted from the
        number of CPUs to determine the maximum number of threads to use.
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
            n_threads = max(os.cpu_count() - reserve, 1)

    logger.info("Setting thread limit to {}", n_threads)
    threadpoolctl.threadpool_limits(limits=n_threads)
