import numpy as np
import pint_pal
import threadpoolctl
import os

def test_thread_limit():
    """
    Tests setting the number of threads using threadpoolctl.
    """
    n_threads_orig = threadpoolctl.threadpool_info()[0]["num_threads"]

    expected_num = max(os.cpu_count() - 2, 1)
    pint_pal.set_thread_limit()
    for lib_info in threadpoolctl.threadpool_info():
        assert lib_info["num_threads"] == expected_num

    pint_pal.set_thread_limit(1)
    for lib_info in threadpoolctl.threadpool_info():
        assert lib_info["num_threads"] == 1

    pint_pal.set_thread_limit(n_threads_orig)
