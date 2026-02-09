import os

def set_threads(num):
    # Set all Relevant Environment Variables for Multi-Threaded Workloads
    os.environ["OMP_NUM_THREADS"] = num       # openmp
    os.environ["OPENBLAS_NUM_THREADS"] = num   # openblas
    os.environ["MKL_NUM_THREADS"] = num        # mkl
    os.environ["NUMEXPR_NUM_THREADS"] = num    # numexpr
    os.environ["VECLIB_MAXIMUM_THREADS"] = num # acclerate
