import os, sys

def test_numpy_not_imported_too_early():
    import pint_pal
    assert "numpy" not in sys.modules

def test_env_vars_set():
    import pint_pal
    pint_pal.set_thread_limit(1)
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "1"
    assert os.environ["VECLIB_MAXIMUM_THREADS"] == "1"
