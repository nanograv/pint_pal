import os
import pytest
from pint_pal.set_threads import set_threads


def test_set_threads():
    """
    Simple unit test to check number of threads is set as expected
    """
    set_threads(4)

    assert os.environ["OMP_NUM_THREADS"] == 4
