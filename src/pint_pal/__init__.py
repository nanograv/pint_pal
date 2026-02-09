import pint_pal.packageconfiguration # must go before any use of pint_pal.config
from pint_pal.packageconfiguration import set_data_root, reset_config
from pint_pal.threads import set_thread_limit
import pint_pal.checkin
import pint_pal.logger
from pint_pal.set_threads import set_threads
import os, sys
from pathlib import Path

# In Jupyter, if Tempo2 is installed in the kernel environment,
# but not in the base environment (possible when using conda or pixi),
# libstempo may not be able to find the correct path to the Tempo2
# shared directory unless the TEMPO2 environment variable is set.
# This will set it automatically as long as Tempo2 exists in the
# "obvious" place, and so should reduce the need to manually set it.
if 'TEMPO2' not in os.environ:
    env_base_dir = Path(sys.executable).parent.parent
    tempo2_dir = env_base_dir / 'share' / 'tempo2'
    if tempo2_dir.exists():
        os.environ['TEMPO2'] = str(tempo2_dir)

# Set number of threads, default being max(N_CPU - 2, 1)
# but with ability to read from various systems
# This may be overwritten by what is provided in TimingConfiguration
# but ensures that without a call to TimingConfiguration, the values
# are set to this default.
NUM_THREADS = str(max(os.cpu_count() - 2, 1)) 

# For HPC Systems with a SLURM Job Manager
if "SLURM_TASKS_PER_NODE" in os.environ:
    NUM_THREADS = os.environ["SLURM_TASKS_PER_NODE"]

# For HPC Systems with a PBS Job Manager
if "NCPUS" in os.environ:
    NUM_THREADS = os.environ["NCPUS"]

        
from . import _version
__version__ = _version.get_versions()['version']
