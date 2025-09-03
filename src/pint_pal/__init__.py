import pint_pal.packageconfiguration # must go before any use of pint_pal.config
from pint_pal.packageconfiguration import set_data_root, reset_config
import pint_pal.checkin

from . import _version
__version__ = _version.get_versions()['version']
