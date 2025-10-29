import pint_pal
from ruamel.yaml import YAML
import os.path
yaml = YAML(typ='safe')

class PackageConfiguration:
    """
    A class to manage package-level configuration data.
    It shouldn't be necessary for the user to create an instance of this class.
    Instead, the single, global instance can be managed using the `set_data_root()`
    and `reset_config()` functions.
    """
    __slots__ = [
        'PACKAGE_DIR',
        'DATA_ROOT',
        'RUN_FTEST',
        'RUN_NOISE_ANALYSIS',
        'CHECK_EXCISION',
        'USE_EXISTING_NOISE_DIR',
        'USE_TOA_PICKLE',
        'LOG_LEVEL',
        'LOG_TO_FILE',
        'LATEST_BIPM',
        'LATEST_EPHEM',
        'PLANET_SHAPIRO',
        'CORRECT_TROPOSPHERE',
        'FREQUENCY_RATIO',
        'MAX_SOLARWIND_DELAY',
        'LATEST_TOA_RELEASE',
    ]

    def __init__(self, config_file, data_root):
        """
        Initialize the PackageConfiguration from a YAML configuration file.
        """
        self.update(config_file)
        self.PACKAGE_DIR = os.path.dirname(__file__)
        self.DATA_ROOT = data_root

    def update(self, config_file):
        """
        Update configuration values without changing the data root.
        Can be called on its own to change only some configuration options.
        """
        with open(config_file, 'r') as f:
            config = yaml.load(f)

        for key in self.__slots__:
            if key in config:
                setattr(self, key, config[key])

def set_data_root(path):
    """
    Set the root directory of the data repository to be used with PINT Pal.
    PINT Pal will search this directory for a configuration file specifying settings
    such as the appropriate JPL ephemeris and version of TT(BIPM) to check for when
    validating timing models.

    It will also be treated as the base directory when resolving paths in YAML
    configuration files. This allows notebooks (or scripts) using YAML files within
    the data repository, which specify paths relative to the data root, to be run
    from other locations.

    The default value of `data_root` is '.' (the current working directory), which
    is sufficient in cases where either (1) no data repository is in use, or
    (2) all scripts and notebooks are run from the root of the data repository.
    """
    data_root = os.path.realpath(os.path.expanduser(path))
    pint_pal.config.DATA_ROOT = data_root

    config_file = os.path.join(data_root, 'pint_pal_project.yaml')
    try:
        pint_pal.config.update(config_file)
    except FileNotFoundError:
        pass

def reset_config():
    """
    Reset the data root and config variables to the default values.
    """
    config_file = os.path.join(os.path.dirname(__file__), 'defaults.yaml')
    pint_pal.config = PackageConfiguration(config_file, '.')

reset_config()
