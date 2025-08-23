from ruamel.yaml import YAML
import os.path
yaml = YAML(typ='safe')
PACKAGE_DIR = os.path.dirname(__file__)
DATA_ROOT = '.'

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
    global DATA_ROOT
    DATA_ROOT = os.path.realpath(os.path.expanduser(path))
    try:
        read_config_file(os.path.join(DATA_ROOT, 'pint_pal_project.yaml'))
    except FileNotFoundError:
        pass

def reset_data_root():
    """
    Reset the data root and config variables to the default values.
    """
    global DATA_ROOT
    DATA_ROOT = '.'
    read_config_file(os.path.join(PACKAGE_DIR, 'defaults.yaml'))

def read_config_file(config_file):
    """
    Read a configuration file, along the lines of `defaults.yaml`, and load the results
    into a location that can be accessed by other PINT Pal code.
    """
    with open(config_file, 'r') as f:
        config = yaml.load(f)

    global RUN_FTEST
    global RUN_NOISE_ANALYSIS
    global CHECK_EXCISION
    global USE_EXISTING_NOISE_DIR
    global USE_TOA_PICKLE
    global LATEST_BIPM
    global LATEST_EPHEM
    global PLANET_SHAPIRO
    global CORRECT_TROPOSPHERE
    global FREQUENCY_RATIO
    global MAX_SOLARWIND_DELAY
    global LATEST_TOA_RELEASE

    if 'RUN_FTEST' in config:
        RUN_FTEST = config['RUN_FTEST']
    if 'RUN_NOISE_ANALYSIS' in config:
        RUN_NOISE_ANALYSIS = config['RUN_NOISE_ANALYSIS']
    if 'CHECK_EXCISION' in config:
        CHECK_EXCISION = config['CHECK_EXCISION']
    if 'USE_EXISTING_NOISE_DIR' in config:
        USE_EXISTING_NOISE_DIR = config['USE_EXISTING_NOISE_DIR']
    if 'USE_TOA_PICKLE' in config:
        USE_TOA_PICKLE = config['USE_TOA_PICKLE']
    if 'LATEST_BIPM' in config:
        LATEST_BIPM = config['LATEST_BIPM']
    if 'LATEST_EPHEM' in config:
        LATEST_EPHEM = config['LATEST_EPHEM']
    if 'PLANET_SHAPIRO' in config:
        PLANET_SHAPIRO = config['PLANET_SHAPIRO']
    if 'CORRECT_TROPOSPHERE' in config:
        CORRECT_TROPOSPHERE = config['CORRECT_TROPOSPHERE']
    if 'FREQUENCY_RATIO' in config:
        FREQUENCY_RATIO = config['FREQUENCY_RATIO']
    if 'MAX_SOLARWIND_DELAY' in config:
        MAX_SOLARWIND_DELAY = config['MAX_SOLARWIND_DELAY']
    if 'LATEST_TOA_RELEASE' in config:
        LATEST_TOA_RELEASE = config['LATEST_TOA_RELEASE']

read_config_file(os.path.join(PACKAGE_DIR, 'defaults.yaml'))
