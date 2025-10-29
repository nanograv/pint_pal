'''
Unit tests for packageconfiguration.py
'''
import os.path
from ruamel.yaml import YAML
yaml = YAML(typ='safe')

import pint_pal
from pint_pal.packageconfiguration import PackageConfiguration

def test_read_config():
    """
    Test that the defaults.yaml file has been read in, and all
    keys in the file have been loaded into pint_pal.config
    """
    pint_pal_dir = os.path.dirname(pint_pal.__file__)
    defaults_yaml = os.path.join(pint_pal_dir, 'defaults.yaml')
    with open(defaults_yaml, 'r') as f:
        config_dict = yaml.load(f)

    for key in config_dict.keys():
        assert getattr(pint_pal.config, key) == config_dict[key]
    assert pint_pal.config.PACKAGE_DIR == pint_pal_dir
    assert pint_pal.config.DATA_ROOT == '.'

def test_update_config():
    """
    Check that updating and resetting the configuration works
    """
    default_bipm = pint_pal.config.LATEST_BIPM
    default_ephem = pint_pal.config.LATEST_EPHEM

    tests_dir = os.path.dirname(__file__)
    config_file = os.path.join(tests_dir, 'configs', 'pint_pal_project.yaml')
    pint_pal.config.update(config_file)

    assert pint_pal.config.LATEST_BIPM == "BIPM2019" # matches test file
    assert pint_pal.config.LATEST_EPHEM == "DE430"   # matches test file

    pint_pal.reset_config()

    assert pint_pal.config.LATEST_BIPM == default_bipm
    assert pint_pal.config.LATEST_EPHEM == default_ephem
