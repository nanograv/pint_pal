"""
This module provides tools for round-tripping yamls and adding fields/blocks
if desired; ruyamel.yaml preserves order of existing fields and comments.
"""

from ruamel.yaml import YAML
import argparse
import glob
from astropy import log
import numpy as np
from timing_analysis.defaults import *
import os

yaml = YAML()
RELEASE = f'/nanograv/timing/releases/15y/toagen/releases/{LATEST_TOA_RELEASE}/' 

def fix_toa_info(yaml_file,current_release=RELEASE,overwrite=True,extension='fix'):
    """Checks/fixes tim-directory, toas, toa-type from existing yaml; writes new one.

    Parameters
    ==========
    yaml_file: str, input file
    current_release: str, optional
        desired tim-directory (current release by default)
    overwrite: bool, optional
        write yaml with same name (true), or add extenion (false)
    extension: str, optional
        extention added to output filename if overwrite=False
    """
    file_only = yaml_file.split('/')[-1]
    source,toa_type,init_ext = file_only.split('.')

    # ASP/GASP files don't currently have nb/wb in them, so all are nb. GAH!
    # This is why I'm removing {toa-type} from the glob for now
    # Determine TOAs available for the source in the latest release
    release_toas = [gg.split('/')[-1] for gg in glob.glob(f'{current_release}{source}*.tim')]
    if toa_type == 'nb':
        release_toas = [x for x in release_toas if 'wb' not in x]
    elif toa_type == 'wb':
        release_toas = [x for x in release_toas if 'wb' in x]
    else:
        raise ValueError(f'Wrong file format (does not include nb/wb): {yaml_file}')

    release_toas.sort()
    changes = []
    config = read_yaml(yaml_file)

    # Check toa-type; currently case-sensitive
    if toa_type.upper() == config.get('toa-type'):
        pass
    else:
        log.warning(f'{source} yaml filename is not consistent with toa-type; check this!')

    # Check tim-directory and fix if necessary
    yamltimdir = config.get('tim-directory')
    log.debug(f'{source} tim-directory')
    log.debug(f'yaml: {yamltimdir}')
    log.debug(f'want: {current_release}')
    if not yamltimdir:
        log.error(f'{source} has no tim-directory listed. WTF.')
    elif yamltimdir == current_release:
        log.info(f'{source} tim-directory matches latest release.')
    else:
        log.warning(f'{source} tim-directory does not match latest release; fixing it.')
        config['tim-directory'] = current_release
        changes.append('tim-directory')

    # Check toas and fix if necessary
    yamltoas = config.get('toas')
    yamltoas.sort()
    log.debug(f'{source} toas')
    log.debug(f'yaml: {yamltoas}')
    log.debug(f'want: {release_toas}')
    if not yamltoas: 
        log.error(f'{source} has no {toa_type} toas listed. WTF.')
    elif yamltoas == release_toas:
        log.info(f'{source} current {toa_type} toas are all in use.')
    else:
        log.warning(f'{source} {toa_type} toas do not match those available; updating.')
        config['toas'] = release_toas
        changes.append('toas')

    out_yaml = get_outfile(yaml_file,overwrite=overwrite,extension=extension)

    # If there were changes, write an updated yaml file
    if not changes:
        log.info(f'{yaml_file} TOA info up to date; no new file will be written.')
        pass
    else:
        log.info(f'{source} changes were made to: {changes}')
        write_yaml(config, out_yaml)

def add_niterations(yaml_file,overwrite=True,extension='fix',insert_after='fitter'):
    """Adds n-iterations field to yaml file

    Parameters
    ==========
    yaml_file: str, input file
    overwrite: bool, optional
        write yaml with same name (true), or add extenion (false)
    extension: str, optional
        extention added to output filename if overwrite=False
    insert_after: str, optional
        field after which to insert this field in the yaml
    """
    config = read_yaml(yaml_file)
    out_yaml = get_outfile(yaml_file,overwrite=overwrite,extension=extension)

    if not config.get('n-iterations'):
        # n-iterations field goes after fitter by default (insert_after)
        insert_ind = list(config).index(insert_after)+1
        config.insert(insert_ind,'n-iterations',1)

        log.info(f'Adding n-iterations to {out_yaml}.')
        write_yaml(config, out_yaml)
    else:
        log.info(f'{yaml_file} already contains n-iterations field.')

def add_block_field(yaml_file,block_key,key,value,overwrite=True,extension='fix'):
    """Add key/value to specified block in the yaml

    Parameters
    ==========
    yaml_file: str, input file
    block_key: str, yaml block to which key/value should be added
    key: str, yaml field name
    value: variable type, set value associated with key
    overwrite: bool, optional
        write yaml with same name (true), or add extenion (false)
    extension: str, optional
        extention added to output filename if overwrite=False
    """
    config = read_yaml(yaml_file)
    out_yaml = get_outfile(yaml_file,overwrite=overwrite,extension=extension)

    #d.keys().index(k) if we eventually want to add one key after another
    if key in config[block_key]:
        config[block_key][key] = value
        log.info(f'Config {block_key}/{key} already exists in {yaml_file}; setting it to {value}.')
    else:
        config[block_key][key] = value
        log.info(f'Adding {block_key}/{key} to {yaml_file}; setting it to {value}.')

    write_yaml(config, out_yaml)

def add_noise_block(yaml_file,overwrite=True,extension='fix',insert_after='bipm',
        noise_dir=None):
    """Adds noise block to yaml file

    Parameters
    ==========
    yaml_file: str, input file
    overwrite: bool, optional
        write yaml with same name (true), or add extenion (false)
    extension: str, optional
        extention added to output filename if overwrite=False
    insert_after: str, optional
        field after which to insert this block in the yaml
    noise_dir: str, optional
        base directory where existing noise results can be found
    """
    config = read_yaml(yaml_file)
    out_yaml = get_outfile(yaml_file,overwrite=overwrite,extension=extension)

    if not config.get('noise'):
        # noise block goes after bipm by default (insert_after)
        insert_ind = list(config).index(insert_after) + 1
        noise_block = {'results-dir':noise_dir}
        config.insert(insert_ind,'noise',noise_block,'control noise runs, apply results')
        log.info(f'Adding standard noise block to {out_yaml}.')
        write_yaml(config, out_yaml)
    else:
        log.info(f'{yaml_file} already contains noise block.')

def add_dmx_block(yaml_file,overwrite=True,extension='fix',insert_after='noise'):
    """Adds dmx block to yaml file

    Parameters
    ==========
    yaml_file: str, input file
    overwrite: bool, optional
        write yaml with same name (true), or add extenion (false)
    extension: str, optional
        extention added to output filename if overwrite=False
    insert_after: str, optional
        field after which to insert this block in the yaml
    """
    config = read_yaml(yaml_file)
    out_yaml = get_outfile(yaml_file,overwrite=overwrite,extension=extension)

    if not config.get('dmx'):
        # dmx block goes after noise by default (insert_after)
        insert_ind = list(config).index(insert_after) + 1
        dmx_block = {'ignore-dmx':False,'fratio':1.1,'max-sw-delay':0.1,'custom-dmx':[]}
        config.insert(insert_ind,'dmx',dmx_block,'control dmx windowing/fixing')
        log.info(f'Adding standard dmx block to {out_yaml}.')
        write_yaml(config, out_yaml)
    else:
        log.info(f'{yaml_file} already contains dmx block.')

def add_outlier_block(yaml_file,overwrite=True,extension='fix',insert_after='dmx'):
    """Adds outlier block to yaml file

    Parameters
    ==========
    yaml_file: str, input file
    overwrite: bool, optional
        write yaml with same name (true), or add extenion (false)
    extension: str, optional
        extention added to output filename if overwrite=False
    insert_after: str, optional
        field after which to insert this block in the yaml
    """
    config = read_yaml(yaml_file)
    out_yaml = get_outfile(yaml_file,overwrite=overwrite,extension=extension)

    if not config.get('outlier'):
        # outlier block goes after dmx by default (insert_after)
        insert_ind = list(config).index(insert_after) + 1
        outlier_block = {'method':'gibbs','n-burn':1000,'n-samples':20000}
        config.insert(insert_ind,'outlier',outlier_block,'control outlier analysis runs')
        log.info(f'Adding standard outlier block to {out_yaml}.')
        write_yaml(config, out_yaml)
    else:
        log.info(f'{yaml_file} already contains outlier block.')

def add_results_block(yaml_file,overwrite=True,extension='fix',insert_before='ignore',
        set_to_current=True):
    """Adds intermediate-results block to yaml file

    Parameters
    ==========
    yaml_file: str, input file
    overwrite: bool, optional
        write yaml with same name (true), or add extenion (false)
    extension: str, optional
        extention added to output filename if overwrite=False
    insert_before: str, optional
        field before which to insert this block in the yaml
    set_to_current: bool, optional
        set noise-dir and excised-tim approrpiately for current results available
    """
    config = read_yaml(yaml_file)
    out_yaml = get_outfile(yaml_file,overwrite=overwrite,extension=extension)

    # current settings
    if set_to_current:
        noise_dir = f"/nanograv/share/15yr/timing/intermediate/noise-chains/"
        excised_timfile = f"{config.get('source')}.{config.get('toa-type').lower()}_excise.tim"
        excised_timpath = f"/nanograv/share/15yr/timing/intermediate/excised-tims/"
        excised_tim = os.path.join(excised_timpath,excised_timfile)
        if not os.path.exists(excised_tim):
            log.warning(f"Excised tim file does not yet exist: {excised_timfile}")
            #excised_tim = None # meh, add it anyway
    else:
        noise_dir = None
        excised_tim = None

    if not config.get('intermediate-results'):
        # results block goes before ignore by default (insert_before)
        insert_ind = list(config).index(insert_before) 
        results_block = {'noise-dir':noise_dir,'excised-tim':excised_tim}
        config.insert(insert_ind,'intermediate-results',results_block,'use results from previous runs')
        log.info(f'Adding intermediate results block to {out_yaml}.')
        write_yaml(config, out_yaml)
    else:
        log.info(f'{yaml_file} already contains noise block.')

def add_check_block(yaml_file,overwrite=True,extension='fix',insert_after='ignore'):
    """Adds check block to yaml file

    Parameters
    ==========
    yaml_file: str, input file
    overwrite: bool, optional
        write yaml with same name (true), or add extenion (false)
    extension: str, optional
        extention added to output filename if overwrite=False
    insert_after: str, optional
        field after which to insert this block in the yaml
    """
    config = read_yaml(yaml_file)
    out_yaml = get_outfile(yaml_file,overwrite=overwrite,extension=extension)

    if not config.get('check'):
        # check block goes after ignore by default (insert_after)
        insert_ind = list(config).index(insert_after) + 1
        check_block = {
            'toa-outliers':None,
            'dmx-outliers':None,
            'unusual-params':None,
            'other':None,
            'cleared':False,
            }
        config.insert(insert_ind,'check',check_block,'check before final')
        log.info(f'Adding standard check block to {out_yaml}.')
        write_yaml(config, out_yaml)
    else:
        log.info(f'{yaml_file} already contains check block.')

def check_cleared(yaml_file,overwrite=True,extension='fix'):
    """Assigns 'cleared' status based on other check block fields

    Parameters
    ==========
    yaml_file: str, input file
    overwrite: bool, optional
        write yaml with same name (true), or add extenion (false)
    extension: str, optional
        extention added to output filename if overwrite=False
    """
    config = read_yaml(yaml_file)
    out_yaml = get_outfile(yaml_file,overwrite=overwrite,extension=extension)

    if not config.get('check'):
        log.warning(f'{yaml_file} check block does not exist.')
    else:
        cleared = True
        for v in config.get('check').keys():  
            if not config['check'][v]:
                pass
            else:
                log.info(f'{yaml_file} not clear due to {v}.')
                cleared = False
                break

        if cleared:
            log.info(f'{yaml_file} check is cleared.')
            config['check']['cleared'] = True
            write_yaml(config, out_yaml)

def curate_comments(yaml_file,overwrite=True,extension='fix'):
    """Standardizes info comments on specific yaml fields

    Parameters
    ==========
    yaml_file: str, yaml filename
    overwrite: bool, optional
        write yaml with same name (true), or add extenion (false)
    extension: str, optional
        extention added to output filename if overwrite=False
    """
    config = read_yaml(yaml_file)
    out_yaml = get_outfile(yaml_file,overwrite=overwrite,extension=extension)

    # currently assumes these fields exist, should check explicitly
    config.yaml_add_eol_comment("parameters not included here will be frozen",'free-params')
    config.yaml_add_eol_comment("toa excision",'ignore')

    toplevel_keys = dict(config).keys()
    ignore_keys = dict(config['ignore']).keys()
    dmx_keys = dict(config['dmx']).keys()

    if config.get('ignore').get('bad-toa'): 
        try:
            config['ignore'].yaml_add_eol_comment("designated by [name,chan,subint]",'bad-toa')
        except:
            pass
    else:
        if 'bad-toa' in ignore_keys:
            log.info('bad-toa field exists, is not set.')
        else:
            log.warning('bad-toa field does not exist.')

    if config.get('ignore').get('bad-range'):
        try:
            config['ignore'].yaml_add_eol_comment("designated by [mjd_start,mjd_end,(backend optional)]",'bad-range')
        except:
            pass
    else:
        if 'bad-range' in ignore_keys:
            log.info('bad-range field exists, is not set.')
        else:
            log.warning('bad-range field does not exist.')

    if config.get('ignore').get('bad-file'):
        try:
            config['ignore'].yaml_add_eol_comment("designated by basename string {backend}_{mjd}_{source}",'bad-file')
        except:
            pass
    else:
        if 'bad-file' in ignore_keys:
            log.info('bad-file field exists, is not set.')
        else:
            log.warning('bad-file field does not exist.')

    config['dmx'].yaml_add_eol_comment("finer binning when solar wind delay > threshold (us)",'max-sw-delay')
    config['dmx'].yaml_add_eol_comment("designated by [mjd_low,mjd_hi,binsize]",'custom-dmx')
    write_yaml(config, out_yaml)

def set_field(yaml_file,field,value,overwrite=True,extension='fix'):
    """Sets yaml field to provided value

    Parameters
    ==========
    yaml_file: str, yaml filename
    field: str, valid yaml key 
    value: str/list/dict, desired value
    overwrite: bool, optional
        write yaml with same name (true), or add extenion (false)
    extension: str, optional
        extention added to output filename if overwrite=False
    """
    config = read_yaml(yaml_file)
    out_yaml = get_outfile(yaml_file,overwrite=overwrite,extension=extension)

    valid_keys = ['source','par-directory','tim-directory','timing-model',
            'compare-model','toas','free-params','free-dmx','toa-type','fitter',
            'n-iterations','ephem','bipm','noise','results-dir','dmx','ignore-dmx',
            'fratio','max-sw-delay','custom-dmx','intermediate-results','noise-dir',
            'compare-noise-dir','excised-tim','no-corner','ignore','mjd-start',
            'mjd-end','snr-cut','bad-toa','bad-range','bad-file','changelog']

    if field not in valid_keys:
        log.warning(f'Provided field ({field}) not valid.')
    else:
        if field == 'ephem' and isinstance(value,str):
            config['ephem'] = value
        elif field == 'mjd-end' and isinstance(value,float):
            config['ignore']['mjd-end'] = value
        elif field == 'timing-model' and isinstance(value,str):
            config['timing-model'] = value
        elif field == 'compare-model' and isinstance(value,str):
            config['compare-model'] = value
        # changelog appends value to existing
        elif field == 'changelog' and isinstance(value,str):
            config['changelog'].append(value)
        elif field == 'noise-dir' and isinstance(value,str):
            config['intermediate-results']['noise-dir'] = value
        elif field == 'compare-noise-dir' and isinstance(value,str):
            config['intermediate-results']['compare-noise-dir'] = value
        elif field == 'no-corner' and isinstance(value,bool):
            config['intermediate-results']['no-corner'] = value
        elif field == 'excised-tim' and isinstance(value,str):
            config['intermediate-results']['excised-tim'] = value
        else:
            log.error(f'Provided field ({field}) is valid, but not yet implemented in set_field(); doing nothing.')

    write_yaml(config, out_yaml)

def fix_badfile(yaml_file):
    """Suggests bad-file replacements (if strings) with one-element lists

    Parameters
    ==========
    yaml_file: str, yaml filename
    """
    config = read_yaml(yaml_file)
    try:
        n_be = len(config['ignore']['bad-file'])
        log.info(f'{yaml_file}: {n_be} bad-file entries to fix...')
        for i in range(n_be):
            be_field = config['ignore']['bad-file'][i]
            if isinstance(be_field,str):
                print(f'  - [{be_field}]')
    except TypeError:
        log.info(f'{yaml_file}: 0 bad-file entries.')
    except KeyError:
        log.info(f'{yaml_file}: no bad-file field in the ignore block.')

def read_yaml(yaml_file):
    """Reads a yaml file, returns the object

    Parameters
    ==========
    yaml_file: str, input file
    """
    with open(yaml_file) as FILE:
        config = yaml.load(FILE)
    return config

def get_outfile(yaml_file,overwrite=True,extension='fix'):
    """Determines output filename

    Parameters
    ==========
    yaml_file: str, input file 
    overwrite: bool, optional
        write yaml with same name (true), or add extenion (false)
    extension: str, optional
        extention added to output filename if overwrite=False
    """
    if overwrite:
        out_yaml = yaml_file
    else:
        out_yaml = f'{yaml_file}.{extension}'
    return out_yaml

def write_yaml(config_object, yaml_filename):
    """Dumps config object to a yaml file

    Parameters
    ==========
    config_object: `ruamel.yaml.YAML()` object
    yaml_filename: str, output file
    """
    with open(yaml_filename, 'w') as FILE:
        config = yaml.dump(config_object, FILE)


def main():
    """Applies checks, ensures yaml is updated properly"""
    parser = argparse.ArgumentParser(
        description="Automated YAML checker/updater",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "files",
        type=str,
        nargs="+",
        help="YAML files to check/update"
    )
    parser.add_argument(
        "--check",
        "-c",
        action="store_true",
        default=False,
        help="check input yaml",
    )
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        default=False,
        help="overwrite input yaml file(s)",
    )
    parser.add_argument(
        "--roundtrip",
        "-rt",
        action="store_true",
        default=False,
        help="read/write input yaml file(s)",
    )
    parser.add_argument(
        "--addnoise",
        action="store_true",
        default=False,
        help="add noise block to input yaml file(s)",
    )
    parser.add_argument(
        "--addoutlier",
        action="store_true",
        default=False,
        help="add outlier block to input yaml file(s)",
    )
    parser.add_argument(
        "--addresults",
        action="store_true",
        default=False,
        help="add results block to input yaml file(s)",
    )
    parser.add_argument(
        "--addcheck",
        action="store_true",
        default=False,
        help="add check block to input yaml file(s)",
    )
    parser.add_argument(
        "--checkcleared",
        action="store_true",
        default=False,
        help="assign cleared: true if check block is empty",
    )
    parser.add_argument(
        "--bkv",
        nargs=3,
        help="add block/key/value (3 items) to instantiate new yaml field",
    )
    args = parser.parse_args()

    # maybe we want a "initiate" function to make a new yaml?
    if args.check:
        for ff in args.files:
            log.setLevel('DEBUG')
            fix_toa_info(ff,overwrite=args.overwrite)
            add_niterations(ff,overwrite=args.overwrite)
            add_noise_block(ff,overwrite=args.overwrite)
            add_dmx_block(ff,overwrite=args.overwrite)
            #curate_comments(ff,overwrite=args.overwrite)
            # + outlier block?
            # + check block?
    elif args.roundtrip:
        for ff in args.files:
            config = read_yaml(ff)
            outfile = get_outfile(ff,overwrite=args.overwrite,extension='rt')
            write_yaml(config,outfile)
    elif args.addnoise:
        for ff in args.files:
            add_noise_block(ff,overwrite=args.overwrite)
    elif args.addoutlier:
        for ff in args.files:
            add_outlier_block(ff,overwrite=args.overwrite)
    elif args.addresults:
        for ff in args.files:
            add_results_block(ff,overwrite=args.overwrite)
    elif args.addcheck:
        for ff in args.files:
            add_check_block(ff,overwrite=args.overwrite)
    elif args.checkcleared:
        for ff in args.files:
            check_cleared(ff,overwrite=args.overwrite)
    if args.bkv:
        block, key, value = args.bkv
        for ff in args.files:
            add_block_field(ff,block,key,value,overwrite=args.overwrite)

if __name__ == "__main__":
    main()
