import sys
import numpy as np
import astropy.units as u
from astropy import log
import logging
import matplotlib.pyplot as plt
import time
import warnings
from datetime import datetime
from datetime import date
import yaml
import os
import timing_analysis.par_checker as pc

# Read tim/par files
import pint.toa as toa
import pint.models as models
import pint.residuals
from pint.modelutils import model_equatorial_to_ecliptic

from pint.models.parameter import maskParameter
from pint.models.timing_model import Component

def write_par(fitter,toatype='',addext='',outfile=None):
    """Writes a timing model object to a par file in the working directory.

    Parameters
    ==========
    fitter: `pint.fitter` object
    toatype: str, optional
        if set, adds nb/wb.par
    addext: str, optional
        if set, adds extension to date
    outfile: str, optional
        if set, overrides default naming convention
    """
    if outfile is None:
        source = fitter.get_allparams()['PSR'].value
        date_str = date.today().strftime('%Y%m%d')
        if toatype:
            outfile = f'{source}_PINT_{date_str}{addext}.{toatype.lower()}.par'
        else:
            outfile = f'{source}_PINT_{date_str}{addext}.par'

    with open(outfile, 'w') as fout:
        fout.write(fitter.model.as_parfile())

def write_tim(fitter,toatype='',addext='',outfile=None):
    """Writes TOAs to a tim file in the working directory.

    Parameters
    ==========
    fitter: `pint.fitter` object
    toatype: str, optional
        if set, adds nb/wb.par
    addext: str, optional
        if set, adds extension to date
    outfile: str, optional
        if set, overrides default naming convention
    """
    if outfile is None:
        source = fitter.get_allparams()['PSR'].value
        date_str = date.today().strftime('%Y%m%d')
        if toatype:
            outfile = f'{source}_PINT_{date_str}{addext}.{toatype.lower()}.tim'
        else:
            outfile = f'{source}_PINT_{date_str}{addext}.tim'

    fitter.toas.write_TOA_file(outfile, format='tempo2')
    add_cut_tims(outfile)

def add_cut_tims(timfile):
    """Temporary cludge to add cut TOAs back into output tim file (with cut flags).

    Parameters
    ==========
    timfile: str
        tim file to extend with commented TOAs including -cut flags
    """
    from glob import glob
    import subprocess

    cut_tims = glob('*cut.tim')
    commented_lines = []
    for ct in cut_tims:
        with open(ct,'r') as f: commented_lines.extend(f.readlines())
    
    commented_lines = [cl for cl in commented_lines if '-cut' in cl]  # remove unnecessary FORMAT lines

    with open(timfile,'a+') as f:
        for line in commented_lines:
            f.write(f'C {line}')

    for f in cut_tims:
        subprocess.run(['rm',f])

def write_include_tim(source,tim_file_list):
    """Writes file listing tim files to load as one PINT toa object (using INCLUDE).

    Parameters
    ==========
    source: string
        pulsar name
    tim_file_list: list
        tim files to include

    Returns
    =======
    out_tim: tim filename string
    """
    out_tim = '%s.tim' % (source)
    f = open(out_tim,'w')

    for tf in tim_file_list:
        f.write('INCLUDE %s\n' % (tf))

    f.close()
    return out_tim

def center_epochs(model,toas):
    """Center PEPOCH (POSEPOCH, DMEPOCH) using min/max TOA values.

    Parameters
    ==========
    model: `pint.model.TimingModel` object
    toas: `pint.toa.TOAs` object

    Returns
    =======
    model: `pint.model.TimingModel` object
        with centered epoch(s)
    """
    midmjd=np.round((toas.get_mjds().value.max()+toas.get_mjds().value.min())/2.)
    model.change_pepoch(midmjd)

    if model.DMEPOCH.value is None:
        model.DMEPOCH.quantity = midmjd
    else:
        model.change_dmepoch(midmjd)

    if model.POSEPOCH.value is None:
        model.POSEPOCH.quantity = midmjd
    else:
        model.change_posepoch(midmjd)

    if hasattr(model, "TASC") or hasattr(model, "T0"):
        model.change_binary_epoch(midmjd)

    return model

def check_fit(fitter,skip_check=None):
    """Check that pertinent parameters are unfrozen.

    Note: process of doing this robustly for binary models is not yet automated. Checks are
    functions from par_checker.py.

    Parameters
    ==========
    fitter: `pint.fitter` object
    skip_check: list of checks to be skipped (examples: 'spin'; 'spin,astrometry')
                can be a list object or a string with comma-separated values
    """
    if skip_check:
        if type(skip_check)==str:
            skiplist = skip_check.split(',')
        else:
            skiplist = skip_check
    else:
        skiplist = []

    if 'spin' in skiplist:
        log.info("Skipping spin parameter check")
    else:
        pc.check_spin(fitter.model)

    if 'astrometry' in skiplist:
        log.info("Skipping astrometry parameter check")
    else:
        pc.check_astrometry(fitter.model)

def add_feJumps(mo,rcvrs):
    """Automatically add appropriate jumps based on receivers present

    Parameters
    ==========
    mo: `pint.model.TimingModel` object
    rcvrs: list
        receivers present in TOAs
    """
    # Might want a warning here if no jumps are necessary.
    if len(rcvrs) <= 1:
        return

    if not 'PhaseJump' in mo.components.keys():
        log.info("No frontends JUMPed.")
        log.info(f"Adding frontend JUMP {rcvrs[0]}")
        all_components = Component.component_types
        phase_jump_instance = all_components['PhaseJump']()
        mo.add_component(phase_jump_instance)

        mo.JUMP1.key = '-fe'
        mo.JUMP1.key_value = [rcvrs[0]]
        mo.JUMP1.value = 0.0
        mo.JUMP1.frozen = False

    phasejump = mo.components['PhaseJump']
    all_jumps = phasejump.get_jump_param_objects()
    jump_rcvrs = [x.key_value[0] for x in all_jumps if x.key == '-fe']
    missing_fe_jumps = list(set(rcvrs) - set(jump_rcvrs))

    if len(missing_fe_jumps):
        if len(missing_fe_jumps) == 1:
            log.info('Exactly one frontend not JUMPed.')
        else:
            log.info(f"Frontends not JUMPed: {missing_fe_jumps}...")
    else:
        log.warning("All frontends are JUMPed. One JUMP should be removed from the .par file.")
    if len(missing_fe_jumps) > 1:
        for j in missing_fe_jumps[:-1]:
            log.info(f"Adding frontend JUMP {j}")
            JUMPn = maskParameter('JUMP',key='-fe',key_value=[j],value=0.0,units=u.second)
            phasejump.add_param(JUMPn,setup=True)

def add_feDMJumps(mo,rcvrs):
    """Automatically add appropriate dmjumps based on receivers present

    Parameters
    ==========
    mo: `pint.model.TimingModel` object
    rcvrs: list
        receivers present in TOAs
    """

    if not 'DispersionJump' in mo.components.keys():
        log.info("No frontends DMJUMPed.")
        log.info(f"Adding frontend DMJUMP {rcvrs[0]}")
        all_components = Component.component_types
        dmjump_instance = all_components['DispersionJump']()
        mo.add_component(dmjump_instance)

        mo.DMJUMP1.key = '-fe'
        mo.DMJUMP1.key_value = [rcvrs[0]]
        mo.DMJUMP1.value = 0.0
        mo.DMJUMP1.frozen = False

    dmjump = mo.components['DispersionJump']
    all_dmjumps = [getattr(dmjump, param) for param in dmjump.params]
    dmjump_rcvrs = [x.key_value[0] for x in all_dmjumps if x.key == '-fe']
    missing_fe_dmjumps = list(set(rcvrs) - set(dmjump_rcvrs))

    if len(missing_fe_dmjumps):
        log.info(f"Frontends not DMJUMPed: {missing_fe_dmjumps}")
    else:
        log.info(f"All frontends are DMJUMPed.")
    if len(missing_fe_dmjumps):
        for j in missing_fe_dmjumps:
            log.info(f"Adding frontend DMJUMP {j}")
            DMJUMPn = maskParameter('DMJUMP',key='-fe',key_value=[j],value=0.0,units=u.pc*u.cm**-3)
            dmjump.add_param(DMJUMPn,setup=True)

def large_residuals(fo,threshold_us,threshold_dm=None,*,n_sigma=None,max_sigma=None,prefit=False,ignore_ASP_dms=True,print_bad=True):
    """Quick and dirty routine to find outlier residuals based on some threshold.
    Automatically deals with Wideband vs. Narrowband fitters.

    Parameters
    ==========
    fo: `pint.fitter` object
    threshold_us: float
        not a quantity, but threshold for TOA residuals larger (magnitude) than some delay in microseconds; if None, will not look at TOA residuals
    threshold_dm: float
        not a quantity, but threshold for DM residuals larger (magnitude) than some delay in pc cm**-3; if None, will not look at DM residuals
    n_sigma: float or None
        If not None, only discard TOAs and/or DMs that are at least this many sigma as well as large
    max_sigma: float or None
        If not None, also discard all TOAs and/or DMs with claimed uncertainties larger than this many microseconds
    prefit: bool
        If True, will explicitly examine the prefit residual objects in the pinter.fitter object; this will give the same result as when prefit=False but no fit has yet been performed.
    ignore_ASP_dms: bool
        If True, it will not flag/excise any TOAs from ASP or GASP data based on DM criteria
    print_bad: bool
        If True, prints bad-toa lines that can be copied directly into a yaml file

    Returns
    =======
    PINT TOA object of filtered TOAs
    """

    # check if using wideband TOAs, as this changes how to access the residuals

    if fo.is_wideband:
        is_wideband = True
        if prefit:
            time_resids = fo.resids_init.toa.time_resids.to_value(u.us)
            dm_resids = fo.resids_init.dm.resids.value
        else:
            time_resids = fo.resids.toa.time_resids.to_value(u.us)
            dm_resids = fo.resids.dm.resids.value
        dm_errors = fo.toas.get_dm_errors().value
        bes = fo.toas.get_flag_value('be')[0]  # For ignoring G/ASP DMs
        c_dm = np.zeros(len(dm_resids), dtype=bool)
    else:
        is_wideband = False
        if prefit:
            time_resids = fo.resids_init.time_resids.to_value(u.us)
        else:
            time_resids = fo.resids.time_resids.to_value(u.us)
        if threshold_dm is not None:
            log.warning('Thresholding of wideband DM measurements can only be performed with WidebandTOAFitter and wideband TOAs; threshold_dm will be ignored.')
            threshold_dm = None

    toa_errors = fo.toas.get_errors().to_value(u.us)
    c_toa = np.zeros(len(time_resids), dtype=bool)

    if threshold_us is not None:
        c_toa |= np.abs(time_resids) > threshold_us
        if n_sigma is not None:
            c_toa &= np.abs(time_resids/toa_errors) > n_sigma
        if max_sigma is not None:
            c_toa |= toa_errors > max_sigma
    if threshold_dm is not None:
        c_dm |= np.abs(dm_resids) > threshold_dm
        if n_sigma is not None:
            c_dm &= np.abs(dm_resids/dm_errors) > n_sigma
        if max_sigma is not None:
            c_dm |= dm_errors > max_sigma
        if ignore_ASP_dms:
            c_dm &= np.logical_not([be.endswith('ASP') for be in bes])
    if threshold_us is None and threshold_dm is None:
        raise ValueError("You must specify one or both of threshold_us and threshold_dm to be not None.")
    if is_wideband:
        c = c_toa | c_dm
    else:
        c = c_toa

    badlist = np.where(c)
    names = fo.toas.get_flag_value('name')[0]
    chans = fo.toas.get_flag_value('chan')[0]
    subints = fo.toas.get_flag_value('subint')[0]
    for ibad in badlist[0]:
        name = names[ibad]
        chan = chans[ibad]
        subint = subints[ibad]
        if print_bad: print(f"    - ['{name}',{chan},{subint}]")
    mask = ~c
    log.info(f'Selecting {sum(mask)} TOAs of {fo.toas.ntoas} ({sum(c)} removed) based on large_residual() criteria.')
    return fo.toas[mask]

def compare_models(fo,model_to_compare=None,verbosity='check',threshold_sigma=3.,nodmx=True):
    """Wrapper function to compare post-fit results to a user-specified comparison model.

    Parameters
    ==========
    fo: `pint.fitter` object
    model_to_compare: string or Nonetype, optional
        model to compare with the post-fit model
    verbosity: string, optional
        verbosity of output from model.compare
        options are "max", "med", "min", "check". Use ?model.compare for more info.
    threshold_sigma: float, optional
        sigma cutoff for parameter comparison
    nodmx: bool, optional
        when True, omit DMX comparison

    Returns
    =======
    str or None
        returns ascii table when verbosity is not set to "check"; also returns astropy.log statements
    """

    if model_to_compare is not None:
        comparemodel=models.get_model(model_to_compare)
    else:
        comparemodel=fo.model_init
    return comparemodel.compare(fo.model,verbosity=verbosity,nodmx=nodmx,threshold_sigma=threshold_sigma)

def remove_noise(model, noise_components=['ScaleToaError','ScaleDmError',
    'EcorrNoise','PLRedNoise']):
    """Removes noise model components from the input timing model.

    Parameters
    ==========
    model: PINT model object
    noise_components: list of model component names to remove from model
    """
    log.info('Removing pre-existing noise parameters...')
    for component in noise_components:
        if component in model.components:
            log.info(f"Removing {component} from model.")
            model.remove_component(component)
    return

def get_receivers(toas):
    """Returns a list of receivers present in the tim file(s)

    Parameters
    ==========
    toas: `pint.toa.TOAs` object

    Returns
    =======
    receivers: list of strings
        unique set of receivers present in input toas
    """
    receivers = list(set([str(f) for f in set(toas.get_flag_value('fe')[0])]))
    return receivers

def git_config_info():
    """Reports user's git config (name/email) with log.info"""
    gitname = os.popen('git config --get user.name').read().rstrip()
    gitemail = os.popen('git config --get user.email').read().rstrip()
    log.info(f'Your git config user.name is: {gitname}')
    log.info('...to change this, in a terminal: git config user.name "First Last"')
    log.info(f'Your git config user.email is: {gitemail}')
    log.info('...to change this, in a terminal: git config user.email "first.last@nanograv.org"')

def new_changelog_entry(tag, note):
    """Checks for valid tag and auto-generates entry to be copy/pasted into .yaml changelog block.

    Your NANOGrav email (before the @) and the date will be printed automatically. The "tag"
    describes the type of change, and the "note" is a short (git-commit-like) description of
    the change. Entry should be manually appended to .yaml by the user.

    Valid tags:
      - INIT: creation of the .yaml file
      - ADD or REMOVE: adding or removing a parameter
      - BINARY: change in the binary model (e.g. ELL1 -> DD)
      - NOISE: changes in noise parameters, unusual values of note
      - CURATE: notable changes in TOA excision, or adding TOAs
      - NOTE: for anything else
      - TEST: for testing!
    """
    VALID_TAGS = ['INIT','ADD','REMOVE','BINARY','NOISE','CURATE','NOTE','TEST']
    vtstr = ', '.join(VALID_TAGS)
    if tag not in VALID_TAGS:
        log.error(f'{tag} is not a valid tag; valid tags are: {vtstr}.')
    else:
        # Read the git user.email from .gitconfig, return exception if not set
        stream = os.popen('git config --get user.email')
        username = stream.read().rstrip().split('@')[0]

        if not username:
            log.error('Update your git config with... git config --global user.email \"your.email@nanograv.org\"')
        else:
            # Date in YYYY-MM-DD format
            now = datetime.now()
            date = now.strftime('%Y-%m-%d')
            print(f'  - \'{date} {username} {tag}: {note}\'')

def log_notebook_to_file(source, toa_type, base_dir="."):
    """Activate logging to an autogenerated file name.

    This removes all but the first log handler, so it may behave surprisingly 
    if run multiple times not from a notebook.
    """

    if len(log.handlers)>1:
        # log.handlers[0] is the notebook output
        for h in log.handlers[1:]:
            log.removeHandler(h)
        # Start a new log file every time you reload the yaml
    log_file_name = os.path.join(
            base_dir, 
            f"{source}.{toa_type.lower()}.{time.strftime('%Y-%m-%d_%H:%M:%S')}.log")
    fh = logging.FileHandler(log_file_name)
    fh.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    log.addHandler(fh)


_showwarning_orig = None
def _showwarning(*args, **kwargs):
    warning = args[0]
    message = str(args[0])
    mod_path = args[2]
    # Now that we have the module's path, we look through sys.modules to
    # find the module object and thus the fully-package-specified module
    # name.  The module.__file__ is the original source file name.
    mod_name = None
    mod_path, ext = os.path.splitext(mod_path)
    for name, mod in list(sys.modules.items()):
        try:
            # Believe it or not this can fail in some cases:
            # https://github.com/astropy/astropy/issues/2671
            path = os.path.splitext(getattr(mod, '__file__', ''))[0]
        except Exception:
            continue
        if path == mod_path:
            mod_name = mod.__name__
            break
    if mod_name is not None:
        log.warning(message, extra={'origin': mod_name})
    else:
        log.warning(message)



def log_warnings():
    """Route warnings through the Astropy log mechanism.

    Astropy claims to do this but only for warnings that are subclasses of AstropyUserWarning.
    See https://github.com/astropy/astropy/issues/11500 ; if resolved there this can be simpler.
    """
    global _showwarning_orig
    if _showwarning_orig is None:
        _showwarning_orig = warnings.showwarning
        warnings.showwarning = _showwarning
