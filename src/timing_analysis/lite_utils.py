import sys
import numpy as np
import astropy.units as u
from astropy import log
from astropy.io import fits
import logging
import matplotlib.pyplot as plt
import time
import warnings
from datetime import datetime
from datetime import date
import yaml
import os
import timing_analysis.par_checker as pc
from ipywidgets import widgets
import pypulse
import glob

# Read tim/par files
import pint.toa as toa
import pint.models as models
import pint.residuals
from pint.modelutils import model_equatorial_to_ecliptic

from pint.models.parameter import maskParameter
from pint.models.timing_model import Component

def convert_pint_to_tempo_timfile(tim_path, op_path, psr_name = "test", timing_pkg = 'tempo'):
    """
    Function to convert PINT produced timfile to tempo-compatible parfile.
    Add "MODE 1" to tim file to use uncertainties
    Change "arecibo" to "ao" if it is an Arecibo pulsar
    -----------------------
    
    Input:
    path_to_tim: path to timfile
    op_path: path to write modified timfile
    psr_name: Name of pulsar; used for writing op filename;
              can obtain from `convert_pint_to_tempo_parfile()` function.
    
    Returns:
    None
    
    Example usage:
    >>> test_tim = "./J1903+0327_PINT_20210925.nb.tim"
    >>> psr_name = "J1903+0327"
    >>> convert_pint_to_tempo_timfile(test_tim, op_path = "./", psr_name = psr_name)
    """
    
    with open(tim_path) as ff:
        tim_lines = ff.readlines()
        
    new_tim = []

    for ii in range(len(tim_lines)):

        if ii == 0:
            if timing_pkg == 'tempo':
                new_tim.append("MODE 1\n")    
            
            new_tim.append(tim_lines[ii])

        else:
            entries = tim_lines[ii].split(" ")
            if ("arecibo" not in entries) or (timing_pkg == 'tempo2'):
                new_tim.append(tim_lines[ii])
            else:
                ent_arr = np.array(entries)

                idx = np.where(ent_arr == 'arecibo')[0][0]
                entries[idx] = 'ao'

                new_entry = " ".join(entries)
                new_tim.append(new_entry)
                
    with open(op_path + '/' + psr_name + '.tim', 'w') as oo:
        for ll in new_tim:
            oo.write(ll)
            
def convert_pint_to_tempo_parfile(path_to_par, op_path = "./", timing_pkg = 'tempo'):
    """
    Function to convert PINT produced parfile to tempo-compatible parfile.
    Removes CHI2 and SWM from the parfile.
    Changes EFAC/EQUAD to T2EFAC/T2EQUAD.
    If converting to tempo2 parfile, make sure ECL IERS2003 is set
    -----------------------
    
    Input:
    path_to_par: path to parfile
    op_path: path to write modified parfile
    timing_pkg: options: tempo, tempo2; if tempo2, sets ECL IERS2003.
    
    Returns:
    psr_name: Name of pulsar (convenience for tim file conversion)
    
    Example usage:
    >>> test_par = "results/J1903+0327_PINT_20210925.nb.par"
    >>> psr_name = convert_pint_to_tempo_parfile(test_par, op_path = "./", timing_pkg = 'tempo2')
    >>> print(psr_name)
     'J1903+0327'
    """
    
    with open(path_to_par, 'r') as ff:
    
        par_lines = ff.readlines()
    
    new_par = []

    for ii in range(len(par_lines)):

        entries = par_lines[ii].split(" ")
        
        if ii == 0:
            if timing_pkg == 'tempo2':
                new_par.append("MODE 1\n")
                
        if 'PSR' in entries:
            psr_name = entries[-1].split('\n')[0]
        
        if "CHI2" in entries:
            continue

        elif "SWM" in entries:
            continue
            
        elif "A1DOT" in entries:
            entries[0] = "XDOT"
            
            new_entry = " ".join(entries)
            
            new_par.append(new_entry)
            
        elif "STIGMA" in entries:
            entries[0] = "VARSIGMA"
            
            new_entry = " ".join(entries)
            
            new_par.append(new_entry)
            
        elif "NHARMS" in entries:
            entries[-1] = str(int(float(entries[-1]))) + '\n'
            
            new_entry = " ".join(entries)
            
            new_par.append(new_entry)
            
        elif ("ECL" in entries) and (timing_pkg == 'tempo2'):
            entries[-1] = "IERS2003\n"
            
            new_entry = " ".join(entries)
            
            new_par.append(new_entry)

        elif "EFAC" in entries:
            entries[0] = "T2EFAC"

            new_entry = " ".join(entries)

            new_par.append(new_entry)

        elif "EQUAD" in entries:
            entries[0] = "T2EQUAD"

            new_entry = " ".join(entries)

            new_par.append(new_entry)
            
        elif ("T2CMETHOD" in entries) and (timing_pkg == 'tempo2'):
            entries[0] = "#T2CMETHOD"
            
            new_entry = " ".join(entries)
            
            new_par.append(new_entry)

        else:
            new_par.append(par_lines[ii])
            
    with open(op_path + '/' + psr_name + '.par', 'w') as oo:
        for ll in new_par:
            oo.write(ll)
            
    return psr_name

def write_par(fitter,toatype='',addext='',outfile=None, fmt=None):
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
    fmt: str, optional
        if set, writes a tempo/tempo2-friendly par file
    """
    if fmt is None:
        fmt = 'PINT'
    # Error if fmt is not supported (tempo/tempo2)?

    if outfile is None:
        source = fitter.get_allparams()['PSR'].value
        date_str = date.today().strftime('%Y%m%d')
        if toatype:
            outfile = f'{source}_{fmt}_{date_str}{addext}.{toatype.lower()}.par'
        else:
            outfile = f'{source}_{fmt}_{date_str}{addext}.par'

    with open(outfile, 'w') as fout:
        if fmt == 'PINT':
            fout.write(fitter.model.as_parfile())
        else:
            fout.write(fitter.model.as_parfile(format=fmt))

def write_tim(fitter,toatype='',addext='',outfile=None,commentflag=None):
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
    commentflag: str or None, optional
        if a string, and that string is a TOA flag,
        that TOA will be commented in the output file;
        if None (or non-string), no TOAs will be commented.
    """
    if outfile is None:
        source = fitter.get_allparams()['PSR'].value
        date_str = date.today().strftime('%Y%m%d')
        if toatype:
            outfile = f'{source}_PINT_{date_str}{addext}.{toatype.lower()}.tim'
        else:
            outfile = f'{source}_PINT_{date_str}{addext}.tim'

    
    fitter.toas.write_TOA_file(outfile, format='tempo2',commentflag=commentflag)

def find_excise_file(outfile_basename,intermediate_results='/nanograv/share/15yr/timing/intermediate/'):
    """Writes TOAs to a tim file in the working directory.

    Parameters
    ==========
    outfile_basename: str
        e.g. J1234+5678.nb, use tc.get_outfile_basename()
    intermediate_results: str, optional
        base directory where intermediate results are stored
    """
    outlier_dir = os.path.join(intermediate_results,'outlier',outfile_basename)
    excise_file_only = f'{outfile_basename}_excise.tim'
    excise_file = os.path.join(outlier_dir,excise_file_only)
    noc_file = excise_file_only.replace('.tim','-noC.tim')

    # Check for existence of excise file, return filename (else, None)
    if os.path.exists(excise_file):
        # Check for 'C ' instances
        with open(excise_file,'r') as fi:
            timlines = fi.readlines()
            Ncut = 0
            for i in range(len(timlines)):
                if timlines[i].startswith('C '):
                    timlines[i] = timlines[i].lstrip('C ')
                    Ncut += 1

        # If any, remove them and write noc_file to read, else read the existing file
        if Ncut:
            log.info(f"Removing {Ncut} instances of 'C ', writing {noc_file}.")
            with open(noc_file,'w') as fo:
                fo.writelines(timlines)
            excise_file = noc_file
        else:
            pass

        return excise_file

    else:
        log.warning(f'Excise file does not exist: {excise_file}')
        return None 

def write_include_tim(source,tim_file_list):
    """Writes file listing tim files to load as one PINT toa object (using INCLUDE).
       DEPRECATED...?

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
            
def get_flag_val_list(toas, flag):
    """Returns a list of receivers present in the tim file(s)

    Parameters
    ==========
    toas: `pint.toa.TOAs` object
    flag: string
          Name of the flag you're using for jumps

    Returns
    =======
    flaglist: list of strings
        unique value of whatever the flag was in input toas, designated using 'flag'. This is for jumps
    """
    flaglist = list(set([str(f) for f in set(toas.get_flag_value(flag)[0])]))
    if 'None' in flaglist: flaglist.remove('None')
    if 'unknown' in flaglist: flaglist.remove('unknown')
    return flaglist


def add_flag_jumps(mo,flag,flaglist,base=False):
    """Automatically add appropriate jumps based on jump flag present

    Parameters
    ==========
    mo: `pint.model.TimingModel` object
    flag: string
        the name of the flag you're jumping on (e.g. 'fe' or 'f')
    flaglist: list
        list of values of that flag in this dataset.
    base: bool
          Is this the flag you're using as your "one not jumped" category?
    """
    flagval = '-' + flag
    # Might want a warning here if no jumps are necessary.
    if len(flaglist) <= 1:
        return

    if not 'PhaseJump' in mo.components.keys():
        log.info("No JUMPed.")
        log.info(f"Adding flag JUMPs {flaglist[0]}")
        all_components = Component.component_types
        phase_jump_instance = all_components['PhaseJump']()
        mo.add_component(phase_jump_instance)

        mo.JUMP1.key = '-' + flag
        mo.JUMP1.key_value = [flaglist[0]]
        mo.JUMP1.value = 0.0
        mo.JUMP1.frozen = False

    phasejump = mo.components['PhaseJump']
    all_jumps = phasejump.get_jump_param_objects()
    jump_flag = [x.key_value[0] for x in all_jumps if x.key == flagval]
    missing_jumps = list(set(flaglist) - set(jump_flag))

    if base == True:
        if len(missing_jumps):
            if len(missing_jumps) == 1:
                log.info('Exactly one' + flag + ' not JUMPed.')
            else:
                log.info(flag + f" not JUMPed: {missing_jumps}...")
        else:
            log.warning("All " + flag + " are JUMPed. One JUMP should be removed from the .par file if this is your base flag.")
        if len(missing_jumps) > 1:
            for j in missing_jumps[:-1]:
                log.info(f"Adding frontend JUMP {j}")
                JUMPn = maskParameter('JUMP',key=flagval,key_value=[j],value=0.0,units=u.second)
                phasejump.add_param(JUMPn,setup=True)
    else:
        if len(missing_jumps):
            if len(missing_jumps) == 0:
                log.info('All' + flag + ' JUMPed.')
            else:
                log.info(flag + f" not JUMPed: {missing_jumps}...")
        else:
            log.warning("All " + flag + " are JUMPed. One JUMP should be removed from the .par file if this is your base flag.")
        if len(missing_jumps) >= 1:
            for j in missing_jumps[:-1]:
                log.info(f"Adding frontend JUMP {j}")
                JUMPn = maskParameter('JUMP',key=flagval,key_value=[j],value=0.0,units=u.second)
                phasejump.add_param(JUMPn,setup=True)

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
        if print_bad: print(f"  - [{name}, {chan}, {subint}]")
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

def get_receivers_epta(toas):
    """Returns a list of receivers present in the tim file(s)

    Parameters
    ==========
    toas: `pint.toa.TOAs` object

    Returns
    =======
    receivers: list of strings
        unique set of receivers present in input toas
    """
    receivers_r= list(set([str(f) for f in set(toas.get_flag_value('r')[0])]))
    return receivers_r

    
def get_receivers_ipta(toas):
    """Returns a list of receivers present in the tim file(s)

    Parameters
    ==========
    toas: `pint.toa.TOAs` object

    Returns
    =======
    receivers: list of strings
        unique set of receivers present in input toas
    """
    receivers_r= list(set([str(f) for f in set(toas.get_flag_value('r')[0])]))
    receivers_fe = list(set([str(f) for f in set(toas.get_flag_value('fe')[0])]))
    receivers_all = receivers_r + receivers_fe
    receivers = receivers_all.remove('None')
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
      - READY_FOR: indicate state of completion for release version
      - ADD or REMOVE: adding or removing a parameter
      - BINARY: change in the binary model (e.g. ELL1 -> DD)
      - NOISE: changes in noise parameters, unusual values of note
      - CURATE: notable changes in TOA excision, or adding TOAs
      - NOTE: for anything else
      - TEST: for testing!
    """
    VALID_TAGS = ['INIT','READY_FOR','ADD','REMOVE','BINARY','NOISE','CURATE','NOTE','TEST']
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
            f"{source}.{toa_type.lower()}.{time.strftime('%Y-%m-%d_%H%M%S')}.log")
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

def get_cut_colors(palette='pastel'):
    """Get dictionary mapping cut flags to colors
    
    Parameters
    ==========
    palette: str
        Seaborn color palette name (default "pastel")
    
    Returns
    =======
    color_dict: dict
        Dictionary mapping cut flags to colors in the specified palette
    """
    import seaborn as sns
    palette = sns.color_palette(palette, 11)
    color_dict = {
        'good':palette[2],
        'dmx':palette[0],
        'snr':palette[1],
        'badrange':palette[3],
        'outlier10':palette[4],
        'epochdrop':palette[5],
        'orphaned':palette[6],
        'maxout':palette[7],
        'simul':palette[8],
        'poorfebe':palette[9],
        'eclipsing':palette[10],
        'badtoa':'k',
        'badfile':'C3',
        'mjdstart':'C9',
        'mjdend': 'C9',
    }
    return color_dict

def get_cutsDict(toas):
    """Gather useful information about cuts by telescope, febe, flag value

    Parameters
    ==========
    toas: `pint.toa.TOAs` object

    Returns
    =======
    cutsDict: dict
        cuts and total number of TOAs per febe, telescope, cut flag
    """
    cuts = np.array([f['cut'] if 'cut' in f else None for f in toas.orig_table['flags']])
    set_cuts = set(cuts)
    tottoa = len(cuts)

    tels = np.array([t[5] for t in toas.orig_table])
    set_tels = set(tels)

    fbs = np.array([f['f'] for f in toas.orig_table['flags']])
    set_fbs = set(fbs)

    fcutDict = {}
    for fb in set_fbs:
        inds = np.where(fbs == fb)[0]
        ntot = len(inds)
        ncuts = sum(x is not None for x in cuts[inds])
        fcutDict[fb] = [ntot,ncuts]

    tcutDict = {}
    for tel in set_tels:
        inds = np.where(tels == tel)[0]
        ntot = len(inds)
        ncuts = sum(x is not None for x in cuts[inds])
        tcutDict[tel] = [ntot,ncuts]
    
    ncut_manual = 0
    ccutDict = {}
    for c in set_cuts:
        ncut = list(cuts).count(c)
        if c: ccutDict[c] = [tottoa,ncut]
        else: ccutDict['good'] = [tottoa,ncut]
    if ncut_manual: ccutDict['manual'] = [tottoa,ncut_manual]
        
    cutsDict = {'f':fcutDict,
                'tel':tcutDict,
                'cut':ccutDict,
               }
    return cutsDict

def cut_summary(toas, tc, print_summary=False, donut=True, legend=True, save=False):
    """Basic summary of cut TOAs, associated reasons

    Parameters
    ==========
    toas: `pint.toa.TOAs` object
    tc: `timing_analysis.timingconfiguration.TimingConfiguration` object
    print_summary: bool, optional
        Print reasons for cuts and respective nTOA/percentages
    donut: bool, optional
        Make a donut chart showing reasons/percentages for cuts
    legend: bool, optional
        Include a legend rather than labeling slices
    save: bool, optional
        Save a png of the resulting plot.

    Returns
    =======
    cutsDict: dict
        Number total/cut TOAs per telescope, febe, cut flag. 
    """
    color_dict = get_cut_colors()
    cutsDict = get_cutsDict(toas)

    # gather info for title (may also be useful for other features in the future)
    tel = [t[5] for t in toas.table]
    settel = set(tel)

    fe = [str(t[6]['fe']) for t in toas.table]
    setfe = set(fe)

    mashtel = ''.join(settel)
    flavor = f"{tc.get_outfile_basename()} ({mashtel}; {', '.join(setfe)})"

    # kwarg that makes it possible to break this down by telescope/backend...?
    nTOA = len(toas)
    cdc = cutsDict['cut']
    cuts_dict = {}
    for c in cdc.keys():
        ncut = cdc[c][1]
        cuts_dict[c] = ncut
        if print_summary: print(f'{c}: {ncut} ({100*ncut/nTOA:.1f}%)')

    nTOAcut = np.array(list(cuts_dict.values()))
    sizes = nTOAcut/nTOA
    labels = [f"{cdk} ({cuts_dict[cdk]})" for cdk in cuts_dict.keys()]
    colors = [color_dict[cdk] for cdk in cuts_dict.keys()]

    fig1, ax1 = plt.subplots()
    ax1.axis('equal')
    fig1.suptitle(flavor)
    if legend:
        ax1.pie(sizes, colors=colors, autopct='%1.1f%%', pctdistance=0.8, normalize=True)
        ax1.legend(labels,bbox_to_anchor=(0., -0.2, 1., 0.2), loc='lower left',
           ncol=3, mode="expand", borderaxespad=0.)
    else:
        ax1.pie(sizes, autopct='%1.1f%%', labels=labels, pctdistance=0.8, colors=colors, normalize=True)
    if donut:
        donut_hole=plt.Circle( (0,0), 0.6, color='white')
        p=plt.gcf()
        p.gca().add_artist(donut_hole)
    if save:
        plt.savefig(f"{mashtel}_{tc.get_outfile_basename()}_donut.png",bbox_inches='tight')
        plt.close()
    return cutsDict

def plot_cuts_by_backend(toas, backend, marker='o', marker_size=10, palette='pastel', save=False,
                        source_name=None, using_wideband=False):
    """Plot TOAs for a single backend in the frequency-time plane, colored by reason for excision (if any)

    Parameters
    ==========
    toas: `pint.toa.TOAs` object
    backend: str
        Backend for which to make the plot
    marker: str, optional
        Marker to use in scatterplot
    marker_size: int, optional
        Size of markers in scatterplot
    palette: str, optional
        Seaborn color palette name
    save: bool, optional
        Save a png of the plot
    source_name: str, optional
        Explicitly input source name, if desired, for output filename
    using_wideband: bool, optional
        TOAs are WB

    Returns
    =======
    fig: `matplotlib.figure.Figure` object
    ax: `matplotlib.axes._subplots.AxesSubplot` object
        Figure and axes -- can be used to modify plot
    """
    if source_name is not None:
        psr = source_name
    else:
        psr = toas.table[0]['flags']['tmplt'].split('.')[0]
    color_dict = get_cut_colors(palette)

    ntoas_total = sum(1 for t in toas.orig_table if t['flags']['be'] == backend)
    ntoas_cut = sum(1 for t in toas.orig_table if t['flags']['be'] == backend and 'cut' in t['flags'])

    def matches(t, backend, cut_type):
        matches_be = t['flags']['be'] == backend
        if cut_type == 'good':
            matches_cut_type = 'cut' not in t['flags']
        else:
            matches_cut_type = 'cut' in t['flags'] and t['flags']['cut'] == cut_type
        return matches_be and matches_cut_type
    
    fig, ax = plt.subplots(figsize=(9.6, 4.8), constrained_layout=True)

    for cut_type, color in color_dict.items():
        pairs = np.array([(t['tdbld'], t['freq']) for t in toas.orig_table if matches(t, backend, cut_type)])
        if pairs.size > 0:
            mjd, freq = pairs.T
            ax.scatter(mjd, freq, marker=marker, color=color, s=marker_size, label=cut_type)
    ax.set_xlabel('MJD')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title(f'{backend} ({ntoas_total} total TOAs, {ntoas_cut} cut)')
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
    if save:
        if using_wideband:
            plt.savefig(f'{psr}-{backend}-excision_wb.png', dpi=150)
        else:
            plt.savefig(f'{psr}-{backend}-excision_nb.png', dpi=150)
    return fig, ax

def plot_cuts_all_backends(toas, marker='o', marker_size=10, palette='pastel', save=False,
                          source_name=None, using_wideband=False):
    """Plot TOAs for each backend in the frequency-time plane, colored by reason for excision (if any)

    Parameters
    ==========
    toas: `pint.toa.TOAs` object
    using_wideband: bool, optional
        TOAs are WB
    marker: str, optional
        Marker to use in scatterplots
    marker_size: int, optional
        Size of markers in scatterplots
    palette: str, optional
        Seaborn color palette name
    save: bool, optional
        Save a png of each plot
    source_name: str, optional
        Explicitly input source name, if desired, for output filename
    using_wideband: bool, optional
        TOAs are WB
    """
    backends = set(t['flags']['be'] for t in toas.orig_table)
    for backend in backends:
        plot_cuts_by_backend(toas, backend, marker, marker_size, palette, save, using_wideband=using_wideband,
                            source_name=source_name)
    plt.show()
       
def display_excise_dropdowns(file_matches, toa_matches, all_YFp=False, all_GTpd=False, all_profile=False):
    """Displays dropdown boxes from which the files/plot types of interest can be chosen during manual excision. This should be run after tc.get_investigation_files(); doing so will display two lists of dropdowns (separated by bad_toa and bad_file). The user then chooses whatever combinations of files/plot types they'd like to display, and runs a cell below the dropdowns containing the read_excise_dropdowns function.
    
    Parameters
    ==========
    file_matches: a list of *.ff files matching bad files in YAML
    toa_matches: lists with *.ff files matching bad toas in YAML, bad subband #, bad subint #
    all_YFp (optional, default False): if True, defaults all plots to YFp
    all_GTpd (optional, default False): if True, defaults all plots to GTpd
    all_profile (optional, default False): if True, defaults all plots to profile vs. phase
    
    Returns (note: these are separate for now for clarity and freedom to use the subint/subband info in bad-toas)
    =======
    file_dropdowns: list of dropdown widgets containing short file names and file extension dropdowns for bad-files
    pav_file_drop: list of dropdown widget objects indicating plot type to be chosen for bad-files
    toa_dropdowns: list of dropdown widget objects containing short file names and extensions for bad-toas
    pav_toa_drop: list of dropdown widget objects indicating plot type to be chosen for bad-toas
    """
    
    ext_list = ['.ff','None','.calib','.zap']
    if all_YFp:
        pav_list = ['YFp (time vs. phase)','GTpd (frequency vs. phase)','Profile (intensity vs. phase)','None']
    elif all_GTpd:
        pav_list = ['GTpd (frequency vs. phase)','YFp (time vs. phase)','Profile (intensity vs. phase)','None']
    elif all_profile:
        pav_list = ['Profile (intensity vs. phase)','YFp (time vs. phase)','GTpd (frequency vs. phase)','None']
    else:
        pav_list = ['None','YFp (time vs. phase)','GTpd (frequency vs. phase)','Profile (intensity vs. phase)']    
   
    # Files: easy
    short_file_names = [e.split('/')[-1].rpartition('.')[0] for e in file_matches]
    file_dropdowns = [widgets.Dropdown(description=s, style={'description_width': 'initial'},
                                  options=ext_list, layout={'width': 'max-content'}) for s in short_file_names]    
    pav_file_drop = [widgets.Dropdown(options=pav_list) for s in short_file_names]
    file_output = widgets.HBox([widgets.VBox(children=file_dropdowns),widgets.VBox(children=pav_file_drop)])
    if len(file_matches) != 0:
        print('Bad-files in YAML:')
        display(file_output)
    
    # TOAs: difficult, annoying, need to worry about uniqueness
    short_toa_names = [t[0].split('/')[-1].rpartition('.')[0] for t in toa_matches]
    toa_inds = np.unique(short_toa_names, return_index=True)[1] # because np.unique sorts it
    short_toa_unique = [short_toa_names[index] for index in sorted(toa_inds)] # unique
    toa_dropdowns = [widgets.Dropdown(description=s, style={'description_width': 'initial'},
                                  options=ext_list, layout={'width': 'max-content'}) for s in short_toa_unique]
    pav_toa_drop = [widgets.Dropdown(options=pav_list) for s in short_toa_unique] 
    toa_output = widgets.HBox([widgets.VBox(children=toa_dropdowns),widgets.VBox(children=pav_toa_drop)])
    if len(toa_matches) != 0:
        print('Bad-toas in YAML:')
        display(toa_output)
    return file_dropdowns, pav_file_drop, toa_dropdowns, pav_toa_drop

def read_excise_dropdowns(select_list, pav_list, matches):
    """Reads selections for files/plots chosen via dropdown.
    
    Parameters
    ==========
    select_list: list of dropdown widget objects indicating which (if any) file extension was selected for a given matching file
    pav_list: list of dropdown widget objects indicating what type of plot was chosen
    matches: list of full paths to all matching files
    
    Returns
    =======
    plot_list: lists of full paths to files of interest and plot types chosen
    """   
    if len(matches) != 0 and isinstance(matches[0],list): # toa entries
        toa_nm = []
        toa_subband = []
        toa_subint = []
        for i in range(len(matches)):
            toa_nm.append(matches[i][0])
            toa_subband.append(matches[i][1])
            toa_subint.append(matches[i][2])
        toa_unique_ind = np.unique(toa_nm, return_index=True)[1]
        toa_nm_unique = [toa_nm[index] for index in sorted(toa_unique_ind)]
        toa_subband_unique = [toa_subband[index] for index in sorted(toa_unique_ind)]
        toa_subint_unique = [toa_subint[index] for index in sorted(toa_unique_ind)]        
    plot_list = []
    for i in range(len(select_list)):
        if (select_list[i].value != 'None') and (pav_list[i].value != 'None'):
            if isinstance(matches[0], list): # toa entries
                plot_list.append([toa_nm_unique[i].rpartition('/')[0] + '/' + select_list[i].description.split(' ')[0] + select_list[i].value, pav_list[i].value, toa_subband_unique[i], toa_subint_unique[i]])                
            else: # bad-file entries
                plot_list.append([matches[i].rpartition('/')[0] + '/' + select_list[i].description + 
                                  select_list[i].value,pav_list[i].value])
    return plot_list

def make_detective_plots(plot_list, match_list):
    """Makes pypulse plots for selected combinations of file/plot type (pav -YFp or -GTpd style).
    
    Parameters
    ==========
    plot_list: lists of full paths to files of interest and plot types chosen
    match_list: list of full paths to all matching files
    
    Returns
    =======
    None; displays plots in notebook.
    """
    for l in range(len(plot_list)):                
        if len(plot_list[l]) <= 2: # bad file entries
            if plot_list[l][1] == 'YFp (time vs. phase)':
                ar = pypulse.Archive(plot_list[l][0],prepare=True)
                ar.fscrunch()
                ar.imshow()
            elif plot_list[l][1] == 'GTpd (frequency vs. phase)':
                ar = pypulse.Archive(plot_list[l][0],prepare=True)
                ar.tscrunch()
                ar.imshow()
            elif plot_list[l][1] == 'Profile (intensity vs. phase)':
                ar = pypulse.Archive(plot_list[l][0],prepare=True)
                ar.fscrunch()
                ar.tscrunch()
                ar.plot()               
        elif len(plot_list[l]) > 2: # toa entries
            for m in range(len(match_list)):                
                if plot_list[l][0].rpartition('.')[0] in match_list[m][0]:
                    log.info(f'[Subband, subint] from bad-toa entry: [{match_list[m][1]},{match_list[m][2]}]')                    
                    if plot_list[l][1] == 'Profile (intensity vs. phase)':
                        #ar.plot(chan=match_list[m][1], subint=match_list[m][2], pol=0)
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,7))
                        ar = pypulse.Archive(plot_list[l][0], prepare=True)
                        ar.plot(subint=match_list[m][2], pol=0, chan=match_list[m][1], ax=ax1, show=False)
                        ar.fscrunch()
                        ar.plot(subint=0, pol=0, chan=0, ax=ax2, show=False)
                        plt.show()
                    elif plot_list[l][1] == 'GTpd (frequency vs. phase)':
                        ar = pypulse.Archive(plot_list[l][0],prepare=True)
                        ar.tscrunch()
                        ar.imshow()
                    elif plot_list[l][1] == 'YFp (time vs. phase)':
                        ar = pypulse.Archive(plot_list[l][0],prepare=True)
                        ar.fscrunch()
                        ar.imshow()
        
        
def display_cal_dropdowns(file_matches, toa_matches):
    """ Display dropdowns for all cal files that are associated with either bad_file or bad_toa entries
    
    Parameters
    ==========
    file_matches: a list of *.ff files matching bad files in YAML
    toa_matches: lists with *.ff files matching bad toas in YAML, bad subband #, bad subint #
    """
    toa_cal_list = [i[0] for i in toa_matches]
    cal_matches = file_matches + toa_cal_list
    cal_matches_inds = np.unique(cal_matches, return_index=True)[1] # because np.unique sorts it
    cal_matches_unique = [cal_matches[index] for index in sorted(cal_matches_inds)] # unique
    cal_stem = [c.rpartition('/')[0] for c in cal_matches_unique]
    full_cal_files = []
    for c,s in zip(cal_matches_unique,cal_stem):
        hdu = fits.open(c)
        data = hdu[1].data
        hdu.close()
        calfile = data['CAL_FILE']
        full_cal_files.append(s + '/' + calfile[-1].split(' ')[-1])
    cal_plot_types = ['None','Amplitude vs. freq.','Single-axis cal sol\'n vs. freq. (pacv)','On-pulse Stokes vs. freq. (pacv -csu)']
    cal_dropdowns = [widgets.Dropdown(description=c.rpartition('/')[-1], style={'description_width': 'initial'}, options=cal_plot_types, layout={'width': 'max-content'}) for c in cal_matches_unique]
    cal_output = widgets.HBox([widgets.VBox(children=cal_dropdowns)])
    display(cal_output)
    return cal_dropdowns, full_cal_files
    
def read_plot_cal_dropdowns(cal_select_list, full_cal_files):
    """Reads selections for files/plots chosen via dropdown.
    
    Parameters
    ==========
    cal_select_list: list of dropdown widget objects indicating which (if any) cal was selected
    full_cal_files: list of all full paths to cal files
    
    Returns
    =======
    None; displays plots in notebook
    """   
    for c,f in zip(cal_select_list,full_cal_files):
        if c.value != 'None':
            if os.path.isfile(f):
                log.info(f'Making cal plot corresponding to {c.description}')
                cal_archive = pypulse.Archive(f)
                cal = cal_archive.getPulsarCalibrator()
                if c.value == 'Amplitude vs. freq.':
                    cal.plot("AB")
                elif c.value == 'Single-axis cal sol\'n vs. freq. (pacv)':
                    cal.pacv()
                elif c.value == 'On-pulse Stokes vs. freq. (pacv -csu)':
                    cal.pacv_csu()
            else:
                warn = f.rpartition('/')[-1]
                log.warning(f'{warn}: This .cf file doesn\'t seem to exist!')

def display_auto_ex(tc, mo, cutkeys=['epochdrop', 'outlier10'], plot_type='profile'):
    """Displays profiles, freq vs. phase, or time vs. phase for TOAs with cut flags.
    
    Parameters
    ==========
    tc: `timing_analysis.timingconfiguration.TimingConfiguration` object
    mo: `pint.model.TimingModel` object
    cutkeys: any valid -cut keys (default = ['epochdrop', 'outlier10'])
    plot_type: str specifying plot type (profile [default], GTpd, or YFp)
    
    Returns
    =======
    None; displays plots in notebook
    """   
    # Want to start from the excise file to assure all cuts are found regardless of notebook settings
    t = pint.toa.get_TOAs(tc.get_excised(), model=mo)
    log.info(f'Displaying plots for the following cut flags: {cutkeys}')            
    ff_list = sorted(glob.glob('/nanograv/timing/releases/15y/toagen/data/*/*/*.ff'))
    cuts = t.get_flag_value('cut')
    files = t.get_flag_value('name')
    chans = t.get_flag_value('chan')
    subints = t.get_flag_value('subint')
    match_inds = []
    for c in cutkeys:
        match_inds.append(np.where(np.array(cuts[0]) == c)[0])
    merge_matches = []
    for i in range(len(match_inds)):
        for ii in match_inds[i]:
            merge_matches.append(ii)
    plot_list = []
    for i in merge_matches:
        match = [m for m in ff_list if files[0][i] in m]
        plot_list.append([cuts[0][i], files[0][i], match[0], chans[0][i], subints[0][i]])   
    if plot_type == 'profile':
        for p in plot_list:
            log.info(f'Plotting profile for [chan, subint] = [{p[3]}, {p[4]}]')
            log.info(f'Cut flag for this TOA: {p[0]}')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,7))
            ar = pypulse.Archive(p[2], prepare=True)
            ar.plot(subint=int(p[4]), pol=0, chan=int(p[3]), ax=ax1, show=False)
            ar.fscrunch()
            ar.plot(subint=0, pol=0, chan=0, ax=ax2, show=False)
            plt.show()
    else:
        set_matches = set([p[2] for p in plot_list])
        for m in set_matches:
            ar = pypulse.Archive(m,prepare=True)
            if plot_type == 'GTpd':
                ar.tscrunch()
                ar.imshow()
            elif plot_type == 'YFp':
                ar.fscrunch()
                ar.imshow()

    return plot_list

def highlight_cut_resids(toas,model,tc_object,cuts=['badtoa','badfile'],multi=False,ylim_good=True,save=True):
    """ Plot residuals vs. time, highlight specified cuts (default: badtoa/badfile) 
    
    Parameters
    ==========
    toas: `pint.toa.TOAs` object 
    model: `pint.model.TimingModel` object 
    tc_object: `timing_analysis.timingconfiguration` object
    cuts: list, optional
        cuts to highlight in residuals plot (default: manual cuts)
    multi: bool, optional
        plot both full ylim and zoom-in, supercedes ylim_good (default: False)
    ylim_good: bool, optional
        set ylim to that of uncut TOAs (default: True)
    save: bool (default: True)
        saves the output plot
    """
    toas.table = toas.orig_table
    fo = tc_object.construct_fitter(toas,model)
    using_wideband = tc_object.get_toa_type() == 'WB'

    # get resids/errors/mjds
    if using_wideband: time_resids = fo.resids_init.residual_objs['toa'].time_resids.to_value(u.us)
    else: time_resids = fo.resids_init.time_resids.to_value(u.us)
    errs = fo.toas.get_errors().to(u.us).value
    mjds = fo.toas.get_mjds().value

    if multi:
        figsize = (12,6.5)
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        axes = [ax1, ax2]
    else:    
        figsize = (12,3)
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
        axes = [ax1]
    
    import seaborn as sns
    valid_cuts = ['snr','simul','orphaned','maxout','outlier10','dmx','epochdrop','badfile','badtoa','badrange','poorfebe']
    sns.color_palette()

    # find appropriate indices & plot remaining TOAs
    toa_cut_flags = np.array([t['flags']['cut'] if 'cut' in t['flags'] else None for t in toas.orig_table])
    uncut_inds = np.where(toa_cut_flags==None)[0]
    for ax in axes:
        ax.errorbar(mjds[uncut_inds],time_resids[uncut_inds],yerr=errs[uncut_inds],fmt='x',alpha=0.5,color='gray')
        uncut_ylim = ax.get_ylim() # ylim for plot with good TOAs only

        for c in cuts:
            if c in valid_cuts:
                cut_inds = np.where(toa_cut_flags==c)[0]
                ax.errorbar(mjds[cut_inds],time_resids[cut_inds],yerr=errs[cut_inds],fmt='x',label=c)
            else:
                log.warning(f"Unrecognized cut: {c}")

        ax.grid(True)
        ax.set_ylabel('Residual ($\mu$s)')

    ax1.legend(loc='upper center', bbox_to_anchor= (0.5, 1.2), ncol=len(cuts))
    #plt.title(f'{model.PSR.value} highlighted cuts',y=1.2)
    ax1.text(0.975,0.05,tc_object.get_source(),transform=ax1.transAxes,size=18,c='lightgray',
        horizontalalignment='right', verticalalignment='bottom')

    if multi:
        ax1.set_xticklabels([])
        ax2.set_ylim(uncut_ylim)
        ax2.set_xlabel('MJD')
    else:
        ax1.set_xlabel('MJD')
        if ylim_good:
            ax1.set_ylim(uncut_ylim)

    if save:
        if using_wideband:
            plt.savefig(f'{model.PSR.value}_manual_hl_wb.png', dpi=150)
        else:
            plt.savefig(f'{model.PSR.value}_manual_hl_nb.png', dpi=150)
    
    # reset cuts for additional processing
    from timing_analysis.utils import apply_cut_select
    apply_cut_select(toas,reason='resumption after highlighting cuts')

def check_toa_version(toas):
    """ Throws a warning if TOA version does not match the version of PINT in use

    Parameters
    ==========
    toas: `pint.toa.TOAs` object
    """
    if pint.__version__ != toas.pintversion:
        log.warning(f"TOA pickle object created with an earlier version of PINT; this may cause problems.")

def check_tobs(toas,required_tobs_yrs=2.0):
    """ Throws a warning if observation timespan is insufficient

    Parameters
    ==========
    toas: `pint.toa.TOAs` object
    """
    timespan = (toas.last_MJD-toas.first_MJD).to_value('yr')
    if timespan < required_tobs_yrs:
        log.warning(f"Observation timespan ({timespan:.2f} yrs) does not meet requirements for inclusion")

def get_cut_files(toas,cut_flag):
    """ Returns set of files where cut flag is present

    Parameters
    ==========
    toas: `pint.toa.TOAs` object
    """
    toa_cut_flags = np.array([t['flags']['cut'] if 'cut' in t['flags'] else None for t in toas.orig_table])
    cut_inds = np.where(toa_cut_flags==cut_flag)[0]
    filenames = np.array([t['flags']['name'] for t in toas.orig_table])
    return set(filenames[cut_inds])

def check_convergence(fitter):
    """Computes decrease in chi2 after fit and checks for convergence

    Parameters
    ==========
    fitter: `pint.fitter` object
    """
    chi2_decrease = fitter.resids_init.chi2-fitter.resids.chi2
    print(f"chi-squared decreased during fit by {chi2_decrease}")
    if hasattr(fitter, "converged") and fitter.converged:
        print("Fitter has converged")
    else:
        if abs(chi2_decrease)<0.01:
            print("Fitter has probably converged")
        elif chi2_decrease<0:
            log.warning("Fitter has increased chi2!")
        else:
            log.warning("Fitter may not have converged")
    if chi2_decrease > 0.01:
        log.warning("Input par file is not fully fitted")

def file_look(filenm, plot_type = 'profile'):
    """ Plots profile, GTpd, or YFp for a single file
    
    Parameters
    ==========
    filenm: file name of observation of interest
    plot_type: choose if you'd like to see the profile vs. phase, frequency vs. phase, or time vs. phase (valid values = 'profile', 'GTpd', 'YFp')
    """
    ff_list = sorted(glob.glob('/nanograv/timing/releases/15y/toagen/data/*/*/*.ff'))
    fmatch = [f for f in ff_list if filenm in f]
    if len(fmatch) == 0: 
        log.warning(f'File can\'t be found: {filenm}')
    else:
        if plot_type == 'profile':
            for f in fmatch:
                ar = pypulse.Archive(f, prepare=True)
                ar.fscrunch()
                ar.plot(subint=0, pol=0, chan=0, show=True)
        elif plot_type == 'GTpd':
            for f in fmatch:
                ar = pypulse.Archive(f, prepare=True)
                ar.tscrunch()
                ar.imshow()
        elif plot_type == 'YFp':
            for f in fmatch:
                ar = pypulse.Archive(f, prepare=True)
                ar.fscrunch()
                ar.imshow()
        else:
            log.warning(f'Unrecognized plot type: {plot_type}')

def nonexcised_file_look(toas, plot_type = 'profile'):
    """ Plots profile, GTpd, or YFp for all non-excised files
    WARNING: THIS PRODUCES A LOT OF OUTPUT!!
    
    Parameters
    ==========
    toas: `pint.toa.TOAs` object (usually 'to' here)
    plot_type: choose if you'd like to see the profile vs. phase, frequency vs. phase, or time vs. phase (valid values = 'profile', 'GTpd', 'YFp')
    """
    ff_list = sorted(glob.glob('/nanograv/timing/releases/15y/toagen/data/*/*/*.ff'))
    fnames = toas['name']
    matches = []
    for f in ff_list:
        for ff in fnames:
            if ff in f and f not in matches:
                matches.append(f)
                if plot_type == 'profile':
                    ar = pypulse.Archive(f, prepare=True)
                    ar.fscrunch()
                    ar.plot(subint=0, pol=0, chan=0, show=True)
                elif plot_type == 'GTpd':
                    ar = pypulse.Archive(f, prepare=True)
                    ar.tscrunch()
                    ar.imshow()
                elif plot_type == 'YFp':
                    ar = pypulse.Archive(f, prepare=True)
                    ar.fscrunch()
                    ar.imshow()
                else:
                    log.warning(f'Unrecognized plot type: {plot_type}')

def dmx_mjds_to_files(mjds,toas,dmxDict,mode='nb',file_only=False):
    """ Find files in DMX windows associated with input mjds
    
    Tool to facilitate the process of inspecting data associated with outlier
    or unusual DMX measurements.
    
    Parameters
    ==========
    mjds: list
        List of MJDs associated with unusual DMX values
    toas: `pint.toa.TOAs` object
    dmxDict: dictionary
        dmxout information (mjd, val, err, r1, r2), e.g. for nb, wb, respectively
    mode: str, optional
        (default: 'nb')
    file_only: bool, optional
        if True, return matching file names only, not full paths (default: False)
    """
    match_files = []
    ff_path = sorted(glob.glob('/nanograv/timing/releases/15y/toagen/data/*/*/*.ff'))
    
    for mm in mjds:
        mjd_diff = dmxDict[mode]['mjd']-mm
        closest_ind = np.argmin(np.absolute(mjd_diff))
        closest_r1 = dmxDict[mode]['r1'][closest_ind] # early edge of DMX window
        closest_r2 = dmxDict[mode]['r2'][closest_ind] #  late edge of DMX window
        
        # check that mm is between r1 and r2?
        if ((mm > closest_r1) and (mm < closest_r2)):
            mjd_in_closest_window = True
        else:
            mjd_in_closest_window = False
            log.warning(f"Selected MJD ({mm}) is outside the nearest DMX window: {closest_r1} - {closest_r2}")

        if mjd_in_closest_window:
            # get toas inside DMX window
            toa_mjds = np.array([tm for tm in toas.orig_table['mjd_float']])
            dmxwin_inds = np.where((toa_mjds>closest_r1) & (toa_mjds<closest_r2))[0]
        
            # find corresponding files
            toa_allfiles = np.array([tf['name'] for tf in toas.orig_table['flags']])
            files_to_investigate = list(set(toa_allfiles[dmxwin_inds]))
        
            # check that there is a match, len(fti) > 0?
        
            # get full paths for files_to_investigate, unless file_only=True
            if not file_only:
                for fti in files_to_investigate:
                    match = [fp for fp in ff_path if fti in fp]
                    if match: match_files.extend(match)  # though match really shouldn't have len > 1.
                    else: print(f"No match to file: {fti}")
            else:
                match_files = files_to_investigate
        
    return match_files

def convert_enterprise_equads(model):
    """ EQUADS from enterprise v3.3.0 and earlier follow temponest convention, rather than
    that in Tempo/Tempo2/PINT; this function applies a simple conversion.
    For example, with a given EFAC/EQUAD pair:

    EQUAD (pint) = EQUAD (enterprise) / EFAC

    For more information, see https://github.com/nanograv/enterprise/releases/tag/v3.3.0
    """
    efacs = [x for x in model.keys() if 'EFAC' in x]
    equads = [x for x in model.keys() if 'EQUAD' in x]

    if not len(efacs):
        log.warning('There are no white noise parameters in this timing model.')
        return model
    else:
        for ef,eq in zip(efacs,equads):
            old_eq = model[eq].value
            new_eq = old_eq/model[ef].value
            model[eq].value = new_eq
            log.info(f"{eq} has been converted from {old_eq} to {new_eq}.")
        return model
