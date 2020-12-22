import numpy as np
import astropy.units as u
from astropy import log
import matplotlib.pyplot as plt
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

def write_par(fitter,toatype='',addext=''):
    """Writes a timing model object to a par file in the working directory.

    Parameters
    ==========
    fitter: `pint.fitter` object
    toatype: str, optional
        if set, adds nb/wb.par
    addext: str, optional
        if set, adds extension to date
    """
    source = fitter.get_allparams()['PSR'].value
    date_str = date.today().strftime('%Y%m%d')
    if toatype:
        outfile = '%s_PINT_%s%s.%s.par' % (source,date_str,addext,toatype.lower())
    else:
        outfile = '%s_PINT_%s%s.par' % (source,date_str,addext)

    fout=open(outfile,'w')
    fout.write(fitter.model.as_parfile())
    fout.close()


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
    midmjd=(toas.get_mjds().value.max()+toas.get_mjds().value.min())/2.
    model.change_pepoch(midmjd)

    try:
        model.change_posepoch(midmjd)
    except:
        pass

    try:
        model.change_dmepoch(midmjd)
    except:
        pass

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

    log.info(f"Frontends not JUMPed: {missing_fe_jumps}")
    if len(missing_fe_jumps) > 1:
        for j in missing_fe_jumps[:-1]:
            log.info(f"Adding Frontend JUMP {j}")
            JUMPn = maskParameter('JUMP',key='-fe',key_value=[j],value=0.0,units=u.second)
            phasejump.add_param(JUMPn,setup=True)

def check_toas_model(fitter,center=True,summary=True):
    """Runs basic checks on previously-loaded timing model & TOA objects.

    Checks that ephem and bipm_version have been set to the latest available versions; checks
    for equatorial astrometric parameters (converts to ecliptic, if necessary); also checks
    source name, and for appropriate number of jumps/dmjumps. Checks are functions from par_checker.py.

    Parameters
    ==========
    fitter: `pint.fitter` object
    center: boolean, optional
        if true, center PEPOCH, DMEPOCH, POSEPOCH (default: True)
    summary: boolean, optional
        if true, print TOA summary (default: True)

    Returns
    =======
    None
    """
    # Get TOA and model objects from fitter
    to = fitter.toas
    mo = fitter.model

    # Check ephem/bipm
    pc.check_ephem(to)
    pc.check_bipm(to)

    # Identify receivers present
    receivers = set([str(f) for f in set(to.get_flag_value('fe')[0])])

    # Convert to/add AstrometryEcliptic component model if necessary.
    if 'AstrometryEquatorial' in mo.components:
        msg = "AstrometryEquatorial in model components; switching to AstrometryEcliptic."
        log.warning(msg)
        model_equatorial_to_ecliptic(mo)

    # Basic checks on timing model
    pc.check_name(mo)
    add_feJumps(mo,list(receivers))
    pc.check_jumps(mo,receivers)
    if  fitter.__class__.__name__ == 'WidebandTOAFitter':
        pc.check_dmjumps(mo,receivers)

    # Center epochs?
    if center:
        center_epochs(mo,to)

    # Print summary?
    if summary:
        to.print_summary()

def large_residuals(fo,threshold_us):
    """Quick and dirty routine to find outlier residuals based on some threshold.

    Parameters
    ==========
    fo: `pint.fitter` object
    threshold_us: float
        not a quantity, but threshold for residuals larger (magnitude) than some delay in microseconds

    Returns
    =======
    None
        prints bad-toa lines that can be copied directly into a yaml file
    """

    badtoalist = np.where(np.abs(fo.resids_init.time_resids.to(u.us)) > threshold_us*u.us)
    for i in badtoalist[0]:
        name = fo.toas.get_flag_value('name')[0][i]
        chan = fo.toas.get_flag_value('chan')[0][i]
        subint = fo.toas.get_flag_value('subint')[0][i]
        print('    - [\'%s\',%i,%i]'%(name,chan,subint))

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

def print_changelog(config_file):
    """Function to print changelog from YAML in human-readable format in the notebook. 
    Takes that YAML ("config_file") as its only argument.
    """
    # Read from YAML
    stream = open(config_file, 'r')
    configDict = yaml.safe_load(stream)
    # If there's a changelog, write out its contents. If not, complain.
    if 'changelog' in configDict.keys():
        print('YAML changelog as of %s GMT:'%(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        if configDict['changelog'] is not None:
            for cl in configDict['changelog']:
                print('  - %s'%(cl))
        else:
            print('  - No changelog entries appear in our records, so they don\'t exist.\n')
    else:
        print('YAML config file doesn\'t include a changelog. Please append \'changelog:\' to it and add entries below that.')

def new_changelog_entry(tag, note):
    """Checks for valid tag and auto-generates entry to be copy/pasted into .yaml changelog block.

    Your NANOGrav email (before the @) and the date will be printed automatically. The "tag"
    describes the type of change, and the "note" is a short (git-commit-like) description of
    the change. Entry should be manually appended to .yaml by the user.

    Valid tags:
      - INIT: creation of the .yaml file
      - ADD or REMOVE: adding or removing a parameter
      - BINARY: change in the binary model or binary parameters
      - NOISE: changes in noise parameters
      - CURATE: adding / removing TOAs, or changing S/N threshold
    """
    VALID_TAGS = ['TEST','INIT','ADD','REMOVE','BINARY','NOISE','CURATE']
    vtstr = ', '.join(VALID_TAGS)
    if tag not in VALID_TAGS:
        msg = f'{tag} is not a valid tag; valid tags are: {vtstr}.'
        log.error(msg)
    else:
        # Read the git user.email from .gitconfig, return exception if not set
        stream = os.popen('git config --get user.email')
        username = stream.read().rstrip().split('@')[0]

        if not username:
            msg = 'Update your git config with... git config --global user.email \"your.email@nanograv.org\"'
            log.error(msg)
        else:
            # Date in YYYY-MM-DD format
            now = datetime.now()
            date = now.strftime('%Y-%m-%d')
            print(f'  - \'{date} {username} {tag}: {note}\'')
