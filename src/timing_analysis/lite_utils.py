import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from datetime import date
import timing_analysis.par_checker as pc

# Read tim/par files
import pint.toa as toa
import pint.models as models
import pint.residuals
from pint.modelutils import model_equatorial_to_ecliptic

from pint.models.parameter import maskParameter
from pint.models.timing_model import Component 

def write_par(fitter,addext=''):
    """Writes a timing model object to a par file in the working directory.

    Parameters
    ==========
    fitter: `pint.fitter` object
    """
    source = fitter.get_allparams()['PSR']
    date_str = date.today().strftime('%Y%m%d')
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

def plot_res(fitter,restype='prefit'):
    """Simple plotter for prefit/postfit residuals.

    Parameters
    ==========
    fitter: `pint.fitter` object
    restype: string, optional
        controls type of residuals plotted, prefit/postfit

    Raises
    ======
    ValueError
        If restype is not recognized.
    """
    fo = fitter
    obslist = list(fitter.toas.observatories)

    # Select toas from each observatory.
    for obso in obslist:

        select_array = (fitter.toas.get_obss()==obso)
        fitter.toas.select(select_array)

        if 'pre' in restype:
            plt.errorbar(
            fitter.toas.get_mjds(), fitter.resids_init.time_resids.to(u.us)[select_array], fitter.toas.get_errors().to(u.us), fmt="x",label=obso
            )
            restype_str = 'Pre'
        elif 'post' in restype:
            plt.errorbar(
            fitter.toas.get_mjds(), fitter.resids.time_resids.to(u.us)[select_array], fitter.toas.get_errors().to(u.us), fmt="x", label=obso
            )
            restype_str = 'Post'
        else:
            raise ValueError("Residual type (%s) not recognized. Try prefit/postfit." % (restype)) 

        plt.title("%s %s-Fit Timing Residuals" % (fitter.model.PSR.value,restype_str))
        plt.xlabel("MJD")
        plt.ylabel("Residual (us)")
        plt.grid()
        plt.legend()

        fitter.toas.unselect()

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

def apply_snr_cut(toas,snr_cut,summary=False):
    """Imposes desired signal-to-noise ratio cut

    Parameters
    ==========
    toas: `pint.toa.TOAs` object
    snr_cut: float
        selects TOAs with snr > snr_cut
    summary: boolean, optional
        print toa summary
    """
    # Might want a warning here. SNR cut should happen before others for intended effect. 
    # toas.unselect()
    toas.select((np.array(toas.get_flag_value('snr')) > snr_cut)[0])
    
    if summary:    
        toas.print_summary()

def apply_mjd_cut(toas,configDict,summary=False):
    """Imposes cuts based on desired start/stop times (MJD)

    Parameters
    ==========
    toas: `pint.toa.TOAs` object
    configDict: Dict
        configuration parameters read directly from yaml
    summary: boolean, optional
        print toa summary
    """

    if configDict['ignore']['mjd-start']:
        mjd_min = configDict['ignore']['mjd-start']
        select_min = (toas.get_mjds() > mjd_min*u.d)
    else:
        select_min = np.array([True]*len(toas))


    if configDict['ignore']['mjd-end']:
        mjd_max = configDict['ignore']['mjd-end']
        select_max = (toas.get_mjds() < mjd_max*u.d)
    else:
        select_max = np.array([True]*len(toas))

    toas.select(select_min & select_max)

    if summary:
        toas.print_summary()

def load_and_check(configDict,usepickle=False):
    """Loads toas/model objects using configuration info, runs basic checks.

    Checks that ephem and bipm_version have been set to the latest available versions; checks
    for equatorial astrometric parameters (converts to ecliptic, if necessary); also checks
    source name, and for appropriate number of jumps. Checks are functions from par_checker.py.

    Parameters
    ==========
    configDict: Dict
        configuration parameters read directly from yaml
    usepickle: boolean, optional
        produce TOA pickle object

    Returns
    =======
    to: `pint.toa.TOAs` object
        passes all checks
    mo: `pint.model.TimingModel` object
        passes all checks
    """
    source = configDict['source']
    tim_path = configDict['tim-directory']
    tim_files = [tim_path+tf for tf in configDict['toas']]
    tim_filename = write_include_tim(source,tim_files)
    to = toa.get_TOAs(tim_filename, usepickle=usepickle, bipm_version=configDict['bipm'], ephem=configDict['ephem'])

    # Check ephem/bipm
    pc.check_ephem(to)
    pc.check_bipm(to)

    # Identify receivers present
    receivers = set(to.get_flag_value('fe')[0])

    # Load the timing model
    par_path = configDict['par-directory']
    mo = models.get_model(par_path+configDict['timing-model'])

    # Convert to/add AstrometryEcliptic component model if necessary.
    if 'AstrometryEquatorial' in mo.components:
        model_equatorial_to_ecliptic(mo)

    # Basic checks on timing model
    pc.check_name(mo)
    add_feJumps(mo,list(receivers))
    pc.check_jumps(mo,receivers)

    return to, mo

def check_fit(fitter):
    """Check that pertinent parameters are unfrozen.

    Note: process of doing this robustly for binary models is not yet automated. Checks are
    functions from par_checker.py.

    Parameters
    ==========
    fitter: `pint.fitter` object 
    """
    pc.check_spin(fitter.model)
    pc.check_astrometry(fitter.model)

def select_out_toa(to,badtoa_list):
    """Select out individual bad TOAs according to yaml bad-toa lines

    Parameters
    ==========
    toas: `pint.toa.TOAs` object
    badtoa_list : list
        list with fields associated with TOA flags 'name', 'chan', 'subint'
    """
    name,chan,subint = badtoa_list
    name_match = np.array([(n == name) for n in to.get_flag_value('name')[0]])
    chan_match = np.array([(ch == chan) for ch in to.get_flag_value('chan')[0]])
    subint_match = np.array([(si == subint) for si in to.get_flag_value('subint')[0]])
    match = name_match * subint_match * chan_match
    
    if np.sum(match) == 1:
        toa_number = np.where(match==True)[0][0]
        print('Zapping TOA: %s' % (toa_number))
        to.select(np.invert(name_match * subint_match * chan_match))
    else:
        # Probably should have a warning or something here.
        pass

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

    if len(missing_fe_jumps) > 1:
        for j in missing_fe_jumps[:-1]:
            JUMPn = maskParameter('JUMP',key='fe',key_value=[j],value=0.0,units=u.second)
            phasejump.add_param(JUMPn,setup=True)

def apply_range_cut(to,badrange_list):
    """According to the bad-range list in your yaml, mask epochs between two mjds. Can be done for multiple date ranges.
    Parameters
    ==========
    to: `pint.toa.TOAs` object
    badrange_list : list
        list with entries [start_mjd,end_mjd] to designate what will be masked
    """
    mjd_start,mjd_end = badrange_list
    min_crit = (to.get_mjds() > mjd_start*u.d)
    max_crit = (to.get_mjds() < mjd_end*u.d)
    to.select(np.logical_xor(min_crit, max_crit))
    
def apply_epoch_cut(to,badepoch):
    """According to the bad-epoch entries in your yaml, mask TOAs containing that basename (badepoch).
    Parameters
    ==========
    to: `pint.toa.TOAs` object
    badepoch : string
        basename to designate what will be masked
    """
    be = badepoch
    base_in_list = np.array([(be not in n) for n in to.get_flag_value('name')[0]])
    to.select(base_in_list)
