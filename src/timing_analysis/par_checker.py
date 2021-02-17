""" This is a set of utilities which check par files for completeness """

import re
import copy
from astropy import log
import astropy.units as u
from timing_analysis.defaults import *

def check_if_fit(model, *param):
    """
    Check if a parameter or set of parameters
    exists and are fit

    Parameters
    ==========
    model: PINT model object
    param: parameter string(s)

    Raises
    ======
    ValueError
        If (any) parameter is frozen,
        the parameter is not present
    """
    for p in param:
        if not hasattr(model, p):
            raise ValueError("%s parameter is not present in par file."%p)
        if getattr(model, p).frozen:
            raise ValueError("%s parameter is frozen."%p)
    return


def has_and_check_if_fit(model, *param):
    """
    Convenience function for
    if hasattr(model, param):
        check_if_fit(model, param)

    Similar to check_if_fit() except that it will first
    test to see if the parameter exists, and if it does not,
    no error will be issued

    Parameters
    ==========
    model: PINT model object
    param: parameter string(s)
    """
    for p in param:
        if hasattr(model, p):
            check_if_fit(model, p)



def check_name(model):
    """Check the pulsar name

    Parameters
    ==========
    model: PINT model object

    Warnings
    ========
    UserWarning
        If PSR parameter is not in standard J epoch format, issue a warning.
    """
    name = model.PSR.value
    if not re.match("^((J[0-9]{4}[+-][0-9]{4})|(B[0-9]{4}[+-][0-9]{2}))$", name):
        msg = "PSR parameter is not in the proper format."
        log.warning(msg)
    else:
        msg = f"PSR parameter is {name}."
        log.info(msg)
    return


def check_spin(model):
    """Check spin parameters in par file.

    Parameters
    ==========
    model: PINT model object

    Raises
    ======
    ValueError
        If parameters provided are fixed
    """

    check_if_fit(model, "F0", "F1")

    # check for spindown parameters other than F0 or F1, i.e., any
    # such parameter that consists of the letter 'F' followed by
    # an integer
    for p in model.components['Spindown'].params:
        if p.startswith('F') and len(p)>1 and p[1:].isdigit():
            n = int(p[1:])
            if n < 0 or n >1:
                msg = 'Unexpected spin parameter %s should be removed' % (p,)
                log.warning(msg)

    return


def check_astrometry(model):
    """Check astrometry parameters in par file.

    Parameters
    ==========
    model: PINT model object

    Raises
    ======
    ValueError
        If parameters are not in ecliptic coordinates, or
        any astrometric parameter is fixed.
    """
    if "AstrometryEcliptic" not in model.components.keys():
        raise ValueError("Astrometric parameters not in ecliptic coordinates.")
    check_if_fit(model, "ELONG", "ELAT", "PMELONG", "PMELAT", "PX")
    return


def check_binary(model):
    """
    Check binary parameters in par file.
    The following models are checked for: ELL1, ELL1H, DD, DDK
    Performs multiple consistency checks for parameters and subsequent parameters,
    making sure they are fit.

    Parameters
    ==========
    model: PINT model object

    Raises
    ======
    ValueError
        If parameters are frozen or models are improperly defined
    """
    if not hasattr(model, 'binary_model_name'):
        return

    name = getattr(model, 'binary_model_name')

    if name == "ELL1":
        # Check base models, including either PB or FB formulations
        check_if_fit(model, "A1", "TASC", "EPS1", "EPS2")
        if hasattr(model, "PB") and hasattr(model, "FB0"):
            raise ValueError("Both PB and FB0 are defined")
        elif hasattr(model, "PB"):
            formalismFB = False
            check_if_fit(model, "PB")
        elif hasattr(model, "FB0"):
            formalismFB = True
            check_if_fit(model, "FB0")
        else:
            raise ValueError("Either PB or FB0 are required in the ELL1 model.")

        ### Need to include further checks here for higher-order FB terms,
        ### and to make sure they are sequentially fit for


        if not formalismFB:
            has_and_check_if_fit(model, "PBDOT")
            if hasattr(model, "SINI") and hasattr(model, "M2"):
                check_if_fit(model, "SINI", "M2")
            elif hasattr(model, "SINI") or hasattr(model, "M2"):
                raise ValueError("Both SINI and M2 are required when fitting for companion mass.")

        has_and_check_if_fit(model, "XDOT")
        if hasattr(model, "EPS1DOT") and hasattr(model, "EPS2DOT"):
            check_if_fit(model, "EPS1DOT", "EPS2DOT")
        elif hasattr(model, "EPS1DOT") or hasattr(model, "EPS2DOT"):
            raise ValueError("Both EPS1DOT and EPS2DOT are required when fitting for eccentricity derivatives.")



    elif name == "ELL1H":
        check_if_fit(model, "A1", "PB", "TASC", "EPS1", "EPS2", "H3") #H3 is rquired for ELL1H?
        has_and_check_if_fit(model, "PBDOT", "XDOT", "H4")

        if hasattr(model, "EPS1DOT") and hasattr(model, "EPS2DOT"):
            check_if_fit(model, "EPS1DOT", "EPS2DOT")
        elif hasattr(model, "EPS1DOT") or hasattr(model, "EPS2DOT"):
            raise ValueError("Both EPS1DOT and EPS2DOT are required when fitting for eccentricity derivatives.")



    elif name == "DD":
        check_if_fit(model, "A1" "E", "T0", "PB", "OM")
        has_and_check_if_fit(model, "PBDOT", "XDOT", "OMDOT", "EDOT")

        if hasattr(model, "SINI") and hasattr(model, "M2"):
            check_if_fit(model, "SINI", "M2")
        elif hasattr(model, "SINI") or hasattr(model, "M2"):
            raise ValueError("Both SINI and M2 are required when fitting for companion mass.")
    elif name == "DDK":
        check_if_fit(model, "A1", "E", "T0", "PB", "OM", "M2", "K96", "KOM", "KIN")
        has_and_check_if_fit(model, "PBDOT", "XDOT", "OMDOT", "EDOT")

        if hasattr(model, "SINI") and hasattr(model, "M2"):
            check_if_fit(model, "SINI", "M2")
        #elif hasattr(model, "SINI") or hasattr(model, "M2"):
    elif name == "T2":

        ### Are there plans for PINT to support the T2 model?

        check_if_fit(model, "A1", "ECC", "T0", "PB", "OM", "M2", "KOM", "KIN")
        # maybe check if SINI = KIN?
    else:
        raise ValueError("Atypical binary model used")

def check_jumps(model,receivers,fitter_type=None):
    """Checks for correct type/number of JUMP/DMJUMPs in the par file

    Parameters
    ==========
    model: PINT model object
    receivers: list of receiver strings

    Raises
    ======
    ValueError
        If there is not one un-jumped receiver at the end, or
        no TOAs were recorded with that receiver

    ValueError
        If not all receivers are dm-jumped at the end, or
        no TOAs were recorded with that receiver
    """
    jumps = []
    rcvrs = copy.copy(receivers)

    for p in model.params:
        if p.startswith("JUMP"):
            jumps.append(p)

    for jump in jumps:
        j = getattr(model, jump)
        value = j.key_value[0]
        if value not in rcvrs:
            raise ValueError("Receiver %s not used in TOAs"%value)
        rcvrs.remove(value)

    length = len(rcvrs)
    if length > 1:
        raise ValueError("%i receivers require JUMPs"%length)
    elif length == 0:
        raise ValueError("All receivers have JUMPs, one must be removed")

    # Check DMJUMPS for wideband models
    if fitter_type == 'WidebandTOAFitter':
        dmjumps = []
        rcvrs = copy.copy(receivers)

        for p in model.params:
            if p.startswith("DMJUMP"):
                dmjumps.append(p)

        for dmjump in dmjumps:
            j = getattr(model, dmjump)
            value = j.key_value[0]
            if value not in rcvrs:
                raise ValueError("Receiver %s not used in TOAs"%value)
            rcvrs.remove(value)

        length = len(rcvrs)
        if length:
            raise ValueError("%i receivers require DMJUMPs"%length)
    else:
        pass

    return

def check_ephem(toa):
    """Check that the ephemeris matches the latest version.

    Parameters
    ==========
    model: PINT toa object

    Warnings
    ========
    UserWarning
        If ephemeris is not set to the latest version.
    """
    if toa.ephem != LATEST_EPHEM:
        msg = f"Wrong Solar System ephemeris in use ({toa.ephem}); should be {LATEST_EPHEM}."
        log.warning(msg)
    else:
        msg = f"Current Solar System ephemeris in use is {toa.ephem}."
        log.info(msg)
    return

def check_bipm(toa):
    """Check that BIPM correction matches the latest version.

    Parameters
    ==========
    model: PINT toa object

    Warnings
    ========
    UserWarning
        If BIPM correction is not set to the latest version.
    """
    if toa.clock_corr_info['bipm_version'] != LATEST_BIPM:
        msg = f"Wrong bipm_version ({toa.clock_corr_info['bipm_version']}); should be {LATEST_BIPM}."
        log.warning(msg)
    else:
        msg = f"BIPM version in use is {toa.clock_corr_info['bipm_version']}."
        log.info(msg)
    return

def check_ecliptic(model):
    """Check that the parfile uses ecliptic coordinates.

    Parameters
    ==========
    model: PINT toa object

    Warnings
    ========
    UserWarning
        If not all model components are in ecliptic coordinates.
    """
    # Convert to/add AstrometryEcliptic component model if necessary.
    if 'AstrometryEquatorial' in model.components:
        msg = "AstrometryEquatorial in model components; switching to AstrometryEcliptic."
        log.warning(msg)
        model_equatorial_to_ecliptic(model)
    elif 'AstrometryEcliptic' in model.components:
        msg = "AstrometryEcliptic in model components."
        log.info(msg)
    else:
        msg = "Neither AstrometryEcliptic nor AstrometryEquatorial in model components."
        log.warning(msg)
    return

def check_troposphere(model):
    """Check that the model will (or will not) correct for the troposphere.

    Parameters
    =========
    model: PINT toa object

    Warnings
    ========
    UserWarning
        If CORRECT_TROPOSPHERE is not set to the default boolean and/or if the component is not in the model.
    """
    if 'TroposphereDelay' not in model.components.keys():
        from pint.models.timing_model import Component
        troposphere_delay = Component.component_types["TroposphereDelay"]
        model.add_component(troposphere_delay())
        msg = "Added TroposphereDelay to model components."
        log.warning(msg)
    tropo = model.components['TroposphereDelay'].CORRECT_TROPOSPHERE.value
    if tropo != CORRECT_TROPOSPHERE:
        model.components['TroposphereDelay'].CORRECT_TROPOSPHERE.set( \
                CORRECT_TROPOSPHERE)
        msg = "Switching CORRECT_TROPOSPHERE setting."
        log.warning(msg)
    tropo = model.components['TroposphereDelay'].CORRECT_TROPOSPHERE.value
    msg = f"CORRECT_TROPOSPHERE is set to {tropo}."
    log.info(msg)
    return

def check_planet_shapiro(model):
    """Check that the model will (or will not) correct for the planets' Shapiro delays.

    Parameters
    =========
    model: PINT toa object

    Warnings
    ========
    UserWarning
        If PLANET_SHAPIRO is not set to the default boolean and/or if the component is not in the model.
    """
    if 'SolarSystemShapiro' not in model.components.keys():
        from pint.models.timing_model import Component
        sss_delay = Component.component_types["SolarSystemShapiro"]
        model.add_component(sss_delay())
        msg = "Added SolarSystemShapiro to model components."
        log.warning(msg)
    sss = model.components['SolarSystemShapiro'].PLANET_SHAPIRO.value
    if sss != PLANET_SHAPIRO:
        model.components['SolarSystemShapiro'].PLANET_SHAPIRO.set( \
                PLANET_SHAPIRO)
        msg = "Switching PLANET_SHAPIRO setting."
        log.warning(msg)
    sss = model.components['SolarSystemShapiro'].PLANET_SHAPIRO.value
    msg = f"PLANET_SHAPIRO is set to {sss}."
    log.info(msg)
    return

def check_settings(model, toas, check_these=['name', 'ephem', 'bipm',
    'ecliptic', 'troposphere', 'planet_shapiro', 'bad_lo_range']):
    """Umbrella function to run numerous check_ functions."""
    requires_model = {'name', 'ecliptic', 'troposphere', 'planet_shapiro'}
    requires_toas = {'ephem', 'bipm', 'bad_lo_range'}
    requires_both = {}
    for thing in check_these:
        if thing in requires_model:
            globals()[f"check_{thing}"](model)
        elif thing in requires_toas:
            globals()[f"check_{thing}"](toas)
        elif thing in requires_both:
            globals()[f"check_{thing}"](model,toas)
        else:
            log.warning(f"check_{thing} not executed.")
    return

def check_bad_lo_range(toas):
    """Check: no Arecibo TOAs exist in MJD range affected by bad LO (57984-58447)

    15-yr specific mitigation strategy for excising affected data. Will raise
    log.warning if Arecibo TOAs exist in this range.

    Parameters
    ==========
    toas: `pint.toa.TOAs` object
    """
    bad_lo_start = (toas.get_mjds() > 57984.0*u.d)
    bad_lo_end = (toas.get_mjds() < 58447.0*u.d)
    lo_check = (bad_lo_start & bad_lo_end)
    ao_check = (toas.get_obss() == 'arecibo')

    if any(ao_check & lo_check):
        log.warning('Data affected by Arecibo bad LO should be excised (add bad-range to .yaml).')
        print('''
Add the following line(s) to the ignore block of your .yaml file...

  bad-range:
  - [57984,58447,"PUPPI"]
        ''')
    elif any(ao_check):
        log.info('Arecibo data affected by bad LO (MJD 57984-58447) has been ignored.')
