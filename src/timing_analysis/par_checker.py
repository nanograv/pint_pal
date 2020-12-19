""" This is a set of utilities which check par files for completeness """

import re
from astropy import log
import copy

# Latest versions of bipm/ephem tracked here; update as necessary.
LATEST_BIPM = "BIPM2019"
LATEST_EPHEM = "DE438"

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

    # check for spin parameters other than F0 or F1
    for p in model.params:
        if p[0]=='F' and (len(p)==2 and p[1]>='2') or (len(p)>2 and p[1]>='0' and p[1]<='9'):
            msg = 'Unexpected spin parameter %s should be removed' % (p,)
            log.warn(msg)
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



def check_jumps(model, receivers):
    """Check that there are the correct number of jumps in the par file.

    Parameters
    ==========
    model: PINT model object
    receivers: list of receiver strings

    Raises
    ======
    ValueError
        If there is not one un-jumped receiver at the end, or
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
    return

def check_dmjumps(model, receivers):
    """Check that there are the correct number of dmjumps in the par file.

    Parameters
    ==========
    model: PINT model object
    receivers: list of receiver strings

    Raises
    ======
    ValueError
        If not all receivers are dm-jumped at the end, or
        no TOAs were recorded with that receiver
    """
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
    return

def check_ephem(toa):
    """Check that the ephemeris matches the latest version.

    Parameters
    ==========
    model: PINT toa object

    Raises
    ======
    ValueError
        If ephemeris is not set to the latest version.
    """
    if toa.ephem != LATEST_EPHEM:
        msg = "Wrong ephem (%s); should be %s." % (toa.ephem,LATEST_EPHEM)
        log.warning(msg)
    return

def check_bipm(toa):
    """Check that BIPM correction matches the latest version.

    Parameters
    ==========
    model: PINT toa object

    Raises
    ======
    ValueError
        If BIPM correction is not set to the latest version.
    """
    if toa.clock_corr_info['bipm_version'] != LATEST_BIPM:
        msg = "Wrong bipm_version (%s); should be %s." % (toa.clock_corr_info['bipm_version'],LATEST_BIPM)
        log.warning(msg)
    return
