""" This is a set of utilities which check par files for completeness """

import os, sys
import re
import copy
from astropy import log
import astropy.units as u
sys.path.append("/home/jovyan/work/timing_analysis/src/")
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
        if not getattr(model, p).frozen:
            print("Good: %s parameter is fit."%p)
    return

#def check_if_wrongly_included(model, *param):
#    for p in param:

                  
def check_if_fit_WIP(model, *param):
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
    
    FB_pars = {'FB1','FB2','FB3','FB4','FB5','FB6'}
    
    
    for p in param:
        print(p)
        globals()
        p = globals()["thing"]
        print(p)
        print(globals()[f"model.{thing}.value"])
        #globals()[f"print(model.{thing}.value"]
    
    #print(model.param[0].value)
    
    for p in param:
        if hasattr(model, p):
            print(p.value)
            #if (model.PB.value is not None)
        if not hasattr(model, p):
            raise ValueError("%s parameter is not present in par file."%p)
        if getattr(model, p).frozen:
            raise ValueError("%s parameter is frozen."%p)
        else:
            print("Good: %s parameter is fit."%p)
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
        print(p)
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
    
    #######################
    #### Start of ELL1 ####
    #######################

    if name == "ELL1":
        # Check base models, including either PB or FB formulations
        # This block of the script correctly determines formalismFB
        check_if_fit(model, "A1", "TASC", "EPS1", "EPS2")
        if (model.PB.value is not None) and (model.FB0.value is None):
            formalismFB = False
            check_if_fit(model, "PB")
        elif (model.PB.value is None) and (model.FB0.value is not None):
            formalismFB = True
            check_if_fit(model, "FB0")
        elif (model.PB.value is None) and (model.FB0.value is None):
            raise ValueError("Neither PB nor FB0 are set to a value")
        elif (model.PB.value is not None) and (model.FB0.value is not None):
            raise ValueError("Both PB and FB0 are defined")
        else:
            raise ValueError("Either PB or FB0 are required in the ELL1 model.")
        
        # Check for higher-order FB terms, making sure they are fit for sequentially
        if formalismFB:
            # We know FB0 is fit because of the above, so just check FB1 and up
            # Start with highest-order FB derivative. Should we start higher than FB5?
            if hasattr(model, "FB6"):
                print("model has FB6")
                check_if_fit(model, "FB6", "FB5", "FB4", "FB3", "FB2", "FB1")
            elif hasattr(model, "FB5") and not hasattr(model, "FB6"):
                print("model has FB5 but not FB6")
                check_if_fit(model, "FB5", "FB4", "FB3", "FB2", "FB1")
            elif hasattr(model, "FB4") and not hasattr(model, "FB6") and not hasattr(model, "FB5"):
                print("model has FB4 but not FB5 or FB6")
                check_if_fit(model, "FB4", "FB3", "FB2", "FB1")
            elif hasattr(model, "FB3") and not hasattr(model, "FB6") and not hasattr(model, "FB5") and not hasattr(model, "FB4"):
                print("model has FB3 but not FB4, FB5, or FB6")
                check_if_fit(model, "FB3", "FB2", "FB1")
            elif hasattr(model, "FB2") and not hasattr(model, "FB6") and not hasattr(model, "FB5") and not hasattr(model, "FB4") and not hasattr(model, "FB3"):
                print("model has FB2 but not FB3, FB4, FB5, or FB6")
                check_if_fit(model, "FB2", "FB1")
            elif hasattr(model, "FB1") and not hasattr(model, "FB6") and not hasattr(model, "FB5") and not hasattr(model, "FB4") and not hasattr(model, "FB3") and not hasattr("FB2"):
                print("model has FB1 but not FB2, FB3, FB4, FB5, or FB6")
                check_if_fit(model, "FB1")
            else:
                raise ValueError("The options given were not sufficient to describe the presence/absence of FB1 through FB6.")
        
        # We already know if PB is in the fit so don't check again
        if not formalismFB:
            if model.PBDOT.value is not None:
                check_if_fit(model, "PBDOT")
            else:
                print("PBDOT is not included in the model.")
                
        # Check for M2 and SINI
        if (model.SINI.value is not None) and (model.M2.value is not None):
            check_if_fit(model, "M2", "SINI")
        elif (model.SINI.value is not None) and (model.M2.value is None):
            raise ValueError("Both SINI and M2 are required when fitting for companion mass.")
        elif (model.SINI.value is None) and (model.M2.value is not None):
            raise ValueError("Both SINI and M2 are required when fitting for companion mass.")
        elif (model.SINI.value is None) and (model.M2.value is None):
            print("M2 and SINI are not included in the model.")
        else:
            raise ValueError("Please check: there is a problem with the inclusion and/or exclusion of M2 and/or SINI in the model.")
               
        # Check for A1DOT (XDOT)
        if model.A1DOT.value is not None:
            check_if_fit(model, "A1DOT")
        else:
            print("A1DOT is not included in the model.")
            
        # Check for eccentricity parameter derivatives
        if (model.EPS1DOT.value is not None) and (model.EPS2DOT.value is not None):
            check_if_fit(model, "EPS1DOT", "EPS1DOT")
        elif (model.EPS1DOT.value is not None) and (model.EPS2DOT.value is None):
            raise ValueError("Both EPS1DOT and EPS2DOT are required when fitting for eccentricity derivatives.")
        elif (model.EPS1DOT.value is None) and (model.EPS2DOT.value is not None):
            raise ValueError("Both EPS1DOT and EPS2DOT are required when fitting for eccentricity derivatives.")
        elif (model.EPS1DOT.value is None) and (model.EPS2DOT.value is None):
            print("EPS1DOT and EPS2DOT are not included in the model.")
        else:
            #raise ValueError("There is an oddity with the inclusion/exclusion in the model of EPS1DOT and/or EPS2DOT.")
            raise ValueError("Please check: there is a problem with the inclusion and/or exclusion of EPS1DOT and/or EPS2DOT in the model.")
                
        # Check if H3 or H4 are in the model
        if model.H3.value is not None:
            print("H3 is present in this parfile. Either remove H3 or change the binary model to ELL1H.")
        if model.H4.value is not None:
            print("H4 is present in this parfile. Either remove H4 or change the binary model to ELL1H.")
    

    ######################
    ### Start of ELL1H ###
    ######################

    elif name == "ELL1H":
        check_if_fit(model, "A1", "PB", "TASC", "EPS1", "EPS2", "H3")

        # Determine formalismFB
        if (model.PB.value is not None) and (model.FB0.value is None):
            formalismFB = False
            check_if_fit(model, "PB")
        elif (model.PB.value is None) and (model.FB0.value is not None):
            formalismFB = True
            check_if_fit(model, "FB0")
        elif (model.PB.value is None) and (model.FB0.value is None):
            raise ValueError("Neither PB nor FB0 are set to a value")
        elif (model.PB.value is not None) and (model.FB0.value is not None):
            raise ValueError("Both PB and FB0 are defined")
        else:
            raise ValueError("Either PB or FB0 are required in the ELL1H model.")
        
        # Check for higher-order FB terms, making sure they are fit for sequentially
        if formalismFB:
            # We know FB0 is fit because of the above, so just check FB1 and up
            # Start with highest-order FB derivative. Should we start higher than FB5?
            if hasattr(model, "FB6"):
                print("model has FB6")
                check_if_fit(model, "FB6", "FB5", "FB4", "FB3", "FB2", "FB1")
            elif hasattr(model, "FB5") and not hasattr(model, "FB6"):
                print("model has FB5 but not FB6")
                check_if_fit(model, "FB5", "FB4", "FB3", "FB2", "FB1")
            elif hasattr(model, "FB4") and not hasattr(model, "FB6") and not hasattr(model, "FB5"):
                print("model has FB4 but not FB5 or FB6")
                check_if_fit(model, "FB4", "FB3", "FB2", "FB1")
            elif hasattr(model, "FB3") and not hasattr(model, "FB6") and not hasattr(model, "FB5") and not hasattr(model, "FB4"):
                print("model has FB3 but not FB4, FB5, or FB6")
                check_if_fit(model, "FB3", "FB2", "FB1")
            elif hasattr(model, "FB2") and not hasattr(model, "FB6") and not hasattr(model, "FB5") and not hasattr(model, "FB4") and not hasattr(model, "FB3"):
                print("model has FB2 but not FB3, FB4, FB5, or FB6")
                check_if_fit(model, "FB2", "FB1")
            elif hasattr(model, "FB1") and not hasattr(model, "FB6") and not hasattr(model, "FB5") and not hasattr(model, "FB4") and not hasattr(model, "FB3") and not hasattr("FB2"):
                print("model has FB1 but not FB2, FB3, FB4, FB5, or FB6")
                check_if_fit(model, "FB1")
            else:
                raise ValueError("The options given were not sufficient to describe the presence/absence of FB1 through FB6.")
            
        if not formalismFB:
            if model.PBDOT.value is not None:
                check_if_fit(model, "PBDOT")
            else:
                print("PBDOT is not included in the model.")
                
        if (model.SINI.value is not None) and (model.M2.value is not None):
            check_if_fit(model, "M2", "SINI")
        elif (model.SINI.value is not None) and (model.M2.value is None):
            raise ValueError("Both SINI and M2 are required when fitting for companion mass.")
        elif (model.SINI.value is None) and (model.M2.value is not None):
            raise ValueError("Both SINI and M2 are required when fitting for companion mass.")
        elif (model.SINI.value is None) and (model.M2.value is None):
            print("M2 and SINI are not included in the model.")
        else:
            raise ValueError("Please check: there is a problem with the inclusion and/or exclusion of M2 and/or SINI in the model.")
                
        if model.A1DOT.value is not None:
            check_if_fit(model, "A1DOT")
        else:
            print("A1DOT is not included in the model.")
            
        if (model.EPS1DOT.value is not None) and (model.EPS2DOT.value is not None):
            check_if_fit(model, "EPS1DOT", "EPS1DOT")
        elif (model.EPS1DOT.value is not None) and (model.EPS2DOT.value is None):
            raise ValueError("Both EPS1DOT and EPS2DOT are required when fitting for eccentricity derivatives.")
        elif (model.EPS1DOT.value is None) and (model.EPS2DOT.value is not None):
            raise ValueError("Both EPS1DOT and EPS2DOT are required when fitting for eccentricity derivatives.")
        elif (model.EPS1DOT.value is None) and (model.EPS2DOT.value is None):
            print("EPS1DOT and EPS2DOT are not included in the model.")
        else:
            #raise ValueError("There is an oddity with the inclusion/exclusion in the model of EPS1DOT and/or EPS2DOT.")
            raise ValueError("Please check: there is a problem with the inclusion and/or exclusion of EPS1DOT and/or EPS2DOT in the model.")

        if (model.H4.value is not None):
            check_if_fit(model, "H4")
        else:
            print("H4 is not included in the model")


    ###################
    ### Start of DD ###
    ###################

    elif name == "DD":
        check_if_fit(model, "A1" "E", "T0", "PB", "OM")
 
        if (model.SINI.value is not None) and (model.M2.value is not None):
            check_if_fit(model, "M2", "SINI")
        elif (model.SINI.value is not None) and (model.M2.value is None):
            raise ValueError("Both SINI and M2 are required when fitting for companion mass.")
        elif (model.SINI.value is None) and (model.M2.value is not None):
            raise ValueError("Both SINI and M2 are required when fitting for companion mass.")
        elif (model.SINI.value is None) and (model.M2.value is None):
            print("M2 and SINI are not included in the model.")
        else:
            raise ValueError("Please check: there is a problem with the inclusion and/or exclusion of M2 and/or SINI in the model.")
                
        if model.A1DOT.value is not None:
            check_if_fit(model, "A1DOT")
        else:
            print("A1DOT is not included in the model.")

        if model.PBDOT.value is not None:
            check_if_fit(model, "PBDOT")
        else:
            print("PBDOT is not included in the model.")

        if model.EDOT is not None:
            check_if_fit(model, "EDOT")
        else:
            print("EDOT is not included in the model.")

        if model.OMDOT is not None:
            check_if_fit(model, "OMDOT")
        else:
            print("OMDOT is not included in the model.")

        if model.GAMMA is not None:
            check_if_fit(model,"GAMMA")
        else:
            print("GAMMA is not included in the model.")

        # Include DTHETA check? #


    ####################
    ### Start of DDK ###
    ####################

    elif name == "DDK":
        check_if_fit(model, "A1", "ECC", "T0", "PB", "OM", "M2", "K96", "KOM", "KIN")
 
        if model.A1DOT.value is not None:
            print("Please remove A1DOT from this DDK parfile.")

        if model.OMDOT.value is not None:
            print("Please remove OMDOT from this DDK parfile.")

        if model.PBDOT.value is not None:
            check_if_fit(model, "PBDOT")
        else:
            print("PBDOT is not included in the model.")

        if model.EDOT is not None:
            check_if_fit(model, "EDOT")
        else:
            print("EDOT is not included in the model.")

        if model.GAMMA is not None:
            check_if_fit(model,"GAMMA")
        else:
            print("GAMMA is not included in the model.")



    """
    elif name == "T2":

        ### Are there plans for PINT to support the T2 and/or DDK model?

        check_if_fit(model, "A1", "ECC", "T0", "PB", "OM", "M2", "KOM", "KIN")
        # maybe check if SINI = KIN?
    else:
        raise ValueError("Atypical binary model used")
    """


def check_jumps(model,receivers,toa_type=None):
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
    if toa_type == 'WB':
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
    'ecliptic', 'troposphere', 'planet_shapiro', 'bad_lo_range','toa_release']):
    """Umbrella function to run numerous check_ functions."""
    requires_model = {'name', 'ecliptic', 'troposphere', 'planet_shapiro'}
    requires_toas = {'ephem', 'bipm', 'bad_lo_range', 'toa_release'}
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

def check_toa_release(toas):
    """Check: TOAs being used are from the most recent release

    Parameters
    ==========
    toas: `pint.toa.TOAs` object
    """
    release_flags = toas.get_flag_value('ver')[0]
    if len(set(release_flags)) > 1:
        log.error(f'TOAs from multiple releases should not be combined: {set(release_flags)}')
    else:
        if release_flags[0] == LATEST_TOA_RELEASE:
            log.info(f'All TOAs are from the latest release ({LATEST_TOA_RELEASE}).')
        else:
            log.warning(f'TOAs in use are from an old release {release_flags[0]}, not {LATEST_TOA_RELEASE}; update tim-directory in the .yaml accordingly.')


