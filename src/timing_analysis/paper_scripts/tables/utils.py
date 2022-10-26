#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import numpy as np
import astropy.units as u
from astropy import log
from pint.utils import weighted_mean
import pint.residuals as Resid
import pint.models.parameter
from pint.models import get_model
import os
import time
from subprocess import check_output, check_call, CalledProcessError
import glob
# Import some software so we have appropriate versions
import pint
import astropy
import scipy.stats
import copy

ALPHA = 0.0027

def whiten_resids(fitter, restype = 'postfit'):
    """
    Function to whiten residuals. If no reddened residuals, input will be returned.

    Inputs:
    ---------
    fitter [object, dictionary]: PINT fitter class or dictionary output from ecorr_average() function.
    restype ['string']: Type of residuals, pre or post fit, to plot from fitter object. Options are:
        'prefit' - plot the prefit residuals.
        'postfit' - plot the postfit residuals (default)

    Returns:
    ---------
    wres [array]: Array of whitened timing residuals.
    """
    # Check if input is the epoch averaged dictionary, should only be used if epoch averaged NB TOAs
    if type(fitter) is dict:
        rs = fitter['time_resids']
        noise_rs = fitter['noise_resids']
        # Now check if red noise residuals
        if "pl_red_noise" in noise_rs:
            wres = rs - noise_rs['pl_red_noise']
        else:
            log.warning("No red noise, residuals already white. Returning input residuals...")
            wres = rs
    # if not assume it's a PINT fitter class object
    else:
        # Check if WB or NB
        if fitter.is_wideband:
            if restype == 'postfit':
                time_resids = fitter.resids.residual_objs['toa'].time_resids
                noise_resids = fitter.resids.noise_resids
            else:
                time_resids = fitter.resids_init.residual_objs['toa'].time_resids
                noise_resids = fitter.resids_init.noise_resids
        else:
            if restype == 'postfit':
                time_resids = fitter.resids.time_resids
                noise_resids = fitter.resids.noise_resids
            elif restype == 'prefit':
                time_resids = fitter.resids_init.time_resids
                noise_resids = fitter.resids_init.noise_resids
            else:
                raise ValueError("Unrecognized residual type: %s. Please choose from 'prefit' or 'postfit'."%(restype))
            # Get number of residuals
        num_res = len(time_resids)
        # Check that the key is in the dictionary
        if "pl_red_noise" in noise_resids:
            wres = time_resids - noise_resids['pl_red_noise'][:num_res]
        else:
            log.warning("No red noise, residuals already white. Returning input residuals...")
            wres = time_resids
    return wres

def rms_by_backend(resids, errors, rcvr_backends, dm = False):
    """
    Function to take a fitter, list of residuals errors, and backends and compute the rms and weighted rms
    for either time residuals or DM residuals if a wideband residuals.

    Inputs:
    ----------
    resids [list]: List of residuals.
    errors [list]: List of residual errors.
    rcvr_backends [list]: List of backends.
    dm [boolean]: If True, will do computation with DM residuals [defaut: False].
    
    Returns:
    ----------
    rs_dict [dictionary]: Dictionary of rms and wieghted rms residuals for each backend-reciever combination.
    """
    # Define output dictionary
    rs_dict = {}
    # Get RMS of all residuals
    # Now loop through and compute on a per receiver-backend status
    RCVR_BCKNDS = np.sort(list(set(rcvr_backends)))
    if dm:
        avg_RMS_ALL = np.std(resids)
        # Get the weighted rms of averaged residuals
        weights = 1.0 / (errors ** 2)
        wmean, werr, wsdev = weighted_mean(resids, weights, sdev=True)
        avg_WRMS_ALL = wsdev
        # Add to dictionary
        rs_dict['All'] = {'rms':avg_RMS_ALL, 'wrms':avg_WRMS_ALL}
        for r_b in RCVR_BCKNDS:
            # Get indices of receiver-backend toas
            inds = np.where(rcvr_backends==r_b)[0]
            # Select them
            rms = np.std(resids[inds])
            weights = 1.0 / (errors ** 2)
            wmean, werr, wsdev = weighted_mean(resids[inds], weights[inds], sdev=True)
            wrms = wsdev
            rs_dict[r_b] = {'rms':rms, 'wrms':wrms}
    else:
        avg_RMS_ALL = np.std(resids).to(u.us)
        # Get the weighted rms of averaged residuals
        weights = 1.0 / (errors.to(u.s) ** 2)
        wmean, werr, wsdev = weighted_mean(resids, weights, sdev=True)
        avg_WRMS_ALL = wsdev.to(u.us)
        # Add to dictionary
        rs_dict['All'] = {'rms':avg_RMS_ALL, 'wrms':avg_WRMS_ALL}
        for r_b in RCVR_BCKNDS:
            # Get indices of receiver-backend toas
            inds = np.where(rcvr_backends==r_b)[0]
            # Select them
            rms = np.std(resids[inds].to(u.us))
            weights = 1.0 / (errors.to(u.s) ** 2)
            wmean, werr, wsdev = weighted_mean(resids[inds], weights[inds], sdev=True)
            wrms = wsdev.to(u.us)
            rs_dict[r_b] = {'rms':rms, 'wrms':wrms}
    # return the dictionary
    return rs_dict


def resid_stats(fitter, epoch_avg = False, whitened = False, dm_stats = False, print_pretty = False):
    """
    Function to get statistics for the residuals. This includes the RMS and WRMS for all residuals, as well as
    per-backend. Option for epoch averaged or not epoch averaged. If dm_stats are also returned, then there
    will be a second output dictionary for the DM stats by receiver-backend combo.

    Inputs:
    ----------
    fitter [object]: PINT fitter class object, post-fit.
    epoch_avg [boolean]: If True, will output stats for epoch averaged residulas, else will be for 
        non-epoch averaged residuals [default: False].
    whitened [boolean]: If True, will output stats for whitened residulas, else will be for 
        non-whitened residuals [default: False].
    dm_stats [boolean]: If True, will also output the stats for the DM residuals for wideband fitters.
        Note this will return an additional dictionary, dm_stats [default: False].
    print_pretty [boolean]: If True, will nicely print the W/RMS per receiver-backend combo [default: False].

    Returns:
    ----------
    rs_dict [nested dictionary]:

        First set of keys are Receiver-Backend combos, e.g. L-wide_ASP, S-wide_PUPPI. For the W/RMS of all residuals,
        the key is 'All'.

        Within each Receiver-Backend combo the flags are:
            rms [astropy quantity: rms of the residuals for the receiver-backend combo in microseconds.
            wrms [astropy quantity: weighted rms of the residuals for the receiver-backend combo in microseconds.
     
    dm_dict [nested dictionary]: Same as rs_dict but for DM residuals with unit of pc cm^-3.
    """
    # Check if fitter is WB or not
    if fitter.is_wideband:
        resids = fitter.resids.residual_objs['toa']
        dm_resids = fitter.resids.residual_objs['dm']
        NB = False
        if epoch_avg:
            log.warning("Warning, cannot epoch average wideband residuals, will skip epoch averaging.")
    else:
        resids = fitter.resids
        NB = True

    # get rcvr backend combos for averaged residuals
    rcvr_bcknds = np.array(resids.toas.get_flag_value('f')[0])

    # Compute epoch averaged stats
    if epoch_avg and NB:
        avg = fitter.resids.ecorr_average(use_noise_model=True)
        avg_rcvr_bcknds = []
        for iis in avg['indices']:
            avg_rcvr_bcknds.append(rcvr_bcknds[iis[0]])
        avg_rcvr_bcknds = np.array(avg_rcvr_bcknds)
        # compute averaged, whitened
        if whitened:
            wres_avg = whiten_resids(avg)
            rs_dict = rms_by_backend(wres_avg.to(u.us), avg['errors'], avg_rcvr_bcknds)
        # compute averaged
        else:
            rs_dict = rms_by_backend(avg['time_resids'], avg['errors'], avg_rcvr_bcknds)

    # Compute whitened
    elif whitened:
        wres = whiten_resids(fitter)
        #rs_dict = rms_by_backend(wres.to(u.us), fitter.toas.get_errors(), rcvr_bcknds)
        rs_dict = rms_by_backend(wres.to(u.us), resids.get_data_error(), rcvr_bcknds)

    # If not averaged or whitened, compute with functions that already exist
    if not epoch_avg and not whitened:
        # Define dictionary for return values
        rs_dict = {}
        # Get total RMS and WRMS
        RMS_ALL = resids.time_resids.std().to(u.us) # astropy quantity
        WRMS_ALL = resids.rms_weighted() # astropy quantity, units are us

        # Now split up by backend
        RCVR_BCKNDS = np.sort(list(set(rcvr_bcknds)))
        # Turn into dictionary to return
        rs_dict['All'] = {'rms':RMS_ALL, 'wrms':WRMS_ALL}
        for r_b in RCVR_BCKNDS:
            # Get indices of receiver-backend toas
            inds = np.where(rcvr_bcknds==r_b)[0]
            # Select them
            fitter.toas.select(inds)
            # Create new residual object
            r = Resid.Residuals(fitter.toas, fitter.model)
            # Get new RMS, WRMS
            rs_dict[r_b] = {'rms':r.time_resids.std().to(u.us), 'wrms':r.rms_weighted()}
            # Unselect them
            fitter.toas.unselect()

    # print output if desired
    if print_pretty:
        rs_keys = rs_dict.keys()
        for k in rs_keys:
            l = "# WRMS(%s) = %.3f %s" %(k, rs_dict[k]['wrms'].value, rs_dict[k]['wrms'].unit)
            print(l)
            l = "#  RMS(%s) = %.3f %s" %(k, rs_dict[k]['rms'].value, rs_dict[k]['rms'].unit)
            print(l)

    # Check if dm stats are desired
    if dm_stats:
        if not NB:
            dm_dict = rms_by_backend(dm_resids.resids, dm_resids.get_data_error(), rcvr_bcknds, dm = True)
            if print_pretty:
                print()
                dm_keys = dm_dict.keys()
                for k in rs_keys:
                    l = "# WRMS(%s) = %.6f %s" %(k, dm_dict[k]['wrms'].value, dm_dict[k]['wrms'].unit)
                    print(l)
                    l = "#  RMS(%s) = %.6f %s" %(k, dm_dict[k]['rms'].value, dm_dict[k]['rms'].unit)
                    print(l)
        else:
            log.warning("Cannot compute DM Stats, not Wideband timing data.")

    # Return the dictionary
    if dm_stats and not NB:
        return rs_dict, dm_dict
    else:
        return rs_dict
