#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:30:59 2020

@author: bshapiroalbert
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from astropy import log
import astropy.units as u
# Import PINT
import pint.toa as toa
import pint.models as model
import pint.fitter as fitter
import pint.utils as pu
import subprocess
# import extra util functions brent wrote
import timing_analysis.utils as ub
import os

# color blind friends colors and markers?
#CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
#MARKERS = ['.', 'v', 's', 'x', '^', 'D', 'p', 'P', '*']

# Color scheme for consistent reciever-backend combos, same as published 12.5 yr
colorschemes = {'thankful_2':{
              "327_ASP":         "#BE0119",
              "327_PUPPI":       "#BE0119",
              "430_ASP":         "#FD9927",
              "430_PUPPI":       "#FD9927",
              "L-wide_ASP":      "#6BA9E2",
              "L-wide_PUPPI":    "#6BA9E2",
              "Rcvr1_2_GASP":    "#407BD5",
              "Rcvr1_2_GUPPI":   "#407BD5",
              "Rcvr_800_GASP":   "#61C853",
              "Rcvr_800_GUPPI":  "#61C853",
              "S-wide_ASP":      "#855CA0",
              "S-wide_PUPPI":    "#855CA0",
             }}
# marker dictionary to be used if desired, currently all 'x'
markers = {"327_ASP":        "x",
          "327_PUPPI":       "x",
          "430_ASP":         "x",
          "430_PUPPI":       "x",
          "L-wide_ASP":      "x",
          "L-wide_PUPPI":    "x",
          "Rcvr1_2_GASP":    "x",
          "Rcvr1_2_GUPPI":   "x",
          "Rcvr_800_GASP":   "x",
          "Rcvr_800_GUPPI":  "x",
          "S-wide_ASP":      "x",
          "S-wide_PUPPI":    "x",
            }
# Define the color map option
colorscheme = colorschemes['thankful_2']

def call(x):
    subprocess.call(x,shell=True)

def plot_residuals_time(fitter, restype = 'postfit', plotsig = False, avg = False, whitened = False, \
                        save = False, legend = True, title = True, axs = None, **kwargs):
    """
    Make a plot of the residuals vs. time


    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    restype ['string'] : Type of residuals, pre or post fit, to plot from fitter object. Options are:
        'prefit' - plot the prefit residuals.
        'postfit' - plot the postfit residuals (default)
        'both' - overplot both the pre and post-fit residuals.
    plotsig [boolean] : If True plot number of measurements v. residuals/uncertainty, else v. residuals
        [default: False].
    avg [boolean] : If True and not wideband fitter, will compute and plot epoch-average residuals [default: False].
    whitened [boolean] : If True will compute and plot whitened residuals [default: False].
    save [boolean] : If True will save plot with the name "resid_v_mjd.png" Will add averaged/whitened
         as necessary [default: False].
    legend [boolean] : If False, will not print legend with plot [default: True].
    title [boolean] : If False, will not print plot title [default: True].
    axs [string] : If not None, should be defined subplot value and the figure will be used as part of a
         larger figure [default: None].
    
    Optional Arguements:
    --------------------
    res [list/array] : List or array of residual values to plot. Will override values from fitter object.
    errs [list/array] : List or array of residual error values to plot. Will override values from fitter object.
    mjds [list/array] : List or array of TOA MJDs to plot. Will override values from toa object.
    rcvr_bcknds[list/array] : List or array of TOA receiver-backend combinations. Will override values from toa object.
    figsize [tuple] : Size of the figure passed to matplotlib [default: (10,4)].
    fmt ['string'] : matplotlib format option for markers [default: ('x')]
    color ['string'] : matplotlib color option for plot [default: color dictionary in plot_utils.py file]
    alpha [float] : matplotlib alpha options for plot points [default: 0.5]
    """
    # Check if wideband
    if "Wideband" in fitter.__class__.__name__:
        NB = False
        if avg == True:
            raise ValueError("Cannot epoch average wideband residuals, please change 'avg' to False.")
    else:
        NB = True
    
    # Check if want epoch averaged residuals
    if avg == True and restype == 'prefit':
        avg_dict = fitter.resids_init.ecorr_average(use_noise_model=True)
    elif avg == True and restype == 'postfit':
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
    elif avg == True and restype == 'both':
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
        avg_dict_pre = fitter.resids_init.ecorr_average(use_noise_model=True)
        
        
    # Get residuals
    if 'res' in kwargs.keys():
        res = kwargs['res']
    else:
        if restype == 'prefit':
            if NB == True:
                if avg == True:
                    res = avg_dict['time_resids'].to(u.us)
                else:
                    res = fitter.resids_init.time_resids.to(u.us)
            else:
                res = fitter.resids_init.residual_objs['toa'].time_resids.to(u.us)
        elif restype == 'postfit':
            if NB == True:
                if avg == True:
                    res = avg_dict['time_resids'].to(u.us)
                else:
                    res = fitter.resids.time_resids.to(u.us)
            else:
                res = fitter.resids.residual_objs['toa'].time_resids.to(u.us)
        elif restype == 'both':
            if NB == True:
                if avg == True:
                    res = avg_dict['time_resids'].to(u.us)
                    res_pre = avg_dict_pre['time_resids'].to(u.us)
                else:
                    res = fitter.resids.time_resids.to(u.us)
                    res_pre = fitter.resids_init.time_resids.to(u.us)
            else:
                res = fitter.resids.residual_objs['toa'].time_resids.to(u.us)
                res_pre = fitter.resids_init.residual_objs['toa'].time_resids.to(u.us)
        else:
            raise ValueError("Unrecognized residual type: %s. Please choose from 'prefit', 'postfit', or 'both'."\
                             %(restype))
    
    # Check if we want whitened residuals
    if whitened == True and ('res' not in kwargs.keys()):
        if avg == True:
            if restype != 'both':
                res = ngu.whiten_resids(avg_dict, restype=restype)
            else:
                res = ngu.whiten_resids(avg_dict_pre, restype='prefit')
                res_pre = ngu.whiten_resids(avg_dict, restype='postfit')
                res_pre = res_pre.to(u.us)
            res = res.to(u.us)    
        else:
            if restype != 'both':
                res = ngu.whiten_resids(fitter, restype=restype)
            else:
                res = ngu.whiten_resids(fitter, restype='prefit')
                res_pre = ngu.whiten_resids(fitter, restype='postfit')
                res_pre = res_pre.to(u.us)
            res = res.to(u.us)
    
    # Get errors
    if 'errs' in kwargs.keys():
        errs = kwargs['errs']
    else:
        if restype == 'prefit':
            if avg == True:
                errs = avg_dict['errors'].to(u.us)
            else:
                errs = fitter.toas.get_errors().to(u.us)
        elif restype == 'postfit':
            if NB == True:
                if avg == True:
                    errs = avg_dict['errors'].to(u.us)
                else:
                    errs = fitter.resids.get_data_error().to(u.us)
            else:
                errs = fitter.resids.residual_objs['toa'].get_data_error().to(u.us)
        elif restype == 'both':
            if NB == True:
                if avg == True:
                    errs = avg_dict['errors'].to(u.us)
                    errs_pre = avg_dict_pre['errors'].to(u.us)
                else:
                    errs = fitter.resids.get_data_error().to(u.us)
                    errs_pre = fitter.toas.get_errors().to(u.us)
            else:
                errs = fitter.resids.residual_objs['toa'].get_data_error().to(u.us)
                errs_pre = fitter.toas.get_errors().to(u.us)
    
    # Get MJDs
    if 'mjds' in kwargs.keys():
        mjds = kwargs['mjds']
    else:
        mjds = fitter.toas.get_mjds().value
        if avg == True:
            mjds = avg_dict['mjds'].value
    # Convert to years
    years = (mjds - 51544.0)/365.25 + 2000.0
    
    # Get receiver backends
    if 'rcvr_bcknds' in kwargs.keys():
        rcvr_bcknds = kwargs['rcvr_bcknds']
    else:
        rcvr_bcknds = np.array(fitter.toas.get_flag_value('f')[0])
        if avg == True:
            avg_rcvr_bcknds = []
            for iis in avg_dict['indices']:
                avg_rcvr_bcknds.append(rcvr_bcknds[iis[0]])
            rcvr_bcknds = np.array(avg_rcvr_bcknds)
    # Get the set of unique receiver-bandend combos
    RCVR_BCKNDS = set(rcvr_bcknds)

    if axs == None:
        if 'figsize' in kwargs.keys():
            figsize = kwargs['figsize']
        else:
            figsize = (10,4)
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
    else:
        ax1 = axs
    for i, r_b in enumerate(RCVR_BCKNDS):
        inds = np.where(rcvr_bcknds==r_b)[0]
        if not inds.tolist():
            r_b_label = ""
        else:
            r_b_label = rcvr_bcknds[inds][0]
        # Get plot preferences
        if 'fmt' in kwargs.keys():
            mkr = kwargs['fmt']
        else:
            mkr = markers[r_b_label]
            if restype == 'both':
                mkr_pre = '.'
        if 'color' in kwargs.keys():
            clr = kwargs['color']
        else:
            clr = colorscheme[r_b_label]
        if 'alpha' in kwargs.keys():
            alpha = kwargs['alpha']
        else:
            alpha = 0.5
        if plotsig:
            sig = res[inds]/errs[inds]
            ax1.errorbar(years[inds], sig, yerr=len(errs[inds])*[1], fmt=mkr, \
                     color=clr, label=r_b_label, alpha = alpha, picker=True)
            if restype == 'both':
                sig_pre = res_pre[inds]/errs_pre[inds]
                ax1.errorbar(years[inds], sig_pre, yerr=len(errs_pre[inds])*[1], fmt=mkr_pre, \
                         color=clr, label=r_b_label+" Prefit", alpha = alpha, picker=True)
        else:
            ax1.errorbar(years[inds], res[inds], yerr=errs[inds], fmt=mkr, \
                     color=clr, label=r_b_label, alpha = alpha, picker=True)
            if restype == 'both':
                ax1.errorbar(years[inds], res_pre[inds], yerr=errs_pre[inds], fmt=mkr_pre, \
                     color=clr, label=r_b_label+" Prefit", alpha = alpha, picker=True)
            
    # Set second axis
    ax1.set_xlabel(r'Year')
    ax1.grid(True)
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    mjd0  = ((ax1.get_xlim()[0])-2004.0)*365.25+53005.
    mjd1  = ((ax1.get_xlim()[1])-2004.0)*365.25+53005.
    if (mjd1-mjd0>1200.): mjdstep=500.
    elif (mjd1-mjd0>600.): mjdstep=200.
    else: mjdstep=100.
    mjd0 = int(mjd0/mjdstep)*mjdstep + mjdstep
    mjd1 = int(mjd1/mjdstep)*mjdstep
    mjdr = np.arange(mjd0,mjd1+mjdstep,mjdstep)
    yrr = (mjdr - 51544.0)/365.25 + 2000.0
    ax2.set_xticks(yrr)
    ax2.set_xticklabels(["%5d" % m for m in mjdr])
    if plotsig:
        if avg and whitened:
            ax1.set_ylabel('Average Residual/Uncertainty \n (Whitened)', multialignment='center')
        elif avg and not whitened:
            ax1.set_ylabel('Average Residual/Uncertainty')
        elif whitened and not avg:
            ax1.set_ylabel('Residual/Uncertainty \n (Whitened)', multialignment='center')
        else:
            ax1.set_ylabel('Residual/Uncertainty')
    else:
        if avg and whitened:
            ax1.set_ylabel('Average Residual ($\mu$s) \n (Whitened)', multialignment='center')
        elif avg and not whitened:
            ax1.set_ylabel('Average Residual ($\mu$s)')
        elif whitened and not avg:
            ax1.set_ylabel('Residual ($\mu$s) \n (Whitened)', multialignment='center')
        else:
            ax1.set_ylabel('Residual ($\mu$s)')
    if legend:
        ax1.legend(loc='upper center', bbox_to_anchor= (0.5, 1.0+1.0/figsize[1]), ncol=len(RCVR_BCKNDS))
    if title:
        plt.title("%s %s timing residuals" % (fitter.model.PSR.value, restype), y=1.0+1.0/figsize[1])
    if axs == None:
        plt.tight_layout()
    if save:
        ext = ""
        if whitened:
            ext += "_whitened"
        if avg:
            ext += "_averaged"
        if NB:
            ext += "_NB"
        else:
            ext += "_WB"
        if restype == 'prefit':
            ext += "_prefit"
        elif restype == 'postfit':
            ext += "_postfit"
        elif restype == "both":
            ext += "_pre_post_fit"
        plt.savefig("%s_resid_v_mjd%s.png" % (fitter.model.PSR.value, ext))
        
    return

def plot_dmx_time(toas, fitter, figsize=(10,4), savedmx = False, save = False, legend = True, axs = None, NB = False, \
                  nbDMXfile = None, compare = False):
    """
    Make a plot of DMX vs. time


    Arguments
    ---------
    toas: PINT toas
    res: residuals generated by PINT
    figsize: size of the figure passed to matplotlib [default: (10,4)]
    savedmx: default is False, else input is string file name to save dmxparse values to in the style of the TEMPO
          dmxparse.py script.
    save: boolean, if True, will save plot as "dmx_v_time.png" Will add WB or compare if necessary
    legend: boolean, if True, will print legend with plots
    axs: Default is None, if not None, should be defined subplot value and the figure will be used as part of a
         larger figure.
    NB: If True, will label DMX plot as narrowband, if False (Default) assumes plotting Wideband timing DMX values.
    compare: If True and WB is True, then we will plot both the narrow band DMX values and the WB DMX values together
             as a comparison.
    nbDMXfile: Name of the narrowband dmxparse style file with DMX values, errors, and epochs.
    """
    if axs == None:
        #ipython.magic("matplotlib inline")
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
    else:
        ax1 = axs

    if NB == False or (NB == True and compare == True):
        # get dmx dictionary from pint dmxparse function
        dmx_dict = pu.dmxparse(fitter, save=savedmx)
        DMXs = dmx_dict['dmxs'].value
        DMX_vErrs = dmx_dict['dmx_verrs'].value
        DMX_center_MJD = dmx_dict['dmxeps'].value
        DMX_center_Year = (DMX_center_MJD- 51544.0)/365.25 + 2000.0

        ax1.errorbar(DMX_center_Year, DMXs*10**3, yerr=DMX_vErrs*10**3, fmt='.', c = 'gray', marker = 's', \
                         label="Wideband")

    if NB == True:
        # Make sure that narrowband DMX file exists, if not assume fitter is narrowband TOAs
        if nbDMXfile == None or not os.path.isfile(nbDMXfile):
            log.warn("Narrowband DMX file does not exist, assuming fitter is narrowband...")
            dmx_dict = pu.dmxparse(fitter, save=savedmx)
            nb_dmx = dmx_dict['dmxs'].value
            nb_dmx_var = dmx_dict['dmx_verrs'].value
            dmx_epochs = dmx_dict['dmxeps'].value
        # Otherwise get values from input NB dmxparse out file
        else:
            # Get the values from the DMX parse file
            dmx_epochs, nb_dmx, nb_dmx_var, nb_dmx_r1, nb_dmx_r2 = np.loadtxt(nbDMXfile, unpack=True, \
                                                                              usecols=(0,1,2,3,4))
        dmx_mid_yr = (dmx_epochs- 51544.0)/365.25 + 2000.0
        # Now plot these points regardless
        ax1.errorbar(dmx_mid_yr, nb_dmx*10**3, yerr = nb_dmx_var*10**3,\
                     fmt = '.', color = 'k', marker = 'o', \
                     label='Narrowband')

    ax1.set_xlabel(r'Year')
    ax1.grid(True)
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    mjd0  = ((ax1.get_xlim()[0])-2004.0)*365.25+53005.
    mjd1  = ((ax1.get_xlim()[1])-2004.0)*365.25+53005.
    if (mjd1-mjd0>1200.): mjdstep=500.
    elif (mjd1-mjd0>600.): mjdstep=200.
    else: mjdstep=100.
    mjd0 = int(mjd0/mjdstep)*mjdstep + mjdstep
    mjd1 = int(mjd1/mjdstep)*mjdstep
    mjdr = np.arange(mjd0,mjd1+mjdstep,mjdstep)
    yrr = (mjdr - 51544.0)/365.25 + 2000.0
    ax2.set_xticks(yrr)
    ax2.set_xticklabels(["%5d" % m for m in mjdr])
    ax1.set_ylabel(r"DMX ($10^{-3}$ pc cm$^{-3}$)")
    if legend:
        ax1.legend(loc='best')
    if axs == None:
        plt.tight_layout()
    if save:
        ext = ""
        if NB and not compare:
            ext += "_NB"
        if compare:
            ext += "_compare"
        plt.savefig("dmx_v_time%s.png" % (ext))
    return

# Now we want to make wideband DM vs. time plot, this uses the premade dm_resids from PINT
def plot_wb_dm_time(toas, fitter, figsize=(10,4), save = False, legend = True, axs = None, mean_sub = True):
    """
    Make a plot of Wideband timing DM vs. time


    Arguments
    ---------
    toas: PINT toas
    res: residuals generated by PINT
    figsize: size of the figure passed to matplotlib [default: (10,4)]
    save: boolean, if True, will save plot as "dmx_v_time.png" Will add WB or compare if necessary
    legend: boolean, if True, will print legend with plots
    axs: Default is None, if not None, should be defined subplot value and the figure will be used as part of a
         larger figure.
    mean_sub: boolean, if True will mean subtract DM values, otherwise will not.
    """
    if axs == None:
        #ipython.magic("matplotlib inline")
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
    else:
        ax1 = axs

    # Get the DM residuals
    dm_resids = fitter.resids.residual_objs[1].resids.value
    dm_error = fitter.resids.residual_objs[1].data_error.value

    # If we don't want mean subtraced data we add the mean
    if not mean_sub:
        DM0 = np.average(fitter.resids.residual_objs[1].dm_data,
                         weights=(fitter.resids.residual_objs[1].dm_error)**-2)
        dm_resids += DM0.value
        ylabel = r"DM [cm$^{-3}$ pc]"
    else:
        ylabel = r"$\Delta$DM [cm$^{-3}$ pc]"

    # Now we need to get some of the values from the toas
    mjds = toas.get_mjds().value
    years = (mjds - 51544.0)/365.25 + 2000.0
    rcvr_bcknds = np.array(toas.get_flag_value('f')[0])
    RCVR_BCKNDS = set(rcvr_bcknds)
    rcvrs = np.array(toas.get_flag_value('fe')[0])
    RCVRS = set(rcvrs)

    # Now plot them by reciever backend
    for i, r_b in enumerate(RCVR_BCKNDS):
        inds = np.where(rcvr_bcknds==r_b)[0]
        if not inds.tolist():
            r_b_label = ""
        else:
            r_b_label = rcvr_bcknds[inds][0]

        # Do plotting command
        ax1.errorbar(years[inds], dm_resids[inds], yerr=dm_error[inds], fmt=markers[r_b_label], \
                     color=colorscheme[r_b_label], label=r_b_label, alpha = 0.5)

    # Now add the rest of the plotting commands
    ax1.set_xlabel(r'Year')
    ax1.grid(True)
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    mjd0  = ((ax1.get_xlim()[0])-2004.0)*365.25+53005.
    mjd1  = ((ax1.get_xlim()[1])-2004.0)*365.25+53005.
    if (mjd1-mjd0>1200.): mjdstep=500.
    elif (mjd1-mjd0>600.): mjdstep=200.
    else: mjdstep=100.
    mjd0 = int(mjd0/mjdstep)*mjdstep + mjdstep
    mjd1 = int(mjd1/mjdstep)*mjdstep
    mjdr = np.arange(mjd0,mjd1+mjdstep,mjdstep)
    yrr = (mjdr - 51544.0)/365.25 + 2000.0
    ax2.set_xticks(yrr)
    ax2.set_xticklabels(["%5d" % m for m in mjdr])
    ax1.set_ylabel(ylabel)

    if legend:
        ax1.legend(loc='upper center', bbox_to_anchor= (0.5, 1.0+1.0/figsize[1]), ncol=len(RCVR_BCKNDS))
    if axs == None:
        plt.tight_layout()
    if save:
        plt.savefig("%s_dm_v_time.png" % (fitter.model.PSR.name))
    return

def measurements_v_res(toas, res, nbin = 50, figsize=(10,4), plotsig=False, fromPINT = True, errs = None,\
                        rcvr_bcknds = None, avg = False, whitened = False, save = False, legend = True, axs = None):
    """
    Make a histogram of number of measurements v. residuals


    Arguments
    ---------
    toas: PINT toas
    res: residuals generated by PINT
    nbin: int number of bins desired in the histogram. default is 50
    figsize: size of the figure passed to matplotlib [default: (10,4)]
    plotsig: boolean, if True plot number of measurements v. residuals/uncertainty, else v. residuals
    fromPINT: boolean, if True, will get values from PINT objects, if False, will expect necessary values from
              other inputs
    errs: Default in None. If fromPINT is False, errs is an array of residual errors in units of microseconds
    rcvr_bcknds: Default in None. If fromPINT is False, rcvr_bcknds is an array of strings of reciever-banckend combos
    avg: boolean, if True will change x-axis to "Average Residual"
    whitened: boolean, if True will change x-axis to "Residual (Whitened)"
    save: boolean, if True will save plot with the name "measurements_v_resids.png" Will add averaged/whitened
         as necessary
    legend: boolean, if True will print legend with plots
    axs: Default is None, if not None, should be defined subplot value and the figure will be used as part of a
         larger figure.
    """
    if fromPINT:
        errs = toas.get_errors().value
        rcvr_bcknds = np.array(toas.get_flag_value('f')[0])
        RCVR_BCKNDS = set(rcvr_bcknds)
    else:
        errs = errs
        RCVR_BCKNDS = set(rcvr_bcknds)

    if axs == None:
        #ipython.magic("matplotlib inline")
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
    else:
        ax1 = axs
    xmax=0
    for i, r_b in enumerate(RCVR_BCKNDS):
        inds = np.where(rcvr_bcknds==r_b)[0]
        if not inds.tolist():
            r_b_label = ""
        else:
            r_b_label = rcvr_bcknds[inds][0]
        if plotsig:
            sig = res[inds]/errs[inds]
            ax1.hist(sig,nbin,histtype='step', color=colorscheme[r_b_label], label=r_b_label)
            xmax = max(xmax,max(sig),max(-sig))
        else:
            ax1.hist(res[inds],nbin,histtype='step', color=colorscheme[r_b_label], label=r_b_label)
            xmax = max(xmax,max(res[inds]),max(-res[inds]))
    ax1.grid(True)
    ax1.set_ylabel("Number of measurements")
    if plotsig:
        if avg and whitened:
            ax1.set_xlabel('Average Residual/Uncertainty \n (Whitened)', multialignment='center')
        elif avg and not whitened:
            ax1.set_xlabel('Average Residual/Uncertainty')
        elif whitened and not avg:
            ax1.set_xlabel('Residual/Uncertainty \n (Whitened)', multialignment='center')
        else:
            ax1.set_xlabel('Residual/Uncertainty')
    else:
        if avg and whitened:
            ax1.set_xlabel('Average Residual ($\mu$s) \n (Whitened)', multialignment='center')
        elif avg and not whitened:
            ax1.set_xlabel('Average Residual ($\mu$s)')
        elif whitened and not avg:
            ax1.set_xlabel('Residual ($\mu$s) \n (Whitened)', multialignment='center')
        else:
            ax1.set_xlabel('Residual ($\mu$s)')
    ax1.set_xlim(-1.1*xmax,1.1*xmax)
    if legend:
        ax1.legend(loc='upper center', bbox_to_anchor= (0.5, 1.0+1.0/figsize[1]), ncol=len(RCVR_BCKNDS))
    if axs == None:
        plt.tight_layout()
    if save:
        ext = ""
        if whitened:
            ext += "_whitened"
        if avg:
            ext += "_averaged"
        plt.savefig("measurements_v_resid%s.png" % (ext))
    return


def plot_residuals_orb(fitter, restype = 'postfit', plotsig = False, avg = False, whitened = False, \
                        save = False, legend = True, title = True, axs = None, **kwargs):
    """
    Make a plot of the residuals vs. orbital phase.
    NOTE - CURRENTLY CANNOT PLOT EPOCH AVERAGED RESIDUALS


    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    restype ['string'] : Type of residuals, pre or post fit, to plot from fitter object. Options are:
        'prefit' - plot the prefit residuals.
        'postfit' - plot the postfit residuals (default)
        'both' - overplot both the pre and post-fit residuals.
    plotsig [boolean] : If True plot number of measurements v. residuals/uncertainty, else v. residuals
        [default: False].
    avg [boolean] : If True and not wideband fitter, will compute and plot epoch-average residuals [default: False].
    whitened [boolean] : If True will compute and plot whitened residuals [default: False].
    save [boolean] : If True will save plot with the name "resid_v_mjd.png" Will add averaged/whitened
         as necessary [default: False].
    legend [boolean] : If False, will not print legend with plot [default: True].
    title [boolean] : If False, will not print plot title [default: True].
    axs [string] : If not None, should be defined subplot value and the figure will be used as part of a
         larger figure [default: None].
    
    Optional Arguements:
    --------------------
    res [list/array] : List or array of residual values to plot. Will override values from fitter object.
    errs [list/array] : List or array of residual error values to plot. Will override values from fitter object.
    orbphase [list/array] : List or array of orbital phases to plot. Will override values from model object.
    rcvr_bcknds[list/array] : List or array of TOA receiver-backend combinations. Will override values from toa object.
    figsize [tuple] : Size of the figure passed to matplotlib [default: (10,4)].
    fmt ['string'] : matplotlib format option for markers [default: ('x')]
    color ['string'] : matplotlib color option for plot [default: color dictionary in plot_utils.py file]
    alpha [float] : matplotlib alpha options for plot points [default: 0.5]
    """
    # Check if wideband
    if "Wideband" in fitter.__class__.__name__:
        NB = False
        if avg == True:
            raise ValueError("Cannot epoch average wideband residuals, please change 'avg' to False.")
    else:
        NB = True
    
    # Check if want epoch averaged residuals
    if avg == True and restype == 'prefit':
        avg_dict = fitter.resids_init.ecorr_average(use_noise_model=True)
    elif avg == True and restype == 'postfit':
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
    elif avg == True and restype == 'both':
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
        avg_dict_pre = fitter.resids_init.ecorr_average(use_noise_model=True)
        
        
    # Get residuals
    if 'res' in kwargs.keys():
        res = kwargs['res']
    else:
        if restype == 'prefit':
            if NB == True:
                if avg == True:
                    res = avg_dict['time_resids'].to(u.us)
                else:
                    res = fitter.resids_init.time_resids.to(u.us)
            else:
                res = fitter.resids_init.residual_objs['toa'].time_resids.to(u.us)
        elif restype == 'postfit':
            if NB == True:
                if avg == True:
                    res = avg_dict['time_resids'].to(u.us)
                else:
                    res = fitter.resids.time_resids.to(u.us)
            else:
                res = fitter.resids.residual_objs['toa'].time_resids.to(u.us)
        elif restype == 'both':
            if NB == True:
                if avg == True:
                    res = avg_dict['time_resids'].to(u.us)
                    res_pre = avg_dict_pre['time_resids'].to(u.us)
                else:
                    res = fitter.resids.time_resids.to(u.us)
                    res_pre = fitter.resids_init.time_resids.to(u.us)
            else:
                res = fitter.resids.residual_objs['toa'].time_resids.to(u.us)
                res_pre = fitter.resids_init.residual_objs['toa'].time_resids.to(u.us)
        else:
            raise ValueError("Unrecognized residual type: %s. Please choose from 'prefit', 'postfit', or 'both'."\
                             %(restype))
    
    # Check if we want whitened residuals
    if whitened == True and ('res' not in kwargs.keys()):
        if avg == True:
            if restype != 'both':
                res = ngu.whiten_resids(avg_dict, restype=restype)
            else:
                res = ngu.whiten_resids(avg_dict_pre, restype='prefit')
                res_pre = ngu.whiten_resids(avg_dict, restype='postfit')
                res_pre = res_pre.to(u.us)
            res = res.to(u.us)    
        else:
            if restype != 'both':
                res = ngu.whiten_resids(fitter, restype=restype)
            else:
                res = ngu.whiten_resids(fitter, restype='prefit')
                res_pre = ngu.whiten_resids(fitter, restype='postfit')
                res_pre = res_pre.to(u.us)
            res = res.to(u.us)
    
    # Get errors
    if 'errs' in kwargs.keys():
        errs = kwargs['errs']
    else:
        if restype == 'prefit':
            if avg == True:
                errs = avg_dict['errors'].to(u.us)
            else:
                errs = fitter.toas.get_errors().to(u.us)
        elif restype == 'postfit':
            if NB == True:
                if avg == True:
                    errs = avg_dict['errors'].to(u.us)
                else:
                    errs = fitter.resids.get_data_error().to(u.us)
            else:
                errs = fitter.resids.residual_objs['toa'].get_data_error().to(u.us)
        elif restype == 'both':
            if NB == True:
                if avg == True:
                    errs = avg_dict['errors'].to(u.us)
                    errs_pre = avg_dict_pre['errors'].to(u.us)
                else:
                    errs = fitter.resids.get_data_error().to(u.us)
                    errs_pre = fitter.toas.get_errors().to(u.us)
            else:
                errs = fitter.resids.residual_objs['toa'].get_data_error().to(u.us)
                errs_pre = fitter.toas.get_errors().to(u.us)
    
    # Get MJDs
    if 'orbphase' in kwargs.keys():
        orbphase = kwargs['orbphase']
    else:
        mjds = fitter.toas.get_mjds().value
        if avg == True:
            mjds = avg_dict['mjds'].value
    
    # Get receiver backends
    if 'rcvr_bcknds' in kwargs.keys():
        rcvr_bcknds = kwargs['rcvr_bcknds']
    else:
        rcvr_bcknds = np.array(fitter.toas.get_flag_value('f')[0])
        if avg == True:
            avg_rcvr_bcknds = []
            for iis in avg_dict['indices']:
                avg_rcvr_bcknds.append(rcvr_bcknds[iis[0]])
            rcvr_bcknds = np.array(avg_rcvr_bcknds)
    # Get the set of unique receiver-bandend combos
    RCVR_BCKNDS = set(rcvr_bcknds)

    # Now we need to the orbital phases; start with binary model name
    #orbphase = fitter.model.orbital_phase(mjds, radians = False)
    #"""
    binary_model_name = 'Binary'+fitter.model.binary_model_name
    # Now get the orbital phases
    if avg == True:
        orbphase = fitter.model.orbital_phase(mjds, radians = False)
    else:
        # Now get the phases in units of orbits
        delay = fitter.model.delay(fitter.toas)
        orbit_phase = fitter.model.components[binary_model_name].binary_instance.orbits()
        # Correct negative orbital phases
        neg_orb_phs_idx = np.where(orbit_phase<0.0)[0]
        orbit_phase[neg_orb_phs_idx] = orbit_phase[neg_orb_phs_idx] + np.abs(np.floor(orbit_phase[neg_orb_phs_idx]))
        # Get just the phases from 0-1
        orbphase = np.abs(np.modf(orbit_phase)[0])
    #"""

    if axs == None:
        if 'figsize' in kwargs.keys():
            figsize = kwargs['figsize']
        else:
            figsize = (10,4)
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
    else:
        ax1 = axs
    for i, r_b in enumerate(RCVR_BCKNDS):
        inds = np.where(rcvr_bcknds==r_b)[0]
        if not inds.tolist():
            r_b_label = ""
        else:
            r_b_label = rcvr_bcknds[inds][0]
        # Get plot preferences
        if 'fmt' in kwargs.keys():
            mkr = kwargs['fmt']
        else:
            mkr = markers[r_b_label]
            if restype == 'both':
                mkr_pre = '.'
        if 'color' in kwargs.keys():
            clr = kwargs['color']
        else:
            clr = colorscheme[r_b_label]
        if 'alpha' in kwargs.keys():
            alpha = kwargs['alpha']
        else:
            alpha = 0.5
        if plotsig:
            sig = res[inds]/errs[inds]
            ax1.errorbar(orbphase[inds], sig, yerr=len(errs[inds])*[1], fmt=mkr, \
                     color=clr, label=r_b_label, alpha = alpha)
            if restype == 'both':
                sig_pre = res_pre[inds]/errs_pre[inds]
                ax1.errorbar(orbphase[inds], sig_pre, yerr=len(errs_pre[inds])*[1], fmt=mkr_pre, \
                         color=clr, label=r_b_label+" Prefit", alpha = alpha)
        else:
            ax1.errorbar(orbphase[inds], res[inds], yerr = errs[inds], fmt=mkr, \
                     color=clr, label=r_b_label, alpha = alpha)
            if restype == 'both':
                ax1.errorbar(orbphase[inds], res_pre[inds], yerr=errs_pre[inds], fmt=mkr_pre, \
                     color=clr, label=r_b_label+" Prefit", alpha = alpha)
    # Set second axis
    ax1.set_xlabel(r'Orbital Phase')
    ax1.grid(True)
    if plotsig:
        if avg and whitened:
            ax1.set_ylabel('Average Residual/Uncertainty \n (Whitened)', multialignment='center')
        elif avg and not whitened:
            ax1.set_ylabel('Average Residual/Uncertainty')
        elif whitened and not avg:
            ax1.set_ylabel('Residual/Uncertainty \n (Whitened)', multialignment='center')
        else:
            ax1.set_ylabel('Residual/Uncertainty')
    else:
        if avg and whitened:
            ax1.set_ylabel('Average Residual ($\mu$s) \n (Whitened)', multialignment='center')
        elif avg and not whitened:
            ax1.set_ylabel('Average Residual ($\mu$s)')
        elif whitened and not avg:
            ax1.set_ylabel('Residual ($\mu$s) \n (Whitened)', multialignment='center')
        else:
            ax1.set_ylabel('Residual ($\mu$s)')
    if legend:
        ax1.legend(loc='upper center', bbox_to_anchor= (0.5, 1.0+1.0/figsize[1]), ncol=len(RCVR_BCKNDS))
    if title:
        plt.title("%s %s timing residuals" % (fitter.model.PSR.value, restype), y=1.0+1.0/figsize[1])
    if axs == None:
        plt.tight_layout()
    if save:
        ext = ""
        if whitened:
            ext += "_whitened"
        if avg:
            ext += "_averaged"
        if NB:
            ext += "_NB"
        else:
            ext += "_WB"
        if restype == 'prefit':
            ext += "_prefit"
        elif restype == 'postfit':
            ext += "_postfit"
        elif restype == "both":
            ext += "_pre_post_fit"
        plt.savefig("%s_resid_v_orbphase%s.png" % (fitter.model.PSR.value, ext))
        
    return


def plot_toas_freq(toas, figsize=(10,4), save = False, legend = True, axs = None):
    """
    Make a plot of frequency v. toa


    Arguments
    ---------
    toas: PINT toas
    figsize: size of the figure passed to matplotlib [default: (10,4)]
    save: if True will save plot with the name "freq_v_time.png"
    legend: boolean, if True will add legend to plot
    axs: Default is None, if not None, should be defined subplot value and the figure will be used as part of a
         larger figure.
    """

    freqs = toas.get_freqs().value
    mjds = toas.get_mjds().value
    errs = toas.get_errors().value
    years = (mjds - 51544.0)/365.25 + 2000.0
    rcvr_bcknds = np.array(toas.get_flag_value('f')[0])
    RCVR_BCKNDS = set(rcvr_bcknds)

    if axs == None:
        #ipython.magic("matplotlib inline")
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
    else:
        ax1 = axs
    for i, r_b in enumerate(RCVR_BCKNDS):
        inds = np.where(rcvr_bcknds==r_b)[0]
        if not inds.tolist():
            r_b_label = ""
        else:
            r_b_label = rcvr_bcknds[inds][0]
        ax1.scatter(years[inds], freqs[inds], marker=markers[r_b_label], color=colorscheme[r_b_label], label=r_b_label)
    # Set second axis
    ax1.set_xlabel(r'Year')
    ax1.grid(True)
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    mjd0  = ((ax1.get_xlim()[0])-2004.0)*365.25+53005.
    mjd1  = ((ax1.get_xlim()[1])-2004.0)*365.25+53005.
    if (mjd1-mjd0>1200.): mjdstep=500.
    elif (mjd1-mjd0>600.): mjdstep=200.
    else: mjdstep=100.
    mjd0 = int(mjd0/mjdstep)*mjdstep + mjdstep
    mjd1 = int(mjd1/mjdstep)*mjdstep
    mjdr = np.arange(mjd0,mjd1+mjdstep,mjdstep)
    yrr = (mjdr - 51544.0)/365.25 + 2000.0
    ax2.set_xticks(yrr)
    ax2.set_xticklabels(["%5d" % m for m in mjdr])
    ax1.set_ylabel(r"Frequency (MHz)")
    if legend:
        ax1.legend(loc='upper center', bbox_to_anchor= (0.5, 1.0+1.0/figsize[1]), ncol=len(RCVR_BCKNDS))
    if axs == None:
        plt.tight_layout()
    if save:
        plt.savefig("freq_v_time.png")
    return


def fd_res_v_freq(toas, fitter, figsize=(4,4), plotsig = False, fromPINT = True, res = None, errs = None,\
                        mjds = None, rcvr_bcknds = None, freqs = None, avg = False, whitened = False, save = False, \
                        legend = True, axs = None, comp_FD = True):
    """
    Make a plot of the residuals vs. frequency, can do WB as well. Note, if WB fitter, comp_FD may not work.


    Arguments
    ---------
    toas: PINT toas
    fitter: PINT fitter object
    figsize: size of the figure passed to matplotlib [default: (4,4)]
    plotsig: boolean, if True plot number of measurements v. residuals/uncertainty, else v. residuals
    fromPINT: boolean, if True, will get values from PINT objects, if False, will expect necessary values from
              other inputs
    NB: boolean, if True, assumes narrowband TOAs, if False (default) assumes WB TOAs
    res: residuals generated by PINT
    errs: Default in None. If fromPINT is False, errs is an array of residual errors in units of microseconds
    mjds: Default in None. If fromPINT is False, mjds is an array of mjds
    rcvr_bcknds: Default in None. If fromPINT is False, rcvr_bcknds is an array of strings of reciever-banckend combos
    freqs: Default in None. If fromPINT is False, freqs is an array of frequencies for each residual
    avg: boolean, if True will change y-axis to "Average Residual".
    whitened: boolean, if True will change y-axis to "Residual (Whitened)"
    save: boolean, if True will save plot with the name "resid_v_mjd.png" Will add averaged/whitened
         as necessary
    legend: boolean, if False, will not print legend with plot
    axs: Default is None, if not None, should be defined subplot value and the figure will be used as part of a
         larger figure.
    comp_FD: boolean, if True, will plot the residuals v. frequency with FD included, FD subtracted, and best fit
        without FD.
    """
    # Check if fitter is wideband or not
    if "Wideband" in fitter.__class__.__name__:
        NB = False
        resids = fitter.resids.residual_objs[0]
        dm_resids = fitter.resids.residual_objs[1]
    else:
        NB = True
        resids = fitter.resids

    if fromPINT:
        freqs = toas.get_freqs().value
        mjds = toas.get_mjds().value
        errs = toas.get_errors().value
        years = (mjds - 51544.0)/365.25 + 2000.0
        rcvr_bcknds = np.array(toas.get_flag_value('f')[0])
        RCVR_BCKNDS = set(rcvr_bcknds)
        res = resids.time_resids.to(u.us).value
    else:
        freqs = freqs
        res = res
        mjds = mjds
        years = (mjds - 51544.0)/365.25 + 2000.0
        errs = errs
        RCVR_BCKNDS = set(rcvr_bcknds)

    if comp_FD:
        if axs != None:
            log.warn("Cannot do full comparison with three panels")
            axs = None
        if figsize == (4,4):
            figsize = (4,12)
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(313)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(311)
    else:
        if axs == None:
            fig = plt.figure(figsize=figsize)
            ax1 = fig.add_subplot(111)
        else:
            ax1 = axs

    # Make the plot of residual vs. frequency
    for i, r_b in enumerate(RCVR_BCKNDS):
        inds = np.where(rcvr_bcknds==r_b)[0]
        if not inds.tolist():
            r_b_label = ""
        else:
            r_b_label = rcvr_bcknds[inds][0]
        if plotsig:
            sig = res[inds]/errs[inds]
            ax1.errorbar(freqs[inds], sig, yerr=len(errs[inds])*[1], fmt=markers[r_b_label], \
                     color=colorscheme[r_b_label], label=r_b_label, alpha = 0.5)
        else:
            ax1.errorbar(freqs[inds], res[inds], yerr=errs[inds], fmt=markers[r_b_label], \
                     color=colorscheme[r_b_label], label=r_b_label, alpha = 0.5)
        # assign axis labels
        ax1.set_xlabel(r'Frequency (MHz)')
        ax1.grid(True)
        if plotsig:
            if avg and whitened:
                ylabel = 'Average Residual/Uncertainty \n (Whitened)'
                #ax1.set_ylabel('Average Residual/Uncertainty \n (Whitened)', multialignment='center')
            elif avg and not whitened:
                ylabel = 'Average Residual/Uncertainty'
            elif whitened and not avg:
                ylabel ='Residual/Uncertainty \n (Whitened)'
            else:
                ylabel ='Residual/Uncertainty'
        else:
            if avg and whitened:
                ylabel = 'Average Residual ($\mu$s) \n (Whitened)'
            elif avg and not whitened:
                ylabel = 'Average Residual ($\mu$s)'
            elif whitened and not avg:
                ylabel = 'Residual ($\mu$s) \n (Whitened)'
            else:
                ylabel = 'Residual ($\mu$s)'
        ax1.set_ylabel(ylabel)

    # Now if we want to show the other plots, we plot them
    if comp_FD:
        # Plot the residuals with FD parameters subtracted
        cur_fd = [param for param in fitter.model.params if "FD" in param]
        # Get the FD offsets
        FD_offsets = np.zeros(np.size(res))
        # Also need FD line
        sorted_freqs = np.linspace(np.min(freqs), np.max(freqs), 1000)
        FD_line = np.zeros(np.size(sorted_freqs))
        for i, fd in enumerate(cur_fd):
            fd_val = getattr(fitter.model, fd).value * 10**6 # convert to microseconds
            FD_offsets += fd_val * np.log(freqs/1000.0)**(i+1)
            FD_line += fd_val * np.log(sorted_freqs/1000.0)**(i+1)
        # Now edit residuals
        fd_cor_res = res + FD_offsets

        # Now we need to redo the fit without the FD parameters
        psr_fitter_nofd = copy.deepcopy(fitter)
        try:
            psr_fitter_nofd.model.remove_component('FD')
        except:
            log.warn("No FD parameters in the initial timing model...")

        # Check if fitter is wideband or not
        if "Wideband" in psr_fitter_nofd.__class__.__name__:
            resids = psr_fitter_nofd.resids.residual_objs[0]
            avg = False
        else:
            resids = psr_fitter_nofd.resids

        psr_fitter_nofd.fit_toas(1)
        # Now we need to figure out if these need to be whitened and/or averaged
        if avg:
            avg = psr_fitter_nofd.resids.ecorr_average(use_noise_model=True)
            avg_rcvr_bcknds = rcvr_bcknds
            if whitened:
                # need to whiten and average
                wres_avg = ub.whiten_resids(avg)
                res_nofd = wres_avg.to(u.us).value
            else:
                # need to average
                res_nofd = avg['time_resids'].to(u.us).value
        elif whitened:
            # Need to whiten
            wres_nofd = ub.whiten_resids(psr_fitter_nofd)
            res_nofd = wres_nofd.to(u.us).value
        else:
            res_nofd = resids.time_resids.to(u.us).value

        # Now plot
        for i, r_b in enumerate(RCVR_BCKNDS):
            inds = np.where(rcvr_bcknds==r_b)[0]
            if not inds.tolist():
                r_b_label = ""
            else:
                r_b_label = rcvr_bcknds[inds][0]
            if plotsig:
                sig = fd_cor_res[inds]/errs[inds]
                ax3.errorbar(freqs[inds], sig, yerr=len(errs[inds])*[1], fmt=markers[r_b_label], \
                         color=colorscheme[r_b_label], label=r_b_label, alpha = 0.5)

                sig_nofd = res_nofd[inds]/errs[inds]
                ax2.errorbar(freqs[inds], sig_nofd, yerr=len(errs[inds])*[1], fmt=markers[r_b_label], \
                         color=colorscheme[r_b_label], label=r_b_label, alpha = 0.5)
            else:
                ax3.errorbar(freqs[inds], fd_cor_res[inds], yerr=errs[inds], fmt=markers[r_b_label], \
                         color=colorscheme[r_b_label], label=r_b_label, alpha = 0.5)

                ax2.errorbar(freqs[inds], res_nofd[inds], yerr=errs[inds], fmt=markers[r_b_label], \
                         color=colorscheme[r_b_label], label=r_b_label, alpha = 0.5)

            ax3.plot(sorted_freqs, FD_line, c = 'k', ls = '--')
            # assign axis labels
            ax3.set_xlabel(r'Frequency (MHz)')
            ax3.set_ylabel(ylabel)
            ax3.grid(True)
            ax2.set_xlabel(r'Frequency (MHz)')
            ax2.set_ylabel(ylabel)
            ax2.grid(True)


    if legend:
        if comp_FD:
            ax3.legend(loc='upper center', bbox_to_anchor= (0.5, 1.0+1.0/figsize[1]), ncol=len(RCVR_BCKNDS))
        else:
            ax1.legend(loc='upper center', bbox_to_anchor= (0.5, 1.0+1.0/figsize[1]), ncol=len(RCVR_BCKNDS))
    #plt.tight_layout()
    if save:
        ext = ""
        if whitened:
            ext += "_whitened"
        if avg:
            ext += "_averaged"
        if NB:
            ext += "_NB"
        else:
            ext += "_WB"
        plt.savefig("FD_resid_v_freq%s.png" % (ext))
    return


"""
We also offer some options for convenience plotting functions, one that will show all possible summary plots, and
another that will show just the summary plots that are typically created in finalize_timing.py in that order.
"""
def summary_plots(toas, model, fitter, res, title = None, legends = False, avg = False, whitened = False, \
                  save = False, fromPINT = True, wres = None, avg_res = None, avg_errs = None, avg_mjds = None, \
                  avg_toas = None, avg_rcvr_bcknds = None, avg_wres = None):
    """
    Function to make a composite set of summary plots for sets of TOAs
    NOTE - This is note the same set of plots as will be in the pdf writer

    Inputs:
    toas - PINT toa object
    model - PINT model object
    fitter - PINT fitter object
    res - list of residuals in units of microseconds (with quantity attribute removed, eg just the value)
    title - Title of plot
    legends - if True, print legends for all plots, else only add legends to residual v. time plots
    avg - If True, also plot the averaged residuals. If whitened also True, will also plot whitened averaged resids
    whitened - If True, also plot the whitened residuals
    save - If True, will save the summary plot as "summary_plots.png"
    fromPINT - If True, will compute average and whitened residuals inside function. If False, will expect
               inputs to be provided as described below
    wres - list of whitened residuals in units of microseconds (with quantity attribute removed, eg just the value)
    avg_res - epoch averaged residuals in units of microseconds (with quantity attribute removed, eg just the value)
    avg_errs - errors on the averaged residuals (also in microseconds)
    avg_mjds - MJD dates of the epoch averaged residuals
    avg_toas - PINT TOA class object of the epoch averaged residuals for orbital phase plots
    avg_rcvr_bcknds - list of receiver-backend combinations corresponding to the epoch averaged residuals
    avg_wres - list of whitened, epoch averaged residuals in units of microseconds (with quantity attribute removed,
               eg just the value)
    """
    # First determine if the averaged/whitened residuals are given or computed
    EPHEM = "DE436" # JPL ephemeris used
    BIPM = "BIPM2017" # BIPM timescale used
    if fromPINT:
        if whitened:
            # Get the whitned residuals
            wres = ub.whiten_resids(fitter)
            wres = wres.to(u.us).value
        if avg:
            # Get the epoch averaged residuals
            avg = fitter.resids.ecorr_average(use_noise_model=True)
            avg_res = avg['time_resids'].to(u.us).value
            avg_errs = avg['errors'].value
            avg_mjds = avg['mjds'].value
            avg_wres = ub.whiten_resids(avg)
            avg_wres = avg_wres.to(u.us).value
            # get rcvr backend combos for averaged residuals
            rcvr_bcknds = np.array(toas.get_flag_value('f')[0])
            avg_rcvr_bcknds = []
            for iis in avg['indices']:
                avg_rcvr_bcknds.append(rcvr_bcknds[iis[0]])
            avg_rcvr_bcknds = np.array(avg_rcvr_bcknds)
            # Turn the list into TOA class object for orbital phase plots
            avg_toas = []
            for ii in range(len(avg['mjds'].value)):
                a = toa.TOA((avg['mjds'][ii].value), freq=avg['freqs'][ii])
                avg_toas.append(a)
            avg_toas = toa.get_TOAs_list(avg_toas, ephem=EPHEM, bipm_version=BIPM)

    # Define the figure
    #ipython.magic("matplotlib inline")
    # Determine how long the figure size needs to be
    figlength = 18
    gs_rows = 6
    if whitened:
        figlength += 18
        gs_rows += 4
    if avg:
        figlength += 18
        gs_rows += 4
    if whitened and avg:
        figlength += 18
        gs_rows += 4
    # adjust size if not in a binary
    if not hasattr(model, 'binary_model_name'):
        sub_rows = 1
        sub_len = 3
        if whitened:
            sub_rows += 1
            sub_len += 3
        if avg:
            sub_rows += 1
            sub_len += 3
        if whitened and avg:
            sub_rows += 1
            sub_len += 3
        figlength -= sub_len
        gs_rows -= sub_rows

    fig = plt.figure(figsize = (12,figlength)) # not sure what we'll need for a fig size
    if title != None:
        plt.title(title, y = 1.015, size = 16)
    gs = fig.add_gridspec(gs_rows, 2)

    count = 0
    k = 0
    # First plot is all residuals vs. time.
    ax0 = fig.add_subplot(gs[count, :])
    plot_residuals_time(toas, res, figsize=(12,3), axs = ax0)
    k += 1

    # Plot the residuals divided by uncertainty vs. time
    ax1 = fig.add_subplot(gs[count+k, :])
    plot_residuals_time(toas, res, figsize=(12,3), legend = False, plotsig = True, axs = ax1)
    k += 1

    # Second plot is residual v. orbital phase
    if hasattr(model, 'binary_model_name'):
        ax2 = fig.add_subplot(gs[count+k, :])
        plot_residuals_orb(toas, res, model, figsize=(12,3), legend = False, axs = ax2)
        k += 1

    # Now add the measurement vs. uncertainty
    ax3_0 = fig.add_subplot(gs[count+k, 0])
    ax3_1 = fig.add_subplot(gs[count+k, 1])
    measurements_v_res(toas, res, nbin = 50, figsize=(6,3), plotsig=False, legend = False, axs = ax3_0)
    measurements_v_res(toas, res, nbin = 50, figsize=(6,3), plotsig=True, legend = False, axs = ax3_1)
    k += 1

    # and the DMX vs. time
    ax4 = fig.add_subplot(gs[count+k, :])
    plot_dmx_time(toas, fitter, figsize=(12,3), legend = False, axs = ax4)
    k += 1

    # And residual vs. Frequency
    ax5 = fig.add_subplot(gs[count+k, :])
    plot_toas_freq(toas, figsize=(12,3),legend = False, axs =ax5)
    k += 1

    # Now if whitened add the whitened residual plots
    if whitened:
        ax6 = fig.add_subplot(gs[count+k, :])
        plot_residuals_time(toas, wres, figsize=(12,3), whitened = True, axs = ax6)
        k += 1

        # Plot the residuals divided by uncertainty vs. time
        ax7 = fig.add_subplot(gs[count+k, :])
        plot_residuals_time(toas, wres, figsize=(12,3), legend = False, plotsig = True, whitened = True, axs = ax7)
        k += 1

        # Second plot is residual v. orbital phase
        if hasattr(model, 'binary_model_name'):
            ax8 = fig.add_subplot(gs[count+k, :])
            plot_residuals_orb(toas, wres, model, figsize=(12,3), legend = False, whitened = True, axs = ax8)
            k += 1

        # Now add the measurement vs. uncertainty
        ax9_0 = fig.add_subplot(gs[count+k, 0])
        ax9_1 = fig.add_subplot(gs[count+k, 1])
        measurements_v_res(toas, wres, nbin = 50, figsize=(6,3), plotsig=False, legend = False, whitened = True,\
                           axs = ax9_0)
        measurements_v_res(toas, wres, nbin = 50, figsize=(6,3), plotsig=True, legend = False, whitened = True,\
                           axs = ax9_1)
        k += 1

    # Now plot the average residuals
    if avg:
        ax10 = fig.add_subplot(gs[count+k, :])
        plot_residuals_time(toas, avg_res, figsize=(12,3), fromPINT = False, errs = avg_errs, mjds = avg_mjds, \
                   rcvr_bcknds = avg_rcvr_bcknds, avg = True, axs = ax10)
        k += 1

        # Plot the residuals divided by uncertainty vs. time
        ax11 = fig.add_subplot(gs[count+k, :])
        plot_residuals_time(toas, avg_res, figsize=(12,3), fromPINT = False, legend = False, plotsig = True, errs = avg_errs, \
                            mjds = avg_mjds, rcvr_bcknds = avg_rcvr_bcknds, avg = True, axs = ax11)
        k += 1

        # Second plot is residual v. orbital phase
        if hasattr(model, 'binary_model_name'):
            ax12 = fig.add_subplot(gs[count+k, :])
            plot_residuals_orb(avg_toas, avg_res, model, figsize=(12,3), fromPINT = False, legend = False, \
                               errs = avg_errs, mjds = avg_mjds, \
                               rcvr_bcknds = avg_rcvr_bcknds, avg = True, axs = ax12)
            k += 1

        # Now add the measurement vs. uncertainty
        ax13_0 = fig.add_subplot(gs[count+k, 0])
        ax13_1 = fig.add_subplot(gs[count+k, 1])
        measurements_v_res(toas, avg_res, nbin = 50, figsize=(6,3), fromPINT = False, plotsig=False, errs = avg_errs, \
                       rcvr_bcknds = avg_rcvr_bcknds, legend = False, avg = True, axs = ax13_0)
        measurements_v_res(toas, avg_res, nbin = 50, figsize=(6,3), fromPINT = False, plotsig=True, errs = avg_errs, \
                       rcvr_bcknds = avg_rcvr_bcknds, legend = False, avg = True, axs = ax13_1)
        k += 1

    # Now plot the whitened average residuals
    if avg and whitened:
        ax14 = fig.add_subplot(gs[count+k, :])
        plot_residuals_time(toas, avg_wres, figsize=(12,3), fromPINT = False, errs = avg_errs, mjds = avg_mjds, \
                   rcvr_bcknds = avg_rcvr_bcknds, avg = True, whitened = True, axs = ax14)
        k += 1

        # Plot the residuals divided by uncertainty vs. time
        ax15 = fig.add_subplot(gs[count+k, :])
        plot_residuals_time(toas, avg_wres, figsize=(12,3), fromPINT = False, legend = False, plotsig = True, errs = avg_errs, \
                            mjds = avg_mjds, rcvr_bcknds = avg_rcvr_bcknds, avg = True, whitened = True, axs = ax15)
        k += 1

        # Second plot is residual v. orbital phase
        if hasattr(model, 'binary_model_name'):
            ax16 = fig.add_subplot(gs[count+k, :])
            plot_residuals_orb(avg_toas, avg_wres, model, figsize=(12,3), fromPINT = False, legend = False, \
                               errs = avg_errs, mjds = avg_mjds, \
                               rcvr_bcknds = avg_rcvr_bcknds, avg = True, whitened = True, axs = ax16)
            k += 1

        # Now add the measurement vs. uncertainty
        ax17_0 = fig.add_subplot(gs[count+k, 0])
        ax17_1 = fig.add_subplot(gs[count+k, 1])
        measurements_v_res(toas, avg_wres, nbin = 50, figsize=(6,3), fromPINT = False, plotsig=False, \
                           errs = avg_errs, \
                           rcvr_bcknds = avg_rcvr_bcknds, legend = False, avg = True, whitened = True, axs = ax17_0)
        measurements_v_res(toas, avg_wres, nbin = 50, figsize=(6,3), fromPINT = False, plotsig=True, \
                           errs = avg_errs, \
                           rcvr_bcknds = avg_rcvr_bcknds, legend = False, avg = True, whitened = True, axs = ax17_1)
        k += 1

    plt.tight_layout()
    if save:
        plt.savefig("summary_plots.png")

    return


"""We also define a function to output the summary plots exactly as is done in finalize_timing.py (for now)"""
def summary_plots_ft(toas, model, fitter, res, title = None, legends = False,\
                  save = False, fromPINT = True, wres = None, avg_res = None, avg_errs = None, avg_mjds = None, \
                  avg_toas = None, avg_rcvr_bcknds = None, avg_wres = None):
    """
    Function to make a composite set of summary plots for sets of TOAs
    NOTE - This is note the same set of plots as will be in the pdf writer

    Inputs:
    toas - PINT toa object
    model - PINT model object
    fitter - PINT fitter object
    res - list of residuals in units of microseconds (with quantity attribute removed, eg just the value)
    title - Title of plot
    legends - if True, print legends for all plots, else only add legends to residual v. time plots
    save - If True, will save the summary plot as "summary_plots_FT.png"
    fromPINT - If True, will compute average and whitened residuals inside function. If False, will expect
               inputs to be provided as described below
    wres - list of whitened residuals in units of microseconds (with quantity attribute removed, eg just the value)
    avg_res - epoch averaged residuals in units of microseconds (with quantity attribute removed, eg just the value)
    avg_errs - errors on the averaged residuals (also in microseconds)
    avg_mjds - MJD dates of the epoch averaged residuals
    avg_toas - PINT TOA class object of the epoch averaged residuals for orbital phase plots
    avg_rcvr_bcknds - list of receiver-backend combinations corresponding to the epoch averaged residuals
    avg_wres - list of whitened, epoch averaged residuals in units of microseconds (with quantity attribute removed,
               eg just the value)
    """
    # First determine if the averaged/whitened residuals are given or computed
    EPHEM = "DE436" # JPL ephemeris used
    BIPM = "BIPM2017" # BIPM timescale used
    if fromPINT:
        # Get the whitned residuals
        wres = ub.whiten_resids(fitter)
        wres = wres.to(u.us).value
        # Get the epoch averaged residuals
        avg = fitter.resids.ecorr_average(use_noise_model=True)
        avg_res = avg['time_resids'].to(u.us).value
        avg_errs = avg['errors'].value
        avg_mjds = avg['mjds'].value
        avg_wres = ub.whiten_resids(avg)
        avg_wres = avg_wres.to(u.us).value
        # get rcvr backend combos for averaged residuals
        rcvr_bcknds = np.array(toas.get_flag_value('f')[0])
        avg_rcvr_bcknds = []
        for iis in avg['indices']:
            avg_rcvr_bcknds.append(rcvr_bcknds[iis[0]])
        avg_rcvr_bcknds = np.array(avg_rcvr_bcknds)
        # Turn the list into TOA class object for orbital phase plots
        avg_toas = []
        for ii in range(len(avg['mjds'].value)):
            a = toa.TOA((avg['mjds'][ii].value), freq=avg['freqs'][ii])
            avg_toas.append(a)
        avg_toas = toa.get_TOAs_list(avg_toas, ephem=EPHEM, bipm_version=BIPM)

    # Define the figure
    #ipython.magic("matplotlib inline")
    # Determine how long the figure size needs to be
    figlength = 18*3
    gs_rows = 13
    if not hasattr(model, 'binary_model_name'):
        figlength -= 9
        gs_rows -= 3

    fig = plt.figure(figsize = (12,figlength)) # not sure what we'll need for a fig size
    if title != None:
        plt.title(title, y = 1.015, size = 16)
    gs = fig.add_gridspec(gs_rows, 2)

    count = 0
    k = 0
    # First plot is all residuals vs. time.
    ax0 = fig.add_subplot(gs[count, :])
    plot_residuals_time(toas, res, figsize=(12,3), axs = ax0)
    k += 1

    # Then the epoch averaged residuals v. time
    ax10 = fig.add_subplot(gs[count+k, :])
    plot_residuals_time(toas, avg_res, figsize=(12,3), fromPINT = False, errs = avg_errs, mjds = avg_mjds, \
               rcvr_bcknds = avg_rcvr_bcknds, avg = True, axs = ax10)
    k += 1

    # Epoch averaged vs. orbital phase
    if hasattr(model, 'binary_model_name'):
        ax12 = fig.add_subplot(gs[count+k, :])
        plot_residuals_orb(avg_toas, avg_res, model, figsize=(12,3), fromPINT = False, legend = False, \
                           errs = avg_errs, mjds = avg_mjds, \
                           rcvr_bcknds = avg_rcvr_bcknds, avg = True, axs = ax12)
        k += 1

    # And DMX vs. time
    ax4 = fig.add_subplot(gs[count+k, :])
    plot_dmx_time(toas, fitter, figsize=(12,3), legend = False, axs = ax4)
    k += 1

    # Whitened residuals v. time
    ax6 = fig.add_subplot(gs[count+k, :])
    plot_residuals_time(toas, wres, figsize=(12,3), whitened = True, axs = ax6)
    k += 1

    # Whitened epoch averaged residuals v. time
    ax15 = fig.add_subplot(gs[count+k, :])
    plot_residuals_time(toas, avg_wres, figsize=(12,3), fromPINT = False, legend = False, plotsig = False, \
                        errs = avg_errs, mjds = avg_mjds, rcvr_bcknds = avg_rcvr_bcknds, avg = True, \
                        whitened = True, axs = ax15)
    k += 1

    # Whitened epoch averaged residuals v. orbital phase
    if hasattr(model, 'binary_model_name'):
        ax16 = fig.add_subplot(gs[count+k, :])
        plot_residuals_orb(avg_toas, avg_wres, model, figsize=(12,3), fromPINT = False, legend = False, \
                           errs = avg_errs, mjds = avg_mjds, \
                           rcvr_bcknds = avg_rcvr_bcknds, avg = True, whitened = True, axs = ax16)
        k += 1

    # Now add the measurement vs. uncertainty for both all reaiduals and epoch averaged
    ax3_0 = fig.add_subplot(gs[count+k, 0])
    ax3_1 = fig.add_subplot(gs[count+k, 1])
    measurements_v_res(toas, wres, nbin = 50, figsize=(6,3), plotsig=False, legend = False, whitened = True,\
                           axs = ax3_0)
    measurements_v_res(toas, avg_wres, nbin = 50, figsize=(6,3), fromPINT = False, plotsig=False, \
                       errs = avg_errs, \
                       rcvr_bcknds = avg_rcvr_bcknds, legend = False, avg = True, whitened = True, axs = ax3_1)
    k += 1

    # Whitened residual/uncertainty v. time
    ax26 = fig.add_subplot(gs[count+k, :])
    plot_residuals_time(toas, wres, figsize=(12,3), plotsig = True, whitened = True, axs = ax26)
    k += 1

    # Epoch averaged Whitened residual/uncertainty v. time
    ax25 = fig.add_subplot(gs[count+k, :])
    plot_residuals_time(toas, avg_wres, figsize=(12,3), fromPINT = False, legend = False, plotsig = True, \
                        errs = avg_errs, mjds = avg_mjds, rcvr_bcknds = avg_rcvr_bcknds, avg = True, \
                        whitened = True, axs = ax25)
    k += 1

    # Epoch averaged Whitened residual/uncertainty v. orbital phase
    if hasattr(model, 'binary_model_name'):
        ax36 = fig.add_subplot(gs[count+k, :])
        plot_residuals_orb(avg_toas, avg_wres, model, figsize=(12,3), fromPINT = False, legend = False, \
                           errs = avg_errs, mjds = avg_mjds, plotsig = True, \
                           rcvr_bcknds = avg_rcvr_bcknds, avg = True, whitened = True, axs = ax36)
        k += 1

    # Now add the measurement vs. uncertainty for both all reaiduals/uncertainty and epoch averaged/uncertainty
    ax17_0 = fig.add_subplot(gs[count+k, 0])
    ax17_1 = fig.add_subplot(gs[count+k, 1])
    measurements_v_res(toas, wres, nbin = 50, figsize=(6,3), plotsig=True, legend = False, whitened = True,\
                           axs = ax17_0)
    measurements_v_res(toas, avg_wres, nbin = 50, figsize=(6,3), fromPINT = False, plotsig=True, \
                       errs = avg_errs, \
                       rcvr_bcknds = avg_rcvr_bcknds, legend = False, avg = True, whitened = True, axs = ax17_1)
    k += 1

    # Now plot the frequencies of the TOAs vs. time
    ax5 = fig.add_subplot(gs[count+k, :])
    plot_toas_freq(toas, figsize=(12,3),legend = False, axs =ax5)
    k += 1

    plt.tight_layout()
    if save:
        plt.savefig("summary_plots_FT.png")

    return

def plots_for_summary_pdf_nb(toas, model, fitter, title = None, legends = False,\
                  save = False, fromPINT = True, wres = None, avg_res = None, avg_errs = None, avg_mjds = None, \
                  avg_toas = None, avg_rcvr_bcknds = None, avg_wres = None):
    """
    Function to make a composite set of summary plots for sets of TOAs to be put into a summary pdf.
    This is for Narrowband timing only. We need a slightly different set of summary plots for this.

    Inputs:
    toas - PINT toa object
    model - PINT model object
    fitter - PINT fitter object
    title - Title of plot
    legends - if True, print legends for all plots, else only add legends to residual v. time plots
    save - If True, will save the summary plot as "summary_plots_FT.png"
    fromPINT - If True, will compute average and whitened residuals inside function. If False, will expect
               inputs to be provided as described below
    wres - list of whitened residuals in units of microseconds (with quantity attribute removed, eg just the value)
    avg_res - epoch averaged residuals in units of microseconds (with quantity attribute removed, eg just the value)
    avg_errs - errors on the averaged residuals (also in microseconds)
    avg_mjds - MJD dates of the epoch averaged residuals
    avg_toas - PINT TOA class object of the epoch averaged residuals for orbital phase plots
    avg_rcvr_bcknds - list of receiver-backend combinations corresponding to the epoch averaged residuals
    avg_wres - list of whitened, epoch averaged residuals in units of microseconds (with quantity attribute removed,
               eg just the value)
    """
    # First determine if the averaged/whitened residuals are given or computed
    EPHEM = "DE435" # JPL ephemeris used
    BIPM = "BIPM1.5015" # BIPM timescale used
    res = fitter.resids.time_resids.to(u.us).value
    if fromPINT:
        # Get the whitned residuals
        wres = ub.whiten_resids(fitter)
        wres = wres.to(u.us).value
        # Get the epoch averaged residuals
        avg = fitter.resids.ecorr_average(use_noise_model=True)
        avg_res = avg['time_resids'].to(u.us).value
        avg_errs = avg['errors'].value
        avg_mjds = avg['mjds'].value
        avg_wres = ub.whiten_resids(avg)
        avg_wres = avg_wres.to(u.us).value
        # get rcvr backend combos for averaged residuals
        rcvr_bcknds = np.array(toas.get_flag_value('f')[0])
        avg_rcvr_bcknds = []
        for iis in avg['indices']:
            avg_rcvr_bcknds.append(rcvr_bcknds[iis[0]])
        avg_rcvr_bcknds = np.array(avg_rcvr_bcknds)
        # Turn the list into TOA class object for orbital phase plots
        avg_toas = []
        for ii in range(len(avg['mjds'].value)):
            a = toa.TOA((avg['mjds'][ii].value), freq=avg['freqs'][ii])
            avg_toas.append(a)
        avg_toas = toa.get_TOAs_list(avg_toas, ephem=EPHEM, bipm_version=BIPM)

    # Need to make four sets of plots
    for ii in range(4):
        if ii != 3:
            fig = plt.figure(figsize=(8,10.0),dpi=100)
        else:
            fig = plt.figure(figsize=(8,5),dpi=100)
        if title != None:
            plt.title(title, y = 1.08, size = 14)
        if ii == 0:
            gs = fig.add_gridspec(nrows = 4, ncols = 1)

            ax0 = fig.add_subplot(gs[0,:])
            ax1 = fig.add_subplot(gs[1,:])
            ax2 = fig.add_subplot(gs[2,:])
            ax3 = fig.add_subplot(gs[3,:])
            plot_residuals_time(toas, res, figsize=(8, 2.5), axs = ax0, NB = True)
            plot_residuals_time(toas, avg_res, figsize=(8,2.5), fromPINT = False, errs = avg_errs, mjds = avg_mjds, \
               rcvr_bcknds = avg_rcvr_bcknds, avg = True, axs = ax1, legend = False, NB = True)
            if hasattr(model, 'binary_model_name'):
                plot_residuals_orb(avg_toas, avg_res, model, figsize=(8,2.5), fromPINT = False, legend = False, \
                                   errs = avg_errs, mjds = avg_mjds, \
                                   rcvr_bcknds = avg_rcvr_bcknds, avg = True, axs = ax2)
            plot_dmx_time(toas, fitter, figsize=(8,2.5), legend = False, axs = ax3, NB = True)
            plt.tight_layout()
            plt.savefig("%s_summary_plot_1.png" % (model.PSR.value))
            plt.close()
        elif ii == 1:
            if hasattr(model, 'binary_model_name'):
                gs = fig.add_gridspec(4,2)
                ax2 = fig.add_subplot(gs[2,:])
                ax3 = fig.add_subplot(gs[3,0])
                ax4 = fig.add_subplot(gs[3,1])
            else:
                gs = fig.add_gridspec(3,2)
                ax3 = fig.add_subplot(gs[2,0])
                ax4 = fig.add_subplot(gs[2,1])
            ax0 = fig.add_subplot(gs[0,:])
            ax1 = fig.add_subplot(gs[1,:])
            plot_residuals_time(toas, wres, figsize=(8,2.5), whitened = True, axs = ax0, NB = True)
            plot_residuals_time(toas, avg_wres, figsize=(8,2.5), fromPINT = False, legend = False, plotsig = False, \
                        errs = avg_errs, mjds = avg_mjds, rcvr_bcknds = avg_rcvr_bcknds, avg = True, \
                        whitened = True, axs = ax1, NB = True)
            if hasattr(model, 'binary_model_name'):
                plot_residuals_orb(avg_toas, avg_wres, model, figsize=(8,2.5), fromPINT = False, legend = False, \
                           errs = avg_errs, mjds = avg_mjds, \
                           rcvr_bcknds = avg_rcvr_bcknds, avg = True, whitened = True, axs = ax2)
            measurements_v_res(toas, wres, nbin = 50, figsize=(4,2.5), plotsig=False, legend = False, whitened = True,\
                           axs = ax3)
            measurements_v_res(toas, avg_wres, nbin = 50, figsize=(4,2.5), fromPINT = False, plotsig=False, \
                       errs = avg_errs, \
                       rcvr_bcknds = avg_rcvr_bcknds, legend = False, avg = True, whitened = True, axs = ax4)
            plt.tight_layout()
            plt.savefig("%s_summary_plot_2.png" % (model.PSR.value))
            plt.close()
        elif ii == 2:
            if hasattr(model, 'binary_model_name'):
                gs = fig.add_gridspec(4,2)
                ax2 = fig.add_subplot(gs[2,:])
                ax3 = fig.add_subplot(gs[3,0])
                ax4 = fig.add_subplot(gs[3,1])
            else:
                gs = fig.add_gridspec(3,2)
                ax3 = fig.add_subplot(gs[2,0])
                ax4 = fig.add_subplot(gs[2,1])
            ax0 = fig.add_subplot(gs[0,:])
            ax1 = fig.add_subplot(gs[1,:])
            plot_residuals_time(toas, wres, figsize=(8,2.5), plotsig = True, whitened = True, axs = ax0, NB = True)
            plot_residuals_time(toas, avg_wres, figsize=(8,2.5), fromPINT = False, legend = False, plotsig = True, \
                        errs = avg_errs, mjds = avg_mjds, rcvr_bcknds = avg_rcvr_bcknds, avg = True, \
                        whitened = True, axs = ax1, NB = True)
            if hasattr(model, 'binary_model_name'):
                plot_residuals_orb(avg_toas, avg_wres, model, figsize=(8,2.5), fromPINT = False, legend = False, \
                           errs = avg_errs, mjds = avg_mjds, plotsig = True, \
                           rcvr_bcknds = avg_rcvr_bcknds, avg = True, whitened = True, axs = ax2)

            measurements_v_res(toas, wres, nbin = 50, figsize=(4,2.5), plotsig=True, legend = False, whitened = True,\
                           axs = ax3)
            measurements_v_res(toas, avg_wres, nbin = 50, figsize=(4,2.5), fromPINT = False, plotsig=True, \
                       errs = avg_errs, \
                       rcvr_bcknds = avg_rcvr_bcknds, legend = False, avg = True, whitened = True, axs = ax4)
            plt.tight_layout()
            plt.savefig("%s_summary_plot_3.png" % (model.PSR.value))
            plt.close()
        elif ii == 3:
            gs = fig.add_gridspec(1,1)
            ax0 = fig.add_subplot(gs[0])
            plot_toas_freq(toas, figsize=(8,4),legend = True, axs =ax0)
            plt.tight_layout()
            plt.savefig("%s_summary_plot_4.png" % (model.PSR.value))
            plt.close()

def plots_for_summary_pdf_wb(toas, model, fitter, title = None, legends = False,\
                  save = False, fromPINT = True, wres = None):
    """
    Function to make a composite set of summary plots for sets of TOAs to be put into a summary pdf.
    Only for Wideband Timing. We need a slightly different set of summary plots here for this.

    Inputs:
    toas - PINT toa object
    model - PINT model object
    fitter - PINT fitter object
    title - Title of plot
    legends - if True, print legends for all plots, else only add legends to residual v. time plots
    save - If True, will save the summary plot as "summary_plots_FT.png"
    fromPINT - If True, will compute average and whitened residuals inside function. If False, will expect
               inputs to be provided as described below
    wres - list of whitened residuals in units of microseconds (with quantity attribute removed, eg just the value)
    """
    # First determine if the averaged/whitened residuals are given or computed
    EPHEM = "DE435" # JPL ephemeris used
    BIPM = "BIPM1.5015" # BIPM timescale used
    # Get the residuals
    res = fitter.resids.residual_objs[0].time_resids.to(u.us).value
    dm_resids = fitter.resids.residual_objs[1]
    if fromPINT:
        # Get the whitned residuals
        wres = ub.whiten_resids(fitter)
        wres = wres.to(u.us).value
        # get rcvr backend combos for averaged residuals
        rcvr_bcknds = np.array(toas.get_flag_value('f')[0])

    # Need to make four sets of plots
    for ii in range(4):
        if ii != 3:
            fig = plt.figure(figsize=(8,10.0),dpi=100)
        else:
            fig = plt.figure(figsize=(8,5),dpi=100)
        if title != None:
            plt.title(title, y = 1.08, size = 14)
        if ii == 0:
            if hasattr(model, 'binary_model_name'):
                gs = fig.add_gridspec(nrows = 4, ncols = 1)
                ax2 = fig.add_subplot(gs[2,:])
                ax3 = fig.add_subplot(gs[3,:])
            else:
                gs = fig.add_gridspec(nrows = 3, ncols = 1)
                ax3 = fig.add_subplot(gs[2,:])
            ax0 = fig.add_subplot(gs[0,:])
            ax1 = fig.add_subplot(gs[1,:])

            plot_residuals_time(toas, res, figsize=(8, 2.5), axs = ax0)
            #plot_wb_dm_time(toas, fitter, figsize=(10,4), legend = False, axs = ax1, mean_sub = True)
            fd_res_v_freq(toas, fitter, figsize=(12,4), plotsig = False, fromPINT = True, \
              legend = False, axs = ax1, comp_FD = False)
            if hasattr(model, 'binary_model_name'):
                plot_residuals_orb(toas, res, model, figsize=(8,2.5), fromPINT = True, legend = False, \
                                        axs = ax2)
            plot_dmx_time(toas, fitter, figsize=(8,2.5), legend = False, axs = ax3)
            plt.tight_layout()
            plt.savefig("%s_summary_plot_1.png" % (model.PSR.value))
            plt.close()
        elif ii == 1:
            if hasattr(model, 'binary_model_name'):
                gs = fig.add_gridspec(4,2)
                ax2 = fig.add_subplot(gs[2,:])
                ax3 = fig.add_subplot(gs[3,0])
                ax4 = fig.add_subplot(gs[3,1])
            else:
                gs = fig.add_gridspec(3,2)
                ax3 = fig.add_subplot(gs[2,0])
                ax4 = fig.add_subplot(gs[2,1])
            ax0 = fig.add_subplot(gs[0,:])
            ax1 = fig.add_subplot(gs[1,:])

            plot_residuals_time(toas, wres, figsize=(8,2.5), whitened = True, axs = ax0)
            fd_res_v_freq(toas, fitter, figsize=(12,4), plotsig = False, fromPINT = False,
                          res = wres, errs = toas.get_errors().value,\
                          mjds = toas.get_mjds().value, rcvr_bcknds = np.array(toas.get_flag_value('f')[0]),\
                          whitened = True, \
                          legend = False, freqs = toas.get_freqs().value, axs = ax1, comp_FD = False)
            if hasattr(model, 'binary_model_name'):
                plot_residuals_orb(toas, wres, model, figsize=(8,2.5), fromPINT = True, legend = False, \
                            whitened = True, axs = ax2)
            measurements_v_res(toas, wres, nbin = 50, figsize=(4,2.5), plotsig=False, legend = False, whitened = True,\
                           axs = ax3)
            #measurements_v_res(toas, avg_wres, nbin = 50, figsize=(4,2.5), fromPINT = False, plotsig=False, \
            #           errs = avg_errs, \
            #           rcvr_bcknds = avg_rcvr_bcknds, legend = False, avg = True, whitened = True, axs = ax4)
            plt.tight_layout()
            plt.savefig("%s_summary_plot_2.png" % (model.PSR.value))
            plt.close()
        elif ii == 2:
            if hasattr(model, 'binary_model_name'):
                gs = fig.add_gridspec(4,2)
                ax2 = fig.add_subplot(gs[2,:])
                ax3 = fig.add_subplot(gs[3,0])
                ax4 = fig.add_subplot(gs[3,1])
            else:
                gs = fig.add_gridspec(3,2)
                ax3 = fig.add_subplot(gs[2,0])
                ax4 = fig.add_subplot(gs[2,1])
            ax0 = fig.add_subplot(gs[0,:])
            ax1 = fig.add_subplot(gs[1,:])
            plot_residuals_time(toas, wres, figsize=(8,2.5), plotsig = True, whitened = True, axs = ax0)
            fd_res_v_freq(toas, fitter, figsize=(12,4), plotsig = True, fromPINT = False,
                          res = wres, errs = toas.get_errors().value,\
                          mjds = toas.get_mjds().value, rcvr_bcknds = np.array(toas.get_flag_value('f')[0]),\
                          whitened = True, \
                          legend = False, freqs = toas.get_freqs().value, axs = ax1, comp_FD = False)
            if hasattr(model, 'binary_model_name'):
                plot_residuals_orb(toas, wres, model, figsize=(8,2.5), fromPINT = True, legend = False, \
                           plotsig = True, whitened = True, axs = ax2)

            measurements_v_res(toas, wres, nbin = 50, figsize=(4,2.5), plotsig=True, legend = False, whitened = True,\
                           axs = ax3)
            #measurements_v_res(toas, avg_wres, nbin = 50, figsize=(4,2.5), fromPINT = False, plotsig=True, \
            #           errs = avg_errs, \
            #           rcvr_bcknds = avg_rcvr_bcknds, legend = False, avg = True, whitened = True, axs = ax4)
            plt.tight_layout()
            plt.savefig("%s_summary_plot_3.png" % (model.PSR.value))
            plt.close()
        elif ii == 3:
            gs = fig.add_gridspec(1,1)
            ax0 = fig.add_subplot(gs[0])
            plot_toas_freq(toas, figsize=(8,4),legend = True, axs =ax0)
            plt.tight_layout()
            plt.savefig("%s_summary_plot_4.png" % (model.PSR.value))
            plt.close()
