#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:30:59 2020
@author: bshapiroalbert
Code since butchered by many timers.
"""
import numpy as np
import matplotlib.pyplot as plt
import copy
from loguru import logger as log
import astropy.units as u
import yaml

import pint.toa as toa
import pint.models as model
import pint.fitter as fitter
import pint.utils as pu
import subprocess

from pint_pal.utils import *
import os
from pint_pal.timingconfiguration import TimingConfiguration
import pint_pal.lite_utils as lu

PACKAGE_DIR = os.path.dirname(__file__)
with open(os.path.join(PACKAGE_DIR, "plot_settings.yaml"), "r") as cf:
    config = yaml.safe_load(cf)

# plot_settings.yaml now has a NANOGrav 20-yr specific colorscheme (ng20_c).
# If you want to go back to the old colors (or are doing DR3), change this to
# colorschemes["febe"] = config["febe_c"] AND markers["febe"] = config["febe_m"]
colorschemes, markers, labels = {}, {}, {}
colorschemes["obs"] = config["obs_c"]
colorschemes["pta"] = config["pta_c"]
colorschemes["febe"] = config["ng20_c"]
markers["obs"] = config["obs_m"]
markers["pta"] = config["pta_m"]
markers["febe"] = config["ng20_m"]
labels = config["label_names"]


def call(x):
    subprocess.call(x, shell=True)


def set_color_and_marker(colorby):
    if colorby == "pta":
        colorscheme = colorschemes["pta"]
        markerscheme = markers["pta"]
    elif colorby == "obs":
        colorscheme = colorschemes["observatories"]
        markerscheme = markers["observatories"]
    elif colorby == "f":
        colorscheme = colorschemes["febe"]
        markerscheme = markers["febe"]
    return colorscheme, markerscheme


def plot_residuals_time(
    fitter,
    restype="postfit",
    colorby="f",
    plotsig=False,
    avg=False,
    whitened=False,
    save=False,
    legend=True,
    title=True,
    axs=None,
    mixed_ecorr=False,
    **kwargs,
):
    """
    Make a plot of the residuals vs. time


    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    restype ['string'] : Type of residuals, pre or post fit, to plot from fitter object. Options are:
        'prefit' - plot the prefit residuals.
        'postfit' - plot the postfit residuals (default)
        'both' - overplot both the pre and post-fit residuals.
    colorby ['string']: What to use to determine color/markers
        'pta' - color residuals by PTA (default)
        'obs' - color residuals by telescope
        'f'   - color residuals by frontend/backend pair (flag not used by all PTAs).
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


    Optional Arguments:
    --------------------
    res [list/array] : List or array of residual values to plot. Will override values from fitter object.
    errs [list/array] : List or array of residual error values to plot. Will override values from fitter object.
    mjds [list/array] : List or array of TOA MJDs to plot. Will override values from toa object.
    obs[list/array] : List or array of TOA observatories combinations. Will override values from toa object.
    figsize [tuple] : Size of the figure passed to matplotlib [default: (10,4)].
    fmt ['string'] : matplotlib format option for markers [default: ('x')]
    color ['string'] : matplotlib color option for plot [default: color dictionary in plot_utils.py file]
    alpha [float] : matplotlib alpha options for plot points [default: 0.5]
    mixed_ecorr [boolean]: If True, allows avging with mixed ecorr/no ecorr TOAs.
    """
    # Check if wideband
    if fitter.is_wideband:
        NB = False
        if avg == True:
            raise ValueError(
                "Cannot epoch average wideband residuals, please change 'avg' to False."
            )
    else:
        NB = True

    # Check if want epoch averaged residuals
    if avg == True and restype == "prefit" and mixed_ecorr == True:
        avg_dict = fitter.resids_init.ecorr_average(use_noise_model=True)
        no_avg_dict = no_ecorr_average(
            fitter.toas, fitter.resids_init, use_noise_model=True
        )
    elif avg == True and restype == "postfit" and mixed_ecorr == True:
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
        no_avg_dict = no_ecorr_average(fitter.toas, fitter.resids, use_noise_model=True)
    elif avg == True and restype == "both" and mixed_ecorr == True:
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
        no_avg_dict = no_ecorr_average(fitter.toas, fitter.resids, use_noise_model=True)
        avg_dict_pre = fitter.resids_init.ecorr_average(use_noise_model=True)
        no_avg_dict_pre = no_ecorr_average(
            fitter.toas, fitter.resids_init, use_noise_model=True
        )
    elif avg == True and restype == "prefit" and mixed_ecorr == False:
        avg_dict = fitter.resids_init.ecorr_average(use_noise_model=True)
    elif avg == True and restype == "postfit" and mixed_ecorr == False:
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
    elif avg == True and restype == "both" and mixed_ecorr == False:
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
        avg_dict_pre = fitter.resids_init.ecorr_average(use_noise_model=True)

    # Get residuals
    if "res" in kwargs.keys():
        res = kwargs["res"]
    else:
        if restype == "prefit":
            if NB == True:
                if avg == True and mixed_ecorr == True:
                    res = avg_dict["time_resids"].to(u.us)
                    res_no_avg = no_avg_dict["time_resids"].to(u.us)
                elif avg == True and mixed_ecorr == False:
                    res = avg_dict["time_resids"].to(u.us)
                else:
                    res = fitter.resids_init.time_resids.to(u.us)
            else:
                res = fitter.resids_init.residual_objs["toa"].time_resids.to(u.us)
        elif restype == "postfit":
            if NB == True:
                if avg == True and mixed_ecorr == True:
                    res = avg_dict["time_resids"].to(u.us)
                    res_no_avg = no_avg_dict["time_resids"].to(u.us)
                elif avg == True:
                    res = avg_dict["time_resids"].to(u.us)
                else:
                    res = fitter.resids.time_resids.to(u.us)
            else:
                res = fitter.resids.residual_objs["toa"].time_resids.to(u.us)
        elif restype == "both":
            if NB == True:
                if avg == True and mixed_ecorr == True:
                    res = avg_dict["time_resids"].to(u.us)
                    res_no_avg = no_avg_dict["time_resids"].to(u.us)
                    res_pre = avg_dict_pre["time_resids"].to(u.us)
                    res_pre_no_avg = no_avg_dict_pre["time_resids"].to(u.us)
                elif avg == True and mixed_ecorr == False:
                    res = avg_dict["time_resids"].to(u.us)
                    res_pre = avg_dict_pre["time_resids"].to(u.us)
                else:
                    res = fitter.resids.time_resids.to(u.us)
                    res_pre = fitter.resids_init.time_resids.to(u.us)
            else:
                res = fitter.resids.residual_objs["toa"].time_resids.to(u.us)
                res_pre = fitter.resids_init.residual_objs["toa"].time_resids.to(u.us)
        else:
            raise ValueError(
                "Unrecognized residual type: %s. Please choose from 'prefit', 'postfit', or 'both'."
                % (restype)
            )

    # Check if we want whitened residuals
    if whitened == True and ("res" not in kwargs.keys()):
        if avg == True and mixed_ecorr == True:
            if restype != "both":
                res = whiten_resids(avg_dict, restype=restype)
                res_no_avg = whiten_resids(no_avg_dict, restype=restype)
            else:
                res = whiten_resids(avg_dict_pre, restype="prefit")
                res_pre = whiten_resids(avg_dict, restype="postfit")
                res_pre = res_pre.to(u.us)
                res_no_avg = whiten_resids(avg_dict_pre, restype="prefit")
                res_pre_no_avg = whiten_resids(avg_dict, restype="postfit")
                res_pre_no_avg = res_pre_no_avg.to(u.us)
            res = res.to(u.us)
            res_no_avg = res_no_avg.to(u.us)
        elif avg == True and mixed_ecorr == False:
            if restype != "both":
                res = whiten_resids(avg_dict, restype=restype)
            else:
                res = whiten_resids(avg_dict_pre, restype="prefit")
                res_pre = whiten_resids(avg_dict, restype="postfit")
                res_pre = res_pre.to(u.us)
            res = res.to(u.us)
        else:
            if restype != "both":
                res = whiten_resids(fitter, restype=restype)
            else:
                res = whiten_resids(fitter, restype="prefit")
                res_pre = whiten_resids(fitter, restype="postfit")
                res_pre = res_pre.to(u.us)
            res = res.to(u.us)

    # Get errors
    if "errs" in kwargs.keys():
        errs = kwargs["errs"]
    else:
        if restype == "prefit":
            if avg == True and mixed_ecorr == True:
                errs = avg_dict["errors"].to(u.us)
                errs_no_avg = no_avg_dict["errors"].to(u.us)
            elif avg == True and mixed_ecorr == False:
                errs = avg_dict["errors"].to(u.us)
            else:
                errs = fitter.toas.get_errors().to(u.us)
        elif restype == "postfit":
            if NB == True:
                if avg == True and mixed_ecorr == True:
                    errs = avg_dict["errors"].to(u.us)
                    errs_no_avg = no_avg_dict["errors"].to(u.us)
                elif avg == True and mixed_ecorr == False:
                    errs = avg_dict["errors"].to(u.us)
                else:
                    errs = fitter.resids.get_data_error().to(u.us)
            else:
                errs = fitter.resids.residual_objs["toa"].get_data_error().to(u.us)
        elif restype == "both":
            if NB == True:
                if avg == True and mixed_ecorr == True:
                    errs = avg_dict["errors"].to(u.us)
                    errs_pre = avg_dict_pre["errors"].to(u.us)
                    errs_no_avg = no_avg_dict["errors"].to(u.us)
                    errs_no_avg_pre = no_avg_dict_pre["errors"].to(u.us)
                elif avg == True and mixed_ecorr == False:
                    errs = avg_dict["errors"].to(u.us)
                    errs_pre = avg_dict_pre["errors"].to(u.us)
                else:
                    errs = fitter.resids.get_data_error().to(u.us)
                    errs_pre = fitter.toas.get_errors().to(u.us)
            else:
                errs = fitter.resids.residual_objs["toa"].get_data_error().to(u.us)
                errs_pre = fitter.toas.get_errors().to(u.us)
    # Get MJDs
    if "mjds" in kwargs.keys():
        mjds = kwargs["mjds"]
    else:
        mjds = fitter.toas.get_mjds().value
        if avg == True and mixed_ecorr == True:
            mjds = avg_dict["mjds"].value
            mjds_no_avg = no_avg_dict["mjds"].value
            years_no_avg = (mjds_no_avg - 51544.0) / 365.25 + 2000.0

        elif avg == True and mixed_ecorr == False:
            mjds = avg_dict["mjds"].value
    # Convert to years
    years = (mjds - 51544.0) / 365.25 + 2000.0

    # In the end, we'll want to plot both ecorr avg & not ecorr avg at the same time if we have mixed ecorr.
    # Create combined arrays

    if avg == True and mixed_ecorr == True:
        combo_res = np.hstack((res, res_no_avg))
        combo_errs = np.hstack((errs, errs_no_avg))
        combo_years = np.hstack((years, years_no_avg))
        if restype == "both":
            combo_errs_pre = np.hstack((errs_pre, errs_no_avg_pre))
            combo_res_pre = np.hstack((res_pre, res_no_avg_pre))

    # Get colorby flag values (obs, PTA, febe, etc.)
    if "colorby" in kwargs.keys():
        cb = kwargs["colorby"]
    else:
        cb = np.array(fitter.toas[colorby])
        # .      Seems to run a little faster but not robust to obs?
        #        cb = np.array(fitter.toas.get_flag_value(colorby)[0])
        if avg == True:
            avg_cb = []
            for iis in avg_dict["indices"]:
                avg_cb.append(cb[iis[0]])
            if mixed_ecorr == True:
                no_avg_cb = []
                for jjs in no_avg_dict["indices"]:
                    no_avg_cb.append(cb[jjs])
                no_ecorr_cb = np.array(no_avg_cb)

            cb = np.array(avg_cb)

    # Get the set of unique flag values
    if avg == True and mixed_ecorr == True:
        cb = np.hstack((cb, no_ecorr_cb))

    CB = set(cb)

    colorscheme, markerscheme = set_color_and_marker(colorby)

    if "figsize" in kwargs.keys():
        figsize = kwargs["figsize"]
    else:
        figsize = (10, 5)
    if axs == None:
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
    else:
        fig = plt.gcf()
        ax1 = axs

    for i, c in enumerate(CB):
        inds = np.where(cb == c)[0]
        if not inds.tolist():
            cb_label = ""
        else:
            cb_label = cb[inds][0]
        # Get plot preferences
        if "fmt" in kwargs.keys():
            mkr = kwargs["fmt"]
        else:
            try:
                mkr = markers[cb_label]
                if restype == "both":
                    mkr_pre = "."
            except Exception:
                mkr = "x"
                log.log(1, "Color by Flag doesn't have a marker label!!")
        if "color" in kwargs.keys():
            clr = kwargs["color"]
        else:
            try:
                clr = colorscheme[cb_label]
            except Exception:
                clr = "k"
                log.log(1, "Color by Flag doesn't have a color!!")
        if "alpha" in kwargs.keys():
            alpha = kwargs["alpha"]
        else:
            alpha = 0.5
        if avg == True and mixed_ecorr == True:
            if plotsig:
                combo_sig = combo_res[inds] / combo_[inds]
                ax1.errorbar(
                    combo_years[inds],
                    combo_sig,
                    yerr=len(combo_errs[inds]) * [1],
                    fmt=mkr,
                    color=clr,
                    label=cb_label,
                    alpha=alpha,
                    picker=True,
                )
                if restype == "both":
                    combo_sig_pre = combo_res_pre[inds] / combo_errs_pre[inds]
                    ax1.errorbar(
                        combo_years[inds],
                        combo_sig_pre,
                        yerr=len(combo_errs_pre[inds]) * [1],
                        fmt=mkr_pre,
                        color=clr,
                        label=cb_label + " Prefit",
                        alpha=alpha,
                        picker=True,
                    )
            else:
                ax1.errorbar(
                    combo_years[inds],
                    combo_res[inds],
                    yerr=combo_errs[inds],
                    fmt=mkr,
                    color=clr,
                    label=cb_label,
                    alpha=alpha,
                    picker=True,
                )
                if restype == "both":
                    ax1.errorbar(
                        combo_years[inds],
                        combo_res_rpe[inds],
                        yerr=combo_errs_pre[inds],
                        fmt=mkr_pre,
                        color=clr,
                        label=cb_label + " Prefit",
                        alpha=alpha,
                        picker=True,
                    )

        else:
            if plotsig:
                sig = res[inds] / errs[inds]
                ax1.errorbar(
                    years[inds],
                    sig,
                    yerr=len(errs[inds]) * [1],
                    fmt=mkr,
                    color=clr,
                    label=cb_label,
                    alpha=alpha,
                    picker=True,
                )
                if restype == "both":
                    sig_pre = res_pre[inds] / errs_pre[inds]
                    ax1.errorbar(
                        years[inds],
                        sig_pre,
                        yerr=len(errs_pre[inds]) * [1],
                        fmt=mkr_pre,
                        color=clr,
                        label=cb_label + " Prefit",
                        alpha=alpha,
                        picker=True,
                    )
            else:
                ax1.errorbar(
                    years[inds],
                    res[inds],
                    yerr=errs[inds],
                    fmt=mkr,
                    color=clr,
                    label=cb_label,
                    alpha=alpha,
                    picker=True,
                )
                if restype == "both":
                    ax1.errorbar(
                        years[inds],
                        res_pre[inds],
                        yerr=errs_pre[inds],
                        fmt=mkr_pre,
                        color=clr,
                        label=cb_label + " Prefit",
                        alpha=alpha,
                        picker=True,
                    )

    # Set second axis
    ax1.set_xlabel(r"Year")
    ax1.grid(True)
    ax2 = ax1.twiny()
    mjd0 = ((ax1.get_xlim()[0]) - 2004.0) * 365.25 + 53005.0
    mjd1 = ((ax1.get_xlim()[1]) - 2004.0) * 365.25 + 53005.0
    ax2.set_xlim(mjd0, mjd1)
    if plotsig:
        if avg and whitened:
            ax1.set_ylabel(
                "Average Residual/Uncertainty \n (Whitened)", multialignment="center"
            )
        elif avg and not whitened:
            ax1.set_ylabel("Average Residual/Uncertainty")
        elif whitened and not avg:
            ax1.set_ylabel(
                "Residual/Uncertainty \n (Whitened)", multialignment="center"
            )
        else:
            ax1.set_ylabel("Residual/Uncertainty")
    else:
        if avg and whitened:
            ax1.set_ylabel(
                "Average Residual ($\\mu$s) \n (Whitened)", multialignment="center"
            )
        elif avg and not whitened:
            ax1.set_ylabel(r"Average Residual ($\mu$s)")
        elif whitened and not avg:
            ax1.set_ylabel("Residual ($\\mu$s) \n (Whitened)", multialignment="center")
        else:
            ax1.set_ylabel(r"Residual ($\mu$s)")
    if legend:
        if len(CB) > 5:
            ncol = int(np.ceil(len(CB) / 2))
            y_offset = 1.15
        else:
            ncol = len(CB)
            y_offset = 1.0
        ax1.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, y_offset + 1.0 / figsize[1]),
            ncol=ncol,
        )
    if title:
        if len(CB) > 5:
            y_offset = 1.1
        else:
            y_offset = 1.0
        plt.title(
            "%s %s timing residuals" % (fitter.model.PSR.value, restype),
            y=y_offset + 1.0 / figsize[1],
        )
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
        if restype == "prefit":
            ext += "_prefit"
        elif restype == "postfit":
            ext += "_postfit"
        elif restype == "both":
            ext += "_pre_post_fit"
        plt.savefig("%s_resid_v_mjd%s.png" % (fitter.model.PSR.value, ext))

    if axs == None:
        # Define clickable points
        text = ax2.text(0, 0, "")

        # Define point highlight color
        stamp_color = "#FD9927"

        def onclick(event):
            # Get X and Y axis data
            xdata = mjds
            if plotsig:
                ydata = (res / errs).decompose().value
            else:
                ydata = res.value
            # Get x and y data from click
            xclick = event.xdata
            yclick = event.ydata
            # Calculate scaled distance, find closest point index
            d = np.sqrt(((xdata - xclick) / 10.0) ** 2 + (ydata - yclick) ** 2)
            ind_close = np.where(np.min(d) == d)[0]
            # highlight clicked point
            ax2.scatter(xdata[ind_close], ydata[ind_close], marker="x", c=stamp_color)
            # Print point info
            text.set_position((xdata[ind_close], ydata[ind_close]))
            if plotsig:
                text.set_text(
                    "TOA Params:\n MJD: %s \n Res/Err: %.2f \n Index: %s"
                    % (xdata[ind_close][0], ydata[ind_close], ind_close[0])
                )
            else:
                text.set_text(
                    "TOA Params:\n MJD: %s \n Res: %.2f \n Index: %s"
                    % (xdata[ind_close][0], ydata[ind_close], ind_close[0])
                )

        fig.canvas.mpl_connect("button_press_event", onclick)

    return


def plot_FD_delay(
    fitter=None,
    model_object=None,
    save=False,
    title=True,
    axs=None,
    legend=True,
    show_bin=True,
    **kwargs,
):
    """
    Make a plot of frequency (MHz) vs the time delay (us) implied by FD parameters.
    Z. Arzoumanian, The NANOGrav Nine-year Data Set: Observations, Arrival
        Time Measurements, and Analysis of 37 Millisecond Pulsars, The
        Astrophysical Journal, Volume 813, Issue 1, article id. 65, 31 pp.(2015).
        Eq.(2):
        FDdelay = sum(c_i * (log(obs_freq/1GHz))^i)

    This can be run with EITHER a PINT fitter object OR PINT model object. If run with a model object, the user will need to specify which frequencies they would like to plot FD delays over.

    Arguments
    ----------

    fitter[object] : The PINT fitter object.
    model[object] : The PINT model object. Can be used instead of fitter
    save [boolean] : If True will save plot with the name "FD_delay.png"[default: False].
    title [boolean] : If False, will not print plot title [default: True].
    axs [string] : If not None, should be defined subplot value and the figure will be used as part of a
         larger figure [default: None].

    Optional Arguments:
    --------------------
    freqs [list/array] : List or array of frequencies (MHz) to plot. Will override values from toa object.
    show_bin [boolean] : Show the delay corresponding to 1 bin of the profile for comparison. (requires fitter)
    figsize [tuple] : Size of the figure passed to matplotlib [default: (8,4)].
    ls ['string'] : matplotlib format option for linestyle [default: ('-')]
    color ['string'] : matplotlib color option for plot [default: green]
    alpha [float] : matplotlib alpha options for error regions [default: 0.2]
    loc ['string'] : matplotlib legend location [default: 'upper right'] Only used when legend = True
    """

    # Make sure that either a fitter or model object has been specified
    if fitter == None and model_object == None:
        raise Exception("Need to specify either a fitter or model object")

    # Get frequencies
    if "freqs" in kwargs.keys():
        freqs = kwargs["freqs"]
    elif model_object is not None:
        raise Exception(
            "Using a PINT model object. Need to add list/array of frequencies to calculate FD delay over"
        )
    else:
        freqs = fitter.toas.get_freqs().value
        freqs = np.sort(freqs)

    # Get FD delay in units of milliseconds as a function of frequency. This will eventually by available in PINT and become redundant. PINT version may need to be modified to allow for calculation of error regions
    def get_FD_delay(pint_model_object, freqs):
        FD_map = model.TimingModel.get_prefix_mapping(pint_model_object, "FD")
        FD_names = list(FD_map.values())
        FD_names.reverse()
        FD_vals = []
        FD_uncert = []
        for i in FD_names:
            FD_vals.append(
                pint_model_object.get_params_dict(which="all", kind="value")[i]
            )
            FD_uncert.append(
                pint_model_object.get_params_dict(which="all", kind="uncertainty")[i]
            )
        FD_vals.append(0.0)
        FD_uncert.append(0.0)
        FD_vals = np.array(FD_vals)
        FD_uncert = np.array(FD_uncert)
        delay = np.polyval(FD_vals, np.log10(freqs))
        delta_delay_plus = np.polyval(FD_uncert + FD_vals, np.log10(freqs))
        delta_delay_minus = np.polyval(FD_vals - FD_uncert, np.log10(freqs))
        if len(FD_vals) - 1 > 1:
            FD_phrase = "FD1-%s" % (len(FD_vals) - 1)
        else:
            FD_phrase = "FD1"
        return delay * 1e6, delta_delay_plus * 1e6, delta_delay_minus * 1e6, FD_phrase

    # Get FD params if fitter object is given
    if fitter is not None:
        # Check if the fitter object has FD parameters
        try:
            FD_delay, FD_delay_err_plus, FD_delay_err_minus, legend_text = get_FD_delay(
                fitter.model, freqs * 1e-3
            )
            psr_name = fitter.model.PSR.value
            """For when new version of PINT is default on pint_pal
        FD_delay = pint.models.frequency_dependent.FD.FD_delay(fitter.model,freqs)

            """
            if show_bin:
                nbins = fitter.toas["nbin"].astype(int).min()
                P0 = 1 / fitter.model.F0.value
                P0_bin_max = P0 / nbins
        except:
            print("No FD parameters in this model! Exitting...")

    # Get FD params if model object is given
    if model_object is not None:
        # Check if the model object has FD parameters
        try:
            FD_delay, FD_delay_err_plus, FD_delay_err_minus, legend_text = get_FD_delay(
                model_object, freqs * 1e-3
            )
            psr_name = model_object.PSR.value
            """For when new version of PINT is default on pint_pal
        FD_delay = pint.models.frequency_dependent.FD.FD_delay(fitter.model,freqs)

            """
            if show_bin:
                print(
                    "show_bin requires a fitter object, cannot be used with the model alone"
                )
                show_bin = False
        except:
            print("No FD parameters in this model! Exitting...")

    # Get plotting preferences.
    if "figsize" in kwargs.keys():
        figsize = kwargs["figsize"]
    else:
        figsize = (8, 4)
    if axs == None:
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
    else:
        fig = plt.gcf()
        ax1 = axs
    if "ls" in kwargs.keys():
        linestyle = kwargs["ls"]
    else:
        linestyle = "-"
    if "color" in kwargs.keys():
        clr = kwargs["color"]
    else:
        clr = "green"
    if "alpha" in kwargs.keys():
        alpha = kwargs["alpha"]
    else:
        alpha = 0.2
    if "loc" in kwargs.keys():
        loc = kwargs["loc"]
    else:
        loc = "upper right"
    # Plot frequency (MHz) vs delay (microseconds)
    ax1.plot(freqs, FD_delay, label=legend_text, color=clr, ls=linestyle)
    ax1.fill_between(
        freqs, FD_delay_err_plus, FD_delay_err_minus, color=clr, alpha=alpha
    )
    if show_bin:
        if (FD_delay > 0).any():
            ax1.axhline(P0_bin_max * 1e6, label="1 profile bin")
        if (FD_delay < 0).any():
            ax1.axhline(-P0_bin_max * 1e6, label="1 profile bin")
    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel(r"Delay ($\mu$s)")
    if title:
        ax1.set_title("%s FD Delay" % psr_name)
    if legend:
        ax1.legend(loc=loc)
    if axs == None:
        plt.tight_layout()
    if save:
        plt.savefig("%s_fd_delay.png" % psr_name)

    return


def plot_residuals_freq(
    fitter,
    restype="postfit",
    colorby="f",
    plotsig=False,
    avg=False,
    mixed_ecorr=False,
    whitened=False,
    save=False,
    legend=True,
    title=True,
    axs=None,
    **kwargs,
):
    """
    Make a plot of the residuals vs. frequency


    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    restype ['string'] : Type of residuals, pre or post fit, to plot from fitter object. Options are:
        'prefit' - plot the prefit residuals.
        'postfit' - plot the postfit residuals (default)
        'both' - overplot both the pre and post-fit residuals.
    colorby ['string']:  What to use to determine color/markers
        'pta' - color residuals by PTA (default)
        'obs' - color residuals by telescope
        'f'   - color residuals by frontend/backend pair (flag not used by all PTAs).
    plotsig [boolean] : If True plot number of measurements v. residuals/uncertainty, else v. residuals
        [default: False].
    avg [boolean] : If True and not wideband fitter, will compute and plot epoch-average residuals [default: False].
    mixed_ecorr [boolean]: If True, supports combining already-epoch-averaged residuals with epoch-averaging [default: False]
    whitened [boolean] : If True will compute and plot whitened residuals [default: False].
    save [boolean] : If True will save plot with the name "resid_v_freq.png" Will add averaged/whitened
         as necessary [default: False].
    legend [boolean] : If False, will not print legend with plot [default: True].
    title [boolean] : If False, will not print plot title [default: True].
    axs [string] : If not None, should be defined subplot value and the figure will be used as part of a
         larger figure [default: None].

    Optional Arguments:
    --------------------
    res [list/array] : List or array of residual values to plot. Will override values from fitter object.
    errs [list/array] : List or array of residual error values to plot. Will override values from fitter object.
    freqs [list/array] : List or array of frequencies (MHz) to plot. Will override values from toa object.
    rcvr_bcknds[list/array] : List or array of TOA receiver-backend combinations. Will override values from toa object.
    figsize [tuple] : Size of the figure passed to matplotlib [default: (10,4)].
    fmt ['string'] : matplotlib format option for markers [default: ('x')]
    color ['string'] : matplotlib color option for plot [default: color dictionary in plot_utils.py file]
    alpha [float] : matplotlib alpha options for plot points [default: 0.5]
    """
    # Check if wideband
    if fitter.is_wideband:
        NB = False
        if avg == True:
            raise ValueError(
                "Cannot epoch average wideband residuals, please change 'avg' to False."
            )
    else:
        NB = True

    # Check if want epoch averaged residuals
    if avg == True and restype == "prefit" and mixed_ecorr == True:
        avg_dict = fitter.resids_init.ecorr_average(use_noise_model=True)
        no_avg_dict = no_ecorr_average(
            fitter.toas, fitter.resids_init, use_noise_model=True
        )
    elif avg == True and restype == "postfit" and mixed_ecorr == True:
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
        no_avg_dict = no_ecorr_average(fitter.toas, fitter.resids, use_noise_model=True)
    elif avg == True and restype == "both" and mixed_ecorr == True:
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
        no_avg_dict = no_ecorr_average(fitter.toas, fitter.resids, use_noise_model=True)
        avg_dict_pre = fitter.resids_init.ecorr_average(use_noise_model=True)
        no_avg_dict_pre = no_ecorr_average(
            fitter.toas, fitter.resids_init, use_noise_model=True
        )
    elif avg == True and restype == "prefit" and mixed_ecorr == False:
        avg_dict = fitter.resids_init.ecorr_average(use_noise_model=True)
    elif avg == True and restype == "postfit" and mixed_ecorr == False:
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
    elif avg == True and restype == "both" and mixed_ecorr == False:
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
        avg_dict_pre = fitter.resids_init.ecorr_average(use_noise_model=True)

    # Get residuals
    if "res" in kwargs.keys():
        res = kwargs["res"]
    else:
        if restype == "prefit":
            if NB == True:
                if avg == True and mixed_ecorr == True:
                    res = avg_dict["time_resids"].to(u.us)
                    res_no_avg = no_avg_dict["time_resids"].to(u.us)
                elif avg == True and mixed_ecorr == False:
                    res = avg_dict["time_resids"].to(u.us)
                else:
                    res = fitter.resids_init.time_resids.to(u.us)
            else:
                res = fitter.resids_init.residual_objs["toa"].time_resids.to(u.us)
        elif restype == "postfit":
            if NB == True:
                if avg == True and mixed_ecorr == True:
                    res = avg_dict["time_resids"].to(u.us)
                    res_no_avg = no_avg_dict["time_resids"].to(u.us)
                elif avg == True:
                    res = avg_dict["time_resids"].to(u.us)
                else:
                    res = fitter.resids.time_resids.to(u.us)
            else:
                res = fitter.resids.residual_objs["toa"].time_resids.to(u.us)
        elif restype == "both":
            if NB == True:
                if avg == True and mixed_ecorr == True:
                    res = avg_dict["time_resids"].to(u.us)
                    res_no_avg = no_avg_dict["time_resids"].to(u.us)
                    res_pre = avg_dict_pre["time_resids"].to(u.us)
                    res_pre_no_avg = no_avg_dict_pre["time_resids"].to(u.us)
                elif avg == True and mixed_ecorr == False:
                    res = avg_dict["time_resids"].to(u.us)
                    res_pre = avg_dict_pre["time_resids"].to(u.us)
                else:
                    res = fitter.resids.time_resids.to(u.us)
                    res_pre = fitter.resids_init.time_resids.to(u.us)
            else:
                res = fitter.resids.residual_objs["toa"].time_resids.to(u.us)
                res_pre = fitter.resids_init.residual_objs["toa"].time_resids.to(u.us)
        else:
            raise ValueError(
                "Unrecognized residual type: %s. Please choose from 'prefit', 'postfit', or 'both'."
                % (restype)
            )

    # Check if we want whitened residuals
    if whitened == True and ("res" not in kwargs.keys()):
        if avg == True and mixed_ecorr == True:
            if restype != "both":
                res = whiten_resids(avg_dict, restype=restype)
                res_no_avg = whiten_resids(no_avg_dict, restype=restype)
            else:
                res = whiten_resids(avg_dict_pre, restype="prefit")
                res_pre = whiten_resids(avg_dict, restype="postfit")
                res_pre = res_pre.to(u.us)
                res_no_avg = whiten_resids(avg_dict_pre, restype="prefit")
                res_pre_no_avg = whiten_resids(avg_dict, restype="postfit")
                res_pre_no_avg = res_pre_no_avg.to(u.us)
            res = res.to(u.us)
            res_no_avg = res_no_avg.to(u.us)
        elif avg == True and mixed_ecorr == False:
            if restype != "both":
                res = whiten_resids(avg_dict, restype=restype)
            else:
                res = whiten_resids(avg_dict_pre, restype="prefit")
                res_pre = whiten_resids(avg_dict, restype="postfit")
                res_pre = res_pre.to(u.us)
            res = res.to(u.us)
        else:
            if restype != "both":
                res = whiten_resids(fitter, restype=restype)
            else:
                res = whiten_resids(fitter, restype="prefit")
                res_pre = whiten_resids(fitter, restype="postfit")
                res_pre = res_pre.to(u.us)
            res = res.to(u.us)

    # Get errors
    if "errs" in kwargs.keys():
        errs = kwargs["errs"]
    else:
        if restype == "prefit":
            if avg == True and mixed_ecorr == True:
                errs = avg_dict["errors"].to(u.us)
                errs_no_avg = no_avg_dict["errors"].to(u.us)
            elif avg == True and mixed_ecorr == False:
                errs = avg_dict["errors"].to(u.us)
            else:
                errs = fitter.toas.get_errors().to(u.us)
        elif restype == "postfit":
            if NB == True:
                if avg == True and mixed_ecorr == True:
                    errs = avg_dict["errors"].to(u.us)
                    errs_no_avg = no_avg_dict["errors"].to(u.us)
                elif avg == True and mixed_ecorr == False:
                    errs = avg_dict["errors"].to(u.us)
                else:
                    errs = fitter.resids.get_data_error().to(u.us)
            else:
                errs = fitter.resids.residual_objs["toa"].get_data_error().to(u.us)
        elif restype == "both":
            if NB == True:
                if avg == True and mixed_ecorr == True:
                    errs = avg_dict["errors"].to(u.us)
                    errs_pre = avg_dict_pre["errors"].to(u.us)
                    errs_no_avg = no_avg_dict["errors"].to(u.us)
                    errs_no_avg_pre = no_avg_dict_pre["errors"].to(u.us)
                elif avg == True and mixed_ecorr == False:
                    errs = avg_dict["errors"].to(u.us)
                    errs_pre = avg_dict_pre["errors"].to(u.us)
                else:
                    errs = fitter.resids.get_data_error().to(u.us)
                    errs_pre = fitter.toas.get_errors().to(u.us)
            else:
                errs = fitter.resids.residual_objs["toa"].get_data_error().to(u.us)
                errs_pre = fitter.toas.get_errors().to(u.us)

    # In the end, we'll want to plot both ecorr avg & not ecorr avg at the same time if we have mixed ecorr.
    # Create combined arrays

    if avg == True and mixed_ecorr == True:
        combo_res = np.hstack((res, res_no_avg))
        combo_errs = np.hstack((errs, errs_no_avg))
        if restype == "both":
            combo_errs_pre = np.hstack((errs_pre, errs_no_avg_pre))
            combo_res_pre = np.hstack((res_pre, res_no_avg_pre))

    # Get freqs
    if "freqs" in kwargs.keys():
        freqs = kwargs["freqs"]
    else:
        freqs = fitter.toas.get_freqs().value

    # Get colorby flag values (obs, PTA, febe, etc.)
    if "colorby" in kwargs.keys():
        cb = kwargs["colorby"]
    else:
        cb = np.array(fitter.toas[colorby])
        # .      Seems to run a little faster but not robust to obs?
        #        cb = np.array(fitter.toas.get_flag_value(colorby)[0])
        if avg == True:
            avg_cb = []
            for iis in avg_dict["indices"]:
                avg_cb.append(cb[iis[0]])
            if mixed_ecorr == True:
                no_avg_cb = []
                for jjs in no_avg_dict["indices"]:
                    no_avg_cb.append(cb[jjs])
                no_ecorr_cb = np.array(no_avg_cb)

            cb = np.array(avg_cb)

    # Get the set of unique flag values
    if avg == True and mixed_ecorr == True:
        cb = np.hstack((cb, no_ecorr_cb))

    CB = set(cb)

    colorscheme, markerscheme = set_color_and_marker(colorby)

    if "figsize" in kwargs.keys():
        figsize = kwargs["figsize"]
    else:
        figsize = (10, 4)
    if axs == None:
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
    else:
        fig = plt.gcf()
        ax1 = axs

    for i, c in enumerate(CB):
        inds = np.where(cb == c)[0]
        if not inds.tolist():
            cb_label = ""
        else:
            cb_label = cb[inds][0]
        # Get plot preferences
        if "fmt" in kwargs.keys():
            mkr = kwargs["fmt"]
        else:
            try:
                mkr = markerscheme[cb_label]
                if restype == "both":
                    mkr_pre = "."
            except Exception:
                mkr = "x"
                if restype == "both":
                    mkr_pre = "."
                log.log(1, "Color by Flag doesn't have a marker label!!")
        if "color" in kwargs.keys():
            clr = kwargs["color"]
        else:
            try:
                clr = colorscheme[cb_label]
            except Exception:
                clr = "k"
                log.log(1, "Color by Flag doesn't have a color!!")
        if "alpha" in kwargs.keys():
            alpha = kwargs["alpha"]
        else:
            alpha = 0.5

        if avg and mixed_ecorr:
            if plotsig:
                combo_sig = combo_res[inds] / combo_errs[inds]
                ax1.errorbar(
                    freqs[inds],
                    combo_sig,
                    yerr=len(combo_errs[inds]) * [1],
                    fmt=mkr,
                    color=clr,
                    label=cb_label,
                    alpha=alpha,
                    picker=True,
                )
                if restype == "both":
                    combo_sig_pre = combo_res_pre[inds] / combo_errs_pre[inds]
                    ax1.errorbar(
                        freqs[inds],
                        combo_sig_pre,
                        yerr=len(combo_errs_pre[inds]) * [1],
                        fmt=mkr_pre,
                        color=clr,
                        label=cb_label + " Prefit",
                        alpha=alpha,
                        picker=True,
                    )
            else:
                ax1.errorbar(
                    freqs[inds],
                    combo_res[inds],
                    yerr=combo_errs[inds],
                    fmt=mkr,
                    color=clr,
                    label=cb_label,
                    alpha=alpha,
                    picker=True,
                )
                if restype == "both":
                    ax1.errorbar(
                        freqs[inds],
                        combo_res_pre[inds],
                        yerr=combo_errs_pre[inds],
                        fmt=mkr_pre,
                        color=clr,
                        label=cb_label + " Prefit",
                        alpha=alpha,
                        picker=True,
                    )
        else:
            if plotsig:
                sig = res[inds] / errs[inds]
                ax1.errorbar(
                    freqs[inds],
                    sig,
                    yerr=len(errs[inds]) * [1],
                    fmt=mkr,
                    color=clr,
                    label=cb_label,
                    alpha=alpha,
                    picker=True,
                )
                if restype == "both":
                    sig_pre = res_pre[inds] / errs_pre[inds]
                    ax1.errorbar(
                        freqs[inds],
                        sig_pre,
                        yerr=len(errs_pre[inds]) * [1],
                        fmt=mkr_pre,
                        color=clr,
                        label=cb_label + " Prefit",
                        alpha=alpha,
                        picker=True,
                    )
            else:
                ax1.errorbar(
                    freqs[inds],
                    res[inds],
                    yerr=errs[inds],
                    fmt=mkr,
                    color=clr,
                    label=cb_label,
                    alpha=alpha,
                    picker=True,
                )
                if restype == "both":
                    ax1.errorbar(
                        freqs[inds],
                        res_pre[inds],
                        yerr=errs_pre[inds],
                        fmt=mkr_pre,
                        color=clr,
                        label=cb_label + " Prefit",
                        alpha=alpha,
                        picker=True,
                    )

    # Set axis
    ax1.set_xlabel(r"Frequency (MHz)")
    ax1.grid(True)
    if plotsig:
        if avg and whitened:
            ax1.set_ylabel(
                "Average Residual/Uncertainty \n (Whitened)", multialignment="center"
            )
        elif avg and not whitened:
            ax1.set_ylabel("Average Residual/Uncertainty")
        elif whitened and not avg:
            ax1.set_ylabel(
                "Residual/Uncertainty \n (Whitened)", multialignment="center"
            )
        else:
            ax1.set_ylabel("Residual/Uncertainty")
    else:
        if avg and whitened:
            ax1.set_ylabel(
                "Average Residual ($\\mu$s) \n (Whitened)", multialignment="center"
            )
        elif avg and not whitened:
            ax1.set_ylabel(r"Average Residual ($\mu$s)")
        elif whitened and not avg:
            ax1.set_ylabel("Residual ($\\mu$s) \n (Whitened)", multialignment="center")
        else:
            ax1.set_ylabel(r"Residual ($\mu$s)")
    if legend:
        if len(CB) > 5:
            ncol = int(np.ceil(len(CB) / 2))
            y_offset = 1.15
        else:
            ncol = len(CB)
            y_offset = 1.0
        ax1.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, y_offset + 1.0 / figsize[1]),
            ncol=ncol,
        )
    if title:
        if len(CB) > 5:
            y_offset = 1.1
        else:
            y_offset = 1.0
        plt.title(
            "%s %s frequency residuals" % (fitter.model.PSR.value, restype),
            y=y_offset + 1.0 / figsize[1],
        )
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
        if restype == "prefit":
            ext += "_prefit"
        elif restype == "postfit":
            ext += "_postfit"
        elif restype == "both":
            ext += "_pre_post_fit"
        plt.savefig("%s_resid_v_freq%s.png" % (fitter.model.PSR.value, ext))

    if axs == None:
        # Define clickable points
        text = ax1.text(0, 0, "")
        stamp_color = "#FD9927"

        def onclick(event):
            # Get X and Y axis data
            xdata = freqs
            if plotsig:
                ydata = (res / errs).decompose().value
            else:
                ydata = res.value
            # Get x and y data from click
            xclick = event.xdata
            yclick = event.ydata
            # Calculate scaled distance, find closest point index
            d = np.sqrt(((xdata - xclick) / 10.0) ** 2 + (ydata - yclick) ** 2)
            ind_close = np.where(np.min(d) == d)[0]
            # highlight clicked point
            ax1.scatter(xdata[ind_close], ydata[ind_close], marker="x", c=stamp_color)
            # Print point info
            text.set_position((xdata[ind_close], ydata[ind_close]))
            if plotsig:
                text.set_text(
                    "TOA Params:\n Frequency: %s \n Res/Err: %.2f \n Index: %s"
                    % (xdata[ind_close][0], ydata[ind_close], ind_close[0])
                )
            else:
                text.set_text(
                    "TOA Params:\n Frequency: %s \n Res: %.2f \n Index: %s"
                    % (xdata[ind_close][0], ydata[ind_close], ind_close[0])
                )

        fig.canvas.mpl_connect("button_press_event", onclick)

    return


def plot_dmx_time(
    fitter,
    savedmx=False,
    save=False,
    legend=True,
    axs=None,
    title=True,
    compare=False,
    **kwargs,
):
    """
    Make a plot of DMX vs. time


    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    savedmx [boolean] : If True will save dmxparse values to output file with `"psrname"_dmxparse.nb/wb/.out`.
    save [boolean] : If True will save plot with the name "resid_v_mjd.png" Will add averaged/whitened
         as necessary [default: False].
    legend [boolean] : If False, will not print legend with plot [default: True].
    title [boolean] : If False, will not print plot title [default: True].
    axs [string] : If not None, should be defined subplot value and the figure will be used as part of a
         larger figure [default: None].

    Optional Arguments:
    --------------------
    dmx [list/array] : List or array of DMX values to plot. Will override values from fitter object.
    errs [list/array] : List or array of DMX error values to plot. Will override values from fitter object.
    mjds [list/array] : List or array of DMX mjd epoch values to plot. Will override values from fitter object.
    figsize [tuple] : Size of the figure passed to matplotlib [default: (10,4)].
    fmt ['string'] : matplotlib format option for markers [default: ('x')]
    color ['string'] : matplotlib color option for plot [default: color dictionary in plot_utils.py file]
    alpha [float] : matplotlib alpha options for plot points [default: 0.5]
    """
    # Get pulsar name
    psrname = fitter.model.PSR.value

    # Check if wideband
    if fitter.is_wideband:
        NB = False
        dmxname = "%s_dmxparse.wb.out" % (psrname)
    else:
        NB = True
        dmxname = "%s_dmxparse.nb.out" % (psrname)

    # Get plotting dmx and error values for WB
    if "dmx" in kwargs.keys():
        DMXs = kwargs["dmx"]
    else:
        # get dmx dictionary from pint dmxparse function
        dmx_dict = pu.dmxparse(fitter, save="dmxparse.out")
        DMXs = dmx_dict["dmxs"].value
        DMX_vErrs = dmx_dict["dmx_verrs"].value
        DMX_center_MJD = dmx_dict["dmxeps"].value
        DMX_center_Year = (DMX_center_MJD - 51544.0) / 365.25 + 2000.0
        # move file name
        if savedmx:
            os.rename("dmxparse.out", dmxname)

    # Double check/overwrite errors if necessary
    if "errs" in kwargs.keys():
        DMX_vErrs = kwargs["errs"]
    # Double check/overwrite dmx mjd epochs if necessary
    if "mjds" in kwargs.keys():
        DMX_center_MJD = kwargs["mjds"]
        DMX_center_Year = (DMX_center_MJD - 51544.0) / 365.25 + 2000.0

    # If we want to compare WB to NB, we need to look for the right output file
    if compare == True:
        # Look for other dmx file
        if NB:
            # log.log(1, "Searching for file: %s_dmxparse.wb.out" % (psrname))
            if not os.path.isfile("%s_dmxparse.wb.out" % (psrname)):
                raise RuntimeError("Cannot find Wideband DMX parse output file.")
            else:
                # Get the values from the DMX parse file
                dmx_epochs, nb_dmx, nb_dmx_var, nb_dmx_r1, nb_dmx_r2 = np.loadtxt(
                    "%s_dmxparse.wb.out" % (psrname),
                    unpack=True,
                    usecols=(0, 1, 2, 3, 4),
                )
        else:
            # log.log(1, "Searching for file: %s_dmxparse.nb.out" % (psrname))
            if not os.path.isfile("%s_dmxparse.nb.out" % (psrname)):
                raise RuntimeError("Cannot find Narrowband DMX parse output file.")
            else:
                # Get the values from the DMX parse file
                dmx_epochs, nb_dmx, nb_dmx_var, nb_dmx_r1, nb_dmx_r2 = np.loadtxt(
                    "%s_dmxparse.nb.out" % (psrname),
                    unpack=True,
                    usecols=(0, 1, 2, 3, 4),
                )
        dmx_mid_yr = (dmx_epochs - 51544.0) / 365.25 + 2000.0

    # Define the plotting function
    if axs == None:
        if "figsize" in kwargs.keys():
            figsize = kwargs["figsize"]
        else:
            figsize = (10, 4)
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
    else:
        ax1 = axs
    # Get plot preferences
    if "fmt" in kwargs.keys():
        mkr = kwargs["fmt"]
    else:
        mkr = "s"
        if compare:
            mkr_nb = "o"
    if "color" in kwargs.keys():
        clr = kwargs["color"]
    else:
        clr = "gray"
        if compare:
            clr_nb = "k"
    if "alpha" in kwargs.keys():
        alpha = kwargs["alpha"]
    else:
        alpha = 1.0
    # Not actually plot
    if NB and not compare:
        ax1.errorbar(
            DMX_center_Year,
            DMXs * 10**3,
            yerr=DMX_vErrs * 10**3,
            fmt=".",
            c=clr,
            marker=mkr,
            label="Narrowband",
        )
    elif not NB and not compare:
        ax1.errorbar(
            DMX_center_Year,
            DMXs * 10**3,
            yerr=DMX_vErrs * 10**3,
            fmt=".",
            c=clr,
            marker=mkr,
            label="Wideband",
        )
    elif compare:
        if NB:
            ax1.errorbar(
                DMX_center_Year,
                DMXs * 10**3,
                yerr=DMX_vErrs * 10**3,
                fmt=".",
                c=clr,
                marker=mkr,
                label="Narrowband",
            )
            ax1.errorbar(
                dmx_mid_yr,
                nb_dmx * 10**3,
                yerr=nb_dmx_var * 10**3,
                fmt=".",
                color=clr_nb,
                marker=mkr_nb,
                label="Wideband",
            )
        else:
            ax1.errorbar(
                DMX_center_Year,
                DMXs * 10**3,
                yerr=DMX_vErrs * 10**3,
                fmt=".",
                c=clr,
                marker=mkr,
                label="Wideband",
            )
            ax1.errorbar(
                dmx_mid_yr,
                nb_dmx * 10**3,
                yerr=nb_dmx_var * 10**3,
                fmt=".",
                color=clr_nb,
                marker=mkr_nb,
                label="Narrowband",
            )

    # Set second axis
    ax1.set_xlabel(r"Year")
    ax1.grid(True)
    ax2 = ax1.twiny()
    mjd0 = ((ax1.get_xlim()[0]) - 2004.0) * 365.25 + 53005.0
    mjd1 = ((ax1.get_xlim()[1]) - 2004.0) * 365.25 + 53005.0
    ax2.set_xlim(mjd0, mjd1)
    ax1.set_ylabel(r"DMX ($10^{-3}$ pc cm$^{-3}$)")
    if legend:
        ax1.legend(loc="best")
    if title:
        if NB and not compare:
            plt.title("%s narrowband dmx" % (psrname), y=1.0 + 1.0 / figsize[1])
        elif not NB and not compare:
            plt.title("%s wideband dmx" % (psrname), y=1.0 + 1.0 / figsize[1])
        elif compare:
            plt.title(
                "%s narrowband and wideband dmx" % (psrname), y=1.0 + 1.0 / figsize[1]
            )
    if axs == None:
        plt.tight_layout()
    if save:
        ext = ""
        if NB and not compare:
            ext += "_NB"
        if not NB and not compare:
            ext += "_WB"
        if compare:
            ext += "_NB_WB_compare"
        plt.savefig("%s_dmx_v_time%s.png" % (psrname, ext))

    if axs == None:
        # Define clickable points
        text = ax1.text(0, 0, "")
        # Define color for highlighting points
        stamp_color = "#FD9927"

        def onclick(event):
            # Get X and Y axis data
            xdata = DMX_center_Year
            ydata = DMXs * 10**3
            # Get x and y data from click
            xclick = event.xdata
            yclick = event.ydata
            # Calculate scaled distance, find closest point index
            d = np.sqrt(((xdata - xclick) / 1000.0) ** 2 + (ydata - yclick) ** 2)
            ind_close = np.where(np.min(d) == d)[0]
            # highlight clicked point
            ax2.scatter(xdata[ind_close], ydata[ind_close], marker="s", c=stamp_color)
            # Print point info
            text.set_position((xdata[ind_close], ydata[ind_close]))
            text.set_text(
                "DMX Params:\n MJD: %s \n DMX: %.2f \n Index: %s"
                % (xdata[ind_close][0], ydata[ind_close], ind_close[0])
            )

        fig.canvas.mpl_connect("button_press_event", onclick)

    return


def plot_dmxout(dmxout_files, labels, psrname=None, outfile=None, model=None):
    """Make simple dmx vs. time plot with dmxout file(s) as input

    Parameters
    ==========
    dmxout_files: list/str
        list of dmxout files to plot (or str if only one)
    labels: list/str
        list of labels for dmx timeseries (or str if only one)
    psrname: str, optional
        pulsar name
    outfile: str, optional
        save figure and write to outfile if set
    model: `pint.model.TimingModel` object, optional
        if provided, plot excess DM based on a basic SWM (NE_SW = 10.0)

    Returns
    =======
    dmxDict: dictionary
        dmxout information (mjd, val, err, r1, r2) for each label
    """
    from astropy.time import Time

    if isinstance(dmxout_files, str):
        dmxout_files = [dmxout_files]
    if isinstance(labels, str):
        labels = [labels]

    figsize = (10, 4)
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(r"Year")
    ax1.set_ylabel(r"DMX ($10^{-3}$ pc cm$^{-3}$)")
    ax1.grid(True)
    ax2 = ax1.twiny()
    ax2.set_xlabel("MJD")

    dmxDict = {}
    for ii, (df, lab) in enumerate(zip(dmxout_files, labels)):
        dmxmjd, dmxval, dmxerr, dmxr1, dmxr2 = np.loadtxt(
            df, unpack=True, usecols=range(0, 5)
        )
        idmxDict = {
            "mjd": dmxmjd,
            "val": dmxval,
            "err": dmxerr,
            "r1": dmxr1,
            "r2": dmxr2,
        }
        ax2.errorbar(
            dmxmjd,
            dmxval * 10**3,
            yerr=dmxerr * 10**3,
            label=lab,
            marker="o",
            ls="",
            markerfacecolor="none",
        )
        dmxDict[lab] = idmxDict

    # set ax1 lims (year) based on ax2 lims (mjd)
    mjd_xlo, mjd_xhi = ax2.get_xlim()
    dy_xlo = Time(mjd_xlo, format="mjd").decimalyear
    dy_xhi = Time(mjd_xhi, format="mjd").decimalyear
    ax1.set_xlim(dy_xlo, dy_xhi)

    # capture ylim
    orig_ylim = ax2.get_ylim()

    if psrname:
        ax1.text(
            0.975,
            0.05,
            psrname,
            transform=ax1.transAxes,
            size=18,
            c="lightgray",
            horizontalalignment="right",
            verticalalignment="bottom",
        )
    if model:
        from pint.simulation import make_fake_toas_fromMJDs
        from pint_pal.lite_utils import remove_noise

        fake_mjds = np.linspace(
            np.min(dmxmjd), np.max(dmxmjd), num=int(np.max(dmxmjd) - np.min(dmxmjd))
        )
        fake_mjdTime = Time(fake_mjds, format="mjd")

        # copy the model and add sw component
        mo_swm = copy.deepcopy(model)
        remove_noise(mo_swm)  # Not necessary here and avoids lots of warnings
        mo_swm.NE_SW.value = 10.0

        # generate fake TOAs and calculate excess DM due to solar wind
        fake_toas = make_fake_toas_fromMJDs(fake_mjdTime, mo_swm)
        sun_dm_delays = mo_swm.solar_wind_dm(fake_toas) * 10**3  # same scaling as above
        ax2.plot(fake_mjds, sun_dm_delays, c="lightgray", label="Excess DM")

    # don't change ylim based on excess dm trace, if plotted
    ax2.set_ylim(orig_ylim)
    ax2.legend(loc="best")

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
    return dmxDict


def plot_dmx_diffs_nbwb(dmxDict, show_missing=True, psrname=None, outfile=None):
    """Uses output dmxDict from plot_dmxout() to plot diffs between simultaneous nb-wb values

    Parameters
    ==========
    dmxDict: dictionary
        dmxout information (mjd, val, err, r1, r2) for nb, wb, respectively (check both exist)
    show_missing: bool, optional
        if one value is missing (nb/wb), indicate the epoch and which one
    psrname: str, optional
        pulsar name
    outfile: str, optional
        save figure and write to outfile if set

    Returns
    =======
    None?
    """
    # should check that both nb/wb entries exist first...
    nbmjd = dmxDict["nb"]["mjd"]
    wbmjd = dmxDict["wb"]["mjd"]
    allmjds = set(list(nbmjd) + list(wbmjd))

    # May need slightly more curation if nb/wb mjds are *almost* identical
    wbonly = allmjds - set(nbmjd)
    nbonly = allmjds - set(wbmjd)
    both = set(nbmjd).intersection(set(wbmjd))

    # assemble arrays of common inds for plotting later; probably a better way to do this
    nb_common_inds = []
    wb_common_inds = []
    for b in both:
        nb_common_inds.append(np.where(nbmjd == b)[0][0])
        wb_common_inds.append(np.where(wbmjd == b)[0][0])

    nb_common_inds, wb_common_inds = np.array(nb_common_inds), np.array(wb_common_inds)

    nbdmx, nbdmxerr = dmxDict["nb"]["val"], dmxDict["nb"]["err"]
    wbdmx, wbdmxerr = dmxDict["wb"]["val"], dmxDict["wb"]["err"]

    # propagate errors as quadrature sum, though Michael thinks geometric mean might be better?
    nbwb_dmx_diffs = nbdmx[nb_common_inds] - wbdmx[wb_common_inds]
    nbwb_err_prop = np.sqrt(
        nbdmxerr[nb_common_inds] ** 2 + wbdmxerr[wb_common_inds] ** 2
    )

    # make the plot
    from astropy.time import Time

    figsize = (10, 4)
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(r"Year")
    ax1.set_ylabel(r"$\Delta$DMX ($10^{-3}$ pc cm$^{-3}$)")
    ax1.grid(True)
    ax2 = ax1.twiny()
    ax2.set_xlabel("MJD")

    botharray = np.array(list(both))
    mjdbothTime = Time(botharray, format="mjd")
    dybothTime = mjdbothTime.decimalyear

    minmjd, maxmjd = np.sort(botharray)[0], np.sort(botharray)[-1]
    ax2.set_xlim(minmjd, maxmjd)

    ax1.errorbar(
        dybothTime,
        nbwb_dmx_diffs * 1e3,
        yerr=nbwb_err_prop * 1e3,
        marker="o",
        ls="",
        markerfacecolor="none",
        label="nb - wb",
    )

    # want arrows indicating missing nb/wb DMX values to difference
    if show_missing:
        stddiffs = np.std(nbwb_dmx_diffs)
        mjdnbonlyTime = Time(np.array(list(nbonly)), format="mjd")
        dynbonlyTime = mjdnbonlyTime.decimalyear
        ax1.scatter(
            dynbonlyTime,
            np.zeros(len(nbonly)) + stddiffs * 1e3,
            marker="v",
            c="r",
            label="nb only",
        )
        nbonlystr = [str(no) for no in nbonly]
        if nbonlystr:
            log.warning(
                f"nb-only measurements available for MJDs: {', '.join(nbonlystr)}"
            )

        mjdwbonlyTime = Time(np.array(list(wbonly)), format="mjd")
        dywbonlyTime = mjdwbonlyTime.decimalyear
        ax1.scatter(
            dywbonlyTime,
            np.zeros(len(wbonly)) - stddiffs * 1e3,
            marker="^",
            c="r",
            label="wb only",
        )
        wbonlystr = [str(wo) for wo in wbonly]
        if wbonlystr:
            log.warning(
                f"wb-only measurements available for MJDs: {', '.join(wbonlystr)}"
            )

    if psrname:
        ax1.text(
            0.975,
            0.05,
            psrname,
            transform=ax1.transAxes,
            size=18,
            c="lightgray",
            horizontalalignment="right",
            verticalalignment="bottom",
        )
    plt.tight_layout()
    ax1.legend(loc="best")
    if outfile:
        plt.savefig(outfile)

    return None


# Now we want to make wideband DM vs. time plot, this uses the premade dm_resids from PINT
def plot_dm_residuals(
    fitter,
    restype="postfit",
    plotsig=False,
    save=False,
    legend=True,
    title=True,
    axs=None,
    mean_sub=True,
    **kwargs,
):
    """
    Make a plot of Wideband timing DM residuals v. time.


    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    restype ['string'] : Type of residuals, pre or post fit, to plot from fitter object. Options are:
        'prefit' - plot the prefit residuals.
        'postfit' - plot the postfit residuals (default)
        'both' - overplot both the pre and post-fit residuals.
    plotsig [boolean] : If True plot number of measurements v. residuals/uncertainty, else v. residuals
        [default: False].
    save [boolean] : If True will save plot with the name "resid_v_mjd.png" Will add averaged/whitened
         as necessary [default: False].
    legend [boolean] : If False, will not print legend with plot [default: True].
    title [boolean] : If False, will not print plot title [default: True].
    axs [string] : If not None, should be defined subplot value and the figure will be used as part of a
         larger figure [default: None].
    mean_sub [boolean] : If False, will not mean subtract the DM residuals to be centered on zero [default: True]

    Optional Arguments:
    --------------------
    dmres [list/array] : List or array of DM residual values to plot. Will override values from fitter object.
    errs [list/array] : List or array of DM residual error values to plot. Will override values from fitter object.
    mjds [list/array] : List or array of DM residual mjds to plot. Will override values from fitter object.
    rcvr_bcknds[list/array] : List or array of TOA receiver-backend combinations. Will override values from toa object.
    figsize [tuple] : Size of the figure passed to matplotlib [default: (10,4)].
    fmt ['string'] : matplotlib format option for markers [default: ('x')]
    color ['string'] : matplotlib color option for plot [default: color dictionary in plot_utils.py file]
    alpha [float] : matplotlib alpha options for plot points [default: 0.5]
    """
    markers, colorscheme = plot_settings()

    # Check if wideband
    if not fitter.is_wideband:
        raise RuntimeError(
            "Error: Narrowband TOAs have no DM residuals, use `plot_dmx_time() instead."
        )

    # Get the DM residuals
    if "dmres" in kwargs.keys():
        dm_resids = kwargs["dmres"]
    else:
        if restype == "postfit":
            dm_resids = fitter.resids.residual_objs["dm"].resids.value
        elif restype == "prefit":
            dm_resids = fitter.resids_init.residual_objs["dm"].resids.value
        elif restype == "both":
            dm_resids = fitter.resids.residual_objs["dm"].resids.value
            dm_resids_init = fitter.resids_init.residual_objs["dm"].resids.value

    # Get the DM residual errors
    if "errs" in kwargs.keys():
        dm_error = kwargs["errs"]
    else:
        if restype == "postfit":
            dm_error = fitter.resids.residual_objs["dm"].get_data_error().value
        elif restype == "prefit":
            dm_error = fitter.resids_init.residual_objs["dm"].get_data_error().value
        elif restype == "both":
            dm_error = fitter.resids.residual_objs["dm"].get_data_error().value
            dm_error_init = (
                fitter.resids_init.residual_objs["dm"].get_data_error().value
            )

    # Get the MJDs
    if "mjds" in kwargs.keys():
        mjds = kwargs["mjds"]
    else:
        mjds = fitter.toas.get_mjds().value
    years = (mjds - 51544.0) / 365.25 + 2000.0

    # Get the receiver-backend combos
    if "rcvr_bcknds" in kwargs.keys():
        rcvr_bcknds = kwargs["rcvr_bcknds"]
    else:
        rcvr_bcknds = np.array(fitter.toas.get_flag_value("f")[0])
    # Get the set of unique receiver-bandend combos
    RCVR_BCKNDS = set(rcvr_bcknds)

    # If we don't want mean subtraced data we add the mean
    if not mean_sub:
        if "dmres" in kwargs.keys():
            dm_avg = dm_resids
        else:
            dm_avg = fitter.resids.residual_objs["dm"].dm_data
        if "errs" in kwargs.keys():
            dm_avg_err = dm_error
        else:
            dm_avg_err = fitter.resids.residual_objs["dm"].get_data_error().value
        DM0 = np.average(dm_avg, weights=(dm_avg_err) ** -2)
        dm_resids += DM0.value
        if restype == "both":
            dm_resids_init += DM0.value
        if plotsig:
            ylabel = r"DM/Uncertainty"
        else:
            ylabel = r"DM [cm$^{-3}$ pc]"
    else:
        if plotsig:
            ylabel = r"$\Delta$DM/Uncertainty"
        else:
            ylabel = r"$\Delta$DM [cm$^{-3}$ pc]"

    if axs == None:
        if "figsize" in kwargs.keys():
            figsize = kwargs["figsize"]
        else:
            figsize = (10, 4)
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
    else:
        ax1 = axs
    for i, r_b in enumerate(RCVR_BCKNDS):
        inds = np.where(rcvr_bcknds == r_b)[0]
        if not inds.tolist():
            r_b_label = ""
        else:
            r_b_label = rcvr_bcknds[inds][0]
        # Get plot preferences
        if "fmt" in kwargs.keys():
            mkr = kwargs["fmt"]
        else:
            mkr = markers[r_b_label]
            if restype == "both":
                mkr_pre = "."
        if "color" in kwargs.keys():
            clr = kwargs["color"]
        else:
            clr = colorscheme[r_b_label]
        if "alpha" in kwargs.keys():
            alpha = kwargs["alpha"]
        else:
            alpha = 0.5

        # Do plotting command
        if restype == "both":
            if plotsig:
                dm_sig = dm_resids[inds] / dm_error[inds]
                dm_sig_pre = dm_resids_init[inds] / dm_error[inds]
                ax1.errorbar(
                    years[inds],
                    dm_sig,
                    yerr=len(dm_error[inds]) * [1],
                    fmt=markers[r_b_label],
                    color=colorscheme[r_b_label],
                    label=r_b_label,
                    alpha=0.5,
                )
                ax1.errorbar(
                    years[inds],
                    dm_sig_pre,
                    yerr=len(dm_error_init[inds]) * [1],
                    fmt=markers[r_b_label],
                    color=colorscheme[r_b_label],
                    label=r_b_label + " Prefit",
                    alpha=0.5,
                )
            else:
                ax1.errorbar(
                    years[inds],
                    dm_resids[inds],
                    yerr=dm_error[inds],
                    fmt=markers[r_b_label],
                    color=colorscheme[r_b_label],
                    label=r_b_label,
                    alpha=0.5,
                )
                ax1.errorbar(
                    years[inds],
                    dm_resids_init[inds],
                    yerr=dm_error_init[inds],
                    fmt=markers[r_b_label],
                    color=colorscheme[r_b_label],
                    label=r_b_label + " Prefit",
                    alpha=0.5,
                )
        else:
            if plotsig:
                dm_sig = dm_resids[inds] / dm_error[inds]
                ax1.errorbar(
                    years[inds],
                    dm_sig,
                    yerr=len(dm_error[inds]) * [1],
                    fmt=markers[r_b_label],
                    color=colorscheme[r_b_label],
                    label=r_b_label,
                    alpha=0.5,
                )
            else:
                ax1.errorbar(
                    years[inds],
                    dm_resids[inds],
                    yerr=dm_error[inds],
                    fmt=markers[r_b_label],
                    color=colorscheme[r_b_label],
                    label=r_b_label,
                    alpha=0.5,
                )

    # Set second axis
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel(r"Year")
    ax1.grid(True)
    ax2 = ax1.twiny()
    mjd0 = ((ax1.get_xlim()[0]) - 2004.0) * 365.25 + 53005.0
    mjd1 = ((ax1.get_xlim()[1]) - 2004.0) * 365.25 + 53005.0
    ax2.set_xlim(mjd0, mjd1)

    if legend:
        if len(RCVR_BCKNDS) > 5:
            ncol = int(np.ceil(len(RCVR_BCKNDS) / 2))
            y_offset = 1.15
        else:
            ncol = len(RCVR_BCKNDS)
            y_offset = 1.0
        ax1.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, y_offset + 1.0 / figsize[1]),
            ncol=ncol,
        )
    if title:
        if len(RCVR_BCKNDS) > 5:
            y_offset = 1.1
        else:
            y_offset = 1.0
        plt.title(
            "%s %s DM residuals" % (fitter.model.PSR.value, restype),
            y=y_offset + 1.0 / figsize[1],
        )
    if axs == None:
        plt.tight_layout()
    if save:
        ext = ""
        if restype == "postfit":
            ext += "_postfit"
        elif restype == "prefit":
            ext += "_prefit"
        elif restype == "both":
            ext += "_prefit_and_postfit"
        plt.savefig("%s_dm_resids_v_time%s.png" % (fitter.model.PSR.value, ext))

    if axs == None:
        # Define clickable points
        text = ax2.text(0, 0, "")

        # Define point highlight color
        if "430_ASP" in RCVR_BCKNDS or "430_PUPPI" in RCVR_BCKNDS:
            stamp_color = "#61C853"
        else:
            stamp_color = "#FD9927"

        def onclick(event):
            # Get X and Y axis data
            xdata = mjds
            if plotsig:
                ydata = dm_resids / dm_error
            else:
                ydata = dm_resids
            # Get x and y data from click
            xclick = event.xdata
            yclick = event.ydata
            # Calculate scaled distance, find closest point index
            d = np.sqrt(((xdata - xclick) / 1000.0) ** 2 + (ydata - yclick) ** 2)
            ind_close = np.where(np.min(d) == d)[0]
            # highlight clicked point
            ax2.scatter(xdata[ind_close], ydata[ind_close], marker="x", c=stamp_color)
            # Print point info
            text.set_position((xdata[ind_close], ydata[ind_close]))
            text.set_text(
                "DM Params:\n MJD: %s \n Res: %.6f \n Index: %s"
                % (xdata[ind_close][0], ydata[ind_close], ind_close[0])
            )

        fig.canvas.mpl_connect("button_press_event", onclick)

    return


def plot_measurements_v_res(
    fitter,
    restype="postfit",
    plotsig=False,
    nbin=50,
    avg=False,
    whitened=False,
    save=False,
    legend=True,
    title=True,
    axs=None,
    **kwargs,
):
    """
    Make a histogram of number of measurements v. residuals


    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    restype ['string'] : Type of residuals, pre or post fit, to plot from fitter object. Options are:
        'prefit' - plot the prefit residuals.
        'postfit' - plot the postfit residuals (default)
        'both' - overplot both the pre and post-fit residuals.
    plotsig [boolean] : If True plot number of measurements v. residuals/uncertainty, else v. residuals
        [default: False].
    nbin [int] : Number of bins to use in the histogram [default: 50]
    avg [boolean] : If True and not wideband fitter, will compute and plot epoch-average residuals [default: False].
    whitened [boolean] : If True will compute and plot whitened residuals [default: False].
    save [boolean] : If True will save plot with the name "resid_v_mjd.png" Will add averaged/whitened
         as necessary [default: False].
    legend [boolean] : If False, will not print legend with plot [default: True].
    title [boolean] : If False, will not print plot title [default: True].
    axs [string] : If not None, should be defined subplot value and the figure will be used as part of a
         larger figure [default: None].

    Optional Arguments:
    --------------------
    res [list/array] : List or array of residual values to plot. Will override values from fitter object.
    errs [list/array] : List or array of residual error values to plot. Will override values from fitter object.
    rcvr_bcknds[list/array] : List or array of TOA receiver-backend combinations. Will override values from toa object.
    figsize [tuple] : Size of the figure passed to matplotlib [default: (10,4)].
    fmt ['string'] : matplotlib format option for markers [default: ('x')]
    color ['string'] : matplotlib color option for plot [default: color dictionary in plot_utils.py file]
    alpha [float] : matplotlib alpha options for plot points [default: 0.5]
    """
    markers, colorscheme = plot_settings()

    # Check if wideband
    if fitter.is_wideband:
        NB = False
        if avg == True:
            raise ValueError(
                "Cannot epoch average wideband residuals, please change 'avg' to False."
            )
    else:
        NB = True

    # Check if want epoch averaged residuals
    if avg == True and restype == "prefit":
        avg_dict = fitter.resids_init.ecorr_average(use_noise_model=True)
    elif avg == True and restype == "postfit":
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
    elif avg == True and restype == "both":
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
        avg_dict_pre = fitter.resids_init.ecorr_average(use_noise_model=True)

    # Get residuals
    if "res" in kwargs.keys():
        res = kwargs["res"]
    else:
        if restype == "prefit":
            if NB == True:
                if avg == True:
                    res = avg_dict["time_resids"].to(u.us)
                else:
                    res = fitter.resids_init.time_resids.to(u.us)
            else:
                res = fitter.resids_init.residual_objs["toa"].time_resids.to(u.us)
        elif restype == "postfit":
            if NB == True:
                if avg == True:
                    res = avg_dict["time_resids"].to(u.us)
                else:
                    res = fitter.resids.time_resids.to(u.us)
            else:
                res = fitter.resids.residual_objs["toa"].time_resids.to(u.us)
        elif restype == "both":
            if NB == True:
                if avg == True:
                    res = avg_dict["time_resids"].to(u.us)
                    res_pre = avg_dict_pre["time_resids"].to(u.us)
                else:
                    res = fitter.resids.time_resids.to(u.us)
                    res_pre = fitter.resids_init.time_resids.to(u.us)
            else:
                res = fitter.resids.residual_objs["toa"].time_resids.to(u.us)
                res_pre = fitter.resids_init.residual_objs["toa"].time_resids.to(u.us)
        else:
            raise ValueError(
                "Unrecognized residual type: %s. Please choose from 'prefit', 'postfit', or 'both'."
                % (restype)
            )

    # Check if we want whitened residuals
    if whitened == True and ("res" not in kwargs.keys()):
        if avg == True:
            if restype != "both":
                res = whiten_resids(avg_dict, restype=restype)
            else:
                res = whiten_resids(avg_dict_pre, restype="prefit")
                res_pre = whiten_resids(avg_dict, restype="postfit")
                res_pre = res_pre.to(u.us)
            res = res.to(u.us)
        else:
            if restype != "both":
                res = whiten_resids(fitter, restype=restype)
            else:
                res = whiten_resids(fitter, restype="prefit")
                res_pre = whiten_resids(fitter, restype="postfit")
                res_pre = res_pre.to(u.us)
            res = res.to(u.us)

    # Get errors
    if "errs" in kwargs.keys():
        errs = kwargs["errs"]
    else:
        if restype == "prefit":
            if avg == True:
                errs = avg_dict["errors"].to(u.us)
            else:
                errs = fitter.toas.get_errors().to(u.us)
        elif restype == "postfit":
            if NB == True:
                if avg == True:
                    errs = avg_dict["errors"].to(u.us)
                else:
                    errs = fitter.resids.get_data_error().to(u.us)
            else:
                errs = fitter.resids.residual_objs["toa"].get_data_error().to(u.us)
        elif restype == "both":
            if NB == True:
                if avg == True:
                    errs = avg_dict["errors"].to(u.us)
                    errs_pre = avg_dict_pre["errors"].to(u.us)
                else:
                    errs = fitter.resids.get_data_error().to(u.us)
                    errs_pre = fitter.toas.get_errors().to(u.us)
            else:
                errs = fitter.resids.residual_objs["toa"].get_data_error().to(u.us)
                errs_pre = fitter.toas.get_errors().to(u.us)

    # Get receiver backends
    if "rcvr_bcknds" in kwargs.keys():
        rcvr_bcknds = kwargs["rcvr_bcknds"]
    else:
        rcvr_bcknds = np.array(fitter.toas.get_flag_value("f")[0])
        if avg == True:
            avg_rcvr_bcknds = []
            for iis in avg_dict["indices"]:
                avg_rcvr_bcknds.append(rcvr_bcknds[iis[0]])
            rcvr_bcknds = np.array(avg_rcvr_bcknds)
    # Get the set of unique receiver-bandend combos
    RCVR_BCKNDS = set(rcvr_bcknds)

    if axs == None:
        if "figsize" in kwargs.keys():
            figsize = kwargs["figsize"]
        else:
            figsize = (10, 4)
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
    else:
        ax1 = axs

    xmax = 0
    for i, r_b in enumerate(RCVR_BCKNDS):
        inds = np.where(rcvr_bcknds == r_b)[0]
        if not inds.tolist():
            r_b_label = ""
        else:
            r_b_label = rcvr_bcknds[inds][0]
        # Get plot preferences
        if "color" in kwargs.keys():
            clr = kwargs["color"]
        else:
            clr = colorscheme[r_b_label]
        if plotsig:
            sig = res[inds] / errs[inds]
            ax1.hist(
                sig,
                nbin,
                histtype="step",
                color=colorscheme[r_b_label],
                label=r_b_label,
            )
            xmax = max(xmax, max(sig), max(-sig))
            if restype == "both":
                sig_pre = res_pre[inds] / errs_pre[inds]
                ax1.hist(
                    sig_pre,
                    nbin,
                    histtype="step",
                    color=colorscheme[r_b_label],
                    linestyle="--",
                    label=r_b_label + " Prefit",
                )
        else:
            ax1.hist(
                res[inds],
                nbin,
                histtype="step",
                color=colorscheme[r_b_label],
                label=r_b_label,
            )
            xmax = max(xmax, max(res[inds]), max(-res[inds]))
            if restype == "both":
                ax1.hist(
                    res[inds],
                    nbin,
                    histtype="step",
                    color=colorscheme[r_b_label],
                    linestyle="--",
                    label=r_b_label + " Prefit",
                )

    ax1.grid(True)
    ax1.set_ylabel("Number of measurements")
    if plotsig:
        if avg and whitened:
            ax1.set_xlabel(
                "Average Residual/Uncertainty \n (Whitened)", multialignment="center"
            )
        elif avg and not whitened:
            ax1.set_xlabel("Average Residual/Uncertainty")
        elif whitened and not avg:
            ax1.set_xlabel(
                "Residual/Uncertainty \n (Whitened)", multialignment="center"
            )
        else:
            ax1.set_xlabel("Residual/Uncertainty")
    else:
        if avg and whitened:
            ax1.set_xlabel(
                "Average Residual ($\\mu$s) \n (Whitened)", multialignment="center"
            )
        elif avg and not whitened:
            ax1.set_xlabel(r"Average Residual ($\mu$s)")
        elif whitened and not avg:
            ax1.set_xlabel("Residual ($\\mu$s) \n (Whitened)", multialignment="center")
        else:
            ax1.set_xlabel(r"Residual ($\mu$s)")
    ax1.set_xlim(-1.1 * xmax, 1.1 * xmax)
    if legend:
        if len(RCVR_BCKNDS) > 5:
            ncol = int(np.ceil(len(RCVR_BCKNDS) / 2))
            y_offset = 1.15
        else:
            ncol = len(RCVR_BCKNDS)
            y_offset = 1.0
        ax1.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, y_offset + 1.0 / figsize[1]),
            ncol=ncol,
        )
    if title:
        if len(RCVR_BCKNDS) > 5:
            y_offset = 1.1
        else:
            y_offset = 1.0
        plt.title(
            "%s %s residual measurements" % (fitter.model.PSR.value, restype),
            y=y_offset + 1.0 / figsize[1],
        )
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
        if restype == "prefit":
            ext += "_prefit"
        elif restype == "postfit":
            ext += "_postfit"
        elif restype == "both":
            ext += "_pre_post_fit"
        plt.savefig("%s_resid_measurements%s.png" % (fitter.model.PSR.value, ext))

    return


def plot_measurements_v_dmres(
    fitter,
    restype="postfit",
    plotsig=False,
    nbin=50,
    save=False,
    legend=True,
    title=True,
    axs=None,
    mean_sub=True,
    **kwargs,
):
    """
    Make a histogram of number of measurements v. residuals


    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    restype ['string'] : Type of residuals, pre or post fit, to plot from fitter object. Options are:
        'prefit' - plot the prefit residuals.
        'postfit' - plot the postfit residuals (default)
        'both' - overplot both the pre and post-fit residuals.
    plotsig [boolean] : If True plot number of measurements v. residuals/uncertainty, else v. residuals
        [default: False].
    nbin [int] : Number of bins to use in the histogram [default: 50]
    save [boolean] : If True will save plot with the name "resid_v_mjd.png" Will add averaged/whitened
         as necessary [default: False].
    legend [boolean] : If False, will not print legend with plot [default: True].
    title [boolean] : If False, will not print plot title [default: True].
    axs [string] : If not None, should be defined subplot value and the figure will be used as part of a
         larger figure [default: None].
    mean_sub [boolean] : If False, will not mean subtract the DM residuals to be centered on zero [default: True]

    Optional Arguments:
    --------------------
    dmres [list/array] : List or array of residual values to plot. Will override values from fitter object.
    errs [list/array] : List or array of residual error values to plot. Will override values from fitter object.
    rcvr_bcknds[list/array] : List or array of TOA receiver-backend combinations. Will override values from toa object.
    figsize [tuple] : Size of the figure passed to matplotlib [default: (10,4)].
    fmt ['string'] : matplotlib format option for markers [default: ('x')]
    color ['string'] : matplotlib color option for plot [default: color dictionary in plot_utils.py file]
    """
    markers, colorscheme = plot_settings()

    # Check if wideband
    if not fitter.is_wideband:
        raise ValueError(
            "Narrowband Fitters have have no DM residuals, please use `plot_measurements_v_dmres` instead."
        )

    # Get the DM residuals
    if "dmres" in kwargs.keys():
        dm_resids = kwargs["dmres"]
    else:
        if restype == "postfit":
            dm_resids = fitter.resids.residual_objs["dm"].resids.value
        elif restype == "prefit":
            dm_resids = fitter.resids_init.residual_objs["dm"].resids.value
        elif restype == "both":
            dm_resids = fitter.resids.residual_objs["dm"].resids.value
            dm_resids_init = fitter.resids_init.residual_objs["dm"].resids.value

    # Get the DM residual errors
    if "errs" in kwargs.keys():
        dm_error = kwargs["errs"]
    else:
        if restype == "postfit":
            dm_error = fitter.resids.residual_objs["dm"].get_data_error().value
        elif restype == "prefit":
            dm_error = fitter.resids_init.residual_objs["dm"].get_data_error().value
        elif restype == "both":
            dm_error = fitter.resids.residual_objs["dm"].get_data_error().value
            dm_error_init = (
                fitter.resids_init.residual_objs["dm"].get_data_error().value
            )

    # Get the receiver-backend combos
    if "rcvr_bcknds" in kwargs.keys():
        rcvr_bcknds = kwargs["rcvr_bcknds"]
    else:
        rcvr_bcknds = np.array(fitter.toas.get_flag_value("f")[0])
    # Get the set of unique receiver-bandend combos
    RCVR_BCKNDS = set(rcvr_bcknds)

    # If we don't want mean subtraced data we add the mean
    if not mean_sub:
        if "dmres" in kwargs.keys():
            dm_avg = dm_resids
        else:
            dm_avg = fitter.resids.residual_objs["dm"].dm_data
        if "errs" in kwargs.keys():
            dm_avg_err = dm_error
        else:
            dm_avg_err = fitter.resids.residual_objs["dm"].get_data_error().value
        DM0 = np.average(dm_avg, weights=(dm_avg_err) ** -2)
        dm_resids += DM0.value
        if restype == "both":
            dm_resids_init += DM0.value
        if plotsig:
            xlabel = r"DM/Uncertainty"
        else:
            xlabel = r"DM [cm$^{-3}$ pc]"
    else:
        if plotsig:
            xlabel = r"$\Delta$DM/Uncertainty"
        else:
            xlabel = r"$\Delta$DM [cm$^{-3}$ pc]"

    if axs == None:
        if "figsize" in kwargs.keys():
            figsize = kwargs["figsize"]
        else:
            figsize = (10, 4)
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
    else:
        ax1 = axs
    for i, r_b in enumerate(RCVR_BCKNDS):
        inds = np.where(rcvr_bcknds == r_b)[0]
        if not inds.tolist():
            r_b_label = ""
        else:
            r_b_label = rcvr_bcknds[inds][0]
        # Get plot preferences
        if "color" in kwargs.keys():
            clr = kwargs["color"]
        else:
            clr = colorscheme[r_b_label]

        if plotsig:
            sig = dm_resids[inds] / dm_error[inds]
            ax1.hist(
                sig,
                nbin,
                histtype="step",
                color=colorscheme[r_b_label],
                label=r_b_label,
            )
            if restype == "both":
                sig_pre = dm_resids_init[inds] / dm_error_init[inds]
                ax1.hist(
                    sig_pre,
                    nbin,
                    histtype="step",
                    color=colorscheme[r_b_label],
                    linestyle="--",
                    label=r_b_label + " Prefit",
                )
        else:
            ax1.hist(
                dm_resids[inds],
                nbin,
                histtype="step",
                color=colorscheme[r_b_label],
                label=r_b_label,
            )
            if restype == "both":
                ax1.hist(
                    dm_resids_init[inds],
                    nbin,
                    histtype="step",
                    color=colorscheme[r_b_label],
                    linestyle="--",
                    label=r_b_label + " Prefit",
                )

    ax1.grid(True)
    ax1.set_ylabel("Number of measurements")
    ax1.set_xlabel(xlabel)
    if legend:
        if len(RCVR_BCKNDS) > 5:
            ncol = int(np.ceil(len(RCVR_BCKNDS) / 2))
            y_offset = 1.15
        else:
            ncol = len(RCVR_BCKNDS)
            y_offset = 1.0
        ax1.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, y_offset + 1.0 / figsize[1]),
            ncol=ncol,
        )
    if title:
        if len(RCVR_BCKNDS) > 5:
            y_offset = 1.1
        else:
            y_offset = 1.0
        plt.title(
            "%s %s DM residual measurements" % (fitter.model.PSR.value, restype),
            y=y_offset + 1.0 / figsize[1],
        )
    if axs == None:
        plt.tight_layout()
    if save:
        ext = ""
        if restype == "prefit":
            ext += "_prefit"
        elif restype == "postfit":
            ext += "_postfit"
        elif restype == "both":
            ext += "_pre_post_fit"
        plt.savefig("%s_DM_resid_measurements%s.png" % (fitter.model.PSR.value, ext))

    return


def plot_residuals_orb(
    fitter,
    restype="postfit",
    colorby="f",
    plotsig=False,
    avg=False,
    mixed_ecorr=False,
    whitened=False,
    save=False,
    legend=True,
    title=True,
    axs=None,
    **kwargs,
):
    """
    Make a plot of the residuals vs. orbital phase.


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

    Optional Arguments:
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
    if fitter.is_wideband:
        NB = False
        if avg == True:
            raise ValueError(
                "Cannot epoch average wideband residuals, please change 'avg' to False."
            )
    else:
        NB = True

    # Check if want epoch averaged residuals
    if avg == True and restype == "prefit" and mixed_ecorr == True:
        avg_dict = fitter.resids_init.ecorr_average(use_noise_model=True)
        no_avg_dict = no_ecorr_average(
            fitter.toas, fitter.resids_init, use_noise_model=True
        )
    elif avg == True and restype == "postfit" and mixed_ecorr == True:
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
        no_avg_dict = no_ecorr_average(fitter.toas, fitter.resids, use_noise_model=True)
    elif avg == True and restype == "both" and mixed_ecorr == True:
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
        no_avg_dict = no_ecorr_average(fitter.toas, fitter.resids, use_noise_model=True)
        avg_dict_pre = fitter.resids_init.ecorr_average(use_noise_model=True)
        no_avg_dict_pre = no_ecorr_average(
            fitter.toas, fitter.resids_init, use_noise_model=True
        )
    elif avg == True and restype == "prefit" and mixed_ecorr == False:
        avg_dict = fitter.resids_init.ecorr_average(use_noise_model=True)
    elif avg == True and restype == "postfit" and mixed_ecorr == False:
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
    elif avg == True and restype == "both" and mixed_ecorr == False:
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
        avg_dict_pre = fitter.resids_init.ecorr_average(use_noise_model=True)

    # Get residuals
    if "res" in kwargs.keys():
        res = kwargs["res"]
    else:
        if restype == "prefit":
            if NB == True:
                if avg == True and mixed_ecorr == True:
                    res = avg_dict["time_resids"].to(u.us)
                    res_no_avg = no_avg_dict["time_resids"].to(u.us)
                elif avg == True and mixed_ecorr == False:
                    res = avg_dict["time_resids"].to(u.us)
                else:
                    res = fitter.resids_init.time_resids.to(u.us)
            else:
                res = fitter.resids_init.residual_objs["toa"].time_resids.to(u.us)
        elif restype == "postfit":
            if NB == True:
                if avg == True and mixed_ecorr == True:
                    res = avg_dict["time_resids"].to(u.us)
                    res_no_avg = no_avg_dict["time_resids"].to(u.us)
                elif avg == True:
                    res = avg_dict["time_resids"].to(u.us)
                else:
                    res = fitter.resids.time_resids.to(u.us)
            else:
                res = fitter.resids.residual_objs["toa"].time_resids.to(u.us)
        elif restype == "both":
            if NB == True:
                if avg == True and mixed_ecorr == True:
                    res = avg_dict["time_resids"].to(u.us)
                    res_no_avg = no_avg_dict["time_resids"].to(u.us)
                    res_pre = avg_dict_pre["time_resids"].to(u.us)
                    res_pre_no_avg = no_avg_dict_pre["time_resids"].to(u.us)
                elif avg == True and mixed_ecorr == False:
                    res = avg_dict["time_resids"].to(u.us)
                    res_pre = avg_dict_pre["time_resids"].to(u.us)
                else:
                    res = fitter.resids.time_resids.to(u.us)
                    res_pre = fitter.resids_init.time_resids.to(u.us)
            else:
                res = fitter.resids.residual_objs["toa"].time_resids.to(u.us)
                res_pre = fitter.resids_init.residual_objs["toa"].time_resids.to(u.us)
        else:
            raise ValueError(
                "Unrecognized residual type: %s. Please choose from 'prefit', 'postfit', or 'both'."
                % (restype)
            )

    # Check if we want whitened residuals
    if whitened == True and ("res" not in kwargs.keys()):
        if avg == True and mixed_ecorr == True:
            if restype != "both":
                res = whiten_resids(avg_dict, restype=restype)
                res_no_avg = whiten_resids(no_avg_dict, restype=restype)
            else:
                res = whiten_resids(avg_dict_pre, restype="prefit")
                res_pre = whiten_resids(avg_dict, restype="postfit")
                res_pre = res_pre.to(u.us)
                res_no_avg = whiten_resids(avg_dict_pre, restype="prefit")
                res_pre_no_avg = whiten_resids(avg_dict, restype="postfit")
                res_pre_no_avg = res_pre_no_avg.to(u.us)
            res = res.to(u.us)
            res_no_avg = res_no_avg.to(u.us)
        elif avg == True and mixed_ecorr == False:
            if restype != "both":
                res = whiten_resids(avg_dict, restype=restype)
            else:
                res = whiten_resids(avg_dict_pre, restype="prefit")
                res_pre = whiten_resids(avg_dict, restype="postfit")
                res_pre = res_pre.to(u.us)
            res = res.to(u.us)
        else:
            if restype != "both":
                res = whiten_resids(fitter, restype=restype)
            else:
                res = whiten_resids(fitter, restype="prefit")
                res_pre = whiten_resids(fitter, restype="postfit")
                res_pre = res_pre.to(u.us)
            res = res.to(u.us)

    # Get errors
    if "errs" in kwargs.keys():
        errs = kwargs["errs"]
    else:
        if restype == "prefit":
            if avg == True and mixed_ecorr == True:
                errs = avg_dict["errors"].to(u.us)
                errs_no_avg = no_avg_dict["errors"].to(u.us)
            elif avg == True and mixed_ecorr == False:
                errs = avg_dict["errors"].to(u.us)
            else:
                errs = fitter.toas.get_errors().to(u.us)
        elif restype == "postfit":
            if NB == True:
                if avg == True and mixed_ecorr == True:
                    errs = avg_dict["errors"].to(u.us)
                    errs_no_avg = no_avg_dict["errors"].to(u.us)
                elif avg == True and mixed_ecorr == False:
                    errs = avg_dict["errors"].to(u.us)
                else:
                    errs = fitter.resids.get_data_error().to(u.us)
            else:
                errs = fitter.resids.residual_objs["toa"].get_data_error().to(u.us)
        elif restype == "both":
            if NB == True:
                if avg == True and mixed_ecorr == True:
                    errs = avg_dict["errors"].to(u.us)
                    errs_pre = avg_dict_pre["errors"].to(u.us)
                    errs_no_avg = no_avg_dict["errors"].to(u.us)
                    errs_no_avg_pre = no_avg_dict_pre["errors"].to(u.us)
                elif avg == True and mixed_ecorr == False:
                    errs = avg_dict["errors"].to(u.us)
                    errs_pre = avg_dict_pre["errors"].to(u.us)
                else:
                    errs = fitter.resids.get_data_error().to(u.us)
                    errs_pre = fitter.toas.get_errors().to(u.us)
            else:
                errs = fitter.resids.residual_objs["toa"].get_data_error().to(u.us)
                errs_pre = fitter.toas.get_errors().to(u.us)

    # Get MJDs
    if "orbphase" not in kwargs.keys():
        mjds = fitter.toas.get_mjds().value
        if avg == True:
            mjds = avg_dict["mjds"].value
            if mixed_ecorr == True:
                mjds_no_avg = no_avg_dict["mjds"].value

    # Now we need to the orbital phases; start with binary model name
    if "orbphase" in kwargs.keys():
        orbphase = kwargs["orbphase"]
    else:
        orbphase = fitter.model.orbital_phase(mjds, radians=False)
        if avg and mixed_ecorr:
            no_avg_orbphase = fitter.model.orbital_phase(mjds_no_avg, radians=False)

    # In the end, we'll want to plot both ecorr avg & not ecorr avg at the same time if we have mixed ecorr.
    # Create combined arrays

    if avg == True and mixed_ecorr == True:
        combo_res = np.hstack((res, res_no_avg))
        combo_errs = np.hstack((errs, errs_no_avg))
        combo_orbphase = np.hstack((orbphase, no_avg_orbphase))
        if restype == "both":
            combo_errs_pre = np.hstack((errs_pre, errs_no_avg_pre))
            combo_res_pre = np.hstack((res_pre, res_no_avg_pre))

    # Get colorby flag values (obs, PTA, febe, etc.)
    if "colorby" in kwargs.keys():
        cb = kwargs["colorby"]
    else:
        cb = np.array(fitter.toas[colorby])
        # .      Seems to run a little faster but not robust to obs
        #        cb = np.array(fitter.toas.get_flag_value(colorby)[0])
        if avg == True:
            avg_cb = []
            for iis in avg_dict["indices"]:
                avg_cb.append(cb[iis[0]])
            if mixed_ecorr == True:
                no_avg_cb = []
                for jjs in no_avg_dict["indices"]:
                    no_avg_cb.append(cb[jjs])
                no_ecorr_cb = np.array(no_avg_cb)

            cb = np.array(avg_cb)

    # Get the set of unique flag values
    if avg == True and mixed_ecorr == True:
        cb = np.hstack((cb, no_ecorr_cb))
    CB = set(cb)

    colorscheme, markerscheme = set_color_and_marker(colorby)

    if "figsize" in kwargs.keys():
        figsize = kwargs["figsize"]
    else:
        figsize = (10, 4)
    if axs == None:
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
    else:
        fig = plt.gcf()
        ax1 = axs
    for i, c in enumerate(CB):
        inds = np.where(cb == c)[0]
        if not inds.tolist():
            cb_label = ""
        else:
            cb_label = cb[inds][0]
        # Get plot preferences
        if "fmt" in kwargs.keys():
            mkr = kwargs["fmt"]
        else:
            try:
                mkr = markerscheme[cb_label]
                if restype == "both":
                    mkr_pre = "."
            except Exception:
                mkr = "x"
                log.log(1, "Color by flag value doesn't have a marker label!!")
        if "color" in kwargs.keys():
            clr = kwargs["color"]
        else:
            try:
                clr = colorscheme[cb_label]
            except Exception:
                clr = "k"
                log.log(1, "Color by flag value doesn't have a color!!")
        if "alpha" in kwargs.keys():
            alpha = kwargs["alpha"]
        else:
            alpha = 0.5
        if avg and mixed_ecorr:
            if plotsig:
                combo_sig = combo_res[inds] / combo_errs[inds]
                ax1.errorbar(
                    combo_orbphase[inds],
                    combo_sig,
                    yerr=len(combo_errs[inds]) * [1],
                    fmt=mkr,
                    color=clr,
                    label=cb_label,
                    alpha=alpha,
                )
                if restype == "both":
                    combo_sig_pre = combo_res_pre[inds] / combo_errs_pre[inds]
                    ax1.errorbar(
                        combo_orbphase[inds],
                        combo_sig_pre,
                        yerr=len(combo_errs_pre[inds]) * [1],
                        fmt=mkr_pre,
                        color=clr,
                        label=cb_label + " Prefit",
                        alpha=alpha,
                    )
            else:
                ax1.errorbar(
                    combo_orbphase[inds],
                    combo_res[inds],
                    yerr=combo_errs[inds],
                    fmt=mkr,
                    color=clr,
                    label=cb_label,
                    alpha=alpha,
                )
                if restype == "both":
                    ax1.errorbar(
                        combo_orbphase[inds],
                        combo_res_pre[inds],
                        yerr=combo_errs_pre[inds],
                        fmt=mkr_pre,
                        color=clr,
                        label=cb_label + " Prefit",
                        alpha=alpha,
                    )
        else:
            if plotsig:
                sig = res[inds] / errs[inds]
                ax1.errorbar(
                    orbphase[inds],
                    sig,
                    yerr=len(errs[inds]) * [1],
                    fmt=mkr,
                    color=clr,
                    label=cb_label,
                    alpha=alpha,
                )
                if restype == "both":
                    sig_pre = res_pre[inds] / errs_pre[inds]
                    ax1.errorbar(
                        orbphase[inds],
                        sig_pre,
                        yerr=len(errs_pre[inds]) * [1],
                        fmt=mkr_pre,
                        color=clr,
                        label=cb_label + " Prefit",
                        alpha=alpha,
                    )
            else:
                ax1.errorbar(
                    orbphase[inds],
                    res[inds],
                    yerr=errs[inds],
                    fmt=mkr,
                    color=clr,
                    label=cb_label,
                    alpha=alpha,
                )
                if restype == "both":
                    ax1.errorbar(
                        orbphase[inds],
                        res_pre[inds],
                        yerr=errs_pre[inds],
                        fmt=mkr_pre,
                        color=clr,
                        label=cb_label + " Prefit",
                        alpha=alpha,
                    )
    # Set second axis
    ax1.set_xlabel(r"Orbital Phase")
    ax1.grid(True)
    if plotsig:
        if avg and whitened:
            ax1.set_ylabel(
                "Average Residual/Uncertainty \n (Whitened)", multialignment="center"
            )
        elif avg and not whitened:
            ax1.set_ylabel("Average Residual/Uncertainty")
        elif whitened and not avg:
            ax1.set_ylabel(
                "Residual/Uncertainty \n (Whitened)", multialignment="center"
            )
        else:
            ax1.set_ylabel("Residual/Uncertainty")
    else:
        if avg and whitened:
            ax1.set_ylabel(
                "Average Residual ($\\mu$s) \n (Whitened)", multialignment="center"
            )
        elif avg and not whitened:
            ax1.set_ylabel(r"Average Residual ($\mu$s)")
        elif whitened and not avg:
            ax1.set_ylabel("Residual ($\\mu$s) \n (Whitened)", multialignment="center")
        else:
            ax1.set_ylabel(r"Residual ($\mu$s)")
    if legend:
        if len(CB) > 5:
            ncol = int(np.ceil(len(CB) / 2))
            y_offset = 1.15
        else:
            ncol = len(CB)
            y_offset = 1.0
        ax1.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, y_offset + 1.0 / figsize[1]),
            ncol=ncol,
        )
    if title:
        if len(CB) > 5:
            y_offset = 1.1
        else:
            y_offset = 1.0
        plt.title(
            "%s %s timing residuals" % (fitter.model.PSR.value, restype),
            y=y_offset + 1.0 / figsize[1],
        )
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
        if restype == "prefit":
            ext += "_prefit"
        elif restype == "postfit":
            ext += "_postfit"
        elif restype == "both":
            ext += "_pre_post_fit"
        plt.savefig("%s_resid_v_orbphase%s.png" % (fitter.model.PSR.value, ext))

    if axs == None:
        # Define clickable points
        text = ax1.text(0, 0, "")
        stamp_color = "#FD9927"
        # Define color for highlighting points
        # if "430_ASP" in RCVR_BCKNDS or "430_PUPPI" in RCVR_BCKNDS:
        #    stamp_color = "#61C853"
        # else:
        #    stamp_color = "#FD9927"

        def onclick(event):
            # Get X and Y axis data
            xdata = orbphase
            if plotsig:
                ydata = (res / errs).decompose().value
            else:
                ydata = res.value
            # Get x and y data from click
            xclick = event.xdata
            yclick = event.ydata
            # Calculate scaled distance, find closest point index
            d = np.sqrt((xdata - xclick) ** 2 + ((ydata - yclick) / 100.0) ** 2)
            ind_close = np.where(np.min(d) == d)[0]
            # highlight clicked point
            ax1.scatter(xdata[ind_close], ydata[ind_close], marker="x", c=stamp_color)
            # Print point info
            text.set_position((xdata[ind_close], ydata[ind_close]))
            if plotsig:
                text.set_text(
                    "TOA Params:\n Phase: %.5f \n Res/Err: %.2f \n Index: %s"
                    % (xdata[ind_close][0], ydata[ind_close], ind_close[0])
                )
            else:
                text.set_text(
                    "TOA Params:\n Phase: %.5f \n Res: %.2f \n Index: %s"
                    % (xdata[ind_close][0], ydata[ind_close], ind_close[0])
                )

        fig.canvas.mpl_connect("button_press_event", onclick)

    return

def plot_residuals_serial(fitter, restype = 'postfit', colorby='pta', plotsig = False, avg = False, whitened = False, \
                        save = False, legend = True, title = True, axs = None, mixed_ecorr=False, epoch_lines=False, \
                        milli=False, mjd_order=False, **kwargs):
    """
    Make a serial plot of the residuals. (TOAs are evenly spaced along the x-axis)
    In the default setup this will emulate pintk/tempo2 and TOAs appear in the order they are listed in the par file
    If mjd_order=True, TOAs are first sorted according to MJD.


    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    restype ['string'] : Type of residuals, pre or post fit, to plot from fitter object. Options are:
        'prefit' - plot the prefit residuals.
        'postfit' - plot the postfit residuals (default)
        'both' - overplot both the pre and post-fit residuals.
    colorby ['string']: What to use to determine color/markers
        'pta' - color residuals by PTA (default)
        'obs' - color residuals by telescope 
        'f'   - color residuals by frontend/backend pair (flag not used by all PTAs).
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
        

    Optional Arguments:
    --------------------
    res [list/array] : List or array of residual values to plot. Will override values from fitter object.
    errs [list/array] : List or array of residual error values to plot. Will override values from fitter object.
    mjds [list/array] : List or array of TOA MJDs to plot. Will override values from toa object.
    obs[list/array] : List or array of TOA observatories combinations. Will override values from toa object.
    figsize [tuple] : Size of the figure passed to matplotlib [default: (10,4)].
    fmt ['string'] : matplotlib format option for markers [default: ('x')]
    color ['string'] : matplotlib color option for plot [default: color dictionary in plot_utils.py file]
    alpha [float] : matplotlib alpha options for plot points [default: 0.5]
    mixed_ecorr [boolean]: If True, allows avging with mixed ecorr/no ecorr TOAs.
    epoch_lines [boolean]: If True, plot a vertical line at the first TOA of each observation file.
    milli [boolean]: If True, plot y-axis in milliseconds rather than microseconds.
    mjd_order [boolean]: If True, the TOAs are sorted by MJD before being plotted.
    """
    # Check if wideband
    if fitter.is_wideband:
        NB = False
        if avg == True:
            raise ValueError("Cannot epoch average wideband residuals, please change 'avg' to False.")
    else:
        NB = True
        
    # Check if want epoch averaged residuals
    if avg == True and restype == 'prefit' and mixed_ecorr == True:
        avg_dict = fitter.resids_init.ecorr_average(use_noise_model=True)
        no_avg_dict = no_ecorr_average(fitter.toas, fitter.resids_init,use_noise_model=True)
    elif avg == True and restype == 'postfit' and mixed_ecorr == True:
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
        no_avg_dict = no_ecorr_average(fitter.toas, fitter.resids,use_noise_model=True)
    elif avg == True and restype == 'both' and mixed_ecorr == True:
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
        no_avg_dict = no_ecorr_average(fitter.toas, fitter.resids,use_noise_model=True)
        avg_dict_pre = fitter.resids_init.ecorr_average(use_noise_model=True)
        no_avg_dict_pre = no_ecorr_average(fitter.toas, fitter.resids_init,use_noise_model=True)
    elif avg == True and restype == 'prefit' and mixed_ecorr == False:
        avg_dict = fitter.resids_init.ecorr_average(use_noise_model=True)
    elif avg == True and restype == 'postfit' and mixed_ecorr==False:
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
    elif avg == True and restype == 'both' and mixed_ecorr == False:
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
        avg_dict_pre = fitter.resids_init.ecorr_average(use_noise_model=True)


    # adjust to millisec
    if milli:
        unit = u.ms
        unitstr = "ms"
    else:
        unit = u.us
        unitstr = r"$\mu$s"

    # Get residuals
    if 'res' in kwargs.keys():
        res = kwargs['res']
    else:
        if restype == 'prefit':
            if NB == True:
                if avg == True and mixed_ecorr == True:
                    res = avg_dict['time_resids'].to(unit)
                    res_no_avg = no_avg_dict['time_resids'].to(unit)
                elif avg==True and mixed_ecorr == False:
                    res = avg_dict['time_resids'].to(unit)
                else:
                    res = fitter.resids_init.time_resids.to(unit)
            else:
                res = fitter.resids_init.residual_objs['toa'].time_resids.to(unit)
        elif restype == 'postfit':
            if NB == True:
                if avg == True and mixed_ecorr == True:
                    res = avg_dict['time_resids'].to(unit)
                    res_no_avg = no_avg_dict['time_resids'].to(unit)
                elif avg == True:
                    res = avg_dict['time_resids'].to(unit)
                else:
                    res = fitter.resids.time_resids.to(unit)
            else:
                res = fitter.resids.residual_objs['toa'].time_resids.to(unit)
        elif restype == 'both':
            if NB == True:
                if avg == True and mixed_ecorr == True:
                    res = avg_dict['time_resids'].to(unit)
                    res_no_avg = no_avg_dict['time_resids'].to(unit)
                    res_pre = avg_dict_pre['time_resids'].to(unit)
                    res_pre_no_avg = no_avg_dict_pre['time_resids'].to(unit)
                elif avg == True and mixed_ecorr == False:
                    res = avg_dict['time_resids'].to(unit)
                    res_pre = avg_dict_pre['time_resids'].to(unit)
                else:
                    res = fitter.resids.time_resids.to(unit)
                    res_pre = fitter.resids_init.time_resids.to(unit)
            else:
                res = fitter.resids.residual_objs['toa'].time_resids.to(unit)
                res_pre = fitter.resids_init.residual_objs['toa'].time_resids.to(unit)
        else:
            raise ValueError("Unrecognized residual type: %s. Please choose from 'prefit', 'postfit', or 'both'."\
                             %(restype))

    # Check if we want whitened residuals
    if whitened == True and ('res' not in kwargs.keys()):
        if avg == True and mixed_ecorr == True: 
            if restype != 'both':
                res = whiten_resids(avg_dict, restype=restype)
                res_no_avg = whiten_resids(no_avg_dict, restype=restype)
            else:
                res = whiten_resids(avg_dict_pre, restype='prefit')
                res_pre = whiten_resids(avg_dict, restype='postfit')
                res_pre = res_pre.to(unit)
                res_no_avg = whiten_resids(avg_dict_pre, restype='prefit')
                res_pre_no_avg = whiten_resids(avg_dict, restype='postfit')
                res_pre_no_avg = res_pre_no_avg.to(unit)
            res = res.to(unit)
            res_no_avg = res_no_avg.to(unit)
        elif avg == True and mixed_ecorr == False: 
            if restype != 'both':
                res = whiten_resids(avg_dict, restype=restype)
            else:
                res = whiten_resids(avg_dict_pre, restype='prefit')
                res_pre = whiten_resids(avg_dict, restype='postfit')
                res_pre = res_pre.to(unit)
            res = res.to(unit)        
        else:
            if restype != 'both':
                res = whiten_resids(fitter, restype=restype)
            else:
                res = whiten_resids(fitter, restype='prefit')
                res_pre = whiten_resids(fitter, restype='postfit')
                res_pre = res_pre.to(unit)
            res = res.to(unit)

    # Get errors
    if 'errs' in kwargs.keys():
        errs = kwargs['errs']
    else:
        if restype == 'prefit':
            if avg == True and mixed_ecorr == True:
                errs = avg_dict['errors'].to(unit)
                errs_no_avg = no_avg_dict['errors'].to(unit)
            elif avg == True and mixed_ecorr == False:
                errs = avg_dict['errors'].to(unit)
            else:
                errs = fitter.toas.get_errors().to(unit)
        elif restype == 'postfit':
            if NB == True:
                if avg == True and mixed_ecorr == True:
                    errs = avg_dict['errors'].to(unit)
                    errs_no_avg = no_avg_dict['errors'].to(unit)
                elif avg == True and mixed_ecorr == False:
                    errs = avg_dict['errors'].to(unit) 
                else:
                    errs = fitter.resids.get_data_error().to(unit)
            else:
                errs = fitter.resids.residual_objs['toa'].get_data_error().to(unit)
        elif restype == 'both':
            if NB == True:
                if avg == True and mixed_ecorr == True:
                    errs = avg_dict['errors'].to(unit)
                    errs_pre = avg_dict_pre['errors'].to(unit)
                    errs_no_avg = no_avg_dict['errors'].to(unit)
                    errs_no_avg_pre = no_avg_dict_pre['errors'].to(unit)
                elif avg == True and mixed_ecorr == False:
                    errs = avg_dict['errors'].to(unit)
                    errs_pre = avg_dict_pre['errors'].to(unit)
                else:
                    errs = fitter.resids.get_data_error().to(unit)
                    errs_pre = fitter.toas.get_errors().to(unit)
            else:
                errs = fitter.resids.residual_objs['toa'].get_data_error().to(unit)
                errs_pre = fitter.toas.get_errors().to(unit)
    # Get MJDs
    if 'mjds' in kwargs.keys():
        mjds = kwargs['mjds']
    else:
        mjds = fitter.toas.get_mjds().value
        if avg == True and mixed_ecorr == True :
            mjds = avg_dict['mjds'].value
            mjds_no_avg = no_avg_dict['mjds'].value
            years_no_avg = (mjds_no_avg - 51544.0)/365.25 + 2000.0

        elif avg == True and mixed_ecorr == False:
            mjds = avg_dict['mjds'].value
    # Convert to years
    years = (mjds - 51544.0)/365.25 + 2000.0
    
    # In the end, we'll want to plot both ecorr avg & not ecorr avg at the same time if we have mixed ecorr.
    # Create combined arrays
            
    if avg == True and mixed_ecorr == True:    
        combo_res = np.hstack((res, res_no_avg))
        combo_errs = np.hstack((errs, errs_no_avg))
        combo_years = np.hstack((years, years_no_avg))
        if restype =='both':
            combo_errs_pre = np.hstack((errs_pre, errs_no_avg_pre))     
            combo_res_pre = np.hstack((res_pre, res_no_avg_pre))
  
    # Get colorby flag values (obs, PTA, febe, etc.)
    if 'colorby' in kwargs.keys():
        cb = kwargs['colorby']
    else:
        cb = np.array(fitter.toas[colorby])
#.      Seems to run a little faster but not robust to obs?
#        cb = np.array(fitter.toas.get_flag_value(colorby)[0])
        if avg == True:
            avg_cb = []
            for iis in avg_dict['indices']:
                avg_cb.append(cb[iis[0]])
            if mixed_ecorr == True:
                no_avg_cb = []
                for jjs in no_avg_dict['indices']:
                    no_avg_cb.append(cb[jjs])
                no_ecorr_cb = np.array(no_avg_cb)
                
            cb = np.array(avg_cb)
            
    # Get the set of unique flag values
    if avg==True and mixed_ecorr==True:
        cb = np.hstack((cb,no_ecorr_cb))
    
    CB = set(cb)

    
    if colorby== 'pta':
        colorscheme = colorschemes['pta']
    elif colorby == 'obs':
        colorscheme = colorschemes['observatories']
    elif colorby == 'f':
        colorscheme = colorschemes['febe']
  

    if 'figsize' in kwargs.keys():
        figsize = kwargs['figsize']
    else:
        figsize = (10,5)
    if axs == None:
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111)
    else:
        fig = plt.gcf()
        ax1 = axs

    # if want tempo2 version, where serial = order in tim file
    if not mjd_order:
        x = range(len(inds))
    # if want serial = order in mjd
    else:
        x = np.argsort(mjds) 

    # plot vertical line at the first TOA of each observation file
    if epoch_lines and not avg:
        names = fitter.toas["name"]
        for nm in np.unique(names):
            inds_name = np.where(names==nm)[0]
            x_nm = x[inds_name]
            ax1.axvline(min(x_nm), c="k", alpha=0.1)

    for i, c in enumerate(CB):
        inds = np.where(cb==c)[0]
        if not inds.tolist():
            cb_label = ""
        else:
            cb_label = cb[inds][0]
        # Get plot preferences
        if 'fmt' in kwargs.keys():
            mkr = kwargs['fmt']
        else:
            try:
                mkr = markers[cb_label]
                if restype == 'both':
                    mkr_pre = '.'
            except Exception:
                mkr = 'x'
                log.log(1, "Color by Flag doesn't have a marker label!!")
        if 'color' in kwargs.keys():
            clr = kwargs['color']
        else:
            try:
                clr = colorscheme[cb_label]
            except Exception:
                clr = 'k'
                log.log(1, "Color by Flag doesn't have a color!!")
        if 'alpha' in kwargs.keys():
            alpha = kwargs['alpha']
        else:
            alpha = 0.5

        if 'label' in kwargs.keys():
            label = kwargs['label']
        else:
            label = cb_label

        x_subsec = x[inds]
        if avg == True and mixed_ecorr == True:
            if plotsig:
                combo_sig = combo_res[inds]/combo_errs[inds]
                ax1.errorbar(x_subsec, combo_sig, yerr=len(combo_errs[inds])*[1], fmt=mkr, \
                         color=clr, label=label, alpha = alpha, picker=True)
                if restype == 'both':
                    combo_sig_pre = combo_res_pre[inds]/combo_errs_pre[inds]
                    ax1.errorbar(x_subsec, combo_sig_pre, yerr=len(combo_errs_pre[inds])*[1], fmt=mkr_pre, \
                             color=clr, label=label+" Prefit", alpha = alpha, picker=True)
            else:
                ax1.errorbar(x_subsec, combo_res[inds], yerr=combo_errs[inds], fmt=mkr, \
                         color=clr, label=label, alpha = alpha, picker=True)
                if restype == 'both':
                    ax1.errorbar(x_subsec, combo_res_pre[inds], yerr=combo_errs_pre[inds], fmt=mkr_pre, \
                         color=clr, label=label+" Prefit", alpha = alpha, picker=True)
                  
        else:
            if plotsig:
                sig = res[inds]/errs[inds]
                ax1.errorbar(x_subsec, sig, yerr=len(errs[inds])*[1], fmt=mkr, \
                         color=clr, label=label, alpha = alpha, picker=True)
                if restype == 'both':
                    sig_pre = res_pre[inds]/errs_pre[inds]
                    ax1.errorbar(x_subsec, sig_pre, yerr=len(errs_pre[inds])*[1], fmt=mkr_pre, \
                             color=clr, label=label+" Prefit", alpha = alpha, picker=True)
            else:
                ax1.errorbar(x_subsec, res[inds], yerr=errs[inds], fmt=mkr, \
                         color=clr, label=label, alpha = alpha, picker=True)
                if restype == 'both':
                    ax1.errorbar(x_subsec, res_pre[inds], yerr=errs_pre[inds], fmt=mkr_pre, \
                         color=clr, label=label+" Prefit", alpha = alpha, picker=True)

    # Set second axis
    ax1.set_xlabel(r'TOA Number')

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
            ax1.set_ylabel(f'Average Residual ({unitstr}) \n (Whitened)', multialignment='center')
        elif avg and not whitened:
            ax1.set_ylabel(f'Average Residual ({unitstr})')
        elif whitened and not avg:
            ax1.set_ylabel(f'Residual ({unitstr}) \n (Whitened)', multialignment='center')
        else:
            ax1.set_ylabel(f'Residual ({unitstr})')
    if legend:
        if len(CB) > 5:
            ncol = int(np.ceil(len(CB)/2))
            y_offset = 1.15
        else:
            ncol = len(CB)
            y_offset = 1.0
        ax1.legend(loc='upper center', bbox_to_anchor= (0.5, y_offset+1.0/figsize[1]), ncol=ncol)
    if title:
        if len(CB) > 5:
            y_offset = 1.1
        else:
            y_offset = 1.0
        if isinstance(title, str):
            title_str = title
        else:
            title_str = "%s %s timing residuals" % (fitter.model.PSR.value, restype)
        plt.title(title_str, y=y_offset+1.0/figsize[1])
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
    
    if axs == None:
        # Define clickable points
        text = ax1.text(0,0,"")

        # Define point highlight color
        stamp_color = "#FD9927"

        def onclick(event):
            # Get X and Y axis data
            xdata = x
            if plotsig:
                ydata = (res/errs).decompose().value
            else:
                ydata = res.value
            # Get x and y data from click
            xclick = event.xdata
            yclick = event.ydata
            # Calculate scaled distance, find closest point index
            d = np.sqrt(((xdata - xclick)/10.0)**2 + (ydata - yclick)**2)
            ind_close = np.where(np.min(d) == d)[0]
            # highlight clicked point
            ax1.scatter(xdata[ind_close], ydata[ind_close], marker = 'x', c = stamp_color)
            # Print point info
            text.set_position((xdata[ind_close], ydata[ind_close]))
            if plotsig:
                text.set_text("TOA Params:\n MJD: %s \n Res/Err: %.2f \n Index: %s" % (xdata[ind_close][0], ydata[ind_close], ind_close[0]))
            else:
                text.set_text("TOA Params:\n MJD: %s \n Res: %.2f \n Index: %s" % (xdata[ind_close][0], ydata[ind_close], ind_close[0]))

        fig.canvas.mpl_connect('button_press_event', onclick)

    return

def plot_fd_res_v_freq(
    fitter,
    plotsig=False,
    comp_FD=True,
    avg=False,
    whitened=False,
    save=False,
    legend=True,
    title=True,
    axs=None,
    **kwargs,
):
    """
    Make a plot of the residuals vs. frequency, can do WB as well. Note, if WB fitter, comp_FD may not work.
    If comp_FD is True, the panels are organized as follows:
    Top: Residuals with FD parameters subtracted and the FD curve plotted as a dashed line.
    Middle: Best fit residuals with no FD parameters.
    Bottom: Residuals with FD correction included.
    Note - This function may take a while to run if there are many TOAs.

    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    plotsig [boolean] : If True plot number of measurements v. residuals/uncertainty, else v. residuals
        [default: False].
    comp_FD [boolean]: If True, will plot the residuals v. frequency with FD included, FD subtracted, and best fit
        without FD.
    avg [boolean] : If True and not wideband fitter, will compute and plot epoch-average residuals [default: False].
    whitened [boolean] : If True will compute and plot whitened residuals [default: False].
    save [boolean] : If True will save plot with the name "resid_v_mjd.png" Will add averaged/whitened
         as necessary [default: False].
    legend [boolean] : If False, will not print legend with plot [default: True].
    title [boolean] : If False, will not print plot title [default: True].
    axs [string] : If not None, should be defined subplot value and the figure will be used as part of a
         larger figure [default: None].

    Optional Arguments:
    --------------------
    res [list/array] : List or array of residual values to plot. Will override values from fitter object.
    errs [list/array] : List or array of residual error values to plot. Will override values from fitter object.
    freqs [list/array] : List or array of TOA frequencies to plot. Will override values from toa object.
    rcvr_bcknds[list/array] : List or array of TOA receiver-backend combinations. Will override values from toa object.
    figsize [tuple] : Size of the figure passed to matplotlib [default: (10,4)].
    fmt ['string'] : matplotlib format option for markers [default: ('x')]
    color ['string'] : matplotlib color option for plot [default: color dictionary in plot_utils.py file]
    alpha [float] : matplotlib alpha options for plot points [default: 0.5]
    """
    # Check if fitter is wideband or not
    if fitter.is_wideband:
        NB = False
        if avg == True:
            raise ValueError(
                "Cannot epoch average wideband residuals, please change 'avg' to False."
            )
    else:
        NB = True

    # Check if want epoch averaged residuals
    if avg:
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)

    # Get residuals
    if "res" in kwargs.keys():
        res = kwargs["res"]
    else:
        if NB == True:
            if avg == True:
                res = avg_dict["time_resids"].to(u.us)
            else:
                res = fitter.resids.time_resids.to(u.us)
        else:
            res = fitter.resids.residual_objs["toa"].time_resids.to(u.us)

    # Check if we want whitened residuals
    if whitened == True and ("res" not in kwargs.keys()):
        if avg == True:
            res = whiten_resids(avg_dict)
            res = res.to(u.us)
        else:
            res = whiten_resids(fitter)
            res = res.to(u.us)

    # Get errors
    if "errs" in kwargs.keys():
        errs = kwargs["errs"]
    else:
        if NB == True:
            if avg == True:
                errs = avg_dict["errors"].to(u.us)
            else:
                errs = fitter.resids.get_data_error().to(u.us)
        else:
            errs = fitter.resids.residual_objs["toa"].get_data_error().to(u.us)

    # Get receiver backends
    if "rcvr_bcknds" in kwargs.keys():
        rcvr_bcknds = kwargs["rcvr_bcknds"]
    else:
        rcvr_bcknds = np.array(fitter.toas.get_flag_value("f")[0])
        if avg == True:
            avg_rcvr_bcknds = []
            for iis in avg_dict["indices"]:
                avg_rcvr_bcknds.append(rcvr_bcknds[iis[0]])
            rcvr_bcknds = np.array(avg_rcvr_bcknds)
    # Get the set of unique receiver-bandend combos
    RCVR_BCKNDS = set(rcvr_bcknds)

    # get frequencies
    if "freqs" in kwargs.keys():
        freqs = kwargs["freqs"]
    else:
        if avg == True:
            freqs = avg_dict["freqs"].value
        else:
            freqs = fitter.toas.get_freqs().value

    # Check if comparing the FD parameters
    if comp_FD:
        if axs != None:
            log.warning("Cannot do full comparison with three panels")
            axs = None
        if "figsize" in kwargs.keys():
            figsize = kwargs["figsize"]
        else:
            figsize = (4, 12)
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(313)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(311)
    else:
        if "figsize" in kwargs.keys():
            figsize = kwargs["figsize"]
        else:
            figsize = (4, 4)
        if axs == None:
            fig = plt.figure(figsize=figsize)
            ax1 = fig.add_subplot(111)
        else:
            ax1 = axs

    # Make the plot of residual vs. frequency
    for i, r_b in enumerate(RCVR_BCKNDS):
        inds = np.where(rcvr_bcknds == r_b)[0]
        if not inds.tolist():
            r_b_label = ""
        else:
            r_b_label = rcvr_bcknds[inds][0]
        # Get plot preferences
        if "fmt" in kwargs.keys():
            mkr = kwargs["fmt"]
        else:
            mkr = markers[r_b_label]
        if "color" in kwargs.keys():
            clr = kwargs["color"]
        else:
            clr = colorscheme[r_b_label]
        if "alpha" in kwargs.keys():
            alpha = kwargs["alpha"]
        else:
            alpha = 1.0
        if plotsig:
            sig = res[inds] / errs[inds]
            ax1.errorbar(
                freqs[inds],
                sig,
                yerr=len(errs[inds]) * [1],
                fmt=mkr,
                color=clr,
                label=r_b_label,
                alpha=alpha,
            )
        else:
            ax1.errorbar(
                freqs[inds],
                res[inds],
                yerr=errs[inds],
                fmt=mkr,
                color=clr,
                label=r_b_label,
                alpha=alpha,
            )
        # assign axis labels
        ax1.set_xlabel(r"Frequency (MHz)")
        ax1.grid(True)
        if plotsig:
            if avg and whitened:
                ylabel = "Average Residual/Uncertainty \n (Whitened)"
            elif avg and not whitened:
                ylabel = "Average Residual/Uncertainty"
            elif whitened and not avg:
                ylabel = "Residual/Uncertainty \n (Whitened)"
            else:
                ylabel = "Residual/Uncertainty"
        else:
            if avg and whitened:
                ylabel = "Average Residual ($\\mu$s) \n (Whitened)"
            elif avg and not whitened:
                ylabel = "Average Residual ($\\mu$s)"
            elif whitened and not avg:
                ylabel = "Residual ($\\mu$s) \n (Whitened)"
            else:
                ylabel = "Residual ($\\mu$s)"
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
            fd_val = getattr(fitter.model, fd).value * 10**6  # convert to microseconds
            FD_offsets += fd_val * np.log(freqs / 1000.0) ** (i + 1)
            FD_line += fd_val * np.log(sorted_freqs / 1000.0) ** (i + 1)
        # Now edit residuals
        fd_cor_res = res.value + FD_offsets

        # Now we need to redo the fit without the FD parameters
        psr_fitter_nofd = copy.deepcopy(fitter)
        try:
            psr_fitter_nofd.model.remove_component("FD")
        except:
            log.warning("No FD parameters in the initial timing model...")

        # Check if fitter is wideband or not
        if psr_fitter_nofd.is_wideband:
            resids = psr_fitter_nofd.resids.residual_objs["toa"]
        else:
            resids = psr_fitter_nofd.resids

        psr_fitter_nofd.fit_toas(1)
        # Now we need to figure out if these need to be whitened and/or averaged
        if avg:
            avg = psr_fitter_nofd.resids.ecorr_average(use_noise_model=True)
            avg_rcvr_bcknds = rcvr_bcknds
            if whitened:
                # need to whiten and average
                wres_avg = whiten_resids(avg)
                res_nofd = wres_avg.to(u.us).value
            else:
                # need to average
                res_nofd = avg["time_resids"].to(u.us).value
        elif whitened:
            # Need to whiten
            wres_nofd = whiten_resids(psr_fitter_nofd)
            res_nofd = wres_nofd.to(u.us).value
        else:
            res_nofd = resids.time_resids.to(u.us).value

        # Now plot
        for i, r_b in enumerate(RCVR_BCKNDS):
            inds = np.where(rcvr_bcknds == r_b)[0]
            if not inds.tolist():
                r_b_label = ""
            else:
                r_b_label = rcvr_bcknds[inds][0]
            # Get plot preferences
            if "fmt" in kwargs.keys():
                mkr = kwargs["fmt"]
            else:
                mkr = markers[r_b_label]
            if "color" in kwargs.keys():
                clr = kwargs["color"]
            else:
                clr = colorscheme[r_b_label]
            if "alpha" in kwargs.keys():
                alpha = kwargs["alpha"]
            else:
                alpha = 1.0
            if plotsig:
                sig = fd_cor_res[inds] / errs[inds]
                ax3.errorbar(
                    freqs[inds],
                    sig.value,
                    yerr=len(errs[inds]) * [1],
                    fmt=mkr,
                    color=clr,
                    label=r_b_label,
                    alpha=alpha,
                )

                sig_nofd = res_nofd[inds] / errs[inds].value
                ax2.errorbar(
                    freqs[inds],
                    sig_nofd,
                    yerr=len(errs[inds]) * [1],
                    fmt=mkr,
                    color=clr,
                    label=r_b_label,
                    alpha=alpha,
                )
            else:
                ax3.errorbar(
                    freqs[inds],
                    fd_cor_res[inds],
                    yerr=errs[inds].value,
                    fmt=mkr,
                    color=clr,
                    label=r_b_label,
                    alpha=alpha,
                )

                ax2.errorbar(
                    freqs[inds],
                    res_nofd[inds],
                    yerr=errs[inds].value,
                    fmt=mkr,
                    color=clr,
                    label=r_b_label,
                    alpha=alpha,
                )

            ax3.plot(sorted_freqs, FD_line, c="k", ls="--")
            # assign axis labels
            ax3.set_xlabel(r"Frequency (MHz)")
            ax3.set_ylabel(ylabel)
            ax3.grid(True)
            ax2.set_xlabel(r"Frequency (MHz)")
            ax2.set_ylabel(ylabel)
            ax2.grid(True)

    if legend:
        if comp_FD:
            ax3.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0 + 1.0 / figsize[1]),
                ncol=int(len(RCVR_BCKNDS) / 2),
            )
        else:
            ax1.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0 + 1.0 / figsize[1]),
                ncol=int(len(RCVR_BCKNDS) / 2),
            )
    if title:
        plt.title(
            "%s FD Paramter Check" % (fitter.model.PSR.value), y=1.0 + 1.0 / figsize[1]
        )
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
        plt.savefig("%s_FD_resid_v_freq%s.png" % (fitter.model.PSR.value, ext))
    return


"""
We also offer some options for convenience plotting functions, one that will show all possible summary plots, and
another that will show just the summary plots that are typically created in finalize_timing.py in that order.
"""


def summary_plots(
    fitter, title=None, legends=False, save=False, avg=True, whitened=True
):
    """
    Function to make a composite set of summary plots for sets of TOAs.
    NOTE - This is noe the same set of plots as will be in the pdf writer

    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    title [boolean] : If True, will add titles to ALL plots [default: False].
    legend [boolean] : If True, will add legends to ALL plots [default: False].
    save [boolean] : If True will save plot with the name "psrname_summary.png" Will add averaged/whitened
         as necessary [default: False].
    avg [boolean] : If True and not wideband fitter, will make plots of epoch averaged residuals [default: True].
    whitened [boolean] : If True will make plots of whitened residuals [default: True].
    """

    if fitter.is_wideband:
        if avg == True:
            raise ValueError(
                "Cannot epoch average wideband residuals, please change 'avg' to False."
            )
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
    if not hasattr(fitter.model, "binary_model_name"):
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

    fig = plt.figure(figsize=(12, figlength))  # not sure what we'll need for a fig size
    if title != None:
        plt.title(title, y=1.015, size=16)
    gs = fig.add_gridspec(gs_rows, 2)

    count = 0
    k = 0
    # First plot is all residuals vs. time.
    ax0 = fig.add_subplot(gs[count, :])
    plot_residuals_time(fitter, title=False, axs=ax0, figsize=(12, 3))
    k += 1

    # Plot the residuals divided by uncertainty vs. time
    ax1 = fig.add_subplot(gs[count + k, :])
    plot_residuals_time(
        fitter, title=False, legend=False, plotsig=True, axs=ax1, figsize=(12, 3)
    )
    k += 1

    # Second plot is residual v. orbital phase
    if hasattr(fitter.model, "binary_model_name"):
        ax2 = fig.add_subplot(gs[count + k, :])
        plot_residuals_orb(fitter, title=False, legend=False, axs=ax2, figsize=(12, 3))
        k += 1

    # Now add the measurement vs. uncertainty
    ax3_0 = fig.add_subplot(gs[count + k, 0])
    ax3_1 = fig.add_subplot(gs[count + k, 1])
    plot_measurements_v_res(
        fitter,
        nbin=50,
        plotsig=False,
        title=False,
        legend=False,
        axs=ax3_0,
        figsize=(6, 3),
    )
    plot_measurements_v_res(
        fitter,
        nbin=50,
        plotsig=True,
        title=False,
        legend=False,
        axs=ax3_1,
        figsize=(6, 3),
    )
    k += 1

    # and the DMX vs. time
    ax4 = fig.add_subplot(gs[count + k, :])
    plot_dmx_time(
        fitter,
        savedmx="dmxparse.out",
        legend=False,
        title=False,
        axs=ax4,
        figsize=(12, 3),
    )
    k += 1

    # And residual vs. Frequency
    ax5 = fig.add_subplot(gs[count + k, :])
    plot_residuals_freq(fitter, title=False, legend=False, axs=ax5, figsize=(12, 3))
    k += 1

    # Now if whitened add the whitened residual plots
    if whitened:
        ax6 = fig.add_subplot(gs[count + k, :])
        plot_residuals_time(
            fitter, title=False, whitened=True, axs=ax6, figsize=(12, 3)
        )
        k += 1

        # Plot the residuals divided by uncertainty vs. time
        ax7 = fig.add_subplot(gs[count + k, :])
        plot_residuals_time(
            fitter,
            title=False,
            legend=False,
            plotsig=True,
            whitened=True,
            axs=ax7,
            figsize=(12, 3),
        )
        k += 1

        # Second plot is residual v. orbital phase
        if hasattr(fitter.model, "binary_model_name"):
            ax8 = fig.add_subplot(gs[count + k, :])
            plot_residuals_orb(
                fitter,
                title=False,
                legend=False,
                whitened=True,
                axs=ax8,
                figsize=(12, 3),
            )
            k += 1

        # Now add the measurement vs. uncertainty
        ax9_0 = fig.add_subplot(gs[count + k, 0])
        ax9_1 = fig.add_subplot(gs[count + k, 1])
        plot_measurements_v_res(
            fitter,
            nbin=50,
            plotsig=False,
            title=False,
            legend=False,
            whitened=True,
            axs=ax9_0,
            figsize=(6, 3),
        )
        plot_measurements_v_res(
            fitter,
            nbin=50,
            plotsig=True,
            title=False,
            legend=False,
            whitened=True,
            axs=ax9_1,
            figsize=(6, 3),
        )
        k += 1

    # Now plot the average residuals
    if avg:
        ax10 = fig.add_subplot(gs[count + k, :])
        plot_residuals_time(fitter, title=False, avg=True, axs=ax10, figsize=(12, 3))
        k += 1

        # Plot the residuals divided by uncertainty vs. time
        ax11 = fig.add_subplot(gs[count + k, :])
        plot_residuals_time(
            fitter,
            title=False,
            legend=False,
            plotsig=True,
            avg=True,
            axs=ax11,
            figsize=(12, 3),
        )
        k += 1

        # Second plot is residual v. orbital phase
        if hasattr(fitter.model, "binary_model_name"):
            ax12 = fig.add_subplot(gs[count + k, :])
            plot_residuals_orb(
                fitter, title=False, legend=False, avg=True, axs=ax12, figsize=(12, 3)
            )
            k += 1

        # Now add the measurement vs. uncertainty
        ax13_0 = fig.add_subplot(gs[count + k, 0])
        ax13_1 = fig.add_subplot(gs[count + k, 1])
        plot_measurements_v_res(
            fitter,
            nbin=50,
            plotsig=False,
            title=False,
            legend=False,
            avg=True,
            axs=ax13_0,
            figsize=(6, 3),
        )
        plot_measurements_v_res(
            fitter,
            nbin=50,
            plotsig=True,
            title=False,
            legend=False,
            avg=True,
            axs=ax13_1,
            figsize=(6, 3),
        )
        k += 1

    # Now plot the whitened average residuals
    if avg and whitened:
        ax14 = fig.add_subplot(gs[count + k, :])
        plot_residuals_time(fitter, avg=True, whitened=True, axs=ax14, figsize=(12, 3))
        k += 1

        # Plot the residuals divided by uncertainty vs. time
        ax15 = fig.add_subplot(gs[count + k, :])
        plot_residuals_time(
            fitter,
            title=False,
            legend=False,
            plotsig=True,
            avg=True,
            whitened=True,
            axs=ax15,
            figsize=(12, 3),
        )
        k += 1

        # Second plot is residual v. orbital phase
        if hasattr(fitter.model, "binary_model_name"):
            ax16 = fig.add_subplot(gs[count + k, :])
            plot_residuals_orb(
                fitter,
                title=False,
                legend=False,
                avg=True,
                whitened=True,
                axs=ax16,
                figsize=(12, 3),
            )
            k += 1

        # Now add the measurement vs. uncertainty
        ax17_0 = fig.add_subplot(gs[count + k, 0])
        ax17_1 = fig.add_subplot(gs[count + k, 1])
        plot_measurements_v_res(
            fitter,
            nbin=50,
            plotsig=False,
            title=False,
            legend=False,
            avg=True,
            whitened=True,
            axs=ax17_0,
            figsize=(6, 3),
        )
        plot_measurements_v_res(
            fitter,
            nbin=50,
            plotsig=True,
            title=False,
            legend=False,
            avg=True,
            whitened=True,
            axs=ax17_1,
            figsize=(6, 3),
        )
        k += 1

    plt.tight_layout()
    if save:
        plt.savefig("%s_summary_plots.png" % (fitter.model.PSR.value))

    return


"""We also define a function to output the summary plots exactly as is done in finalize_timing.py (for now)"""


def summary_plots_ft(fitter, title=None, legends=False, save=False):
    """
    Function to make a composite set of summary plots for sets of TOAs
    NOTE - This is note the same set of plots as will be in the pdf writer

    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    title [boolean] : If True, will add titles to ALL plots [default: False].
    legend [boolean] : If True, will add legends to ALL plots [default: False].
    save [boolean] : If True will save plot with the name "psrname_summary.png" Will add averaged/whitened
         as necessary [default: False].
    """
    # Define the figure
    # Determine how long the figure size needs to be
    figlength = 18 * 3
    gs_rows = 13
    if not hasattr(fitter.model, "binary_model_name"):
        figlength -= 9
        gs_rows -= 3
    if fitter.is_wideband:
        figlength -= 9
        gs_rows -= 3

    fig = plt.figure(figsize=(12, figlength))  # not sure what we'll need for a fig size
    if title != None:
        plt.title(title, y=1.015, size=16)
    gs = fig.add_gridspec(gs_rows, 2)

    count = 0
    k = 0
    # First plot is all residuals vs. time.
    ax0 = fig.add_subplot(gs[count, :])
    plot_residuals_time(fitter, title=False, axs=ax0, figsize=(12, 3))
    k += 1

    # Then the epoch averaged residuals v. time
    if not fitter.is_wideband:
        ax10 = fig.add_subplot(gs[count + k, :])
        plot_residuals_time(
            fitter, title=False, legend=False, avg=True, axs=ax10, figsize=(12, 3)
        )
        k += 1

    # Epoch averaged vs. orbital phase
    if hasattr(fitter.model, "binary_model_name"):
        if not fitter.is_wideband:
            ax12 = fig.add_subplot(gs[count + k, :])
            plot_residuals_orb(
                fitter, title=False, legend=False, avg=True, axs=ax12, figsize=(12, 3)
            )
            k += 1
        else:
            ax12 = fig.add_subplot(gs[count + k, :])
            plot_residuals_orb(
                fitter, title=False, legend=False, axs=ax12, figsize=(12, 3)
            )
            k += 1

    # And DMX vs. time
    ax4 = fig.add_subplot(gs[count + k, :])
    plot_dmx_time(
        fitter,
        savedmx="dmxparse.out",
        legend=False,
        title=False,
        axs=ax4,
        figsize=(12, 3),
    )
    k += 1

    # Whitened residuals v. time
    ax6 = fig.add_subplot(gs[count + k, :])
    plot_residuals_time(fitter, whitened=True, axs=ax6, figsize=(12, 3))
    k += 1

    # Whitened epoch averaged residuals v. time
    if not fitter.is_wideband:
        ax15 = fig.add_subplot(gs[count + k, :])
        plot_residuals_time(
            fitter,
            title=False,
            legend=False,
            plotsig=False,
            avg=True,
            whitened=True,
            axs=ax15,
            figsize=(12, 3),
        )
        k += 1

    # Whitened epoch averaged residuals v. orbital phase
    if hasattr(fitter.model, "binary_model_name"):
        if not fitter.is_wideband:
            ax16 = fig.add_subplot(gs[count + k, :])
            plot_residuals_orb(
                fitter,
                title=False,
                legend=False,
                avg=True,
                whitened=True,
                axs=ax16,
                figsize=(12, 3),
            )
            k += 1
        else:
            ax16 = fig.add_subplot(gs[count + k, :])
            plot_residuals_orb(
                fitter,
                title=False,
                legend=False,
                avg=False,
                whitened=True,
                axs=ax16,
                figsize=(12, 3),
            )
            k += 1

    # Now add the measurement vs. uncertainty for both all reaiduals and epoch averaged
    ax3_0 = fig.add_subplot(gs[count + k, 0])
    ax3_1 = fig.add_subplot(gs[count + k, 1])
    plot_measurements_v_res(
        fitter,
        nbin=50,
        title=False,
        legend=False,
        plotsig=False,
        whitened=True,
        axs=ax3_0,
        figsize=(6, 3),
    )
    if not fitter.is_wideband:
        plot_measurements_v_res(
            fitter,
            nbin=50,
            title=False,
            legend=False,
            avg=True,
            whitened=True,
            axs=ax3_1,
            figsize=(6, 3),
        )
        k += 1
    else:
        plot_measurements_v_res(
            fitter,
            nbin=50,
            title=False,
            legend=False,
            avg=False,
            whitened=False,
            axs=ax3_1,
            figsize=(6, 3),
        )
        k += 1

    # Whitened residual/uncertainty v. time
    ax26 = fig.add_subplot(gs[count + k, :])
    plot_residuals_time(
        fitter,
        plotsig=True,
        title=False,
        legend=False,
        whitened=True,
        axs=ax26,
        figsize=(12, 3),
    )
    k += 1

    # Epoch averaged Whitened residual/uncertainty v. time
    if not fitter.is_wideband:
        ax25 = fig.add_subplot(gs[count + k, :])
        plot_residuals_time(
            fitter,
            title=False,
            legend=False,
            plotsig=True,
            avg=True,
            whitened=True,
            axs=ax25,
            figsize=(12, 3),
        )
        k += 1

    # Epoch averaged Whitened residual/uncertainty v. orbital phase
    if hasattr(fitter.model, "binary_model_name"):
        if not fitter.is_wideband:
            ax36 = fig.add_subplot(gs[count + k, :])
            plot_residuals_orb(
                fitter,
                title=False,
                legend=False,
                plotsig=True,
                avg=True,
                whitened=True,
                axs=ax36,
                figsize=(12, 3),
            )
            k += 1
        else:
            ax36 = fig.add_subplot(gs[count + k, :])
            plot_residuals_orb(
                fitter,
                title=False,
                legend=False,
                plotsig=True,
                avg=False,
                whitened=True,
                axs=ax36,
                figsize=(12, 3),
            )
            k += 1

    # Now add the measurement vs. uncertainty for both all reaiduals/uncertainty and epoch averaged/uncertainty
    ax17_0 = fig.add_subplot(gs[count + k, 0])
    ax17_1 = fig.add_subplot(gs[count + k, 1])
    plot_measurements_v_res(
        fitter,
        nbin=50,
        plotsig=True,
        title=False,
        legend=False,
        whitened=True,
        axs=ax17_0,
        figsize=(6, 3),
    )
    if not fitter.is_wideband:
        plot_measurements_v_res(
            fitter,
            nbin=50,
            title=False,
            plotsig=True,
            legend=False,
            avg=True,
            whitened=True,
            axs=ax17_1,
            figsize=(6, 3),
        )
        k += 1
    else:
        plot_measurements_v_res(
            fitter,
            nbin=50,
            title=False,
            plotsig=True,
            legend=False,
            avg=False,
            whitened=False,
            axs=ax17_1,
            figsize=(6, 3),
        )
        k += 1

    # Now plot the frequencies of the TOAs vs. time
    ax5 = fig.add_subplot(gs[count + k, :])
    plot_residuals_freq(fitter, title=False, legend=False, axs=ax5, figsize=(12, 3))
    k += 1

    plt.tight_layout()
    if save:
        plt.savefig("%s_summary_plots_FT.png" % (fitter.model.PSR.value))

    return


# JUST THE PLOTS FOR THE PDF WRITERS LEFT
def plots_for_summary_pdf_nb(fitter, title=None, legends=False):
    """
    Function to make a composite set of summary plots for sets of TOAs to be put into a summary pdf.
    This is for Narrowband timing only. For Wideband timing, use `plots_for_summary_pdf_wb`.
    By definition, this function will save all plots as "psrname"_summary_plot_#.nb.png, where # is
    and integer from 1-4.

    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    title [boolean] : If True, will add titles to ALL plots [default: False].
    legend [boolean] : If True, will add legends to ALL plots [default: False].
    """

    if fitter.is_wideband:
        raise ValueError(
            "Cannot use this function with WidebandTOAFitter, please use `plots_for_summary_pdf_wb` instead."
        )
    # Need to make four sets of plots
    for ii in range(4):
        if ii != 3:
            fig = plt.figure(figsize=(8, 10.0), dpi=100)
        else:
            fig = plt.figure(figsize=(8, 5), dpi=100)
        if title != None:
            plt.title(title, y=1.08, size=14)
        if ii == 0:
            gs = fig.add_gridspec(nrows=4, ncols=1)

            ax0 = fig.add_subplot(gs[0, :])
            ax1 = fig.add_subplot(gs[1, :])
            ax2 = fig.add_subplot(gs[2, :])
            ax3 = fig.add_subplot(gs[3, :])
            # Plot residuals v. time
            plot_residuals_time(fitter, title=False, axs=ax0, figsize=(8, 2.5))
            # Plot averaged residuals v. time
            if "ecorr_noise" in fitter.model.get_components_by_category().keys():
                plot_residuals_time(
                    fitter,
                    avg=True,
                    axs=ax1,
                    title=False,
                    legend=False,
                    figsize=(8, 2.5),
                )
            else:
                log.warning(
                    "ECORR not in model, cannot generate epoch averaged residuals. Plots will show all residuals."
                )
                plot_residuals_time(
                    fitter,
                    avg=False,
                    axs=ax1,
                    title=False,
                    legend=False,
                    figsize=(8, 2.5),
                )
            # Plot residuals v orbital phase
            if hasattr(fitter.model, "binary_model_name"):
                if "ecorr_noise" in fitter.model.get_components_by_category().keys():
                    plot_residuals_orb(
                        fitter,
                        title=False,
                        legend=False,
                        avg=True,
                        axs=ax2,
                        figsize=(8, 2.5),
                    )
                else:
                    plot_residuals_orb(
                        fitter,
                        title=False,
                        legend=False,
                        avg=False,
                        axs=ax2,
                        figsize=(8, 2.5),
                    )
            # plot dmx v. time
            if "dispersion_dmx" in fitter.model.get_components_by_category().keys():
                plot_dmx_time(
                    fitter,
                    savedmx="dmxparse.out",
                    legend=False,
                    title=False,
                    axs=ax3,
                    figsize=(8, 2.5),
                )
            else:
                log.warning("No DMX bins in timing model, cannot plot DMX v. Time.")
            plt.tight_layout()
            plt.savefig("%s_summary_plot_1_nb.png" % (fitter.model.PSR.value))
            plt.close()
        elif ii == 1:
            if hasattr(fitter.model, "binary_model_name"):
                gs = fig.add_gridspec(4, 2)
                ax2 = fig.add_subplot(gs[2, :])
                ax3 = fig.add_subplot(gs[3, 0])
                ax4 = fig.add_subplot(gs[3, 1])
            else:
                gs = fig.add_gridspec(3, 2)
                ax3 = fig.add_subplot(gs[2, 0])
                ax4 = fig.add_subplot(gs[2, 1])
            ax0 = fig.add_subplot(gs[0, :])
            ax1 = fig.add_subplot(gs[1, :])
            # plot whitened residuals v time
            plot_residuals_time(
                fitter, title=False, whitened=True, axs=ax0, figsize=(8, 2.5)
            )
            # plot whitened, epoch averaged residuals v time
            if "ecorr_noise" in fitter.model.get_components_by_category().keys():
                plot_residuals_time(
                    fitter,
                    title=False,
                    legend=False,
                    avg=True,
                    whitened=True,
                    axs=ax1,
                    figsize=(8, 2.5),
                )
            else:
                plot_residuals_time(
                    fitter,
                    title=False,
                    legend=False,
                    avg=False,
                    whitened=True,
                    axs=ax1,
                    figsize=(8, 2.5),
                )
            # Plot whitened, epoch averaged residuals v orbital phase
            if hasattr(fitter.model, "binary_model_name"):
                if "ecorr_noise" in fitter.model.get_components_by_category().keys():
                    plot_residuals_orb(
                        fitter,
                        title=False,
                        legend=False,
                        avg=True,
                        whitened=True,
                        axs=ax2,
                        figsize=(8, 2.5),
                    )
                else:
                    plot_residuals_orb(
                        fitter,
                        title=False,
                        legend=False,
                        avg=False,
                        whitened=True,
                        axs=ax2,
                        figsize=(8, 2.5),
                    )
            # plot number of whitened residuals histogram
            plot_measurements_v_res(
                fitter,
                nbin=50,
                title=False,
                legend=False,
                whitened=True,
                axs=ax3,
                figsize=(4, 2.5),
            )
            # plot number of whitened, epoch averaged residuals histogram
            if "ecorr_noise" in fitter.model.get_components_by_category().keys():
                plot_measurements_v_res(
                    fitter,
                    nbin=50,
                    title=False,
                    legend=False,
                    avg=True,
                    whitened=True,
                    axs=ax4,
                    figsize=(4, 2.5),
                )
            else:
                plot_measurements_v_res(
                    fitter,
                    nbin=50,
                    title=False,
                    legend=False,
                    avg=False,
                    whitened=True,
                    axs=ax4,
                    figsize=(4, 2.5),
                )
            plt.tight_layout()
            plt.savefig("%s_summary_plot_2_nb.png" % (fitter.model.PSR.value))
            plt.close()
        elif ii == 2:
            if hasattr(fitter.model, "binary_model_name"):
                gs = fig.add_gridspec(4, 2)
                ax2 = fig.add_subplot(gs[2, :])
                ax3 = fig.add_subplot(gs[3, 0])
                ax4 = fig.add_subplot(gs[3, 1])
            else:
                gs = fig.add_gridspec(3, 2)
                ax3 = fig.add_subplot(gs[2, 0])
                ax4 = fig.add_subplot(gs[2, 1])
            ax0 = fig.add_subplot(gs[0, :])
            ax1 = fig.add_subplot(gs[1, :])
            # plot whitened residuals/uncertainty v. time
            plot_residuals_time(
                fitter,
                plotsig=True,
                title=False,
                whitened=True,
                axs=ax0,
                figsize=(8, 2.5),
            )
            # plot whitened, epoch averaged residuals/uncertainty v. time
            if "ecorr_noise" in fitter.model.get_components_by_category().keys():
                plot_residuals_time(
                    fitter,
                    title=False,
                    legend=False,
                    plotsig=True,
                    avg=True,
                    whitened=True,
                    axs=ax1,
                    figsize=(8, 2.5),
                )
            else:
                plot_residuals_time(
                    fitter,
                    title=False,
                    legend=False,
                    plotsig=True,
                    avg=False,
                    whitened=True,
                    axs=ax1,
                    figsize=(8, 2.5),
                )
            # plot whitened, epoch averaged residuals/uncertainty v. orbital phase
            if hasattr(fitter.model, "binary_model_name"):
                if "ecorr_noise" in fitter.model.get_components_by_category().keys():
                    plot_residuals_orb(
                        fitter,
                        title=False,
                        legend=False,
                        plotsig=True,
                        avg=True,
                        whitened=True,
                        axs=ax2,
                        figsize=(8, 2.5),
                    )
                else:
                    plot_residuals_orb(
                        fitter,
                        title=False,
                        legend=False,
                        plotsig=True,
                        avg=False,
                        whitened=True,
                        axs=ax2,
                        figsize=(8, 2.5),
                    )
            # plot number of whitened residuals/uncertainty histogram
            plot_measurements_v_res(
                fitter,
                nbin=50,
                plotsig=True,
                title=False,
                legend=False,
                whitened=True,
                axs=ax3,
                figsize=(4, 2.5),
            )
            # plot number of whitened, epoch averaged residuals/uncertainties histogram
            if "ecorr_noise" in fitter.model.get_components_by_category().keys():
                plot_measurements_v_res(
                    fitter,
                    nbin=50,
                    plotsig=True,
                    title=False,
                    legend=False,
                    avg=True,
                    whitened=True,
                    axs=ax4,
                    figsize=(4, 2.5),
                )
            else:
                plot_measurements_v_res(
                    fitter,
                    nbin=50,
                    plotsig=True,
                    title=False,
                    legend=False,
                    avg=False,
                    whitened=True,
                    axs=ax4,
                    figsize=(4, 2.5),
                )
            plt.tight_layout()
            plt.savefig("%s_summary_plot_3_nb.png" % (fitter.model.PSR.value))
            plt.close()
        elif ii == 3:
            gs = fig.add_gridspec(1, 1)
            ax0 = fig.add_subplot(gs[0])
            plot_residuals_freq(
                fitter, title=False, legend=True, axs=ax0, figsize=(8, 4)
            )
            plt.tight_layout()
            plt.savefig("%s_summary_plot_4_nb.png" % (fitter.model.PSR.value))
            plt.close()


def plots_for_summary_pdf_wb(fitter, title=None, legends=False):
    """
    Function to make a composite set of summary plots for sets of TOAs to be put into a summary pdf.
    This is for Wideband timing only. For Narrowband timing, use `plots_for_summary_pdf_nb`.
    By definition, this function will save all plots as "psrname"_summary_plot_#.wb.png, where # is
    and integer from 1-4.

    Arguments
    ---------
    fitter [object] : The PINT fitter object.
    title [boolean] : If True, will add titles to ALL plots [default: False].
    legend [boolean] : If True, will add legends to ALL plots [default: False].
    """
    if not fitter.is_wideband:
        raise ValueError(
            "Cannot use this function with non-WidebandTOAFitter, please use `plots_for_summary_pdf_nb` instead."
        )
    # Need to make four sets of plots
    for ii in range(4):
        if ii != 3:
            fig = plt.figure(figsize=(8, 10.0), dpi=100)
        else:
            fig = plt.figure(figsize=(8, 5), dpi=100)
        if title != None:
            plt.title(title, y=1.08, size=14)
        if ii == 0:
            if hasattr(fitter.model, "binary_model_name"):
                gs = fig.add_gridspec(nrows=4, ncols=1)
                ax2 = fig.add_subplot(gs[2, :])
                ax3 = fig.add_subplot(gs[3, :])
            else:
                gs = fig.add_gridspec(nrows=3, ncols=1)
                ax3 = fig.add_subplot(gs[2, :])
            ax0 = fig.add_subplot(gs[0, :])
            ax1 = fig.add_subplot(gs[1, :])
            # Plot time residuals v. time
            plot_residuals_time(fitter, title=False, axs=ax0, figsize=(8, 2.5))
            # Plot DM residuals v. time
            plot_dm_residuals(
                fitter, save=False, legend=False, title=False, axs=ax1, figsize=(8, 2.5)
            )
            # Plot time residuals v. orbital phase
            if hasattr(fitter.model, "binary_model_name"):
                plot_residuals_orb(
                    fitter, title=False, legend=False, axs=ax2, figsize=(8, 2.5)
                )
            plot_dmx_time(
                fitter,
                savedmx="dmxparse.out",
                legend=False,
                title=False,
                axs=ax3,
                figsize=(8, 2.5),
            )
            plt.tight_layout()
            plt.savefig("%s_summary_plot_1_wb.png" % (fitter.model.PSR.value))
            plt.close()
        elif ii == 1:
            if hasattr(fitter.model, "binary_model_name"):
                gs = fig.add_gridspec(3, 2)
                ax2 = fig.add_subplot(gs[1, :])
                ax3 = fig.add_subplot(gs[2, 0])
                ax4 = fig.add_subplot(gs[2, 1])
            else:
                gs = fig.add_gridspec(2, 2)
                ax3 = fig.add_subplot(gs[1, 0])
                ax4 = fig.add_subplot(gs[1, 1])
            ax0 = fig.add_subplot(gs[0, :])
            # ax1 = fig.add_subplot(gs[1,:])
            # Plot whitened time residuals v. time
            plot_residuals_time(
                fitter, title=False, whitened=True, axs=ax0, figsize=(8, 2.5)
            )
            # Plot whitened time residuals v. time
            if hasattr(fitter.model, "binary_model_name"):
                plot_residuals_orb(
                    fitter,
                    title=False,
                    legend=False,
                    whitened=True,
                    axs=ax2,
                    figsize=(8, 2.5),
                )
            # Plot number of whitened residuals histograms
            plot_measurements_v_res(
                fitter,
                nbin=50,
                title=False,
                plotsig=False,
                legend=False,
                whitened=True,
                axs=ax3,
                figsize=(4, 2.5),
            )
            # plot number of DM residuals histograms
            plot_measurements_v_dmres(
                fitter, nbin=50, legend=False, title=False, axs=ax4
            )
            plt.tight_layout()
            plt.savefig("%s_summary_plot_2_wb.png" % (fitter.model.PSR.value))
            plt.close()
        elif ii == 2:
            if hasattr(fitter.model, "binary_model_name"):
                gs = fig.add_gridspec(4, 2)
                ax2 = fig.add_subplot(gs[2, :])
                ax3 = fig.add_subplot(gs[3, 0])
                ax4 = fig.add_subplot(gs[3, 1])
            else:
                gs = fig.add_gridspec(3, 2)
                ax3 = fig.add_subplot(gs[2, 0])
                ax4 = fig.add_subplot(gs[2, 1])
            ax0 = fig.add_subplot(gs[0, :])
            ax1 = fig.add_subplot(gs[1, :])
            # plot whitened time residuals/uncertainty v time
            plot_residuals_time(
                fitter,
                plotsig=True,
                title=False,
                whitened=True,
                axs=ax0,
                figsize=(8, 2.5),
            )
            # Plot DM residuals/uncertainty v. time
            plot_dm_residuals(
                fitter,
                plotsig=True,
                save=False,
                legend=False,
                title=False,
                axs=ax1,
                figsize=(8, 2.5),
            )
            # Plot whitened time residuals/uncertainty v orbital phase
            if hasattr(fitter.model, "binary_model_name"):
                plot_residuals_orb(
                    fitter,
                    title=False,
                    legend=False,
                    plotsig=True,
                    whitened=True,
                    axs=ax2,
                    figsize=(8, 2.5),
                )
            # plot number of whitened time residuals/uncertainty histograms
            plot_measurements_v_res(
                fitter,
                nbin=50,
                title=False,
                plotsig=True,
                legend=False,
                whitened=True,
                axs=ax3,
                figsize=(4, 2.5),
            )
            # plot number of DM residuals/uncertainty histograms
            plot_measurements_v_dmres(
                fitter, plotsig=True, nbin=50, legend=False, title=False, axs=ax4
            )
            plt.tight_layout()
            plt.savefig("%s_summary_plot_3_wb.png" % (fitter.model.PSR.value))
            plt.close()
        elif ii == 3:
            gs = fig.add_gridspec(1, 1)
            ax0 = fig.add_subplot(gs[0])
            plot_residuals_freq(
                fitter, title=False, legend=True, axs=ax0, figsize=(8, 4)
            )
            plt.tight_layout()
            plt.savefig("%s_summary_plot_4_wb.png" % (fitter.model.PSR.value))
            plt.close()


def plot_settings(colorby="f"):
    """
    Initialize plot rc params, define color scheme
    """
    fig_width_pt = 620
    inches_per_pt = 1.0 / 72.27  # Convert pt to inches
    golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean * 2  # height in inches
    fig_size = [fig_width, fig_height]
    fontsize = 20  # for xlabel, backend labels
    plotting_params = {
        "backend": "pdf",
        "axes.labelsize": 12,
        "lines.markersize": 4,
        "font.size": 12,
        "xtick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.major.size": 6,
        "ytick.minor.size": 3,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "lines.markeredgewidth": 1,
        "axes.linewidth": 1.2,
        "legend.fontsize": 10,
        "xtick.labelsize": 12,
        "ytick.labelsize": 10,
        "savefig.dpi": 400,
        "path.simplify": True,
        "font.family": "serif",
        "font.serif": "Times",
        "text.usetex": True,
        "figure.figsize": fig_size,
        "text.latex.preamble": r"\usepackage{amsmath}",
    }

    plt.rcParams.update(plotting_params)

    colorscheme, markerscheme = set_color_and_marker(colorby)
    return markerscheme, colorscheme


def get_fitter(yaml):
    """
    Get the fitter and model from a given YAML

    Parameters
    ==========
    yaml: str
        yaml to use for locating latest results

    """
    tc = TimingConfiguration(yaml)
    mo, to = tc.get_model_and_toas(excised=True, usepickle=True)
    tc.manual_cuts(to)
    receivers = lu.get_receivers(to)
    if tc.get_toa_type() == "WB":
        lu.add_feDMJumps(mo, receivers)
    else:
        lu.add_feJumps(mo, receivers)
    fo = tc.construct_fitter(to, mo)
    return fo, mo


def get_avg_years(fo_nb, fo_wb, avg_dict):
    """
    Get MJDS for each data set in years

    Parameters
    ==========
    fo: fitter object
    mo: model object
    avg_dict: from fo.resids.ecorr_average()

    """
    mjd_nb = fo_nb.toas.get_mjds().value
    years_nb = (mjd_nb - 51544.0) / 365.25 + 2000.0
    mjd_wb = fo_wb.toas.get_mjds().value
    years_wb = (mjd_wb - 51544.0) / 365.25 + 2000.0
    mjds_avg = avg_dict["mjds"].value
    years_avg = (mjds_avg - 51544.0) / 365.25 + 2000.0
    return years_nb, years_wb, years_avg


def get_backends(fo_nb, fo_wb, avg_dict):
    """
    Grab backends via flags to make plotting easier

    Parameters
    ==========
    fo: fitter object
    mo: model object
    avg_dict: from fo.resids.ecorr_average()

    """
    rcvr_bcknds_nb = np.array(fo_nb.toas.get_flag_value("f")[0])
    rcvr_set_nb = set(rcvr_bcknds_nb)
    rcvr_bcknds_wb = np.array(fo_wb.toas.get_flag_value("f")[0])
    rcvr_set_wb = set(rcvr_bcknds_wb)
    avg_rcvr_bcknds = []
    for iis in avg_dict["indices"]:
        avg_rcvr_bcknds.append(rcvr_bcknds_nb[iis[0]])
    rcvr_bcknds_avg = np.array(avg_rcvr_bcknds)
    rcvr_set_avg = set(rcvr_bcknds_avg)
    return rcvr_bcknds_nb, rcvr_bcknds_wb, rcvr_bcknds_avg


def get_DMX_info(fo):
    """
    Get DMX timeseries info from dmxparse

    Parameters
    ==========
    fo: fitter object

    """
    dmx_dict = pint.utils.dmxparse(fo)
    DMXs = dmx_dict["dmxs"].value
    DMX_vErrs = dmx_dict["dmx_verrs"].value
    DMX_center_MJD = dmx_dict["dmxeps"].value
    DMX_center_Year = (DMX_center_MJD - 51544.0) / 365.25 + 2000.0
    return DMXs, DMX_vErrs, DMX_center_Year


def plot_by_color(ax, x, y, err, bknds, rn_off, be_legend, be_format):
    """
    Plot color-divided-by-receiver/BE points on any axis

    Parameters
    ==========
    ax: axis for plotting
    x: x values to plot
    y: y values to plot
    err: error bars to plot
    bknds: list of backend flags associated with TOAs
    rn_off: the DC red noise offset to subtract (prior to PINT fix)

    """
    markers, colorscheme = plot_settings()
    for i, r_b in enumerate(set(bknds)):
        inds = np.where(bknds == r_b)[0]
        if not inds.tolist():
            r_b_label = ""
        else:
            r_b_label = bknds[inds][0]
        mkr = markers[r_b_label]
        clr = colorscheme[r_b_label]
        ax.errorbar(
            x[inds],
            y[inds] - (rn_off * u.us),
            yerr=err[inds],
            fmt=mkr,
            color=clr,
            label=r_b_label,
            alpha=0.5,
        )

    ylim = max(np.abs(y - (rn_off * u.us))).value + 0.6 * max(np.abs(err)).value
    ax.set_ylim(-1 * ylim * 1.08, ylim * 1.08)

    if be_legend:
        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        fixed_labels = [label_names[l] for l in labels]
        if be_format == "vert":
            plt.legend(handles, fixed_labels, loc=(1.005, 0), fontsize=12)
        if be_format == "horiz":
            plt.legend(
                handles,
                fixed_labels,
                loc="lower left",
                ncol=len(fixed_labels),
                borderpad=0.1,
                columnspacing=0.1,
            )
            ax.set_ylim(-1 * ylim * 1.2, ylim * 1.08)


def rec_labels(axs, bcknds, years_avg):
    """
    Mark transitions between backends

    Parameters
    ==========
    axs: axis for plotting
    x: x values to plot
    y: y values to plot
    err: error bars to plot
    bknds: list of backend flags associated with TOAs
    rn_off: the DC red noise offset to subtract (prior to PINT fix)

    """
    guppi = 2010.1
    puppi = 2012.1

    has_asp = False
    has_puppi = False
    has_gasp = False
    has_guppi = False
    has_ao = False
    has_gbt = False
    has_yuppi = False

    for r in bcknds:
        if "ASP" in r:
            has_asp = True
        if "PUPPI" in r:
            has_puppi = True
        if "GASP" in r:
            has_gasp = True
        if "GUPPI" in r:
            has_guppi = True
        if "YUPPI" in r:
            has_yuppi = True

    if has_asp and has_puppi:
        for a in axs:
            has_ao = True
            a.axvline(puppi, linewidth=0.75, color="k", linestyle="--", alpha=0.6)
    if has_gasp and has_guppi:
        for a in axs:
            has_gbt = True
            a.axvline(guppi, linewidth=0.75, color="k", linestyle="--", alpha=0.6)

    ycoord = 1.1
    x_min_yr = min(years_avg)
    x_max_yr = max(years_avg)

    tform = axs[0].get_xaxis_transform()
    va = ha = "center"

    if has_ao and has_gbt:
        if has_yuppi:
            axs[0].text(
                (puppi + x_max_yr) / 2.0,
                ycoord,
                "PUPPI/GUPPI/YUPPI",
                transform=tform,
                va=va,
                ha=ha,
            )
        else:
            axs[0].text(
                (puppi + x_max_yr) / 2.0,
                ycoord,
                "PUPPI/GUPPI",
                transform=tform,
                va=va,
                ha=ha,
            )
        axs[0].text(
            (guppi + x_min_yr) / 2.0, ycoord, "ASP/GASP", transform=tform, va=va, ha=ha
        )
        axs[0].text(
            (guppi + puppi) / 2.0, ycoord, "ASP/GUPPI", transform=tform, va=va, ha=ha
        )
    elif has_ao and not has_gbt:
        if has_yuppi:
            axs[0].text(
                (puppi + x_max_yr) / 2.0,
                ycoord,
                "PUPPI/YUPPI",
                transform=tform,
                va=va,
                ha=ha,
            )
        else:
            axs[0].text(
                (puppi + x_max_yr) / 2.0, ycoord, "PUPPI", transform=tform, va=va, ha=ha
            )
        axs[0].text(
            (puppi + x_min_yr) / 2.0 - 0.2, ycoord, "ASP", transform=tform, va=va, ha=ha
        )
    elif not has_ao and has_gbt:
        if has_yuppi:
            axs[0].text(
                (puppi + x_max_yr) / 2.0,
                ycoord,
                "GUPPI/YUPPI",
                transform=tform,
                va=va,
                ha=ha,
            )
        else:
            axs[0].text(
                (guppi + x_max_yr) / 2.0, ycoord, "GUPPI", transform=tform, va=va, ha=ha
            )
        axs[0].text(
            (guppi + x_min_yr) / 2.0, ycoord, "GASP", transform=tform, va=va, ha=ha
        )
    if has_puppi and not has_asp and not has_gasp and not has_guppi:
        if has_yuppi:
            axs[0].text(
                (x_min_yr + x_max_yr) / 2.0,
                ycoord,
                "PUPPI/YUPPI",
                transform=tform,
                va=va,
                ha=ha,
            )
        else:
            axs[0].text(
                (x_min_yr + x_max_yr) / 2.0,
                ycoord,
                "PUPPI",
                transform=tform,
                va=va,
                ha=ha,
            )
    if has_guppi and not has_asp and not has_gasp and not has_puppi:
        if has_yuppi:
            axs[0].text(
                (x_min_yr + x_max_yr) / 2.0,
                ycoord,
                "GUPPI/YUPPI",
                transform=tform,
                va=va,
                ha=ha,
            )
        else:
            axs[0].text(
                (x_min_yr + x_max_yr) / 2.0,
                ycoord,
                "GUPPI",
                transform=tform,
                va=va,
                ha=ha,
            )
    if has_yuppi and not has_guppi and not has_puppi:
        axs[0].text(
            (x_min_yr + x_max_yr) / 2.0, ycoord, "YUPPI", transform=tform, va=va, ha=ha
        )


def rn_sub(testing, rn_subtract, fo_nb, fo_wb):
    if rn_subtract:
        if testing:
            rn_nb = 0.0
            rn_wb = 0.0
        else:
            rn_nb = fo_nb.current_state.xhat[0] * fo_nb.current_state.M[0, 0] * 1e6
            rn_wb = fo_wb.current_state.xhat[0] * fo_wb.current_state.M[0, 0] * 1e6
    else:
        rn_nb = 0.0
        rn_wb = 0.0
    return rn_nb, rn_wb
