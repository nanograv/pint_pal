"""
This module provides tools for checking the finality of yamls
and gathering results for hand-off to DWG
"""

from timing_analysis.yamlio import *
from timing_analysis.timingconfiguration import TimingConfiguration
from astropy import log
from datetime import datetime
import subprocess
import shutil
import argparse
import glob
import os
import numpy as np
from matplotlib import pyplot as plt
import timing_analysis.lite_utils as lu
import pint.utils
import astropy.units as u

# accessible to functions here, apparently
TA_PATH = "/home/jovyan/work/timing_analysis/" # assume running from here?
INTERMED_PATH = "/nanograv/share/15yr/timing/intermediate/"
TA_RESULTS = os.path.join(TA_PATH,"results")
TA_CONFIGS = os.path.join(TA_PATH,"configs")

def make_release_dir(type, overwrite=False):
    """
    Make new release directory to contain latest results

    Parameters
    ==========
    type: str
        narrowband (nb) or wideband (wb)
    overwrite: bool, optional
        overwrite existing files if release directory already exists (default: False)
    """
    now = datetime.now()
    Ymd = now.strftime("%Y%m%d")

    cmd = subprocess.Popen(["git","rev-parse","--short","HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
    githash, stderr = cmd.communicate()
    githash = githash.strip().decode() # decode() since a bytestring is returned initially (b'[hash]')
    release_dir = f"{INTERMED_PATH}{Ymd}.Release.{type}.{githash}"
    if not os.path.isdir(release_dir):
        log.info(f"Making new release directory: {release_dir}")
        os.mkdir(release_dir)
    elif os.path.isdir(release_dir) and overwrite:
        log.warning(f"Overwriting files in release directory: {release_dir}")
    else:
        log.warning(f"Release directory already exists: {release_dir}")

    return release_dir

def check_cleared(type):
    """
    Check that all yamls of the specified type have been cleared

    Parameters
    ==========
    type: str
        narrowband (nb), wideband (wb), or both (nbwb)
    """
    if type == "nbwb":
        yamls = glob.glob(f"{TA_CONFIGS}/*.yaml")
    else:
        yamls = glob.glob(f"{TA_CONFIGS}/*.{type}.yaml")
    for y in yamls:
        tc = TimingConfiguration(y)
        if not tc.get_check_cleared():
            log.warning(f"{tc.get_source()} has not been cleared.")
            
    return yamls

def check_dupes_copy(results, release_dir, add_base=None):
    """
    Check for duplicate results (copy if no duplicates)

    Parameters
    ==========
    results: list
        list (should be one element) of results file(s) to copy
    release_dir: str
        path to release directory
    add_base: str, optional
        optional basename added to file being copied
    """
    if len(results) != 1:
        log.warning("Multiple/no matching results files found.")
        print(results)
    else:
        file2copy = os.path.basename(results[0])
        if add_base:
            file2copy = f"{add_base}.{file2copy}"

        dest_file = f"{release_dir}/{file2copy}"
        shutil.copyfile(results[0], dest_file)

def locate_copy_results(yamls,type,destination=None):
    """
    Get latest results from yamls, copy to release directory

    Parameters
    ==========
    yamls: list
        yamls to use for locating latest results
    type: str
        narrowband (nb) or wideband (wb)
    destination: str
        path to release directory
    """
    for y in yamls:
        tc = TimingConfiguration(y)
        source = tc.get_source()
        noise_dir = tc.get_noise_dir()
        latest_yaml = [y]
        latest_par = [f"{TA_PATH}{tc.get_model_path()}"]
        latest_tim = glob.glob(f"{noise_dir}results/{source}_*.tim") # underscore to avoid duplicating split-tel results
        noise_chains = glob.glob(f"{noise_dir}{source}_{type}/chain_1.txt")
        noise_pars = glob.glob(f"{noise_dir}{source}_{type}/pars.txt")

        log.info(f"Locating/copying files for {source}...")
        check_dupes_copy(latest_tim, destination)
        check_dupes_copy(latest_par, destination)
        check_dupes_copy(latest_yaml, destination)
        check_dupes_copy(noise_chains, destination, add_base=f"{source}.{type}")
        check_dupes_copy(noise_pars, destination, add_base=f"{source}.{type}")
       
def plot_settings():
    """
    Initialize plot rc params, define color scheme
    """
    fig_width_pt = 620
    inches_per_pt = 1.0/72.27               # Convert pt to inches
    golden_mean = (np.sqrt(5)-1.0)/2.0      # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean*2       # height in inches
    fig_size = [fig_width,fig_height]
    fontsize = 20  # for xlabel, backend labels
    plotting_params = {'backend': 'pdf', 'axes.labelsize': 12, 'lines.markersize': 4, 'font.size': 12, 'xtick.major.size': 6, 'xtick.minor.size': 3, 'ytick.major.size': 6, 'ytick.minor.size': 3, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5, 'lines.markeredgewidth': 1, 'axes.linewidth': 1.2, 'legend.fontsize': 10, 'xtick.labelsize': 12, 'ytick.labelsize': 10, 'savefig.dpi': 400, 'path.simplify': True, 'font.family': 'serif', 'font.serif': 'Times', 'text.usetex': True, 'figure.figsize': fig_size, 'text.latex.preamble': [r'\usepackage{amsmath}', r'\usepackage{apjfonts}']}

    plt.rcParams.update(plotting_params)

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
        "1.5GHz_YUPPI":    "#45062E",
        "3GHz_YUPPI":      "#E5A4CB",
        "6GHz_YUPPI":      "#40635F",
        "CHIME":           "#ECE133",
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
          "1.5GHz_YUPPI":    "x",
          "3GHz_YUPPI":      "x",
          "6GHz_YUPPI":      "x",
          "CHIME":           "x",
           }
    
    # Define the color map option
    colorscheme = colorschemes['thankful_2']
    
    return markers, colorscheme

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
    if tc.get_toa_type() == 'WB':
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
    years_nb = (mjd_nb - 51544.0)/365.25 + 2000.0
    mjd_wb = fo_wb.toas.get_mjds().value 
    years_wb = (mjd_wb - 51544.0)/365.25 + 2000.0
    mjds_avg = avg_dict['mjds'].value
    years_avg = (mjds_avg - 51544.0)/365.25 + 2000.0
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
    rcvr_bcknds_nb = np.array(fo_nb.toas.get_flag_value('f')[0])
    rcvr_set_nb = set(rcvr_bcknds_nb)
    rcvr_bcknds_wb = np.array(fo_wb.toas.get_flag_value('f')[0])
    rcvr_set_wb = set(rcvr_bcknds_wb)
    avg_rcvr_bcknds = []
    for iis in avg_dict['indices']:
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
    DMXs = dmx_dict['dmxs'].value
    DMX_vErrs = dmx_dict['dmx_verrs'].value
    DMX_center_MJD = dmx_dict['dmxeps'].value
    DMX_center_Year = (DMX_center_MJD - 51544.0)/365.25 + 2000.0
    return DMXs, DMX_vErrs, DMX_center_Year

def plot_by_color(ax, x, y, err, bknds, rn_off, be_legend):
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
        inds = np.where(bknds==r_b)[0]
        if not inds.tolist():
            r_b_label = ""
        else:
            r_b_label = bknds[inds][0]
        mkr = markers[r_b_label]
        clr = colorscheme[r_b_label]
        ax.errorbar(x[inds], y[inds] - (rn_off * u.us), yerr=err[inds], fmt=mkr, color=clr, label=r_b_label, alpha=0.5)
        
    ylim = (max(np.abs(y - (rn_off * u.us))).value + 0.6 * max(np.abs(err)).value)
    ax.set_ylim(-1 * ylim * 1.08, ylim * 1.08)
    
    if be_legend:
        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        plt.legend(handles, labels, loc=(1.005, 0), fontsize=12)
        

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
        if 'ASP' in r:
            has_asp = True
        if 'PUPPI' in r:
            has_puppi = True
        if 'GASP' in r:
            has_gasp = True
        if 'GUPPI' in r:
            has_guppi = True
        if 'YUPPI' in r:
            has_yuppi = True

    if has_asp and has_puppi:
        for a in axs:
            has_ao = True
            a.axvline(puppi, linewidth=0.75, color='k', linestyle='--', alpha=0.6)
    if has_gasp and has_guppi:
        for a in axs:
            has_gbt = True
            a.axvline(guppi, linewidth=0.75, color='k', linestyle='--', alpha=0.6)
    
    ycoord = 1.1
    x_min_yr = min(years_avg)
    x_max_yr = max(years_avg)
    
    tform = axs[0].get_xaxis_transform()
    va = ha = 'center'
    
    if has_ao and has_gbt:
        if has_yuppi:
            axs[0].text((puppi+x_max_yr)/2., ycoord, 'PUPPI/GUPPI/YUPPI', transform=tform, va=va, ha=ha)
        else:
            axs[0].text((puppi+x_max_yr)/2., ycoord, 'PUPPI/GUPPI', transform=tform, va=va, ha=ha)
        axs[0].text((guppi+x_min_yr)/2., ycoord, 'ASP/GASP', transform=tform, va=va, ha=ha)
        axs[0].text((guppi+puppi)/2., ycoord, 'ASP/GUPPI', transform=tform, va=va, ha=ha)        
    elif has_ao and not has_gbt:
        if has_yuppi:
            axs[0].text((puppi+x_max_yr)/2., ycoord, 'PUPPI/YUPPI', transform=tform, va=va, ha=ha)
        else:
            axs[0].text((puppi+x_max_yr)/2., ycoord, 'PUPPI', transform=tform, va=va, ha=ha)
        axs[0].text((puppi+x_min_yr)/2. - 0.2, ycoord, 'ASP', transform=tform, va=va, ha=ha)        
    elif not has_ao and has_gbt:
        if has_yuppi:
            axs[0].text((puppi+x_max_yr)/2., ycoord, 'GUPPI/YUPPI', transform=tform, va=va, ha=ha)
        else:
            axs[0].text((guppi+x_max_yr)/2., ycoord, 'GUPPI', transform=tform, va=va, ha=ha)
        axs[0].text((guppi+x_min_yr)/2., ycoord, 'GASP', transform=tform, va=va, ha=ha)
    if has_puppi and not has_asp and not has_gasp and not has_guppi:
        if has_yuppi:
            axs[0].text((x_min_yr+x_max_yr)/2., ycoord, 'PUPPI/YUPPI', transform=tform, va=va, ha=ha)
        else:
            axs[0].text((x_min_yr+x_max_yr)/2., ycoord, 'PUPPI', transform=tform, va=va, ha=ha)
    if has_guppi and not has_asp and not has_gasp and not has_puppi:
        if has_yuppi:
            axs[0].text((x_min_yr+x_max_yr)/2., ycoord, 'GUPPI/YUPPI', transform=tform, va=va, ha=ha)
        else:
            axs[0].text((x_min_yr+x_max_yr)/2., ycoord, 'GUPPI', transform=tform, va=va, ha=ha)
    if has_yuppi and not has_guppi and not has_puppi:
        axs[0].text((x_min_yr+x_max_yr)/2., ycoord, 'YUPPI', transform=tform, va=va, ha=ha)
            
def rn_sub(testing, rn_subtract, fo_nb, fo_wb):
    if rn_subtract:
        if testing:
            rn_nb = 0.0
            rn_wb = 0.0
        else:
            rn_nb = fo_nb.current_state.xhat[0] * fo_nb.current_state.M[0,0] * 1e6
            rn_wb = fo_wb.current_state.xhat[0] * fo_wb.current_state.M[0,0] * 1e6
    else:
        rn_nb = 0.0
        rn_wb = 0.0
    return rn_nb, rn_wb

def main():

    parser = argparse.ArgumentParser(
        description="Make release directory; copy final nb/wb TA data products there",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        nargs=1,
        help="Release type: nb, wb, or both (nbwb)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite files in release directory",
    )
    args = parser.parse_args()

    if not args.type:
        print("Please provide a release type.")
    elif args.type[0] not in ['nb','wb','nbwb']:
        print(args.type)
        print("Unrecognized release type.")
    else:
        if args.type[0] != "nbwb":
            # make directory
            rel_dir = make_release_dir(args.type[0], overwrite=args.overwrite)

            # get yamls
            yamls = check_cleared(args.type[0])

            # locate results and copy them to release directory
            locate_copy_results(yamls,args.type[0],rel_dir)

        else: # nbwb
            rel_dir = make_release_dir(args.type[0], overwrite=args.overwrite)
            nb_yamls = check_cleared('nb')
            locate_copy_results(nb_yamls,'nb',rel_dir) # works for nb/wb separately
            wb_yamls = check_cleared('wb')
            locate_copy_results(wb_yamls,'wb',rel_dir)

if __name__ == "__main__":
    main()
