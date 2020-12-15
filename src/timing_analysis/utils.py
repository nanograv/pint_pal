#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import astropy.units as u
from astropy import log
from pint.utils import weighted_mean
import pint.residuals as Resid
import os
import time
from subprocess import check_output
import glob
# Import some software so we have appropriate versions
import pint
import astropy
# import enterprise_extensions as e_e # NOTE - enterprise_extensions has no attribute __version__
from timing_analysis.ftester import get_fblist, param_check

ALPHA = 0.0027

def whiten_resids(fitter, restype = 'postfit'):
    """
    Function to whiten residuals. Returns a residual class object. If no reddened residuals, input will be returned.

    Arguements:
    fitter: PINT fitter class or dictionary output from ecorr_average() function
    restype ['string'] : Type of residuals, pre or post fit, to plot from fitter object. Options are:
        'prefit' - plot the prefit residuals.
        'postfit' - plot the postfit residuals (default)
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
        if "Wideband" in fitter.__class__.__name__:
            resids = fitter.resids.residual_objs['toa']
        else:
            if restype == 'postfit':
                resids = fitter.resids
            elif restype == 'prefit':
                resids = fitter.resids_init
            else:
                raise ValueError("Unrecognized residual type: %s. Please choose from 'prefit' or 'postfit'."%(restype))
        # Get number of residuals
        num_res = len(resids.time_resids)
        # Check that the key is in the dictionary
        if "pl_red_noise" in resids.noise_resids:
            wres = resids.time_resids - resids.noise_resids['pl_red_noise'][:num_res]
        else:
            log.warning("No red noise, residuals already white. Returning input residuals...")
            wres = resids.time_resids
    return wres

def rms_by_backend(resids, errors, rcvr_backends, dm = False):
    """
    Function to take a fitter, list of residuals errors, and backends and compute the rms and weighted rms
    for either time residuals or DM residuals if a wideband residuals.

    Inputs:
    ----------
    resids : list
        List of residuals
    errors : list
        List of residual errors
    rcvr_backends : list
        List of backends
    dm : bool
        If True, will do computation with DM residuals
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
    fitter : class
        PINT fitter class object, post-fit.
    epoch_avg : bool
        If True, will output stats for epoch averaged residulas, else will be for non-epoch averaged residuals.
    whitened : bool
        If True, will output stats for whitened residulas, else will be for non-whitened residuals.
    dm_stats : bool
        If True, will also output the stats for the DM residuals for wideband fitters. Note this will return an
        additional dictionary, dm_stats.
    print_pretty : bool
        If True, will print the W/RMS per receiver-backend combo.

    Output:

    Nested dictionary :

        First set of keys are Receiver-Backend combos, e.g. L-wide_ASP, S-wide_PUPPI. For the W/RMS of all residuals,
        the key is 'All'.

        Within each Receiver-Backend combo the flags are:
            rms : astropy quantity
                rms of the residuals for the receiver-backend combo in microseconds
            wrms : astropy quantity
                weighted rms of the residuals for the receiver-backend combo in microseconds
    """
    # Check if fitter is WB or not
    if "Wideband" in fitter.__class__.__name__:
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



# Define helper functions
def year(mjd):
    return (mjd - 51544.0)/365.25 + 2000.0


def report_ptest(label, rms, chi2, ndof, dmrms = None, Fstatistic=None, alpha=ALPHA):
    if Fstatistic is None:
        if dmrms != None:
            line = "%42s %7.3f %16.6f %9.2f %5d --" % (label, rms, dmrms, chi2, ndof)
        else:
            line = "%42s %7.3f %9.2f %5d --" % (label, rms, chi2, ndof)
    elif Fstatistic:
        if dmrms != None:
            line = "%42s %7.3f %16.6f %9.2f %5d %.3g" % (label, rms, dmrms, chi2, ndof, Fstatistic)
        else:
            line = "%42s %7.3f %9.2f %5d %.3g" % (label, rms, chi2, ndof, Fstatistic)
        if Fstatistic < alpha:
            line += " ***"
    else:
        if dmrms != None:
            line = "%42s %7.3f %16.6f %9.2f %5d xxx" % (label, rms, dmrms, chi2, ndof)
        else:
            line = "%42s %7.3f %9.2f %5d xxx" % (label, rms, chi2, ndof)
    return line


def get_Ftest_lines(Ftest_dict, fitter):
    """
    Function to get nicely formatted lines from F-test dictionary.

    Input
    ----------
    Ftest_dict: Dictionary of F-test results output by the `run_Ftests()` function.
    fitter: The PINT fitter object, which contains
        the TOAs and the models
    """
    ftest_lines = []
    for fk in Ftest_dict.keys():
        if 'FD' in fk:
            for ffk in Ftest_dict[fk].keys():
                if ffk == 'NoFD':
                    label = 'no FD'
                    cur_fd = [param for param in fitter.model.params if "FD" in param]
                    ftest_lines.append("\nTesting FD terms (%s enabled):" % (cur_fd))
                    ft = None
                else:
                    label = "FD1 through %s" % (ffk)
                    ft = Ftest_dict[fk][ffk]['ft']
                l = report_ptest(label, Ftest_dict[fk][ffk]['resid_wrms_test'].value , Ftest_dict[fk][ffk]['chi2_test'],
                                 Ftest_dict[fk][ffk]['dof_test'], Fstatistic=ft)
                ftest_lines.append(l)
        # Get the FB parameter lines
        elif 'FB' in fk:
            # Get the value of fbmax, note, may need fixes somewhere
            try:
                fbmax = (int(max(Ftest_dict[fk].keys())[-1]))
            except:
                fbmax = (int(max(Ftest_dict[fk].keys())[-2]))
            fblist = get_fblist(fitter)
            fbused = (len(fblist)>0)
            fbp = [fblist[ifb] for ifb in sorted(fblist.keys())]  # sorted list of fb parameters
            ftest_lines.append("\nTesting FB parameters, present list: "+" ".join(fbp))
            ftest_lines.append("\nTesting removal of FB parameters:")
            for i in range(1,len(fblist)):
                p = [fbp[j] for j in range(i,len(fbp))]
                ffk = 'FB%s+'%i
                l = report_ptest(" ".join(p), Ftest_dict[fk][ffk]['resid_wrms_test'].value, Ftest_dict[fk][ffk]['chi2_test'], Ftest_dict[fk][ffk]['dof_test'], Fstatistic=Ftest_dict[fk][ffk]['ft'], alpha=ALPHA)
                ftest_lines.append(l)
            ftest_lines.append("Testing addition of FB parameters:")
            for i in range(len(fblist),fbmax+1):
                p = ["FB%d" % (j) for j in range(len(fblist),i+1)]
                ffk = 'FB%s'%i
                l = report_ptest(" ".join(p), Ftest_dict[fk][ffk]['resid_wrms_test'].value, Ftest_dict[fk][ffk]['chi2_test'], Ftest_dict[fk][ffk]['dof_test'], Fstatistic=Ftest_dict[fk][ffk]['ft'], alpha=ALPHA)
                ftest_lines.append(l)


        elif 'initial' in fk:
            l = report_ptest(fk, Ftest_dict[fk]['resid_wrms_test'].value , Ftest_dict[fk]['chi2_test'],
                             Ftest_dict[fk]['dof_test'], Fstatistic=None)
            ftest_lines.append(l)
        elif "Add" in fk:
            ftest_lines.append('Testing additional parameters:')
            for ffk in Ftest_dict[fk].keys():
                if ffk == 'Binary':
                    for fffk in Ftest_dict[fk][ffk].keys():
                        l = report_ptest(fffk, Ftest_dict[fk][ffk][fffk]['resid_wrms_test'].value , Ftest_dict[fk][ffk][fffk]['chi2_test'],
                                 Ftest_dict[fk][ffk][fffk]['dof_test'], Fstatistic=Ftest_dict[fk][ffk][fffk]['ft'])
                        ftest_lines.append(l)
                else:
                    l = report_ptest(ffk, Ftest_dict[fk][ffk]['resid_wrms_test'].value , Ftest_dict[fk][ffk]['chi2_test'],
                                 Ftest_dict[fk][ffk]['dof_test'], Fstatistic=Ftest_dict[fk][ffk]['ft'])
                    ftest_lines.append(l)
        elif "Remove" in fk:
            ftest_lines.append('\nTesting removal of parameters:')
            for ffk in Ftest_dict[fk].keys():
                if ffk == 'Binary':
                    for fffk in Ftest_dict[fk][ffk].keys():
                        l = report_ptest(fffk, Ftest_dict[fk][ffk][fffk]['resid_wrms_test'].value , Ftest_dict[fk][ffk][fffk]['chi2_test'],
                                 Ftest_dict[fk][ffk][fffk]['dof_test'], Fstatistic=Ftest_dict[fk][ffk][fffk]['ft'])
                        ftest_lines.append(l)
                else:
                    l = report_ptest(ffk, Ftest_dict[fk][ffk]['resid_wrms_test'].value , Ftest_dict[fk][ffk]['chi2_test'],
                                 Ftest_dict[fk][ffk]['dof_test'], Fstatistic=Ftest_dict[fk][ffk]['ft'])
                    ftest_lines.append(l)
        elif fk == 'F':
            # Get current number of spin frequency derivatives
            current_freq_deriv = 1
            for i in range(2,21):
                p = "F%d" % i
                if p in fitter.model.params:
                    current_freq_deriv = i
            ftest_lines.append("Testing spin freq derivs (%s enabled):" % (current_freq_deriv))
            for ffk in Ftest_dict[fk].keys():
                l = report_ptest(ffk, Ftest_dict[fk][ffk]['resid_wrms_test'].value , Ftest_dict[fk][ffk]['chi2_test'],
                                 Ftest_dict[fk][ffk]['dof_test'], Fstatistic=Ftest_dict[fk][ffk]['ft'])
                ftest_lines.append(l)

    return ftest_lines


def pdf_writer(fitter, parfile, rs_dict, Ftest_dict, dm_dict = None, append=None):
    """
    Function to take output from timing notebook functions and write things out nicely in a summary pdf.

    Input
    ----------
    fitter: The PINT fitter object, which contains
        the TOAs and the models
    parfile: Name of parfile used to generate residuals.
    rs_dict: Dictionary of residual stats output by the `resid_stats()` function.
    Ftest_dict: Dictionary of F-test results output by the `run_Ftests()` function.
    dm_dict: Dictionary of DM residual stats output by the `resid_stats()` function for WB timing.
        Input is optional, if `None` (Default value), will not write out the DM residual stats.
    append: defualt is `None`, else should be a string to the path to the texfile to append output to.

    """
    # Check if fitter is wideband or not
    if "Wideband" in fitter.__class__.__name__:
        NB = False
        resids = fitter.resids.residual_objs['toa']
        dm_resids = fitter.resids.residual_objs['dm']
        # TO DO: Double check how to konw if has_dmdata should be True
        if 'DMDATA' in fitter.model.params:
            if int(fitter.model.DMDATA) == 1:
                has_dmdata = True
            else:
                has_dmdata = False
        else:
            has_dmdata = False
    else:
        NB = True
        resids = fitter.resids

    # Start the latex pdf text (from old finalize timing script)
    psr = fitter.model.PSR.value.replace('-','$-$')
    write_header = True
    if append is not None:
        texfile = append
        if os.path.exists(texfile):
            write_header = False
        fsum = open(texfile,'a')
    else:
        texfile = fitter.model.PSR.value + '.summary.tex'
        fsum = open(texfile,'w')
    if write_header:
        fsum.write(r'\documentclass[11pt]{article}' + '\n')
        fsum.write(r'\usepackage{graphicx}' + '\n')
        fsum.write(r'\addtolength{\hoffset}{-2.5cm}' + '\n')
        fsum.write(r'\addtolength{\textwidth}{5.0cm}' + '\n')
        fsum.write(r'\addtolength{\voffset}{-2.5cm}' + '\n')
        fsum.write(r'\addtolength{\textheight}{5.0cm}' + '\n')
        fsum.write(r'\usepackage{fancyhdr}' + '\n')
        fsum.write(r'\pagestyle{fancy}' + '\n')
        fsum.write(r'\lhead{\leftmark}' + '\n')
        fsum.write(r'\rhead{\thepage}' + '\n')
        fsum.write(r'\cfoot{}' + '\n')
        fsum.write(r'\begin{document}' + '\n')
    else:
        fsum.write(r'\clearpage' + '\n')
        fsum.write(r'\newpage' + '\n')

    # Get some values from the fitter
    start = fitter.toas.first_MJD.value
    finish = fitter.toas.last_MJD.value
    span = finish - start

    # Write beginning header info
    fsum.write(r'\section*{PSR ' + psr + '\markboth{' + psr + '}{}}\n')

    fsum.write(r'Summary generated on ' + time.ctime() \
            + ' by ' + check_output('whoami').strip().decode("utf-8")  \
            + r'\\' + '\n')
    fsum.write(r'Input files: \verb@' + parfile + r'@, ' \
            + r'\verb@' + fitter.toas.filename + r'@\\' + '\n')
    fsum.write('Span: %.1f years (%.1f -- %.1f)\\\\\n ' % (span/365.24,
        year(float(start)), year(float(finish))))

    if NB:
        avg_dict = fitter.resids.ecorr_average(use_noise_model=True)
        mjdlist = np.sort(avg_dict['mjds'].value)
    else:
        mjdlist = np.sort(fitter.toas.get_mjds().value)
    maxepoch = 6.5
    nepoch = 1
    m0 = mjdlist[0]
    for m in mjdlist:
        if m > m0+maxepoch:
            nepoch += 1
            m0 = m
    fsum.write('Epochs (defined as observations within %.1f-day spans): %d\\\\\n' % (maxepoch,nepoch))

    # Check if WB in not
    fsum.write('Wideband data: %s\n' %{False:'No',True:'Yes'}[not NB])
    if not NB:
        fsum.write('\\\\DMDATA: %s\n' %{False:'No',True:'Yes'}[has_dmdata])

    # Write out the timing model
    fsum.write(r'\subsection*{Timing model}' + '\n')
    fsum.write(r'\begin{verbatim}' + '\n')
    # Get the parfile lines
    parlines = fitter.model.as_parfile().split('\n')
    for l in parlines:
        if l.startswith('DMX'): continue
        fsum.write(l+"\n")
    fsum.write(r'\end{verbatim}' + '\n')

    # Write out the residual stats
    fsum.write(r'\subsection*{Residual stats}' + '\n')
    fsum.write(r'\begin{verbatim}' + '\n')
    rs_keys = rs_dict.keys()
    for k in rs_keys:
        l = "# WRMS(%s) = %.3f %s" %(k, rs_dict[k]['wrms'].value, rs_dict[k]['wrms'].unit)
        fsum.write(l + '\n')
        l = "#  RMS(%s) = %.3f %s" %(k, rs_dict[k]['rms'].value, rs_dict[k]['rms'].unit)
        fsum.write(l + '\n\n')
    fsum.write(r'\end{verbatim}' + '\n')

    # Write out DM residual stats if desired
    if not NB and dm_dict != None:
        fsum.write(r'\subsection*{DM Residual stats}' + '\n')
        fsum.write(r'\begin{verbatim}' + '\n')
        rs_keys = rs_dict.keys()
        for k in rs_keys:
            l = "# WRMS(%s) = %.6f %s" %(k, dm_dict[k]['wrms'].value, dm_dict[k]['wrms'].unit)
            fsum.write(l + '\n')
            l = "#  RMS(%s) = %.6f %s" %(k, dm_dict[k]['rms'].value, dm_dict[k]['rms'].unit)
            fsum.write(l + '\n\n')
        fsum.write(r'\end{verbatim}' + '\n')

    # Get lines to write for F-tests
    hdrline = "%42s %7s %9s %5s %s" % ("", "RMS(us)", "Chi2", "NDOF", "Ftest")
    ftest_lines = get_Ftest_lines(Ftest_dict, fitter)
    # Write F-test results
    fsum.write(r'\subsection*{Parameter tests}' + '\n')
    fsum.write("F-test results used PINT\n")
    fsum.write(r'\begin{verbatim}' + '\n')
    fsum.write(hdrline + '\n')
    for l in ftest_lines:
        fsum.write(l + '\n')
    fsum.write(r'\end{verbatim}' + '\n')

    # Write Epochs section
    fsum.write(r'\subsection*{Epochs near center of data span?}' + '\n')
    tmidspan = 0.5*(float(finish)+float(start))
    fsum.write('Middle of data span: midspan = %.0f\\\\\n' % (tmidspan))
    dtpepoch = float(fitter.model.PEPOCH.value)-tmidspan
    fsum.write('PEPOCH - midspan = $%.0f$ days = $%.1f$ years\\\\\n'  % ( dtpepoch, dtpepoch/365.24))
    if param_check('TASC', fitter, check_enabled=True):
        dttasc = float(fitter.model.TASC.value)-tmidspan
        fsum.write('TASC - midspan = $%.0f$ days = $%.1f$ years\\\\\n'  % ( dttasc, dttasc/365.24))
    if param_check('T0', fitter, check_enabled=True):
        dtt0 = float(fitter.model.T0.value)-tmidspan
        fsum.write('TASC - midspan = $%.0f$ days = $%.1f$ years\\\\\n'  % ( dtt0, dtt0/365.24))

    fsum.write('\n')

    # Write out if reduced chi squared is close to 1
    fsum.write(r'\subsection*{Reduced $\chi^2$ close to 1.00?}' + '\n')
    chi2_0 = Ftest_dict['initial']['chi2_test']
    ndof_0 = Ftest_dict['initial']['dof_test']
    rchi= chi2_0/ndof_0
    fsum.write('Reduced $\chi^2$ is %f/%d = %f\n' % (chi2_0,ndof_0,rchi))
    if rchi<0.95 or rchi>1.05:
        fsum.write('\\\\ Warning: $\chi^2$ is far from 1.00\n')

    # Check for more than one jumped receiver
    fsum.write(r'\subsection*{Receivers and JUMPs}' + '\n')
    groups = set(np.array(resids.toas.get_flag_value('f')[0]))
    receivers = set([g.replace("_GUPPI","").replace("_GASP","").replace("_PUPPI","").replace("_ASP","") for g in groups])

    jumped = []
    for p in fitter.model.params:
        if "JUMP" in p and "DM" not in p:
            jumped.append(getattr(fitter.model, p).key_value[0])
    if len(jumped)==0:
        print("WARNING: no JUMPs")
        jumped = ()
    nnotjumped = 0
    fsum.write('{\\setlength{\\topsep}{6pt}%\n\\setlength{\\partopsep}{0pt}%\n')  # starts a new environment
    fsum.write('\\begin{tabbing}\\hspace*{72pt}\\=\\kill\n')
    fsum.write('Receivers:\\\\[4pt]')
    for r in receivers:
        fsum.write('\n')
        fsum.write(r.replace("_","\\_"))
        if r in jumped:
            fsum.write('\\> JUMP')
        else:
            nnotjumped += 1
        fsum.write('\\\\')
    if len(receivers)>0:
        fsum.write('[4pt]')
    if nnotjumped==1:
        fsum.write('One non-JUMPed receiver.  Good.\\\\')
    else:
        fsum.write('Warning: %d non-JUMPed receivers.\\\\' % (nnotjumped,))
    fsum.write('\end{tabbing}\n')
    fsum.write('}\n\n')   # ends environment started above.

    # Check pulsar name
    fsum.write(r'\subsection*{Pulsar name check in .par file}' + '\n')
    fsum.write('Name in .par file: %s\\\\\n' % (fitter.model.PSR.value))
    if fitter.model.PSR.value.startswith("B") or fitter.model.PSR.value.startswith("J"):
        fsum.write('OK: starts with B or J\n')
    else:
        fsum.write('Warning: does not start with B or J\n')

    # Write if there are bad DMX ranges

    # NOTE - CURRENTLY CANNOT DO THIS, NEED DMX CHECKER FIRST

    fsum.write(r'\subsection*{Check for bad DMX ranges, less than 10\% bandwidth}' + '\n')
    """
    if not is_wideband:
        if len(baddmx)==0:
            fsum.write('No bad dmx ranges\\\\\n')
        else:
            fsum.write('Bad DMX ranges found, %d out of %d DMX ranges:\\\\\n' % (len(baddmx),ndmx))
            for l in baddmx:
                fsum.write('{\\tt '+l+'}\\\\\n')
    else:
        fsum.write('No fractional bandwidth check for DMX ranges with wideband data!\\\\\n')
    """
    if not NB:
        fsum.write('No fractional bandwidth check for DMX ranges with wideband data!\\\\\n')

    # Write out software versions used
    fsum.write(r'\subsection*{Software versions used in Analysis:}' + '\n')
    fsum.write('PINT: %s\\\\\n' % (pint.__version__))
    fsum.write('astropy: %s\\\\\n' % (astropy.__version__))
    fsum.write('numpy: %s\\\\\n' % (np.__version__))
    try:
        import enterprise
        fsum.write('enterprise: %s\\\\\n' % (enterprise.__version__))
    except:
        pass
    try:
        import PTMCMCSampler
        fsum.write('PTMCMCSampler: %s\\\\\n' % (PTMCMCSampler.__version__))
    except:
        pass
    try:
        psrchive_v = check_output(["psrchive", "--version"]).decode("utf-8")
        fsum.write('PSRCHIVE: %s\\\\\n' % (psrchive_v))
    except:
        pass
    
    # Write out the plots - Assuming we have already made the summary plot previous to this
    # TODO Fix the plots...
    plot_file_list = np.sort(glob.glob("%s*summary_plot_*" % (fitter.model.PSR.value)))
    for plot_file in plot_file_list:
        fsum.write(r'\begin{figure}[p]' + '\n')
        #fsum.write(r'\begin{center}' + '\n')
        #fsum.write(r'\vspace*{-2.0em}' + '\n')
        fsum.write(r'\centerline{\includegraphics[]{' + plot_file + '}}\n')
        #fsum.write(r'\end{center}' + '\n')
        fsum.write(r'\end{figure}' + '\n')

    if append is None:

        fsum.write(r'\end{document}' + '\n')
        fsum.close()

        os.system('pdflatex -interaction=batchmode '
                + texfile + ' 2>&1 > /dev/null')

def write_if_changed(filename, contents):
    """Write contents to filename, touching the file only if it does not already contain them.
    
    contents should be a string and the file will be a text file.
    """
    if os.path.exists(filename):
        if contents == open(filename).read():
            return
    with open(filename, "w") as f:
        f.write(contents)