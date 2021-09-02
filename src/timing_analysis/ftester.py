""" This is a set of utilities which performs F-test calculations for parameter inclusion/exclusion """


'''
fitter.ftest('list of PINT parameter objects', 'list of corresponding timing model components', remove=True/False, full_output = True) and this will output a dictionary with keys: ft, resid_rms_test, resid_wrms_test, chi2_test, dof_test, where the test values are of the nested modeled that was tested.

'''



import warnings
#from pint.models import (
#    parameter as p,
#)
import timing_analysis.PINT_parameters as pparams
from pint.models.timing_model import Component
import copy
import astropy.units as u
import numpy as np

ALPHA = 0.0027

def param_check(name, fitter, check_enabled=True):
    """
    Checks if a timing model parameter is in the timing model.
    
    Inputs:
    --------
    name [string]: Name of the timing model parameter to be checked.
    fitter [object]: The PINT fitter object.
    check_enaabled [boolean]: If True, check if the model parameter is fit for as well [default: True].
    
    Returns:
    --------
    [boolean]: Returns either True or False depending on if the parameters is in the model and/or enabled.
    
    """
    if not name in fitter.model.params:
        return False
    # If frozen is False, then it's fit for, if frozen is True, parameter not fit for
    if check_enabled and getattr(fitter.model, "{:}".format(name)).frozen:
        return False
    return True

def get_fblist(fitter):
    """
    Returns the list of FB parameters for the ELL1 model.

    Input:
    --------
    fitter [object]: The PINT fitter object.

    Returns:
    --------
    fblist [dictionary]: Dictionary of FB parameters currently in the timing model.
    """
    fblist = {}
    for p in fitter.model.params:
        if p.startswith("FB") and not p.startswith("FBJ") and getattr(fitter.model, p).value != None:
            if len(p)==2:
                fblist[0] = p
            else:
                fblist[int(p[2:])] = p
    fbbad = False
    k = sorted(fblist.keys())
    for i, ifb in zip(range(len(fblist)),sorted(fblist.keys())):
        if i!=ifb:
            fbbad = True
    if fbbad:
        print_bad("FB parameters not a series of integers: "+
             " ".join([fblist[i] for i in k]) )
    return fblist

def binary_params_ftest(bparams, fitter, remove):
    """
    Returns the list of parameters and components to put in the check for binary models.

    Input:
    --------
    bparams [list]: List of timing model binary parameters to check for.
    fitter [object]: The PINT fitter object.
    remove [boolean]: If True, check to see if parameters need to be removed, if False check to see
        what parameters need to be added.

    Returns:
    --------
    plist [list]: List of PINT parameters to iterate through for the F-tests.
    clist [list]: List of PINT components to iterate through for the F-tests.
    """
    # list of parameters as strings to remove or add in the F-test
    p_test = []
    # list of PINT parameters to test
    pint_params = []
    # And their associated timing model components
    pint_comps = []
    # get binary model to add to component
    b_ext = fitter.model.binary_model_name
    # If we want to test removing parameters from the binary model
    if remove:
        # Figure out what is in the model and needs to be removed
        for bp in bparams:
            if param_check(bp, fitter):
                p_test.append(bp)
    else:
        # Figure out what is in the model and needs to be removed
        for bp in bparams:
            if not param_check(bp, fitter):
                p_test.append(bp)
    # Check M2 SINI specifically
    if 'M2' in p_test and 'SINI' in p_test:
        pint_params.append([pparams.M2, pparams.SINI])
        pint_comps.append([pparams.M2_Component+b_ext, pparams.SINI_Component+b_ext])
    # Get the rest of the parameters and components
    for pr in p_test:
        if pr == 'M2' or pr == 'SINI' or pr == 'EPS1DOT' or pr == 'EPS2DOT':
            pass
        else:
            # Check for H3/H4 -> This may not be quite right, may need to change later
            if pr == 'H3':
                pass
            else:
                pint_params.append([getattr(pparams, pr)])
                pint_comps.append([getattr(pparams, pr+'_Component')+b_ext])
    # Check EPS1/2DOT specifically
    if 'EPS1DOT' in p_test and 'EPS2DOT' in p_test:
        pint_params.append([pparams.EPS1DOT, pparams.EPS2DOT])
        pint_comps.append([pparams.EPS1DOT_Component+b_ext, pparams.EPS2DOT_Component+b_ext])
    # Check H3 and H4 specifically
    if 'H3' in p_test and 'H4' in p_test:
        pint_params.append([pparams.H3, pparams.H4])
        pint_comps.append([pparams.H3_Component+b_ext, pparams.H4_Component+b_ext])
    # return the values
    return pint_params, pint_comps

def report_ptest(label, ftest_dict = None, alpha=ALPHA):
    """
    Nicely prints the results of F-tests in a human-readable format.
    
    Input:
    --------
    label [string]: Name of the parameter(s) that were added/removed for the F-test.
    ftest_dict [dictionary]: Dictionary of output values from the PINT `ftest()` function. If `None`, will
        print a line of NaNs for each reported value.
    alpha [float]: Value to compare for F-statistic significance. If the F-statistic is lower than alpha, 
        the timing model parameters are deemed statistically significant to the timing model.
    """
    # If F-test fails, print line of NaNs
    if ftest_dict == None:
        line = "%42s %7.3f %9.2f %5f %.3f" % (label, np.nan, np.nan, np.nan, np.nan)
    # Else print the computed values
    else:
        # Get values from input dictionary
        rms = ftest_dict['resid_wrms_test'].value # weighted root mean square of timing residuals
        chi2 = ftest_dict['chi2_test'] # chi-squared value of the fit of the F-tested model
        ndof = ftest_dict['dof_test'] # number of degrees of freedom in the F-tested model
        if "dm_resid_wrms_test" in ftest_dict.keys():
            dmrms = ftest_dict['dm_resid_wrms_test'].value # weighted root mean square of DM residuals
        else:
            dmrms = None
        Fstatistic=  ftest_dict['ft'] # F-statistic from the F-test comparison
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
    print(line)
    
def summarize_Ftest(Ftest_dict, fitter, alpha = ALPHA):
    """
    Function to determine and print what parameters are recommended to be added from F-test dictionary.

    Input:
    ----------
    Ftest_dict [dictionary]: Dictionary of F-test results output by the `run_Ftests()` function.
    fitter [object]: The PINT fitter object.
    alpha [float]: Value to compare for F-statistic significance. If the F-statistic is lower than alpha, 
        the timing model parameters are deemed statistically significant to the timing model.
    
    """
    add_params = []
    remove_params = []
    fd_add = []
    fd_add_ft = []
    fd_remove = []
    fd_remove_ft = []
    for fk in Ftest_dict.keys():
        if 'FB' in fk:
            try:
                fbmax = (int(max(Ftest_dict[fk].keys())[-1]))
            except (IndexError, ValueError):
                fbmax = (int(max(Ftest_dict[fk].keys())[-2]))
            fblist = get_fblist(fitter)
            fbused = (len(fblist)>0)
            fbp = [fblist[ifb] for ifb in sorted(fblist.keys())]  # sorted list of fb parameters
            # Remoeing FB parameters
            for i in range(1,len(fblist)):
                p = [fbp[j] for j in range(i,len(fbp))]
                ffk = 'FB%s+'%i
                if Ftest_dict[fk][ffk]['ft'] is not None:
                    if Ftest_dict[fk][ffk]['ft'] > alpha and Ftest_dict[fk][ffk]['ft']:
                        remove_params.append(ffk)
            # Adding FB parameters
            for i in range(len(fblist),fbmax+1):
                p = ["FB%d" % (j) for j in range(len(fblist),i+1)]
                ffk = 'FB%s'%i
                if Ftest_dict[fk][ffk]['ft'] is not None:
                    if Ftest_dict[fk][ffk]['ft'] <= alpha and Ftest_dict[fk][ffk]['ft']:
                        add_params.append(ffk)
        # Check which parameters should be removed
        elif "Remove" in fk:
            for ffk in Ftest_dict[fk].keys():
                if ffk == 'Binary':
                    for fffk in Ftest_dict[fk][ffk].keys():
                        if Ftest_dict[fk][ffk][fffk] is not None:
                            if Ftest_dict[fk][ffk][fffk]['ft'] > alpha and Ftest_dict[fk][ffk][fffk]['ft']:
                                remove_params.append(fffk)
                elif ffk == 'FD':
                    for fffk in Ftest_dict[fk][ffk].keys():
                        if Ftest_dict[fk][ffk][fffk] is not None:
                            if Ftest_dict[fk][ffk][fffk]['ft'] > alpha and Ftest_dict[fk][ffk][fffk]['ft']:
                                fd_remove.append(fffk)
                                fd_remove_ft.append(Ftest_dict[fk][ffk][fffk]['ft'])
                else:
                    if Ftest_dict[fk][ffk]['ft'] is not None:
                        if Ftest_dict[fk][ffk]['ft'] > alpha and Ftest_dict[fk][ffk]['ft']:
                            # Policy is never to remove parallax
                            if ffk != 'PX':
                                remove_params.append(ffk)
        # Check which parameters should be added
        elif "Add" in fk:
            for ffk in Ftest_dict[fk].keys():
                if ffk == 'Binary':
                    for fffk in Ftest_dict[fk][ffk].keys():
                        if Ftest_dict[fk][ffk][fffk] is not None:
                            if Ftest_dict[fk][ffk][fffk]['ft'] <= alpha and Ftest_dict[fk][ffk][fffk]['ft']:
                                add_params.append(fffk)
                elif ffk == 'FD':
                    for fffk in Ftest_dict[fk][ffk].keys():
                        if Ftest_dict[fk][ffk][fffk] is not None:
                            if Ftest_dict[fk][ffk][fffk]['ft'] <= alpha and Ftest_dict[fk][ffk][fffk]['ft']:
                                fd_add.append(fffk)
                                fd_add_ft.append(Ftest_dict[fk][ffk][fffk]['ft'])
                else:
                    if Ftest_dict[fk][ffk]['ft'] is not None:
                        if Ftest_dict[fk][ffk]['ft'] <= alpha and Ftest_dict[fk][ffk]['ft']:
                            add_params.append(ffk)
        # Policy is to never add additional spin derivatives in general
        elif fk == 'F':
            pass

    # Now return which parameters to add/remove
    if fd_remove:
        remove_params.append(fd_remove[np.where(fd_remove_ft==min(fd_remove_ft))[0][0]])
        remove_statement = "F-tests recommend removing the following parameters: " + " ".join(remove_params)
    else:
        remove_statement = "F-tests do not recommend removing any parameters."
    if fd_add:
        add_params.append(fd_add[np.where(fd_add_ft==min(fd_add_ft))[0][0]])
        add_statement = "F-tests recommend adding the following parameters: " + " ".join(add_params)
    else:
        add_statement = "F-tests do not recommend adding any parameters."
    return add_statement, remove_statement
    
def reset_params(params):
    """
    Resets parameter values to defaults as assigned in pint_parameters.py. Most are reset to zero.
    
    Inputs:
    --------
    params [list]: List of pint parameters to reset the values of within the model.
    """
    for p in params:
        if p.name == 'M2':
            p.value = 0.25
            p.uncertainty = 0.0
        elif p.name == 'SINI':
            p.value = 0.8
            p.uncertainty = 0.0
        else:
            p.value = 0.0
            p.uncertainty = 0.0

def run_Ftests(fitter, alpha=ALPHA, FDnparams = 5, NITS = 1):
    """
    This is the main convenience function to run the various F-tests below. This includes F-tests for F2, PX, 
    binary parameters, FD parameters, etc. As part of the function, the tests, parameters, RMS of residuals, chi2,
    degrees of freedom, F-statistic values are all printed in a nice human readable format.

    Input:
    --------
    fitter [object]: The PINT fitter object.
    alpha [float]: The F-test significance value. If the F-statistic is lower than alpha, 
        the timing model parameters are deemed statistically significant to the timing model [default: 0.0027].
    FDnparams [int]: Maximum number of FD parameters to test [default: 5].
    NITS [int]: Number of fit iterations to run during FD parameter F-tests when adding FD parameters to the
        timing model after each F-test is run. Should only need to be increased if FD parameter F-tests do
        not appear to be converged [default: 1].
        
    Returns:
    ---------
    retdict [dictionary]: A nested dictionary of all of the different F-tests done and subsequent reported values for each
        F-test. These include keys for added ['Add'] or removed ['Remove'] parameters, the initial values ['initial']
        and within each of those, further nested dictionaries of parameters [e.g. 'PX'], and the reported values.
    """
    # Check if fitter is wideband or not
    if fitter.is_wideband:
        NB = False
    #    resids = fitter.resids.residual_objs['toa']
    #    dm_resids = fitter.resids.residual_objs['dm']
    else:
        NB = True
    #    resids = fitter.resids
    resids = fitter.resids

    # Define return dictionary, format tbd
    retdict = {}
    # Start with initial printing from Finalize Timing:
    print("Testing additional parameters (*** = significant):")
    if NB:
        hdrline = "%42s %7s %9s %5s %s" % ("", "RMS(us)", "Chi2", "NDOF", "Ftest")
    else:
        hdrline = "%42s %7s %9s %9s %5s %s" % ("", "RMS(us)", "DM RMS(pc cm^-3)", "Chi2", "NDOF", "Ftest")
    print(hdrline)
    if NB:
        base_rms = resids.time_resids.std().to(u.us)
        base_wrms = resids.rms_weighted() # assumes the input fitter has been fit already
    else:
        base_rms = resids.residual_objs['toa'].time_resids.std().to(u.us)
        base_wrms = resids.residual_objs['toa'].rms_weighted() # assumes the input fitter has been fit already
    base_chi2 = resids.chi2
    base_ndof = resids.dof
    if NB:
        # Add initial values to F-test dictionary
        retdict['initial'] = {'ft':None, 'resid_rms_test':base_rms, 'resid_wrms_test':base_wrms, 'chi2_test':base_chi2, 'dof_test':base_ndof}
    else:
        dm_resid_rms_test = fitter.resids.residual_objs['dm'].resids.std()
        dm_resid_wrms_test = fitter.resids.residual_objs['dm'].rms_weighted()
        # Add initial values to F-test dictionary
        retdict['initial'] = {'ft':None, 'resid_rms_test':base_rms, 'resid_wrms_test':base_wrms, 'chi2_test':base_chi2, 'dof_test':base_ndof, "dm_resid_rms_test": dm_resid_rms_test, "dm_resid_wrms_test": dm_resid_wrms_test}
    # Now report the values
    report_ptest("initial", retdict['initial'])
    # Check adding binary parameters
    print("Testing additional parameters:")
    retdict['Add'] = {}
    if hasattr(fitter.model, "binary_model_name"):
        if fitter.model.binary_model_name == 'DD' or fitter.model.binary_model_name == 'BT':
            binarydict = check_binary_DD(fitter, alpha=ALPHA, remove = False, NITS=NITS)
        elif fitter.model.binary_model_name == 'DDK':
            binarydict = check_binary_DDK(fitter, alpha=ALPHA, remove = False, NITS=NITS)
        elif fitter.model.binary_model_name == 'ELL1':
            binarydict = check_binary_ELL1(fitter, alpha=ALPHA, remove = False, NITS=NITS)
        elif fitter.model.binary_model_name == 'ELL1H':
            binarydict = check_binary_ELL1H(fitter, alpha=ALPHA, remove = False, NITS=NITS)
        retdict['Add']['Binary'] = binarydict
    # Test adding FD parameters
    FDdict = check_FD(fitter, alpha=ALPHA, remove = False, maxcomponent=FDnparams, NITS = NITS)
    retdict['Add']['FD'] = FDdict
    print("Testing removal of parameters:")
    retdict['Remove'] = {}
    # Check parallax, NOTE - cannot remove PX if binary model is DDK, so check for that.
    if hasattr(fitter.model, "binary_model_name"):
        if fitter.model.binary_model_name == 'DDK':
            print(" PX, KOM and KIN cannot be removed in DDK.")
        else:
            PXdict = check_PX(fitter, alpha=ALPHA, NITS=NITS)
            retdict['Remove']['PX'] = PXdict['PX']
    else:
        PXdict = check_PX(fitter, alpha=ALPHA, NITS=NITS)
        retdict['Remove']['PX'] = PXdict['PX']
    # Check removing binary parameters
    if hasattr(fitter.model, "binary_model_name"):
        if fitter.model.binary_model_name == 'DD' or fitter.model.binary_model_name == 'BT':
            binarydict = check_binary_DD(fitter, alpha=ALPHA, remove = True, NITS=NITS)
        elif fitter.model.binary_model_name == 'DDK':
            binarydict = check_binary_DDK(fitter, alpha=ALPHA, remove = True, NITS=NITS)
        elif fitter.model.binary_model_name == 'ELL1':
            binarydict = check_binary_ELL1(fitter, alpha=ALPHA, remove = True, NITS=NITS)
        elif fitter.model.binary_model_name == 'ELL1H':
            binarydict = check_binary_ELL1H(fitter, alpha=ALPHA, remove = True, NITS=NITS)
        retdict['Remove']['Binary'] = binarydict
    # Test removing FD parameters
    FDdict = check_FD(fitter, alpha=ALPHA, remove = True, maxcomponent=FDnparams, NITS = NITS)
    retdict['Remove']['FD'] = FDdict
    # Get current number of spin frequency derivatives
    current_freq_deriv = 1
    for i in range(2,21):
        p = "F%d" % i
        if p in fitter.model.params:
            current_freq_deriv = i
    print("Testing spin freq derivs (%s enabled):" % (current_freq_deriv))
    # NOTE - CURRENTLY ONLY TESTS F2
    F2dict = check_F2(fitter, alpha=ALPHA, NITS=NITS)
    retdict['F'] = F2dict
    # Now check FB parameters
    if hasattr(fitter.model, "binary_model_name"):
        if fitter.model.binary_model_name == 'ELL1':
            fblist = get_fblist(fitter)
            if fblist:
                FBdict = check_FB(fitter, alpha=ALPHA, fbmax = 5, NITS=NITS)
                retdict['FB'] = FBdict
    
    # Print a summary of the F-tests results and suggestions
    add_statement, remove_statement = summarize_Ftest(retdict, fitter, alpha = ALPHA)
    print(add_statement)
    print(remove_statement)

    return retdict


def check_F2(fitter, alpha=ALPHA, NITS=1):
    """
    Check the significance of F2 with an F-test.
    In general, we do not allow F2 to be added but we will still check as in the past.

    Input:
    --------
    fitter [object]: The PINT fitter object.
    alpha [float]: The F-test significance value. If the F-statistic is lower than alpha, 
        the timing model parameters are deemed statistically significant to the timing model [default: 0.0027].

    Returns:
    --------
    retdict [dictionary]: Returns the dictionary output from the F-tests.
    """
    # Add dictionary for return values
    retdict = {}
    # Run the F2 F-test
    try:
        ftest_dict = fitter.ftest(pparams.F2, pparams.F2_Component, remove=False, full_output=True, maxiter=NITS)
    # If there's an error running the F-test in the fit for some reason, we catch it
    except ValueError as e:
        warnings.warn(f"Error when running F-test for F2: {e}")
        ftest_dict = None
    # Add to dictionary
    retdict['F2'] = ftest_dict
    # Report the values
    report_ptest('F2', ftest_dict, alpha = alpha)
    # This edits the values in the file for some reason, want to reset them to zeros
    reset_params([pparams.F2])
    # Return the dictionary
    return retdict


def check_PX(fitter, alpha=ALPHA, NITS=1):
    """
    Check the significance of PX with an F-test.
    In general, we fit PX but we still wish to test for this as in the past.

    Input:
    --------
    fitter [object]: The PINT fitter object.
    alpha [float]: The F-test significance value. If the F-statistic is lower than alpha, 
        the timing model parameters are deemed statistically significant to the timing model [default: 0.0027].

    Returns:
    --------
    retdict [dictionary]: Returns the dictionary output from the F-tests.
    """
    # Add dictionary for return values
    retdict = {}
    # Run the parallax F-test
    try:
        ftest_dict = fitter.ftest(pparams.PX, pparams.PX_Component, remove=True, full_output=True, maxiter=NITS)
    # If there's an error running the F-test in the fit for some reason, we catch it
    except ValueError as e:
        warnings.warn(f"Error when running F-test for PX: {e}")
        ftest_dict = None
    # Add to dictionary
    retdict['PX'] = ftest_dict
    # Report the values
    report_ptest('PX', ftest_dict, alpha = alpha)
    # This edits the values in the file for some reason, want to reset them to zeros
    reset_params([pparams.PX])
    # Return the dictionary
    return retdict

def check_FD(fitter, alpha=ALPHA, remove=False, maxcomponent=5, NITS = 1):
    """
    Check adding FD parameters with an F-test.

    Input:
    --------
    fitter [object]: The PINT fitter object.
    alpha [float]: The F-test significance value. If the F-statistic is lower than alpha, 
        the timing model parameters are deemed statistically significant to the timing model [default: 0.0027].
    remove [boolean]: If True, will do and report F-test values for removing parameters.
        If False, will look for and report F-test values for adding parameters [default: False].
    maxcomponent [int]: Maximum number of FD parameters to add to the model [default: 5]. If remove=True,
        This parameter is ignored.
    NITS [int]: Number of fit iterations to run when adding FD parameters to the timing model after 
        each F-test is run. Should only need to be increased if FD parameter F-tests do not appear 
        to be converged [default: 1].

    Returns:
    --------
    retdict [dictionary]: Returns the dictionary output from the F-tests.
    """
    # Print how many FD currently enabled
    cur_fd = [param for param in fitter.model.free_params if "FD" in param]
    if remove:
        print("Testing removing FD terms (", cur_fd, "enabled):")
    else:
        print("Testing adding FD terms (", cur_fd, "enabled):")
    
    fitter_fd = copy.deepcopy(fitter)
    
    # Check if timing model has FD component in it (includes wideband models)
    if "FD" not in fitter.model.components.keys():
        try:
            all_components = Component.component_types
            fd_class = all_components["FD"]
            fd = fd_class()
            fitter_fd.model.add_component(fd, validate=False)
        except ValueError:
            warnings.warn("FD Component already in timing model.")
    # Add dictionary for return values
    retdict = {}
    
    # Get list of FD parameters to add or remove
    param_list = []
    component_list = []
    if remove:
        for i in range(len(cur_fd), 0, -1):
            param_list.append([getattr(pparams, f'FD{q}') for q in range(len(cur_fd),i-1,-1)])
            component_list.append([getattr(pparams, f"FD{q}_Component") for q in range(len(cur_fd),i-1,-1)])
    else:
        for i in range(len(cur_fd)+1, maxcomponent+1):
            param_list.append([getattr(pparams, f'FD{q}') for q in range(len(cur_fd)+1,i+1)])
            component_list.append([getattr(pparams, f"FD{q}_Component") for q in range(len(cur_fd)+1,i+1)])
    for i in range(len(param_list)):
        d_label = ""
        for fd in param_list[i]:
            d_label += f"{fd.name}, "
        d_label = d_label[:-2]
        # Run F-test
        try:
            ftest_dict = fitter_fd.ftest(param_list[i], component_list[i], maxiter=NITS, remove=remove, full_output=True)
        # If there's an error running the F-test in the fit for some reason, we catch it
        except ValueError as e:
            warnings.warn(f"Error when running F-test for {d_label}: {e}")
            ftest_dict = None
        # Add to dictionary to return
        retdict[d_label] = ftest_dict
        # Report the values
        #report_ptest('FD1 through FD%s'%i, ftest_dict, alpha = alpha)
        report_ptest(d_label, ftest_dict, alpha = alpha)
        # This edits the values in the file for some reason, want to reset them to zeros
        reset_params(param_list[i])

    # Return the dictionary
    return retdict

def check_binary_DD(fitter, alpha=ALPHA, remove = False, NITS=1):
    """
    Check the binary parameter F-tests for the DD binary model, either removing or adding parameters.

    Input:
    --------
    fitter [object]: The PINT fitter object.
    alpha [float]: The F-test significance value. If the F-statistic is lower than alpha, 
        the timing model parameters are deemed statistically significant to the timing model [default: 0.0027].
    remove [boolean]: If True, will do and report F-test values for removing parameters.
             If False, will look for and report F-test values for adding parameters [default: False].
             Parameters to check:
                1. M2, SINI
                2. PBDOT
                3. XDOT -> A1DOT
                4. OMDOT
                5. EDOT

    Returns:
    ---------
    retdict [dictionary]: Returns the dictionary output from the F-tests.
    """
    # Add dictionary for return values
    retdict = {}
    # Params to check
    DDparams = ['M2', 'SINI', 'PBDOT', 'A1DOT', 'OMDOT' ,'EDOT']
    # Get the components and run the F-test
    # get list of parameters for F-tests
    pint_params, pint_comps = binary_params_ftest(DDparams, fitter, remove)
    # Now get the list of components and parameters to run the F-test; Check M2 SINI specifically
    for ii in range(len(pint_params)):
        try:
            ftest_dict = fitter.ftest(pint_params[ii], pint_comps[ii], remove=remove, full_output=True, maxiter=NITS)
        except ValueError as e:
            warnings.warn(f"Error when running F-test for {[p.name for p in pint_params[ii]]}: {e}")
            ftest_dict = None
        # Get dictionary label
        if len(pint_params[ii]) > 1:
            d_label = "M2, SINI"
        else:
            d_label = pint_params[ii][0].name
        # Add the dictionary
        retdict[d_label] = ftest_dict
        # Report the values
        report_ptest(d_label, ftest_dict, alpha = alpha)
        # Reset the parameters
        reset_params(pint_params[ii])
    # Return the dictionary
    return retdict

def check_binary_DDK(fitter, alpha=ALPHA, remove = False, NITS=1):
    """
    Check the binary parameter F-tests for the DDK binary model, either removing or adding parameters.

    Input:
    --------
    fitter [object]: The PINT fitter object.
    alpha [float]: The F-test significance value. If the F-statistic is lower than alpha, 
        the timing model parameters are deemed statistically significant to the timing model [default: 0.0027].
    remove [boolean: If True, will do and report F-test values for removing parameters.
             If False, will look for and report F-test values for adding parameters [default: False].
             Parameters to check:
                1. M2, SINI
                2. PBDOT
                3. XDOT -> A1DOT
                4. OMDOT
                5. EDOT

    Returns:
    ---------
    retdict [dictionary]: Returns the dictionary output from the F-tests.
    """
    # Add dictionary for return values
    retdict = {}
    # Params to check
    DDKparams = ['M2', 'SINI', 'PBDOT', 'A1DOT', 'OMDOT' ,'EDOT']
    # Get the components and run the F-test
    # get list of parameters for F-tests
    pint_params, pint_comps = binary_params_ftest(DDKparams, fitter, remove)
    # Now get the list of components and parameters to run the F-test; Check M2 SINI specifically
    for ii in range(len(pint_params)):
        try:
            ftest_dict = fitter.ftest(pint_params[ii], pint_comps[ii], remove=remove, full_output=True, maxiter=NITS)
        except ValueError as e:
            warnings.warn(f"Error when running F-test for {[p.name for p in pint_params[ii]]}: {e}")
            ftest_dict = None
        # Get dictionary label
        if len(pint_params[ii]) > 1:
            d_label = "M2, SINI"
        else:
            d_label = pint_params[ii][0].name
        # Add the dictionary
        retdict[d_label] = ftest_dict
        # Report the values
        report_ptest(d_label, ftest_dict, alpha = alpha)
        # Reset the parameters
        reset_params(pint_params[ii])
    # Return the dictionary
    return retdict


def check_binary_ELL1(fitter, alpha=ALPHA, remove = False, NITS=1):
    """
    Check the binary parameter F-tests for the ELL1 binary model, either removing or adding parameters.

    Input:
    --------
    fitter [object]: The PINT fitter object.
    alpha [float]: The F-test significance value. If the F-statistic is lower than alpha, 
        the timing model parameters are deemed statistically significant to the timing model [default: 0.0027].
    remove [boolean]: If True, will do and report F-test values for removing parameters.
             If False, will look for and report F-test values for adding parameters [default: False].
             Parameters to check:
                1. PB
                    a. M2, SINI
                    b. PBDOT
                    c. XDOT -> A1DOT
                    d. EPS1DOT, EPS2DOT
                2. FB0, FB1
                    a. FB2-FB6+ ( These are checked by running `check_FB()` )
                    b. XDOT -> A1DOT
                    c. EPS1DOT, EPS2DOT

    Returns:
    ---------
    retdict [dictionary]: Returns the dictionary output from the F-tests.
    """
    # Define dictionary for returned value
    retdict = {}
    # Check if FB is used (Use the same function as in previous finalize time, but PINT-ified
    fblist = get_fblist(fitter)
    fbused = (len(fblist)>0)
    # Get the parameters that are used (not including FB values)
    if not fbused:
        ELL1params = ['M2', 'SINI', 'PBDOT', 'A1DOT', 'EPS1DOT' ,'EPS2DOT']
    else:
        ELL1params = ['A1DOT', 'EPS1DOT' ,'EPS2DOT']
    # Check not using FB first
    if not fbused:
        # get list of parameters for F-tests
        pint_params, pint_comps = binary_params_ftest(ELL1params, fitter, remove)
        # Now get the list of components and parameters to run the F-test; Check M2 SINI specifically
        for ii in range(len(pint_params)):
            try:
                ftest_dict = fitter.ftest(pint_params[ii], pint_comps[ii], remove=remove, full_output=True, maxiter=NITS)
            except ValueError as e:
                warnings.warn(f"Error when running F-test for {[p.name for p in pint_params[ii]]}: {e}")
                ftest_dict = None
            # Get dictionary label
            if len(pint_params[ii]) > 1 and (pint_params[ii][0].name == 'M2' or pint_params[ii][0].name == 'SINI'):
                d_label = "M2, SINI"
            elif len(pint_params[ii])>1 and (pint_params[ii][0].name == 'EPS1DOT' or pint_params[ii][0].name == 'EPS2DOT'):
                d_label = "EPS1DOT, EPS2DOT"
            else:
                d_label = pint_params[ii][0].name
            # Add the dictionary
            retdict[d_label] = ftest_dict
            # Report the values
            report_ptest(d_label, ftest_dict, alpha = alpha)
            # Reset the parameters
            reset_params(pint_params[ii])
        # Return the dictionary
        return retdict
    # If we use FB parameters, check that, this will be more like the FD parameters
    else:
        # Now check other parameters, not FB
        pint_params, pint_comps = binary_params_ftest(ELL1params, fitter, remove)
        # Now get the list of components and parameters to run the F-test; Check M2 SINI specifically
        for ii in range(len(pint_params)):
            try:
                ftest_dict = fitter.ftest(pint_params[ii], pint_comps[ii], remove=remove, full_output=True, maxiter=NITS)
            except ValueError as e:
                warnings.warn(f"Error when running F-test for {[p.name for p in pint_params[ii]]}: {e}")
                ftest_dict = None
            # Get dictionary label
            if len(pint_params[ii]) > 1 and (pint_params[ii][0].name == 'M2' or pint_params[ii][0].name == 'SINI'):
                d_label = "M2, SINI"
            elif len(pint_params[ii])>1 and (pint_params[ii][0].name == 'EPS1DOT' or pint_params[ii][0].name == 'EPS2DOT'):
                d_label = "EPS1DOT, EPS2DOT"
            else:
                d_label = pint_params[ii][0].name
            # Add the dictionary
            retdict[d_label] = ftest_dict
            # Report the values
            report_ptest(d_label, ftest_dict, alpha = alpha)
            # Reset the parameters
            reset_params(pint_params[ii])
        # Return the dictionary
        return retdict

def check_binary_ELL1H(fitter, alpha=ALPHA, remove = False, NITS=1):
    """
    Check the binary parameter F-tests for the ELL1H binary model, either removing or adding parameters.

    Input:
    --------
    fitter [object]: The PINT fitter object.
    alpha [float]: The F-test significance value. If the F-statistic is lower than alpha, 
        the timing model parameters are deemed statistically significant to the timing model [default: 0.0027].
    remove [boolean]: If True, will do and report F-test values for removing parameters.
             If False, will look for and report F-test values for adding parameters [default: False].
             Parameters to check:
                1. PBDOT
                2. XDOT -> A1DOT
                3. EPS1DOT, EPS2DOT
                4. H3
                    a. H4

    Returns:
    ---------
    retdict [dictionary]: Returns the dictionary output from the F-tests.
    """
    # Define dictionary for returned value
    retdict = {}
    # Get the parameters that are used
    ELL1Hparams = ['PBDOT', 'A1DOT', 'EPS1DOT' ,'EPS2DOT', 'H3', 'H4']
    # get list of parameters for F-tests
    pint_params, pint_comps = binary_params_ftest(ELL1Hparams, fitter, remove)
    # Now get the list of components and parameters to run the F-test; Check M2 SINI specifically
    for ii in range(len(pint_params)):
        try:
            ftest_dict = fitter.ftest(pint_params[ii], pint_comps[ii], remove=remove, full_output=True, maxiter=NITS)
        except ValueError as e:
            warnings.warn(f"Error when running F-test for {[p.name for p in pint_params[ii]]}: {e}")
            ftest_dict = None
        # Get dictionary label
        if len(pint_params[ii]) > 1 and (pint_params[ii][0].name == 'H3' or pint_params[ii][0].name == 'H4'):
            d_label = "H3, H4"
        elif len(pint_params[ii])>1 and (pint_params[ii][0].name == 'EPS1DOT' or pint_params[ii][0].name == 'EPS2DOT'):
            d_label = "EPS1DOT, EPS2DOT"
        else:
            d_label = pint_params[ii][0].name
        # Add the dictionary
        retdict[d_label] = ftest_dict
        # Report the values
        report_ptest(d_label, ftest_dict, alpha = alpha)
        # Reset the parameters
        reset_params(pint_params[ii])
    # Return the dictionary
    return retdict

def check_FB(fitter, alpha=ALPHA, fbmax = 5, NITS=1):
    """
    Check the FB parameter F-tests for the ELL1 binary model doing both removing and addtion.

    Input:
    --------
    fitter [object]: The PINT fitter object.
    alpha [float]: The F-test significance value. If the F-statistic is lower than alpha, 
        the timing model parameters are deemed statistically significant to the timing model [default: 0.0027].
    fbmax [int]: Number of FB parameters to check in the F-tests [default: 5].

    Returns:
    ---------
    retdict [dictionary]: Returns the dictionary output from the F-tests.
    """
    # Define dictionary for returned value
    retdict = {}
    # Check if FB is used (Use the same function as in previous finalize time, but PINT-ified
    fblist = get_fblist(fitter)
    fbused = (len(fblist)>0)
    if fbused:
        # First check FB parameters - print the corrent number of FB parameters used
        fbp = [fblist[ifb] for ifb in sorted(fblist.keys())]  # sorted list of fb parameters
        print("Testing FB parameters, present list: "+" ".join(fbp))
        # Now run the F-tests, first try removing FB parmaters
        print("Testing removal of FB parameters:")
        for i in range(1,len(fblist)):
            p = [fbp[j] for j in range(i,len(fbp))]
            param_list = [(getattr(pparams, fp)) for fp in p]
            component_list = [(getattr(pparams, "%s_Component"%(fp))+"ELL1") for fp in p]
            # Run F-test
            try:
                ftest_dict = fitter.ftest(param_list, component_list, remove=True, full_output=True, maxiter=NITS)
            except ValueError as e:
                warnings.warn(f"Error when running F-test for {[p.name for p in pint_params[ii]]}: {e}")
                ftest_dict = None
            # Add to dictionary to return
            retdict['FB%s+'%i] = ftest_dict
            # Report the values
            report_ptest(" ".join(p), ftest_dict, alpha = alpha)
            # This edits the values in the file for some reason, want to reset them to zeros
            reset_params(param_list)
    # Now try adding FB parameters
        print("Testing addition of FB parameters:")
        for i in range(len(fblist),fbmax+1):
            p = ["FB%d" % (j) for j in range(len(fblist),i+1)]
            param_list = [(getattr(pparams, fp)) for fp in p]
            component_list = [(getattr(pparams, "%s_Component"%(fp))+"ELL1") for fp in p]
            # Run F-test
            try:
                ftest_dict = fitter.ftest(param_list, component_list, remove=False, full_output=True, maxiter=NITS)
            except ValueError as e:
                warnings.warn(f"Error when running F-test for {[p.name for p in pint_params[ii]]}: {e}")
                ftest_dict = None
            # Add to dictionary to return
            retdict['FB%s'%i] = ftest_dict
            # Report the values
            report_ptest(" ".join(p), ftest_dict, alpha = alpha)
            # This edits the values in the file for some reason, want to reset them to zeros
            reset_params(param_list)
        # Now return the dictionary
        return retdict
    else:
        warnings.warn("No FB parameters in the initial timing model...")
        return False

    
