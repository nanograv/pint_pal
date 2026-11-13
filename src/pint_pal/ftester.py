""" This is a set of utilities which performs F-test calculations for parameter inclusion/exclusion

The test fits are set up and run by `pint_pal.parallel_fit.TestFitter`, on one or more CPUs. Each test adds or removes
parameters from the timing model, and is compared with a baseline fit of the unchanged model. The results are returned
in a nested dictionary in which each test has the keys: ft, chi2_test, dof_test.
"""



from loguru import logger as log
import pint.fitter
from pint.utils import FTest
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pint_pal.timingconfiguration import TimingConfiguration
from pint_pal.parallel_fit import TestFitter
import numpy as np
import re

ALPHA = 0.0027 # 3-sigma
ALPHA_ADD = 0.0027 # 3-sigma
ALPHA_REMOVE = 0.1336 # 1.5-sigma

def param_check(
    name: str,
    fitter: pint.fitter.Fitter,
    check_enabled: bool = True
) -> bool:
    """
    Checks if a timing model parameter is in the timing model.
    
    Parameters
    ----------
    name: str
        Name of the timing model parameter to be checked.
    fitter : pint.fitter.Fitter
        The PINT Fitter object.
    check_enabled: bool
        If True, check if the model parameter is fit for as well [default: True].
    
    Returns
    -------
    bool
        Returns either True or False depending on if the parameters is in the model and/or enabled.
    """
    
    if not name in fitter.model.params:
        return False
    # If frozen is False, then it's fit for, if frozen is True, parameter not fit for
    if check_enabled and getattr(fitter.model, "{:}".format(name)).frozen:
        return False
    return True

def get_fblist(fitter: pint.fitter.Fitter) -> dict:
    """
    Returns the list of FB parameters for the ELL1 model.

    Parameters
    ----------
    fitter : pint.fitter.Fitter
        The PINT Fitter object.

    Returns
    -------
    dict
        Dictionary of FB parameters currently in the timing model.
    """
    fblist = {}
    for p in fitter.model.params:
        if p.startswith("FB") and getattr(fitter.model, p).value != None:
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
        log.warning("FB parameters not a series of integers: "+
                    " ".join([fblist[i] for i in k]) )
    return fblist

def report_ptest(
    label: str,
    ftest_dict: dict | None = None,
    alpha: float = ALPHA
) -> str:
    """
    Nicely prints the results of F-tests in a human-readable format.
    
    Parameters
    ----------
    label: str
        Name of the parameter(s) that were added/removed for the F-test.
    ftest_dict: dict
        Dictionary of output values from the F-test. If `None`, will print a line of NaNs for each reported value.
    alpha: float
        Value to compare for F-statistic significance. If the F-statistic is lower than alpha,
        the timing model parameters are deemed statistically significant to the timing model.
    """
    # If F-test fails, print line of NaNs
    if ftest_dict == None:
        line = "%42s %9.2f %5f %.3f" % (label, np.nan, np.nan, np.nan)
    # Else print the computed values
    else:
        # Get values from input dictionary
        chi2 = ftest_dict['chi2_test'] # chi-squared value of the fit of the F-tested model
        ndof = ftest_dict['dof_test'] # number of degrees of freedom in the F-tested model
        Fstatistic = ftest_dict['ft'] # F-statistic from the F-test comparison
        line = "%42s %9.2f %5d" % (label, chi2, ndof)
        if Fstatistic is None:
            line += " --"
        elif not np.isnan(Fstatistic):
            line += " %.3g" % Fstatistic
            if Fstatistic < alpha:
                line += " ***"
        else:
            line += " xxx"
    print(line)
    return line
   
def checkFLogs(
    add_params: list,
    remove_params: list,
    tc: "TimingConfiguration"
) -> tuple[list, list]:
    """
    Checks config file for previous F-test recommendations to avoid looping.

    Parameters
    ----------
    add_params : list
        List of parameters recommended to be added.
    remove_params : list
        List of parameters recommended to be removed.
    tc : pint_pal.timingconfiguration.TimingConfiguration
        The pint_pal TimingConfiguration object.

    Returns
    -------
    (list, list)
        Sets of added, removed parameter recommendations
    """
    if add_params:
        add_params = [item.strip() for item in add_params[0].split(',')]
    else:
        add_params = []

    if remove_params:
        remove_params = [item.strip() for item in remove_params[0].split(',')]
    else:
        remove_params = []
    
    if 'FTest' in tc.config:
        oldFtests = tc.config['FTest']
        
        # - Compare old results with new to see if duplicates or loop occur
        for item in sorted(add_params):
            if item in oldFtests['Remove']:
                log.warning(f'Parameter {item} has already been recommended to be removed.')
                add_params.remove(item)
        for item in sorted(remove_params):
            if item in oldFtests['Add']:
                log.warning(f'Parameter {item} has already been recommended to be added.')
                remove_params.remove(item)

    else:
        print("No FTest in config.  Continuing...")

    return sorted(set(add_params)), sorted(set(remove_params))

def summarize_Ftest(
    Ftest_dict: dict,
    fitter: pint.fitter.Fitter,
    tc: "TimingConfiguration",
    alpha_add: float = ALPHA_ADD,
    alpha_remove: float = ALPHA_REMOVE
) -> tuple[str, str]:
    """
    Function to determine and print what parameters are recommended to be added from F-test dictionary.

    Parameters
    ----------
    Ftest_dict : dict
        Dictionary of F-test results output by the `run_Ftests()` function.
    fitter : pint.fitter.Fitter
        The PINT Fitter object.
    tc : pint_pal.timingconfiguration.TimingConfiguration
        The pint_pal TimingConfiguration object.
    alpha_add : float
        Value to compare for F-statistic significance when adding a parameter. If the F-statistic is lower than alpha_add, the timing model parameters are deemed statistically significant to (add to) the timing model.
    alpha_remove : float
        Value to compare for F-statistic significance when removing a parameter. If the F-statistic is lower than alpha_remove, the timing model parameters are deemed statistically significant to remove from timing model.

    Returns
    -------
    (str, str)
        Add, Remove text statements
    """
    
    add_params = []
    remove_params = []
    fd_add = []
    fd_add_ft = []
    fd_remove = []
    fd_remove_ft = []
    for fk in Ftest_dict.keys():
        if 'FB' in fk:
            fbmax = max([int(re.match(r"FB(\d+)\+?", k).group(1)) for k in Ftest_dict[fk]])
            fblist = get_fblist(fitter)
            fbused = (len(fblist)>0)
            fbp = [fblist[ifb] for ifb in sorted(fblist.keys())]  # sorted list of fb parameters
            # Removing FB parameters
            for i in range(1,len(fblist)):
                p = [fbp[j] for j in range(i,len(fbp))]
                ffk = 'FB%s+'%i
                if Ftest_dict[fk][ffk] is not None:
                    if Ftest_dict[fk][ffk]['ft'] is not None:
                        if Ftest_dict[fk][ffk]['ft'] > alpha_remove:
                            remove_params.append(ffk)
            # Adding FB parameters
            for i in range(len(fblist),fbmax+1):
                p = ["FB%d" % (j) for j in range(len(fblist),i+1)]
                ffk = 'FB%s'%i
                if Ftest_dict[fk][ffk] is not None:
                    if Ftest_dict[fk][ffk]['ft'] is not None:
                        if Ftest_dict[fk][ffk]['ft'] <= alpha_add:
                            add_params.append(ffk)
        # Check which parameters should be removed
        elif "Remove" in fk:
            for ffk in Ftest_dict[fk].keys():
                if ffk == 'Binary':
                    for fffk in Ftest_dict[fk][ffk].keys():
                        if Ftest_dict[fk][ffk][fffk] is not None:
                            if Ftest_dict[fk][ffk][fffk]['ft'] > alpha_remove:
                                remove_params.append(fffk)
                elif ffk == 'FD':
                    for fffk in Ftest_dict[fk][ffk].keys():
                        if Ftest_dict[fk][ffk][fffk] is not None:
                            if Ftest_dict[fk][ffk][fffk]['ft'] > alpha_remove:
                                fd_remove.append(fffk)
                                fd_remove_ft.append(Ftest_dict[fk][ffk][fffk]['ft'])
                else:
                    if Ftest_dict[fk][ffk] is not None: # This is added because some parameters fail ftest due to max convergence limit and hence are defaulted to None
                        if Ftest_dict[fk][ffk]['ft'] is not None:
                            if Ftest_dict[fk][ffk]['ft'] > alpha_remove:
                                # Policy is never to remove parallax
                                if ffk != 'PX':
                                    remove_params.append(ffk)
                    elif Ftest_dict[fk][ffk] is None:
                        print(f"Due to max iter limit, {ffk} hasn't converged and hence removing it can't be justified") 
        # Check which parameters should be added
        elif "Add" in fk:
            for ffk in Ftest_dict[fk].keys():
                if ffk == 'Binary':
                    for fffk in Ftest_dict[fk][ffk].keys():
                        if Ftest_dict[fk][ffk][fffk] is not None:
                            if Ftest_dict[fk][ffk][fffk]['ft'] <= alpha_add:
                                add_params.append(fffk)
                elif ffk == 'FD':
                    for fffk in Ftest_dict[fk][ffk].keys():
                        if Ftest_dict[fk][ffk][fffk] is not None:
                            if Ftest_dict[fk][ffk][fffk]['ft'] <= alpha_add:
                                fd_add.append(fffk)
                                fd_add_ft.append(Ftest_dict[fk][ffk][fffk]['ft'])
                else:
                    if Ftest_dict[fk][ffk] is not None:
                        if Ftest_dict[fk][ffk]['ft'] is not None:
                            if Ftest_dict[fk][ffk]['ft'] <= alpha_add:
                                add_params.append(ffk)
        # Policy is to never add additional spin derivatives in general
        elif fk == 'F':
            pass
    
    # Determine which FD parameters to add/remove
    if fd_remove:
        remove_params.append(fd_remove[np.argmin(fd_remove_ft)])
    if fd_add:
        add_params.append(fd_add[np.argmin(fd_add_ft)])
    
    # Check the config for previous FTest results
    add_params,remove_params = checkFLogs(add_params,remove_params,tc)

    # Now return which parameters to add/remove
    if remove_params:
        remove_statement = "F-tests recommend removing the following parameters: " + " ".join(remove_params)
    else:
        remove_statement = "F-tests do not recommend removing any parameters."
    if add_params:
        add_statement = "F-tests recommend adding the following parameters: " + " ".join(add_params)
    else:
        add_statement = "F-tests do not recommend adding any parameters."
   
    if 'FTest' in tc.config:
        oldFtests = tc.config['FTest']
        add_params = add_params + oldFtests["Add"]
        remove_params = remove_params + oldFtests["Remove"]    
 
    configAdd="[" + ",".join(sorted(set(add_params))) + "]"
    configRemove="[" + ",".join(sorted(set(remove_params))) + "]" 
    
    print('\n')
    print('Copy and paste below into .yaml file:')
    print('FTest:\n' \
        '  Add:',configAdd,'\n'\
        '  Remove:',configRemove)
    print('\n')
    
    return add_statement, remove_statement
    
def run_Ftests(
    fitter: pint.fitter.Fitter,
    tc: "TimingConfiguration",
    alpha_add: float = ALPHA_ADD,
    alpha_remove: float = ALPHA_REMOVE,
    FDnparams: int = 5,
    NITS: int = 1,
    FBnparams: int = 5,
    ncpu: int = 1
) -> dict:
    """
    This is the main convenience function to run the F-tests for F2, PX, binary parameters, FD and FB parameters.
    The test fits are set up and run by `pint_pal.parallel_fit.TestFitter`. The tests, chi2, degrees of freedom
    and F-statistic values are all printed in a nice human readable format.

    Parameters
    ----------
    fitter : pint.fitter.Fitter
        The PINT Fitter object.
    tc : pint_pal.timingconfiguration.TimingConfiguration
        The pint_pal TimingConfiguration object.
    alpha_add : float
        Value to compare for F-statistic significance when adding a parameter. If the F-statistic is lower than alpha_add, the timing model parameters are deemed statistically significant to (add to) the timing model.
    alpha_remove : float
        Value to compare for F-statistic significance when removing a parameter. If the F-statistic is lower than alpha_remove, the timing model parameters are deemed statistically significant to remove from timing model.
    FDnparams : int
        Maximum number of FD parameters to test.
    NITS : int
        Maximum number of fit iterations for each test fit [default: 1].
    FBnparams : int
        Maximum number of FB parameters to test [default: 5].
    ncpu : int
        Number of CPUs to run the test fits on. With 1, the fits are run one after the other [default: 1].

    Returns
    -------
    dict
        A nested dictionary of all of the different F-tests done and subsequent reported values for each
        F-test. These include keys for added ['Add'] or removed ['Remove'] parameters, the initial values ['initial']
        and within each of those, further nested dictionaries of parameters [e.g. 'PX'], and the reported values.
    """
    # Fit every test model, and the baseline (the current model, refit with the same settings)
    tf = TestFitter(model=fitter.model, toas=fitter.toas, maxFD=FDnparams, maxFB=FBnparams)
    results = tf.run(ncpu=ncpu, maxiter=NITS)
    baseline = results['baseline']

    # Define return dictionary
    retdict = {'initial': {'ft': None, 'chi2_test': baseline['chi2'], 'dof_test': baseline['dof']},
               'Add': {}, 'Remove': {}}
    for name, test in results.items():
        if name in ('meta', 'baseline'):
            continue
        # The parameters added or removed in this test are the ones set differently from the baseline
        params = [p for p, value, base_value in zip(tf.parnames, test['initvalues'], baseline['initvalues'])
                  if value != base_value]
        remove = params[0] in tf.removepars
        # Compare the simpler model with the more complex one; a failed fit (chi2 is NaN) gives None
        if np.isnan(test['chi2']):
            ftest_dict = None
        else:
            if remove:
                ft = FTest(test['chi2'], test['dof'], baseline['chi2'], baseline['dof'])
            else:
                ft = FTest(baseline['chi2'], baseline['dof'], test['chi2'], test['dof'])
            ftest_dict = {'ft': ft, 'chi2_test': test['chi2'], 'dof_test': test['dof']}
        # File the test under the same keys and labels as before
        section = retdict['Remove'] if remove else retdict['Add']
        if params[0].startswith('FB'):
            # 'FB<i>+' removes FB<i> and above; 'FB<i>' adds terms up to FB<i>
            numbers = sorted(int(p[2:]) for p in params)
            key = f"FB{numbers[0]}+" if remove else f"FB{numbers[-1]}"
            retdict.setdefault('FB', {})[key] = ftest_dict
        elif params[0].startswith('FD'):
            # Added terms are listed upwards (FD3, FD4), removed terms downwards (FD2, FD1)
            params = sorted(params, key=lambda p: int(p[2:]), reverse=remove)
            section.setdefault('FD', {})[", ".join(params)] = ftest_dict
        elif params == ['F2']:
            retdict.setdefault('F', {})['F2'] = ftest_dict
        elif params == ['PX']:
            section['PX'] = ftest_dict
        else:
            section.setdefault('Binary', {})[", ".join(params)] = ftest_dict
    # List the FD tests from fewest to most terms
    for section in (retdict['Add'], retdict['Remove']):
        if 'FD' in section:
            section['FD'] = dict(sorted(section['FD'].items(), key=lambda item: item[0].count(',')))

    # Print the table, then a summary of the F-tests results and suggestions
    print_Ftest_table(retdict, fitter, alpha_add=alpha_add, alpha_remove=alpha_remove)
    add_statement, remove_statement = summarize_Ftest(retdict, fitter, tc, alpha_add=alpha_add, alpha_remove=alpha_remove)
    print(add_statement)
    print(remove_statement)

    return retdict


def print_Ftest_table(
    Ftest_dict: dict,
    fitter: pint.fitter.Fitter,
    alpha_add: float = ALPHA_ADD,
    alpha_remove: float = ALPHA_REMOVE
) -> None:
    """
    Prints the F-test results from `run_Ftests()` as a table.

    Parameters
    ----------
    Ftest_dict : dict
        Dictionary of F-test results output by the `run_Ftests()` function.
    fitter : pint.fitter.Fitter
        The PINT Fitter object.
    alpha_add : float
        Added parameters are marked significant ('***') if the F-statistic is lower than alpha_add.
    alpha_remove : float
        Removed parameters are marked significant ('***') if the F-statistic is lower than alpha_remove.
    """
    print("Testing additional parameters (*** = significant):")
    print("%42s %9s %5s %s" % ("", "Chi2", "NDOF", "Ftest"))
    report_ptest("initial", Ftest_dict['initial'])
    # FD terms currently in the model, must be of the format FDi to avoid FDJUMP parameters
    cur_fd = [param for param in fitter.model.free_params
              if param.startswith("FD") and param[2:].isdigit()]
    for action, alpha in (('Add', alpha_add), ('Remove', alpha_remove)):
        if action == 'Add':
            print("Testing additional parameters:")
        else:
            print("Testing removal of parameters:")
        for key, value in Ftest_dict[action].items():
            if key == 'Binary':
                for label, ftest_dict in value.items():
                    report_ptest(label, ftest_dict, alpha = alpha)
            elif key == 'FD':
                print(f"Testing {'adding' if action == 'Add' else 'removing'} FD terms (", cur_fd, "enabled):")
                for label, ftest_dict in value.items():
                    report_ptest(label, ftest_dict, alpha = alpha)
            else:
                report_ptest(key, value, alpha = alpha)
    # Get current number of spin frequency derivatives
    current_freq_deriv = 1
    for i in range(2,21):
        p = "F%d" % i
        if p in fitter.model.params:
            current_freq_deriv = i
    if 'F' in Ftest_dict:
        print("Testing spin freq derivs (%s enabled):" % (current_freq_deriv))
        report_ptest('F2', Ftest_dict['F']['F2'], alpha = alpha_add)
    if 'FB' in Ftest_dict:
        fblist = get_fblist(fitter)
        fbp = [fblist[ifb] for ifb in sorted(fblist.keys())]  # sorted list of fb parameters
        print("Testing FB parameters, present list: "+" ".join(fbp))
        print("Testing removal of FB parameters:")
        for key, ftest_dict in Ftest_dict['FB'].items():
            if key.endswith('+'):
                i = int(key[2:-1])
                report_ptest(" ".join(fbp[i:]), ftest_dict, alpha = alpha_remove)
        print("Testing addition of FB parameters:")
        for key, ftest_dict in Ftest_dict['FB'].items():
            if not key.endswith('+'):
                i = int(key[2:])
                report_ptest(" ".join(f"FB{j}" for j in range(len(fbp), i+1)), ftest_dict, alpha = alpha_add)
