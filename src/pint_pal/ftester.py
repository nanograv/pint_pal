""" This is a set of utilities which performs F-test calculations for parameter inclusion/exclusion.

The F-tests are run with `pint_pal.parallel_fit.TestFitter`, which fits a set of nested timing models: one for
each parameter (or linked group of parameters) to be added to or removed from the timing model, plus a
baseline fit with no parameters added or removed. Each test fit is compared with the baseline fit using
`pint.utils.FTest()`.

`runFtest_parallel()` runs the fits in parallel on several CPUs, and `run_Ftests()` runs exactly the same fits
one at a time, for computers that aren't set up for parallel processing. Both process the results in the
same way and return the same output: a nested dictionary of the F-test results (`Ftest_dict`), in the same
layout as earlier versions of this module, for example:

    {'initial': {...},
     'Add':    {'Binary': {'PBDOT': {...}, 'M2, SINI': {...}}, 'FD': {'FD4': {...}, 'FD4, FD5': {...}}},
     'Remove': {'PX': {...}, 'Binary': {...}, 'FD': {'FD3': {...}, 'FD3, FD2': {...}}},
     'F':      {'F2': {...}},
     'FB':     {'FB2+': {...}, 'FB3': {...}},
     'FTest':  {'Add': [...], 'Remove': [...]},
     'fits':   {...}}

where each test entry ({...}) has the keys 'ft', 'resid_rms_test', 'resid_wrms_test', 'chi2_test' and 'dof_test'
(and 'dm_resid_rms_test' and 'dm_resid_wrms_test' for wideband data), or is None if the fit failed. 'FTest' holds
the recommendations to copy into the config .yaml file, and 'fits' the raw output of the `TestFitter` fits.
"""

import concurrent.futures
import copy
import json
import os
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.time import Time
from loguru import logger as log
import pint.fitter
import pint.gridutils
import pint.models
from pint.utils import FTest

import pint_pal.parallel_fit as par_fit

if TYPE_CHECKING:
    from pint_pal.timingconfiguration import TimingConfiguration

ALPHA_ADD = 0.0027  # 3-sigma; parameters are recommended to be added if the F-statistic is <= ALPHA_ADD
ALPHA_REMOVE = 0.1336  # 1.5-sigma; parameters are recommended to be removed if the F-statistic is > ALPHA_REMOVE
ALPHA = ALPHA_ADD  # default significance value


def param_check(
    name: str,
    fitter: pint.fitter.Fitter,
    check_enabled: bool = True
) -> bool:
    """
    Checks if a timing model parameter is in the timing model.

    Parameters
    ----------
    name : str
        Name of the timing model parameter to be checked.
    fitter : pint.fitter.Fitter
        The PINT Fitter object.
    check_enabled : bool
        If True, check if the model parameter is fit for as well [default: True].

    Returns
    -------
    bool
        Returns either True or False depending on if the parameter is in the model and/or enabled.
    """
    if name not in fitter.model.params:
        return False
    # If frozen is False, then it's fit for, if frozen is True, parameter not fit for
    if check_enabled and getattr(fitter.model, name).frozen:
        return False
    return True


def get_fblist(fitter: pint.fitter.Fitter) -> dict[int, str]:
    """
    Returns the list of FB parameters for the ELL1 model.

    Parameters
    ----------
    fitter : pint.fitter.Fitter
        The PINT Fitter object.

    Returns
    -------
    dict
        Dictionary of FB parameters currently in the timing model, keyed by FB index.
    """
    fblist = {}
    for p in fitter.model.params:
        # Skip FB jumps (FBJ*) and parameters that have no value set
        if p.startswith("FB") and not p.startswith("FBJ") and getattr(fitter.model, p).value is not None:
            fblist[int(p[2:]) if len(p) > 2 else 0] = p
    # FB parameters should be a contiguous series FB0, FB1, ..., FBn
    if sorted(fblist) != list(range(len(fblist))):
        log.warning("FB parameters not a series of integers: " +
                    " ".join(fblist[i] for i in sorted(fblist)))
    return fblist


def _valid_ft(ft: float | bool | None) -> bool:
    """
    Checks if an F-statistic is a usable number (not None, False, or NaN). E.g. PINT's FTest returns NaN when
    the two models have the same degrees of freedom. An F-statistic of exactly 0 is valid: it means the
    difference is so significant that the probability underflows.

    Parameters
    ----------
    ft : float, bool or None
        The F-statistic.

    Returns
    -------
    bool
        True if the F-statistic can be used.
    """
    if ft is None or isinstance(ft, (bool, np.bool_)):
        return False
    return bool(np.isfinite(ft))


def _recommendation_statements(
    add_params: list[str],
    remove_params: list[str]
) -> tuple[str, str]:
    """
    Returns the human-readable statements of which parameters are recommended to be added/removed.

    Parameters
    ----------
    add_params : list of str
        Parameters recommended to be added.
    remove_params : list of str
        Parameters recommended to be removed.

    Returns
    -------
    (str, str)
        Add, Remove text statements.
    """
    if add_params:
        add_statement = "F-tests recommend adding the following parameters: " + " ".join(add_params)
    else:
        add_statement = "F-tests do not recommend adding any parameters."
    if remove_params:
        remove_statement = "F-tests recommend removing the following parameters: " + " ".join(remove_params)
    else:
        remove_statement = "F-tests do not recommend removing any parameters."
    return add_statement, remove_statement


# ---------------------------------------------------------------------------------------------------------
# Running the F-test fits
# ---------------------------------------------------------------------------------------------------------

class _SerialExecutor(concurrent.futures.Executor):
    """
    Executor that runs each task immediately in the current process. Passing this to `TestFitter.run()` makes
    PINT run the F-test fits one at a time through exactly the same code as the parallel workers (a fresh copy
    of the fitter for each test, and the same handling of failed fits), so the serial and parallel F-tests
    give identical results.
    """

    def submit(self, fn: Callable, /, *args, **kwargs) -> concurrent.futures.Future:
        """
        Runs a task immediately and returns its (already completed) result.

        Parameters
        ----------
        fn : callable
            The task to run, called as ``fn(*args, **kwargs)``.

        Returns
        -------
        concurrent.futures.Future
            A completed future holding the result, or the exception the task raised.
        """
        future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as e:
            future.set_exception(e)
        return future


def _has_value(model: pint.models.TimingModel, name: str) -> bool:
    """
    Checks if a parameter is in the timing model with a nonzero value (as `TestFitter` decides).

    Parameters
    ----------
    model : pint.models.TimingModel
        The PINT timing model.
    name : str
        Name of the timing model parameter.

    Returns
    -------
    bool
        True if the parameter is in the model with a value that is set and nonzero.
    """
    return name in model.params and model[name].value is not None and model[name].value != 0


def _residual_stats(fitter: pint.fitter.Fitter) -> dict:
    """
    Returns the residual statistics reported for each F-test, as PINT's `Fitter.ftest()` reports them.

    Parameters
    ----------
    fitter : pint.fitter.Fitter
        A PINT Fitter object that has been fit.

    Returns
    -------
    dict
        'resid_rms_test' and 'resid_wrms_test' (the RMS and weighted RMS of the TOA residuals, in us), and for
        wideband data 'dm_resid_rms_test' and 'dm_resid_wrms_test' (the same for the DM residuals).
    """
    # Wideband residuals are split into TOA and DM residuals
    toa_resids = fitter.resids.residual_objs['toa'] if fitter.is_wideband else fitter.resids
    stats = {'resid_rms_test': toa_resids.time_resids.std().to(u.us),
             'resid_wrms_test': toa_resids.rms_weighted()}
    if fitter.is_wideband:
        stats['dm_resid_rms_test'] = fitter.resids.residual_objs['dm'].resids.std()
        stats['dm_resid_wrms_test'] = fitter.resids.residual_objs['dm'].rms_weighted()
    return stats


class _FitWorker:
    """
    Runs one F-test fit, with each tested parameter either fixed at a given value or fit for.

    This follows the same steps as PINT's `pint.gridutils.WrappedFitter.doonefit()`, which `TestFitter` normally
    uses (a fresh copy of the fitter, the same fit, and the same handling of failed fits), but it also returns
    the residual statistics of the fit. Those need all of the fitted parameter values, which `doonefit()` can't
    return for MJD parameters such as T0.
    """

    def __init__(self, fitter: pint.fitter.Fitter, parnames: list[str], fitargs: dict):
        """
        Parameters
        ----------
        fitter : pint.fitter.Fitter
            The PINT Fitter object for the maximal model (`TestFitter.f0`).
        parnames : list of str
            Names of the tested parameters, in the order of each test's values.
        fitargs : dict
            Additional arguments passed to `fit_toas()`, e.g. maxiter.
        """
        self.fitter = fitter
        self.parnames = parnames
        self.fitargs = fitargs

    def __call__(self, parvalues: list) -> dict:
        """
        Runs the fit for one test.

        Parameters
        ----------
        parvalues : list
            Value of each tested parameter: None to fit for it, otherwise the value to fix it at.

        Returns
        -------
        dict
            'chi2' (NaN if the fit failed), 'dof', 'results' (fitted values of the tested parameters) and
            'resid_stats' (see `_residual_stats()`, or None if the fit failed).
        """
        fitter = copy.deepcopy(self.fitter)
        parstrings = []
        for name, value in zip(self.parnames, parvalues):
            # Freeze the parameters with a value at that value, and fit for the others
            param = getattr(fitter.model, name)
            if value is not None:
                param.frozen = True
                param.quantity = value
            else:
                param.frozen = False
            parstrings.append(f"{name} = {value}")
        failed = False
        try:
            fitter.fit_toas(**self.fitargs)
        except (pint.fitter.InvalidModelParameters, pint.fitter.StepProblem):
            log.warning(f"Fit may not be converged for {','.join(parstrings)}, but returning anyway")
        except pint.fitter.MaxiterReached:
            log.warning(f"Max iterations reached for {','.join(parstrings)}: returning NaN")
            failed = True
        except Exception as e:
            log.warning(f"Unexpected exception {e} for {','.join(parstrings)}: returning NaN")
            failed = True
        return {'chi2': np.nan if failed else fitter.resids.chi2,
                'dof': fitter.resids.dof,
                'results': {name: getattr(fitter.model, name).quantity for name in self.parnames},
                'resid_stats': None if failed else _residual_stats(fitter)}


# Binary parameters F-tested for the BT model, which `parallel_fit.TestFitter` doesn't cover. These are the DD
# parameters without M2 and SINI, since PINT's BT model has no Shapiro delay
_BT_EXTRAPARNAMES = ["PBDOT", "A1DOT", "OMDOT", "EDOT"]


class _TestFitter(par_fit.TestFitter):
    """
    `parallel_fit.TestFitter`, with the fits run by `_FitWorker` (so that the residual statistics of each fit are
    also returned), and with these changes to which binary parameters are F-tested:

    - BT: `TestFitter` doesn't test any binary parameters for BT, so PBDOT, A1DOT, OMDOT and EDOT are tested
      (the DD parameters without M2 and SINI, which PINT's BT model doesn't have).
    - ELL1H: only H3 is tested, not H4. H4 is left out of the tests unless it is already in the timing model.
      If it is, it is kept so that removing H3 also removes H4 (H4 can't be used without H3), and the test of
      removing H4 on its own is ignored when the results are processed. Leaving H4 out also avoids adding H4 to
      models that use STIGMA, which PINT doesn't allow ("ELL1H can use H4 or STIGMA but not both").
    """

    def setextrapars(self, maxFB: int = 5, maxFD: int = 5) -> None:
        """
        Defines the extra parameters to test, as `TestFitter.setextrapars()` does, with the BT and ELL1H
        changes above.

        Parameters
        ----------
        maxFB : int
            Maximum FB number to include [default: 5].
        maxFD : int
            Maximum FD number to include [default: 5].
        """
        super().setextrapars(maxFB=maxFB, maxFD=maxFD)
        binary = self.m_base["BINARY"].value if "BINARY" in self.m_base.params else None
        if binary == "BT":
            # Insert before the FD parameters so the binary tests are listed with the other parameters
            fd_index = next((i for i, p in enumerate(self.extraparnames) if re.fullmatch(r"FD\d+", p)),
                            len(self.extraparnames))
            new_pars = [p for p in _BT_EXTRAPARNAMES if p not in self.extraparnames]
            self.extraparnames[fd_index:fd_index] = new_pars
        elif binary == "ELL1H" and "H4" in self.extraparnames and not _has_value(self.m_base, "H4"):
            self.extraparnames.remove("H4")

    def run(
        self,
        executor: concurrent.futures.Executor | None = None,
        ncpu: int | None = None,
        chunksize: int = 1,
        printprogress: bool = True,
        **fitargs
    ) -> dict:
        """
        Runs the fits, as `TestFitter.run()` does, but with `_FitWorker` so that the residual statistics of each
        fit are also returned.

        Parameters
        ----------
        executor : concurrent.futures.Executor or None
            Executor to run the fits with. If None, a process pool with ncpu workers is created, or the fits are
            run one at a time if ncpu is 1 [default: None].
        ncpu : int or None
            Number of CPUs to run the fits on, if executor is None. If None, all CPUs are used [default: None].
        chunksize : int
            Number of fits sent to each worker at a time [default: 1].
        printprogress : bool
            Show a progress bar, if `tqdm` is installed [default: True].
        **fitargs
            Additional arguments passed to `fit_toas()`, e.g. maxiter.

        Returns
        -------
        dict
            `TestFitter`'s test information, with 'chi2', 'dof', 'results' and 'resid_stats' added for each test.
        """
        worker = _FitWorker(self.f0, self.parnames, fitargs)
        if executor is None:
            if ncpu == 1:
                executor = _SerialExecutor()
            else:
                executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=ncpu or os.cpu_count(),
                    initializer=pint.gridutils.set_log, initargs=(log,))
        with executor as e:
            outputs = e.map(worker, self.parvalues, chunksize=chunksize)
            if printprogress and pint.gridutils.tqdm is not None:
                outputs = pint.gridutils.tqdm(outputs, total=len(self.parvalues), ascii=True)
            outputs = list(outputs)
        for key, info in self.testinfo.items():
            if key != "meta":
                info.update(outputs[info["number"]])
        self.testinfo["meta"]["stoptime"] = Time.now().iso
        return self.testinfo


def _run_test_fits(
    fitter: pint.fitter.Fitter,
    executor: concurrent.futures.Executor | None = None,
    ncpu: int | None = None,
    maxiter: int = 20,
    FDnparams: int = 5,
    FBnparams: int = 5
) -> dict:
    """
    Runs the baseline fit and all of the F-test fits with `parallel_fit.TestFitter`.

    Parameters
    ----------
    fitter : pint.fitter.Fitter
        The PINT Fitter object. Its current model and TOAs are what is tested.
    executor : concurrent.futures.Executor or None
        Executor to run the fits with. If None, one is created with ncpu workers [default: None].
    ncpu : int or None
        Number of CPUs to run the fits on, if executor is None [default: None].
    maxiter : int
        Maximum number of fit iterations for each test [default: 20].
    FDnparams : int
        Maximum number of FD parameters to test [default: 5].
    FBnparams : int
        Maximum number of FB parameters to test, for ELL1 models using FB [default: 5].

    Returns
    -------
    dict
        Output of the `TestFitter` run, including the 'baseline' fit, with each fit's residual statistics.
    """
    # Pass the fitter's current model and TOAs (rather than the original .par/.tim files) to test exactly what
    # has been fit, including any TOA excision. TestFitter includes the baseline fit in its tests
    try:
        tf = _TestFitter(model=fitter.model, toas=fitter.toas, maxFB=FBnparams, maxFD=FDnparams)
    except TypeError as e:
        # Older versions of parallel_fit only take parfile/timfile (and lack the baseline fit and nested FD/FB tests)
        if "model" in str(e) or "toas" in str(e):
            raise ImportError(
                f"The pint_pal.parallel_fit being imported ({par_fit.__file__}) is an older version whose "
                "TestFitter doesn't accept model=/toas=. Check that the updated file is named parallel_fit.py, "
                "that pint_pal is installed from the folder you updated (e.g. pip install -e), and restart the "
                "Python kernel.") from e
        raise
    return tf.run(executor=executor, ncpu=ncpu, maxiter=maxiter)


# ---------------------------------------------------------------------------------------------------------
# Processing the F-test results
# ---------------------------------------------------------------------------------------------------------

def _chi2_problem(chi2_simple: float, chi2_complex: float) -> str | None:
    """
    Checks that the more complex of two nested fits doesn't have a noticeably higher chi2 than the simpler one,
    which can't happen when both fits are converged. PINT's FTest() returns 1.0 in that case, which would
    wrongly recommend removing a parameter.

    Parameters
    ----------
    chi2_simple : float
        Chi-squared of the fit with fewer parameters.
    chi2_complex : float
        Chi-squared of the fit with more parameters.

    Returns
    -------
    str or None
        A description of the problem, or None if there is no problem.
    """
    tolerance = max(1e-3, 1e-6 * abs(chi2_simple))
    if chi2_complex > chi2_simple + tolerance:
        return "the fit with more parameters has a higher chi2, so a fit (often the baseline) is not converged"
    return None


def _evaluate_tests(results: dict) -> tuple[dict, list[dict]]:
    """
    Computes the F-statistic of each test against the baseline fit.

    Parameters
    ----------
    results : dict
        Output of `run_Ftests()` or `runFtest_parallel()`.

    Returns
    -------
    baseline : dict
        The baseline fit, with keys 'chi2' and 'dof'.
    tests : list of dict
        One dictionary per test, with keys 'key' (the test's key in results), 'action' ('add' or 'remove'),
        'params' (list of parameter names), 'chi2', 'dof', 'ft' (the F-statistic, or None if it can't be used)
        and 'problem' (why it can't be used, or None).
    """
    if 'baseline' not in results or not np.isfinite(results['baseline']['chi2']):
        raise ValueError("F-test results have no valid 'baseline' fit to compare against.")
    baseline = results['baseline']

    tests = []
    for key, test in results.items():
        # Skip the run metadata and the baseline fit, everything else is a test
        if key in ('meta', 'baseline'):
            continue
        # The test string is of the form "Combination <n>: <add/remove> <PAR1> and <PAR2> ..."
        action, parList = test['string'].split(':', 1)[1].strip().split(" ", 1)
        chi2 = test['chi2']
        dof = test['dof']
        # The simpler and more complex of the two models being compared
        simple, complex_ = ((chi2, dof), (baseline['chi2'], baseline['dof'])) if action == 'remove' else \
            ((baseline['chi2'], baseline['dof']), (chi2, dof))
        # A failed fit returns a NaN chi2; FTest() would then return 1.0, which would wrongly recommend removal
        if not np.isfinite(chi2):
            problem = "the fit failed"
        else:
            problem = _chi2_problem(simple[0], complex_[0])
        # FTest(chi2_1, dof_1, chi2_2, dof_2) compares the simpler model (1) with the more complex model (2)
        ft = None if problem else FTest(simple[0], simple[1], complex_[0], complex_[1])
        tests.append({'key': key, 'action': action, 'params': parList.split(" and "), 'chi2': chi2, 'dof': dof,
                      'ft': ft, 'problem': problem})
    return baseline, tests


def _raw_results(Ftest_dict: dict) -> dict:
    """
    Returns the raw output of the `TestFitter` fits from an F-test results dictionary.

    Parameters
    ----------
    Ftest_dict : dict
        Output of `run_Ftests()` or `runFtest_parallel()` (which holds the raw output under 'fits'), or the raw
        output itself.

    Returns
    -------
    dict
        The raw output of the `TestFitter` fits, including the 'baseline' fit.
    """
    return Ftest_dict['fits'] if 'fits' in Ftest_dict else Ftest_dict


def _test_entry(chi2: float, dof: int, ft: float | None, stats: dict | None) -> dict:
    """
    Returns one test's entry in the nested F-test results dictionary, in the format of PINT's `Fitter.ftest()`.

    Parameters
    ----------
    chi2 : float
        Chi-squared of the tested model.
    dof : int
        Degrees of freedom of the tested model.
    ft : float or None
        The F-statistic (None for the initial model).
    stats : dict or None
        Residual statistics (see `_residual_stats()`). If None, the RMS values are NaN.

    Returns
    -------
    dict
        Keys 'ft', 'resid_rms_test', 'resid_wrms_test', 'chi2_test', 'dof_test', and for wideband data
        'dm_resid_rms_test' and 'dm_resid_wrms_test'.
    """
    stats = stats or {'resid_rms_test': np.nan * u.us, 'resid_wrms_test': np.nan * u.us}
    entry = {'ft': ft,
             'resid_rms_test': stats['resid_rms_test'],
             'resid_wrms_test': stats['resid_wrms_test'],
             'chi2_test': chi2,
             'dof_test': dof}
    for key in ('dm_resid_rms_test', 'dm_resid_wrms_test'):
        if key in stats:
            entry[key] = stats[key]
    return entry


def _build_Ftest_dict(results: dict) -> dict:
    """
    Arranges the F-test results in the nested dictionary used by earlier versions of this module (and read by
    `pint_pal.utils.pdf_writer()`):

    - 'initial': the baseline fit.
    - 'Add' and 'Remove': 'Binary' (binary parameters) and 'FD' (FD parameters) dictionaries, and other
      parameters (e.g. 'PX') directly.
    - 'F': spin frequency derivatives that are tested for addition (e.g. 'F2').
    - 'FB': FB parameters, keyed 'FB<i>+' for removing FB<i> and above, and 'FB<i>' for adding up to FB<i>.

    Each test is keyed by the names of its parameters (e.g. 'M2, SINI', 'FD4, FD5', or 'FD3, FD2' for removals),
    and its entry is in the format of PINT's `Fitter.ftest()` (see `_test_entry()`), or None if the fit failed.
    'ft' is NaN for a test that can't be F-tested (e.g. its chi2 moved the wrong way).

    Parameters
    ----------
    results : dict
        Raw output of the `TestFitter` fits, including the 'baseline' fit.

    Returns
    -------
    dict
        The nested F-test results dictionary.
    """
    baseline, tests = _evaluate_tests(results)
    Ftest_dict = {'initial': _test_entry(baseline['chi2'], baseline['dof'], None, baseline.get('resid_stats')),
                  'Add': {}, 'Remove': {}}
    for test in tests:
        action, pars = test['action'], test['params']
        raw = results[test['key']]
        if test['problem'] == "the fit failed":
            entry = None
        else:
            ft = test['ft'] if test['ft'] is not None else np.nan
            entry = _test_entry(test['chi2'], test['dof'], ft, raw.get('resid_stats'))
        section = Ftest_dict['Add' if action == 'add' else 'Remove']
        indices = sorted(int(p[2:]) for p in pars) if all(re.fullmatch(r"F[DB]\d+", p) for p in pars) else None
        if indices and pars[0].startswith("FD"):
            # FD tests are labelled with the FD parameters in the order they are added or removed
            order = indices if action == 'add' else indices[::-1]
            section.setdefault('FD', {})[", ".join(f"FD{i}" for i in order)] = entry
        elif indices and pars[0].startswith("FB"):
            # 'FB<i>+' removes FB<i> and above, 'FB<i>' adds up to FB<i>
            key = f"FB{indices[0]}+" if action == 'remove' else f"FB{indices[-1]}"
            Ftest_dict.setdefault('FB', {})[key] = entry
        elif action == 'add' and all(re.fullmatch(r"F\d+", p) for p in pars):
            Ftest_dict.setdefault('F', {})[", ".join(pars)] = entry
        elif pars == ['PX'] or all(re.fullmatch(r"F\d+", p) for p in pars):
            section[", ".join(pars)] = entry
        else:
            section.setdefault('Binary', {})[", ".join(pars)] = entry
    # List the FD tests from fewest to most parameters (e.g. 'FD3', then 'FD3, FD2'), as earlier versions did
    for section in (Ftest_dict['Add'], Ftest_dict['Remove']):
        if 'FD' in section:
            section['FD'] = dict(sorted(section['FD'].items(), key=lambda item: len(item[0].split(","))))
    return Ftest_dict


def report_ptest(label: str, ftest_dict: dict | None = None, alpha: float = ALPHA) -> str:
    """
    Formats the results of an F-test as a line of a human-readable table.

    Parameters
    ----------
    label : str
        Name of the parameter(s) that were added/removed for the F-test.
    ftest_dict : dict or None
        The test's entry in the nested F-test results dictionary (see `_test_entry()`). If None (the fit
        failed), a line of NaNs is returned.
    alpha : float
        The F-test is marked significant ('***') if the F-statistic is lower than alpha [default: 0.0027].

    Returns
    -------
    str
        The formatted line: the label, the weighted RMS of the residuals (and of the DM residuals, for wideband
        data), chi2, degrees of freedom, and the F-statistic ('--' for the initial model, 'xxx' if the test
        can't be used).
    """
    # If the fit failed, give a line of NaNs
    if ftest_dict is None:
        return "%42s %7.3f %9.2f %5f %.3f" % (label, np.nan, np.nan, np.nan, np.nan)
    line = "%42s %7.3f" % (label, ftest_dict['resid_wrms_test'].to_value(u.us))
    # weighted root mean square of DM residuals (wideband only)
    if 'dm_resid_wrms_test' in ftest_dict:
        line += " %16.6f" % ftest_dict['dm_resid_wrms_test'].value
    line += " %9.2f %5d" % (ftest_dict['chi2_test'], ftest_dict['dof_test'])
    Fstatistic = ftest_dict['ft']
    if Fstatistic is None:
        line += " --"
    elif _valid_ft(Fstatistic):
        line += " %.3g%s" % (Fstatistic, " ***" if Fstatistic < alpha else "")
    else:
        line += " xxx"
    return line


def get_Ftest_lines(
    Ftest_dict: dict,
    fitter: pint.fitter.Fitter | None = None,
    alpha_add: float = ALPHA_ADD,
    alpha_remove: float = ALPHA_REMOVE,
    alpha: float | None = None
) -> list[str]:
    """
    Returns the F-test results as lines of a human-readable table, in the same layout as earlier versions of
    this module, e.g. to print or to write to a summary pdf.

    Parameters
    ----------
    Ftest_dict : dict
        Output of `run_Ftests()` or `runFtest_parallel()`.
    fitter : pint.fitter.Fitter or None
        The PINT Fitter object. If given, the enabled FD parameters, the number of spin frequency derivatives and
        the FB parameters in the model are listed in the section headings [default: None].
    alpha_add : float
        Add tests are marked significant ('***') if the F-statistic is lower than alpha_add [default: 0.0027].
    alpha_remove : float
        Removal tests are marked significant ('***') if the F-statistic is lower than alpha_remove
        [default: 0.1336].
    alpha : float or None
        Older name for alpha_add, kept so existing calls still work. If given, overrides alpha_add.

    Returns
    -------
    list of str
        Lines of the table, starting with the column headings and ending with a line explaining the markers.
    """
    if alpha is not None:
        alpha_add = alpha
    if 'initial' not in Ftest_dict:
        Ftest_dict = _build_Ftest_dict(Ftest_dict)
    wideband = 'dm_resid_wrms_test' in Ftest_dict['initial']
    if wideband:
        lines = ["%42s %7s %9s %9s %5s %s" % ("", "RMS(us)", "DM RMS(pc cm^-3)", "Chi2", "NDOF", "Ftest")]
    else:
        lines = ["%42s %7s %9s %5s %s" % ("", "RMS(us)", "Chi2", "NDOF", "Ftest")]
    lines.append(report_ptest("initial", Ftest_dict['initial']))

    cur_fd = ([p for p in fitter.model.free_params if re.fullmatch(r"FD\d+", p)] if fitter is not None else None)
    for action, heading in (('Add', "Testing additional parameters:"), ('Remove', "Testing removal of parameters:")):
        alpha_test = alpha_add if action == 'Add' else alpha_remove
        lines.append(heading)
        for key, entry in Ftest_dict.get(action, {}).items():
            if key == 'Binary':
                lines += [report_ptest(k, v, alpha=alpha_test) for k, v in entry.items()]
            elif key == 'FD':
                verb = "adding" if action == 'Add' else "removing"
                lines.append(f"Testing {verb} FD terms" + (f" ( {cur_fd} enabled):" if cur_fd is not None else ":"))
                lines += [report_ptest(k, v, alpha=alpha_test) for k, v in entry.items()]
            else:
                lines.append(report_ptest(key, entry, alpha=alpha_test))

    if 'F' in Ftest_dict:
        n_deriv = max([int(p[1:]) for p in fitter.model.params if re.fullmatch(r"F\d+", p)] or [1]) \
            if fitter is not None else None
        lines.append("Testing spin freq derivs" + (f" ({n_deriv} enabled):" if n_deriv is not None else ":"))
        lines += [report_ptest(k, v, alpha=alpha_add) for k, v in Ftest_dict['F'].items()]

    if 'FB' in Ftest_dict:
        fbp = [p for _, p in sorted(get_fblist(fitter).items())] if fitter is not None else None
        def fb_label(key):
            # Label FB tests by their parameters, e.g. 'FB1+' -> 'FB1 FB2', 'FB4' -> 'FB3 FB4' (needs the fitter)
            if fbp is None:
                return key
            i = int(key.rstrip('+')[2:])
            return " ".join(fbp[i:]) if key.endswith('+') else " ".join(f"FB{j}" for j in range(len(fbp), i + 1))
        if fbp is not None:
            lines.append("Testing FB parameters, present list: " + " ".join(fbp))
        lines.append("Testing removal of FB parameters:")
        lines += [report_ptest(fb_label(k), v, alpha=alpha_remove) for k, v in Ftest_dict['FB'].items() if k.endswith('+')]
        lines.append("Testing addition of FB parameters:")
        lines += [report_ptest(fb_label(k), v, alpha=alpha_add) for k, v in Ftest_dict['FB'].items()
                  if not k.endswith('+')]

    # Explain the markers, since '***' means the opposite outcome for 'add' and 'remove' tests
    lines.append(f"*** = significant: worth adding (F < {alpha_add:g}) or worth keeping (F < {alpha_remove:g}); "
                 "xxx = fit failed or not converged, or F-test not possible")
    return lines


def _step_through_nested(
    chain: list[dict],
    action: str,
    baseline: dict,
    alpha_add: float = ALPHA_ADD,
    alpha_remove: float = ALPHA_REMOVE
) -> tuple[list[str], list[tuple]]:
    """
    Steps through a nested series of FD or FB tests one term at a time, as repeated runs of the F-tests would.

    Each step compares a test with the previous (smaller) test in the series, or with the baseline for the
    first step, so each term is tested on its own given the terms before it. For removals, e.g. with FD1 and
    FD2 in the model: FD2 is tested against the baseline ("remove FD2"), then FD1 is tested by comparing
    "remove FD1 and FD2" with "remove FD2" (the FD1-only model), which is the test a second run would make.
    Stepping stops at the first term that fails, so a term that is needed on its own is never added or
    removed just because it was tested together with others.

    Parameters
    ----------
    chain : list of dict
        The nested tests (as returned by `_evaluate_tests()`), sorted by number of parameters.
    action : str
        'add' or 'remove'.
    baseline : dict
        The baseline fit, with keys 'chi2' and 'dof'.
    alpha_add : float
        F-statistic threshold for adding parameters [default: 0.0027].
    alpha_remove : float
        F-statistic threshold for removing parameters [default: 0.1336].

    Returns
    -------
    chosen : list of str
        Parameters to add or remove (empty if the first step fails or the tests aren't nested).
    steps : list of tuple
        (new parameters, F-statistic, passed) for each step taken.
    """
    chosen = []
    previous = baseline
    steps = []
    for test in chain:
        # The tests should be nested, each one containing all of the parameters of the previous one
        # (if they aren't, any recommendation could break the series, so none is made)
        if not set(chosen) <= set(test['params']):
            log.warning(f"The {action} tests for {', '.join(test['params'])} are not nested, so no change is "
                        "recommended for them. Check that parallel_fit.py uses the instance's linked parameters.")
            return [], steps
        new_pars = [p for p in test['params'] if p not in chosen]
        # A fit that failed or isn't converged can't be compared, so stop there
        simple, complex_ = (test, previous) if action == 'remove' else (previous, test)
        if test['problem'] or _chi2_problem(simple['chi2'], complex_['chi2']):
            steps.append((new_pars, None, False))
            break
        # FTest(chi2_1, dof_1, chi2_2, dof_2) compares the simpler model (1) with the more complex model (2)
        ft = FTest(simple['chi2'], simple['dof'], complex_['chi2'], complex_['dof'])
        if action == 'remove':
            passed = _valid_ft(ft) and ft > alpha_remove
        else:
            passed = _valid_ft(ft) and ft <= alpha_add
        steps.append((new_pars, ft, passed))
        if not passed:
            break
        chosen = test['params']
        previous = test
    return chosen, steps


def _describe_steps(prefix: str, action: str, steps: list[tuple]) -> str:
    """
    Returns a one-line description of the steps taken through a nested FD or FB series.

    Parameters
    ----------
    prefix : str
        'FD' or 'FB'.
    action : str
        'add' or 'remove'.
    steps : list of tuple
        The steps, as returned by `_step_through_nested()`.

    Returns
    -------
    str
        The description.
    """
    verb = 'remove' if action == 'remove' else 'add'
    parts = []
    for new_pars, ft, passed in steps:
        result = ("fit failed or not converged" if ft is None else f"F={ft:.3g}") + (f", {verb}" if passed else ", stop")
        parts.append(f"{', '.join(new_pars)} ({result})")
    return f"{prefix} {'removal' if action == 'remove' else 'addition'}, one term at a time: " + "; ".join(parts)


def parseParallelResults(
    results: dict,
    alpha_add: float = ALPHA_ADD,
    alpha_remove: float = ALPHA_REMOVE,
    verbose: bool = True
) -> tuple[list[str], list[str]]:
    """
    Checks the results of the F-tests and exports the parameters to be added/removed.

    The following policies are applied: spin frequency derivatives are never recommended to be added,
    parallax is never recommended to be removed, and for the ELL1H binary model only H3 is tested (a test
    of H4 on its own is ignored).

    FD and FB parameters each form a series (FD1, FD2, ...), and their tests are nested (e.g. "add FD4" and
    "add FD4 and FD5"; "remove FD2" and "remove FD1 and FD2"). Each series is stepped through one term at a
    time (see `_step_through_nested()`), so the result is the same as re-running the F-tests until nothing
    changes. One consistent decision is made for each series:

    - If adding the next term is significant, the terms that pass are recommended to be added and nothing
      in that series is removed: a significant higher-order term means the lower-order terms are kept.
      Adding takes precedence because it requires much stronger evidence (alpha_add) than removing.
    - Otherwise, terms are recommended to be removed from the top down, until one is significant.

    Parameters
    ----------
    results : dict
        Output of `run_Ftests()` or `runFtest_parallel()`, or the raw output of the `TestFitter` fits.
    alpha_add : float
        F-statistic threshold for adding parameters; added if F <= alpha_add [default: 0.0027].
    alpha_remove : float
        F-statistic threshold for removing parameters; removed if F > alpha_remove [default: 0.1336].
    verbose : bool
        If True, print how the FD/FB series were decided [default: True].

    Returns
    -------
    (list of str, list of str)
        Parameters recommended to be added, and parameters recommended to be removed.
    """
    baseline, tests = _evaluate_tests(_raw_results(results))

    n_wrong_way = sum(1 for test in tests if test['problem'] and test['problem'] != "the fit failed")
    if n_wrong_way:
        log.warning(f"{n_wrong_way} of {len(tests)} tests have chi2 moving the wrong way (worse with more "
                    "parameters). The baseline fit is probably not converged: refit the model, or increase "
                    "maxiter, before relying on these F-tests.")

    add_params = []
    remove_params = []
    # Nested FD and FB tests, keyed by (prefix, action); these are decided separately below
    nested = {}

    for test in tests:
        action, pars, ft = test['action'], test['params'], test['ft']
        if test['problem']:
            log.warning(f"'{action} {', '.join(pars)}' can't be F-tested: {test['problem']}.")
        # Keep nested FD and FB tests aside, including the ones that fail, since they are stepped through below
        prefix = next((pre for pre in ('FD', 'FB') if all(re.fullmatch(pre + r"\d+", p) for p in pars)), None)
        if prefix is not None:
            nested.setdefault((prefix, action), []).append(test)
            continue
        if ft is None:
            continue
        if action == 'remove':
            recommend = _valid_ft(ft) and ft > alpha_remove
        else:
            recommend = _valid_ft(ft) and ft <= alpha_add
        if not recommend:
            continue

        # Policy is to never add additional spin derivatives in general
        if action == 'add' and any(re.fullmatch(r"F\d+", p) for p in pars):
            continue
        # Policy is never to remove parallax
        if action == 'remove' and 'PX' in pars:
            continue
        # For ELL1H only H3 is tested; H4 on its own is ignored (see `_TestFitter`)
        if pars == ['H4']:
            continue
        if action == 'remove':
            remove_params.extend(pars)
        else:
            add_params.extend(pars)

    # Decide each FD and FB series one term at a time. Adding and removing within the same series would give
    # an invalid model (e.g. FD1 and FD3 without FD2), so if a term is added, nothing in that series is removed
    for prefix in ('FD', 'FB'):
        steps_by_action = {}
        chosen_by_action = {}
        for action in ('add', 'remove'):
            chain = sorted(nested.get((prefix, action), []), key=lambda t: len(t['params']))
            chosen_by_action[action], steps_by_action[action] = _step_through_nested(
                chain, action, baseline, alpha_add=alpha_add, alpha_remove=alpha_remove)
        to_add, to_remove = chosen_by_action['add'], chosen_by_action['remove']
        if verbose:
            for action in ('add', 'remove'):
                # Only describe series that go beyond a single step, or that lead to a change
                if len(steps_by_action[action]) > 1 or chosen_by_action[action]:
                    print(_describe_steps(prefix, action, steps_by_action[action]))
        if to_add:
            add_params.extend(to_add)
            if to_remove and verbose:
                print(f"Not recommending removing {', '.join(to_remove)}: adding {', '.join(to_add)} is "
                      f"significant, so the existing {prefix} terms are kept.")
        else:
            remove_params.extend(to_remove)

    return sorted(set(add_params)), sorted(set(remove_params))


def summarize_Ftest(
    Ftest_dict: dict,
    fitter: pint.fitter.Fitter | None = None,
    alpha_add: float = ALPHA_ADD,
    alpha_remove: float = ALPHA_REMOVE,
    alpha: float | None = None
) -> tuple[str, str]:
    """
    Function to determine what parameters are recommended to be added/removed from the F-test results.

    Parameters
    ----------
    Ftest_dict : dict
        Output of `run_Ftests()` or `runFtest_parallel()`.
    fitter : pint.fitter.Fitter or None
        The PINT Fitter object. [currently unused]
    alpha_add : float
        Parameters are recommended to be added if the F-statistic is lower than or equal to alpha_add, i.e.
        they are statistically significant to the timing model [default: 0.0027].
    alpha_remove : float
        Parameters are recommended to be removed if the F-statistic is greater than alpha_remove, i.e. they
        are not statistically significant to the timing model [default: 0.1336].
    alpha : float or None
        Older name for alpha_add, kept so existing calls still work. If given, overrides alpha_add.

    Returns
    -------
    (str, str)
        Add, Remove text statements.
    """
    if alpha is not None:
        alpha_add = alpha
    add_params, remove_params = parseParallelResults(_raw_results(Ftest_dict), alpha_add=alpha_add,
                                                     alpha_remove=alpha_remove, verbose=False)
    return _recommendation_statements(add_params, remove_params)


def _get_old_ftests(tc: "TimingConfiguration") -> tuple[list[str], list[str]] | None:
    """
    Returns the lists of previously recommended parameters to add and remove from the config 'FTest' entry.

    Parameters
    ----------
    tc : pint_pal.timingconfiguration.TimingConfiguration
        The pint_pal TimingConfiguration object.

    Returns
    -------
    (list of str, list of str) or None
        Previous Add, Remove recommendations, or None if there is no 'FTest' entry in the config.
    """
    if 'FTest' not in tc.config:
        return None
    old_ftests = tc.config['FTest'] or {}
    # An empty YAML entry (e.g. "Add:") is read as None
    return list(old_ftests.get('Add') or []), list(old_ftests.get('Remove') or [])


def checkFLogs(
    add_params: list[str],
    remove_params: list[str],
    tc: "TimingConfiguration"
) -> tuple[list[str], list[str]]:
    """
    Compares the F-test recommendations with previous ones stored in the config to avoid duplicates and
    add/remove loops.

    Parameters
    ----------
    add_params : list of str
        Parameters recommended to be added.
    remove_params : list of str
        Parameters recommended to be removed.
    tc : pint_pal.timingconfiguration.TimingConfiguration
        The pint_pal TimingConfiguration object.

    Returns
    -------
    (list of str, list of str)
        Sorted, unique lists of parameters to add and to remove.
    """
    add_params = sorted(set(add_params))
    remove_params = sorted(set(remove_params))

    old_ftests = _get_old_ftests(tc)
    if old_ftests is None:
        print("\nNo FTest in config.  Continuing...")
        return add_params, remove_params
    old_add, old_remove = old_ftests

    # - Compare old results with new to see if duplicates or loop occur
    for item in add_params:
        if item in old_remove:
            log.warning(f'Parameter {item} has already been recommended to be removed.')
    for item in remove_params:
        if item in old_add:
            log.warning(f'Parameter {item} has already been recommended to be added.')
    add_params = [item for item in add_params if item not in old_remove]
    remove_params = [item for item in remove_params if item not in old_add]

    return add_params, remove_params


def _process_test_results(
    results: dict,
    fitter: pint.fitter.Fitter,
    tc: "TimingConfiguration",
    results_log: bool | str | os.PathLike = False,
    alpha_add: float = ALPHA_ADD,
    alpha_remove: float = ALPHA_REMOVE
) -> dict:
    """
    Post-processing shared by `run_Ftests()` and `runFtest_parallel()`: arranges the results in the nested F-test
    results dictionary, prints the table of results and the recommendations, and prints the combined (new and
    previous) recommendations in a format to copy into the config .yaml file.

    New recommendations that would undo a previous one in the config (e.g. removing a parameter that was
    previously recommended to be added) are dropped with a warning by `checkFLogs()` before they are reported,
    so the printed statements and the .yaml block agree.

    Parameters
    ----------
    results : dict
        Raw output of the `TestFitter` fits, including the 'baseline' fit.
    fitter : pint.fitter.Fitter
        The PINT Fitter object, used for the section headings of the table.
    tc : pint_pal.timingconfiguration.TimingConfiguration
        The pint_pal TimingConfiguration object.
    results_log : bool, str or os.PathLike
        If True, write the raw results to results.json in the current directory; if a path, write them there
        instead [default: False].
    alpha_add : float
        F-statistic threshold for adding parameters [default: 0.0027].
    alpha_remove : float
        F-statistic threshold for removing parameters [default: 0.1336].

    Returns
    -------
    dict
        The nested F-test results dictionary (see `_build_Ftest_dict()`), with 'FTest' (the recommendations
        to add and remove, combined with those already in the config, as in the .yaml block) and 'fits' (the
        raw results).
    """
    if results_log:
        log_file = results_log if isinstance(results_log, (str, os.PathLike)) else "results.json"
        with open(log_file, "w") as fo:
            json.dump(results, fo, indent=4, default=par_fit.astropy_numpy_json_serializer)

    Ftest_dict = _build_Ftest_dict(results)
    print("Testing additional parameters (*** = significant):")
    print("\n".join(get_Ftest_lines(Ftest_dict, fitter, alpha_add=alpha_add, alpha_remove=alpha_remove)))

    new_ap, new_rp = parseParallelResults(results, alpha_add=alpha_add, alpha_remove=alpha_remove)
    # Drop new recommendations that would undo previous ones in the config
    new_ap, new_rp = checkFLogs(new_ap, new_rp, tc)

    # Combine with previous FTest results from the config
    old_ftests = _get_old_ftests(tc) or ([], [])
    ap = sorted(set(new_ap + old_ftests[0]))
    rp = sorted(set(new_rp + old_ftests[1]))

    print('\n')
    print('Copy and paste below into .yaml file:')
    print('FTest:\n  Add:', "[" + ",".join(ap) + "]", '\n  Remove:', "[" + ",".join(rp) + "]")
    print('\n')

    add_statement, remove_statement = _recommendation_statements(new_ap, new_rp)
    print(add_statement)
    print(remove_statement)

    Ftest_dict['FTest'] = {'Add': ap, 'Remove': rp}
    Ftest_dict['fits'] = results
    return Ftest_dict


# ---------------------------------------------------------------------------------------------------------
# Main functions
# ---------------------------------------------------------------------------------------------------------

def run_Ftests(
    fitter: pint.fitter.Fitter,
    tc: "TimingConfiguration",
    results_log: bool | str | os.PathLike = False,
    maxiter: int = 20,
    alpha_add: float = ALPHA_ADD,
    alpha_remove: float = ALPHA_REMOVE,
    FDnparams: int = 5,
    FBnparams: int = 5,
    alpha: float | None = None
) -> dict:
    """
    Runs the F-tests one at a time (serially), for computers that are not set up for parallel processing.
    This runs exactly the same fits as `runFtest_parallel()`, processes the results in the same way, and
    returns the same outputs.

    Parameters
    ----------
    fitter : pint.fitter.Fitter
        The PINT Fitter object. Its current model and TOAs are what is tested.
    tc : pint_pal.timingconfiguration.TimingConfiguration
        The pint_pal TimingConfiguration object.
    results_log : bool, str or os.PathLike
        If True, write the raw results to results.json in the current directory; if a path, write them there
        instead [default: False].
    maxiter : int
        Maximum number of fit iterations for each test [default: 20].
    alpha_add : float
        Parameters are recommended to be added if the F-statistic is lower than or equal to alpha_add, i.e.
        they are statistically significant to the timing model [default: 0.0027].
    alpha_remove : float
        Parameters are recommended to be removed if the F-statistic is greater than alpha_remove, i.e. they
        are not statistically significant to the timing model [default: 0.1336].
    FDnparams : int
        Maximum number of FD parameters to test [default: 5].
    FBnparams : int
        Maximum number of FB parameters to test, for ELL1 models using FB [default: 5].
    alpha : float or None
        Older name for alpha_add, kept so existing calls still work. If given, overrides alpha_add.

    Returns
    -------
    dict
        The nested F-test results dictionary (`Ftest_dict`), in the same layout as earlier versions of this
        module: 'initial', 'Add', 'Remove', 'F' and 'FB' (see `_build_Ftest_dict()`), plus 'FTest' (the
        recommendations to add and remove, combined with those already in the config, as in the .yaml block)
        and 'fits' (the raw output of the `TestFitter` fits).
    """
    if alpha is not None:
        alpha_add = alpha
    results = _run_test_fits(fitter, executor=_SerialExecutor(), maxiter=maxiter,
                             FDnparams=FDnparams, FBnparams=FBnparams)
    return _process_test_results(results, fitter, tc, results_log=results_log,
                                 alpha_add=alpha_add, alpha_remove=alpha_remove)


def runFtest_parallel(
    fitter: pint.fitter.Fitter,
    tc: "TimingConfiguration",
    ncpu: int | None = 4,
    results_log: bool | str | os.PathLike = False,
    maxiter: int = 20,
    alpha_add: float = ALPHA_ADD,
    alpha_remove: float = ALPHA_REMOVE,
    FDnparams: int = 5,
    FBnparams: int = 5
) -> dict:
    """
    Runs the F-tests in parallel with `pint_pal.parallel_fit`, prints the recommendations, and prints
    the combined (new and previous) recommendations in a format to copy into the config .yaml file.
    `run_Ftests()` runs exactly the same fits one at a time and returns the same outputs.

    Parameters
    ----------
    fitter : pint.fitter.Fitter
        The PINT Fitter object. Its current model and TOAs are what is tested.
    tc : pint_pal.timingconfiguration.TimingConfiguration
        The pint_pal TimingConfiguration object.
    ncpu : int or None
        Number of CPUs to run the fits on. If 1, the fits are run one at a time exactly as in `run_Ftests()`.
        If None, all available CPUs are used [default: 4].
    results_log : bool, str or os.PathLike
        If True, write the raw results to results.json in the current directory; if a path, write them there
        instead [default: False].
    maxiter : int
        Maximum number of fit iterations for each test [default: 20].
    alpha_add : float
        Parameters are recommended to be added if the F-statistic is lower than or equal to alpha_add, i.e.
        they are statistically significant to the timing model [default: 0.0027].
    alpha_remove : float
        Parameters are recommended to be removed if the F-statistic is greater than alpha_remove, i.e. they
        are not statistically significant to the timing model [default: 0.1336].
    FDnparams : int
        Maximum number of FD parameters to test [default: 5].
    FBnparams : int
        Maximum number of FB parameters to test, for ELL1 models using FB [default: 5].

    Returns
    -------
    dict
        The nested F-test results dictionary (`Ftest_dict`), in the same layout as earlier versions of this
        module: 'initial', 'Add', 'Remove', 'F' and 'FB' (see `_build_Ftest_dict()`), plus 'FTest' (the
        recommendations to add and remove, combined with those already in the config, as in the .yaml block)
        and 'fits' (the raw output of the `TestFitter` fits).
    """
    if ncpu is not None and ncpu < 1:
        raise ValueError(f"ncpu must be 1 or more (or None to use all CPUs), not {ncpu}.")
    # With one CPU, use the serial executor (as in run_Ftests()) rather than PINT's own single-CPU mode, which
    # reuses one fitter for every test and stops at the first failed fit, so it can give different results
    executor = _SerialExecutor() if ncpu == 1 else None
    results = _run_test_fits(fitter, executor=executor, ncpu=ncpu, maxiter=maxiter,
                             FDnparams=FDnparams, FBnparams=FBnparams)
    return _process_test_results(results, fitter, tc, results_log=results_log,
                                 alpha_add=alpha_add, alpha_remove=alpha_remove)
