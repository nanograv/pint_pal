# Generic imports
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as log
from multiprocessing import Pool

# Outlier/Epochalyptica imports
from pint.fitter import ConvergenceFailure
import copy
from scipy.special import fdtr
from pint_pal.utils import apply_cut_flag, apply_cut_select
from pint_pal.lite_utils import write_tim
from pint_pal.dmx_utils import *

#######################################
#### Discovery outliers functions ####
#######################################

def make_outlier_likelihood_discovery(psr, noise_dict=None, tspan=None, model_kwargs=None):
    """
    Build a discovery ``PulsarLikelihood`` configured for outlier analysis.

    This is a thin wrapper around
    :func:`~pint_pal.discovery_utils.make_single_pulsar_noise_likelihood_discovery`
    that enforces the three requirements of the Wang & Taylor (2022) HMC-Gibbs
    outlier model:

    1. ``outliers=True`` on the measurement noise — introduces the per-TOA
       variance-scaling parameter ``alpha_scaling``.
    2. ``gp_ecorr=True`` — uses the GP-basis ECORR instead of the kernel
       ECORR (required for ``variable=True`` ECORR via
       ``psrl.sample_conditional``).
    3. ``variable=True`` on the ECORR GP — coefficients are sampled rather
       than marginalised so the Gibbs draws can update them.
    4. ``tm_marg=False`` — timing-model GP coefficients are sampled rather
       than marginalised (i.e. ``variable=True`` on the timing GP).

    The caller's ``model_kwargs`` dict is deep-copied so it is never mutated.

    Parameters
    ----------
    psr : Any
        Pulsar object.
    noise_dict : dict, optional
        Noise parameter dictionary. Default is None.
    tspan : float, optional
        Time span for the noise model. Default is None.
    model_kwargs : dict, optional
        Model keyword arguments (same schema as
        ``make_single_pulsar_noise_likelihood_discovery``). Forced keys:
        ``timing_model.tm_marg=False``, ``white_noise.gp_ecorr=True``,
        ``white_noise.variable=True``, ``white_noise.outliers=True``.

    Returns
    -------
    discovery.PulsarLikelihood
        Likelihood configured for outlier detection.
    """
    from pint_pal.discovery_utils import make_single_pulsar_noise_likelihood_discovery
    import copy

    mk = copy.deepcopy(model_kwargs) if model_kwargs is not None else {}

    # -- timing model: sample coefficients rather than marginalise
    if not mk.get('timing_model'):
        mk['timing_model'] = {}
    mk['timing_model']['tm_marg'] = False

    # -- white noise: GP-basis ECORR, variable coefficients, outlier scaling
    if not mk.get('white_noise'):
        mk['white_noise'] = {}
    mk['white_noise']['gp_ecorr'] = True
    mk['white_noise']['variable'] = True
    mk['white_noise']['outliers'] = True

    return make_single_pulsar_noise_likelihood_discovery(
        psr, noise_dict=noise_dict, tspan=tspan, model_kwargs=mk
    )


def run_outlier_analysis_nuts(
    mo,
    to,
    outdir,
    *,
    model_kwargs=None,
    sampler_kwargs=None,
    seed=42,
    resume=False,
    return_sampler_without_sampling=False,
):
    """Run the Wang & Taylor (2022) HMC-Gibbs outlier analysis for a single pulsar.

    1. Builds the outlier-configured ``PulsarLikelihood`` via
       :func:`make_outlier_likelihood_discovery` (enforces ``outliers=True``,
       ``gp_ecorr=True``, ``variable=True``, ``tm_marg=False``).
    2. Merges the standard pint_pal prior dict with the discovery outlier
       priors (``nu``, ``theta_m``).
    3. Builds the numpyro model + Gibbs function from
       ``discovery.models.nanograv_single_pulsar_outlier``.
    4. Wraps them in a ``numpyro.infer.HMCGibbs`` kernel and runs with
       checkpoints via :func:`~pint_pal.discovery_utils.run_nuts_with_checkpoints`.

    The saved feather file contains one column per named numpyro site
    (``efacs``, ``equads``, ``ecorrs``, ``nu``, any RN hyper names,
    ``theta``, ``z_i``, ``alpha_i``, ``q``, ``coeffs``, ``loglike``).
    Vector sites are expanded to ``<name>_0``, ``<name>_1``, … columns.

    Parameters
    ----------
    mo : `pint.model.TimingModel` object
    to : `pint.toa.TOAs` object
    outdir : str
        Directory to write checkpoint / feather output files.
    model_kwargs : dict, optional
        Model configuration (same schema as ``model_noise``).  The outlier
        requirements (``tm_marg``, ``gp_ecorr``, ``variable``, ``outliers``)
        are enforced on top of whatever is passed.
    sampler_kwargs : dict, optional
        Sampler configuration.  Recognised keys (with defaults):

        - ``num_warmup`` (500)
        - ``num_samples`` (2000)
        - ``num_samples_per_checkpoint`` (500)
        - ``max_tree_depth`` (6)
        - ``target_accept_prob`` (0.8)
        - ``diagnostics`` (True)
    seed : int, optional
        Random seed. Default 42.
    resume : bool, optional
        Resume from an existing checkpoint. Default False.
    return_sampler_without_sampling : bool, optional
        If True, build and return the ``numpyro.infer.MCMC`` object without
        running it. Default False.

    Returns
    -------
    numpyro.infer.MCMC or None
        The MCMC object when *return_sampler_without_sampling* is True;
        otherwise None.
    """
    import json
    import numpy as np
    import pandas as pd
    import jax
    import numpyro.infer
    from jax.random import PRNGKey

    from discovery.models.nanograv_single_pulsar_outlier import (
        make_outlier_model,
        make_outlier_gibbs_fn,
        priordict_outlier_default,
        _init_values_from_priordict,
    )
    from pint_pal import discovery_utils as disco_utils
    from enterprise.pulsar import Pulsar

    if model_kwargs is None:
        model_kwargs = {}
    if sampler_kwargs is None:
        sampler_kwargs = {}

    # get enterprise_pulsar object
    e_psr = Pulsar(mo, to)
    log.info(f"Setting up outlier analysis (discovery HMC-Gibbs) for {e_psr.name}")
    os.makedirs(outdir, exist_ok=True)

    # Build outlier PulsarLikelihood
    psrl = make_outlier_likelihood_discovery(
        psr=e_psr,
        noise_dict={},
        tspan=None,
        model_kwargs=model_kwargs,
    )

    # Build prior dict: standard pint_pal priors + outlier extras
    from discovery import priordict_standard as ds_pdict
    prior_dict = priordict_outlier_default.copy()
    pint_pal_priors = json.load(
        open(os.path.join(os.path.dirname(__file__), "discovery_priors.json"))
    )
    prior_dict.update(pint_pal_priors)
    # restore outlier-specific keys that pint_pal_priors may not contain
    for k, v in priordict_outlier_default.items():
        prior_dict.setdefault(k, v)

    # Build the numpyro outlier model + Gibbs function
    model = make_outlier_model(psrl, priordict=prior_dict)
    gibbs_fn = make_outlier_gibbs_fn(psrl)

    init = _init_values_from_priordict(psrl, prior_dict)
    init_strategy = numpyro.infer.util.init_to_value(values=init)

    # Build the HMCGibbs sampler
    nuts_kernel = numpyro.infer.NUTS(
        model,
        init_strategy=init_strategy,
        max_tree_depth=sampler_kwargs.get("max_tree_depth", 6),
        target_accept_prob=sampler_kwargs.get("target_accept_prob", 0.8),
    )
    kernel = numpyro.infer.HMCGibbs(
        nuts_kernel,
        gibbs_fn=jax.jit(gibbs_fn),
        gibbs_sites=["theta", "z_i", "alpha_i", "coeffs", "q"],
    )
    mcmc = numpyro.infer.MCMC(
        kernel,
        num_warmup=sampler_kwargs.get("num_warmup", 500),
        num_samples=sampler_kwargs.get("num_samples", 2000),
    )

    # Attach a to_df method: flatten vector sites to "<name>_i" columns,
    # drop the nested "params" deterministic (it's a dict-of-arrays).
    def _outlier_to_df():
        raw = mcmc.get_samples(group_by_chain=False)
        rows = {}
        for key, val in raw.items():
            if key == "params":  # nested dict — skip
                continue
            arr = np.asarray(val)
            if arr.ndim == 1:          # scalar site
                rows[key] = arr
            else:                      # vector site: expand to key_0, key_1, …
                for i in range(arr.shape[-1]):
                    rows[f"{key}_{i}"] = arr[:, i]
        return pd.DataFrame(rows)

    mcmc.to_df = _outlier_to_df

    if return_sampler_without_sampling:
        return mcmc

    # Run with checkpoints
    disco_utils.run_nuts_with_checkpoints(
        sampler=mcmc,
        num_samples_per_checkpoint=sampler_kwargs.get("num_samples_per_checkpoint", 500),
        rng_key=PRNGKey(seed),
        outdir=outdir,
        file_name=f"{e_psr.name}_outlier_nuts_samples",
        resume=resume,
        diagnostics=sampler_kwargs.get("diagnostics", True),
        model=None,  # loglike already a deterministic in the samples
    )
    return None


## enterprise outlier analysis below ##


def gibbs_run(entPintPulsar,results_dir=None,Nsamples=10000):
    """Necessary set-up to run gibbs sampler, and run it. Return pout.
    """
    # Imports
    import enterprise.signals.parameter as parameter
    from enterprise.signals import utils
    from enterprise.signals import signal_base
    from enterprise.signals.selections import Selection
    from enterprise.signals import white_signals
    from enterprise.signals import gp_signals
    from enterprise.signals.selections import Selection
    from enterprise.signals import selections
    from enterprise.signals import deterministic_signals
    from enterprise_outliers.gibbs_outlier import OutlierGibbs

    # white noise
    efac = parameter.Uniform(0.01,10.0)
    t2equad = parameter.Uniform(-10, -4)
    ecorr = parameter.Uniform(-10, -4)
    selection = selections.Selection(selections.by_backend)

    # white noise
    mn = white_signals.MeasurementNoise(efac=efac, log10_t2equad=t2equad, selection=selection)
    ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=selection)

    # red noise
    pl = utils.powerlaw(log10_A=parameter.Uniform(-18,-11),gamma=parameter.Uniform(0,7))
    rn = gp_signals.FourierBasisGP(spectrum=pl, components=30)

    # timing model
    tm = gp_signals.TimingModel()

    # combined signal
    s = mn + ec + rn + tm 

    # PTA
    pta = signal_base.PTA([s(entPintPulsar)])

    # Steve's code
    gibbs = OutlierGibbs(pta, model='mixture', vary_df=True,theta_prior='beta', vary_alpha=True)
    params = np.array([p.sample() for p in gibbs.params]).flatten()
    gibbs.sample(params, outdir=results_dir,niter=Nsamples, resume=False)
    poutlier = np.mean(gibbs.poutchain, axis = 0)

    #return np.mean(gibbs.poutchain, axis = 0)
    return poutlier

def get_entPintPulsar(model,toas,sort=False,drop_pintpsr=True):
    """Return enterprise.PintPulsar object

    Parameters
    ==========
    model: `pint.model.TimingModel` object
    toas: `pint.toa.TOAs` object
    sort: bool
        optional, default: False
    drop_pintpsr: bool
        optional, default: True; PintPulsar retains model/toas if False

    Returns
    =======
    model: `enterprise.PintPulsar` object
    """
    from enterprise.pulsar import PintPulsar
    return PintPulsar(toas,model,sort=sort,drop_pintpsr=drop_pintpsr)

def calculate_pout(model, toas, tc_object):
    """Determines TOA outlier probabilities using choices specified in the
    timing configuration file's outlier block. Write tim file with pout flags/values.

    Parameters
    ==========
    model: `pint.model.TimingModel` object
    toas: `pint.toa.TOAs` object
    tc_object: `pint_pal.timingconfiguration` object
    """
    method = tc_object.get_outlier_method()
    results_dir = f'outlier/{tc_object.get_outfile_basename()}'
    Nsamples = tc_object.get_outlier_samples()
    Nburnin = tc_object.get_outlier_burn()

    if method == 'hmc':
        epp = get_entPintPulsar(model, toas, drop_pintpsr=False)
        from enterprise_outliers.hmc_outlier import OutlierHMC
        pout = OutlierHMC(epp, outdir=results_dir, Nsamples=Nsamples, Nburnin=Nburnin)
        print('') # Progress bar doesn't print a newline
        # Some sorting will be needed here so pout refers to toas order?
    elif method == 'gibbs':
        epp = get_entPintPulsar(model, toas)
        pout = gibbs_run(epp,results_dir=results_dir,Nsamples=Nsamples)
    else:
        log.error(f'Specified method ({method}) is not recognized.')

    # Apply pout flags, cuts
    for i,oi in enumerate(toas.table['index']):
        toas.orig_table[oi]['flags'][f'pout_{method}'] = str(pout[i])

    # Re-introduce cut TOAs for writing tim file that includes -cut/-pout flags
    toas.table = toas.orig_table
    fo = tc_object.construct_fitter(toas,model)
    pout_timfile = f'{results_dir}/{tc_object.get_outfile_basename()}_pout.tim'
    write_tim(fo,toatype=tc_object.get_toa_type(),outfile=pout_timfile)

    # Need to mask TOAs once again
    apply_cut_select(toas,reason='resumption after write_tim, pout')

def make_pout_cuts(model,toas,tc_object,outpct_threshold=8.0):
    """Apply cut flags to TOAs with outlier probabilities larger than specified threshold.
    Also runs setup_dmx.

    Parameters
    ==========
    toas: `pint.toa.TOAs` object
    tc_object: `pint_pal.timingconfiguration` object
    outpct_threshold: float, optional
       cut file's remaining TOAs (maxout) if X% were flagged as outliers (default set by 5/64=8%) 
    """
    toas = tc_object.apply_ignore(toas,specify_keys=['prob-outlier'])
    apply_cut_select(toas,reason='outlier analysis, specified key')
    toas = setup_dmx(model,toas,frequency_ratio=tc_object.get_fratio(),max_delta_t=tc_object.get_sw_delay())

    # Now cut files if X% or more TOAs/file are flagged as outliers
    if tc_object.get_toa_type() == 'NB':
        tc_object.check_file_outliers(toas,outpct_threshold=outpct_threshold)
        toas = setup_dmx(model,toas,frequency_ratio=tc_object.get_fratio(),max_delta_t=tc_object.get_sw_delay())
    else:
        log.info('Skipping maxout cuts (wideband).')

def Ftest(chi2_1, dof_1, chi2_2, dof_2):
    """
    Ftest(chi2_1, dof_1, chi2_2, dof_2):
        Compute an F-test to see if a model with extra parameters is
        significant compared to a simpler model.  The input values are the
        (non-reduced) chi^2 values and the numbers of DOF for '1' the
        original model and '2' for the new model (with more fit params).
        The probability is computed exactly like Sherpa's F-test routine
        (in Ciao) and is also described in the Wikipedia article on the
        F-test:  http://en.wikipedia.org/wiki/F-test
        The returned value is the probability that the improvement in
        chi2 is due to chance (i.e. a low probability means that the
        new fit is quantitatively better, while a value near 1 means
        that the new model should likely be rejected).
        If the new model has a higher chi^2 than the original model,
        returns value of False
    """
    delta_chi2 = chi2_1 - chi2_2
    if delta_chi2 > 0:
      delta_dof = dof_1 - dof_2
      new_redchi2 = chi2_2 / dof_2
      F = (delta_chi2 / delta_dof) / new_redchi2
      ft = 1.0 - fdtr(delta_dof, dof_2, F)
    else:
      ft = False
    return ft

# This global var allows the (unpickleable) PINT model object
# to be passed to the multiprocessing workers in epochalyptica.
_epoch_args = None

def _set_epoch_args(model, toas, tc_object):
    """Sets arguments for test_one_epoch() into a global variable 
    for use in multiprocessing."""
    global _epoch_args
    _epoch_args = (model, toas, tc_object)

def _test_one_epoch_args(filename):
    """Single-argument wrapper function for test_one_epoch() for use with
    multiprocessing."""
    return test_one_epoch(*_epoch_args, filename)

def test_one_epoch(model, toas, tc_object, filename):
    """Test chi2 for removal of one epoch (filename).  Used internally
    by epochalyptica().

    Returns:
      receiver - receiver name of the removed file
      mjd - MJD of the removed file
      chi2 - post-fit chi2 after removing the file
      ndof - post-fit NDOF after removing the file
      ntoas - number of TOAs remaining after removal
      esum - weighted sum of removed TOA uncertainties
    """
    using_wideband = tc_object.get_toa_type() == 'WB'
    log.info(f"Testing removal of {filename} ntoas={toas.ntoas}")

    maskarray = np.ones(toas.ntoas,dtype=bool)
    receiver = None
    mjd = None
    toaval = None
    dmxindex = None
    dmxlower = None
    dmxupper = None
    esum = 0.0
    # Note, t[1]: mjd, t[2]: mjd (d), t[3]: error (us), t[6]: flags dict
    for index,t in enumerate(toas.table):
        if t[6]['name'] == filename:
            if receiver == None:
                receiver = t[6]['f']
            if mjd == None:
                mjd = int(t[1].value)
            if toaval == None:
                toaval = t[2]
                i = 1
                while dmxindex == None:
                    DMXval = f"DMXR1_{i:04d}"
                    lowerbound = getattr(model.components['DispersionDMX'],DMXval).value
                    DMXval = f"DMXR2_{i:04d}"
                    upperbound = getattr(model.components['DispersionDMX'],DMXval).value
                    if toaval > lowerbound and toaval < upperbound:
                        dmxindex = f"{i:04d}"
                        dmxlower = lowerbound
                        dmxupper = upperbound
                    i += 1
            esum = esum + 1.0 / (float(t[3])**2.0)
            maskarray[index] = False

    toas.select(maskarray)
    numtoas_in_dmxrange = 0
    for toa in toas.table:
        if toa[2] > dmxlower and toa[2] < dmxupper:
            numtoas_in_dmxrange += 1
    newmodel = model
    if numtoas_in_dmxrange == 0:
        log.debug(f"Removing DMX range {dmxindex}")
        newmodel = copy.deepcopy(model)
        newmodel.components['DispersionDMX'].remove_param(f'DMXR1_{dmxindex}')
        newmodel.components['DispersionDMX'].remove_param(f'DMXR2_{dmxindex}')
        newmodel.components['DispersionDMX'].remove_param(f'DMX_{dmxindex}')
    f = tc_object.construct_fitter(toas,newmodel)
    try:
        f.fit_toas(maxiter=tc_object.get_niter())
    except ConvergenceFailure:
        log.info('Failed to converge; moving on with best result.')
    ndof, chi2 = f.resids.dof, f.resids.chi2
    ntoas = toas.ntoas
    esum = 1.0 / np.sqrt(esum)
    toas.unselect()
    return receiver, mjd, chi2, ndof, ntoas, esum

def epochalyptica(model,toas,tc_object,ftest_threshold=1.0e-6,nproc=1):
    """ Test for the presence of remaining bad epochs (files) by removing one at a
        time and examining its impact on the residuals; pre/post reduced
        chi-squared values are assessed using an F-statistic.  

    Parameters:
    ===========
    model: `pint.model.TimingModel` object
    toas: `pint.toa.TOAs` object
    tc_object: `pint_pal.timingconfiguration` object
    ftest_threshold: float
        optional, threshold below which files will be dropped
    nproc: number of parallel processes to use for tests
    """
    using_wideband = tc_object.get_toa_type() == 'WB'
    f_init = tc_object.construct_fitter(toas,model)
    try:
        f_init.fit_toas(maxiter=tc_object.get_niter())
    except ConvergenceFailure:
        log.info('Failed to converge; moving on with best result.')
    ndof_init, chi2_init = f_init.resids.dof, f_init.resids.chi2
    ntoas_init = toas.ntoas  # How does this change for wb?
    redchi2_init = chi2_init / ndof_init

    filenames = sorted(set(toas.get_flag_value('name')[0]))
    outdir = f'outlier/{tc_object.get_outfile_basename()}'
    outfile = os.path.join(outdir,'epochdrop.txt')

    # Check for existence of path and make directories if they don't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fout = open(outfile,'w')
    numfiles = len(filenames)
    log.info(f'There are {numfiles} files to analyze.')
    files_to_drop = []

    # Run tests in parallel
    results = None
    if nproc > 1:
        with Pool(nproc, _set_epoch_args, (f_init.model, toas, tc_object)) as p:
            results = p.map(_test_one_epoch_args, filenames)

    for i, filename in enumerate(filenames):
        if results is not None:
            receiver, mjd, chi2, ndof, ntoas, esum = results[i]
        else: 
            receiver, mjd, chi2, ndof, ntoas, esum = test_one_epoch(f_init.model, toas, tc_object, filename)
        redchi2 = chi2 / ndof
        log.debug(f"After masking TOA(s) from {filename}...")
        log.debug(f"ndof init: {ndof_init}, ndof trial: {ndof}; chi2 init: {chi2_init}, chi2 trial: {chi2}")
        if ndof_init != ndof:
            ftest = Ftest(float(chi2_init),int(ndof_init),float(chi2),int(ndof))
            if ftest < ftest_threshold: files_to_drop.append(filename)
            log.debug(f"ftest: {ftest}")
        else:
            ftest = False
        fout.write(f"{filename} {receiver} {mjd:d} {(ntoas_init - ntoas):d} {ftest:e} {esum}\n")
        fout.flush()
    fout.close()

    # Apply cut flags
    names = np.array([f['name'] for f in toas.orig_table['flags']])
    for ftd in files_to_drop:
        filedropinds = np.where(names==ftd)[0]
        apply_cut_flag(toas,filedropinds,'epochdrop')

    # Make cuts, fix DMX windows if necessary
    if len(files_to_drop):
        apply_cut_select(toas,reason='epoch drop analysis')
        toas = setup_dmx(model,toas,frequency_ratio=tc_object.get_fratio(),max_delta_t=tc_object.get_sw_delay())
    else:
        log.info('No files dropped (epochalyptica).')

    # Re-introduce cut TOAs for writing tim file that includes -cut flags
    toas.table = toas.orig_table
    fo = tc_object.construct_fitter(toas,model)
    excise_timfile = f'{outdir}/{tc_object.get_outfile_basename()}_excise.tim'
    write_tim(fo,toatype=tc_object.get_toa_type(),outfile=excise_timfile)

    # Need to mask TOAs once again
    apply_cut_select(toas,reason='resumption after write_tim (excise)')
