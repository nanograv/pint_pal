# Generic imports
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from astropy import log
from multiprocessing import Pool

# Outlier/Epochalyptica imports
from pint.fitter import ConvergenceFailure
import copy
from scipy.special import fdtr
from pint_pal.utils import apply_cut_flag, apply_cut_select
from pint_pal.lite_utils import write_tim
from pint_pal.dmx_utils import *

# discovery outlier analysis imports
import discovery as ds
from discovery import matrix, selection_backend_flags
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from numpyro import sample, factor, infer, deterministic
from numpyro import distributions as dist

def gibbs_run(entPintPulsar,results_dir=None,Nsamples=10000):
    """
    Run Gibbs sampler for outlier analysis using enterprise PTA model.

    Parameters
    ----------
    entPintPulsar : enterprise.PintPulsar object
        The pulsar object for analysis.
    results_dir : str, optional
        Directory to save results.
    Nsamples : int, optional
        Number of samples to draw.

    Returns
    -------
    poutlier : np.ndarray
        Mean outlier probability chain.
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
    """
    Create an enterprise.PintPulsar object from PINT model and TOAs.

    Parameters
    ----------
    model : pint.model.TimingModel
        Timing model object.
    toas : pint.toa.TOAs
        TOAs object.
    sort : bool, optional
        Whether to sort TOAs.
    drop_pintpsr : bool, optional
        Whether to drop PINT pulsar info.

    Returns
    -------
    PintPulsar
        The enterprise PintPulsar object.
    """
    from enterprise.pulsar import PintPulsar
    return PintPulsar(toas,model,sort=sort,drop_pintpsr=drop_pintpsr)

def calculate_pout(model, toas, tc_object, outlier_sampler_kwargs={}):
    """
    Calculate TOA outlier probabilities and write tim file with flags.

    Parameters
    ----------
    model : pint.model.TimingModel
        Timing model object.
    toas : pint.toa.TOAs
        TOAs object.
    tc_object : pint_pal.timingconfiguration
        Timing configuration object.
    outlier_sampler_kwargs : dict, optional
        Additional keyword arguments for the outlier sampler.
        Should be passed via the config file outlier:sampler_kwargs.

    Returns
    -------
    None
    """
    method = tc_object.get_outlier_method()
    results_dir = f'outlier/{tc_object.get_outfile_basename()}'
    Nsamples = tc_object.get_outlier_samples()
    Nburnin = tc_object.get_outlier_burn()

    if method == 'enterprise-hmc':
        log.info('Running enterprise hmc outlier analysis...')
        epp = get_entPintPulsar(model, toas, drop_pintpsr=False)
        from enterprise_outliers.hmc_outlier import OutlierHMC
        pout = OutlierHMC(epp, outdir=results_dir, Nsamples=Nsamples, Nburnin=Nburnin)
        print('') # Progress bar doesn't print a newline
        # Some sorting will be needed here so pout refers to toas order?
    elif method == 'enterprise-gibbs':
        log.info('Running enterprise gibbs outlier analysis...')
        epp = get_entPintPulsar(model, toas)
        pout = gibbs_run(epp,results_dir=results_dir,Nsamples=Nsamples)
    elif method == 'discovery-gibbs':
        log.info('Running discovery-gibbs outlier analysis...')
        log.info('Loading enterprise pulsar object...')
        psr = get_entPintPulsar(model, toas, drop_pintpsr=False)
        # should put in a checker to make sure these priors match !!
        prior_dict = get_discovery_prior_dictionary()
        log.info('Setting up discovery outlier Gibbs sampler...')
        sampler = setup_sampler_discovery_outlier_analysis(psr, prior_dict=prior_dict)
        log.info('Beginning to sample...')
        sampler.run(jax.random.PRNGKey(0))
        log.info('Sampling complete. Calculating pout for all toas.')
        pout = np.mean(sampler.get_samples()['z_i'], axis=0)
    else:
        msg = 'Specified method ({method}) is not recognized.'
        msg+= 'Please use "enterprise-hmc", "enterprise-gibbs", or "discovery-gibbs".'
        log.error(msg)
        raise ValueError(msg)

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
    """
    Apply cut flags to TOAs with outlier probabilities above threshold and run DMX setup.

    Parameters
    ----------
    model : pint.model.TimingModel
        Timing model object.
    toas : pint.toa.TOAs
        TOAs object.
    tc_object : pint_pal.timingconfiguration
        Timing configuration object.
    outpct_threshold : float, optional
        Percentage threshold for outlier cuts.

    Returns
    -------
    None
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

    Parameters
    ----------
    chi2_1 : float
        Chi-squared of original model.
    dof_1 : int
        Degrees of freedom of original model.
    chi2_2 : float
        Chi-squared of new model.
    dof_2 : int
        Degrees of freedom of new model.

    Returns
    -------
    float or bool
        Probability that improvement is due to chance, or False if not improved.
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
    """
    Set global variable for multiprocessing epoch analysis.

    Parameters
    ----------
    model : pint.model.TimingModel
        Timing model object.
    toas : pint.toa.TOAs
        TOAs object.
    tc_object : pint_pal.timingconfiguration
        Timing configuration object.

    Returns
    -------
    None
    """
    global _epoch_args
    _epoch_args = (model, toas, tc_object)

def _test_one_epoch_args(filename):
    """
    Wrapper for test_one_epoch for multiprocessing.

    Parameters
    ----------
    filename : str
        Filename to test removal.

    Returns
    -------
    tuple
        Results from test_one_epoch.
    """
    return test_one_epoch(*_epoch_args, filename)

def test_one_epoch(model, toas, tc_object, filename):
    """
    Test chi2 for removal of one epoch (filename).  Used internally
    by epochalyptica().

    Parameters
    ----------
    model : pint.model.TimingModel
        Timing model object.
    toas : pint.toa.TOAs
        TOAs object.
    tc_object : pint_pal.timingconfiguration
        Timing configuration object.
    filename : str
        Filename to remove.

    Returns
    ------
    tuple
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

## discovery outlier analysis functions written by Pat Meyers
def makenoise_measurement_rescaled(psr, noisedict={}, scale=1.0, tnequad=False, selection=selection_backend_flags, vectorize=True):
    """
    Create a noise measurement matrix with rescaled errors for a pulsar.

    Parameters
    ----------
    psr : Pulsar object
        Pulsar to analyze.
    noisedict : dict, optional
        Noise parameters.
    scale : float, optional
        Scaling factor for errors.
    tnequad : bool, optional
        Use tnequad instead of t2equad.
    selection : function, optional
        Backend selection function.
    vectorize : bool, optional
        Whether to vectorize output.

    Returns
    -------
    NoiseMatrix1D_novar or NoiseMatrix1D_var
        Noise matrix for measurement errors.
    """
    backend_flags = selection(psr)
    backends = [b for b in sorted(set(backend_flags)) if b != '']

    efacs = [f'{psr.name}_{backend}_efac' for backend in backends]
    toaerr_scaling = f'{psr.name}_alpha_scaling({psr.toas.size})'
    if tnequad:
        log10_tnequads = [f'{psr.name}_{backend}_log10_tnequad' for backend in backends]
        params = efacs + log10_tnequads + [toaerr_scaling]
    else:
        log10_t2equads = [f'{psr.name}_{backend}_log10_t2equad' for backend in backends]
        params = efacs + log10_t2equads + [toaerr_scaling]

    masks = [(backend_flags == backend) for backend in backends]
    logscale = np.log10(scale)

    if all(par in noisedict for par in params):
        if tnequad:
            noise = sum(mask * (noisedict[efac]**2 * (scale * psr.toaerrs)**2 + 10.0**(2 * (logscale + noisedict[log10_tnequad])))
                        for mask, efac, log10_tnequad in zip(masks, efacs, log10_tnequads))
        else:
            noise = sum(mask * noisedict[efac]**2 * ((scale * psr.toaerrs)**2 + 10.0**(2 * (logscale + noisedict[log10_t2equad])))
                        for mask, efac, log10_t2equad in zip(masks, efacs, log10_t2equads))

        return matrix.NoiseMatrix1D_novar(noise)
    else:
        if vectorize:
            toaerrs2, masks = matrix.jnparray(scale**2 * psr.toaerrs**2), matrix.jnparray([mask for mask in masks])

            if tnequad:
                def getnoise(params):
                    efac2  = matrix.jnparray([params[efac]**2 for efac in efacs])
                    equad2 = matrix.jnparray([10.0**(2 * (logscale + params[log10_tnequad])) for log10_tnequad in log10_tnequads])
                    toaerrs2 = toaerrs2 * params[toaerr_scaling]
                    return (masks * (efac2[:,jnp.newaxis] * toaerrs2[jnp.newaxis,:] + equad2[:,jnp.newaxis])).sum(axis=0)
            else:
                def getnoise(params):
                    efac2  = matrix.jnparray([params[efac]**2 for efac in efacs])
                    equad2 = matrix.jnparray([10.0**(2 * (logscale + params[log10_t2equad])) for log10_t2equad in log10_t2equads])
                    return (masks * efac2[:,jnp.newaxis] * ((toaerrs2*params[toaerr_scaling])[jnp.newaxis,:] + equad2[:,jnp.newaxis])).sum(axis=0)
        else:
            toaerrs, masks = matrix.jnparray(scale * psr.toaerrs), [matrix.jnparray(mask) for mask in masks]
            if tnequad:
                def getnoise(params):
                    toaerrs2 = toaerrs2 * params[toaerr_scaling]
                    return sum(mask * (params[efac]**2 * toaerrs**2 + 10.0**(2 * (logscale + params[log10_tnequad])))
                               for mask, efac, log10_tnequad in zip(masks, efacs, log10_tnequads))
            else:
                def getnoise(params):
                    toaerrs2 = toaerrs2 * params[toaerr_scaling]
                    return sum(mask * params[efac]**2 * (toaerrs**2 + 10.0**(2 * (logscale + params[log10_t2equad])))
                               for mask, efac, log10_t2equad in zip(masks, efacs, log10_t2equads))

        getnoise.params = params
        return matrix.NoiseMatrix1D_var(getnoise)
        
def make_conditional_with_tm(psrl):
    """
    Create function to sample from conditional with variable timing model.

    Parameters
    ----------
    psrl : PulsarLikelihood object
        Pulsar likelihood object.

    Returns
    -------
    function
        Conditional sampling function.
    """
    Pvar = psrl.N.P_var
    Nvar = psrl.N.N
    F = psrl.N.F
    Pvar_inv = Pvar.make_inv()

    Nvar_solve_1d = Nvar.make_solve_1d()
    Nvar_solve_2d = Nvar.make_solve_2d()
    y = psrl.y
    def cond(params):

        # Nmy, ldN = Nvar_solve_1d(params, y)
        NmF, ldN = Nvar_solve_2d(params, F)
        FtNm = NmF.T
        FtNmy = FtNm @ y
        # FtNmy = F.T @ Nmy
        FtNmF = F.T @ NmF
        Pinv, ldP = Pvar_inv(params)
        Sigma = Pinv + FtNmF
        ch = jsl.cho_factor(Sigma)
        b_mean = jsl.cho_solve(ch, FtNmy)
        
        return b_mean, Sigma
    return cond

def make_sample_cond_with_tm(psrl):
    """
    Create function to sample from conditional with variable timing model and random key.

    Parameters
    ----------
    psrl : PulsarLikelihood object
        Pulsar likelihood object.

    Returns
    -------
    function
        Conditional sampling function with random key.
    """
    Pvar = psrl.N.P_var
    Nvar = psrl.N.N
    F = psrl.N.F
    Pvar_inv = Pvar.make_inv()

    Nvar_solve_1d = Nvar.make_solve_1d()
    Nvar_solve_2d = Nvar.make_solve_2d()
    y = psrl.y
    def cond(key, params):

        # Nmy, ldN = Nvar_solve_1d(params, y)
        NmF, ldN = Nvar_solve_2d(params, F)
        FtNm = NmF.T
        FtNmy = FtNm @ y
        # FtNmy = F.T @ Nmy
        FtNmF = F.T @ NmF
        Pinv, ldP = Pvar_inv(params)
        Sigma = Pinv + FtNmF
        ch = jsl.cho_factor(Sigma)
        b_mean = jsl.cho_solve(ch, FtNmy)

        noise = matrix.jsp.linalg.solve_triangular(ch[0].T, matrix.jnpnormal(key, (b_mean.shape)), lower=ch[1])
        # matrix.jsp.linalg.solve_triangular(cf[0].T, matrix.jnpnormal(subkey, mu.shape), lower=False)

        nkey, _ = matrix.jnpsplit(key)
        
        return nkey, {k: (noise + b_mean)[slc] for k, slc in psrl.N.index.items()}
    return cond

# def make_clogl(psrl, psr):
#     Pvar = psrl.N.P_var
#     Nvar = psrl.N.N
#     F = matrix.jnparray(psrl.N.F)
#     Pvar_inv = Pvar.make_inv()
#     Psolve_1d = Pvar.make_solve_1d()

#     Nvar_solve_1d = Nvar.make_solve_1d()
#     Nvar_solve_2d = Nvar.make_solve_2d()
#     cvars = psrl.N.index.keys()
#     y = psr.residuals
#     def clogl(params):
#         b = jnp.hstack([params[c] for c in cvars])
#         mean_residuals = F @ b
        
#         yprime = y - mean_residuals
#         # jax.debug.print('{x}', x=yprime[:10])
#         Nmyp, ldN = Nvar_solve_1d(params, yprime)
#         ypNmyp = yprime @ Nmyp

#         Pmb, ldP = Psolve_1d(params, b)
#         bPmb = b @ Pmb
        
#         return -0.5 * (ypNmyp + bPmb + ldP + ldN)
#     return clogl

def make_single_psr_model_scaled_errors(psr, noisedict={}, tm_variable=False):
    """
    Build single pulsar likelihood model with scaled measurement errors.

    Parameters
    ----------
    psr : Pulsar object
        Pulsar to analyze.
    noisedict : dict, optional
        Noise parameters.
    tm_variable : bool, optional
        Whether timing model is variable.

    Returns
    -------
    PulsarLikelihood
        The pulsar likelihood model.
    """
    psrl = ds.PulsarLikelihood([psr.residuals,
                            ds.makegp_timing(psr, svd=True, variable=tm_variable),
                            ds.makegp_ecorr(psr, noisedict=noisedict),      
                            makenoise_measurement_rescaled(psr, noisedict=noisedict),
                            ds.makegp_fourier(psr, ds.freespectrum, 30, name='red_noise')], concat=True)
    return psrl


def make_single_psr_model(psr, noisedict={}, tm_variable=False):
    """
    Build single pulsar likelihood model with default measurement errors.

    Parameters
    ----------
    psr : Pulsar object
        Pulsar to analyze.
    noisedict : dict, optional
        Noise parameters.
    tm_variable : bool, optional
        Whether timing model is variable.

    Returns
    -------
    PulsarLikelihood
        The pulsar likelihood model.
    """
    psrl = ds.PulsarLikelihood([psr.residuals,
                            ds.makegp_timing(psr, svd=True, variable=tm_variable),
                            ds.makegp_ecorr(psr, noisedict=noisedict),      
                            ds.makenoise_measurement(psr, noisedict=noisedict),
                            ds.makegp_fourier(psr, ds.freespectrum, 30, name='red_noise')], concat=True)
    return psrl


def make_gibbs_fn(psrl, prior_dict):
    """
    Create Gibbs sampling function for outlier analysis.

    Parameters
    ----------
    psrl : PulsarLikelihood object
        Pulsar likelihood object.
    prior_dict : dict
        Dictionary of prior parameters.

    Returns
    -------
    function
        Gibbs sampling function.
    """
    make_Nalpha = psrl.N.N.getN
    Nalpha = psrl.N.N
    Nalpha_solve_2d = Nalpha.make_solve_2d()
    Nalpha_solve_1d = Nalpha.make_solve_1d()
    num_resids = psrl.y.size
    ones = jnp.ones(num_resids)
    Tmat = psrl.N.F
    mval = 0.01
    y = psrl.y
    ecorr_params = [p for p in psrl.logL.params if 'ecorr' in p]
    efac_params = [p for p in psrl.logL.params if 'efac' in p]
    equad_params = [p for p in psrl.logL.params if 'equad' in p]
    jcond = jax.jit(make_sample_cond_with_tm(psrl))
    cvars = list(psrl.N.index.keys())
    def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        # draw coefficients
        pardict = hmc_sites.copy()
        pardict.update({efn: ef for ef, efn in zip(hmc_sites['efacs'], efac_params)})
        pardict.update({eqn: eq for eq, eqn in zip(hmc_sites['equads'], equad_params)})
        pardict.update({ecn: ec for ec, ecn in zip(hmc_sites['ecorrs'], ecorr_params)})

        # update with alpha scaling
        pardict.update({f'{psrl.name}_alpha_scaling({y.size})': gibbs_sites['alpha_i']**gibbs_sites['z_i']})
        nkey, coeffs = jcond(rng_key, pardict)

        

        # turn into a single arary, store
        pardict.update(coeffs)
        coeffs = jnp.hstack([coeffs[c] for c in cvars])
        
        means = Tmat @ coeffs
        gibbs_sites['coeffs'] = coeffs
    
        # theta
        gibbs_sites['theta'] = sample("theta", dist.Beta(num_resids * mval + jnp.sum(gibbs_sites['z_i']),
                                                  num_resids * (1-mval) + num_resids - jnp.sum(gibbs_sites['z_i'])),
                                           rng_key=nkey)
    
        # z_i's
        norm_prob_alpha = jnp.exp(dist.Normal(means,jnp.sqrt(make_Nalpha(pardict))).log_prob(y))
        pardict.update({f'{psrl.name}_alpha_scaling({y.size})': ones})
        norm_prob_reg = jnp.exp(dist.Normal(means,jnp.sqrt(make_Nalpha(pardict))).log_prob(y))

        nnkey, _ = jax.random.split(nkey)
        theta = gibbs_sites['theta']

        q = theta * norm_prob_alpha / (theta * norm_prob_alpha + (1 - theta)*norm_prob_reg)
        
        q = jnp.where(q<1, q, 1)
        gibbs_sites['q'] = q
    
        q = deterministic('q', q)
        gibbs_sites['z_i'] = sample("z_i", dist.Binomial(1, q), rng_key=nnkey)
    
        # alphas
        n3key, _ = jax.random.split(nnkey)
        
        yprime = y - means
        tot = yprime @ Nalpha_solve_1d(pardict, y)[0]
        top = 0.5 * (1 + gibbs_sites['z_i'] * tot)
        bot = sample("alpha_i", dist.Gamma(0.5 * (1 + gibbs_sites['z_i'])), rng_key=n3key)
        gibbs_sites['alpha_i'] = top / bot
        return gibbs_sites
    
    return gibbs_fn


def make_numpyro_outlier_model(psrl, psr):
    """
    Create a numpyro model for outlier analysis for a single pulsar.

    Parameters
    ----------
    psrl : PulsarLikelihood object
        Pulsar likelihood object.
    psr : Pulsar object
        Pulsar to analyze.

    Returns
    -------
    function
        Numpyro model function.
    """
    jclogl = jax.jit(psrl.clogL)
    ecorr_high = -5
    ecorr_low = -8.5
    efac_high = 10
    efac_low = 0.1
    equad_high = -5
    equad_low = -8.5
    ecorr_params = [p for p in psrl.logL.params if 'ecorr' in p]
    efac_params = [p for p in psrl.logL.params if 'efac' in p]
    equad_params = [p for p in psrl.logL.params if 'equad' in p]
    def model(rng_key=None):
        # wn params
        pardict = {}
        efacs = sample('efacs', dist.Uniform(efac_low, efac_high).expand([len(efac_params)]), rng_key=rng_key)
        equads = sample('equads', dist.Uniform(equad_low, equad_high).expand([len(equad_params)]), rng_key=rng_key)
        ecorrs = sample('ecorrs', dist.Uniform(ecorr_low, ecorr_high).expand([len(ecorr_params)]), rng_key=rng_key)
        pardict = {efn: ef for ef, efn in zip(efacs, efac_params)}
        pardict.update({eqn: eq for eq, eqn in zip(equads, equad_params)})
        pardict.update({ecn: ec for ec, ecn in zip(ecorrs, ecorr_params)})
    
        # coefficients (doesn't matter, gibbs takes care of this)
        coeffs = sample("coeffs", dist.Uniform(-1e-4, 1e-4).expand([psrl.N.F.shape[-1]]), rng_key=rng_key)
    
        # rn rhos
        log10_rhos = sample(f'{psr.name}_red_noise_log10_rho(30)', dist.Uniform(-9, -4).expand([30]), rng_key=rng_key)
        pardict[f'{psr.name}_red_noise_log10_rho(30)'] = log10_rhos      
    
        # theta (doesn't matter, gibbs takes care of this)
        theta = sample("theta", dist.Uniform(0, 1), rng_key=rng_key)
    
        # z_i (doesn't matter, gibbs takes care of this)
        z_i = sample("z_i", dist.Uniform(0, 1).expand([psr.residuals.size]), rng_key=rng_key)
        q = sample("q", dist.Uniform(0, 1).expand([psr.residuals.size]), rng_key=rng_key)
        # alpha_i (doesn't matter, gibbs takes care of this)
        alpha_i = sample("alpha_i", dist.Uniform(0, 100).expand([psr.residuals.size]), rng_key=rng_key)
        pardict.update({f'{psr.name}_alpha_scaling({psr.residuals.size})': alpha_i**z_i})

        # put coefficients in right form for clogl
        pardict.update({k: coeffs[slc] for k, slc in psrl.N.index.items()})
        factor('clogl', jclogl(pardict))
    return model

def setup_sampler_discovery_outlier_analysis(psr, tc_object, prior_dict={}):
    """
    Setup the numpyro sampler for a discovery outlier analysis.

    Parameters
    ----------
    psr : `enterprise.pulsar` object
    
    tc_object : `pint_pal.TimingConfiguration` object
        this gets used to set up the sampler kwargs.
    
    prior_dict : dict, optional
        if running with non-standard priors, need to update them via this dictionary

    Returns
    -------
    sampler : `numpyro.infer.mcmc.MCMC`
        numpyro object which is ready to perform MonteCarlo sampling
    """
    psrl = make_single_psr_model_scaled_errors(psr, tm_variable=True)
    gibbs_fn = make_gibbs_fn(psrl, prior_dict)
    jgibbs = jax.jit(gibbs_fn)
    numpyro_model = make_numpyro_outlier_model(psrl, psr)
    hmc_kernel = infer.NUTS(numpyro_model, max_tree_depth=10, target_accept_prob=0.99)
    kernel = infer.HMCGibbs(hmc_kernel, gibbs_fn=jgibbs, gibbs_sites=['theta','z_i', 'alpha_i', 'coeffs', 'q'])
    outlier_kwargs = tc_object.config['outlier']
    # need to pass relevant outlier kwargs into the sampler
    sampler = infer.MCMC(kernel, num_warmup=20, num_samples=100, num_chains=1)
    return sampler

def get_discovery_prior_dictionary(override_dict={}):
    """
    Get the prior dictionary for discovery outlier analysis.

    Parameters
    ----------
    override_dict : dict
        Dictionary of parameters to override the default priors.

    Returns
    -------
    dict : dict
        Dictionary of priors.
    """
    prior_dict = ds.priordict_standard.copy()
    prior_dict.update({f'(.*_)?alpha_scaling\\(([0-9]*)\\)': [0.999, 1.001]})
    prior_dict.update({f'(.*_)?timingmodel_coefficients\\(([0-9]*)\\)': [0.999, 1.001]})
    for key, value in override_dict.items():
        prior_dict[key] = value
    return prior_dict
