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
