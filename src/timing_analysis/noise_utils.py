import numpy as np, os
from astropy import log
from astropy.time import Time

from enterprise.pulsar import Pulsar
from enterprise_extensions import models, model_utils, sampler
import corner

import pint.models as pm
from pint.models.parameter import maskParameter

import matplotlib.pyplot as pl

#Imports necessary for e_e noise modeling functions
import functools
from collections import OrderedDict

from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise import constants as const

from enterprise_extensions import model_utils
from enterprise_extensions import deterministic
from enterprise_extensions.timing import timing_block
#from enterprise_extensions.blocks import (white_noise_block, red_noise_block)

import types

from enterprise.signals import utils
from enterprise.signals import gp_priors as gpp

def white_noise_block(vary=False, inc_ecorr=False, gp_ecorr=False,
                      efac1=False, select='backend', name=None, is_wideband = False, wb_efac_sigma = 0.25):
    """
    Returns the white noise block of the model:
        1. EFAC per backend/receiver system
        2. EQUAD per backend/receiver system
        3. ECORR per backend/receiver system
    :param vary:
        If set to true we vary these parameters
        with uniform priors. Otherwise they are set to constants
        with values to be set later.
    :param inc_ecorr:
        include ECORR, needed for NANOGrav channelized TOAs
    :param gp_ecorr:
        whether to use the Gaussian process model for ECORR
    :param efac1:
        use a strong prior on EFAC = Normal(mu=1, stdev=0.1)
    :param is_wideband:
        flag to toggle special normal prior for wideband EFAC and DMEFAC
    :param wb_efac_sigma:
        width of normal prior for wideband EFAC and DMEFAC.
    """

    if select == 'backend':
        # define selection by observing backend
        backend = selections.Selection(selections.by_backend)
        # define selection by nanograv backends
        backend_ng = selections.Selection(selections.nanograv_backends)
    else:
        # define no selection
        backend = selections.Selection(selections.no_selection)

    # white noise parameters
    if vary:
        if efac1:
            efac = parameter.Normal(1.0, 0.1)
        else:
            if is_wideband:
                efac = parameter.Normal(1.0, wb_efac_sigma)
            else:
                efac = parameter.Uniform(0.01, 10.0)
        equad = parameter.Uniform(-8.5, -5)
        if inc_ecorr:
            ecorr = parameter.Uniform(-8.5, -5)
    else:
        efac = parameter.Constant()
        equad = parameter.Constant()
        if inc_ecorr:
            ecorr = parameter.Constant()

    # white noise signals
    ef = white_signals.MeasurementNoise(efac=efac,
                                        selection=backend, name=name)
    eq = white_signals.EquadNoise(log10_equad=equad,
                                  selection=backend, name=name)
    if inc_ecorr:
        if gp_ecorr:
            if name is None:
                ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr,
                                                selection=backend_ng)
            else:
                ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr,
                                                selection=backend_ng, name=name)

        else:
            ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr,
                                                selection=backend_ng,
                                                name=name)

    # combine signals
    if inc_ecorr:
        s = ef + eq + ec
    elif not inc_ecorr:
        s = ef + eq

    return s

def red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=None,
                    components=30, gamma_val=None, coefficients=False,
                    select=None, modes=None, wgts=None,
                    break_flat=False, break_flat_fq=None):
    """
    Returns red noise model:
        1. Red noise modeled as a power-law with 30 sampling frequencies
    :param psd:
        PSD function [e.g. powerlaw (default), turnover, spectrum, tprocess]
    :param prior:
        Prior on log10_A. Default if "log-uniform". Use "uniform" for
        upper limits.
    :param Tspan:
        Sets frequency sampling f_i = i / Tspan. Default will
        use overall time span for indivicual pulsar.
    :param components:
        Number of frequencies in sampling of red noise
    :param gamma_val:
        If given, this is the fixed slope of the power-law for
        powerlaw, turnover, or tprocess red noise
    :param coefficients: include latent coefficients in GP model?
    """
    # red noise parameters that are common
    if psd in ['powerlaw', 'powerlaw_genmodes', 'turnover',
               'tprocess', 'tprocess_adapt', 'infinitepower']:
        # parameters shared by PSD functions
        if prior == 'uniform':
            log10_A = parameter.LinearExp(-20, -11)
        elif prior == 'log-uniform' and gamma_val is not None:
            if np.abs(gamma_val - 4.33) < 0.1:
                log10_A = parameter.Uniform(-20, -11)
            else:
                log10_A = parameter.Uniform(-20, -11)
        else:
            log10_A = parameter.Uniform(-20, -11)

        if gamma_val is not None:
            gamma = parameter.Constant(gamma_val)
        else:
            #gamma = parameter.Uniform(0, 7)
            ##########This is specific for timing##################
            gamma = parameter.Uniform(1.2, 7)

        # different PSD function parameters
        if psd == 'powerlaw':
            pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
        elif psd == 'powerlaw_genmodes':
            pl = gpp.powerlaw_genmodes(log10_A=log10_A, gamma=gamma, wgts=wgts)
        elif psd == 'turnover':
            kappa = parameter.Uniform(0, 7)
            lf0 = parameter.Uniform(-9, -7)
            pl = utils.turnover(log10_A=log10_A, gamma=gamma,
                                lf0=lf0, kappa=kappa)
        elif psd == 'tprocess':
            df = 2
            alphas = gpp.InvGamma(df/2, df/2, size=components)
            pl = gpp.t_process(log10_A=log10_A, gamma=gamma, alphas=alphas)
        elif psd == 'tprocess_adapt':
            df = 2
            alpha_adapt = gpp.InvGamma(df/2, df/2, size=1)
            nfreq = parameter.Uniform(-0.5, 10-0.5)
            pl = gpp.t_process_adapt(log10_A=log10_A, gamma=gamma,
                                    alphas_adapt=alpha_adapt, nfreq=nfreq)
        elif psd == 'infinitepower':
            pl = gpp.infinitepower()

    if psd == 'spectrum':
        if prior == 'uniform':
            log10_rho = parameter.LinearExp(-10, -4, size=components)
        elif prior == 'log-uniform':
            log10_rho = parameter.Uniform(-10, -4, size=components)

        pl = gpp.free_spectrum(log10_rho=log10_rho)

    if select == 'backend':
        # define selection by observing backend
        selection = selections.Selection(selections.by_backend)
    elif select == 'band' or select == 'band+':
        # define selection by observing band
        selection = selections.Selection(selections.by_band)
    else:
        # define no selection
        selection = selections.Selection(selections.no_selection)

    if break_flat:
        log10_A_flat = parameter.Uniform(-20, -11)
        gamma_flat = parameter.Constant(0)
        pl_flat = utils.powerlaw(log10_A=log10_A_flat, gamma=gamma_flat)

        freqs = 1.0 * np.arange(1, components+1) / Tspan
        components_low = sum(f < break_flat_fq for f in freqs)
        if components_low < 1.5:
            components_low = 2

        rn = gp_signals.FourierBasisGP(pl, components=components_low,
                                       Tspan=Tspan, coefficients=coefficients,
                                       selection=selection)

        rn_flat = gp_signals.FourierBasisGP(pl_flat,
                                            modes=freqs[components_low:],
                                            coefficients=coefficients,
                                            selection=selection,
                                            name='red_noise_hf')
        rn = rn + rn_flat
    else:
        rn = gp_signals.FourierBasisGP(pl, components=components,
                                       Tspan=Tspan,
                                       coefficients=coefficients,
                                       selection=selection,
                                       modes=modes)

    if select == 'band+':  # Add the common component as well
        rn = rn + gp_signals.FourierBasisGP(pl, components=components,
                                            Tspan=Tspan,
                                            coefficients=coefficients)

    return rn

def model_singlepsr_noise(psr, tm_var=False, tm_linear=False,
                          tmparam_list=None,
                          red_var=True, psd='powerlaw', red_select=None,
                          noisedict=None, tm_svd=False, tm_norm=True,
                          white_vary=True, components=30, upper_limit=False,
                          is_wideband=False, use_dmdata=False,
                          dmjump_var=False, gamma_val=None, extra_sigs=None,
                          select='backend', wb_efac_sigma = 0.25, coefficients=False):
    """
    Single pulsar noise model
    :param psr: enterprise pulsar object
    :param tm_var: explicitly vary the timing model parameters
    :param tm_linear: vary the timing model in the linear approximation
    :param tmparam_list: an explicit list of timing model parameters to vary
    :param red_var: include red noise in the model
    :param psd: red noise psd model
    :param noisedict: dictionary of noise parameters
    :param tm_svd: boolean for svd-stabilised timing model design matrix
    :param tm_norm: normalize the timing model, or provide custom normalization
    :param white_vary: boolean for varying white noise or keeping fixed
    :param components: number of modes in Fourier domain processes
    :param upper_limit: whether to do an upper-limit analysis
    :param is_wideband: whether input TOAs are wideband TOAs; will exclude
           ecorr from the white noise model
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
           is_wideband
    :param gamma_val: red noise spectral index to fix
    :param extra_sigs: Any additional `enterprise` signals to be added to the
        model.
        
    :return s: single pulsar noise model
    """
    
    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # timing model
    if not tm_var:
        if (is_wideband and use_dmdata):
            if dmjump_var:
                dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
            else:
                dmjump = parameter.Constant()
            if white_vary:
                #dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
                ################Timing specific#######################
                dmefac = parameter.Normal(1.0, wb_efac_sigma)
                log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
                #dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
            else:
                dmefac = parameter.Constant()
                log10_dmequad = parameter.Constant()
                #dmjump = parameter.Constant()
            s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                    log10_dmequad=log10_dmequad, dmjump=dmjump,
                    dmefac_selection=selections.Selection(
                        selections.by_backend),
                    log10_dmequad_selection=selections.Selection(
                        selections.by_backend),
                    dmjump_selection=selections.Selection(
                        selections.by_frontend))
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd, normed=tm_norm,
                                   coefficients=coefficients)
    else:
        # create new attribute for enterprise pulsar object
        psr.tmparams_orig = OrderedDict.fromkeys(psr.t2pulsar.pars())
        for key in psr.tmparams_orig:
            psr.tmparams_orig[key] = (psr.t2pulsar[key].val,
                                      psr.t2pulsar[key].err)
        if not tm_linear:
            s = timing_block(tmparam_list=tmparam_list)
        else:
            pass

    # red noise
    if red_var:
        s += red_noise_block(psd=psd, prior=amp_prior,
                             components=components, gamma_val=gamma_val,
                             coefficients=coefficients, select=red_select)

    if extra_sigs is not None:
        s += extra_sigs
        
    # adding white-noise, and acting on psr objects
    if 'NANOGrav' in psr.flags['pta'] and not is_wideband:
        s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                select=select, is_wideband = False, wb_efac_sigma = wb_efac_sigma)
        model = s2(psr)
    else:
        s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                select=select, is_wideband = True, wb_efac_sigma = wb_efac_sigma)
        model = s3(psr)

    # set up PTA
    pta = signal_base.PTA([model])

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta

def analyze_noise(chaindir = './noise_run_chains/', burn_frac = 0.25, save_corner = True):
    """
    Reads enterprise chain file; produces and saves corner plot; returns WN dictionary and RN (SD) BF

    Parameters
    ==========
    chaindir: path to enterprise noise run chain; Default: './noise_run_chains/'
    burn_frac: fraction of chain to use for burn-in; Default: 0.25
    save_corner: Flag to toggle saving of corner plots; Default: True

    Returns
    =======
    wn_dict: Dictionary of maximum likelihood WN values
    rn_bf: Savage-Dickey BF for RN for given pulsar
    """

    chainfile = chaindir + 'chain_1.txt'
    chain = np.loadtxt(chainfile)
    burn = int(burn_frac * chain.shape[0])
    pars = np.loadtxt(chaindir + 'pars.txt', dtype = str)

    psr_name = pars[0].split('_')[0]

    if save_corner:
        pars_short = [p.split("_",1)[1] for p in pars]
        log.info(f"Chain parameter names are {pars_short}")
        corner.corner(chain[burn:, :-4], labels = pars_short)

        if '_wb' in chaindir:
            figname = f"./{psr_name}_noise_corner_wb.pdf"
        elif '_nb' in chaindir:
            figname = f"./{psr_name}_noise_corner_nb.pdf"
        else:
            figname = f"./{psr_name}_noise_corner.pdf"

        pl.savefig(figname)
        pl.savefig(figname.replace(".pdf",".png"), dpi=300)

        pl.show()

    ml_idx = np.argmax(chain[burn:, -4])

    wn_vals = chain[burn:, :-4][ml_idx]

    wn_dict = dict(zip(pars, wn_vals))

    #Print bayes factor for red noise in pulsar
    rn_bf = model_utils.bayes_fac(chain[burn:, -5])[0]

    return wn_dict, rn_bf

def model_noise(mo, to, vary_red_noise = True, n_iter = int(1e5), using_wideband = False, resume = False, run_noise_analysis = True, wb_efac_sigma = 0.25, base_op_dir = "./"):
    """
    Setup enterprise PTA and perform MCMC noise analysis

    Parameters
    ==========
    mo: PINT (or tempo2) timing model
    to: PINT (or tempo2) TOAs
    red_noise: include red noise in the model
    n_iter: number of MCMC iterations; Default: 1e5; Recommended > 5e4
    using_wideband: Flag to toggle between narrowband and wideband datasets; Default: False
    run_noise_analysis: Flag to toggle execution of noise modeling; Default: True

    Returns
    =======
    None
    """

    if not using_wideband:
        outdir = base_op_dir + mo.PSR.value + '_nb/'
    else:
        outdir = base_op_dir + mo.PSR.value + '_wb/'

    if os.path.exists(outdir) and (run_noise_analysis) and (not resume):
        log.info("INFO: A noise directory for pulsar {} already exists! Re-running noise modeling from scratch".format(mo.PSR.value))
    elif os.path.exists(outdir) and (run_noise_analysis) and (resume):
        log.info("INFO: A noise directory for pulsar {} already exists! Re-running noise modeling starting from previous chain".format(mo.PSR.value))

    if not run_noise_analysis:
        log.info("Skipping noise modeling. Change run_noise_analysis = True to run noise modeling.")
        return None

    #Ensure n_iter is an integer
    n_iter = int(n_iter)

    if n_iter < 1e4:
        log.warning("Such a small number of iterations is unlikely to yield accurate posteriors. STRONGLY recommend increasing the number of iterations to at least 5e4")

    #Create enterprise Pulsar object for supplied pulsar timing model (mo) and toas (to)
    e_psr = Pulsar(mo, to)

    #Setup a single pulsar PTA using enterprise_extensions
    if not using_wideband:
        pta = model_singlepsr_noise(e_psr, white_vary = True, red_var = vary_red_noise, is_wideband = False, use_dmdata = False, dmjump_var = False, wb_efac_sigma = wb_efac_sigma)
    else:
        pta = model_singlepsr_noise(e_psr, is_wideband = True, use_dmdata = True, white_vary = True, red_var = vary_red_noise, dmjump_var = False, wb_efac_sigma = wb_efac_sigma)
        dmjump_params = {}
        for param in mo.params:
            if param.startswith('DMJUMP'):
                dmjump_param = getattr(mo,param)
                dmjump_param_name = f"{pta.pulsars[0]}_{dmjump_param.key_value[0]}_dmjump"
                dmjump_params[dmjump_param_name] = dmjump_param.value
        pta.set_default_params(dmjump_params)

    #setup sampler using enterprise_extensions
    samp = sampler.setup_sampler(pta, outdir = outdir, resume = resume)

    #Initial sample
    x0 = np.hstack([p.sample() for p in pta.params])

    #Start sampling

    samp.sample(x0, n_iter, SCAMweight=30, AMweight=15, DEweight=50,)
    
def convert_to_RNAMP(value):
    """
    Utility function to convert enterprise RN amplitude to tempo2/PINT parfile RN amplitude
    """
    return (86400.*365.24*1e6)/(2.0*np.pi*np.sqrt(3.0)) * 10 ** value

def add_noise_to_model(model, burn_frac = 0.25, save_corner = True, ignore_red_noise = False, using_wideband = False, rn_bf_thres = 1e2, base_dir = None):
    """
    Add WN and RN parameters to timing model.

    Parameters
    ==========
    model: PINT (or tempo2) timing model
    burn_frac: fraction of chain to use for burn-in; Default: 0.25
    save_corner: Flag to toggle saving of corner plots; Default: True
    ignore_red_noise: Flag to manually force RN exclusion from timing model. When False,
        code determines whether
    RN is necessary based on whether the RN BF > 1e3. Default: False
    using_wideband: Flag to toggle between narrowband and wideband datasets; Default: False
    base_dir: directory containing {psr}_nb and {psr}_wb chains directories; if None, will
        check for results in the current working directory './'.

    Returns
    =======
    model: New timing model which includes WN and RN parameters
    """

    # Assume results are in current working directory if not specified
    if not base_dir:
        base_dir = './'

    if not using_wideband:
        chaindir = os.path.join(base_dir,f'{model.PSR.value}_nb/')
    else:
        chaindir = os.path.join(base_dir,f'{model.PSR.value}_wb/')

    log.info(f'Using existing noise analysis results in {chaindir}')
    log.info('Adding new noise parameters to model.')
    wn_dict, rn_bf = analyze_noise(chaindir, burn_frac, save_corner)
    chainfile = chaindir + 'chain_1.txt'
    mtime = Time(os.path.getmtime(chainfile), format="unix")
    log.info(f"Noise chains loaded from {chainfile} created at {mtime.isot}")

    #Create the maskParameter for EFACS
    efac_params = []
    equad_params = []
    rn_params = []

    if not using_wideband:
        ecorr_params = []
    else:
        dmefac_params = []
        dmequad_params = []

    ii = 0
    idx = 0

    for key, val in wn_dict.items():

        if not using_wideband:
            if ii % 3 == 0:
                idx += 1
        else:
            if ii % 5 == 0:
                idx += 1

        psr_name = key.split('_')[0]

        if '_efac' in key:

            param_name = key.split('_efac')[0].split(psr_name)[1][1:]

            tp = maskParameter(name = 'EFAC', index = idx, key = '-f', key_value = param_name,
                               value = val, units = '')
            efac_params.append(tp)

        elif '_equad' in key:

            param_name = key.split('_equad')[0].split(psr_name)[1].split('_log10')[0][1:]

            tp = maskParameter(name = 'EQUAD', index = idx, key = '-f', key_value = param_name,
                               value = 10 ** val / 1e-6, units = 'us')
            equad_params.append(tp)

        elif ('_ecorr' in key) and (not using_wideband):

            param_name = key.split('_ecorr')[0].split(psr_name)[1].split('_log10')[0][1:]

            tp = maskParameter(name = 'ECORR', index = idx, key = '-f', key_value = param_name,
                               value = 10 ** val / 1e-6, units = 'us')
            ecorr_params.append(tp)

        elif ('_dmefac' in key) and (using_wideband):

            param_name = key.split('_dmefac')[0].split(psr_name)[1][1:]

            tp = maskParameter(name = 'DMEFAC', index = idx, key = '-f', key_value = param_name,
                               value = val, units = '')
            dmefac_params.append(tp)

        elif ('_dmequad' in key) and (using_wideband):

            param_name = key.split('_dmequad')[0].split(psr_name)[1].split('_log10')[0][1:]

            tp = maskParameter(name = 'DMEQUAD', index = idx, key = '-f', key_value = param_name,
                               value = 10 ** val, units = 'pc/cm3')
            dmequad_params.append(tp)

        ii += 1

    ef_eq_comp = pm.ScaleToaError()

    if using_wideband:
        dm_comp = pm.noise_model.ScaleDmError()
    else:
        ec_comp = pm.EcorrNoise()

    #Remove the default parameters that come with these components
    ef_eq_comp.remove_param(param = 'EFAC1')
    ef_eq_comp.remove_param(param = 'EQUAD1')
    ef_eq_comp.remove_param(param = 'TNEQ1')
    if using_wideband:
        dm_comp.remove_param(param = 'DMEFAC1')
        dm_comp.remove_param(param = 'DMEQUAD1')
    else:
        ec_comp.remove_param('ECORR1')

    #Add the above ML WN parameters to their respective components
    for ii in range(len(efac_params)):

        ef_eq_comp.add_param(param = efac_params[ii], setup = True)
        ef_eq_comp.add_param(param = equad_params[ii], setup = True)
        if not using_wideband:
            ec_comp.add_param(param = ecorr_params[ii], setup = True)
        else:
            dm_comp.add_param(param = dmefac_params[ii])
            dm_comp.add_param(param = dmequad_params[ii], setup = True)

    #Add the ML RN parameters to their component
    
    log.info(f"The SD Bayes factor for red noise in this pulsar is: {rn_bf}")
    if (rn_bf >= rn_bf_thres or np.isnan(rn_bf)) and (not ignore_red_noise):

        log.info("Including red noise for this pulsar")
        #Add the ML RN parameters to their component
        rn_comp = pm.PLRedNoise()

        rn_keys = np.array([key for key,val in wn_dict.items() if '_red_' in key])
        rn_comp.RNAMP.quantity = convert_to_RNAMP(wn_dict[psr_name + '_red_noise_log10_A'])
        rn_comp.RNIDX.quantity = -1 * wn_dict[psr_name + '_red_noise_gamma']

        #Add red noise to the timing model
        model.add_component(rn_comp, validate = True, force = True)
    else:
        log.info("Not including red noise for this pulsar")

    #Add these components to the input timing model
    model.add_component(ef_eq_comp, validate = True, force = True)
    if not using_wideband:
        model.add_component(ec_comp, validate = True, force = True)
    else:
        model.add_component(dm_comp, validate = True, force = True)

    #Setup and validate the timing model to ensure things are correct
    model.setup()
    model.validate()
    model.noise_mtime = mtime.isot

    return model
