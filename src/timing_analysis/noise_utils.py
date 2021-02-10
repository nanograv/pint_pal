import numpy as np, os
from astropy import log

from enterprise.pulsar import Pulsar
from enterprise_extensions import models, model_utils, sampler
import corner

import pint.models as pm
from pint.models.parameter import maskParameter

import matplotlib.pyplot as pl

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

    chain = np.loadtxt(chaindir + 'chain_1.txt')
    burn = int(burn_frac * chain.shape[0])
    pars = np.loadtxt(chaindir + 'pars.txt', dtype = np.unicode)

    psr_name = pars[0].split('_')[0]

    if save_corner:
        corner.corner(chain[burn:, :-4], labels = pars)

        if '_wb' in chaindir:
            figname = f"./{psr_name}_noise_corner.wb.pdf"
        elif '_nb' in chaindir:
            figname = f"./{psr_name}_noise_corner.nb.pdf"
        else:
            figname = f"./{psr_name}_noise_corner.pdf"

        pl.savefig(figname)

        pl.show()

    ml_idx = np.argmax(chain[burn:, -3])

    wn_vals = chain[burn:, :-4][ml_idx]

    wn_dict = dict(zip(pars, wn_vals))

    #Print bayes factor for red noise in pulsar
    rn_bf = model_utils.bayes_fac(chain[burn:, -5])[0]

    return wn_dict, rn_bf

def model_noise(mo, to, red_noise = True, n_iter = int(1e5), using_wideband = False, resume = False, run_noise_analysis = True):
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
        outdir = './noise_run_chains/' + mo.PSR.value + '_nb/'
    else:
        outdir = './noise_run_chains/' + mo.PSR.value + '_wb/'

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
        pta = models.model_singlepsr_noise(e_psr, white_vary = True, red_var = red_noise, is_wideband = False, use_dmdata = False, dmjump_var = False)
    else:
        pta = models.model_singlepsr_noise(e_psr, is_wideband = True, use_dmdata = True, white_vary = True, red_var = red_noise, dmjump_var = False)
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

def add_noise_to_model(model, burn_frac = 0.25, save_corner = True, ignore_red_noise = False, using_wideband = False, rn_bf_thres = 1e2):
    """
    Add WN and RN parameters to timing model.

    Parameters
    ==========
    model: PINT (or tempo2) timing model
    burn_frac: fraction of chain to use for burn-in; Default: 0.25
    save_corner: Flag to toggle saving of corner plots; Default: True
    ignore_red_noise: Flag to manually force RN exclusion from timing model. When False, code determines whether
    RN is necessary based on whether the RN BF > 1e3. Default: False
    using_wideband: Flag to toggle between narrowband and wideband datasets; Default: False

    Returns
    =======
    model: New timing model which includes WN and RN parameters
    """

    if not using_wideband:
        chaindir = './noise_run_chains/' + model.PSR.value + '_nb/'
    else:
        chaindir = './noise_run_chains/' + model.PSR.value + '_wb/'

    wn_dict, rn_bf = analyze_noise(chaindir, burn_frac, save_corner)

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
    #CONDITIONAL TO ADD RN;
    #MIGHT NEED TO FIDDLE WITH THIS

    msg = f"The SD Bayes factor for red noise in this pulsar is: {rn_bf}"
    log.info(msg)
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

    return model
