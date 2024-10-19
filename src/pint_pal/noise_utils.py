import numpy as np, os, json
import arviz as az
from astropy import log
from astropy.time import Time

from enterprise.pulsar import Pulsar
from enterprise_extensions import models, model_utils, sampler
import corner

import pint.models as pm
from pint.models.parameter import maskParameter

import matplotlib as mpl
import matplotlib.pyplot as pl

import la_forge.core as co
import la_forge.diagnostics as dg
import la_forge.utils as lu

# Imports necessary for e_e noise modeling functions
import functools
from collections import OrderedDict

from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise import constants as const

from enterprise_extensions.sampler import group_from_params, get_parameter_groups
from enterprise_extensions import model_utils
from enterprise_extensions import deterministic
from enterprise_extensions.timing import timing_block

# from enterprise_extensions.blocks import (white_noise_block, red_noise_block)

import types

from enterprise.signals import utils
from enterprise.signals import gp_priors as gpp


def setup_sampling_groups(pta,
                          write_groups=True,
                          outdir='./'):
    """
    Sets sampling groups for PTMCMCSampler.
    The sampling groups can help ensure the sampler does not get stuck.
    The idea is to group parameters which are more highly correlated.

    Params
    ------
    pta: the enterprise pta object 
    write_groups: bool, write the groups to a file
    outdir: str, directory to write the groups to
    
    returns
    -------
    groups: list of lists of indices corresponding to parameter groups
    
    """
        
        # groups
    pnames = pta.param_names
    groups = get_parameter_groups(pta)
    # add per-backend white noise
    backends = np.unique([p[p.index('_')+1:p.index('efac')-1] for p in pnames if 'efac' in p])
    for be in backends:
        groups.append(group_from_params(pta,[be]))
    # group red noise parameters
    exclude = ['linear_timing_model','sw_r2','sw_4p39','measurement_noise',
            'ecorr_sherman-morrison', 'ecorr_fast-sherman-morrison']
    red_signals = [p[p.index('_')+1:] for p in list(pta.signals.keys())
                if not p[p.index('_')+1:] in exclude]
    rn_ct = 0
    for rs in red_signals:
        if len(group_from_params(pta,[rs])) > 0:
            rn_ct += 1
            groups.append(group_from_params(pta,[rs]))
    if rn_ct > 1:
        groups.append(group_from_params(pta,red_signals))
    # add cross chromatic groups
    if 'n_earth' in pnames or 'log10_sigma_ne' in pnames:
        # cross SW and chrom groups
        dmgp_sw = [idx for idx, nm in enumerate(pnames)
                if any([flag in nm for flag in ['dm_gp','n_earth', 'log10_sigma_ne']])]
        groups.append(dmgp_sw)
        if np.any(['chrom' in param for param in pnames]):
            chromgp_sw = [idx for idx, nm in enumerate(pnames)
                        if any([flag in nm for flag in ['chrom_gp','n_earth', 'log10_sigma_ne']])]
            dmgp_chromgp_sw = [idx for idx, nm in enumerate(pnames)
                            if any([flag in nm for flag in ['dm_gp','chrom','n_earth', 'log10_sigma_ne']])]
            groups.append(chromgp_sw)
            groups.append(dmgp_chromgp_sw)
    if np.any(['chrom' in param for param in pnames]):
        # cross dmgp and chromgp group
        dmgp_chromgp = [idx for idx, nm in enumerate(pnames)
                        if any([flag in nm for flag in ['dm_gp','chrom']])]
        groups.append(dmgp_chromgp)
    # everything
    groups.append([i for i in range(len(pnames))])
    # save list of params corresponding to groups
    if write_groups is True:
        with open(f'{outdir}/groups.txt', 'w') as fi:
            for group in groups:
                line = np.array(pnames)[np.array(group)]
                fi.write("[" + " ".join(line) + "]\n")
    # return the groups to be passed to the sampler
    return groups


def analyze_noise(
    chaindir="./noise_run_chains/",
    burn_frac=0.25,
    save_corner=True,
    no_corner_plot=False,
    chaindir_compare=None,
    which_sampler = 'PTMCMCSampler',
):
    """
    Reads enterprise chain file; produces and saves corner plot; returns WN dictionary and RN (SD) BF

    Parameters
    ==========
    chaindir: path to enterprise noise run chain; Default: './noise_run_chains/'
    burn_frac: fraction of chain to use for burn-in; Default: 0.25
    save_corner: Flag to toggle saving of corner plots; Default: True
    no_corner_plot: Flag to toggle saving of corner plots; Default: False
    chaindir_compare: path to noise run chain wish to plot in corner plot for comparison; Default: None
    which_sampler: choose from ['PTMCMCSampler' or 'GibbsSampler' or 'discovery']

    Returns
    =======
    noise_core: la_forge.core object which contains noise chains and run metadata
    noise_dict: Dictionary of maximum a posterior noise values
    rn_bf: Savage-Dickey BF for achromatic RN for given pulsar
    """
    try:
        noise_core = co.Core(chaindir=chaindir)
    except:
        log.error(f"Could not load noise run from {chaindir}")
        ValueError(f"Could not load noise run from {chaindir}")
    if which_sampler == 'PTMCMCSampler':
        noise_core.set_burn(burn_frac)
    elif which_sampler == 'discovery':
        noise_core.set_burn(0)
    else:
        noise_core.set_burn(burn_frac)
    chain = noise_core.chain
    psr_name = noise_core.params[0].split("_")[0]
    pars =  np.array([p for p in noise_core.params if p not in ['lnlike', 'lnpost']])
    if len(pars)+2 != chain.shape[1]:
        chain = chain[:, :len(pars)+2]
    
    # load in same for comparison noise model
    if chaindir_compare is not None:
        compare_core = co.Core(chaindir=chaindir) 
        compare_core.set_burn(noise_core.burn)
        chain_compare = compare_core.chain
        pars_compare = np.array([p for p in compare_core.params if p not in ['lnlike', 'lnpost']])
        if len(pars_compare)+2 != chain_compare.shape[1]:
            chain_compare = chain_compare[:, :len(pars_compare)+2]

        psr_name_compare = pars_compare[0].split("_")[0]
        if psr_name_compare != psr_name:
            log.warning(
                f"Pulsar name from {chaindir_compare} does not match. Will not plot comparison"
            )
            chaindir_compare = None

    if save_corner and not no_corner_plot:
        pars_short = [p.split("_", 1)[1] for p in pars]
        log.info(f"Chain parameter names are {pars_short}")
        log.info(f"Chain parameter convention: {test_equad_convention(pars_short)}")
        if chaindir_compare is not None:
            # need to plot comparison corner plot first so it's underneath
            compare_pars_short = [p.split("_", 1)[1] for p in pars_compare]
            log.info(f"Comparison chain parameter names are {compare_pars_short}")
            log.info(
               f"Comparison chain parameter convention: {test_equad_convention(compare_pars_short)}"
            )
            # don't plot comparison if the parameter names don't match
            if compare_pars_short != pars_short:
                log.warning(
                   "Parameter names for comparison noise chains do not match, not plotting the compare-noise-dir chains"
                )
                chaindir_compare = None
            else:
                normalization_factor = (
                    np.ones(len(chain_compare))
                    * len(chain)
                    / len(chain_compare)
                )
                fig = corner.corner(
                    chain_compare,
                    color="orange",
                    alpha=0.5,
                    weights=normalization_factor,
                    labels=compare_pars_short,
                )
                # normal corner plot
                corner.corner(
                    chain, fig=fig, color="black", labels=pars_short
                )
        if chaindir_compare is None:
            corner.corner(chain, labels=pars_short)

        if "_wb" in chaindir:
            figname = f"./{psr_name}_noise_corner_wb.pdf"
        elif "_nb" in chaindir:
            figname = f"./{psr_name}_noise_corner_nb.pdf"
        else:
            figname = f"./{psr_name}_noise_corner.pdf"

        pl.savefig(figname)
        pl.savefig(figname.replace(".pdf", ".png"), dpi=300)

        pl.show()

    if no_corner_plot:

        from matplotlib.backends.backend_pdf import PdfPages

        if "_wb" in chaindir:
            figbase = f"./{psr_name}_noise_posterior_wb"
        elif "_nb" in chaindir:
            figbase = f"./{psr_name}_noise_posterior_nb"
        else:
            figbase = f"./{psr_name}_noise_posterior"

        pars_short = [p.split("_", 1)[1] for p in pars]
        log.info(f"Chain parameter names are {pars_short}")
        log.info(f"Chain parameter convention: {test_equad_convention(pars_short)}")
        if chaindir_compare is not None:
            # need to plot comparison corner plot first so it's underneath
            compare_pars_short = [p.split("_", 1)[1] for p in pars_compare]
            log.info(f"Comparison chain parameter names are {compare_pars_short}")
            log.info(
                f"Comparison chain parameter convention: {test_equad_convention(compare_pars_short)}"
            )
            # don't plot comparison if the parameter names don't match
            if compare_pars_short != pars_short:
                log.warning(
                    "Parameter names for comparison noise chains do not match, not plotting the compare-noise-dir chains"
                )
                chaindir_compare = None
            else:
                normalization_factor = (
                    np.ones(len(chain_compare))
                    * len(chain)
                    / len(chain_compare)
                )

        # Set the shape of the subplots
        shape = pars.shape[0]

        if "_wb" in chaindir:
            ncols = 4  # number of columns per page
        else:
            ncols = 3

        nrows = 5  # number of rows per page

        mp_idx = noise_core.map_idx
        #mp_idx = np.argmax(chain[:, a])
        if chaindir_compare is not None:
            mp_compare_idx = compare_core.map_idx

        nbins = 20
        pp = 0
        for idx, par in enumerate(pars_short):
            j = idx % (nrows * ncols)
            if j == 0:
                pp += 1
                fig = pl.figure(figsize=(8, 11))

            ax = fig.add_subplot(nrows, ncols, j + 1)
            ax.hist(
                chain[:, idx],
                bins=nbins,
                histtype="step",
                color="black",
                label="Current",
            )
            ax.axvline(chain[:, idx][mp_idx], ls="--", color="black")
            if chaindir_compare is not None:
                ax.hist(
                    chain_compare[:, idx],
                    bins=nbins,
                    histtype="step",
                    color="orange",
                    label="Past",
                )
                ax.axvline(
                    chain_compare[:, idx][mp_compare_idx], ls="--", color="orange"
                )
            if "_wb" in chaindir:
                ax.set_xlabel(par, fontsize=8)
            else:
                ax.set_xlabel(par, fontsize=10)
            ax.set_yticks([])
            ax.set_yticklabels([])

            if j == (nrows * ncols) - 1 or idx == len(pars_short) - 1:
                pl.tight_layout()
                pl.savefig(f"{figbase}_{pp}.pdf")

        # Wasn't working before, but how do I implement a legend?
        # ax[nr][nc].legend(loc = 'best')
        pl.show()
    
    noise_dict = noise_core.get_map_dict()

    # Print bayes factor for red noise in pulsar
    rn_amp_nm = psr_name+"_red_noise_log10_A"
    rn_bf = model_utils.bayes_fac(noise_core(rn_amp_nm), ntol=1, logAmax=-11, logAmin=-20)[0]

    return noise_core, noise_dict, rn_bf


def model_noise(
    mo,
    to,
    which_sampler="PTMCMCSampler",
    vary_red_noise=True,
    n_iter=int(1e5),
    using_wideband=False,
    resume=False,
    run_noise_analysis=True,
    wb_efac_sigma=0.25,
    base_op_dir="./",
    noise_kwargs={},
    sampler_kwargs={},
    return_sampler=False,
):
    """
    Setup enterprise PTA and perform MCMC noise analysis

    Parameters
    ==========
    mo: PINT (or tempo2) timing model
    to: PINT (or tempo2) TOAs
    sampler: choose from ['PTMCMCSampler' or 'GibbsSampler' or 'discovery']
        PTMCMCSampler -- MCMC sampling with the Enterprise likelihood
        GibbsSampler -- enterprise_extension's GibbsSampler with PTMCMC and Enterprise white noise
        discovery -- various numpyro samplers with a discovery likelihood
    red_noise: include red noise in the model
    n_iter: number of MCMC iterations; Default: 1e5; Recommended > 5e4
    using_wideband: Flag to toggle between narrowband and wideband datasets; Default: False
    run_noise_analysis: Flag to toggle execution of noise modeling; Default: True
    noise_kwargs: dictionary of noise model parameters; Default: {}
    sampler_kwargs: dictionary of sampler parameters; Default: {}
    return_sampler: Flag to return the sampler object; Default: False

    Returns
    =======
    None or
    samp: sampler object
    """

    if not using_wideband:
        outdir = base_op_dir + mo.PSR.value + "_nb/"
    else:
        outdir = base_op_dir + mo.PSR.value + "_wb/"

    if os.path.exists(outdir) and (run_noise_analysis) and (not resume):
        log.info(
            "INFO: A noise directory for pulsar {} already exists! Re-running noise modeling from scratch".format(
                mo.PSR.value
            )
        )
    elif os.path.exists(outdir) and (run_noise_analysis) and (resume):
        log.info(
            "INFO: A noise directory for pulsar {} already exists! Re-running noise modeling starting from previous chain".format(
                mo.PSR.value
            )
        )

    if not run_noise_analysis:
        log.info(
            "Skipping noise modeling. Change run_noise_analysis = True to run noise modeling."
        )
        return None

    # Ensure n_iter is an integer
    n_iter = int(n_iter)

    if n_iter < 1e4:
        log.warning(
            "Such a small number of iterations is unlikely to yield accurate posteriors. STRONGLY recommend increasing the number of iterations to at least 5e4"
        )

    # Create enterprise Pulsar object for supplied pulsar timing model (mo) and toas (to)
    e_psr = Pulsar(mo, to)

    if which_sampler == "PTMCMCSampler":
        log.info(f"INFO: Running noise analysis with {which_sampler} for {e_psr.name}")
        # Setup a single pulsar PTA using enterprise_extensions
        if not using_wideband:
            pta = models.model_singlepsr_noise(
                e_psr,
                white_vary=True,
                red_var=vary_red_noise,
                is_wideband=False,
                use_dmdata=False,
                dmjump_var=False,
                wb_efac_sigma=wb_efac_sigma,
                **noise_kwargs,
            )
        else:
            pta = models.model_singlepsr_noise(
                e_psr,
                is_wideband=True,
                use_dmdata=True,
                white_vary=True,
                red_var=vary_red_noise,
                dmjump_var=False,
                wb_efac_sigma=wb_efac_sigma,
                ng_twg_setup=True,
                **noise_kwargs,
            )
            dmjump_params = {}
            for param in mo.params:
                if param.startswith("DMJUMP"):
                    dmjump_param = getattr(mo, param)
                    dmjump_param_name = (
                        f"{pta.pulsars[0]}_{dmjump_param.key_value[0]}_dmjump"
                    )
                    dmjump_params[dmjump_param_name] = dmjump_param.value
            pta.set_default_params(dmjump_params)
        # set groups here
        groups = setup_sampling_groups(pta, write_groups=True, outdir=outdir)
        #######
        # setup sampler using enterprise_extensions
        samp = sampler.setup_sampler(pta,
                                     outdir=outdir,
                                     resume=resume,
                                     groups=groups)
        # Initial sample
        x0 = np.hstack([p.sample() for p in pta.params])
        # Start sampling
        samp.sample(
            x0, n_iter, SCAMweight=30, AMweight=15, DEweight=50, **sampler_kwargs
        )
    elif which_sampler == "GibbsSampler":
        try: 
            from enterprise_extensions import GibbsSampler
        except:
            log.error("Please install the latest enterprise_extensions")
            ValueError("Please install the latest enterprise_extensions")
        log.info(f"INFO: Running noise analysis with {which_sampler} for {e_psr.name}")
        samp = GibbsSampler(
            e_psr,
            **noise_kwargs,
        )
        samp.sample(niter=n_iter, save_path=outdir, **sampler_kwargs)
        pass
    elif which_sampler == "discovery":
        log.info(f"INFO: Running noise analysis with {which_sampler} for {e_psr.name}")
        try:
            import jax
            import xarray as xr
        except ImportError:
            log.error("Please install latest version of jax and/or xarray")
            ValueError("Please install lastest version of jax and/or xarray")
        # get the default settings
        model_defaults, sampler_defaults = get_model_and_sampler_default_settings()
        # update with args passed in
        model_kwargs = model_defaults.update(noise_kwargs)
        sampler_kwargs = sampler_defaults.update(sampler_kwargs)
        os.mkdir(outdir, parents=True, exist_ok=True)
        with open(outdir+"model_kwargs.json", "w") as f:
            json.dump(model_kwargs, f)
        with open(outdir+"sampler_kwargs.json", "w") as f:
            json.dump(sampler_kwargs, f)
        samp, log_x, numpyro_model = setup_discovery_noise(e_psr, model_kwargs, sampler_kwargs)
        # run the sampler
        samp.run(jax.random.key(42))
        # convert to a DataFrame
        df = log_x.to_df(samp.get_samples()['par'])
        # convert DataFrame to dictionary
        samples_dict = df.to_dict(orient='list')
        if sampler_kwargs['numpyro_sampler'] is not 'HMC_GIBBS':
            ln_like = log_likelihood(numpyro_model, samp.get_samples())['ll']
            ln_prior = dist.Normal(0, 10).log_prob(samp.get_samples()['par']).sum(axis=-1)
            ln_post = ln_like + ln_prior
            samples_dict['lnlike'] = ln_like
            samples_dict['lnpost'] = ln_post
        else:
            samples_dict['lnlike'] = None
            samples_dict['lnpost'] = None
        # convert dictionary to ArviZ InferenceData object
        inference_data = az.from_dict(samples_dict)
        # Save to NetCDF file which can be loaded into la_forge
        inference_data.to_netcdf(outdir+"chain.nc")
    else:
        log.error(
            "Invalid sampler specified. Please use 'PTMCMCSampler' or 'GibbsSampler' or 'discovery' "
        )
    if return_sampler:
        return samp


def convert_to_RNAMP(value):
    """
    Utility function to convert enterprise RN amplitude to tempo2/PINT parfile RN amplitude
    """
    return (86400.0 * 365.24 * 1e6) / (2.0 * np.pi * np.sqrt(3.0)) * 10**value


def add_noise_to_model(
    model,
    burn_frac=0.25,
    save_corner=True,
    no_corner_plot=False,
    ignore_red_noise=False,
    using_wideband=False,
    rn_bf_thres=1e2,
    base_dir=None,
    compare_dir=None,
):
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
        base_dir = "./"

    chaindir_compare = compare_dir
    if not using_wideband:
        chaindir = os.path.join(base_dir, f"{model.PSR.value}_nb/")
        if compare_dir is not None:
            chaindir_compare = os.path.join(compare_dir, f"{model.PSR.value}_nb/")
    else:
        chaindir = os.path.join(base_dir, f"{model.PSR.value}_wb/")
        if compare_dir is not None:
            chaindir_compare = os.path.join(compare_dir, f"{model.PSR.value}_wb/")

    log.info(f"Using existing noise analysis results in {chaindir}")
    log.info("Adding new noise parameters to model.")
    noise_dict, rn_bf = analyze_noise(
        chaindir,
        burn_frac,
        save_corner,
        no_corner_plot,
        chaindir_compare=chaindir_compare,
    )
    chainfile = chaindir + "chain_1.txt"
    mtime = Time(os.path.getmtime(chainfile), format="unix")
    log.info(f"Noise chains loaded from {chainfile} created at {mtime.isot}")

    # Create the maskParameter for EFACS
    efac_params = []
    equad_params = []
    ecorr_params = []
    dmefac_params = []
    dmequad_params = []

    efac_idx = 1
    equad_idx = 1
    ecorr_idx = 1
    dmefac_idx = 1
    dmequad_idx = 1

    for key, val in noise_dict.items():

        psr_name = key.split("_")[0]

        if "_efac" in key:

            param_name = key.split("_efac")[0].split(psr_name)[1][1:]

            tp = maskParameter(
                name="EFAC",
                index=efac_idx,
                key="-f",
                key_value=param_name,
                value=val,
                units="",
                convert_tcb2tdb=False,
            )
            efac_params.append(tp)
            efac_idx += 1

        # See https://github.com/nanograv/enterprise/releases/tag/v3.3.0
        # ..._t2equad uses PINT/Tempo2/Tempo convention, resulting in total variance EFAC^2 x (toaerr^2 + EQUAD^2)
        elif "_t2equad" in key:

            param_name = (
                key.split("_t2equad")[0].split(psr_name)[1].split("_log10")[0][1:]
            )

            tp = maskParameter(
                name="EQUAD",
                index=equad_idx,
                key="-f",
                key_value=param_name,
                value=10**val / 1e-6,
                units="us",
                convert_tcb2tdb=False,
            )
            equad_params.append(tp)
            equad_idx += 1

        # ..._tnequad uses temponest convention, resulting in total variance EFAC^2 toaerr^2 + EQUAD^2
        elif "_tnequad" in key:

            param_name = (
                key.split("_tnequad")[0].split(psr_name)[1].split("_log10")[0][1:]
            )

            tp = maskParameter(
                name="EQUAD",
                index=equad_idx,
                key="-f",
                key_value=param_name,
                value=10**val / 1e-6,
                units="us",
                convert_tcb2tdb=False,
            )
            equad_params.append(tp)
            equad_idx += 1

        # ..._equad uses temponest convention; generated with enterprise pre-v3.3.0
        elif "_equad" in key:

            param_name = (
                key.split("_equad")[0].split(psr_name)[1].split("_log10")[0][1:]
            )

            tp = maskParameter(
                name="EQUAD",
                index=equad_idx,
                key="-f",
                key_value=param_name,
                value=10**val / 1e-6,
                units="us",
                convert_tcb2tdb=False,
            )
            equad_params.append(tp)
            equad_idx += 1

        elif ("_ecorr" in key) and (not using_wideband):

            param_name = (
                key.split("_ecorr")[0].split(psr_name)[1].split("_log10")[0][1:]
            )

            tp = maskParameter(
                name="ECORR",
                index=ecorr_idx,
                key="-f",
                key_value=param_name,
                value=10**val / 1e-6,
                units="us",
                convert_tcb2tdb=False,
            )
            ecorr_params.append(tp)
            ecorr_idx += 1

        elif ("_dmefac" in key) and (using_wideband):

            param_name = key.split("_dmefac")[0].split(psr_name)[1][1:]

            tp = maskParameter(
                name="DMEFAC",
                index=dmefac_idx,
                key="-f",
                key_value=param_name,
                value=val,
                units="",
                convert_tcb2tdb=False,
            )
            dmefac_params.append(tp)
            dmefac_idx += 1

        elif ("_dmequad" in key) and (using_wideband):

            param_name = (
                key.split("_dmequad")[0].split(psr_name)[1].split("_log10")[0][1:]
            )

            tp = maskParameter(
                name="DMEQUAD",
                index=dmequad_idx,
                key="-f",
                key_value=param_name,
                value=10**val,
                units="pc/cm3",
                convert_tcb2tdb=False,
            )
            dmequad_params.append(tp)
            dmequad_idx += 1

    # Test EQUAD convention and decide whether to convert
    convert_equad_to_t2 = False
    if test_equad_convention(noise_dict.keys()) == "tnequad":
        log.info(
            "WN paramaters use temponest convention; EQUAD values will be converted once added to model"
        )
        convert_equad_to_t2 = True
        if np.any(["_equad" in p for p in noise_dict.keys()]):
            log.info("WN parameters generated using enterprise pre-v3.3.0")
    elif test_equad_convention(noise_dict.keys()) == "t2equad":
        log.info("WN parameters use T2 convention; no conversion necessary")

    # Create white noise components and add them to the model
    ef_eq_comp = pm.ScaleToaError()
    ef_eq_comp.remove_param(param="EFAC1")
    ef_eq_comp.remove_param(param="EQUAD1")
    ef_eq_comp.remove_param(param="TNEQ1")
    for efac_param in efac_params:
        ef_eq_comp.add_param(param=efac_param, setup=True)
    for equad_param in equad_params:
        ef_eq_comp.add_param(param=equad_param, setup=True)
    model.add_component(ef_eq_comp, validate=True, force=True)

    if len(dmefac_params) > 0 or len(dmequad_params) > 0:
        dm_comp = pm.noise_model.ScaleDmError()
        dm_comp.remove_param(param="DMEFAC1")
        dm_comp.remove_param(param="DMEQUAD1")
        for dmefac_param in dmefac_params:
            dm_comp.add_param(param=dmefac_param, setup=True)
        for dmequad_param in dmequad_params:
            dm_comp.add_param(param=dmequad_param, setup=True)
        model.add_component(dm_comp, validate=True, force=True)
    if len(ecorr_params) > 0:
        ec_comp = pm.EcorrNoise()
        ec_comp.remove_param("ECORR1")
        for ecorr_param in ecorr_params:
            ec_comp.add_param(param=ecorr_param, setup=True)
        model.add_component(ec_comp, validate=True, force=True)

    # Create red noise component and add it to the model
    log.info(f"The SD Bayes factor for red noise in this pulsar is: {rn_bf}")
    if (rn_bf >= rn_bf_thres or np.isnan(rn_bf)) and (not ignore_red_noise):

        log.info("Including red noise for this pulsar")
        # Add the ML RN parameters to their component
        rn_comp = pm.PLRedNoise()

        rn_keys = np.array([key for key, val in noise_dict.items() if "_red_" in key])
        rn_comp.RNAMP.quantity = convert_to_RNAMP(
            noise_dict[psr_name + "_red_noise_log10_A"]
        )
        rn_comp.RNIDX.quantity = -1 * noise_dict[psr_name + "_red_noise_gamma"]

        # Add red noise to the timing model
        model.add_component(rn_comp, validate=True, force=True)
    else:
        log.info("Not including red noise for this pulsar")
        
    # Check to see if dm noise is present
    dm_pars = [key for key in list(noise_dict.keys()) if "_dm_gp" in key]
    if len(dm_pars) > 0:
        ###### POWERLAW DM NOISE ######
        if f'{psr_name}_dm_gp_log10_A' in dm_pars:
            #dm_bf = model_utils.bayes_fac(noise_core(rn_amp_nm), ntol=1, logAmax=-11, logAmin=-20)[0] 
            #log.info(f"The SD Bayes factor for dm noise in this pulsar is: {dm_bf}") 
            log.info('Adding Powerlaw DM GP noise as PLDMNoise to par file')
            # Add the ML RN parameters to their component
            dm_comp = pm.PLDMNoise()
            dm_keys = np.array([key for key, val in noise_dict.items() if "_red_" in key])
            dm_comp.TNDMAMP.quantity = convert_to_RNAMP(
                noise_dict[psr_name + "_dm_gp_log10_A"]
            )
            dm_comp.TNDMIDX.quantity = -1 * noise_dict[psr_name + "_dm_gp_gamma"]
            ##### FIXMEEEEEEE : need to figure out some way to softcode this
            dm_comp.TNDMC.quantitity = 100
            # Add red noise to the timing model
            model.add_component(dm_comp, validate=True, force=True)
        ###### FREE SPECTRAL (WaveX) DM NOISE ######
        elif f'{psr_name}_dm_gp_log10_rho_0' in dm_pars:
            log.info('Adding Free Spectral DM GP as DMWaveXnoise to par file')
            NotImplementedError('DMWaveXNoise not yet implemented')

    # Check to see if higher order chromatic noise is present
    chrom_pars = [key for key in list(noise_dict.keys()) if "_chrom_gp" in key]
    if len(chrom_pars) > 0:
        ###### POWERLAW CHROMATIC NOISE ######
        if f'{psr_name}_chrom_gp_log10_A' in chrom_pars:
            log.info('Adding Powerlaw CHROM GP noise as PLCMNoise to par file')
            # Add the ML RN parameters to their component
            chrom_comp = pm.PLCMNoise()
            chrom_keys = np.array([key for key, val in noise_dict.items() if "_chrom_gp_" in key])
            dm_comp.TNDMAMP.quantity = convert_to_RNAMP(
                noise_dict[psr_name + "_chrom_gp_log10_A"]
            )
            chrom_comp.TNCMIDX.quantity = -1 * noise_dict[psr_name + "_dm_gp_gamma"]
            ##### FIXMEEEEEEE : need to figure out some way to softcode this
            chrom_comp.TNCMC.quantitity = 100
            # Add red noise to the timing model
            model.add_component(dm_comp, validate=True, force=True)
        ###### FREE SPECTRAL (WaveX) DM NOISE ######
        elif f'{psr_name}_chrom_gp_log10_rho_0' in chrom_pars:
            log.info('Adding Free Spectral CHROM GP as CMWaveXnoise to par file')
            NotImplementedError('CMWaveXNoise not yet implemented')
    
    log.info(f"The SD Bayes factor for dm noise in this pulsar is: {rn_bf}")
    if (rn_bf >= rn_bf_thres or np.isnan(rn_bf)) and (not ignore_red_noise):

        log.info("Including red noise for this pulsar")
        # Add the ML RN parameters to their component
        rn_comp = pm.PLRedNoise()

        rn_keys = np.array([key for key, val in noise_dict.items() if "_red_" in key])
        rn_comp.RNAMP.quantity = convert_to_RNAMP(
            noise_dict[psr_name + "_red_noise_log10_A"]
        )
        rn_comp.RNIDX.quantity = -1 * noise_dict[psr_name + "_red_noise_gamma"]

        # Add red noise to the timing model
        model.add_component(rn_comp, validate=True, force=True)
    else:
        log.info("Not including red noise for this pulsar")

    # Setup and validate the timing model to ensure things are correct
    model.setup()
    model.validate()
    model.noise_mtime = mtime.isot

    if convert_equad_to_t2:
        from pint_pal.lite_utils import convert_enterprise_equads

        model = convert_enterprise_equads(model)

    return model


def setup_gibbs_sampler():
    """
    Setup the Gibbs sampler for noise analysis from enterprise extensions
    """
    # check that a sufficiently up-to-date version of enterprise_extensions is installed
    try:
        from enterprise_extensions.gibbs_sampling import gibbs
    except ImportError:
        log.error("Please install the latest version of enterprise_extensions")
        return None
    NotImplementedError("Gibbs sampler not yet implemented")


def setup_discovery_noise(psr,
                          model_kwargs={},
                          sampler_kwargs={}):
    """
    Setup the discovery likelihood with numpyro sampling for noise analysis
    """
    # check that jax, numpyro and discovery are installed
    try:
        import discovery as ds
        import jax
        from jax import numpy as jnp
        import numpyro
        from numpyro import distributions as dist
        from numpyro import infer
        from discovery import prior
        from discovery.prior import (makelogtransform_uniform,
                                 makelogprior_uniform,
                                 sample_uniform)
        from discovery.gibbs import setup_single_psr_hmc_gibbs
        
    except ImportError:
        log.error("Please install the latest version of discovery, numpyro, and/or jax")
        ValueError("Please install the latest version of discovery, numpyro, and/or jax")
    # set up the model
    time_span = ds.getspan([psr])
    model_components = [
        psr.residuals,
        ds.makegp_timing(psr, svd=True),
        ds.makenoise_measurement(psr),
        ds.makegp_ecorr(psr),
        ]
    if model_kwargs['inc_rn']:
        if model_kwargs['rn_psd'] == 'powerlaw':
            model_components.append(ds.makegp_fourier(psr, ds.powerlaw, model_kwargs['rn_nfreqs'], T=time_span, name='red_noise'))
        elif model_kwargs['rn_psd'] == 'free_spectral':
            model_components.append(ds.makegp_fourier(psr, ds.free_spectral, model_kwargs['rn_nfreqs'], T=time_span, name='red_noise'))
    if model_kwargs['inc_dmgp']:
        if model_kwargs['dmgp_psd'] == 'powerlaw':
            model_components.append(ds.makegp_fourier(psr, ds.powerlaw, model_kwargs['dmgp_nfreqs'], T=time_span, name='dm_gp'))
        elif model_kwargs['dmgp_psd'] == 'free_spectral':
            model_components.append(ds.makegp_fourier(psr, ds.free_spectral, model_kwargs['dmgp_nfreqs'], T=time_span, name='dm_gp'))
    if model_kwargs['inc_chrom']:
        if model_kwargs['rn_psd'] == 'powerlaw':
            model_components.append(ds.makegp_fourier(psr, ds.powerlaw, model_kwargs['chromgp_nfreqs'], T=time_span, name='dm_gp'))
        elif model_kwargs['rn_psd'] == 'free_spectral':
            model_components.append(ds.makegp_fourier(psr, ds.free_spectral, model_kwargs['chromgp_nfreqs'], T=time_span, name='dm_gp'))
    psl = ds.PulsarLikelihood(model_components)
    prior = prior.makelogprior_uniform(psl.logL.params, {})
    log_x = makelogtransform_uniform(psl.logL)
    # x0 = sample_uniform(psl.logL.params)
    if sampler_kwargs['numpyro_sampler'] == 'HMC_Gibbs':
        def numpyro_model():
            return None
        gibbs_hmc_kernel = setup_single_psr_hmc_gibbs(
                    psrl=psl, psrs=psr,
                    priordict=ds.priordict_standard,
                    invhdorf=None, nuts_kwargs={})
        sampler = infer.MCMC(gibbs_hmc_kernel,
                    num_warmup=sampler_kwargs['num_warmup'],
                    num_samples=sampler_kwargs['num_warmup'],
                    num_chains=sampler_kwargs['num_chains'], 
                    chain_method=sampler_kwargs['chain_method'],
                    progress_bar=True,
                )
    elif sampler_kwargs['numpyro_sampler'] == 'NUTS':
        def numpyro_model():
            params = jnp.array(numpyro.sample("par", dist.Normal(0,10).expand([len(log_x.params)])))
            numpyro.factor("ll", log_x(params))
        nuts_kernel = infer.NUTS(numpyro_model, num_steps=sampler_kwargs['num_steps'])
        sampler = infer.MCMC(nuts_kernel,
                    num_warmup=sampler_kwargs['num_warmup'],
                    num_samples=sampler_kwargs['num_warmup'],
                    num_chains=sampler_kwargs['num_chains'], 
                    chain_method=sampler_kwargs['chain_method'],
                    progress_bar=True,
                )
    elif sampler_kwargs['numpyro_sampler'] == 'HMC':
        def numpyro_model():
            params = jnp.array(numpyro.sample("par", dist.Normal(0,10).expand([len(log_x.params)])))
            numpyro.factor("ll", log_x(params))
        hmc_kernel = infer.HMC(numpyro_model, num_steps=sampler_kwargs['num_steps'])
        sampler = infer.MCMC(hmc_kernel,
                    num_warmup=sampler_kwargs['num_warmup'],
                    num_samples=sampler_kwargs['num_warmup'],
                    num_chains=sampler_kwargs['num_chains'], 
                    chain_method=sampler_kwargs['chain_method'],
                    progress_bar=True,
                )
    
    return sampler, log_x, numpyro_model


def test_equad_convention(pars_list):
    """
    If (t2/tn)equad present, report convention used.
    See https://github.com/nanograv/enterprise/releases/tag/v3.3.0

    Parameters
    ==========
    pars_list: list of noise parameters from enterprise run (e.g. pars.txt)

    Returns
    =======
    convention_test: t2equad/tnequad/None
    """
    # Test equad convention
    t2_test = np.any(["_t2equad" in p for p in pars_list])
    tn_test = np.any([("_tnequad" in p) or ("_equad" in p) for p in pars_list])
    if t2_test and not tn_test:
        return "t2equad"
    elif tn_test and not t2_test:
        return "tnequad"
    else:
        log.warning(
            "EQUADs not present in parameter list (or something strange is going on)."
        )
        return None

def get_model_and_sampler_default_settings():
    model_defaults = {
        'inc_rn': True,
        'rn_psd': 'powerlaw',
        'rn_nfreqs': 30,
        'inc_dmgp': False,
        'dmgp_psd': 'powerlaw',
        'dmgp_nfreqs': 100,
        'inc_chromgp': False,
        'chromgp_psd': 'powerlaw',
        'chromgp_nfreqs': 100,
        'vary_chrom_idx': False,
        'inc_swgp': False,
        'swgp_psd': 'powerlaw',
        'swgp_nfreqs': 100,
        }
    sampler_defaults = {
        'numpyro_sampler': 'HMC',
        'num_steps': 5,
        'num_warmup': 500,
        'num_samples': 2500,
        'num_chains': 4,
        'chain_method': 'vectorized',
        }
    return model_defaults, sampler_defaults