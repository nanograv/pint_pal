# Generic imports
import os, sys
import json
import matplotlib.pyplot as plt
import numpy as np
import warnings
from functools import partial
import inspect
from loguru import logger as log
from multiprocessing import Pool
from pathlib import Path
from datetime import datetime
import dill as pickle
from typing import Any, Dict, List, Optional, Sequence, Union, Callable

# discovery outlier analysis imports
import pandas as pd
import discovery as ds
import arviz as az
from discovery import matrix, selection_backend_flags
from discovery import prior as ds_prior
from discovery.pulsar import save_chain
import numpy as np
import optax
import jax
from jax import random
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpyro
from numpyro import sample, factor, infer, deterministic
from numpyro import distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.elbo import ELBO
from numpyro.infer.svi import SVIState
#from numpyro.infer import SVI, SVIState
from IPython.display import display, clear_output


## copied from discovery to be deleted later
from pint_pal import solar as ds_solar


warnings.filterwarnings("ignore")
log.disable("pint")
log.remove()
log.add(sys.stderr, colorize=False, enqueue=True)
log.info(f"Using {jax.default_backend()} with {jax.local_device_count()} devices")

def _select_fourier_basis(psr, Nfreqs, tspan, logmode, f_min, nlog, noise_type):
    "Convoluted helper function for setting log/lin Fourier bases of different types"
    if nlog > 0:
        if noise_type == 'red_noise':
            return lambda pulsar, comp, T : ds.log_fourierbasis(
                psr, T=tspan, logmode=logmode,
                f_min=f_min, nlin=Nfreqs, nlog=nlog,
                )
        elif noise_type == 'dm_noise':
            return lambda pulsar, comp, T : ds.log_dm_fourierbasis(
                psr, T=tspan, logmode=logmode,
                f_min=f_min, nlin=Nfreqs, nlog=nlog,
                )
        elif noise_type == 'chromatic':
            return lambda pulsar, comp, T : ds.log_free_chromatic_fourierbasis(
                psr, T=tspan, logmode=logmode,
                f_min=f_min, nlin=Nfreqs, nlog=nlog,
                )
        elif noise_type == 'solar_wind':
            return lambda pulsar, comp, T : ds_solar.log_solardm_fourierbasis(
                psr, T=tspan, logmode=logmode,
                f_min=f_min, nlin=Nfreqs, nlog=nlog,
                )  
    elif nlog == 0:
        if noise_type == 'red_noise':
            return ds.fourierbasis
        elif noise_type == 'dm_noise':
            return ds.dmfourierbasis
        elif noise_type == 'chromatic':
            return ds.dmfourierbasis_alpha
        elif noise_type == 'solar_wind':
            return ds_solar.fourierbasis_solar_dm
    else:
        raise ValueError(f"Invalid nlog value in {noise_type} model. Must be a non-negative integer.")
    



def timing_model_block(
        psr: Any,
        svd: bool = True,
        tm_marg: bool = True,
    ) -> Any:
    """
    Build the timing-model Gaussian-process block for a pulsar.

    Parameters
    ----------
    psr : Any
        Pulsar object.
    svd : bool, optional
        Whether to use SVD for the timing model design matrix. Default is True.
    tm_marg : bool, optional
        If True, marginalize the timing model; otherwise treat it as a variable.
        Default is True.

    Returns
    -------
    Any
        Discovery timing-model GP block from ``ds.makegp_timing``.
    """
    return ds.makegp_timing(psr, svd=svd, variable=not tm_marg)

def white_noise_block(
        psr: Any,
        noise_dict: Dict[str, Any] = {},
        include_ecorr: bool = True,
        gp_ecorr: bool = False,
        tn_equad: bool = True,
        selection: Callable = ds.selection_backend_flags,
    ) -> Any:
    """
    Build the white-noise measurement block.

    Parameters
    ----------
    psr : Any
        Pulsar object.
    noise_dict : dict, optional
        Noise parameter dictionary. Default is empty dict.
    include_ecorr : bool, optional
        Whether to include ECORR terms in the measurement model. Default is True.
    gp_ecorr : bool, optional
        Placeholder to keep a shared interface with other block builders.
        This argument is not used in this function. Default is False.
    tn_equad : bool, optional
        Whether to include EQUAD terms. Default is True.
    selection : Callable, optional
        Backend selection function. Default is discovery.selection_backend_flags.

    Returns
    -------
    Any
        Discovery white-noise block from ``ds.makenoise_measurement``.
    """
    return ds.makenoise_measurement(
        psr,
        tnequad=tn_equad,
        ecorr=include_ecorr,
        selection=selection,
        noisedict=noise_dict,
    )

def gp_ecorr_block(
        psr: Any,
        noise_dict: Dict[str, Any] = {},
        include_ecorr: bool = True, # dummy vars
        gp_ecorr: bool = False,
        tn_equad: bool = True,
        selection: Callable = ds.selection_backend_flags,
        gp_ecorr_name: str = 'ecorrGP'
    ) -> Any:
    """
    Build the Gaussian-process ECORR block.

    Parameters
    ----------
    psr : Any
        Pulsar object.
    noise_dict : dict, optional
        Noise parameter dictionary. Default is empty dict.
    include_ecorr : bool, optional
        Unused placeholder to match ``white_noise_block`` interface.
    gp_ecorr : bool, optional
        Unused placeholder to match ``white_noise_block`` interface.
    tn_equad : bool, optional
        Unused placeholder to match ``white_noise_block`` interface.
    selection : Callable, optional
        Backend selection function. Default is discovery.selection_backend_flags.
    gp_ecorr_name : str, optional
        Name for the GP ECORR component. Default is "ecorrGP".

    Returns
    -------
    Any
        Discovery GP ECORR block from ``ds.makegp_ecorr``.
    """
    return ds.makegp_ecorr(
        psr,
        noisedict=noise_dict,
        selection=selection,
        gp_ecorr_name=gp_ecorr_name,
        )

def red_noise_block(
        psr: Any,
        tspan: Optional[float] = None,
        basis: str = 'fourier',
        prior: str = 'powerlaw',
        Nfreqs: int = 100,
        logmode=2,
        f_min_frac=1/5,
        nlog=0,
        name: str = 'red_noise',
        ) -> Any:
    """
    Build the red-noise Gaussian-process block.

    Parameters
    ----------
    psr : Any
        Pulsar object.
    tspan : float, optional
        Total data span passed to Fourier basis construction. Default is None.
    basis : str, optional
        Basis type for the GP. Currently only ``"fourier"`` is implemented.
        Default is ``"fourier"``.
    prior : str, optional
        Prior type or callable prior for the GP amplitude. Supported string
        values are ``"powerlaw"``, ``"broken_powerlaw"``, and
        ``"freespectrum"``. Default is ``"powerlaw"``.
    Nfreqs : int, optional
        Number of Fourier frequencies. Default is 100.
    logmode : int, optional
        Log-binning mode for hybrid log/linear Fourier bases. Default is -1.
    f_min : float, optional
        Minimum Fourier frequency for hybrid log/linear bases. Default is None.
    nlog : int, optional
        Number of logarithmically spaced frequencies. If ``nlog > 0``,
        ``_select_fourier_basis`` returns a log/linear helper basis.
        Default is 0.
    name : str, optional
        Name of the noise component. Default is "red_noise".

    Returns
    -------
    Any
        Discovery red-noise block from ``ds.makegp_fourier``.
    """
    if tspan is None:
        tspan = ds.getspan(psr)
    if basis == 'fourier':
        if prior == 'powerlaw':
            prior = ds.powerlaw
        elif prior == 'broken_powerlaw':
            prior = ds.broken_powerlaw
        elif prior == 'freespectrum':
            prior = ds.freespectrum
        elif callable(prior):
            pass # pass a callable prior
        else:
            raise ValueError("Invalid *prior* specified for Fourier basis red noise. Try one of: ['powerlaw', 'broken_powerlaw', 'freespectrum']")

        rn = ds.makegp_fourier(
            psr,
            prior,
            Nfreqs,
            T=tspan,
            fourierbasis=_select_fourier_basis(
                psr, Nfreqs, tspan, logmode,
                f_min_frac*1/tspan, # scale f_min_frac to f_min using tspan
                nlog, noise_type='red_noise'
            ),
            name=name
            )
    elif basis == 'interpolation':
        raise NotImplementedError("Interpolation basis for solar wind noise is not yet implemented.")
    else:
        raise ValueError("Invalid basis specified for solar wind noise. Must be 'fourier' or 'interpolation'.")

    return rn

def dm_noise_block(
        psr: Any,
        tspan: Optional[float] = None,
        basis: str = 'fourier',
        prior: str = 'powerlaw',
        Nfreqs: int = 100,
        logmode=2,
        f_min_frac=1/5,
        nlog=0,
        name: str = 'dm_gp',
        ) -> Any:
    """
    Build the dispersion-measure (DM) noise Gaussian-process block.

    Parameters
    ----------
    psr : Any
        Pulsar object.
    tspan : float, optional
        Total data span passed to Fourier basis construction. Default is None.
    basis : str, optional
        Basis type for the GP. Currently only ``"fourier"`` is implemented.
        Default is ``"fourier"``.
    prior : str, optional
        Prior type or callable prior for the GP amplitude. Supported string
        values are ``"powerlaw"``, ``"broken_powerlaw"``, and
        ``"freespectrum"``. Default is ``"powerlaw"``.
    Nfreqs : int, optional
        Number of Fourier frequencies. Default is 100.
    logmode : int, optional
        Log-binning mode for hybrid log/linear Fourier bases. Default is 2.
    f_min_frac : float, optional
        Fractional minimum Fourier frequency for hybrid log/linear bases
        (scaled to an absolute frequency using ``tspan``). Default is None.
    nlog : int, optional
        Number of logarithmically spaced frequencies. If ``nlog > 0``,
        ``_select_fourier_basis`` returns a log/linear helper basis.
        Default is 0.
    name : str, optional
        Name of the noise component. Default is "dm_gp".

    Returns
    -------
    Any
        Discovery DM-noise block from ``ds.makegp_fourier``.
    """
    if tspan is None:
        tspan = ds.getspan(psr)
    if basis == 'fourier':
        if prior == 'powerlaw':
            prior = ds.powerlaw
        elif prior == 'broken_powerlaw':
            prior = ds.broken_powerlaw
        elif prior == 'freespectrum':
            prior = ds.freespectrum
        elif callable(prior):
            pass # pass a callable prior
        else:
            raise ValueError("Invalid *prior* specified for Fourier basis DM noise. Try one of: ['powerlaw', 'broken_powerlaw', 'freespectrum']")

        dm_gp = ds.makegp_fourier(
            psr,
            prior,
            Nfreqs,
            T=tspan,
            fourierbasis=_select_fourier_basis(
                psr, Nfreqs, tspan, logmode,
                f_min_frac*1/tspan, # scale f_min_frac to f_min using tspan
                nlog, noise_type='dm_noise'
            ),
            name=name
            )
    elif basis == 'interpolation':
        raise NotImplementedError("Time domain models for dm noise are not yet implemented.")
    else:
        raise ValueError("Invalid basis specified for dm noise. Must be 'fourier' or 'interpolation'.")

    return dm_gp

def chromatic_noise_block(
        psr: Any,
        tspan: Optional[float] = None,
        basis: str = 'fourier',
        prior: str = 'powerlaw',
        Nfreqs: int = 100,
        logmode=2,
        f_min_frac=1/5,
        nlog=0,
        name: str = 'chrom_gp',
        chromatic_idx: str = 'vary',
        ) -> Any:
    """
    Build the chromatic noise Gaussian-process block.

    Parameters
    ----------
    psr : Any
        Pulsar object.
    basis : str, optional
        Basis type for the GP. Currently only ``"fourier"`` is implemented.
        Default is ``"fourier"``.
    prior : str, optional
        Prior type or callable prior for the GP amplitude. Supported string
        values are ``"powerlaw"``, ``"broken_powerlaw"``, and
        ``"freespectrum"``. Default is ``"powerlaw"``.
    Nfreqs : int, optional
        Number of Fourier frequencies. Default is 100.
    tspan : float, optional
        Total data span passed to Fourier basis construction. Default is None.
    logmode : int, optional
        Log-binning mode for hybrid log/linear Fourier bases. Default is -1.
    f_min : float, optional
        Minimum Fourier frequency for hybrid log/linear bases. Default is None.
    nlog : int, optional
        Number of logarithmically spaced frequencies. If ``nlog > 0``,
        ``_select_fourier_basis`` returns a log/linear helper basis.
        Default is 0.
    name : str, optional
        Name of the noise component. Default is ``"chrom_gp"``.
    chromatic_idx : str, optional
        Reserved argument for chromatic index handling mode. Currently not used
        inside this function. Default is ``"vary"``.

    Returns
    -------
    Any
        Discovery chromatic-noise block from ``ds.makegp_fourier``.
    """
    if tspan is None:
        tspan = ds.getspan(psr)
    if basis == 'fourier':
        if prior == 'powerlaw':
            prior = ds.powerlaw
        elif prior == 'broken_powerlaw':
            prior = ds.broken_powerlaw
        elif prior == 'freespectrum':
            prior = ds.freespectrum
        elif callable(prior):
            pass # pass a callable prior
        else:
            raise ValueError("Invalid *prior* specified for Fourier basis chromatic noise. Try one of: ['powerlaw', 'broken_powerlaw', 'freespectrum']")

        chrom_gp = ds.makegp_fourier(
            psr,
            prior,
            Nfreqs,
            T=tspan,
            fourierbasis=_select_fourier_basis(
                psr, Nfreqs, tspan, logmode,
                f_min_frac*1/tspan, # scale f_min_frac to f_min using tspan
                nlog, noise_type='chromatic'
            ),
            name=name
            )
    else:
        raise ValueError("Invalid *basis* specified for chromatic noise. Supported basis types: ['fourier']")
    return chrom_gp

def solar_wind_noise_block(
        psr: Any,
        tspan: Optional[float] = None,
        basis: str = 'fourier',
        basis_nodes: Optional[np.ndarray] = None,
        interp_dt: Optional[float] = 30.0,
        interp_kind: str = 'linear',
        prior: str = 'powerlaw',
        Nfreqs: int = 100,
        logmode=2,
        f_min_frac=1/5,
        nlog=0,
        name: str = 'sw_gp',
        ) -> Any:
    """
    Build the solar-wind noise Gaussian-process block.

    Parameters
    ----------
    psr : Any
        Pulsar object.
    basis : str, optional
        Basis type for the GP. Default is "fourier".
        Supported options are ["fourier", "interpolation"].
    basis_nodes : np.ndarray, optional
        Nodes for the basis. Default is None. Only used for interpolation basis.
    interp_dt : float, optional
        Time step for interpolation nodes in days. Default is 30. Only used for interpolation basis.
    interp_kind : str, optional
        Interpolation kind for the basis. Default is "linear". Only used for interpolation basis.
    prior : str, optional
        Prior type for the GP amplitude. For Fourier basis this is a PSD. For time domain it is a covariance function.
        Fourier basis supports ["powerlaw"].
        Time-domain interpolation basis supports ["ridge", "square_exponential", "quasi_periodic", "matern"].
        Default is "powerlaw".
    Nfreqs : int, optional
        Number of Fourier frequencies. Default is 100. Only used for Fourier basis.
    tspan : float, optional
        Time span for the Fourier basis. Default is None.
    name : str, optional
        Name of the noise component. Default is "red_noise".

    Returns
    -------
    Any
        Discovery solar-wind noise block from either ``ds.makegp_fourier``
        (Fourier basis) or ``ds_solar.makegp_timedomain_solar_dm``
        (interpolation basis).
    """
    if tspan is None:
        tspan = ds.getspan(psr)
    if basis == 'fourier':
        if prior == 'powerlaw':
            prior = ds.powerlaw
        elif prior == 'broken_powerlaw':
            prior = ds.broken_powerlaw
        elif prior == 'freespectrum':
            prior = ds.freespectrum
        elif callable(prior):
            pass # pass a callable prior
        else:
            raise ValueError("Invalid *prior* specified for Fourier basis solar wind noise. Try one of: ['powerlaw', 'broken_powerlaw', 'freespectrum']")

        sw_gp = ds.makegp_fourier(
            psr,
            prior,
            Nfreqs,
            T=tspan,
            fourierbasis=_select_fourier_basis(
                psr, Nfreqs, tspan, logmode,
                f_min_frac*1/tspan, # scale f_min_frac to f_min using tspan
                nlog, noise_type='solar_wind'
            ),
            name=name
            )
    elif basis == 'interpolation':
        if basis_nodes is None:
            basis_nodes = np.arange(psr.toas.min()/86400, psr.toas.max()/86400, interp_dt)
        # else basis nodes are provided by user.
        td_basis, nodes = ds_solar.custom_blocked_interpolation_basis(
            psr.toas,
            nodes=basis_nodes,
            kind=interp_kind,
        )
        if prior == 'ridge':
            prior_kernel = ds_solar.ridge_kernel()
        elif prior == 'square_exponential':
            prior_kernel = ds_solar.square_exponential_kernel()
        elif prior == 'quasi_periodic':
            prior_kernel = ds_solar.quasi_periodic_kernel()
        elif prior == 'matern':
            prior_kernel = ds_solar.matern_kernel()
        elif prior == 'powerlaw':
            raise ValueError("Power-law prior is not supported for time domain solar wind noise. Must be in ['ridge', 'square_exponential', 'quasi_periodic', 'matern'].")
        else: 
            raise ValueError("Invalid prior specified for time domain solar wind noise. Must be in ['ridge', 'square_exponential', 'quasi_periodic', 'matern'].")
        sw_gp = ds_solar.makegp_timedomain_solar_dm(
            psr,
            covariance=prior_kernel,
            dt=None,
            Umat=td_basis,
            nodes=nodes,
            common=[],
            name=name,
        )
    else:
        raise ValueError("Invalid basis specified for solar wind noise. Must be 'fourier' or 'interpolation'.")

    return sw_gp

def make_single_pulsar_noise_likelihood_discovery(
        psr: Any,
        noise_dict: Optional[Dict[str, Any]] = None,
        tspan: Optional[float] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        return_args: bool = False,
) -> Union[ds.PulsarLikelihood, Sequence[Any]]:
    """
    Build a discover likelihood for a single pulsar noise analysis.

    Parameters
    ----------
    psr : Any
        Pulsar object.
    noise_dictionary : dict, optional
        Dictionary of noise parameters.
    tspan : float, optional
        Time span for the noise model.
    model_kwargs : dict, optional
        Dictionary of model keyword arguments.
    return_args : bool, optional
        If True, return the raw argument list instead of the PulsarLikelihood instance.

    Returns
    -------
    discovery.PulsarLikelihood or list
        PulsarLikelihood instance assembled from components, or the raw args list if return_args is True.
    """
    if noise_dict is None:
        noise_dict = {}
    if tspan is None:
        tspan = ds.getspan([psr])
    # if marg_ne:
    #     psr.Mmat = np.hstack([psr.Mmat, np.array([ds.make_solardm(psr)(1.0)]).T])
    model_kwargs.update({ky: False for ky in ['timing_model', 'white_noise', 'red_noise', 'dm_noise', 'chromatic_noise', 'solar_wind'] if ky not in model_kwargs.keys()})
    # model checks -- update your configs !!
    if not model_kwargs['timing_model']:
        log.error("Timing model must be included in the `noise_run` config block for the likelihood to be properly constructed.")
    if not model_kwargs['white_noise']:
        log.error("White noise must be included in the `noise_run` config block for the likelihood to be properly constructed.")
    if not model_kwargs['red_noise']:
        log.warn("Red noise is not included in the `noise_run` config block. Proceed with caution.")       
    ## add pulsar's time spans to model for signals where not included
    ## this allows individual signals to be passed specific timespans
    for name, mod in model_kwargs.items():
        if mod and name in ['red_noise', 'dm_noise', 'chromatic_noise', 'solar_wind']:
            if 'tspan' not in list(mod.keys()):
                model_kwargs[name].update({'tspan': tspan})
    ## timing model
    args = [psr.residuals]
    if model_kwargs['timing_model']:
        args.append(
            timing_model_block(psr, **model_kwargs['timing_model'])
        )
    else:
        log.error("Timing model must be included in the model_kwargs for the likelihood to be properly constructed.")
    # white noise block + gp ecorr block
    if model_kwargs['white_noise']:
        log.info("Adding white noise to the model.")
        gp_ecorr = model_kwargs['white_noise'].get('gp_ecorr', False)
        if gp_ecorr:
            args.append(
                gp_ecorr_block(
                    psr,
                    noise_dict=noise_dict,
                    **model_kwargs['white_noise'],
                )
            )
            # make sure kernel ecorr is off if basis ecorr is on !
            model_kwargs['white_noise']['include_ecorr'] = False
        args.append(
            white_noise_block(
                psr,
                noise_dict=noise_dict,
                **model_kwargs['white_noise'],
            )
        )
    else:
        log.error("White noise must be included in the model_kwargs for the likelihood to be properly constructed.")
    # red noise block
    if model_kwargs['red_noise']:
        log.info("Adding red noise to the model.")
        args.append(
            red_noise_block(
                psr,
                **model_kwargs['red_noise'],
            )
        )
    if model_kwargs['dm_noise']:
        log.info("Adding DM noise to the model.")
        args.append(
            dm_noise_block(
                psr,
                **model_kwargs['dm_noise']
            )
        )

    if model_kwargs['chromatic_noise']:
        log.info("Adding chromatic noise to the model.")
        args.append(
            chromatic_noise_block(
                psr,
                name='chrom_gp',
                **model_kwargs['chromatic_noise']
            )
        )
    if model_kwargs['solar_wind']:
        log.info("Adding solar wind noise to the model.")
        args.append(
            solar_wind_noise_block(
                psr,
                name='sw_gp',
                **model_kwargs['solar_wind']
            )
        )
    if return_args:
        return args
    else:
        return ds.PulsarLikelihood(tuple(args))

def make_sampler_nuts(
        numpyro_model: Callable,
        sampler_kwargs: Dict[str, Any] = {},
        ) -> infer.MCMC:
    """
    Create a NumPyro NUTS sampler with filtered keyword arguments.

    Parameters
    ----------
    numpyro_model : Callable
        NumPyro model function.
    sampler_kwargs : dict, optional
        Keyword arguments for NUTS and MCMC constructors. Only supported
        parameters are forwarded. Default is empty dict.

    Returns
    -------
    numpyro.infer.MCMC
        Configured MCMC sampler instance.
    """
    nutsargs = dict(
        max_tree_depth=sampler_kwargs.get('max_tree_depth', 8), # these should be good defaults
        dense_mass=sampler_kwargs.get('dense_mass', False),
        forward_mode_differentiation=sampler_kwargs.get('forward_mode_differentiation', False),
        target_accept_prob=sampler_kwargs.get('target_accept_prob', 0.8),
        **{arg: val for arg, val in sampler_kwargs.items() if arg in inspect.getfullargspec(infer.NUTS).args and arg not in {'max_tree_depth', 'dense_mass', 'forward_mode_differentiation', 'target_accept_prob'}}
        )
    #samples_per_checkpoint = int(sampler_kwargs.get('num_samples', 1000) / sampler_kwargs.get('num_checkpoints', 5))
    mcmcargs = dict(
        num_samples=sampler_kwargs.get('num_samples', 2000),
        num_warmup=sampler_kwargs.get('num_warmup', 500),
        **{arg: val for arg, val in sampler_kwargs.items() if arg in inspect.getfullargspec(infer.MCMC).kwonlyargs
              and arg not in {'num_samples', 'num_warmup'}
           }
        )
    sampler = infer.MCMC(infer.NUTS(numpyro_model, **nutsargs), **mcmcargs)
    sampler.to_df = lambda: numpyro_model.to_df(sampler.get_samples())

    return sampler

def make_numpyro_model(input_lnlike: Any, priordict: Dict[str, Any] = {}) -> Callable[[], None]:
    """
    Wrap a discovery likelihood into a NumPyro model.

    Parameters
    ----------
    input_lnlike : Any
        Likelihood object that provides a callable interface and a .params list.
    priordict : dict, optional
        Dictionary of prior overrides. Default is empty dict.

    Returns
    -------
    Callable[[], None]
        NumPyro model function with attached .to_df convenience method.
    """
    priors_dict = ds.priordict_standard.copy()
    priors_dict.update(priordict)
    def numpyro_model() -> None:
        """NumPyro model that samples parameters and factors in the log-likelihood."""
        lnlike = input_lnlike({par: numpyro.sample(par, dist.Uniform(*ds_prior.getprior_uniform(par, priordict)))
            for par in input_lnlike.params})

        numpyro.factor('logl', lnlike)
    numpyro_model.to_df = lambda chain: pd.DataFrame(chain)

    return numpyro_model

def run_nuts_with_checkpoints(
    sampler,
    num_samples_per_checkpoint,
    rng_key,
    outdir="chains",
    file_name="numpyro_samples",
    resume=False,
    diagnostics=True,
):
    """Run NumPyro MCMC and save checkpoints.
    This function performs multiple iterations of MCMC sampling, saving checkpoints
    after each iteration. It saves samples to feather files and the NumPyro MCMC
    state to JSON.
    Parameters
    ----------
    sampler : numpyro.infer.MCMC
        A NumPyro MCMC sampler object.
    num_samples_per_checkpoint : int
        The number of samples to save in each checkpoint.
    rng_key : jax.random.PRNGKey
        The random number generator key for JAX.
    outdir : str | Path
        The directory for output files.
    resume : bool
        Whether to look for a state to resume from.
    Returns
    -------
    None
        This function doesn't return any value but saves the results to disk.
    Side Effects
    ------------
    - Runs the MCMC sampler for the number of iterations required to reach the total sample number.
    - Saves samples data to feather files after each iteration.
    - Writes the NumPyro sampler state to a pickle file after each iteration.
    Example
    -------
    >>> import discovery.samplers.numpyro as ds_numpyro
    >>> # Assume `model` is configured
    >>> npsampler = ds_numpyro.makesampler_nuts(model, num_samples =100, num_warmup=50)
    >>> ds_numpyro.run_nuts_with_checkpoints(npsampler, 10, jax.random.key(42))
    """
    # convert to pathlib object
    # make directory if it doesn't exist
    if not isinstance(outdir, Path):
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)

    samples_file = outdir / f"{file_name}.feather"
    checkpoint_file = outdir / f"{file_name}-checkpoint.pickle"

    if checkpoint_file.is_file() and samples_file.is_file() and resume:
        df = pd.read_feather(samples_file)
        num_samples_saved = df.shape[0]

        with checkpoint_file.open("rb") as f:
            checkpoint = pickle.load(f)

        total_sample_num = sampler.num_samples - num_samples_saved

        sampler.post_warmup_state = checkpoint

    else:
        df = None
        num_samples_saved = 0
        total_sample_num = sampler.num_samples

    num_checkpoints = int(jnp.ceil(total_sample_num / num_samples_per_checkpoint))
    remainder_samples = int(total_sample_num % num_samples_per_checkpoint)

    for checkpoint in range(num_checkpoints):
        if checkpoint == 0:
            sampler.num_samples = num_samples_per_checkpoint
            sampler._set_collection_params()  # Need this to update num_samples
        elif checkpoint == num_checkpoints - 1:
            # We won't need to update the collection params because we've set the post warmup state,
            # and that accomplishes the same goal.
            sampler.num_samples = remainder_samples if remainder_samples != 0 else num_samples_per_checkpoint

        sampler.run(rng_key)

        df_new = sampler.to_df()

        df = pd.concat([df, df_new]) if df is not None else df_new

        save_chain(df, samples_file)

        with checkpoint_file.open("wb") as f:
            pickle.dump(sampler.last_state, f)

        sampler.post_warmup_state = sampler.last_state

        rng_key, _ = jax.random.split(rng_key)

        # checkpoint plot
        if diagnostics:
            try:
                idata = az.from_numpyro(sampler)
                n_vars = len(idata.posterior.data_vars)
                with az.rc_context({"plot.max_subplots": max(40, 2 * n_vars)}):
                    fig_trace = az.plot_trace(idata)
                fig_trace = fig_trace.ravel()[0].figure if hasattr(fig_trace, "ravel") else plt.gcf()
                fig_trace.suptitle(f"Checkpoint {checkpoint + 1}/{num_checkpoints} trace")
                fig_trace.tight_layout()
                trace_file = outdir / f"{file_name}-checkpoint-trace.png"
                fig_trace.savefig(trace_file, dpi=150, bbox_inches="tight")


                chain_method = getattr(sampler, "chain_method", "unknown")
                posterior_sizes = dict(getattr(idata.posterior, "sizes", {}))
                n_chains = int(posterior_sizes.get("chain", 0))
                n_draws = int(posterior_sizes.get("draw", 0))
                clear_output(wait=True)
                display(fig_trace)
                if chain_method in {"parallel", "vectorized", "sequential"} and n_chains >= 2 and n_draws >= 4:
                    rhat = az.rhat(idata)
                    rhat_vals = np.asarray(rhat.to_array(), dtype=float).ravel()
                    rhat_vals = rhat_vals[np.isfinite(rhat_vals)]
                    if len(rhat_vals) > 0:
                        n_high = int(np.sum(rhat_vals > 1.01))
                        print(
                            "R-hat summary: "
                            f"median={np.median(rhat_vals):.4f}, "
                            f"max={np.max(rhat_vals):.4f}, "
                            f"n(>1.01)={n_high}/{len(rhat_vals)}"
                        )
                    else:
                        print("R-hat summary: no finite values available")
                else:
                    print(
                        "R-hat skipped: "
                        f"chain_method={chain_method}, chains={n_chains}, draws={n_draws}. "
                        "Need at least 2 chains and 4 draws per chain."
                    )
                plt.close(fig_trace)
            except Exception as e:
                log.warning(f"Diagnostics plotting failed at checkpoint {checkpoint + 1}: {e}")


def setup_svi(
    model: Callable,
    guide: Callable,
    loss: ELBO | None = None,
    num_warmup_steps: int = 500,
    max_epochs: int = 5000,
    peak_learning_rate: float = 0.01,
    gradient_clipping_val: float | None = None,
) -> SVI:
    """
    Set up Stochastic Variational Inference with AdamW optimizer and cosine decay schedule.

    Parameters
    ----------
    model : Callable
        NumPyro model function.
    guide : Callable
        NumPyro guide function (typically a delta function for MAP estimation).
    loss : ELBO or None, optional
        Evidence Lower Bound loss function. If None, uses Trace_ELBO().
        Default is None.
    num_warmup_steps : int, optional
        Number of warmup steps for learning rate schedule. Default is 500.
    max_epochs : int, optional
        Maximum number of training epochs for learning rate decay. Default is 5000.
    peak_learning_rate : float, optional
        Peak learning rate value. Default is 0.01.
    gradient_clipping_val : float or None, optional
        Maximum global norm for gradient clipping. If None, no clipping is applied.
        Default is None.

    Returns
    -------
    SVI
        Configured NumPyro SVI object with AdamW optimizer and warmup cosine decay
        learning rate schedule.

    Notes
    -----
    The learning rate schedule starts at 0, warms up to peak_learning_rate over num_warmup_steps,
    then decays following a cosine schedule to 1% of peak_learning_rate over max_epochs steps.
    """
    if loss is None:
        loss = Trace_ELBO()
    # Define the learning rate schedule
    learning_rate_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0,
        peak_value=peak_learning_rate,
        warmup_steps=num_warmup_steps,
        decay_steps=max_epochs,
        end_value=peak_learning_rate * 0.01,  # Decay to 10% of the peak
    )
    npyro_optimizer = numpyro.optim.optax_to_numpyro(
        # Gradient clipping if supplied
        (
            optax.adamw(learning_rate=learning_rate_schedule)
            if gradient_clipping_val is None
            else optax.chain(
                optax.clip_by_global_norm(
                    gradient_clipping_val,
                ),
                optax.adamw(learning_rate=learning_rate_schedule),
            )
        ),
    )
    return SVI(model, guide, npyro_optimizer, loss=loss)


@partial(jax.jit, static_argnums=(0, -1))
def run_training_batch(
    svi: SVI,
    svi_state: SVIState,
    rng_key: jax.Array,
    batch_size: int,
) -> SVIState:
    """
    Run SVI parameter updates for a fixed number of steps using jax.lax.scan.

    This function is JIT-compiled for speed. The SVI object and batch_size are
    treated as static arguments (static_argnums=(0, -1)).

    Parameters
    ----------
    svi : SVI
        NumPyro SVI object containing the model, guide, and optimizer.
    svi_state : SVIState
        Current state of the SVI optimizer.
    rng_key : jax.Array
        JAX random number generator key.
    batch_size : int
        Number of SVI update steps to run.

    Returns
    -------
    SVIState
        Final SVI state after batch_size update steps.

    Notes
    -----
    Uses jax.lax.scan for efficient iteration, which is faster than a Python
    for loop inside a JIT-compiled function.
    """

    def body_fn(carry: tuple[SVIState, jax.Array], x: Any) -> tuple[tuple[SVIState, jax.Array], None]:
        """Single SVI update step for jax.lax.scan."""
        svi_state, rng_key = carry
        rng_key, subkey = jax.random.split(rng_key)
        new_svi_state, loss = svi.update(svi_state)
        return (new_svi_state, subkey), None

    # Use lax.scan to loop `body_fn` for `batch_size` iterations
    (final_svi_state, _), _ = jax.lax.scan(
        body_fn, (svi_state, rng_key), xs=None, length=batch_size,
    )

    return final_svi_state

def _stack_hist(arr_list):
    return np.stack([np.asarray(a, dtype=float) for a in arr_list], axis=0)

def _plot_with_iqr(ax, Y, color, title, ylabel, x):
    q25, q50, q75 = np.nanpercentile(Y, [25, 50, 75], axis=0)
    for y in Y:
        ax.plot(x, y, color=color, alpha=0.12, lw=1)
    ax.fill_between(x, q25, q75, color=color, alpha=0.25)
    ax.plot(x, q50, color=color, lw=2)
    ax.set_title(title)
    ax.set_xlabel("Within-batch progress")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)

def _tree_l2_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))

@partial(jax.jit, static_argnums=(0, -1))
def run_training_batch_with_diagnostics(
    svi: SVI,
    svi_state: SVIState,
    rng_key: jax.Array,
    batch_size: int,
):
    def body_fn(carry: tuple[SVIState, jax.Array], _: Any):
        state, key = carry
        key, subkey = jax.random.split(key)

        old_params = svi.get_params(state)

        # gradient norm diagnostic
        grads = jax.grad(svi.loss.loss, argnums=1)(
            state.rng_key, old_params, svi.model, svi.guide
        )
        grad_norm = _tree_l2_norm(grads)

        # keep update behavior same as non-diagnostics fn
        new_state, loss = svi.update(state)

        new_params = svi.get_params(new_state)
        step = jax.tree_util.tree_map(lambda a, b: b - a, old_params, new_params)
        step_norm = _tree_l2_norm(step)
        param_norm = _tree_l2_norm(new_params)

        y = (new_state, loss, grad_norm, step_norm, param_norm)
        return (new_state, subkey), y

    (final_state, _), (svi_states, losses, grad_norms, step_norms, param_norms) = jax.lax.scan(
        body_fn, (svi_state, rng_key), xs=None, length=batch_size
    )
    return final_state, svi_states, losses, grad_norms, step_norms, param_norms


def run_svi_early_stopping(
    rng_key: jax.Array,
    svi: SVI,
    batch_size: int = 1000,
    patience: int = 3,
    max_num_batches: int = 50,
    difference_threshold: float = 1.0,
    diagnostics: bool = False,
    outdir: str | Path | None = None,
    file_prefix: str = "svi",
) -> Dict[str, Any]:
    """
    Run SVI optimization with early stopping based on loss plateau detection.

    Training proceeds in batches of optimization steps. Early stopping is triggered
    when the validation loss fails to improve by more than `difference_threshold` for `patience`
    consecutive batches.

    Parameters
    ----------
    rng_key : jax.Array
        JAX random number generator key.
    svi : SVI
        NumPyro SVI object containing the model, guide, and optimizer.
    batch_size : int, optional
        Number of optimization steps per batch. Default is 1000.
    patience : int, optional
        Number of consecutive batches without improvement (> `difference_threshold` decrease in loss)
        before early stopping is triggered. Default is 3.
    max_num_batches : int, optional
        Maximum number of batches to run. Default is 50.
    difference_threshold : float, optional
        Minimum decrease in loss required to reset the patience counter. Default is 1.0.
    diagnostics : bool, optional
        If True, collect gradient norms and intermediate states at each step.
        This adds computational overhead. Default is False.
    outdir : str | Path | None, optional
        Directory to save diagnostics plots when `diagnostics=True`. If None,
        plots are only displayed and not written to disk. Default is None.
    file_prefix : str, optional
        Prefix used for saved diagnostics plot filenames. Default is "svi".

    Returns
    -------
    dict
        Dictionary of optimized parameter values from the best SVI state
        (lowest validation loss).

    Notes
    -----
    The early stopping criterion requires the loss to improve by at least `difference_threshold`
    to reset the patience counter. This threshold is configurable and may need
    adjustment for different problem scales.

    Examples
    --------
    >>> from jax import random
    >>> rng_key = jax.Array(0)
    >>> svi = setup_svi(model, guide)
    >>> params = run_svi_early_stopping(rng_key, svi, batch_size=1000, patience=3)
    """
    svi_state = svi.init(rng_key)

    svi_states_list = []
    global_norm_grads_list = []
    step_norms_list = []
    param_norms_list = []
    losses_list = []

    best_val_loss = float("inf")
    best_svi_state = svi_state
    patience_counter = 0

    log.info(f"Starting training with batches of {batch_size} steps.")

    final_params = None
    diagnostics_plot_dir = None
    diagnostics_plot_path = None
    if diagnostics and outdir is not None:
        diagnostics_plot_dir = Path(outdir)
        diagnostics_plot_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_plot_path = diagnostics_plot_dir / f"{file_prefix}_svi_diagnostics.png"
    if diagnostics:
        # once, before loop so plot updates in place
        plt.ioff()  # optional: avoids duplicate auto-displays
        # Create persistent figure once
        fig, axs = plt.subplots(2, 4, figsize=(18, 8))
        ax_cum_loss, ax_cum_grad, ax_cum_step, ax_cum_param = axs[0]
        ax_batch_loss, ax_batch_grad, ax_batch_step, ax_batch_param = axs[1]
        # Fixed metric colors
        C_LOSS = "#1f77b4"   # blue
        C_GRAD = "#d62728"   # red
        C_STEP = "#2ca02c"   # green
        C_PARAM = "#9467bd"  # purple
    for batch_num in range(max_num_batches):
        if diagnostics:
            # from run_training_batch_with_diagnostics(...)
            svi_state, batch_svi_states, batch_losses, batch_grad_norms, batch_step_norms, batch_param_norms = (
                run_training_batch_with_diagnostics(svi, svi_state, rng_key, batch_size)
            )

            # store
            svi_states_list.append(batch_svi_states)
            losses_list.append(np.asarray(batch_losses))
            global_norm_grads_list.append(np.asarray(batch_grad_norms))
            step_norms_list.append(np.asarray(batch_step_norms))
            param_norms_list.append(np.asarray(batch_param_norms))

            # cumulative arrays
            cum_losses = np.concatenate(losses_list, axis=0)
            cum_grads  = np.concatenate(global_norm_grads_list, axis=0)
            cum_steps  = np.concatenate(step_norms_list, axis=0)
            cum_params = np.concatenate(param_norms_list, axis=0)

            eps = 1e-16
            cum_grads_safe = np.clip(cum_grads, eps, None)

            # clear only cumulative row every loop
            for ax in [ax_cum_loss, ax_cum_grad, ax_cum_step, ax_cum_param]:
                ax.clear()

            # cumulative row
            ax_cum_loss.plot(cum_losses, color=C_LOSS)
            ax_cum_grad.plot(cum_grads_safe, color=C_GRAD)
            ax_cum_step.plot(cum_steps, color=C_STEP)
            ax_cum_param.plot(cum_params, color=C_PARAM)

            ax_cum_loss.set_title("Cumulative Loss")
            ax_cum_grad.set_title("Cumulative Grad Norm")
            ax_cum_step.set_title("Cumulative Step Norm")
            ax_cum_param.set_title("Cumulative Param Norm")

            ax_cum_loss.set_yscale("symlog", linthresh=1.0)
            ax_cum_grad.set_yscale("log")

            for ax in [ax_cum_loss, ax_cum_grad, ax_cum_step, ax_cum_param]:
                ax.set_xlabel("Global step")
            ax_cum_loss.set_ylabel("Value")

            # ---- batch summary row (clear + redraw as summary over all batches so far) ----
            for ax in [ax_batch_loss, ax_batch_grad, ax_batch_step, ax_batch_param]:
                ax.clear()

            L = _stack_hist(losses_list)
            G = _stack_hist(global_norm_grads_list)
            S = _stack_hist(step_norms_list)
            P = _stack_hist(param_norms_list)

            x = np.linspace(0.0, 1.0, L.shape[1])

            L0 = L[:, [0]]
            loss_frac = (L - L0) / (np.abs(L0) + eps)

            G0 = np.clip(G[:, [0]], eps, None)
            grad_logrel = np.log10(np.clip(G, eps, None) / G0)

            S0 = np.clip(S[:, [0]], eps, None)
            step_logrel = np.log10(np.clip(S, eps, None)+eps / (S0+eps))

            P0 = np.clip(P[:, [0]], eps, None)
            param_logrel = np.log10(np.clip(P, eps, None) / P0)

            _plot_with_iqr(ax_batch_loss, loss_frac, C_LOSS, "Per-batch Loss Change", "frac(loss-loss0)", x)
            _plot_with_iqr(ax_batch_grad, grad_logrel, C_GRAD, "Per-batch Grad Change", "log10(grad/grad0)", x)
            _plot_with_iqr(ax_batch_step, step_logrel, C_STEP, "Per-batch Step Change", "log10(step/step0)", x)
            _plot_with_iqr(ax_batch_param, param_logrel, C_PARAM, "Per-batch Param Change", "log10(param/param0)", x)

            ax_batch_loss.set_yscale("symlog", linthresh=1e-2)
            for ax in [ax_batch_loss, ax_batch_grad, ax_batch_step, ax_batch_param]:
                ax.axhline(0.0, color="k", lw=1, alpha=0.35)

            fig.suptitle(f"SVI Diagnostics (Batch {batch_num + 1}/{max_num_batches})")
            fig.tight_layout()
            if diagnostics_plot_path is not None:
                fig.savefig(diagnostics_plot_path, dpi=150, bbox_inches="tight")
            clear_output(wait=True)
            display(fig)

            # optional: quick text summary per batch
            log.info(
                f"[batch {batch_num + 1}] "
                f"loss(last)={float(batch_losses[-1]):.4g}, "
                f"grad_norm(last)={float(batch_grad_norms[-1]):.4g}, "
                f"step_norm(last)={float(batch_step_norms[-1]):.4g}"
            )

        else:
            svi_state = run_training_batch(svi, svi_state, rng_key, batch_size)
        current_val_loss = svi.evaluate(svi_state)
        total_steps = (batch_num + 1) * batch_size
        log.info(
            f"Batch {batch_num + 1}/{max_num_batches} | Total steps taken: {total_steps}",
        )

        # Early stopping logic
        log.info(f"{current_val_loss=}")
        log.info(f"{best_val_loss=}")
        difference = current_val_loss - best_val_loss if batch_num >= 1 else -np.inf
        if difference < - difference_threshold:
            log.info(
                f"Loss improved from {best_val_loss:.4f} to {current_val_loss:.4f} {difference=}. Saving state.",
            )
            best_val_loss = current_val_loss
            best_svi_state = svi_state
            patience_counter = 0
        else:
            patience_counter += 1
            log.info(
                f"Loss did not improve. Patience: {patience_counter}/{patience} {difference=}",
            )

            if patience_counter >= patience:
                log.info("Early stopping triggered. Halting training.")
                break

            log.info(f"Best loss achieved: {best_val_loss:.4f}")

            final_params = svi.get_params(best_svi_state)
    if diagnostics:
        # close the fig
        plt.close(fig)
    log.info("Optimization complete.")
    # This conditional is entered if we exhaust the max training batches
    # without early stopping
    if final_params is None:
        final_params = svi.get_params(best_svi_state)

    final_params = {
        (name[:-9] if name.endswith("_auto_loc") else name): value
        for name, value in final_params.items()
    }

    if diagnostics:
        return final_params, (svi_states_list, global_norm_grads_list)
    else:
        return final_params, None