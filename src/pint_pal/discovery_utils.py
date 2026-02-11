# Generic imports
import os, sys
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


warnings.filterwarnings("ignore")
log.disable("pint")
log.remove()
log.add(sys.stderr, colorize=False, enqueue=True)
log.info(f"Using {jax.default_backend()} with {jax.local_device_count()} devices")

def get_discovery_prior_dictionary(override_dict: Optional[Dict[str, Any]] = {}) -> Dict[str, Any]:
    """
    Get the prior dictionary for discovery outlier analysis.

    Parameters
    ----------
    override_dict : dict, optional
        Dictionary of parameters to override the default priors.

    Returns
    -------
    dict
        Dictionary of priors.
    """
    prior_dict = ds.priordict_standard.copy()
    prior_dict.update({
        '(.*_)?red_noise_log10_A.*': [-20, -11],
        '(.*_)?red_noise_gamma.*': [0, 7],
        '(.*_)?dm_gp_log10_A.*': [-20, -11],
        '(.*_)?dm_gp_gamma.*': [0, 7],
        '(.*_)?chrom_gp_log10_A.*': [-20, -11],
        '(.*_)?chrom_gp_gamma.*': [-7, 7],
        '(.*_)?swgp_log10_A.*': [-12, -1],
        '(.*_)?swgp_gamma.*': [-7, 7],
    })
    if override_dict:
        for key, value in override_dict.items():
            prior_dict[key] = value
    return prior_dict

def timing_model_block(
        psr,
        svd=True,
        tm_marg=True,
    ):

    return ds.makegp_timing(psr, svd=svd, variable=not tm_marg)

def white_noise_block(
        psr,
        noise_dict={},
        include_ecorr=True,
        gp_ecorr=False,
        tn_equad=True,
        selection=ds.selection_backend_flags,
    ):
    return ds.makenoise_measurement(
        psr,
        tnequad=tn_equad,
        ecorr=include_ecorr,
        selection=selection,
        noisedict=noise_dict,
    )

def gp_ecorr_block(
        psr,
        noise_dict={},
        include_ecorr=True, # dummy vars
        gp_ecorr=False,
        tn_equad=True,
        selection=ds.selection_backend_flags,
        gp_ecorr_name='ecorrGP'
    ):
    return ds.makegp_ecorr(
        psr,
        noisedict=noise_dict,
        selection=selection,
        gp_ecorr_name=gp_ecorr_name,
        )

def red_noise_block(
        psr,
        basis='fourier',
        prior='powerlaw',
        Nfreqs=100,
        time_span: Optional[float] = None,
        name='red_noise',
        ):
    if basis == 'fourier':
        if prior == 'powerlaw':
            prior = ds.powerlaw
        else:
            raise ValueError("Invalid prior specified for solar wind noise. Must be 'powerlaw'.")
        rn = ds.makegp_fourier(psr, prior, Nfreqs, T=time_span, name=name)
    elif basis == 'interpolation':
        raise NotImplementedError("Interpolation basis for solar wind noise is not yet implemented.")
    else:
        raise ValueError("Invalid basis specified for solar wind noise. Must be 'fourier' or 'interpolation'.")

    return rn

def dm_noise_block(
        psr,
        basis='fourier',
        prior='powerlaw',
        Nfreqs=100,
        time_span: Optional[float] = None,
        name='dm_gp',
        ):
    if basis == 'fourier':
        if prior == 'powerlaw':
            prior = ds.powerlaw
        else:
            raise ValueError("Invalid prior specified for dm noise. Must be 'powerlaw'.")
        dmgp = ds.makegp_fourier(psr, prior, Nfreqs, T=time_span, name=name, fourierbasis=ds.dmfourierbasis)
    elif basis == 'interpolation':
        raise NotImplementedError("Interpolation basis for dm noise is not yet implemented.")
    else:
        raise ValueError("Invalid basis specified for dm noise. Must be 'fourier' or 'interpolation'.")

    return dmgp

def chromatic_noise_block(
        psr,
        time_span: Optional[float] = None,
        chromatic_idx='vary',
        ):
    rn = ds.makegp_fourier(psr, ds.powerlaw, 40, T=time_span, name='red_noise',)
    return rn

def solar_wind_noise_block(
        psr,
        basis='fourier',
        prior='powerlaw',
        nfreqs=100,
        time_span: Optional[float] = None,
        name='red_noise',
        ):
    if basis == 'fourier':
        if prior == 'powerlaw':
            prior = ds.powerlaw
        else:
            raise ValueError("Invalid prior specified for solar wind noise. Must be 'powerlaw'.")
        rn = ds.makegp_fourier(psr, prior, nfreqs, T=time_span, name=name,)
    elif basis == 'interpolation':
        raise NotImplementedError("Interpolation basis for solar wind noise is not yet implemented.")
    else:
        raise ValueError("Invalid basis specified for solar wind noise. Must be 'fourier' or 'interpolation'.")

    return rn

def make_single_pulsar_noise_likelihood_discovery(
        psr: Any,
        noise_dict: Optional[Dict[str, Any]] = None,
        time_span: Optional[float] = None,
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
    time_span : float, optional
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
    if time_span is None:
        time_span = ds.getspan([psr])
    # if marg_ne:
    #     psr.Mmat = np.hstack([psr.Mmat, np.array([ds.make_solardm(psr)(1.0)]).T])
    model_kwargs.update({ky: False for ky in ['timing_model', 'white_noise', 'red_noise', 'dm_noise', 'chromatic_noise', 'solar_wind'] if ky not in model_kwargs.keys()})
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
        numpyro_model,
        sampler_kwargs={},
        ) -> infer.MCMC:
    nutsargs = dict(
        max_tree_depth=8,
        dense_mass=False,
        forward_mode_differentiation=False,
        target_accept_prob=0.8,
        **{arg: val for arg, val in sampler_kwargs.items() if arg in inspect.getfullargspec(infer.NUTS).args}
        )
    #samples_per_checkpoint = int(sampler_kwargs.get('num_samples', 1000) / sampler_kwargs.get('num_checkpoints', 5))
    mcmcargs = dict(
        #num_samples=samples_per_checkpoint,
        #num_warmup=sampler_kwargs.get('num_warmup', 500),
        **{arg: val for arg, val in sampler_kwargs.items() if arg in inspect.getfullargspec(infer.MCMC).kwonlyargs
           #and not 'num_samples'
           }
        )
    sampler = infer.MCMC(infer.NUTS(numpyro_model, **nutsargs), **mcmcargs)
    sampler.to_df = lambda: numpyro_model.to_df(sampler.get_samples())

    return sampler

def make_numpyro_model(input_lnlike, priordict={}):
    priors_dict = ds.priordict_standard.copy()
    priors_dict.update(priordict)
    def numpyro_model():
        lnlike = input_lnlike({par: numpyro.sample(par, dist.Uniform(*ds_prior.getprior_uniform(par, priordict)))
            for par in input_lnlike.params})

        numpyro.factor('logl', lnlike)
    numpyro_model.to_df = lambda chain: pd.DataFrame(chain)

    return numpyro_model

def run_discovery_with_checkpoints(
    sampler: numpyro.infer.MCMC,
    num_checkpoints: int = 5,
    outdir: str = ".",
    file_basename: str = "results",
    rng_key: jax.random.PRNGKey = None,
    resume: bool = True,
) -> az.InferenceData:
    """
    Run a NumPyro MCMC sampler in multiple chunks with checkpointing.

    This function executes an MCMC sampler for a given number of checkpoints,
    saving ArviZ inference data at each step and overwriting the NumPyro sampler
    state in a pickle file. The ArviZ outputs are stored per iteration in
    Zarr format and concatenated into a single complete file after all
    checkpoints finish. On resume, the sampler state is reloaded from the last
    pickle file and sampling continues from the last checkpoint.

    Parameters
    ----------
    sampler : numpyro.infer.MCMC
        A NumPyro MCMC sampler object that has been initialized with a kernel.
    num_checkpoints : int, optional (default=10)
        Number of checkpoints (iterations) to run. Each checkpoint generates
        `n_samples_per_checkpoint` posterior samples.
    n_samples_per_checkpoint : int, optional (default=2000)
        Number of samples to draw at each checkpoint.
    outdir : str, optional (default=".")
        Output directory for checkpoints and results.
    file_basename : str, optional (default="results")
        Base name for output files. Files will be saved as
        `{file_basename}-arviz-checkpoint-<i>.zarr` for ArviZ
        and `{file_basename}-numpyro-sampler.pickle` for the sampler state.
    rng_key : jax.random.PRNGKey
        JAX random key used to control the random number generation in sampling.
    resume : bool, optional (default=True)
        Whether to resume from the most recent checkpoint if files exist.
        Otherwise will start overwriting saved files.

    Returns
    -------
    data : arviz.InferenceData
        Results are written to disk:
        - Iteration-specific ArviZ Zarr checkpoints
        - Overwritten NumPyro sampler pickle (latest state only)
        - Final concatenated ArviZ dataset `{file_basename}-arviz-complete.zarr`
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # paths
    sampler_pickle_path = outdir / f"{file_basename}-numpyro-sampler.pickle"

    # Determine resume iteration
    start_iter = 0
    if resume and sampler_pickle_path.exists():
        log.info("Resuming from last saved sampler state...")
        with sampler_pickle_path.open("rb") as input_file:
            sampler = pickle.load(input_file)
        sampler.post_warmup_state = sampler.last_state

        # find last ArviZ checkpoint
        checkpoint_files = sorted(outdir.glob(f"{file_basename}-arviz-checkpoint-*.zarr"))
        if checkpoint_files:
            last_file = checkpoint_files[-1]
            start_iter = int(last_file.stem.split("-")[-1]) + 1
            log.info(f"Resuming from checkpoint {start_iter}")
        else:
            log.info("No ArviZ checkpoints found, starting from 0.")

    for iteration in range(start_iter, num_checkpoints):
        before_sampling = datetime.now().astimezone().replace(microsecond=0)

        # run MCMC
        sampler.run(rng_key)

        after_sampling = datetime.now().astimezone().replace(microsecond=0)
        duration = (after_sampling - before_sampling).total_seconds() / 3600.0
        log.info(f"Saving checkpoint {iteration} after {duration:.2f} hours")

        # convert to ArviZ and save iteration-specific checkpoint
        data = az.from_numpyro(sampler)
        az.to_zarr(data, str(outdir / f"{file_basename}-arviz-checkpoint-{iteration}.zarr"))

        # overwrite NumPyro sampler pickle (always latest state)
        with sampler_pickle_path.open("wb") as output_file:
            pickle.dump(sampler, output_file)

        sampler.post_warmup_state = sampler.last_state

    log.info(f"Finished all {num_checkpoints} checkpoints. Writing concatenated samples...")

    # collect all ArviZ checkpoints into one file
    data = az.concat(
        [az.from_zarr(str(outdir / f"{file_basename}-arviz-checkpoint-{i}.zarr")) for i in range(num_checkpoints)],
        dim="draw",
    )
    az.to_zarr(data, str(outdir / f"{file_basename}-arviz-complete.zarr"))
    
    return data

def setup_svi(
    model: Callable,
    guide: Callable,
    loss: ELBO | None = None,
    num_warmup_steps: int = 500,
    max_epochs: int = 5000,
    peak_lr: float = 0.01,
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
    peak_lr : float, optional
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
    The learning rate schedule starts at 0, warms up to peak_lr over num_warmup_steps,
    then decays following a cosine schedule to 1% of peak_lr over max_epochs steps.
    """
    if loss is None:
        loss = Trace_ELBO()
    # Define the learning rate schedule
    learning_rate_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0,
        peak_value=peak_lr,
        warmup_steps=num_warmup_steps,
        decay_steps=max_epochs,
        end_value=peak_lr * 0.01,  # Decay to 10% of the peak
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

    def body_fn(carry, x):
        svi_state, rng_key = carry
        rng_key, subkey = jax.random.split(rng_key)
        new_svi_state, loss = svi.update(svi_state)
        return (new_svi_state, subkey), None

    # Use lax.scan to loop `body_fn` for `batch_size` iterations
    (final_svi_state, _), _ = jax.lax.scan(
        body_fn, (svi_state, rng_key), xs=None, length=batch_size,
    )

    return final_svi_state


@partial(jax.jit, static_argnums=(0, -1))
def run_training_batch_with_diagnostics(
    svi: SVI,
    svi_state: SVIState,
    rng_key: jax.Array,
    batch_size: int,
) -> tuple[SVIState, jnp.ndarray, jnp.ndarray]:
    """
    Run SVI updates with diagnostic information (gradient norms and intermediate states).

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
    final_svi_state : SVIState
        Final SVI state after batch_size update steps.
    svi_states : jnp.ndarray
        Array of intermediate SVI states at each step.
    global_norm_grads : jnp.ndarray
        Array of global gradient norms at each step, useful for monitoring
        training stability.

    Notes
    -----
    Computing gradients explicitly at each step adds computational overhead
    compared to run_training_batch. Use this function only when diagnostic
    information is needed for monitoring or debugging purposes.
    """

    def body_fn(carry, x):
        svi_state, rng_key, _ = carry
        rng_key, subkey = jax.random.split(rng_key)
        new_svi_state, loss = svi.update(svi_state)
        global_norm_grad = optax.global_norm(
            jax.grad(svi.loss.loss, argnums=1)(
                rng_key, svi.get_params(svi_state), svi.model, svi.guide
            ),
        )
        return (new_svi_state, subkey, global_norm_grad), None

    # Use lax.scan to loop `body_fn` for `batch_size` iterations
    (final_svi_state, _, _), (svi_states, _, global_norm_grads) = jax.lax.scan(
        body_fn, (svi_state, rng_key), xs=None, length=batch_size,
    )

    return final_svi_state, svi_states, global_norm_grads


def run_svi_early_stopping(
    rng_key: jax.Array,
    svi: SVI,
    batch_size: int = 1000,
    patience: int = 3,
    max_num_batches: int = 50,
    diagnostics: bool = False,
) -> dict:
    """
    Run SVI optimization with early stopping based on loss plateau detection.

    Training proceeds in batches of optimization steps. Early stopping is triggered
    when the validation loss fails to improve by more than 1.0 for `patience`
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
        Number of consecutive batches without improvement (>1.0 decrease in loss)
        before early stopping is triggered. Default is 3.
    max_num_batches : int, optional
        Maximum number of batches to run. Default is 50.
    diagnostics : bool, optional
        If True, collect gradient norms and intermediate states at each step.
        This adds computational overhead. Default is False.

    Returns
    -------
    dict
        Dictionary of optimized parameter values from the best SVI state
        (lowest validation loss).

    Notes
    -----
    The early stopping criterion requires the loss to improve by at least 1.0
    to reset the patience counter. This threshold is hardcoded and may need
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

    best_val_loss = float("inf")
    best_svi_state = svi_state
    patience_counter = 0

    log.info(f"Starting training with batches of {batch_size} steps.")

    final_params = None
    for batch_num in range(max_num_batches):
        if diagnostics:
            svi_state, batch_svi_states_list, batch_global_norm_grads = run_training_batch_with_diagnostics(
                svi, svi_state, rng_key, batch_size
            )
            svi_states_list.append(batch_svi_states_list)
            global_norm_grads_list.append(batch_global_norm_grads)
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
        if difference < -1:
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

    log.info("Optimization complete.")
    # This conditional is entered if we exhaust the max training batches
    # without early stopping
    if final_params is None:
        final_params = svi.get_params(best_svi_state)

    return final_params