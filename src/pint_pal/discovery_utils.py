# Generic imports
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as log
from multiprocessing import Pool
from pathlib import Path
from datetime import datetime
import dill as pickle
from typing import Any, Dict, List, Optional, Sequence, Union

# discovery outlier analysis imports
import discovery as ds
import arviz as az
from discovery import matrix, selection_backend_flags
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpyro
from numpyro import sample, factor, infer, deterministic
from numpyro import distributions as dist

def get_discovery_prior_dictionary(override_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
    if override_dict:
        for key, value in override_dict.items():
            prior_dict[key] = value
    return prior_dict

def make_spna_cnm(
        psr: Any,
        noise: Optional[Dict[str, Any]] = None,
        time_span: Optional[float] = None,
        vary_white_noise: bool = True,
        vary_ecorr: bool = True,
        include_rn: bool = True,
        include_dmgp: bool = True,
        include_chromgp: bool = True,
        chromatic_idx: Optional[float] = None,
        include_swgp: bool = True,
        marg_ne: bool = True,
        return_args: bool = False
) -> Union[ds.PulsarLikelihood, Sequence[Any]]:
    """
    Build a SPNA CNM (pulsar likelihood) composed of measurement noise and various Gaussian processes.

    Parameters
    ----------
    psr : object
        Pulsar object used by the discovery library (must provide Mmat, residuals, etc.).
    noise : dict, optional
        Static white-noise parameter dictionary used when vary_white_noise is False.
    time_span : float, optional
        Observation time span; if None, determined via ds.getspan([psr]).
    vary_white_noise : bool
        Whether to construct variable white-noise components.
    vary_ecorr : bool
        Whether epoch-correlated noise should be variable.
    include_rn : bool
        Include red-noise GP if True.
    include_dmgp : bool
        Include DM GP if True.
    include_chromgp : bool
        Include chromatic GP if True.
    chromatic_idx : float, optional
        Fixed chromatic spectral index when including chromatic GP.
    include_swgp : bool
        Include solar-wind GP if True.
    marg_ne : bool
        If True, append solar DM design column to psr.Mmat.
    return_args : bool
        If True, return the list of component arguments instead of a PulsarLikelihood.

    Returns
    -------
    discovery.PulsarLikelihood or list
        PulsarLikelihood instance assembled from components, or the raw args list if return_args is True.
    """
    if noise is None:
        noise = {}
    if time_span is None:
        time_span = ds.getspan([psr])
    if marg_ne:
        psr.Mmat = np.hstack([psr.Mmat, np.array([ds.make_solardm(psr)(1.0)]).T])
    args = []
    if vary_white_noise:
        args.append(ds.makenoise_measurement(psr, tnequad=True))
        args.append(ds.makegp_ecorr(psr))
        args.append(ds.makegp_timing(psr, svd=True, variable=True))
        if include_rn:
            args.append(ds.makegp_fourier(psr, ds.powerlaw, 40, T=time_span, name='red_noise',
                              # use lambda function since the fourierbasis has to be a callable with only psr, comp, T args
                              fourierbasis=lambda psr, comp, T : log_fourierbasis(psr, T=T, logmode=2, f_min=1/(5*time_span), nlin=40, nlog=10)
                        )
            )
    else:
        args.append(
            ds.makenoise_measurement(psr, noise)
        )
        if vary_ecorr:
            args.append(
                ds.makegp_ecorr(psr)
            )
        else:
            args.append(
                ds.makegp_ecorr(psr, noise)
            )
        args.append(
            ds.makegp_timing(psr, svd=True, variable=False)
        )
        if include_rn:
            args.append(
            ds.makegp_fourier(psr, ds.powerlaw, 40, T=time_span, name='red_noise',
                              # use lambda function since the fourierbasis has to be a callable with only psr, comp, T args
                              fourierbasis=lambda psr, comp, T : log_fourierbasis(psr, T=T, logmode=2, f_min=1/(5*time_span), nlin=40, nlog=10)
                             )
            )
    if include_dmgp:
        args.append(ds.makegp_fourier(psr, ds.powerlaw, 200, T=time_span, name='dm_gp',
                                fourierbasis=lambda psr, comp, T : log_fourierbasis_dm(psr, T=T, logmode=2, f_min=1/(5*time_span), nlin=150, nlog=10)
                                 ))
    if include_chromgp:
        if chromatic_idx is None:
            args.append(
                ds.makegp_fourier(psr, ds.powerlaw, 200, T=time_span, name='chrom_gp',
                    fourierbasis=lambda psr, comp, T : log_fourierbasis_chromatic(psr, T=T, logmode=2, f_min=1/(5*time_span), nlin=150, nlog=10)
                )
            )
        else:
            args.append(
                ds.makegp_fourier(psr, ds.powerlaw, 200, T=time_span, name='chrom_gp',
                    fourierbasis=lambda psr, comp, T : log_fourierbasis_chromatic_fixed(psr, alpha=chromatic_idx, 
                        T=T, logmode=2, f_min=1/(5*time_span), nlin=150, nlog=10)
                )
            )
    #args.append(du.makegp_td_sw(psr, ridge_kernel, M, T=time_span))
    if include_swgp:
        args.append(ds.makegp_fourier(psr, ds.powerlaw, 200, T=time_span, name='sw_gp',
                            fourierbasis=lambda psr, comp, T : log_fourierbasis_solardm(psr, T=T, logmode=2, f_min=1/(5*time_span), nlin=150, nlog=10)
                                 ))
    det_sw_subtracted_resids = np.array(psr.residuals - ds.make_solardm(psr)(6.67))
    args.append(det_sw_subtracted_resids)
    if return_args:
        return args
    else:
        return ds.PulsarLikelihood(tuple(args))



def run_discovery_with_checkpoints(
    sampler: numpyro.infer.MCMC,
    n_checkpoints: int = 10,
    n_samples_per_checkpoint: int = 2000,
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
    n_checkpoints : int, optional (default=10)
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

    for iteration in range(start_iter, n_checkpoints):
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

    log.info(f"Finished all {n_checkpoints} checkpoints. Writing concatenated samples...")

    # collect all ArviZ checkpoints into one file
    data = az.concat(
        [az.from_zarr(str(outdir / f"{file_basename}-arviz-checkpoint-{i}.zarr")) for i in range(n_checkpoints)],
        dim="draw",
    )
    az.to_zarr(data, str(outdir / f"{file_basename}-arviz-complete.zarr"))
    
    return data