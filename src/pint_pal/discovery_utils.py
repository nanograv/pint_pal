# Generic imports
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as log
from multiprocessing import Pool
from pathlib import Path
from datetime import datetime
import dill as pickle

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

def run_discovery_with_checkpoints(
    sampler: numpyro.infer.MCMC,
    n_checkpoints: int = 10,
    n_samples_per_checkpoint: int = 2000,
    outdir: str = ".",
    file_basename: str = "results",
    rng_key: jax.random.PRNGKey = None,
    resume: bool = True,
) -> None:
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