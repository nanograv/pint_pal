"""Integration tests for discovery/noise setup workflow.

These tests intentionally use real test fixtures (config/par/tim) to validate
that setup paths work end-to-end for common `noise_run` model/inference
configurations while keeping runtime lightweight.
"""

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pint.models as models
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enterprise.pulsar import Pulsar

import pint_pal.discovery_utils as du
import pint_pal.noise_utils as nu
from pint_pal.timingconfiguration import TimingConfiguration


@pytest.fixture(scope="module")
def real_tc():
    """Load real TimingConfiguration from test fixtures."""
    parent = Path(__file__).parent
    return TimingConfiguration(
        parent / "configs" / "tctest.nb.yaml",
        tim_directory=parent / "tim",
        par_directory=parent / "results",
    )


@pytest.fixture(scope="module")
def real_model_toas(real_tc):
    """Build real model/TOAs from fixture config, preferring excised TOAs."""
    mo, to = real_tc.get_model_and_toas(excised=True, usepickle=False)
    if to is None:
        mo, to = real_tc.get_model_and_toas(apply_initial_cuts=True, usepickle=False)
    return mo, to


@pytest.fixture(scope="module")
def real_pint_model():
    """Load a real PINT timing model from fixtures for add-noise integration."""
    parent = Path(__file__).parent
    parfile = parent / "results" / "J0605+3757_PINT_20220301.nb.par"
    return models.get_model(parfile)


@pytest.fixture(autouse=True)
def _compat_log_warn(monkeypatch):
    """Compatibility shim for older/newer logger API differences."""
    if not hasattr(du.log, "warn"):
        monkeypatch.setattr(du.log, "warn", du.log.warning, raising=False)


def _config_model_kwargs_from_tc(tc):
    """Create a normalized discovery model block from fixture config."""
    base = deepcopy(tc.config.get("outlier", {}).get("model", {}))
    base.setdefault("timing_model", {"svd": True, "tm_marg": False})
    base.setdefault("white_noise", {"gp_ecorr": False, "tn_equad": True, "include_ecorr": True})
    base.setdefault("red_noise", {"basis": "fourier", "Nfreqs": 8, "prior": "powerlaw"})
    base["dm_noise"] = False
    base["chromatic_noise"] = False
    base["solar_wind"] = False
    return base


def _variant_model_kwargs(tc, **overrides):
    """Apply targeted model-block overrides on top of normalized defaults."""
    cfg = _config_model_kwargs_from_tc(tc)
    cfg.update(overrides)
    return cfg


def _enterprise_pulsar_from_real_data(real_model_toas):
    """Build enterprise Pulsar from fixture-backed model/toas."""
    model, toas = real_model_toas
    return Pulsar(model, toas, pint=True, t2=None)


def _assert_discovery_likelihood_builds(psr, model_kwargs):
    """Assert discovery likelihood setup succeeds and exposes log-likelihood."""
    pulsar_likelihood = du.make_single_pulsar_noise_likelihood_discovery(
        psr=psr,
        noise_dict={},
        tspan=None,
        model_kwargs=model_kwargs,
        return_args=False,
    )
    assert hasattr(pulsar_likelihood, "logL")
    return pulsar_likelihood


def _assert_discovery_likelihood_evaluates(pulsar_likelihood):
    """Evaluate discovery likelihood and assert finite logL across API variants."""
    def _midpoint_from_prior(par_name):
        try:
            low, high = du.ds_prior.getprior_uniform(par_name, du.ds.priordict_standard)
            return 0.5 * (float(low) + float(high))
        except (RuntimeError, KeyError, ValueError):
            alias = par_name.replace("_dm_gp_", "_dmgp_")
            if alias != par_name:
                low, high = du.ds_prior.getprior_uniform(alias, du.ds.priordict_standard)
                return 0.5 * (float(low) + float(high))

            if par_name.endswith("_alpha"):
                return 2.0
            if par_name.endswith("_log10_A"):
                return -15.5
            if par_name.endswith("_gamma"):
                return 3.5
            if par_name.endswith("_efac") or par_name.endswith("_dmefac"):
                return 1.0
            if par_name.endswith("_log10_ecorr") or par_name.endswith("_ecorr") or par_name.endswith("_dmequad"):
                return -6.5

            raise AssertionError(f"No prior midpoint mapping available for discovery parameter: {par_name}")

    sampled_params = {}

    for _ in range(128):
        try:
            logl_val = float(pulsar_likelihood.logL(sampled_params))
            assert np.isfinite(logl_val)
            return
        except KeyError as err:
            missing = str(err.args[0])
            if missing in sampled_params:
                raise
            sampled_params[missing] = _midpoint_from_prior(missing)

    raise AssertionError("Failed to evaluate discovery likelihood with finite logL after filling missing parameters.")


def _base_noise_dict(psr_name):
    """Smallest backend noise dictionary required for add-noise setup."""
    return {f"{psr_name}_Rcvr1_2_GUPPI_efac": 1.0}


def test_real_config_and_data_load_for_setup(real_tc, real_model_toas):
    """Sanity-check fixture loading used by downstream integration tests."""
    mo, to = real_model_toas
    assert real_tc.get_source() == "J0605+3757"
    assert mo.PSR.value == "J0605+3757"
    assert len(to.table) > 0


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_real_discovery_likelihood_setup_from_loaded_model(real_tc, real_model_toas):
    """Baseline: fixture model block builds a discovery likelihood."""
    e_psr = _enterprise_pulsar_from_real_data(real_model_toas)
    model_kwargs = _config_model_kwargs_from_tc(real_tc)
    pulsar_likelihood = _assert_discovery_likelihood_builds(e_psr, model_kwargs)
    _assert_discovery_likelihood_evaluates(pulsar_likelihood)


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
@pytest.mark.parametrize(
    "sampler_kwargs",
    [
        {
            "likelihood": "discovery",
            "sampler": "NUTS",
            "num_samples": 4,
            "num_warmup": 2,
            "num_chains": 1,
            "chain_method": "sequential",
            "max_tree_depth": 4,
            "dense_mass": False,
            "diagnostics": False,
        },
        {
            "likelihood": "discovery",
            "sampler": "NUTS",
            "num_samples": 3,
            "num_warmup": 1,
            "num_chains": 1,
            "chain_method": "sequential",
            "max_tree_depth": 6,
            "dense_mass": True,
            "target_accept_prob": 0.85,
            "diagnostics": False,
        },
    ],
)
def test_model_noise_setup_supports_nuts_sampler_kwargs_combinations(real_tc, real_model_toas, tmp_path, sampler_kwargs):
    """`model_noise` should accept different discovery NUTS inference knobs."""
    mo, to = real_model_toas
    model_kwargs = _config_model_kwargs_from_tc(real_tc)

    sampler = nu.model_noise(
        mo,
        to,
        using_wideband=False,
        resume=False,
        run_noise_analysis=True,
        base_op_dir=str(tmp_path),
        model_kwargs=model_kwargs,
        sampler_kwargs=sampler_kwargs,
        seed=123,
        return_sampler_without_sampling=True,
    )

    assert sampler is not None
    assert hasattr(sampler, "run")


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
@pytest.mark.parametrize(
    "variant_name,overrides",
    [
        (
            "red_only",
            {
                "red_noise": {"basis": "fourier", "Nfreqs": 12, "prior": "powerlaw"},
                "dm_noise": False,
                "chromatic_noise": False,
                "solar_wind": False,
            },
        ),
        (
            "dm_only",
            {
                "red_noise": False,
                "dm_noise": {"basis": "fourier", "Nfreqs": 16, "prior": "powerlaw"},
                "chromatic_noise": False,
                "solar_wind": False,
            },
        ),
        (
            "chromatic_only",
            {
                "red_noise": False,
                "dm_noise": False,
                "chromatic_noise": {"basis": "fourier", "Nfreqs": 10, "prior": "powerlaw"},
                "solar_wind": False,
            },
        ),
        (
            "solar_wind_fourier",
            {
                "red_noise": False,
                "dm_noise": False,
                "chromatic_noise": False,
                "solar_wind": {"basis": "fourier", "Nfreqs": 8, "prior": "powerlaw"},
            },
        ),
    ],
)
def test_discovery_likelihood_variant_configs_build(real_tc, real_model_toas, variant_name, overrides):
    """Each supported discovery model combination should build a valid likelihood."""
    e_psr = _enterprise_pulsar_from_real_data(real_model_toas)
    model_kwargs = _variant_model_kwargs(real_tc, **overrides)
    pulsar_likelihood = _assert_discovery_likelihood_builds(e_psr, model_kwargs)
    _assert_discovery_likelihood_evaluates(pulsar_likelihood)


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_discovery_likelihood_variant_solar_wind_interpolation(real_tc, real_model_toas):
    """Interpolation-basis solar wind should build when discovery backend supports it."""
    if not hasattr(du.ds_solar, "custom_blocked_interpolation_basis"):
        pytest.skip("Installed discovery.solar does not expose interpolation basis helper")

    e_psr = _enterprise_pulsar_from_real_data(real_model_toas)
    toas_days = np.asarray(e_psr.toas) / 86400.0
    nodes = np.linspace(float(np.min(toas_days)), float(np.max(toas_days)), 6)

    model_kwargs = _variant_model_kwargs(
        real_tc,
        red_noise=False,
        dm_noise=False,
        chromatic_noise=False,
        solar_wind={
            "basis": "interpolation",
            "basis_nodes": nodes,
            "prior": "square_exponential",
            "interp_kind": "linear",
        },
    )

    pulsar_likelihood = _assert_discovery_likelihood_builds(e_psr, model_kwargs)
    _assert_discovery_likelihood_evaluates(pulsar_likelihood)


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_discovery_likelihood_invalid_basis_raises(real_tc, real_model_toas):
    """Unsupported basis should fail at model construction time."""
    e_psr = _enterprise_pulsar_from_real_data(real_model_toas)

    model_kwargs = _variant_model_kwargs(
        real_tc,
        red_noise={"basis": "interpolation", "Nfreqs": 10, "prior": "powerlaw"},
        dm_noise=False,
        chromatic_noise=False,
        solar_wind=False,
    )

    with pytest.raises(NotImplementedError):
        du.make_single_pulsar_noise_likelihood_discovery(
            psr=e_psr,
            noise_dict={},
            tspan=None,
            model_kwargs=model_kwargs,
            return_args=False,
        )


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_discovery_likelihood_invalid_prior_raises(real_tc, real_model_toas):
    """Unsupported prior should fail with the expected ValueError message."""
    e_psr = _enterprise_pulsar_from_real_data(real_model_toas)

    model_kwargs = _variant_model_kwargs(
        real_tc,
        red_noise=False,
        dm_noise={"basis": "fourier", "Nfreqs": 10, "prior": "definitely_not_a_prior"},
        chromatic_noise=False,
        solar_wind=False,
    )

    with pytest.raises(ValueError, match=r"Invalid \*prior\* specified for Fourier basis DM noise"):
        du.make_single_pulsar_noise_likelihood_discovery(
            psr=e_psr,
            noise_dict={},
            tspan=None,
            model_kwargs=model_kwargs,
            return_args=False,
        )


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
@pytest.mark.parametrize(
    "inference_block",
    [
        {
            "likelihood": "discovery",
            "sampler": "optimizer",
            "num_warmup_steps": 25,
            "max_epochs": 50,
            "peak_learning_rate": 0.01,
            "gradient_clipping_val": None,
            "batch_size": 4,
            "patience": 2,
            "difference_threshold": 0.5,
            "max_num_batches": 2,
            "diagnostics": False,
        },
        {
            "likelihood": "discovery",
            "sampler": "optimizer",
            "num_warmup_steps": 10,
            "max_epochs": 20,
            "peak_learning_rate": 0.003,
            "gradient_clipping_val": 1.0,
            "batch_size": 3,
            "patience": 1,
            "difference_threshold": 0.1,
            "max_num_batches": 1,
            "diagnostics": False,
        },
    ],
)
def test_model_noise_setup_supports_optimizer_sampler_kwargs_combinations(real_tc, real_model_toas, tmp_path, monkeypatch, inference_block):
    """`model_noise` optimizer path should consume different inference settings."""
    mo, to = real_model_toas
    model_kwargs = _config_model_kwargs_from_tc(real_tc)

    captured_calls = {}

    monkeypatch.setattr(
        du,
        "setup_svi",
        lambda model, guide, loss, num_warmup_steps, max_epochs, peak_learning_rate, gradient_clipping_val: captured_calls.update(
            {
                "setup_svi": {
                    "num_warmup_steps": num_warmup_steps,
                    "max_epochs": max_epochs,
                    "peak_learning_rate": peak_learning_rate,
                    "gradient_clipping_val": gradient_clipping_val,
                }
            }
        )
        or "svi",
    )
    monkeypatch.setattr(
        du,
        "run_svi_early_stopping",
        lambda rng_key, svi, batch_size, patience, max_num_batches, difference_threshold, diagnostics, outdir, file_prefix: captured_calls.update(
            {
                "run_svi": {
                    "batch_size": batch_size,
                    "patience": patience,
                    "max_num_batches": max_num_batches,
                    "difference_threshold": difference_threshold,
                    "diagnostics": diagnostics,
                }
            }
        )
        or ({"a": 1.0}, None),
    )

    result = nu.model_noise(
        mo,
        to,
        using_wideband=False,
        resume=False,
        run_noise_analysis=True,
        base_op_dir=str(tmp_path),
        model_kwargs=model_kwargs,
        sampler_kwargs=inference_block,
        seed=456,
        return_sampler_without_sampling=False,
    )

    assert result is None
    assert captured_calls["setup_svi"]["num_warmup_steps"] == inference_block["num_warmup_steps"]
    assert captured_calls["setup_svi"]["max_epochs"] == inference_block["max_epochs"]
    assert captured_calls["setup_svi"]["peak_learning_rate"] == inference_block["peak_learning_rate"]
    assert captured_calls["setup_svi"]["gradient_clipping_val"] == inference_block["gradient_clipping_val"]
    assert captured_calls["run_svi"]["batch_size"] == inference_block["batch_size"]
    assert captured_calls["run_svi"]["patience"] == inference_block["patience"]
    assert captured_calls["run_svi"]["max_num_batches"] == inference_block["max_num_batches"]
    assert captured_calls["run_svi"]["difference_threshold"] == inference_block["difference_threshold"]


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_add_noise_to_model_with_real_model_and_synthetic_noise(real_pint_model):
    """Baseline integration: EFAC/ECORR/red-noise terms are added to a real model."""
    model = deepcopy(real_pint_model)
    psr = model.PSR.value

    noise_dict = {
        f"{psr}_Rcvr1_2_GUPPI_efac": 1.12,
        f"{psr}_Rcvr_800_GUPPI_efac": 0.97,
        f"{psr}_Rcvr1_2_GUPPI_ecorr": -6.0,
        f"{psr}_Rcvr_800_GUPPI_ecorr": -5.8,
        f"{psr}_red_noise_log10_A": -15.3,
        f"{psr}_red_noise_gamma": 4.2,
    }

    out = nu.add_noise_to_model(
        model=model,
        noise_dict=noise_dict,
        model_kwargs={"red_noise": {"Nfreqs": 12}},
        using_wideband=False,
    )

    assert out is model
    assert "ScaleToaError" in model.components
    assert "EcorrNoise" in model.components
    assert "PLRedNoise" in model.components

    rn = model.components["PLRedNoise"]
    assert float(rn.TNREDAMP.value) == pytest.approx(noise_dict[f"{psr}_red_noise_log10_A"])
    assert float(rn.TNREDGAM.value) == pytest.approx(noise_dict[f"{psr}_red_noise_gamma"])
    assert int(rn.TNREDC.value) == 12


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_add_noise_to_model_adds_dm_gp_powerlaw(real_pint_model):
    model = deepcopy(real_pint_model)
    psr = model.PSR.value
    noise_dict = _base_noise_dict(psr)
    noise_dict.update({f"{psr}_dm_gp_log10_A": -13.6, f"{psr}_dm_gp_gamma": 2.1})

    out = nu.add_noise_to_model(model, noise_dict, model_kwargs={"dm_noise": {"Nfreqs": 20}})
    assert out is model
    assert "PLDMNoise" in model.components


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_add_noise_to_model_adds_chromatic_gp_powerlaw(real_pint_model):
    model = deepcopy(real_pint_model)
    psr = model.PSR.value
    noise_dict = _base_noise_dict(psr)
    noise_dict.update({f"{psr}_chrom_gp_log10_A": -13.9, f"{psr}_chrom_gp_gamma": 3.0})

    if hasattr(nu.pm, "PLCMNoise"):
        out = nu.add_noise_to_model(model, noise_dict, model_kwargs={"chromatic_noise": {"Nfreqs": 15}})
        assert out is model
        assert "PLCMNoise" in model.components
    else:
        with pytest.raises(AttributeError, match="PLCMNoise"):
            nu.add_noise_to_model(model, noise_dict, model_kwargs={"chromatic_noise": {"Nfreqs": 15}})


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_add_noise_to_model_deterministic_solar_wind_currently_notimplemented(real_pint_model):
    model = deepcopy(real_pint_model)
    psr = model.PSR.value
    noise_dict = _base_noise_dict(psr)
    noise_dict.update({"n_earth": 6.2, f"{psr}_sw_gp_log10_A": -13.0, f"{psr}_sw_gp_gamma": 2.6})

    with pytest.raises(NotImplementedError):
        nu.add_noise_to_model(model, noise_dict, model_kwargs={"solar_wind": {"Nfreqs": 12}})


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_add_noise_to_model_adds_solar_wind_powerlaw_gp_only(real_pint_model):
    model = deepcopy(real_pint_model)
    psr = model.PSR.value
    noise_dict = _base_noise_dict(psr)
    noise_dict.update({f"{psr}_sw_gp_log10_A": -13.0, f"{psr}_sw_gp_gamma": 2.6})

    if hasattr(nu.pm, "PLSWNoise"):
        out = nu.add_noise_to_model(model, noise_dict, model_kwargs={"solar_wind": {"Nfreqs": 12}})
        assert out is model
        assert "SolarWindDispersion" in model.components
        assert "PLSWNoise" in model.components
    else:
        with pytest.raises(AttributeError, match="PLSWNoise"):
            nu.add_noise_to_model(model, noise_dict, model_kwargs={"solar_wind": {"Nfreqs": 12}})


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
@pytest.mark.parametrize(
    "sw_keys,class_name",
    [
        ({"sw_gp_log10_sigma_ridge": -7.0}, "TimeDomainRideSWNoise"),
        ({"sw_gp_log10_sigma_sq_exp": -7.1, "sw_gp_log10_ell": 1.2}, "TimeDomainSqExpSWNoise"),
        (
            {
                "sw_gp_log10_sigma_quasi_periodic": -7.2,
                "sw_gp_log10_ell": 1.1,
                "sw_gp_log10_gamma_p": -0.2,
                "sw_gp_log10_p": 1.5,
            },
            "TimeDomainQuasiPeriodicSWNoise",
        ),
        ({"sw_gp_log10_sigma_matern": -7.3, "sw_gp_log10_ell": 1.0}, "TimeDomainMaternSWNoise"),
    ],
)
def test_add_noise_to_model_adds_time_domain_solar_wind_variants(real_pint_model, sw_keys, class_name):
    model = deepcopy(real_pint_model)
    psr = model.PSR.value

    noise_dict = _base_noise_dict(psr)
    noise_dict.update({f"{psr}_{k}": v for k, v in sw_keys.items()})

    if hasattr(nu.pm, class_name):
        out = nu.add_noise_to_model(
            model,
            noise_dict,
            model_kwargs={"solar_wind": {"dt": 14.0, "interp_kind": "linear", "nu": 1.5}},
        )
        assert out is model
        assert "SolarWindDispersion" in model.components
        assert class_name in model.components
    else:
        with pytest.raises(AttributeError, match=class_name):
            nu.add_noise_to_model(
                model,
                noise_dict,
                model_kwargs={"solar_wind": {"dt": 14.0, "interp_kind": "linear", "nu": 1.5}},
            )


@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_add_noise_to_model_adds_wideband_dm_white_noise_components(real_pint_model):
    model = deepcopy(real_pint_model)
    psr = model.PSR.value

    noise_dict = {
        f"{psr}_Rcvr1_2_GUPPI_efac": 1.0,
        f"{psr}_Rcvr1_2_GUPPI_dmefac": 1.1,
        f"{psr}_Rcvr1_2_GUPPI_dmequad": -7.0,
    }

    out = nu.add_noise_to_model(model, noise_dict, model_kwargs={}, using_wideband=True)

    assert out is model
    assert "ScaleToaError" in model.components
    assert "ScaleDmError" in model.components
