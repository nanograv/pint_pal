"""Full run-through tests: discovery noise model -> par file -> PINT noise model.

Each case builds the discovery likelihood pint_pal fits for a ``model_kwargs`` config,
assigns every fitted parameter a value, writes those values into the timing model with
``noise_utils.add_noise_to_model``, writes the par file, reads it back with PINT, and
checks that

* every fitted parameter is accounted for in the par file (none silently dropped), and
* the PINT noise model is the model discovery fit: per-TOA white-noise variance, ECORR
  amplitudes, Fourier-GP spectral weights, and time-domain solar-wind node covariance.

The noise-model checks run on both the model ``add_noise_to_model`` returns and the one
read back from its par file: the par file holds one copy of each component, so a
component duplicated in memory only shows up in the returned model.

Models PINT cannot represent must raise NotImplementedError instead (see
``test_noise_setup_integration``); a new discovery parameter without a mapping below
fails here rather than going missing from par files.
"""

import io
import json
import re
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from pint.models import get_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pint_pal.discovery_utils as du
import pint_pal.noise_utils as nu
from tests.test_noise_setup_integration import (  # noqa: F401  (fixtures)
    _config_model_kwargs_from_tc,
    _enterprise_pulsar_from_real_data,
    real_model_toas,
    real_tc,
)

pytestmark = pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")

PRIORS = {
    **du.ds.priordict_standard,
    **json.load(open(Path(nu.__file__).parent / "discovery_priors.json")),
}

# block name in model_kwargs -> (discovery block builder, PINT component)
FOURIER_GPS = {
    "red_noise": (du.red_noise_block, "PLRedNoise"),
    "dm_noise": (du.dm_noise_block, "PLDMNoise"),
    "chromatic_noise": (du.chromatic_noise_block, "PLChromNoise"),
    "solar_wind": (du.solar_wind_noise_block, "PLSWNoise"),
}

# discovery parameter suffix -> PINT parameter written for it. White-noise and ECORR
# parameters are checked through the noise model itself (the EQUAD convention
# conversion makes a per-parameter check less direct), so they map to None here.
PARAMETER_MAP = [
    (r"_efac$", None),
    (r"_log10_(t2|tn)equad$", None),
    (r"_log10_ecorr$", None),
    (r"_red_noise_log10_A$", "TNREDAMP"),
    (r"_red_noise_gamma$", "TNREDGAM"),
    (r"_dm_gp_log10_A$", "TNDMAMP"),
    (r"_dm_gp_gamma$", "TNDMGAM"),
    (r"_chrom_gp_log10_A$", "TNCHROMAMP"),
    (r"_chrom_gp_gamma$", "TNCHROMGAM"),
    (r"_chrom_gp_alpha$", "TNCHROMIDX"),
    (r"_sw_gp_log10_A$", "TNSWAMP"),
    (r"_sw_gp_gamma$", "TNSWGAM"),
    (r"_sw_gp_log10_sigma_(ridge|sq_exp|matern|quasi_periodic)$", "TDSWLOGSIG"),
    (r"_sw_gp_log10_ell$", "TDSWLOGELL"),
    (r"_sw_gp_log10_gamma_p$", "TDSWLOGGAMP"),
    (r"_sw_gp_log10_p$", "TDSWLOGP"),
]

FOURIER = {"basis": "fourier", "Nfreqs": 20, "prior": "powerlaw"}
CASES = {
    "white_noise_tnequad": {},
    "white_noise_t2equad": dict(white_noise={"gp_ecorr": False, "tn_equad": False, "include_ecorr": True}),
    "white_noise_gp_ecorr": dict(white_noise={"gp_ecorr": True, "tn_equad": True, "include_ecorr": True}),
    "red_noise": dict(red_noise=FOURIER),
    "red_noise_log_modes": dict(red_noise={**FOURIER, "nlog": 4}),
    "dm_noise": dict(dm_noise=FOURIER),
    "chromatic_vary_index": dict(chromatic_noise=FOURIER),
    "chromatic_fixed_index": dict(chromatic_noise={**FOURIER, "chromatic_idx": 6.0}),
    "chromatic_with_quadratic": dict(chromatic_noise={**FOURIER, "include_quadratic": True}),
    "solar_wind_fourier": dict(solar_wind=FOURIER),
    **{
        f"solar_wind_{kernel}": dict(solar_wind={"basis": "interpolation", "prior": kernel})
        for kernel in ["ridge", "square_exponential", "quasi_periodic", "matern"]
    },
    "all_fourier_gps": dict(red_noise=FOURIER, dm_noise=FOURIER, chromatic_noise=FOURIER, solar_wind=FOURIER),
}

# The log-spaced modes are weighted differently: discovery uses linBinning's bin
# widths, PINT np.diff of the mode frequencies. Known PINT-side follow-up.
LOG_MODE_CASES = {"red_noise_log_modes"}


def _model_kwargs(tc, overrides):
    mk = _config_model_kwargs_from_tc(tc)
    mk["red_noise"] = False  # each case turns on only what it tests
    mk.update(overrides)
    return mk


def _fit_values(params):
    """A distinct in-prior value for every parameter, so a value written to the wrong
    PINT parameter cannot pass by coincidence."""
    values = {}
    for i, par in enumerate(sorted(params)):
        low, high = du.ds_prior.getprior_uniform(par, PRIORS)
        values[par] = low + (0.2 + 0.6 * (i + 1) / (len(params) + 1)) * (high - low)
    return values


def _gp_block_kwargs(mk, block):
    kwargs = dict(mk[block])
    if block == "chromatic_noise":
        kwargs.pop("include_quadratic", None)
    return kwargs


@pytest.fixture(scope="module")
def e_psr(real_model_toas):
    return _enterprise_pulsar_from_real_data(real_model_toas)


def _full_run_through(real_tc, real_model_toas, e_psr, overrides):
    """discovery likelihood -> fitted values -> add_noise_to_model -> par file -> PINT.

    Returns the returned (in-memory) model and the model read back from its par file.
    """
    model, toas = real_model_toas
    mk = _model_kwargs(real_tc, overrides)
    psl = du.make_single_pulsar_noise_likelihood_discovery(psr=e_psr, noise_dict={}, model_kwargs=mk)
    fitted = _fit_values(psl.logL.params)

    # the fixture model already carries EFAC/EQUAD/ECORR, as a par file from an
    # earlier noise run would; they must be replaced, not added to
    with_noise = nu.add_noise_to_model(deepcopy(model), fitted, model_kwargs=mk)
    reread = get_model(io.StringIO(with_noise.as_parfile()))
    reread.setup()
    return mk, fitted, {"returned": with_noise, "par_file": reread}, toas


@pytest.mark.parametrize("case", list(CASES))
def test_every_fitted_parameter_reaches_the_par_file(real_tc, real_model_toas, e_psr, case):
    mk, fitted, models, _ = _full_run_through(real_tc, real_model_toas, e_psr, CASES[case])
    reread = models["par_file"]

    unmapped = []
    for par, value in fitted.items():
        target = next((pint_par for pattern, pint_par in PARAMETER_MAP if re.search(pattern, par)), False)
        if target is False:
            unmapped.append(par)
        elif target is not None:
            assert getattr(reread, target).value == pytest.approx(value), f"{par} -> {target}"
    assert not unmapped, (
        f"fitted parameters with no par-file mapping: {unmapped}. Write them in "
        "add_noise_to_model (and add them to PARAMETER_MAP), or raise NotImplementedError there."
    )


@pytest.mark.parametrize("case", list(CASES))
def test_pint_white_noise_matches_discovery(real_tc, real_model_toas, e_psr, case):
    mk, fitted, models, toas = _full_run_through(real_tc, real_model_toas, e_psr, CASES[case])
    wn_kwargs = mk["white_noise"]

    measurement = du.white_noise_block(
        e_psr, tn_equad=wn_kwargs.get("tn_equad", True), include_ecorr=False,
    )
    var_discovery = np.asarray(measurement.getN(fitted))
    ecorr_discovery = np.unique(np.asarray(du.gp_ecorr_block(e_psr).Phi.getN(fitted)))

    for which, model in models.items():
        # enterprise reorders the TOAs; _isort maps its order back to PINT's
        var_pint = model.scaled_toa_uncertainty(toas).to_value("s")[e_psr._isort] ** 2
        np.testing.assert_allclose(var_pint, var_discovery, rtol=1e-10, err_msg=which)

        if wn_kwargs.get("include_ecorr", True):
            ecorr = [c for c in model.NoiseComponent_list if type(c).__name__ == "EcorrNoise"]
            assert len(ecorr) == 1, which
            np.testing.assert_allclose(
                np.unique(ecorr[0].get_noise_weights(toas)), ecorr_discovery, rtol=1e-10, err_msg=which
            )


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            c,
            marks=pytest.mark.xfail(strict=True, reason="PINT weights log modes with np.diff, discovery with linBinning bin widths")
            if c in LOG_MODE_CASES else (),
        )
        for c in CASES
        if any(block in CASES[c] and CASES[c][block].get("basis") == "fourier" for block in FOURIER_GPS)
    ],
)
def test_pint_fourier_gp_spectra_match_discovery(real_tc, real_model_toas, e_psr, case):
    mk, fitted, models, toas = _full_run_through(real_tc, real_model_toas, e_psr, CASES[case])

    for block, (builder, component) in FOURIER_GPS.items():
        if not (mk.get(block) and mk[block].get("basis") == "fourier"):
            continue
        gp = builder(e_psr, **_gp_block_kwargs(mk, block))
        phi_discovery = np.asarray(gp.Phi.getN(fitted))
        for which, model in models.items():
            comps = [c for c in model.NoiseComponent_list if type(c).__name__ == component]
            assert len(comps) == 1, f"{which}: {component}"
            # the spans differ by the barycentric offset between enterprise and PINT TOAs
            np.testing.assert_allclose(
                comps[0].get_noise_weights(toas), phi_discovery, rtol=1e-4, err_msg=f"{which}: {component}"
            )


@pytest.mark.parametrize("case", [c for c in CASES if c.startswith("solar_wind_") and c != "solar_wind_fourier"])
def test_pint_time_domain_solar_wind_covariance_matches_discovery(real_tc, real_model_toas, e_psr, case):
    mk, fitted, models, toas = _full_run_through(real_tc, real_model_toas, e_psr, CASES[case])

    gp = du.solar_wind_noise_block(e_psr, **mk["solar_wind"])
    cov_discovery = np.asarray(gp.Phi.getN(fitted))
    for which, model in models.items():
        comps = [c for c in model.NoiseComponent_list if type(c).__name__ == "TimeDomainSWNoise"]
        assert len(comps) == 1, which
        # the nodes differ by the barycentric offset (minutes) against 30-day spacing
        np.testing.assert_allclose(
            comps[0].get_noise_weights(toas), cov_discovery,
            rtol=1e-3, atol=1e-6 * np.abs(cov_discovery).max(), err_msg=which,
        )
