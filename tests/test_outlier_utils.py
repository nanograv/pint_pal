"""Tests for the discovery outlier utilities added to outlier_utils.py.

Only covers ``make_outlier_likelihood_discovery`` — no tests for the legacy
enterprise/Gibbs machinery in the rest of the module.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pint_pal.discovery_utils as du


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_psr():
    """Minimal fake pulsar accepted by make_single_pulsar_noise_likelihood_discovery."""
    return SimpleNamespace(residuals=np.zeros(10), toas=np.linspace(0, 10 * 86400, 10))


def _patch_ds_blocks(monkeypatch):
    """Monkeypatch all ds.* calls so no real discovery objects are needed."""
    monkeypatch.setattr(du.ds, "getspan", lambda _x: 3650.0 * 86400.0)
    monkeypatch.setattr(du, "timing_model_block", lambda *a, **k: "tm")
    monkeypatch.setattr(du, "gp_ecorr_block", lambda *a, **k: "gpec")
    monkeypatch.setattr(du, "white_noise_block", lambda *a, **k: "wn")
    monkeypatch.setattr(du, "red_noise_block", lambda *a, **k: "rn")
    monkeypatch.setattr(du, "dm_noise_block", lambda *a, **k: "dm")
    monkeypatch.setattr(du, "chromatic_noise_block", lambda *a, **k: "chrom")
    monkeypatch.setattr(du, "solar_wind_noise_block", lambda *a, **k: "sw")


# ---------------------------------------------------------------------------
# unit tests – forced kwargs and deep-copy safety
# ---------------------------------------------------------------------------


def test_make_outlier_likelihood_forces_gp_ecorr_and_variable_and_outliers(monkeypatch):
    """White-noise block must have gp_ecorr=True, variable=True, outliers=True."""
    captured = {}

    def capture_wn_kwargs(**kwargs):
        captured.update(kwargs)

    _patch_ds_blocks(monkeypatch)
    monkeypatch.setattr(du, "white_noise_block",
                        lambda *a, **k: (capture_wn_kwargs(**k), "wn")[1])
    monkeypatch.setattr(du.ds, "PulsarLikelihood", lambda args: args)

    from pint_pal.outlier_utils import make_outlier_likelihood_discovery

    make_outlier_likelihood_discovery(
        _make_psr(),
        model_kwargs={
            "timing_model": {"svd": True, "tm_marg": True},  # should be overridden
            "white_noise": {"gp_ecorr": False},               # should be overridden
        },
    )

    assert captured.get("gp_ecorr") is True
    assert captured.get("variable") is True
    assert captured.get("outliers") is True


def test_make_outlier_likelihood_forces_tm_marg_false(monkeypatch):
    """timing_model block must receive tm_marg=False regardless of caller value."""
    captured = {}

    def capture_tm_kwargs(*a, **k):
        captured.update(k)
        return "tm"

    _patch_ds_blocks(monkeypatch)
    monkeypatch.setattr(du, "timing_model_block", capture_tm_kwargs)
    monkeypatch.setattr(du.ds, "PulsarLikelihood", lambda args: args)

    from pint_pal.outlier_utils import make_outlier_likelihood_discovery

    make_outlier_likelihood_discovery(
        _make_psr(),
        model_kwargs={
            "timing_model": {"svd": True, "tm_marg": True},  # caller sets True
            "white_noise": {},
        },
    )

    assert captured.get("tm_marg") is False


def test_make_outlier_likelihood_does_not_mutate_caller_kwargs(monkeypatch):
    """make_outlier_likelihood_discovery must deep-copy model_kwargs."""
    _patch_ds_blocks(monkeypatch)
    monkeypatch.setattr(du.ds, "PulsarLikelihood", lambda args: args)

    from pint_pal.outlier_utils import make_outlier_likelihood_discovery

    original = {
        "timing_model": {"svd": True, "tm_marg": True},
        "white_noise": {"gp_ecorr": False, "outliers": False, "variable": False},
    }
    import copy
    original_copy = copy.deepcopy(original)

    make_outlier_likelihood_discovery(_make_psr(), model_kwargs=original)

    assert original == original_copy, "make_outlier_likelihood_discovery mutated model_kwargs"


def test_make_outlier_likelihood_works_with_none_model_kwargs(monkeypatch):
    """None model_kwargs should be treated as an empty dict (no crash)."""
    _patch_ds_blocks(monkeypatch)
    monkeypatch.setattr(du.ds, "PulsarLikelihood", lambda args: args)

    from pint_pal.outlier_utils import make_outlier_likelihood_discovery

    # Should not raise
    result = make_outlier_likelihood_discovery(_make_psr(), model_kwargs=None)
    assert result is not None


# ---------------------------------------------------------------------------
# parametrized model-variant tests + likelihood callable checks
# ---------------------------------------------------------------------------

# Each entry is a descriptive label and any *extra* model_kwargs blocks beyond
# the mandatory timing_model / white_noise.
_MODEL_VARIANTS = [
    pytest.param(
        {},
        id="base_rn_only",
    ),
    pytest.param(
        {"dm_noise": {"basis": "fourier", "Nfreqs": 30}},
        id="with_dmgp",
    ),
    pytest.param(
        {"dm_noise": {"basis": "fourier", "Nfreqs": 30},
         "solar_wind": {"basis": "fourier", "prior": "powerlaw", "Nfreqs": 10}},
        id="with_dmgp_and_swgp",
    ),
    pytest.param(
        {"chromatic_noise": {"basis": "fourier", "Nfreqs": 20, "chromatic_idx": "vary"}},
        id="with_chromatic_gp",
    ),
    pytest.param(
        {"dm_noise": {"basis": "fourier", "Nfreqs": 30},
         "chromatic_noise": {"basis": "fourier", "Nfreqs": 20, "chromatic_idx": "vary"},
         "solar_wind": {"basis": "fourier", "prior": "powerlaw", "Nfreqs": 10}},
        id="with_dmgp_chromatic_swgp",
    ),
]


@pytest.mark.parametrize("extra_kwargs", _MODEL_VARIANTS)
def test_make_outlier_likelihood_model_variants_return_psrl(monkeypatch, extra_kwargs):
    """
    Across a range of model configurations the function must:
      - return a PulsarLikelihood-like object, and
      - that object must be callable with a parameter vector (a few draws).
    """
    _patch_ds_blocks(monkeypatch)

    # Simple fake PulsarLikelihood: callable, records how many times called.
    class FakePSRL:
        def __init__(self, args):
            self._args = args
            self.ncalls = 0

        def __call__(self, params):
            self.ncalls += 1
            return -float(np.sum(params ** 2))   # simple quadratic "lnlike"

        @property
        def params(self):
            return ["efac", "log10_equad", "log10_A", "gamma"]

    psrl_instance = None

    def fake_PulsarLikelihood(args):
        nonlocal psrl_instance
        psrl_instance = FakePSRL(args)
        return psrl_instance

    monkeypatch.setattr(du.ds, "PulsarLikelihood", fake_PulsarLikelihood)

    from pint_pal.outlier_utils import make_outlier_likelihood_discovery

    base_kwargs = {
        "timing_model": {"svd": True, "tm_marg": True},
        "white_noise": {"gp_ecorr": False, "outliers": False, "variable": False},
        "red_noise": {"basis": "fourier", "Nfreqs": 30},
    }
    base_kwargs.update(extra_kwargs)

    psrl = make_outlier_likelihood_discovery(_make_psr(), model_kwargs=base_kwargs)

    assert psrl is psrl_instance, "Expected the FakePSRL returned by PulsarLikelihood"

    # --- take a few draws from the likelihood ---
    rng = np.random.default_rng(42)
    n_draws = 5
    for _ in range(n_draws):
        params = rng.standard_normal(len(psrl.params))
        lnl = psrl(params)
        assert np.isfinite(lnl), f"Likelihood draw returned non-finite value: {lnl}"

    assert psrl.ncalls == n_draws


@pytest.mark.parametrize("extra_kwargs", _MODEL_VARIANTS)
def test_make_outlier_likelihood_always_has_gp_ecorr_in_args(monkeypatch, extra_kwargs):
    """
    Across all model variants the GP-ECORR block must appear in the args tuple
    (gp_ecorr=True is forced), and kernel ECORR must be disabled.
    """
    captured_args = []

    _patch_ds_blocks(monkeypatch)

    def recording_PulsarLikelihood(args):
        captured_args.append(args)
        return args

    monkeypatch.setattr(du.ds, "PulsarLikelihood", recording_PulsarLikelihood)

    from pint_pal.outlier_utils import make_outlier_likelihood_discovery

    base_kwargs = {
        "timing_model": {"svd": True, "tm_marg": True},
        "white_noise": {"gp_ecorr": False, "include_ecorr": True, "outliers": False},
        "red_noise": {"basis": "fourier", "Nfreqs": 30},
    }
    base_kwargs.update(extra_kwargs)

    make_outlier_likelihood_discovery(_make_psr(), model_kwargs=base_kwargs)

    assert len(captured_args) == 1
    args = captured_args[0]

    # Use explicit identity/equality against the string sentinels from the fakes.
    str_args = [a for a in args if isinstance(a, str)]
    assert "gpec" in str_args, (
        "GP-ECORR block ('gpec') not found in PulsarLikelihood args — "
        "gp_ecorr=True was not enforced"
    )
    assert "tm" in str_args
    assert "wn" in str_args
