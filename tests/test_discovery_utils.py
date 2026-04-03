"""PR-focused unit tests for discovery utility helpers.

These tests emphasize branch coverage and explicit behavior checks while
keeping test scaffolding lightweight with local fakes/monkeypatching.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pint_pal.discovery_utils as du


def _solar_psr(days):
    return SimpleNamespace(toas=np.array(days) * 86400.0)


def test_select_fourier_basis_negative_nlog_raises():
    with pytest.raises(ValueError, match="non-negative"):
        du._select_fourier_basis(
            psr=object(),
            Nfreqs=10,
            tspan=100.0,
            logmode=2,
            f_min=1e-3,
            nlog=-1,
            noise_type="red_noise",
        )


def test_red_noise_block_scales_fmin_and_uses_getspan(monkeypatch):
    calls = {}
    psr = object()
    sentinel_prior = object()

    monkeypatch.setattr(du.ds, "getspan", lambda _psr: 100.0)
    monkeypatch.setattr(du.ds, "powerlaw", sentinel_prior)

    def fake_select(_psr, nfreqs, tspan, logmode, f_min, nlog, noise_type):
        calls["select"] = {
            "Nfreqs": nfreqs,
            "tspan": tspan,
            "logmode": logmode,
            "f_min": f_min,
            "nlog": nlog,
            "noise_type": noise_type,
        }
        return "BASIS"

    def fake_makegp_fourier(_psr, prior, nfreqs, T, fourierbasis, name):
        calls["makegp"] = {
            "prior": prior,
            "Nfreqs": nfreqs,
            "T": T,
            "fourierbasis": fourierbasis,
            "name": name,
        }
        return "RN_BLOCK"

    monkeypatch.setattr(du, "_select_fourier_basis", fake_select)
    monkeypatch.setattr(du.ds, "makegp_fourier", fake_makegp_fourier)

    result = du.red_noise_block(
        psr,
        tspan=None,
        prior="powerlaw",
        Nfreqs=12,
        logmode=3,
        f_min_frac=0.2,
        nlog=4,
        name="rn_custom",
    )

    assert result == "RN_BLOCK"
    assert calls["select"]["noise_type"] == "red_noise"
    assert calls["select"]["f_min"] == pytest.approx(0.2 * (1 / 100.0))
    assert calls["makegp"]["prior"] is sentinel_prior
    assert calls["makegp"]["T"] == 100.0
    assert calls["makegp"]["fourierbasis"] == "BASIS"


def test_dm_noise_block_invalid_basis_raises():
    with pytest.raises(ValueError, match="Invalid basis specified for dm noise"):
        du.dm_noise_block(psr=object(), tspan=100.0, basis="bad_basis")


def test_solar_wind_interpolation_rejects_powerlaw_and_builds_default_nodes(monkeypatch):
    calls = {}

    def fake_custom_blocked_interpolation_basis(toas, nodes, kind):
        calls["toas"] = toas
        calls["nodes"] = nodes
        calls["kind"] = kind
        return "UMAT", "NODES"

    monkeypatch.setattr(
        du.ds_solar,
        "custom_blocked_interpolation_basis",
        fake_custom_blocked_interpolation_basis,
        raising=False,
    )

    with pytest.raises(ValueError, match="Power-law prior is not supported"):
        du.solar_wind_noise_block(
            _solar_psr([10.0, 22.0]),
            basis="interpolation",
            prior="powerlaw",
            interp_dt=5.0,
            interp_kind="linear",
        )

    assert calls["kind"] == "linear"
    assert np.allclose(calls["nodes"], np.arange(10.0, 22.0, 5.0))


def test_basic_noise_blocks_delegate(monkeypatch):
    seen = {}

    monkeypatch.setattr(du.ds, "makegp_timing", lambda psr, svd, variable: (psr, svd, variable))
    monkeypatch.setattr(
        du.ds,
        "makenoise_measurement",
        lambda psr, tnequad, ecorr, selection, noisedict: (psr, tnequad, ecorr, selection, noisedict),
    )
    monkeypatch.setattr(
        du.ds,
        "makegp_ecorr",
        lambda psr, noisedict, selection, gp_ecorr_name: (psr, noisedict, selection, gp_ecorr_name),
    )

    psr = object()
    assert du.timing_model_block(psr, svd=False, tm_marg=False) == (psr, False, True)
    assert du.white_noise_block(psr, noise_dict={"x": 1}, include_ecorr=False, tn_equad=False, selection="sel") == (
        psr,
        False,
        False,
        "sel",
        {"x": 1},
    )
    assert du.gp_ecorr_block(psr, noise_dict={"y": 2}, selection="sel2", gp_ecorr_name="g") == (
        psr,
        {"y": 2},
        "sel2",
        "g",
    )


def test_select_fourier_basis_nlog_zero_returns_expected_objects(monkeypatch):
    monkeypatch.setattr(du.ds, "fourierbasis", object())
    monkeypatch.setattr(du.ds, "dmfourierbasis", object())
    monkeypatch.setattr(du.ds, "dmfourierbasis_alpha", object())
    monkeypatch.setattr(du.ds_solar, "fourierbasis_solar_dm", object(), raising=False)

    assert du._select_fourier_basis(None, 1, 1.0, 0, 0.1, 0, "red_noise") is du.ds.fourierbasis
    assert du._select_fourier_basis(None, 1, 1.0, 0, 0.1, 0, "dm_noise") is du.ds.dmfourierbasis
    assert du._select_fourier_basis(None, 1, 1.0, 0, 0.1, 0, "chromatic") is du.ds.dmfourierbasis_alpha
    assert du._select_fourier_basis(None, 1, 1.0, 0, 0.1, 0, "solar_wind") is du.ds_solar.fourierbasis_solar_dm


def test_select_fourier_basis_nlog_positive_calls_expected_builder(monkeypatch):
    called = {}
    monkeypatch.setattr(
        du.ds,
        "log_dm_fourierbasis",
        lambda psr, T, logmode, f_min, nlin, nlog: called.update(
            dict(psr=psr, T=T, logmode=logmode, f_min=f_min, nlin=nlin, nlog=nlog)
        )
        or "ok",
    )

    fn = du._select_fourier_basis("PSR", 7, 100.0, 3, 0.01, 2, "dm_noise")
    out = fn(None, None, None)

    assert out == "ok"
    assert called["psr"] == "PSR"
    assert called["T"] == 100.0
    assert called["nlin"] == 7
    assert called["nlog"] == 2


def test_block_prior_and_basis_validation_errors():
    with pytest.raises(ValueError, match=r"Invalid \*prior\* specified for Fourier basis red noise"):
        du.red_noise_block(object(), tspan=10.0, prior="bad")
    with pytest.raises(NotImplementedError):
        du.red_noise_block(object(), tspan=10.0, basis="interpolation")
    with pytest.raises(NotImplementedError):
        du.dm_noise_block(object(), tspan=10.0, basis="interpolation")
    with pytest.raises(ValueError, match=r"Invalid \*prior\* specified for Fourier basis chromatic noise"):
        du.chromatic_noise_block(object(), tspan=10.0, prior="bad")
    with pytest.raises(ValueError, match=r"Invalid \*basis\* specified for chromatic noise"):
        du.chromatic_noise_block(object(), tspan=10.0, basis="bad")
    with pytest.raises(ValueError, match="Invalid basis specified for solar wind noise"):
        du.solar_wind_noise_block(object(), tspan=10.0, basis="bad")


def test_solar_wind_interpolation_supported_prior(monkeypatch):
    monkeypatch.setattr(du.ds_solar, "custom_blocked_interpolation_basis", lambda *a, **k: ("U", "N"), raising=False)
    monkeypatch.setattr(du.ds_solar, "matern_kernel", lambda: "K", raising=False)
    monkeypatch.setattr(
        du.ds_solar,
        "makegp_timedomain_solar_dm",
        lambda psr, covariance, dt, Umat, nodes, common, name: {
            "psr": psr,
            "covariance": covariance,
            "Umat": Umat,
            "nodes": nodes,
            "name": name,
        },
        raising=False,
    )

    result = du.solar_wind_noise_block(_solar_psr([0.0, 1.0]), basis="interpolation", basis_nodes=np.array([1.0, 2.0]), prior="matern")
    assert result["covariance"] == "K"
    assert result["Umat"] == "U"
    assert result["name"] == "sw_gp"


def test_make_single_pulsar_noise_likelihood_discovery_builds_all_args(monkeypatch):
    psr = SimpleNamespace(residuals="res", toas=np.array([0.0, 1.0]))

    monkeypatch.setattr(du.ds, "getspan", lambda _x: 123.0)
    monkeypatch.setattr(du, "timing_model_block", lambda *a, **k: "tm")
    monkeypatch.setattr(du, "gp_ecorr_block", lambda *a, **k: "gpec")
    monkeypatch.setattr(du, "white_noise_block", lambda *a, **k: "wn")
    monkeypatch.setattr(du, "red_noise_block", lambda *a, **k: "rn")
    monkeypatch.setattr(du, "dm_noise_block", lambda *a, **k: "dm")
    monkeypatch.setattr(du, "chromatic_noise_block", lambda *a, **k: "chrom")
    monkeypatch.setattr(du, "solar_wind_noise_block", lambda *a, **k: "sw")
    monkeypatch.setattr(du.ds, "PulsarLikelihood", lambda args: ("PL", args))

    model_kwargs = {
        "timing_model": {"svd": True, "tm_marg": False},
        "white_noise": {"gp_ecorr": True, "include_ecorr": True, "tn_equad": True},
        "red_noise": {"basis": "fourier"},
        "dm_noise": {"basis": "fourier"},
        "chromatic_noise": {"basis": "fourier"},
        "solar_wind": {"basis": "fourier", "prior": "powerlaw"},
    }

    out = du.make_single_pulsar_noise_likelihood_discovery(psr, noise_dict={}, tspan=None, model_kwargs=model_kwargs, return_args=False)
    assert out[0] == "PL"
    assert out[1][0] == "res"
    assert out[1][1:] == ("tm", "gpec", "wn", "rn", "dm", "chrom", "sw")
    assert model_kwargs["white_noise"]["include_ecorr"] is False
    assert model_kwargs["red_noise"]["tspan"] == 123.0


def test_make_sampler_nuts_filters_and_attaches_to_df(monkeypatch):
    class FakeNUTS:
        def __init__(self, model, **kwargs):
            self.model = model
            self.kwargs = kwargs

    class FakeMCMC:
        def __init__(self, kernel, **kwargs):
            self.kernel = kernel
            self.kwargs = kwargs

        def get_samples(self):
            return {"a": [1, 2]}

    monkeypatch.setattr(du.infer, "NUTS", FakeNUTS)
    monkeypatch.setattr(du.infer, "MCMC", FakeMCMC)

    class Model:
        @staticmethod
        def to_df(samples):
            return pd.DataFrame(samples)

    sampler = du.make_sampler_nuts(Model, sampler_kwargs={"num_samples": 3, "num_warmup": 2, "max_tree_depth": 7})
    assert sampler.kwargs["num_samples"] == 3
    assert sampler.kernel.kwargs["max_tree_depth"] == 7
    assert list(sampler.to_df().columns) == ["a"]


def test_make_numpyro_model_calls_sample_and_factor(monkeypatch):
    sampled = []
    factored = {}
    monkeypatch.setattr(du.ds_prior, "getprior_uniform", lambda par, pdict: (0.0, 1.0))
    monkeypatch.setattr(du.numpyro, "sample", lambda name, distobj: sampled.append(name) or 0.5)
    monkeypatch.setattr(du.numpyro, "factor", lambda name, val: factored.update({name: val}))

    class LnLike:
        params = ["x", "y"]

        def __call__(self, pars):
            return pars["x"] + pars["y"]

    model = du.make_numpyro_model(LnLike(), {"x": (0.0, 1.0)})
    model()
    assert sampled == ["x", "y"]
    assert factored["logl"] == 1.0


def test_run_nuts_with_checkpoints_saves_chain_and_checkpoint(tmp_path, monkeypatch):
    class FakeSampler:
        def __init__(self):
            self.num_samples = 5
            self.last_state = {"s": 0}
            self.post_warmup_state = None
            self.calls = 0

        def _set_collection_params(self):
            return None

        def run(self, _rng):
            self.calls += 1
            self.last_state = {"s": self.calls}

        def to_df(self):
            return pd.DataFrame({"x": np.arange(self.num_samples)})

    saved = {}
    monkeypatch.setattr(du, "save_chain", lambda df, path: saved.update({"rows": len(df), "path": path}))
    monkeypatch.setattr(du.jax.random, "split", lambda key: (key, key))

    sampler = FakeSampler()
    du.run_nuts_with_checkpoints(
        sampler=sampler,
        num_samples_per_checkpoint=2,
        rng_key=np.array([0, 1]),
        outdir=tmp_path,
        file_name="abc",
        diagnostics=False,
    )

    assert sampler.calls == 3
    assert saved["rows"] == 5
    assert (tmp_path / "abc-checkpoint.pickle").is_file()


def test_setup_svi_uses_optax_and_returns_svi(monkeypatch):
    monkeypatch.setattr(du, "Trace_ELBO", lambda: "ELBO")
    monkeypatch.setattr(du.optax, "warmup_cosine_decay_schedule", lambda **k: ("sched", k))
    monkeypatch.setattr(du.optax, "adamw", lambda learning_rate: ("adamw", learning_rate))
    monkeypatch.setattr(du.optax, "clip_by_global_norm", lambda x: ("clip", x))
    monkeypatch.setattr(du.optax, "chain", lambda *ops: ("chain", ops))
    monkeypatch.setattr(du.numpyro.optim, "optax_to_numpyro", lambda opt: ("nopt", opt))
    monkeypatch.setattr(du, "SVI", lambda model, guide, opt, loss: (model, guide, opt, loss))

    out = du.setup_svi(lambda: None, lambda: None, gradient_clipping_val=1.0)
    assert out[2][0] == "nopt"
    assert out[3] == "ELBO"


def test_run_training_batch_and_diagnostics_helpers(monkeypatch):
    class FakeState:
        def __init__(self, step):
            self.step = step
            self.rng_key = np.array([0, 1], dtype=np.uint32)

        def __add__(self, other):
            return FakeState(self.step + other)

    class FakeSVI:
        def update(self, state):
            return state + 1, float(state.step)

        def get_params(self, state):
            return np.array([state.step, state.step + 1.0])

        class loss:
            @staticmethod
            def loss(rng, params, model, guide):
                return np.sum(params)

        model = object()
        guide = object()

    class FakeLax:
        @staticmethod
        def scan(body_fn, carry, xs=None, length=1):
            ys = []
            for i in range(length):
                carry, y = body_fn(carry, i)
                ys.append(y)
            if ys and isinstance(ys[0], tuple):
                cols = []
                for j in range(len(ys[0])):
                    values = [row[j] for row in ys]
                    try:
                        cols.append(np.array(values, dtype=float))
                    except (TypeError, ValueError):
                        cols.append(values)
                cols = tuple(cols)
                return carry, cols
            return carry, ys

    monkeypatch.setattr(du.jax, "lax", FakeLax)
    monkeypatch.setattr(du.jax.random, "split", lambda key: (key + 1, key + 2))
    monkeypatch.setattr(du.jax, "grad", lambda fn, argnums=1: (lambda *a, **k: np.array([3.0, 4.0])))
    monkeypatch.setattr(du.jax.tree_util, "tree_leaves", lambda tree: [np.asarray(tree)])
    monkeypatch.setattr(du.jax.tree_util, "tree_map", lambda fn, a, b: fn(a, b))

    s1 = du.run_training_batch.__wrapped__(FakeSVI(), FakeState(0), 0, 3)
    assert s1.step == 3

    final_state, states, losses, grad_norms, step_norms, param_norms = du.run_training_batch_with_diagnostics.__wrapped__(
        FakeSVI(), FakeState(0), 0, 2
    )
    assert final_state.step == 2
    assert len(losses) == 2
    assert len(grad_norms) == 2


def test_stack_plot_tree_and_svi_early_stopping(monkeypatch):
    stacked = du._stack_hist([[1, 2], [3, 4]])
    assert stacked.shape == (2, 2)

    class FakeAx:
        def plot(self, *a, **k):
            return None

        def fill_between(self, *a, **k):
            return None

        def set_title(self, *a, **k):
            return None

        def set_xlabel(self, *a, **k):
            return None

        def set_ylabel(self, *a, **k):
            return None

        def grid(self, *a, **k):
            return None

    du._plot_with_iqr(FakeAx(), np.array([[1.0, 2.0], [2.0, 3.0]]), "k", "t", "y", np.array([0.0, 1.0]))
    assert float(du._tree_l2_norm(np.array([3.0, 4.0]))) == pytest.approx(5.0)

    class FakeSVI:
        def init(self, _rng):
            return 0

        def evaluate(self, state):
            return {1: 10.0, 2: 9.0, 3: 9.5}.get(state, 9.5)

        def get_params(self, state):
            return {"a_auto_loc": float(state), "b": 2.0}

    monkeypatch.setattr(du, "run_training_batch", lambda svi, state, rng, batch: state + 1)
    params, diag = du.run_svi_early_stopping(
        np.array([0, 1]),
        FakeSVI(),
        batch_size=1,
        patience=1,
        max_num_batches=5,
        difference_threshold=0.5,
    )
    assert params["a"] == 2.0
    assert params["b"] == 2.0
    assert diag is None


@pytest.mark.parametrize("prior_name,prior_attr", [("powerlaw", "powerlaw"), ("broken_powerlaw", "broken_powerlaw"), ("freespectrum", "freespectrum")])
def test_fourier_blocks_accept_supported_prior_values(monkeypatch, prior_name, prior_attr):
    sentinel_prior = object()
    monkeypatch.setattr(du.ds, prior_attr, sentinel_prior, raising=False)
    monkeypatch.setattr(du, "_select_fourier_basis", lambda *a, **k: "BASIS")
    monkeypatch.setattr(du.ds, "makegp_fourier", lambda psr, prior, *a, **k: prior)

    assert du.red_noise_block(object(), tspan=50.0, prior=prior_name) is sentinel_prior
    assert du.dm_noise_block(object(), tspan=50.0, prior=prior_name) is sentinel_prior
    assert du.chromatic_noise_block(object(), tspan=50.0, prior=prior_name) is sentinel_prior
    assert du.solar_wind_noise_block(object(), tspan=50.0, basis="fourier", prior=prior_name) is sentinel_prior


@pytest.mark.parametrize(
    "prior_name,kernel_attr",
    [
        ("ridge", "ridge_kernel"),
        ("square_exponential", "square_exponential_kernel"),
        ("quasi_periodic", "quasi_periodic_kernel"),
        ("matern", "matern_kernel"),
    ],
)
def test_solar_wind_interpolation_supported_priors(monkeypatch, prior_name, kernel_attr):
    kernel = object()
    monkeypatch.setattr(du.ds_solar, "custom_blocked_interpolation_basis", lambda *a, **k: ("U", "N"), raising=False)
    monkeypatch.setattr(du.ds_solar, kernel_attr, lambda: kernel, raising=False)
    monkeypatch.setattr(du.ds_solar, "makegp_timedomain_solar_dm", lambda psr, covariance, **k: covariance, raising=False)

    out = du.solar_wind_noise_block(_solar_psr([0.0, 2.0]), basis="interpolation", prior=prior_name, basis_nodes=np.array([1.0]))
    assert out is kernel


def test_solar_wind_interpolation_invalid_prior_raises(monkeypatch):
    monkeypatch.setattr(du.ds_solar, "custom_blocked_interpolation_basis", lambda *a, **k: ("U", "N"), raising=False)
    with pytest.raises(ValueError, match="Invalid prior specified for time domain solar wind noise"):
        du.solar_wind_noise_block(_solar_psr([0.0, 1.0]), basis="interpolation", prior="not-a-prior", basis_nodes=np.array([1.0]))


def test_make_single_pulsar_noise_likelihood_respects_disabled_model_fields(monkeypatch):
    psr = SimpleNamespace(residuals="res", toas=np.array([0.0, 1.0]))
    monkeypatch.setattr(du.ds, "getspan", lambda _x: 11.0)
    monkeypatch.setattr(du.log, "warn", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(du, "timing_model_block", lambda *a, **k: "tm")
    monkeypatch.setattr(du, "white_noise_block", lambda *a, **k: "wn")

    model_kwargs = {
        "timing_model": {"svd": True, "tm_marg": False},
        "white_noise": {"gp_ecorr": False, "include_ecorr": True, "tn_equad": True},
        "red_noise": False,
        "dm_noise": False,
        "chromatic_noise": False,
        "solar_wind": False,
    }
    args = du.make_single_pulsar_noise_likelihood_discovery(psr, noise_dict={}, tspan=None, model_kwargs=model_kwargs, return_args=True)
    assert args == ["res", "tm", "wn"]
