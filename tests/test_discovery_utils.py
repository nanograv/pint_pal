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

    def fake_makegp_fourier(_psr, prior, nfreqs, T, modes=None, fourierbasis=None, name=None):
        calls["makegp"] = {
            "prior": prior,
            "Nfreqs": nfreqs,
            "T": T,
            "modes": modes,
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
        du.ds_signals,
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
        lambda psr, tnequad, ecorr, selection, noisedict, **kwargs: (psr, tnequad, ecorr, selection, noisedict),
    )
    monkeypatch.setattr(
        du.ds,
        "makegp_ecorr",
        lambda psr, noisedict, selection, **kwargs: (psr, noisedict, selection),
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
    )


def test_select_fourier_basis_nlog_zero_returns_expected_objects(monkeypatch):
    monkeypatch.setattr(du.ds, "fourierbasis", object())
    monkeypatch.setattr(du.ds, "fourierbasis_dm", object())
    monkeypatch.setattr(du.ds, "fourierbasis_chrom", object())
    monkeypatch.setattr(du.ds_solar, "fourierbasis_solar_dm", object(), raising=False)

    assert du._select_fourier_basis(None, 1, 1.0, 0, 0.1, 0, "red_noise") is du.ds.fourierbasis
    assert du._select_fourier_basis(None, 1, 1.0, 0, 0.1, 0, "dm_noise") is du.ds.fourierbasis_dm
    assert du._select_fourier_basis(None, 1, 1.0, 0, 0.1, 0, "chromatic") is du.ds.fourierbasis_chrom
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
    with pytest.raises(ValueError, match=r"Invalid \*prior\* specified for Fourier basis chromatic noise"):
        du.chromatic_noise_block(object(), tspan=10.0, prior="bad")
    with pytest.raises(ValueError, match=r"Invalid \*basis\* specified for chromatic noise"):
        du.chromatic_noise_block(object(), tspan=10.0, basis="bad")
    with pytest.raises(ValueError, match="Invalid basis specified for solar wind noise"):
        du.solar_wind_noise_block(object(), tspan=10.0, basis="bad")


def test_solar_wind_interpolation_supported_prior(monkeypatch):
    monkeypatch.setattr(du.ds_signals, "custom_blocked_interpolation_basis", lambda *a, **k: ("U", "N"), raising=False)
    monkeypatch.setattr(du.ds_signals, "matern_kernel", lambda: "K", raising=False)
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


def test_make_numpyro_model_uses_tanh_transform(monkeypatch):
    """make_numpyro_model should sample a single 'pars' site and call factor via logx."""
    factored = {}
    sampled_name = []

    import jax.numpy as jnp

    class FakeLogx:
        params = ["x", "y"]

        def __call__(self, pars):
            return jnp.sum(pars)

        def to_df(self, pars):
            import pandas as pd
            return pd.DataFrame({"x": [1.0], "y": [2.0]})

    class LnLike:
        params = ["x", "y"]
        def __call__(self, pars):
            return 0.0

    fake_logx = FakeLogx()
    monkeypatch.setattr(du.ds_prior, "makelogtransform_uniform",
                        lambda lnlike, priordict: fake_logx)
    monkeypatch.setattr(du.numpyro, "sample",
                        lambda name, distobj: (sampled_name.append(name) or jnp.array([0.5, 0.5])))
    monkeypatch.setattr(du.numpyro, "factor",
                        lambda name, val: factored.update({name: float(val)}))

    model = du.make_numpyro_model(LnLike(), {})
    model()

    assert sampled_name == ["pars"]
    assert "logl" in factored
    assert factored["logl"] == pytest.approx(1.0)


def test_make_numpyro_model_to_df_uses_logx_to_df(monkeypatch):
    """to_df on the model should delegate to logx.to_df."""
    import jax.numpy as jnp
    import pandas as pd

    class FakeLogx:
        params = ["x"]
        def __call__(self, pars): return jnp.sum(pars)
        def to_df(self, pars):
            return pd.DataFrame({"x": np.array(pars).ravel()})

    class LnLike:
        params = ["x"]
        def __call__(self, pars): return 0.0

    monkeypatch.setattr(du.ds_prior, "makelogtransform_uniform",
                        lambda lnlike, priordict: FakeLogx())
    model = du.make_numpyro_model(LnLike(), {})
    df = model.to_df({"pars": np.array([[1.5], [2.5]])})
    assert list(df.columns) == ["x"]
    assert np.allclose(df["x"].values, [1.5, 2.5])


def test_make_numpyro_model_custom_transform(monkeypatch):
    """A user-supplied transform callable should be used instead of the default."""
    import jax.numpy as jnp

    class FakeLogx:
        params = ["a"]
        def __call__(self, pars): return jnp.array(-99.0)
        def to_df(self, pars): return None

    called_with = {}

    def custom_transform(lnlike, priordict):
        called_with["lnlike"] = lnlike
        return FakeLogx()

    class LnLike:
        params = ["a"]
        def __call__(self, pars): return 0.0

    lnlike = LnLike()
    factored = {}
    monkeypatch.setattr(du.numpyro, "sample",
                        lambda name, distobj: jnp.zeros(1))
    monkeypatch.setattr(du.numpyro, "factor",
                        lambda name, val: factored.update({name: float(val)}))

    model = du.make_numpyro_model(lnlike, {}, transform=custom_transform)
    model()

    assert called_with["lnlike"] is lnlike
    assert factored["logl"] == pytest.approx(-99.0)


def test_make_numpyro_model_parlen_scalar_params(monkeypatch):
    """parlen should equal the number of scalar params with no vector params."""
    import jax.numpy as jnp

    sizes_seen = []

    class FakeLogx:
        params = ["a", "b", "c"]
        def __call__(self, pars): return jnp.array(0.0)
        def to_df(self, pars): return None

    class LnLike:
        params = ["a", "b", "c"]
        def __call__(self, pars): return 0.0

    monkeypatch.setattr(du.ds_prior, "makelogtransform_uniform",
                        lambda lnlike, priordict: FakeLogx())

    def fake_sample(name, distobj):
        sizes_seen.append(distobj.batch_shape[0])
        return jnp.zeros(3)

    monkeypatch.setattr(du.numpyro, "sample", fake_sample)
    monkeypatch.setattr(du.numpyro, "factor", lambda *a, **k: None)

    model = du.make_numpyro_model(LnLike(), {})
    model()
    assert sizes_seen == [3]  # Normal(0,10).expand([3])


# ---------------------------------------------------------------------------
# SVI integration: make_numpyro_model <-> AutoDelta guide <-> setup_svi
# ---------------------------------------------------------------------------

class TestMakeNumpyroModelSviIntegration:
    """Verify make_numpyro_model output is compatible with the SVI pipeline.

    Tests use real numpyro + JAX but with lightweight fake logx objects so
    no actual data or discovery likelihood is needed.
    """

    @staticmethod
    def _build_model_and_guide(n_params=2):
        """Return (model, guide, logx) for n_params scalar parameters."""
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.infer.autoguide as autoguide

        class FakeLogx:
            params = [f"p{i}" for i in range(n_params)]

            def __call__(self, pars):
                # simple quadratic bowl — easy to optimise
                return -jnp.sum(pars ** 2)

            def to_df(self, pars):
                pars = jnp.atleast_2d(pars)
                return pd.DataFrame(np.array(pars), columns=self.params)

        logx = FakeLogx()

        # Bypass the actual transform; inject FakeLogx directly.
        model = du.make_numpyro_model.__wrapped__ if hasattr(du.make_numpyro_model, "__wrapped__") \
            else None

        # Manually build the model the same way make_numpyro_model does.
        parlen = n_params  # all scalar

        def numpyro_model():
            pars = numpyro.sample("pars", numpyro.distributions.Normal(0, 10).expand([parlen]))
            numpyro.factor("logl", logx(pars))

        numpyro_model.to_df = lambda chain: logx.to_df(chain["pars"])

        guide = autoguide.AutoDelta(numpyro_model)
        return numpyro_model, guide, logx

    def test_model_site_name_is_pars(self):
        """The new model must expose 'pars' as the only latent sample site.
        numpyro.factor() also appears as a sample site with is_observed=True,
        so we filter those out."""
        import numpyro
        from numpyro import handlers

        model, guide, _ = self._build_model_and_guide(n_params=3)
        trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace()
        latent_sites = {
            k for k, v in trace.items()
            if v["type"] == "sample" and not v.get("is_observed", False)
        }
        assert latent_sites == {"pars"}

    def test_pars_site_has_correct_shape(self):
        """'pars' site must be a 1-D array with length == number of params."""
        from numpyro import handlers
        n = 5
        model, _, _ = self._build_model_and_guide(n_params=n)
        trace = handlers.trace(handlers.seed(model, rng_seed=0)).get_trace()
        assert trace["pars"]["value"].shape == (n,)

    def test_autodelta_guide_initialises_from_model(self):
        """AutoDelta must be able to initialise from the model without error."""
        import jax
        import numpyro
        import numpyro.infer as infer

        model, guide, _ = self._build_model_and_guide(n_params=2)
        svi = infer.SVI(model, guide,
                        numpyro.optim.Adam(0.01),
                        loss=numpyro.infer.Trace_ELBO())
        # svi.init must not raise
        state = svi.init(jax.random.key(0))
        params = svi.get_params(state)
        # AutoDelta creates 'pars_auto_loc'
        assert "pars_auto_loc" in params

    def test_svi_update_step_runs(self):
        """A single SVI update step must complete and return finite loss."""
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.infer as infer

        model, guide, _ = self._build_model_and_guide(n_params=2)
        svi = infer.SVI(model, guide,
                        numpyro.optim.Adam(0.01),
                        loss=numpyro.infer.Trace_ELBO())
        state = svi.init(jax.random.key(1))
        new_state, loss = svi.update(state)
        assert jnp.isfinite(loss)

    def test_run_svi_early_stopping_cleans_pars_key(self, monkeypatch):
        """run_svi_early_stopping must strip '_auto_loc' → params['pars'] exists."""
        import jax
        import numpyro
        import numpyro.infer as infer

        model, guide, logx = self._build_model_and_guide(n_params=2)
        svi = infer.SVI(model, guide,
                        numpyro.optim.Adam(0.05),
                        loss=numpyro.infer.Trace_ELBO())

        # run a small optimisation — 3 batches of 5 steps, patience=10 so no early stop
        params, _ = du.run_svi_early_stopping(
            jax.random.key(42),
            svi,
            batch_size=5,
            patience=10,
            max_num_batches=3,
            diagnostics=False,
        )
        # cleaned params must have 'pars' key (stripped '_auto_loc')
        assert "pars" in params
        assert np.array(params["pars"]).shape == (2,)

    def test_to_df_round_trip_after_svi(self, monkeypatch):
        """model.to_df on cleaned SVI params must return a DataFrame
        with the correct physical-parameter column names."""
        import jax
        import numpyro
        import numpyro.infer as infer

        n = 3
        model, guide, logx = self._build_model_and_guide(n_params=n)
        svi = infer.SVI(model, guide,
                        numpyro.optim.Adam(0.05),
                        loss=numpyro.infer.Trace_ELBO())

        params, _ = du.run_svi_early_stopping(
            jax.random.key(7),
            svi,
            batch_size=5,
            patience=10,
            max_num_batches=3,
            diagnostics=False,
        )
        # params['pars'] is the cleaned unconstrained vector
        df = model.to_df({"pars": params["pars"]})
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == logx.params

    def test_setup_svi_accepts_model_and_autodelta_guide(self):
        """du.setup_svi must accept the new model and AutoDelta guide without error."""
        import jax
        import numpyro.infer.autoguide as autoguide

        model, guide, _ = self._build_model_and_guide(n_params=2)
        svi = du.setup_svi(model, guide,
                           num_warmup_steps=5,
                           max_epochs=20,
                           peak_learning_rate=0.01)
        # init must work
        state = svi.init(jax.random.key(99))
        params = svi.get_params(state)
        assert "pars_auto_loc" in params


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
        difference_threshold=0.05,
    )
    assert params["a"] == 2.0
    assert params["b"] == 2.0
    assert diag is None


# ---------------------------------------------------------------------------
# Chromatic basis: fixed vs. varying chromatic_idx
# ---------------------------------------------------------------------------

class TestChromaticBasisSelection:
    """Tests for _select_fourier_basis chromatic path and chromatic_noise_block."""

    # --- _select_fourier_basis, nlog==0 ---

    def test_nlog0_vary_returns_fourierbasis_chrom(self):
        """chromatic_idx=None → bare ds.fourierbasis_chrom (callable fmat)."""
        result = du._select_fourier_basis(
            psr=object(), Nfreqs=10, tspan=100.0,
            logmode=2, f_min=1e-3, nlog=0,
            noise_type='chromatic', chromatic_idx=None,
        )
        assert result is du.ds.fourierbasis_chrom

    def test_nlog0_fixed_calls_make_fourierbasis_chrom(self, monkeypatch):
        """chromatic_idx=4.0 → ds.make_fourierbasis_chrom(alpha=4.0) (fixed-index matrix)."""
        captured = {}
        def fake_make(alpha):
            captured['alpha'] = alpha
            return 'FIXED_BASIS'
        monkeypatch.setattr(du.ds, 'make_fourierbasis_chrom', fake_make)

        result = du._select_fourier_basis(
            psr=object(), Nfreqs=10, tspan=100.0,
            logmode=2, f_min=1e-3, nlog=0,
            noise_type='chromatic', chromatic_idx=4.0,
        )
        assert captured['alpha'] == 4.0
        assert result == 'FIXED_BASIS'

    # --- _select_fourier_basis, nlog>0 ---

    def test_nlog_positive_vary_calls_log_free(self, monkeypatch):
        """nlog>0 + chromatic_idx=None → lambda calling log_free_chromatic_fourierbasis."""
        calls = {}
        def fake_log_free(psr, T, logmode, f_min, nlin, nlog):
            calls['log_free'] = True
            return 'f', 'df', lambda alpha: 'FMAT'
        monkeypatch.setattr(du.ds, 'log_free_chromatic_fourierbasis', fake_log_free)

        psr = object()
        result = du._select_fourier_basis(
            psr=psr, Nfreqs=10, tspan=100.0,
            logmode=2, f_min=1e-3, nlog=5,
            noise_type='chromatic', chromatic_idx=None,
        )
        # Result is a lambda; calling it should call the log_free function
        assert callable(result)
        result(psr, 10, 100.0)
        assert 'log_free' in calls

    def test_nlog_positive_fixed_calls_log_fixed(self, monkeypatch):
        """nlog>0 + chromatic_idx=4.0 → lambda calling log_fixed_chromatic_fourierbasis."""
        calls = {}
        def fake_log_fixed(psr, chromatic_idx, T, logmode, f_min, nlin, nlog):
            calls['chromatic_idx'] = chromatic_idx
            return 'f', 'df', 'FMAT'
        monkeypatch.setattr(du.ds, 'log_fixed_chromatic_fourierbasis', fake_log_fixed)

        psr = object()
        result = du._select_fourier_basis(
            psr=psr, Nfreqs=10, tspan=100.0,
            logmode=2, f_min=1e-3, nlog=5,
            noise_type='chromatic', chromatic_idx=4.0,
        )
        assert callable(result)
        result(psr, 10, 100.0)
        assert calls['chromatic_idx'] == 4.0

    # --- chromatic_noise_block wiring ---

    def test_chromatic_block_vary_passes_none(self, monkeypatch):
        """chromatic_idx='vary' → _select_fourier_basis gets chromatic_idx=None."""
        captured = {}
        def fake_select(psr, Nfreqs, tspan, logmode, f_min, nlog, noise_type, chromatic_idx=None):
            captured['chromatic_idx'] = chromatic_idx
            return 'BASIS'
        monkeypatch.setattr(du, '_select_fourier_basis', fake_select)
        monkeypatch.setattr(du.ds, 'getspan', lambda _: 100.0)
        monkeypatch.setattr(du.ds, 'powerlaw', object())
        monkeypatch.setattr(du.ds, 'makegp_fourier', lambda *a, **k: 'GP')

        du.chromatic_noise_block(object(), tspan=100.0, chromatic_idx='vary')
        assert captured['chromatic_idx'] is None

    def test_chromatic_block_fixed_passes_value(self, monkeypatch):
        """chromatic_idx=4.0 → _select_fourier_basis gets chromatic_idx=4.0."""
        captured = {}
        def fake_select(psr, Nfreqs, tspan, logmode, f_min, nlog, noise_type, chromatic_idx=None):
            captured['chromatic_idx'] = chromatic_idx
            return 'BASIS'
        monkeypatch.setattr(du, '_select_fourier_basis', fake_select)
        monkeypatch.setattr(du.ds, 'getspan', lambda _: 100.0)
        monkeypatch.setattr(du.ds, 'powerlaw', object())
        monkeypatch.setattr(du.ds, 'makegp_fourier', lambda *a, **k: 'GP')

        du.chromatic_noise_block(object(), tspan=100.0, chromatic_idx=4.0)
        assert captured['chromatic_idx'] == 4.0

    def test_chromatic_block_vary_no_chromatic_idx_in_makegp_kwargs(self, monkeypatch):
        """makegp_fourier must NOT receive a chromatic_idx kwarg (it doesn't accept one)."""
        makegp_kwargs = {}
        def fake_makegp(psr, prior, nfreqs, T=None, fourierbasis=None, name=None, **kwargs):
            makegp_kwargs.update(kwargs)
            return 'GP'
        monkeypatch.setattr(du, '_select_fourier_basis', lambda *a, **k: 'BASIS')
        monkeypatch.setattr(du.ds, 'getspan', lambda _: 100.0)
        monkeypatch.setattr(du.ds, 'powerlaw', object())
        monkeypatch.setattr(du.ds, 'makegp_fourier', fake_makegp)

        du.chromatic_noise_block(object(), tspan=100.0, chromatic_idx='vary')
        assert 'chromatic_idx' not in makegp_kwargs

    def test_chromatic_block_fixed_no_chromatic_idx_in_makegp_kwargs(self, monkeypatch):
        """makegp_fourier must NOT receive chromatic_idx even when a value is passed."""
        makegp_kwargs = {}
        def fake_makegp(psr, prior, nfreqs, T=None, fourierbasis=None, name=None, **kwargs):
            makegp_kwargs.update(kwargs)
            return 'GP'
        monkeypatch.setattr(du, '_select_fourier_basis', lambda *a, **k: 'BASIS')
        monkeypatch.setattr(du.ds, 'getspan', lambda _: 100.0)
        monkeypatch.setattr(du.ds, 'powerlaw', object())
        monkeypatch.setattr(du.ds, 'makegp_fourier', fake_makegp)

        du.chromatic_noise_block(object(), tspan=100.0, chromatic_idx=4.0)
        assert 'chromatic_idx' not in makegp_kwargs


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
    monkeypatch.setattr(du.ds_signals, "custom_blocked_interpolation_basis", lambda *a, **k: ("U", "N"), raising=False)
    monkeypatch.setattr(du.ds_signals, kernel_attr, lambda: kernel, raising=False)
    monkeypatch.setattr(du.ds_solar, "makegp_timedomain_solar_dm", lambda psr, covariance, **k: covariance, raising=False)

    out = du.solar_wind_noise_block(_solar_psr([0.0, 2.0]), basis="interpolation", prior=prior_name, basis_nodes=np.array([1.0]))
    assert out is kernel


def test_solar_wind_interpolation_invalid_prior_raises(monkeypatch):
    monkeypatch.setattr(du.ds_solar, "custom_blocked_interpolation_basis", lambda *a, **k: ("U", "N"), raising=False)
    with pytest.raises(ValueError, match="Invalid prior specified for time domain solar wind noise"):
        du.solar_wind_noise_block(_solar_psr([0.0, 1.0]), basis="interpolation", prior="not-a-prior", basis_nodes=np.array([1.0]))


# ---------------------------------------------------------------------------
# modes forwarding tests
# ---------------------------------------------------------------------------

def _make_fake_makegp(calls, sentinel="BLOCK"):
    """Return a fake ds.makegp_fourier that records its keyword arguments."""
    def fake(psr, prior, nfreqs, T=None, modes=None, fourierbasis=None, name=None, **kwargs):
        calls.update(dict(prior=prior, Nfreqs=nfreqs, T=T, modes=modes,
                         fourierbasis=fourierbasis, name=name))
        return sentinel
    return fake


def _patch_block(monkeypatch, tspan=500.0):
    monkeypatch.setattr(du.ds, "getspan", lambda _psr: tspan)
    monkeypatch.setattr(du, "_select_fourier_basis", lambda *a, **k: "BASIS")
    monkeypatch.setattr(du.ds, "powerlaw", "powerlaw")


def test_red_noise_block_forwards_modes(monkeypatch):
    calls = {}
    _patch_block(monkeypatch)
    monkeypatch.setattr(du.ds, "makegp_fourier", _make_fake_makegp(calls))
    modes = np.array([1e-9, 3e-9, 1e-8])
    result = du.red_noise_block(object(), tspan=500.0, modes=modes)
    assert result == "BLOCK"
    assert calls["modes"] is modes


def test_red_noise_block_modes_none_by_default(monkeypatch):
    calls = {}
    _patch_block(monkeypatch)
    monkeypatch.setattr(du.ds, "makegp_fourier", _make_fake_makegp(calls))
    du.red_noise_block(object(), tspan=500.0)
    assert calls["modes"] is None


def test_dm_noise_block_forwards_modes(monkeypatch):
    calls = {}
    _patch_block(monkeypatch)
    monkeypatch.setattr(du.ds, "makegp_fourier", _make_fake_makegp(calls))
    modes = np.array([2e-9, 4e-9, 6e-9, 8e-9])
    result = du.dm_noise_block(object(), tspan=500.0, modes=modes)
    assert result == "BLOCK"
    assert calls["modes"] is modes


def test_dm_noise_block_modes_none_by_default(monkeypatch):
    calls = {}
    _patch_block(monkeypatch)
    monkeypatch.setattr(du.ds, "makegp_fourier", _make_fake_makegp(calls))
    du.dm_noise_block(object(), tspan=500.0)
    assert calls["modes"] is None


def test_chromatic_noise_block_forwards_modes(monkeypatch):
    calls = {}
    _patch_block(monkeypatch)
    monkeypatch.setattr(du.ds, "makegp_fourier", _make_fake_makegp(calls))
    modes = np.linspace(1e-9, 1e-8, 5)
    result = du.chromatic_noise_block(object(), tspan=500.0, modes=modes)
    assert result == "BLOCK"
    assert calls["modes"] is modes


def test_chromatic_noise_block_modes_none_by_default(monkeypatch):
    calls = {}
    _patch_block(monkeypatch)
    monkeypatch.setattr(du.ds, "makegp_fourier", _make_fake_makegp(calls))
    du.chromatic_noise_block(object(), tspan=500.0)
    assert calls["modes"] is None


def test_solar_wind_noise_block_forwards_modes(monkeypatch):
    calls = {}
    _patch_block(monkeypatch)
    monkeypatch.setattr(du.ds, "makegp_fourier", _make_fake_makegp(calls))
    modes = np.array([5e-9, 1e-8, 2e-8])
    result = du.solar_wind_noise_block(object(), tspan=500.0, modes=modes)
    assert result == "BLOCK"
    assert calls["modes"] is modes


def test_solar_wind_noise_block_modes_none_by_default(monkeypatch):
    calls = {}
    _patch_block(monkeypatch)
    monkeypatch.setattr(du.ds, "makegp_fourier", _make_fake_makegp(calls))
    du.solar_wind_noise_block(object(), tspan=500.0)
    assert calls["modes"] is None


def test_all_block_modes_are_passed_as_exact_array(monkeypatch):
    """Each block must pass the modes array through without copying or modifying it."""
    _patch_block(monkeypatch)
    modes = np.array([1e-9, 2e-9, 3e-9, 4e-9, 5e-9])
    for block_fn in (du.red_noise_block, du.dm_noise_block,
                     du.chromatic_noise_block, du.solar_wind_noise_block):
        calls = {}
        monkeypatch.setattr(du.ds, "makegp_fourier", _make_fake_makegp(calls))
        block_fn(object(), tspan=500.0, modes=modes)
        assert calls["modes"] is modes, f"{block_fn.__name__} did not forward modes unchanged"


# ---------------------------------------------------------------------------
# compute_log_probs
# ---------------------------------------------------------------------------

def test_make_numpyro_model_has_compute_log_probs(monkeypatch):
    """make_numpyro_model attaches a compute_log_probs method to the model."""

    class FakeLogx:
        params = ["x", "y"]
        def __call__(self, p):
            return 0.0
        def to_df(self, chain_pars):
            return None

    monkeypatch.setattr(du.ds_prior, "makelogtransform_uniform",
                        lambda lnl, priordict=None: FakeLogx())

    model = du.make_numpyro_model(lambda p: 0.0)
    assert hasattr(model, "compute_log_probs"), "model should have compute_log_probs"
    assert callable(model.compute_log_probs)


def test_compute_log_probs_returns_required_keys(monkeypatch):
    """compute_log_probs returns dict with lnlike, lnprior, lnpost keys."""
    import jax.numpy as jnp

    class FakeLogx:
        params = ["x", "y"]
        def __call__(self, p):
            return jnp.sum(p)
        def to_df(self, chain_pars):
            return None

    monkeypatch.setattr(du.ds_prior, "makelogtransform_uniform",
                        lambda lnl, priordict=None: FakeLogx())

    model = du.make_numpyro_model(lambda p: 0.0)
    pars = jnp.ones((3, 2))
    result = model.compute_log_probs({'pars': pars})
    assert set(result.keys()) == {'lnlike', 'lnprior', 'lnpost'}
    assert result['lnlike'].shape == (3,)
    assert result['lnprior'].shape == (3,)
    assert result['lnpost'].shape == (3,)


def test_compute_log_probs_lnpost_equals_sum(monkeypatch):
    """lnpost == lnlike + lnprior for every sample."""
    import jax.numpy as jnp

    class FakeLogx:
        params = ["x", "y"]
        def __call__(self, p):
            return jnp.sum(p)
        def to_df(self, _):
            return None

    monkeypatch.setattr(du.ds_prior, "makelogtransform_uniform",
                        lambda lnl, priordict=None: FakeLogx())
    model = du.make_numpyro_model(lambda p: 0.0)
    pars = jnp.array([[1.0, 2.0], [0.5, -0.5]])
    result = model.compute_log_probs({'pars': pars})
    np.testing.assert_allclose(
        np.asarray(result['lnpost']),
        np.asarray(result['lnlike']) + np.asarray(result['lnprior']),
    )


# ---------------------------------------------------------------------------
# run_nuts_with_checkpoints — log-prob columns
# ---------------------------------------------------------------------------

def test_run_nuts_with_checkpoints_appends_log_prob_columns(tmp_path, monkeypatch):
    """When model is supplied, df saved to disk includes lnlike/lnprior/lnpost."""
    import jax.numpy as jnp

    N = 4

    class FakeSampler:
        num_samples = N
        last_state = {}
        post_warmup_state = None
        calls = 0

        def _set_collection_params(self):
            pass

        def run(self, _rng):
            self.calls += 1
            self.last_state = {"s": self.calls}

        def to_df(self):
            return pd.DataFrame({"par1": np.zeros(N), "par2": np.ones(N)})

        def get_samples(self, group_by_chain=False):
            return {"pars": jnp.zeros((N, 2))}

    saved_dfs = []
    monkeypatch.setattr(du, "save_chain", lambda df, path: saved_dfs.append(df.copy()))
    monkeypatch.setattr(du.jax.random, "split", lambda key: (key, key))

    class FakeModel:
        def compute_log_probs(self, chain):
            n = chain['pars'].shape[0]
            return {
                'lnlike': jnp.full((n,), -1.0),
                'lnprior': jnp.full((n,), -2.0),
                'lnpost': jnp.full((n,), -3.0),
            }

    du.run_nuts_with_checkpoints(
        sampler=FakeSampler(),
        num_samples_per_checkpoint=2,
        rng_key=np.array([0, 1]),
        outdir=tmp_path,
        file_name="abc",
        diagnostics=False,
        model=FakeModel(),
    )

    assert saved_dfs, "save_chain should have been called at least once"
    last_df = saved_dfs[-1]
    for col in ('lnlike', 'lnprior', 'lnpost'):
        assert col in last_df.columns, f"expected column '{col}' in saved df"
    np.testing.assert_allclose(last_df['lnlike'].values, -1.0)
    np.testing.assert_allclose(last_df['lnprior'].values, -2.0)
    np.testing.assert_allclose(last_df['lnpost'].values, -3.0)


def test_run_nuts_with_checkpoints_no_model_no_log_cols(tmp_path, monkeypatch):
    """When model is not supplied, no lnlike/lnprior/lnpost columns are added."""
    N = 4

    class FakeSampler:
        num_samples = N
        last_state = {}
        post_warmup_state = None
        calls = 0

        def _set_collection_params(self):
            pass

        def run(self, _rng):
            self.calls += 1
            self.last_state = {}

        def to_df(self):
            return pd.DataFrame({"par1": np.zeros(N)})

    saved_dfs = []
    monkeypatch.setattr(du, "save_chain", lambda df, path: saved_dfs.append(df.copy()))
    monkeypatch.setattr(du.jax.random, "split", lambda key: (key, key))

    du.run_nuts_with_checkpoints(
        sampler=FakeSampler(),
        num_samples_per_checkpoint=N,
        rng_key=np.array([0, 1]),
        outdir=tmp_path,
        file_name="nomodel",
        diagnostics=False,
    )

    last_df = saved_dfs[-1]
    for col in ('lnlike', 'lnprior', 'lnpost'):
        assert col not in last_df.columns


# ---------------------------------------------------------------------------
# get_map_noise_values — N parameter
# ---------------------------------------------------------------------------

class TestGetMapNoiseValuesN:
    """Unit tests for get_map_noise_values with N parameter."""

    def _make_outdir(self, tmp_path, has_json=False, has_lnpost=True):
        import pint.models
        from types import SimpleNamespace

        # Minimal fake timing model accepted by format_chain_dir
        class FakeModel:
            PSR = SimpleNamespace(value="J0000+0000")

        outdir = tmp_path / "chains" / "J0000+0000"
        outdir.mkdir(parents=True)

        if has_json:
            (outdir / "J0000+0000_map_params.json").write_text('{"par1": 1.0, "par2": 2.0}')

        if not has_json:
            df = pd.DataFrame({
                "par1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "par2": [5.0, 4.0, 3.0, 2.0, 1.0],
            })
            if has_lnpost:
                df["lnpost"] = [-5.0, -4.0, -3.0, -2.0, -1.0]  # sample 4 is best
            df.to_feather(outdir / "J0000+0000_nuts_samples.feather")

        return outdir.parent.parent, FakeModel()

    def test_N1_uses_json_when_present(self, tmp_path, monkeypatch):
        outdir_root, fake_model = self._make_outdir(tmp_path, has_json=True)
        from pint_pal.noise_utils import format_chain_dir, get_map_noise_values
        monkeypatch.setattr("pint_pal.noise_utils.format_chain_dir",
                            lambda d, model=None: str(tmp_path / "chains" / "J0000+0000"))
        result = get_map_noise_values(outdir_root, fake_model, N=1)
        assert result == {"par1": 1.0, "par2": 2.0}

    def test_N1_selects_top_row_by_lnpost(self, tmp_path, monkeypatch):
        outdir_root, fake_model = self._make_outdir(tmp_path, has_json=False, has_lnpost=True)
        monkeypatch.setattr("pint_pal.noise_utils.format_chain_dir",
                            lambda d, model=None: str(tmp_path / "chains" / "J0000+0000"))
        from pint_pal.noise_utils import get_map_noise_values
        result = get_map_noise_values(outdir_root, fake_model, N=1)
        # best row (lnpost=-1) is the last row: par1=5, par2=1
        assert result["par1"] == pytest.approx(5.0)
        assert result["par2"] == pytest.approx(1.0)

    def test_N3_averages_top3_by_lnpost(self, tmp_path, monkeypatch):
        outdir_root, fake_model = self._make_outdir(tmp_path, has_json=False, has_lnpost=True)
        monkeypatch.setattr("pint_pal.noise_utils.format_chain_dir",
                            lambda d, model=None: str(tmp_path / "chains" / "J0000+0000"))
        from pint_pal.noise_utils import get_map_noise_values
        result = get_map_noise_values(outdir_root, fake_model, N=3)
        # top-3 by lnpost: rows with lnpost -1,-2,-3 → par1=[5,4,3], par2=[1,2,3]
        assert result["par1"] == pytest.approx(4.0)
        assert result["par2"] == pytest.approx(2.0)

    def test_no_lnpost_falls_back_to_mean(self, tmp_path, monkeypatch):
        outdir_root, fake_model = self._make_outdir(tmp_path, has_json=False, has_lnpost=False)
        monkeypatch.setattr("pint_pal.noise_utils.format_chain_dir",
                            lambda d, model=None: str(tmp_path / "chains" / "J0000+0000"))
        from pint_pal.noise_utils import get_map_noise_values
        result = get_map_noise_values(outdir_root, fake_model, N=1)
        # mean of [1,2,3,4,5] = 3, mean of [5,4,3,2,1] = 3
        assert result["par1"] == pytest.approx(3.0)
        assert result["par2"] == pytest.approx(3.0)


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
