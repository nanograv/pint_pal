"""PR-focused unit tests for noise utility helpers.

This module keeps mocking local and explicit while exercising core control
flow, config wiring, and error handling branches.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pint_pal.noise_utils as nu
from pint import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class _DummyPSR:
    value = "J1234+5678"


class _DummyModel:
    PSR = _DummyPSR()


@pytest.fixture
def model():
    parent = Path(__file__).parent
    parfile = parent / "par/B1855+09_NANOGrav_12yv4.gls.par"
    return models.get_model(parfile)



@pytest.mark.filterwarnings("ignore:PINT only supports 'T2CMETHOD IAU2000B'")
def test_format_chain_dir(model):
    assert nu.format_chain_dir("/path/to/chains/B1855+09_nb/", model) == "/path/to/chains/B1855+09_nb/"
    assert nu.format_chain_dir("/path/to/chains/", model) == "/path/to/chains/B1855+09_nb/"
    assert nu.format_chain_dir("/path/to/chains/", model, using_wideband=True) == "/path/to/chains/B1855+09_wb/"


def _formatted_dir(root: str) -> str:
    return nu.format_chain_dir(root, model=_DummyModel())


def _create_chain_outpath(tmp_path: Path) -> Path:
    formatted = _formatted_dir(str(tmp_path))
    outpath = tmp_path / formatted.split(str(tmp_path).rstrip("/") + "/")[-1]
    outpath.mkdir(parents=True, exist_ok=True)
    return outpath


def test_get_map_noise_values_invalid_outdir_raises(tmp_path):
    with pytest.raises(ValueError, match="Invalid outdir"):
        nu.get_map_noise_values(str(tmp_path / "does_not_exist"), _DummyModel())


def test_get_map_noise_values_prefers_map_json(tmp_path):
    outpath = _create_chain_outpath(tmp_path)

    map_file = outpath / "test_map_params.json"
    map_file.write_text(json.dumps({"a": 1, "b": 2.5}), encoding="utf-8")

    result = nu.get_map_noise_values(str(tmp_path), _DummyModel())

    assert result == {"a": 1.0, "b": 2.5}


def test_get_map_noise_values_falls_back_to_feather_numeric_means(tmp_path, monkeypatch):
    outpath = _create_chain_outpath(tmp_path)

    (outpath / "chain.feather").touch()

    fake_df = pd.DataFrame({"x": [1.0, 3.0], "y": [2, 6], "label": ["a", "b"]})
    monkeypatch.setattr(nu.pd, "read_feather", lambda _path: fake_df)

    result = nu.get_map_noise_values(str(tmp_path), _DummyModel())

    assert result == {"x": 2.0, "y": 4.0}


def test_get_map_noise_values_raises_if_no_map_or_feather(tmp_path):
    outpath = _create_chain_outpath(tmp_path)

    with pytest.raises(FileNotFoundError, match=r"No \*_map_dict.json, \*_map_params.json, or \*\.feather"):
        nu.get_map_noise_values(str(tmp_path), _DummyModel())


def test_get_map_noise_values_raises_for_empty_feather(tmp_path, monkeypatch):
    outpath = _create_chain_outpath(tmp_path)
    (outpath / "empty.feather").touch()

    monkeypatch.setattr(nu.pd, "read_feather", lambda _path: pd.DataFrame())

    with pytest.raises(ValueError, match="Feather chain is empty"):
        nu.get_map_noise_values(str(tmp_path), _DummyModel())


def test_get_map_noise_values_raises_for_non_numeric_feather(tmp_path, monkeypatch):
    outpath = _create_chain_outpath(tmp_path)
    (outpath / "nonnumeric.feather").touch()

    monkeypatch.setattr(nu.pd, "read_feather", lambda _path: pd.DataFrame({"name": ["a", "b"]}))

    with pytest.raises(ValueError, match="No numeric columns found"):
        nu.get_map_noise_values(str(tmp_path), _DummyModel())


def test_get_mean_large_likelihoods_top_n_average():
    core = SimpleNamespace(
        chain=np.array([[0.0, 1.0], [2.0, 3.0], [1.0, 5.0]]),
        burn=0,
        params=["x", "lnlike"],
    )
    out = nu.get_mean_large_likelihoods(core, N=2)
    assert out["x"] == pytest.approx((1.0 + 2.0) / 2.0)


def test_analyze_enterprise_noise_invalid_path_raises(monkeypatch):
    monkeypatch.setattr(nu.co, "Core", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad")))
    monkeypatch.setattr(nu.os.path, "isfile", lambda p: False)
    with pytest.raises(ValueError, match="Could not load noise run"):
        nu.analyze_enterprise_noise(chaindir="/nope", save_corner=False, no_corner_plot=False)


def test_analyze_enterprise_noise_happy_path_map(monkeypatch):
    class FakeCore:
        def __init__(self, chaindir=None):
            self.chain = np.array([[1.0, 2.0, 0, 0, 0, 0], [2.0, 3.0, 0, 0, 0, 0]])
            self.params = ["J1234_red_noise_log10_A", "lnlike", "lnpost", "chain_accept", "pt_chain_accept"]
            self.burn = 0
            self.map_idx = 0

        def set_burn(self, b):
            self.burn = b

        def get_map_dict(self):
            return {"a": 1.0}

        def get_param_median(self, p):
            return 1.23

        def __call__(self, name):
            return np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(nu.co, "Core", FakeCore)
    monkeypatch.setattr(nu, "get_model_and_sampler_default_settings", lambda: ({}, {"sampler": "PTMCMCSampler", "likelihood": "enterprise"}))
    monkeypatch.setattr(nu.model_utils, "bayes_fac", lambda arr, **k: [42.0])

    core, noise_dict, rn_bf = nu.analyze_enterprise_noise(
        chaindir="/tmp/x",
        save_corner=False,
        no_corner_plot=False,
        use_noise_point="MAP",
    )
    assert isinstance(core, FakeCore)
    assert noise_dict == {"a": 1.0}
    assert rn_bf == 42.0


def test_model_noise_quick_exit_and_invalid_combo(monkeypatch, tmp_path):
    mo = SimpleNamespace(PSR=SimpleNamespace(value="J1"), NTOA=SimpleNamespace(value=10), params=[])
    to = object()
    outdir = str(tmp_path / "J1_nb")
    monkeypatch.setattr(nu, "format_chain_dir", lambda *a, **k: outdir)
    monkeypatch.setattr(nu.os.path, "exists", lambda p: True)

    assert nu.model_noise(mo, to, run_noise_analysis=True, resume=False, sampler_kwargs={"likelihood": "enterprise", "sampler": "PTMCMCSampler"}) is None

    monkeypatch.setattr(nu.os.path, "exists", lambda p: False)
    monkeypatch.setattr(nu, "Pulsar", lambda *a, **k: SimpleNamespace(name="J1"))
    assert nu.model_noise(mo, to, sampler_kwargs={"likelihood": "bad", "sampler": "bad"}, model_kwargs={}) is None


def test_model_noise_discovery_nuts_returns_sampler(monkeypatch, tmp_path):
    mo = SimpleNamespace(PSR=SimpleNamespace(value="J1"), NTOA=SimpleNamespace(value=10), params=[])
    to = object()
    monkeypatch.setattr(nu, "format_chain_dir", lambda *a, **k: str(tmp_path / "J1_nb"))
    monkeypatch.setattr(nu.os.path, "exists", lambda p: False)
    monkeypatch.setattr(nu, "Pulsar", lambda *a, **k: SimpleNamespace(name="J1"))
    monkeypatch.setattr(nu.disco_utils, "make_single_pulsar_noise_likelihood_discovery", lambda **k: SimpleNamespace(logL=SimpleNamespace(params=[])))
    monkeypatch.setattr(nu.disco_utils, "make_numpyro_model", lambda *a, **k: "model")
    monkeypatch.setattr(nu.disco_utils, "make_sampler_nuts", lambda *a, **k: "sampler")

    out = nu.model_noise(
        mo,
        to,
        model_kwargs={},
        sampler_kwargs={"likelihood": "discovery", "sampler": "NUTS"},
        return_sampler_without_sampling=True,
    )
    assert out == "sampler"


def test_model_noise_uses_config_noise_run_kwargs_for_discovery(monkeypatch, tmp_path):
    calls = {}

    mo = SimpleNamespace(PSR=SimpleNamespace(value="J1"), NTOA=SimpleNamespace(value=10), params=[])
    to = object()

    config = {
        "noise_run": {
            "model": {
                "timing_model": {"svd": True, "tm_marg": False},
                "white_noise": {"gp_ecorr": False, "tn_equad": True, "include_ecorr": True},
                "red_noise": {"basis": "fourier", "Nfreqs": 30, "prior": "powerlaw"},
                "dm_noise": False,
                "chromatic_noise": False,
                "solar_wind": False,
            },
            "inference": {
                "likelihood": "discovery",
                "sampler": "NUTS",
                "num_samples": 7,
                "num_warmup": 3,
            },
        }
    }

    monkeypatch.setattr(nu, "format_chain_dir", lambda *a, **k: str(tmp_path / "J1_nb"))
    monkeypatch.setattr(nu.os.path, "exists", lambda p: False)
    monkeypatch.setattr(nu, "Pulsar", lambda *a, **k: SimpleNamespace(name="J1"))

    def fake_make_likelihood(psr, noise_dict, tspan, model_kwargs, return_args):
        calls["model_kwargs"] = model_kwargs
        return SimpleNamespace(logL=SimpleNamespace(params=[]))

    monkeypatch.setattr(nu.disco_utils, "make_single_pulsar_noise_likelihood_discovery", fake_make_likelihood)
    monkeypatch.setattr(nu.disco_utils, "make_numpyro_model", lambda *a, **k: "logL")

    def fake_make_sampler(logL, sampler_kwargs):
        calls["sampler_kwargs"] = sampler_kwargs
        return "sampler"

    monkeypatch.setattr(nu.disco_utils, "make_sampler_nuts", fake_make_sampler)

    out = nu.model_noise(
        mo,
        to,
        model_kwargs=config["noise_run"]["model"],
        sampler_kwargs=config["noise_run"]["inference"],
        return_sampler_without_sampling=True,
    )

    assert out == "sampler"
    assert calls["model_kwargs"] == config["noise_run"]["model"]
    assert calls["sampler_kwargs"] == config["noise_run"]["inference"]


def test_model_noise_optimizer_uses_config_inference_controls(monkeypatch, tmp_path):
    calls = {}
    mo = SimpleNamespace(PSR=SimpleNamespace(value="J1"), NTOA=SimpleNamespace(value=10), params=[])
    to = object()

    model_cfg = {
        "timing_model": {"svd": True, "tm_marg": False},
        "white_noise": {"gp_ecorr": False, "tn_equad": True, "include_ecorr": True},
        "red_noise": {"basis": "fourier", "Nfreqs": 30, "prior": "powerlaw"},
        "dm_noise": False,
        "chromatic_noise": False,
        "solar_wind": False,
    }
    inf_cfg = {
        "likelihood": "discovery",
        "sampler": "optimizer",
        "num_warmup_steps": 50,
        "max_epochs": 100,
        "peak_learning_rate": 0.02,
        "batch_size": 4,
        "patience": 2,
        "difference_threshold": 0.7,
        "max_num_batches": 3,
        "diagnostics": False,
    }

    monkeypatch.setattr(nu, "format_chain_dir", lambda *a, **k: str(tmp_path / "J1_nb"))
    monkeypatch.setattr(nu.os.path, "exists", lambda p: False)
    monkeypatch.setattr(nu.os, "makedirs", lambda path, exist_ok=True: Path(path).mkdir(parents=True, exist_ok=exist_ok))
    monkeypatch.setattr(nu, "Pulsar", lambda *a, **k: SimpleNamespace(name="J1"))
    monkeypatch.setattr(nu.disco_utils, "make_single_pulsar_noise_likelihood_discovery", lambda **k: SimpleNamespace(logL=SimpleNamespace(params=[])))
    monkeypatch.setattr(nu.disco_utils, "make_numpyro_model", lambda *a, **k: "logL")
    monkeypatch.setattr(nu.numpyro.handlers, "reparam", lambda logL, config=None: logL)
    monkeypatch.setattr(nu.numpyro.infer.autoguide, "AutoDelta", lambda logL: "guide")

    def fake_setup_svi(model, guide, loss, num_warmup_steps, max_epochs, peak_learning_rate, gradient_clipping_val):
        calls["setup_svi"] = {
            "num_warmup_steps": num_warmup_steps,
            "max_epochs": max_epochs,
            "peak_learning_rate": peak_learning_rate,
        }
        return "svi"

    monkeypatch.setattr(nu.disco_utils, "setup_svi", fake_setup_svi)

    def fake_run_svi_early_stopping(rng_key, svi, batch_size, patience, difference_threshold, max_num_batches, diagnostics, outdir, file_prefix):
        calls["run_svi"] = {
            "batch_size": batch_size,
            "patience": patience,
            "difference_threshold": difference_threshold,
            "max_num_batches": max_num_batches,
        }
        return ({"a": 1.0}, None)

    monkeypatch.setattr(nu.disco_utils, "run_svi_early_stopping", fake_run_svi_early_stopping)

    nu.model_noise(
        mo,
        to,
        model_kwargs=model_cfg,
        sampler_kwargs=inf_cfg,
        return_sampler_without_sampling=False,
    )

    assert calls["setup_svi"]["num_warmup_steps"] == inf_cfg["num_warmup_steps"]
    assert calls["setup_svi"]["max_epochs"] == inf_cfg["max_epochs"]
    assert calls["setup_svi"]["peak_learning_rate"] == inf_cfg["peak_learning_rate"]
    assert calls["run_svi"]["batch_size"] == inf_cfg["batch_size"]
    assert calls["run_svi"]["patience"] == inf_cfg["patience"]


def test_model_noise_discovery_nuts_uses_inference_values(monkeypatch, tmp_path):
    calls = {}
    mo = SimpleNamespace(PSR=SimpleNamespace(value="J1"), NTOA=SimpleNamespace(value=10), params=[])
    to = object()

    model_cfg = {
        "timing_model": {"svd": True, "tm_marg": False},
        "white_noise": {"gp_ecorr": False, "tn_equad": True, "include_ecorr": True},
        "red_noise": {"basis": "fourier", "Nfreqs": 20, "prior": "powerlaw"},
        "dm_noise": False,
        "chromatic_noise": False,
        "solar_wind": False,
    }
    inf_cfg = {
        "likelihood": "discovery",
        "sampler": "NUTS",
        "num_samples_per_checkpoint": 5,
        "diagnostics": False,
    }

    monkeypatch.setattr(nu, "format_chain_dir", lambda *a, **k: str(tmp_path / "J1_nb"))
    monkeypatch.setattr(nu.os.path, "exists", lambda p: False)
    monkeypatch.setattr(nu.os, "makedirs", lambda path, exist_ok=True: Path(path).mkdir(parents=True, exist_ok=exist_ok))
    monkeypatch.setattr(nu, "Pulsar", lambda *a, **k: SimpleNamespace(name="J1"))
    monkeypatch.setattr(nu.disco_utils, "make_single_pulsar_noise_likelihood_discovery", lambda **k: SimpleNamespace(logL=SimpleNamespace(params=[])))
    monkeypatch.setattr(nu.disco_utils, "make_numpyro_model", lambda *a, **k: "logL")
    monkeypatch.setattr(nu.disco_utils, "make_sampler_nuts", lambda *a, **k: "sampler")

    def fake_run_nuts_with_checkpoints(**kwargs):
        calls["run_nuts"] = kwargs

    monkeypatch.setattr(nu.disco_utils, "run_nuts_with_checkpoints", fake_run_nuts_with_checkpoints)

    nu.model_noise(
        mo,
        to,
        model_kwargs=model_cfg,
        sampler_kwargs=inf_cfg,
        return_sampler_without_sampling=False,
    )

    assert calls["run_nuts"]["num_samples_per_checkpoint"] == inf_cfg["num_samples_per_checkpoint"]
    assert calls["run_nuts"]["diagnostics"] is False


def test_convert_to_rnamp_positive():
    assert nu.convert_to_RNAMP(-20) > 0


def test_add_noise_to_model_minimal_white_noise(monkeypatch):
    class FakeComp:
        def __init__(self):
            self.params = []

        def remove_param(self, param=None):
            return None

        def add_param(self, param=None, setup=True):
            self.params.append(param)

    class FakeModel:
        def __init__(self):
            self.components = []

        def add_component(self, comp, validate=True, force=True):
            self.components.append(comp)

        def setup(self):
            return None

        def validate(self):
            return None

    monkeypatch.setattr(nu.pm, "ScaleToaError", FakeComp)
    monkeypatch.setattr(nu, "maskParameter", lambda **k: SimpleNamespace(**k))
    monkeypatch.setattr(nu, "test_equad_convention", lambda keys: None)

    model = FakeModel()
    noise_dict = {"J1234_backend_efac": 1.0}
    out = nu.add_noise_to_model(model, noise_dict, model_kwargs={})
    assert out is model
    assert len(model.components) == 1


def test_test_equad_convention_cases():
    assert nu.test_equad_convention(["a_t2equad"]) == "t2equad"
    assert nu.test_equad_convention(["a_tnequad"]) == "tnequad"
    assert nu.test_equad_convention(["none"]) is None


def test_get_init_sample_from_chain_path_and_json(monkeypatch, tmp_path):
    class Prior:
        def __init__(self, v):
            self.v = v

        def sample(self):
            return self.v

    pta = SimpleNamespace(params=[Prior(1.0), Prior(2.0)], param_names=["a", "b"])

    out = nu.get_init_sample_from_chain_path(pta)
    assert np.allclose(out, [1.0, 2.0])

    json_file = tmp_path / "map.json"
    json_file.write_text(json.dumps({"a": 5.0}), encoding="utf-8")
    out2 = nu.get_init_sample_from_chain_path(pta, json_path=str(json_file))
    assert np.allclose(out2, [5.0, 2.0])


def test_make1d_make2d_make_emp_distr(monkeypatch):
    monkeypatch.setattr(nu, "EmpiricalDistribution1D", lambda par, samples, bins: ("1d", par, len(samples), len(bins)))
    monkeypatch.setattr(nu, "EmpiricalDistribution2D", lambda pars, samples, bins: ("2d", tuple(pars), samples.shape, len(bins)))

    d1 = nu.make1d("p", np.array([0.0, 1.0]))
    d2 = nu.make2d(["a", "b"], np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert d1[0] == "1d"
    assert d2[0] == "2d"

    class Core:
        params = ["x_dm_gp", "y_chrom_gp", "lnlike", "lnpost", "a", "b"]

        def __call__(self, p):
            if isinstance(p, list):
                return np.array([[1.0, 2.0], [2.0, 3.0]])
            return np.array([1.0, 2.0, 3.0])

    dists = nu.make_emp_distr(Core())
    assert len(dists) >= 2


def test_log_single_likelihood_evaluation_time(monkeypatch):
    class Prior:
        def sample(self):
            return 1.0

    calls = {"n": 0}

    class PTA:
        params = [Prior(), Prior()]

        def get_lnlikelihood(self, x):
            calls["n"] += 1
            return 0.0

    t = iter([0.0, 1.0])
    monkeypatch.setattr(nu.time, "time", lambda: next(t))
    nu.log_single_likelihood_evaluation_time(PTA(), {})
    assert calls["n"] == 11


def test_get_model_and_sampler_default_settings_shape():
    model_defaults, sampler_defaults = nu.get_model_and_sampler_default_settings()
    assert "tm_svd" in model_defaults
    assert sampler_defaults["sampler"] in {"PTMCMCSampler", "GibbsSampler", "optimizer", "NUTS"}
