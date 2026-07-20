"""Unit tests for GP realization plotting helpers in plot_utils.

Focuses on:
  - _convert_units: correct unit labels and scaling
  - _get_tm_component_signal: column extraction and reference value injection
  - plot_gp_realization: mode 1/2/3 auto-units, ylabel stripping, title content
  - plot_gp_sw_ne: mode 1/2/3 ylabel, legend label, title, ref_ne gating
  - plot_gp_realizations_combined: suptitle mode label, include_tm_values forwarded to SW
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")          # non-interactive backend for CI
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pint_pal.plot_utils as pu


# ---------------------------------------------------------------------------
# Synthetic payload builder
# ---------------------------------------------------------------------------

N_TOAS = 20
N_REAL = 5
N_RN = 4    # red-noise coefficients
N_TM = 3    # timing-model columns (F0, F1, DM)

PSR = "J1234+5678"

# Coefficient draws: shape (n_real, n_coeffs)
_rng = np.random.default_rng(42)
_rn_coeffs  = _rng.standard_normal((N_REAL, N_RN))
_tm_coeffs  = _rng.standard_normal((N_REAL, N_TM))
_sw_coeffs  = _rng.standard_normal((N_REAL, 6))  # 6 SW interpolation nodes
_ecorr_c    = _rng.standard_normal((N_REAL, 2))

# Design-matrix columns: shape (n_toas, n_coeffs)
_rn_F  = _rng.standard_normal((N_TOAS, N_RN)) * 1e-7   # seconds scale
_tm_F  = _rng.standard_normal((N_TOAS, N_TM)) * 1e-6   # d(delay)/d(param)
_sw_F  = _rng.standard_normal((N_TOAS, 6))    * 1e-7
_ecorr_F = _rng.standard_normal((N_TOAS, 2))  * 1e-7

# SW shape factor (seconds per cm^-3)
_sw_shape = np.abs(_rng.standard_normal(N_TOAS)) * 1e-7 + 1e-8

# Frequencies and TOAs
_freqs = _rng.uniform(800, 2000, N_TOAS)
_toas  = np.linspace(55000, 57000, N_TOAS)


def _make_payload(include_sw=False, include_ecorr=False):
    """Build a minimal synthetic payload for use in tests."""
    gp_keys = [
        f"{PSR}_timing_model_coefficients({N_TM})",
        f"{PSR}_red_noise_coefficients({N_RN})",
        f"{PSR}_dm_gp_coefficients({N_RN})",
    ]
    realizations = {
        gp_keys[0]: _tm_coeffs.copy(),
        gp_keys[1]: _rn_coeffs.copy(),
        gp_keys[2]: _rn_coeffs.copy(),
    }
    F_columns = {
        gp_keys[0]: _tm_F.copy(),
        gp_keys[1]: _rn_F.copy(),
        gp_keys[2]: _rn_F.copy(),
    }

    if include_sw:
        sw_key = f"{PSR}_solar_wind_gp_coefficients(6)"
        gp_keys.append(sw_key)
        realizations[sw_key] = _sw_coeffs.copy()
        F_columns[sw_key] = _sw_F.copy()

    if include_ecorr:
        ec_key = f"{PSR}_ecorr_coefficients(2)"
        gp_keys.append(ec_key)
        realizations[ec_key] = _ecorr_c.copy()
        F_columns[ec_key] = _ecorr_F.copy()

    return {
        "pulsar_name": PSR,
        "gp_keys": gp_keys,
        "toas_mjd": _toas.copy(),
        "freqs_mhz": _freqs.copy(),
        "realizations": realizations,
        "F_columns": F_columns,
        "tm_fitpars": ["F0", "F1", "DM"],  # 3 columns matching N_TM
        "sw_shape_at_toas": _sw_shape.tolist() if include_sw else None,
        "solar_conjunctions_mjd": None,
        "sw_nodes_mjd": None,
    }


def _dummy_model(ne_sw=7.0, dm=15.0, f0=300.0):
    """Return a SimpleNamespace that mimics the PINT model attribute access."""
    return SimpleNamespace(
        NE_SW=SimpleNamespace(value=ne_sw),
        DM=SimpleNamespace(value=dm),
        DM1=SimpleNamespace(value=0.0),
        DM2=SimpleNamespace(value=0.0),
        F0=SimpleNamespace(value=f0),
        F1=SimpleNamespace(value=0.0),
        F2=SimpleNamespace(value=0.0),
    )


@pytest.fixture(autouse=True)
def close_plots():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Tests: _convert_units
# ---------------------------------------------------------------------------

class TestConvertUnits:
    _sig = np.ones((3, 10)) * 1e-6   # 1 µs in seconds
    _freqs = np.full(10, 1400.0)

    def test_us_label_and_scaling(self):
        out, lbl = pu._convert_units(self._sig, self._freqs, "red_noise", "us")
        assert "µs" in lbl or "mu" in lbl.lower() or r"\mu" in lbl
        np.testing.assert_allclose(out, 1.0)  # 1e-6 s → 1 µs

    def test_s_label_and_scaling(self):
        out, lbl = pu._convert_units(self._sig, self._freqs, "red_noise", "s")
        assert "(s)" in lbl or "s)" in lbl
        np.testing.assert_allclose(out, 1e-6)  # stays in seconds

    def test_dm_label_for_dm_gp(self):
        _, lbl = pu._convert_units(self._sig, self._freqs, "dm_gp", "dm")
        assert "DM" in lbl or "dm" in lbl.lower()

    def test_dm_full_label_for_dm_gp(self):
        _, lbl = pu._convert_units(self._sig, self._freqs, "dm_gp", "dm_full")
        assert "pc" in lbl.lower() or "cm" in lbl

    def test_us_at_1400_label(self):
        _, lbl = pu._convert_units(self._sig, self._freqs, "chrom", "us@1400")
        assert "1400" in lbl

    def test_ne_label_falls_back_for_non_interp(self):
        """'ne' units without sw_shape fall back to DM-like label."""
        _, lbl = pu._convert_units(self._sig, self._freqs, "sw", "ne")
        assert lbl  # just check it returns something

    def test_unknown_units_raises(self):
        with pytest.raises(ValueError, match="Unknown target_units"):
            pu._convert_units(self._sig, self._freqs, "red_noise", "bananas")


# ---------------------------------------------------------------------------
# Tests: _get_tm_component_signal
# ---------------------------------------------------------------------------

class TestGetTmComponentSignal:
    def test_returns_none_when_no_tm_key(self):
        payload = _make_payload()
        # Remove the timing_model key
        payload["gp_keys"] = [k for k in payload["gp_keys"] if "timing" not in k]
        sig, names = pu._get_tm_component_signal(payload, "dm_gp", PSR)
        assert sig is None
        assert names == []

    def test_dm_columns_selected_for_dm_gp(self):
        payload = _make_payload()
        sig, names = pu._get_tm_component_signal(payload, "dm_gp", PSR)
        # DM is in _TM_PARAMS_FOR_CATEGORY['dm_gp']
        assert "DM" in names
        assert sig is not None
        assert sig.shape == (N_REAL, N_TOAS)

    def test_f0_columns_selected_for_red_noise(self):
        payload = _make_payload()
        sig, names = pu._get_tm_component_signal(payload, "red_noise", PSR)
        assert "F0" in names or "F1" in names
        assert sig is not None

    def test_reference_values_added_mode3(self):
        payload = _make_payload()
        model = _dummy_model(dm=15.0)
        sig_no_ref, _ = pu._get_tm_component_signal(
            payload, "dm_gp", PSR, model=model, include_reference_values=False
        )
        sig_ref, _ = pu._get_tm_component_signal(
            payload, "dm_gp", PSR, model=model, include_reference_values=True
        )
        # Mode 3 adds a non-zero offset; the arrays should differ
        assert not np.allclose(sig_no_ref, sig_ref)

    def test_sw_reference_not_added_by_this_function(self):
        """SW reference values must NOT be added here (handled by plot_gp_sw_ne)."""
        payload = _make_payload(include_sw=True)
        model = _dummy_model(ne_sw=7.0)
        sig_no_ref, _ = pu._get_tm_component_signal(
            payload, "sw", PSR, model=model, include_reference_values=False
        )
        sig_ref, _ = pu._get_tm_component_signal(
            payload, "sw", PSR, model=model, include_reference_values=True
        )
        # SW is skipped — both calls return the same result
        if sig_no_ref is not None and sig_ref is not None:
            np.testing.assert_array_equal(sig_no_ref, sig_ref)


# ---------------------------------------------------------------------------
# Tests: plot_gp_realization — auto-units, ylabel, title
# ---------------------------------------------------------------------------

class TestPlotGpRealization:
    """Test that auto-units, ylabel, and title reflect modes 1/2/3 correctly."""

    def _get_ax_labels(self, payload, gp_key, **kwargs):
        fig, ax = pu.plot_gp_realization(payload, gp_key, **kwargs)
        return ax.get_ylabel(), ax.get_title(), fig

    # --- Mode 1: GP only ---

    def test_mode1_rn_ylabel_is_mus(self):
        payload = _make_payload()
        rn_key = [k for k in payload["gp_keys"] if "red" in k][0]
        ylabel, _, _ = self._get_ax_labels(
            payload, rn_key,
            include_tm_perturbations=False,
            include_tm_values=False,
        )
        assert r"\mu" in ylabel or "µ" in ylabel, f"Expected µs ylabel, got: {ylabel}"

    def test_mode1_rn_ylabel_has_delta(self):
        payload = _make_payload()
        rn_key = [k for k in payload["gp_keys"] if "red" in k][0]
        ylabel, _, _ = self._get_ax_labels(
            payload, rn_key,
            include_tm_perturbations=False,
            include_tm_values=False,
        )
        assert r"\Delta" in ylabel or "Delta" in ylabel, f"Expected Δ in ylabel, got: {ylabel}"

    def test_mode1_title_contains_gp_only(self):
        payload = _make_payload()
        rn_key = [k for k in payload["gp_keys"] if "red" in k][0]
        _, title, _ = self._get_ax_labels(
            payload, rn_key,
            include_tm_perturbations=False,
            include_tm_values=False,
        )
        assert "GP only" in title, f"Expected 'GP only' in title, got: {title}"

    # --- Mode 2: GP + ΔTM ---

    def test_mode2_rn_ylabel_still_has_delta(self):
        payload = _make_payload()
        rn_key = [k for k in payload["gp_keys"] if "red" in k][0]
        model = _dummy_model()
        ylabel, _, _ = self._get_ax_labels(
            payload, rn_key,
            include_tm_perturbations=True,
            include_tm_values=False,
            model=model,
        )
        # Mode 2 stays in Δ units
        assert r"\Delta" in ylabel or "Delta" in ylabel, f"Expected Δ in ylabel (mode 2), got: {ylabel}"

    def test_mode2_ylabel_still_mus_not_seconds(self):
        payload = _make_payload()
        rn_key = [k for k in payload["gp_keys"] if "red" in k][0]
        model = _dummy_model()
        ylabel, _, _ = self._get_ax_labels(
            payload, rn_key,
            include_tm_perturbations=True,
            include_tm_values=False,
            model=model,
        )
        assert r"\mu" in ylabel or "µ" in ylabel, f"Expected µs (not s), got: {ylabel}"
        # Must NOT just say "(s)" without µ
        assert ylabel.strip() != r"$\Delta t$ (s)"

    def test_mode2_title_contains_dtm(self):
        payload = _make_payload()
        rn_key = [k for k in payload["gp_keys"] if "red" in k][0]
        model = _dummy_model()
        _, title, _ = self._get_ax_labels(
            payload, rn_key,
            include_tm_perturbations=True,
            include_tm_values=False,
            model=model,
        )
        assert r"\Delta" in title or "ΔTM" in title or "delta" in title.lower(), \
            f"Expected ΔTM reference in title (mode 2), got: {title}"

    # --- Mode 3: GP + ΔTM + ref ---

    def test_mode3_rn_ylabel_strips_delta(self):
        payload = _make_payload()
        rn_key = [k for k in payload["gp_keys"] if "red" in k][0]
        model = _dummy_model()
        ylabel, _, _ = self._get_ax_labels(
            payload, rn_key,
            include_tm_perturbations=True,
            include_tm_values=True,
            model=model,
        )
        # Δ should be stripped in mode 3
        assert r"\Delta" not in ylabel and "Delta" not in ylabel, \
            f"Δ should be stripped in mode 3 ylabel, got: {ylabel}"

    def test_mode3_rn_ylabel_still_mus(self):
        payload = _make_payload()
        rn_key = [k for k in payload["gp_keys"] if "red" in k][0]
        model = _dummy_model()
        ylabel, _, _ = self._get_ax_labels(
            payload, rn_key,
            include_tm_perturbations=True,
            include_tm_values=True,
            model=model,
        )
        assert r"\mu" in ylabel or "µ" in ylabel, \
            f"Expected µs units in mode 3, got: {ylabel}"

    def test_mode3_title_contains_total(self):
        payload = _make_payload()
        rn_key = [k for k in payload["gp_keys"] if "red" in k][0]
        model = _dummy_model()
        _, title, _ = self._get_ax_labels(
            payload, rn_key,
            include_tm_perturbations=True,
            include_tm_values=True,
            model=model,
        )
        assert "total" in title.lower(), f"Expected 'total' in title (mode 3), got: {title}"

    # --- DM panels stay in DM units ---

    def test_dm_gp_mode1_uses_dm_units(self):
        payload = _make_payload()
        dm_key = [k for k in payload["gp_keys"] if "dm_gp" in k][0]
        ylabel, _, _ = self._get_ax_labels(
            payload, dm_key,
            include_tm_perturbations=False,
            include_tm_values=False,
        )
        assert "DM" in ylabel or "pc" in ylabel.lower() or r"10^{-3}" in ylabel, \
            f"Expected DM units for dm_gp mode 1, got: {ylabel}"

    def test_dm_gp_mode3_ylabel_no_delta(self):
        payload = _make_payload()
        dm_key = [k for k in payload["gp_keys"] if "dm_gp" in k][0]
        model = _dummy_model()
        ylabel, _, _ = self._get_ax_labels(
            payload, dm_key,
            include_tm_perturbations=True,
            include_tm_values=True,
            model=model,
        )
        assert r"\Delta" not in ylabel and "Delta" not in ylabel, \
            f"Δ should be stripped from DM ylabel in mode 3, got: {ylabel}"


# ---------------------------------------------------------------------------
# Tests: plot_gp_sw_ne — mode 1/2/3 labelling, ref_ne gating
# ---------------------------------------------------------------------------

class TestPlotGpSwNe:
    def _sw_key(self, payload):
        return [k for k in payload["gp_keys"] if "solar" in k or "sw" in k.lower()][0]

    def _run(self, payload, **kwargs):
        fig, ax = pu.plot_gp_sw_ne(payload, **kwargs)
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        return ax.get_ylabel(), ax.get_title(), legend_texts, fig

    # --- Mode 1: GP only ---

    def test_mode1_ylabel_has_delta(self):
        payload = _make_payload(include_sw=True)
        ylabel, _, _, _ = self._run(
            payload,
            include_tm_components=False,
            include_tm_values=False,
            model=None,
        )
        assert r"\Delta" in ylabel or "Delta" in ylabel, \
            f"Mode 1 SW ylabel should be Δn_E, got: {ylabel}"

    def test_mode1_title_contains_gp_only(self):
        payload = _make_payload(include_sw=True)
        _, title, _, _ = self._run(
            payload,
            include_tm_components=False,
            include_tm_values=False,
            model=None,
        )
        assert "GP only" in title, f"Expected 'GP only' in mode 1 SW title, got: {title}"

    def test_mode1_legend_shows_delta_ne(self):
        payload = _make_payload(include_sw=True)
        _, _, legend_texts, _ = self._run(
            payload,
            include_tm_components=False,
            include_tm_values=False,
            model=None,
        )
        combined = " ".join(legend_texts)
        assert r"\Delta" in combined or "Δ" in combined, \
            f"Mode 1 legend should show Δn_E, got: {legend_texts}"

    # --- Mode 2: GP + ΔTM (no reference NE_SW) ---

    def test_mode2_ylabel_still_delta(self):
        payload = _make_payload(include_sw=True)
        ylabel, _, _, _ = self._run(
            payload,
            include_tm_components=True,
            include_tm_values=False,
            model=_dummy_model(),
        )
        assert r"\Delta" in ylabel or "Delta" in ylabel, \
            f"Mode 2 SW ylabel should still be Δn_E (not total), got: {ylabel}"

    def test_mode2_ref_ne_not_added(self):
        """With include_tm_values=False, model.NE_SW should not be added."""
        payload = _make_payload(include_sw=True)
        model = _dummy_model(ne_sw=9999.0)  # absurdly large value
        _, title, _, fig2 = self._run(
            payload,
            include_tm_components=False,
            include_tm_values=False,
            model=model,
        )
        # Title should NOT say "total"
        assert "total" not in title.lower(), \
            f"Mode 2 with include_tm_values=False should not add ref, got title: {title}"
        # Median should not be ~9999
        ax = fig2.get_axes()[0]
        lines = [l for l in ax.get_lines() if len(l.get_ydata()) == N_TOAS]
        if lines:
            assert np.abs(np.median(lines[0].get_ydata())) < 100, \
                "NE_SW reference (9999) must not appear when include_tm_values=False"

    def test_mode2_title_dtm(self):
        payload = _make_payload(include_sw=True)
        _, title, _, _ = self._run(
            payload,
            include_tm_components=True,
            include_tm_values=False,
            model=_dummy_model(),
        )
        assert "ΔTM" in title or r"\Delta" in title or "dtm" in title.lower(), \
            f"Mode 2 SW title should reference ΔTM, got: {title}"

    def test_mode2_legend_shows_dtm(self):
        payload = _make_payload(include_sw=True)
        _, _, legend_texts, _ = self._run(
            payload,
            include_tm_components=True,
            include_tm_values=False,
            model=_dummy_model(),
        )
        combined = " ".join(legend_texts)
        assert r"\Delta" in combined or "ΔTM" in combined or "delta" in combined.lower(), \
            f"Mode 2 legend should show ΔTM, got: {legend_texts}"

    # --- Mode 3: GP + ΔTM + ref NE_SW ---

    def test_mode3_ylabel_no_delta(self):
        payload = _make_payload(include_sw=True)
        ylabel, _, _, _ = self._run(
            payload,
            include_tm_components=True,
            include_tm_values=True,
            model=_dummy_model(ne_sw=7.0),
        )
        assert r"\Delta" not in ylabel and "Delta" not in ylabel, \
            f"Mode 3 SW ylabel should be n_E (no Δ), got: {ylabel}"

    def test_mode3_title_contains_total(self):
        payload = _make_payload(include_sw=True)
        _, title, _, _ = self._run(
            payload,
            include_tm_components=True,
            include_tm_values=True,
            model=_dummy_model(ne_sw=7.0),
        )
        assert "total" in title.lower(), f"Mode 3 SW title should say 'total', got: {title}"

    def test_mode3_legend_shows_total(self):
        payload = _make_payload(include_sw=True)
        _, _, legend_texts, _ = self._run(
            payload,
            include_tm_components=True,
            include_tm_values=True,
            model=_dummy_model(ne_sw=7.0),
        )
        combined = " ".join(legend_texts)
        assert "total" in combined.lower(), \
            f"Mode 3 SW legend should say 'total', got: {legend_texts}"

    def test_mode3_ref_ne_shifts_median(self):
        """Adding ref NE_SW=100 should shift the median by ~100 cm^-3."""
        payload = _make_payload(include_sw=True)
        _, _, _, fig_no_ref = self._run(
            payload,
            include_tm_components=False,
            include_tm_values=False,
            model=None,
        )
        _, _, _, fig_ref = self._run(
            payload,
            include_tm_components=False,
            include_tm_values=True,
            model=_dummy_model(ne_sw=100.0),
        )
        ax_no_ref = fig_no_ref.get_axes()[0]
        ax_ref = fig_ref.get_axes()[0]
        lines_no = [l for l in ax_no_ref.get_lines() if len(l.get_ydata()) == N_TOAS]
        lines_r  = [l for l in ax_ref.get_lines()    if len(l.get_ydata()) == N_TOAS]
        if lines_no and lines_r:
            diff = np.median(lines_r[0].get_ydata()) - np.median(lines_no[0].get_ydata())
            assert abs(diff) > 1.0, \
                f"NE_SW=100 should shift median by >1 cm^-3, got diff={diff}"

    def test_no_sw_shape_returns_none(self):
        """Without sw_shape_at_toas the function should warn and return None."""
        payload = _make_payload(include_sw=True)
        payload["sw_shape_at_toas"] = None  # remove shape factor
        result = pu.plot_gp_sw_ne(payload)
        assert result == (None, None)


# ---------------------------------------------------------------------------
# Tests: plot_gp_realizations_combined — suptitle, SW forwarding
# ---------------------------------------------------------------------------

class TestPlotGpRealizationsCombined:
    def _suptitle(self, payload, **kwargs):
        fig, axes = pu.plot_gp_realizations_combined(payload, **kwargs)
        # suptitle is stored as fig.texts[0]
        texts = [t.get_text() for t in fig.texts]
        return " ".join(texts), fig, axes

    def test_suptitle_mode1(self):
        payload = _make_payload()
        title, _, _ = self._suptitle(
            payload,
            include_tm_perturbations=False,
            include_tm_values=False,
        )
        assert "GP only" in title, f"Expected 'GP only' in suptitle (mode 1), got: {title}"

    def test_suptitle_mode2(self):
        payload = _make_payload()
        title, _, _ = self._suptitle(
            payload,
            include_tm_perturbations=True,
            include_tm_values=False,
            model=_dummy_model(),
        )
        assert r"\Delta" in title or "ΔTM" in title or "delta" in title.lower(), \
            f"Expected ΔTM in suptitle (mode 2), got: {title}"

    def test_suptitle_mode3(self):
        payload = _make_payload()
        title, _, _ = self._suptitle(
            payload,
            include_tm_perturbations=True,
            include_tm_values=True,
            model=_dummy_model(),
        )
        assert "total" in title.lower(), f"Expected 'total' in suptitle (mode 3), got: {title}"

    def test_returns_correct_number_of_axes(self):
        payload = _make_payload()
        _, _, axes = self._suptitle(
            payload,
            include_tm_perturbations=False,
            include_tm_values=False,
        )
        assert len(axes) == len(payload["gp_keys"])

    def test_exclude_filters_panels(self):
        payload = _make_payload()
        _, _, axes = self._suptitle(
            payload,
            exclude=["timing_model"],
            include_tm_perturbations=False,
            include_tm_values=False,
        )
        n_expected = sum(
            1 for k in payload["gp_keys"]
            if pu._classify_gp(k, PSR)[1] != "timing_model"
        )
        assert len(axes) == n_expected

    def test_sw_panel_forwards_include_tm_values(self):
        """SW panel must NOT say 'total' when include_tm_values=False."""
        payload = _make_payload(include_sw=True)
        model = _dummy_model(ne_sw=7.0)

        fig_mode2, axes_mode2 = pu.plot_gp_realizations_combined(
            payload,
            include_tm_perturbations=True,
            include_tm_values=False,
            model=model,
        )
        fig_mode3, axes_mode3 = pu.plot_gp_realizations_combined(
            payload,
            include_tm_perturbations=True,
            include_tm_values=True,
            model=model,
        )

        sw_idx = next(
            i for i, k in enumerate(payload["gp_keys"])
            if pu._classify_gp(k, PSR)[1] == "sw"
        )

        title_mode2 = axes_mode2[sw_idx].get_title()
        title_mode3 = axes_mode3[sw_idx].get_title()

        assert "total" not in title_mode2.lower(), \
            f"SW mode 2 panel should not say 'total', got: {title_mode2}"
        assert "total" in title_mode3.lower(), \
            f"SW mode 3 panel should say 'total', got: {title_mode3}"
