"""Independent classic DMX prenoise fits with explicit parameter overrides."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from fnmatch import fnmatchcase
from numbers import Integral
from typing import Any
import warnings

from pint.models.noise_model import NoiseComponent
from pint_pal import dmx_utils as du
from pint_pal import lite_utils as lu


@dataclass
class ClassicDMXRun:
    """Prenoise fitter plus the applied parameter policy and preparation record."""

    prenoise: Any
    fitted: bool
    metadata: dict


def _expand_parameters(names, available):
    """Resolve exact names/globs, rejecting misspelled or absent selections."""
    if isinstance(names, str):
        raise TypeError("Use a list/tuple of parameter names, not a single string.")
    selected = []
    for name in names:
        matches = [p for p in available if fnmatchcase(p, name)]
        if not matches:
            raise ValueError(f"Parameter selection {name!r} matches no model parameters.")
        selected.extend(matches)
    return list(dict.fromkeys(selected))


def run_classic_dmx(
    *,
    model,
    toas,
    timing_config,
    binning="preserve",
    dmx_kwargs=None,
    dm_order=None,
    dm_values=None,
    solar_wind="preserve",
    sw_kwargs=None,
    sw_free=None,
    frozen_params=("DM",),
    unfrozen_params=("DMX_*",),
    fit=True,
    maxiter=None,
):
    """Prepare and optionally fit a classic DMX model, without noise inference.

    All inputs are copied. No files are written. Fit failures propagate.

    Parameters
    ----------
    binning : {'preserve', 'rebuild'}
        Retain PAR DMX parameters, or call ``du.setup_dmx(**dmx_kwargs)``.
        Rebuilding can change the TOA selection; the returned TOAs are used.
    dm_order : int or None
        None preserves derivatives. 0 disables derivatives (PINT's required
        DM1 placeholder is retained at zero and frozen). n>0 keeps/adds
        DM1 through DMn and removes higher orders. Existing values are preserved
        unless ``dm_values`` overrides them using the lite_utils convention.
    solar_wind : {'preserve', 'replace', 'remove'}
        Preserve the input SW model, replace it with deterministic SW using
        ``lu.add_deteriministic_solar_wind_to_model``, or remove deterministic
        SW and SWX. ``replace`` also removes SWX to avoid duplicate SW models.
    sw_kwargs : dict or None
        Replacement settings: NE_SW, SWM (0/1), SWP (only with SWM=1), and
        set_SWEPOCH_to_DMEPOCH. Freeze choices belong in sw_free/parameter lists.
    sw_free : sequence or None
        None follows YAML plus explicit overrides. Otherwise freezes all
        SolarWindDispersion parameters except the named ones: () fixes all,
        ('NE_SW',) fits density, ('SWP',) fits exponent with SWM=1, or both.
        Density derivatives such as NE_SW1 can also be selected if present.
        Conflicts with frozen_params/unfrozen_params raise an error.
    frozen_params, unfrozen_params : sequences
        Explicit overrides to the YAML free list. Exact names and glob patterns
        are supported, e.g. 'DMX_*' or 'DM[0-9]*'. Defaults fix DM and fit all
        DMX amplitudes. To freeze selected DMX amplitudes, remove 'DMX_*' from
        unfrozen_params and use YAML's free-dmx or list the desired amplitudes.
        Names in both override lists are errors, rather than precedence rules.
    fit : bool
        False returns a prepared fitter for inspection; True runs fit_toas.

    Returns
    -------
    ClassicDMXRun
        ``prenoise`` is the fitter, ``fitted`` records whether fitting ran,
        and ``metadata`` records noise removal, TOA indices and final policy.
        The metadata is an in-memory audit record, not a portable cache key.
    """
    if binning not in {"preserve", "rebuild"}:
        raise ValueError("binning must be 'preserve' or 'rebuild'.")
    if solar_wind not in {"preserve", "replace", "remove"}:
        raise ValueError("solar_wind must be 'preserve', 'replace', or 'remove'.")
    if dm_order is not None and (
        isinstance(dm_order, bool) or not isinstance(dm_order, Integral) or dm_order < 0
    ):
        raise ValueError("dm_order must be None or a nonnegative integer.")
    if dm_values is not None and (dm_order is None or dm_order == 0):
        raise ValueError("dm_values requires dm_order > 0.")
    if dmx_kwargs and binning != "rebuild":
        raise ValueError("dmx_kwargs requires binning='rebuild'.")
    if sw_kwargs and solar_wind != "replace":
        raise ValueError("sw_kwargs requires solar_wind='replace'.")
    if solar_wind == "remove" and sw_free is not None:
        raise ValueError("sw_free cannot be used when removing solar wind.")

    mo = copy.deepcopy(model)
    to = copy.deepcopy(toas)
    tc = copy.deepcopy(timing_config)
    original_params = set(mo.params)

    def toa_indices(t):
        return list(t.table["index"]) if "index" in t.table.colnames else None

    input_indices = toa_indices(to)
    # Include every registered PINT noise component, including DM/SW GPs.
    noise_names = [
        name for name, component in mo.components.items()
        if isinstance(component, NoiseComponent)
    ]
    lu.remove_noise(mo, noise_components=noise_names)

    if dm_order is not None:
        # DispersionDM.validate() requires DM1 even for a constant-DM model.
        lu.remove_DMn_above(mo, maximum_order=max(1, int(dm_order)))
        if dm_order:
            lu.add_DMn_and_unfreeze_DM(
                mo, DMn=int(dm_order), DMn_value=dm_values, frozen=True,
            )
        else:
            lu.add_DMn_and_unfreeze_DM(mo, DMn=1, DMn_value=0.0, frozen=True)

    if solar_wind in {"replace", "remove"}:
        for name in ("SolarWindDispersionX", "SolarWindDispersion"):
            if name in mo.components:
                mo.remove_component(name)
    if solar_wind == "replace":
        settings = copy.deepcopy(sw_kwargs or {})
        if "frozen" in settings:
            raise ValueError("Put SW freeze choices in sw_free, not sw_kwargs.")
        swm = settings.get("SWM", 0)
        if swm not in (0, 1):
            raise ValueError("SWM must be 0 or 1.")
        swp = settings.pop("SWP", None)
        if swp is not None and swm != 1:
            raise ValueError("SWP is used only with SWM=1.")
        lu.add_deteriministic_solar_wind_to_model(mo, frozen=True, **settings)
        if swp is not None:
            mo.SWP.value = swp

    if binning == "rebuild":
        to = du.setup_dmx(mo, to, **copy.deepcopy(dmx_kwargs or {}))
    dmx_names = [p for p in mo.params if p.startswith("DMX_")]
    if not dmx_names:
        raise ValueError("No DMX amplitudes found. Use an original PAR or binning='rebuild'.")
    if len(to.table) == 0:
        raise ValueError("No TOAs remain after DMX preparation.")

    # Resolve receiver/JUMP entries through the existing YAML interface.
    config_fitter = tc.construct_fitter(to, mo)
    yaml_free = list(tc.get_free_params(config_fitter))
    removed_params = original_params - set(mo.params)
    # Removing components/derivatives must also remove stale YAML entries.
    ignored_yaml = [p for p in yaml_free if p in removed_params]
    free_params = [p for p in yaml_free if p not in removed_params]
    frozen = _expand_parameters(frozen_params, mo.params)
    unfrozen = _expand_parameters(unfrozen_params, mo.params)
    if dm_order == 0:
        frozen.append("DM1")

    if sw_free is not None:
        if "SolarWindDispersion" not in mo.components:
            raise ValueError("sw_free requires a SolarWindDispersion component.")
        sw_names = list(mo.components["SolarWindDispersion"].params)
        selected_sw = _expand_parameters(sw_free, sw_names)
        frozen += [p for p in sw_names if p not in selected_sw]
        unfrozen += selected_sw

    conflict = set(frozen) & set(unfrozen)
    if conflict:
        raise ValueError(f"Parameters listed as both frozen and unfrozen: {sorted(conflict)}")

    # One policy application: YAML -> remove frozen -> add unfrozen.
    for name in frozen:
        free_params = [p for p in free_params if p != name]
    for name in unfrozen:
        if name not in free_params:
            free_params.append(name)
    free_params = list(dict.fromkeys(free_params))
    unknown = set(free_params) - set(mo.params)
    if unknown:
        raise ValueError(f"YAML free parameters absent from model: {sorted(unknown)}")
    if "SWM" in free_params or "SWEPOCH" in free_params:
        raise ValueError("SWM and SWEPOCH are model settings; keep them frozen.")
    if "SWP" in free_params and mo.SWM.value != 1:
        raise ValueError("Cannot fit SWP with SWM=0; select SWM=1 or freeze SWP.")

    mo.free_params = free_params
    empty_params = mo.find_empty_masks(to, freeze=True)
    final_free = list(mo.free_params)
    populated_dmx = [p for p in dmx_names if p not in empty_params]
    if not populated_dmx:
        raise ValueError("No populated DMX intervals remain for fitting.")
    if "DM" in final_free and all(p in final_free for p in populated_dmx):
        raise ValueError(
            "DM/DMX common-offset degeneracy: DM and every DMX amplitude are free. "
            "Freeze DM (recommended), or constrain at least one populated DMX "
            "amplitude. Free DM derivatives do not remove this degeneracy."
        )
    if all(p in final_free for p in dmx_names) and any(
        p.startswith("DM") and p[2:].isdigit() for p in final_free
    ):
        warnings.warn(
            "DM derivatives and all DMX amplitudes are free; inspect their covariance.",
            UserWarning, stacklevel=2,
        )
    mo.setup()
    mo.validate()
    fitter = tc.construct_fitter(to, mo)
    niter = tc.get_niter() if maxiter is None else maxiter
    if fit:
        fitter.fit_toas(maxiter=niter)

    return ClassicDMXRun(
        prenoise=fitter,
        fitted=bool(fit),
        metadata={
            "stage": "prenoise", "binning": binning,
            "dm_order": dm_order, "solar_wind": solar_wind,
            "removed_noise_components": noise_names,
            "removed_parameters": sorted(removed_params),
            "yaml_free_params": yaml_free, "removed_yaml_entries": ignored_yaml,
            "frozen_overrides": sorted(set(frozen)),
            "unfrozen_overrides": sorted(set(unfrozen)),
            "empty_mask_frozen": [p for p in free_params if p not in final_free],
            "final_free_params": list(fitter.model.free_params),
            "n_toas_input": len(toas.table), "n_toas_used": len(to.table),
            "input_toa_indices": input_indices, "used_toa_indices": toa_indices(to),
            "maxiter": niter,
        },
    )
