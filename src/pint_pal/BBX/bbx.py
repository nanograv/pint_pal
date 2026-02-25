# BBX.py 

from __future__ import annotations

import os
import json
import math
import hashlib
import pickle
import re
import yaml
import copy
import glob

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import (
    Any, Callable, Dict, Iterable, Iterator, List, 
    Mapping, Optional, Sequence, Tuple, Union, Type,
    Literal, TextIO,
)

from pathlib import Path
from io import StringIO

import numpy as np
import astropy.units as u
from astropy import log

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages

from astropy.coordinates import SkyCoord, solar_system_ephemeris, get_body_barycentric_posvel, get_sun
from astropy.time import Time

import pint
import pint.toa
import pint.fitter
import pint.models
import pint.utils
from pint.models.timing_model import TimingModel
from pint.models.dispersion_model import DispersionDM, DispersionDMX

import pint_pal.lite_utils as lu
import pint_pal.noise_utils as nu

from astropy.stats import bayesian_blocks

# Personal modules
from .utils import (
    handle_diagnostics,
    handle_diagnostics_multi,
    combine_repeated_toas,
    mjd_to_year,
    year_to_mjd,
    pulsar_name_to_elat,
    find_toa_by_mjd,
    by_mjd_table,
    find_toa_by_dmx,
    format_gap_summary,
) # utils.py

from .diagnostics import (
    plot_epoch_gap_histogram,
    plot_all_epoch_fit_overlay,
    plot_points_per_epoch_arrays,
    plot_b_snr_vs_time,
    plot_a_vs_b_correlation,
    plot_a_b_time_scatter,
    plot_wls_epoch_summaries,
    plot_data_gaps_diagnostics,
    plot_swx_bb_diagnostics,
    plot_dmx_segmentation_by_slice_diagnostics,
    plot_bchrom_vs_dmx,
) # diagnostics.py

# =============================================================================
# Configuration Containers
# =============================================================================

@dataclass(frozen=True)
class ReceiverSelection:
    """
    Select a subset of receivers/backends to include in the fits.

    Semantics:
    - If include is not None: start from that allow-list.
    - If exclude is not None: drop those entries after include is applied.
    """
    include: Optional[List[str]] = None
    exclude: Optional[List[str]] = None

    def validate(self) -> None:
        # Check for types and emptiness
        if self.include is not None:
            if not isinstance(self.include, list):
                raise TypeError("ReceiverSelection.include must be a list[str] or None.") 
            if len(self.include) == 0:
                raise ValueError("ReceiverSelection.include cannot be an empty list; use None instead.") 
            if not all(isinstance(x, str) and x for x in self.include):
                raise TypeError("ReceiverSelection.include entries must be non-empty strings.")
        if self.exclude is not None:
            if not isinstance(self.exclude, list):
                raise TypeError("ReceiverSelection.exclude must be a list[str] or None.") 
            if len(self.exclude) == 0:
                raise ValueError("ReceiverSelection.exclude cannot be an empty list; use None instead.") 
            if not all(isinstance(x, str) and x for x in self.exclude):
                raise TypeError("ReceiverSelection.exclude entries must be non-empty strings.")

@dataclass(frozen=True)
class PlotStyleConfig:
    """
    Plotting style settings
    """
    dpi: int = 150
    cBB: str = "C0"  # color for BB fits
    cVan: str = "C1" # color for Vanilla DMX fits

    # Load pint_pal plotting settings
    plot_settings_path: str = "plot_settings.yaml"
    backend_receiver_colors_key: str = "ng20_c"

    def load_plot_settings(self) -> Dict[str, Any]:
        with open(self.plot_settings_path, "r") as f:
            return yaml.safe_load(f)

    def load_receiver_colors(self) -> Dict[str, Any]:
        ps = self.load_plot_settings()
        return ps[self.backend_receiver_colors_key]

@dataclass(frozen=True)
class InputConfig:
    """
    Input file pointings (3 ways in notebook to read in)
    """
    pulsar_name: str

    # TimingConfiguration route (optional)
    config_file: Optional[str] = None
    par_directory: Optional[str] = None
    tim_directory: Optional[str] = None

    # Direct discovery route (optional)
    path2par: Optional[str] = None
    path2tim: Optional[str] = None

    # Explicit paths (optional)
    par_file: Optional[str] = None
    tim_file: Optional[str] = None

@dataclass(frozen=True)
class OutputConfig:
    """
    Figure output settings
    """
    diagnostics: str = "off"      # "on" | "off" - toggle display diagnostic plotting
    root_fig_dir: str = "figs"    # where to save the figures
    save_figures: bool = True     # toggle to save plots to disk 
    plot: PlotStyleConfig = field(default_factory=PlotStyleConfig)

    def diag_on(self) -> bool:
        return isinstance(self.diagnostics, str) and self.diagnostics.lower() != "off"


@dataclass(frozen=True)
class PickleConfig:
    """
    Pickle settings
    """
    enabled: bool = True                # toggle pickle saves
    cache_dir: Optional[str] = None     # pointing for pickle cache
    code_version: Optional[str] = None  # for dev/multiple fit runs


# =============================================================================
# Pipeline Stages Configs
# =============================================================================
"""
The pipeline largely works in 4 phases:
Stage 0: Initial parameter inputs - set pointings/global variables, etc.. (Config containers)
Stage 1: Base fits (BaseFitConfig and SolarWindConfig): fit a constant DM value and choice of Solar Wind model. 
   If model: "SWX" and conjunction_anchor - "bb" a bayesian block subpipeline is ran
   for the SWX fits with the SW DM proxy = Sun-Earth-pulsar integral. The fits are pickled with a
   unique hash tag.
Stage 2: BB DM fits (BBXConfig) - existing DMX components/parameters are removed, the BBX pipeline is run, 
   and a new DMX component is created with the BB segmentation. The amplitudes are fit
   with the existing DMX framwork or GPs (*underconstruction*). The fits are pickled with a
   unique hash tag.
Stage 3: Noise Analysis (NoiseAnalysisConfig) - a NG20-like enterprise noise analysis is ran and attached
   to the current fitter. 
"""

@dataclass(frozen=True)
class BaseFitConfig:
    """
    Base Fit: settings for a baseline fit before BBX run (const DM + SWM selected)
    """
    receiver_selection: ReceiverSelection = field(default_factory=ReceiverSelection)
    fitter_cls: Optional[Type[pint.fitter.Fitter]] = None  
    maxiter_f0: Optional[int] = None
    maxiter_f1: Optional[int] = None

    # Sanity checks
    def validate(self) -> None:
        self.receiver_selection.validate()
        if self.fitter_cls is not None and not isinstance(self.fitter_cls, type):
            raise TypeError("BaseFitConfig.fitter_cls must be a class (type) or None.")

@dataclass(frozen=True)
class SolarWindConfig:
    """
    Solar Wind parameters and SWX proxy settings when selected.

    NOTE: for SWX fitting with BB segmentation run:
          model="SWX" with conjunction_anchor="bb" (default)
    """
    model: Optional[str] = "SWX"          # "SWM0" | "SWM1" | "SWX" | None
    ne1au: float = 7.9                    # cm^-3 at 1 AU
    swp: float = 2.0
    conjunction_anchor: str = "bb"       # "center" | "start" | "end" | "bb"
    # Bin lengths when conjunction_anchor in {"center","start","end"} selected
    swx_bin_interval_days: Optional[float] = 365.25
    # kwargs for SWX + conjunction_anchor="bb" (passed into SolarWindProxy.build_swx_bb_edges_from_proxy)
    swx_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        # Normalize for robust validation (runner may also normalize, but config should be stable).
        m = None if self.model is None else str(self.model).upper()
        a = str(self.conjunction_anchor).lower()

        if m not in ("SWM0", "SWM1", "SWX", None):
            raise ValueError(f"SolarWindConfig.model={self.model!r} not recognized.")

        if m == "SWX":
            if a not in ("center", "start", "end", "bb"):
                raise ValueError("SolarWindConfig.conjunction_anchor invalid for SWX.")
            if a != "bb" and self.swx_bin_interval_days is None:
                raise ValueError("SWX with non-'bb' anchor requires swx_bin_interval_days.")

        if m in ("SWM0", "SWM1") and a == "bb":
            raise ValueError("conjunction_anchor='bb' only makes sense for SWX.")

@dataclass(frozen=True)
class BBXConfig:
    """ 
    Bayesian Blocks settings
    """
    fitness: str = "measures" # fitness function from astropy.bayesian_blocks
    p0: float = 0.14          # false alarm probability (segmentation prior)
    
    # Bin interval and data count constraints
    min_toas: int = 8 
    min_time: u.Quantity = 0.8 * u.day
    max_time: u.Quantity = 80 * u.day 
    gap_threshold_days: float = 200.0 # interval length with no data = obs gap (exclude in BB seg.)
    trim_days: float = 1.0            # tolerance for bin refinement

    # Receiver selection
    receiver_selection: ReceiverSelection = field(default_factory=ReceiverSelection)

    # Choose your DM proxy signal source (BB input - fed to `fit_BB_pipeline`) 
    signal_source: str = "chromatic" # "chromatic" (epochwise WLS) | "dmx" (finely binned) | "residuals" (r/ν²)
    proxy_kwargs: Mapping[str, Any] = field(default_factory=dict) # proxy specific

    # Sanity checks
    def validate(self) -> None:
        self.receiver_selection.validate()
        if self.signal_source not in ("chromatic", "dmx", "residuals"):
            raise ValueError(f"BBXConfig.signal_source={self.signal_source!r} invalid.")
        if self.min_toas < 1:
            raise ValueError("BBXConfig.min_toas must be >= 1.")
        if self.min_time <= 0 * u.d:
            raise ValueError("BBXConfig.min_time must be positive.")
        if self.max_time <= 0 * u.d:
            raise ValueError("BBXConfig.max_time must be positive.")
        if self.max_time < self.min_time:
            raise ValueError("BBXConfig.max_time must be >= min_time.")


SignalSource = Literal["chromatic", "dmx", "residuals"]

@dataclass(frozen=True)
class ProxyConfig:
    """
    DM proxy construction settings. This defines *what* time series is fed to BB,
    and *how* it is computed.
    """
    signal_source: SignalSource = "chromatic"

    # chromatic proxy (epochwise WLS)
    epoch_tol_days: float = 6.5
    use_inv_nu2: bool = True
    ref_freq: Optional[float] = None
    min_channels: int = 2
    min_unique_channels: int = 3
    min_x_span: float = 1.066e-07
    min_snr_b: float = 0.0
    clip_resid_outliers: bool = False
    mad_sigma: int = 5
    normalize_x: bool = True
    normalize_method: Literal["span", "std"] = "span"
    return_dm_units: bool = True

    # DMX proxy (fine DMX + dmxparse) 
    dmx_bin_days: float = 1.0
    keep_DM: bool = True
    freeze_DM: bool = False
    fitter_maxiter: Optional[int] = None

    # residuals proxy (r/\nu²) 
    # Nothing to build use 1/\nu² weighted residuals as direct input to bayesian_blocks

    # Escape hatch
    extra: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.signal_source not in ("chromatic", "dmx", "residuals"):
            raise ValueError(f"ProxyConfig.signal_source={self.signal_source!r} invalid.")

        if self.signal_source == "chromatic":
            if self.epoch_tol_days <= 0:
                raise ValueError("ProxyConfig.epoch_tol_days must be positive.")
            if self.min_channels < 1:
                raise ValueError("ProxyConfig.min_channels must be >= 1.")
            if self.min_unique_channels < 1:
                raise ValueError("ProxyConfig.min_unique_channels must be >= 1.")
            if self.min_unique_channels > self.min_channels:
                raise ValueError("min_unique_channels cannot exceed min_channels.")
            if self.min_x_span < 0:
                raise ValueError("ProxyConfig.min_x_span must be >= 0.")
            if self.min_snr_b < 0:
                raise ValueError("ProxyConfig.min_snr_b must be >= 0.")
            if self.mad_sigma < 1:
                raise ValueError("ProxyConfig.mad_sigma must be >= 1.")
            if self.normalize_method not in ("span", "std"):
                raise ValueError("ProxyConfig.normalize_method must be 'span' or 'std'.")

        if self.signal_source == "dmx":
            if self.dmx_bin_days <= 0:
                raise ValueError("ProxyConfig.dmx_bin_days must be positive.")
            if self.fitter_maxiter is not None and self.fitter_maxiter < 1:
                raise ValueError("ProxyConfig.fitter_maxiter must be >= 1 when set.")

@dataclass(frozen=True)
class NoiseAnalysisConfig:
    """
    Noise analysis (NG20-like) settings
    """
    run_noise_analysis: bool = True
    do_refit: bool = True
    save_pickle: bool = True
    base_noise_dir: Optional[str] = None


@dataclass(frozen=True)
class RunConfig:
    """
    Single immutable container for all initialization parameters defined in previous Config objects.
    """
    inp: InputConfig
    out: OutputConfig = field(default_factory=OutputConfig)
    pkl: PickleConfig = field(default_factory=PickleConfig)
    bbx: BBXConfig = field(default_factory=BBXConfig)
    sw: SolarWindConfig = field(default_factory=SolarWindConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    basefit: BaseFitConfig = field(default_factory=BaseFitConfig)
    noise: NoiseAnalysisConfig = field(default_factory=NoiseAnalysisConfig)

    # Sanity checks
    def validate(self) -> None:
        self.bbx.validate()
        self.proxy.validate()
        self.sw.validate()
        self.basefit.validate()
        # enforce consistent receiver policy between BaseFits and BBX
        if self.basefit.receiver_selection != self.bbx.receiver_selection:
            raise ValueError("ReceiverSelection differs between BaseFitConfig and BBXConfig.")
        # par's and tim's must exist 
        if not os.path.exists(self.inp.par_file):
            raise FileNotFoundError(f"[RunConfig.validate] par_file not found: {self.inp.par_file}")
        if not os.path.exists(self.inp.tim_file):
            raise FileNotFoundError(f"[RunConfig.validate] tim_file not found: {self.inp.tim_file}")


# =============================================================================
# File naming Objects: derive compact tags and output paths
# =============================================================================

@dataclass(frozen=True)
class OutputPaths:
    """
    File structuring: derive figure directory and compact tags from RunConfig.

    Notes
    -----
      - OutputPaths is the central path manager for directories + naming conventions.
      - Runners define *stems* (lightweight, local), and call fig_path()/artifact_path().
      - Plotters never derive paths.
    """
    fig_dir: str
    fit_tag: str
    config_tag: str
    rcvr_tag: str

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _rcvr_tag(rs: "ReceiverSelection") -> str:
        """Derive receiver/backend compact tags."""
        if rs.include and rs.exclude:
            incl = "_".join(sorted(r[:4] for r in rs.include))
            excl = "_".join(sorted(r[:4] for r in rs.exclude))
            return f"incl_{incl}__excl_{excl}"
        if rs.include:
            return "incl_" + "_".join(sorted(r[:4] for r in rs.include))
        if rs.exclude:
            return "excl_" + "_".join(sorted(r[:4] for r in rs.exclude))
        return "allrcvrs"

    @staticmethod
    def _sw_config_tag(sw: "SolarWindConfig") -> str:
        """Derive solar-wind compact tags."""
        if sw.model == "SWM0":
            return f"NE{sw.ne1au}"
        if sw.model == "SWM1":
            return f"NE{sw.ne1au}_SWP{sw.swp}"
        if sw.model == "SWX":
            conj_tag = {"center": "C", "start": "S", "end": "E", "bb": "BB"}.get(
                sw.conjunction_anchor, "N"
            )
            if conj_tag == "BB":
                return f"NE{sw.ne1au}_SWP{sw.swp}_{conj_tag}"
            interval_tag = (
                f"{int(sw.swx_bin_interval_days)}d" if sw.swx_bin_interval_days is not None else "None"
            )
            return f"NE{sw.ne1au}_SWP{sw.swp}_{conj_tag}_{interval_tag}"
        return str(sw.model)

    @classmethod
    def from_config(cls, cfg: "RunConfig") -> "OutputPaths":
        """
        Derive figure directory and tags.

        Example resulting fig_dir:
          figs/J0030+0451/measures_p0_0.14/SWX/NE7.9_SWP2.0_BB/allrcvrs
        """
        p = cfg.inp.pulsar_name
        fit_tag = f"{cfg.bbx.fitness}_p0_{cfg.bbx.p0}"
        config_tag = cls._sw_config_tag(cfg.sw)
        rcvr_tag = cls._rcvr_tag(cfg.bbx.receiver_selection)

        fig_dir = os.path.join(
            cfg.out.root_fig_dir,
            p,
            fit_tag,
            str(cfg.sw.model),
            config_tag,
            rcvr_tag,
        )

        return cls(fig_dir=fig_dir, fit_tag=fit_tag, config_tag=config_tag, rcvr_tag=rcvr_tag)

    # -------------------------------------------------------------------------
    # Central path constructors
    # -------------------------------------------------------------------------

    def fig_path(self, stem: str, ext: str = ".png") -> str:
        """
        Return full path for a figure.

        Parameters
        ----------
        stem : str
            Filename without extension (runner-defined).
            Example: "J0030+0451_timing_model_comparison"
        ext : str
            File extension including dot. Default ".png".
        """
        ext = ext if ext.startswith(".") else f".{ext}"
        return os.path.join(self.fig_dir, f"{stem}{ext}")

    def artifact_path(self, stem: str, ext: str) -> str:
        """
        Return full path for a non-figure diagnostic artifact (e.g., .npy, .json, .csv).
        """
        ext = ext if ext.startswith(".") else f".{ext}"
        return os.path.join(self.fig_dir, f"{stem}{ext}")

    def ensure_fig_dir(self, *, enabled: bool) -> None:
        """
        Create fig_dir if any saving is enabled (runner decides enabled = out.save_figures).
        """
        if enabled:
            os.makedirs(self.fig_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    def write_metadata(self, cfg: "RunConfig") -> None:
        """
        Write lightweight metadata json (only when saving is enabled).
        """
        if not cfg.out.save_figures:
            return

        self.ensure_fig_dir(enabled=True)

        meta: Dict[str, Any] = {
            "pulsarName": cfg.inp.pulsar_name,
            "solarWindModel": str(cfg.sw.model),
            "config_tag": self.config_tag,
            "fitFunc": cfg.bbx.fitness,
            "p0": cfg.bbx.p0,
            "min_toas": cfg.bbx.min_toas,
            "min_time": str(cfg.bbx.min_time),
            "max_time": str(cfg.bbx.max_time),
            "gap_threshold": str(cfg.bbx.gap_threshold_days * u.d),
            "trim_days": cfg.bbx.trim_days,
            "include": list(cfg.bbx.receiver_selection.include) if cfg.bbx.receiver_selection.include else None,
            "exclude": list(cfg.bbx.receiver_selection.exclude) if cfg.bbx.receiver_selection.exclude else None,
            "signal_source": cfg.proxy.signal_source,
            "diagnostics": cfg.out.diagnostics,
            "save_figures": bool(cfg.out.save_figures),
            "code_version": cfg.pkl.code_version,
        }

        meta_path = self.artifact_path("metadata", ".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)

# =============================================================================
# Shared Run Context
# =============================================================================

@dataclass
class RunContext:
    """
    Container for runtime state/data products  (e.g. pipeline outputs)
    """
    pulsar_name: Optional[str] = None
    par_file: Optional[str] = None
    tim_file: Optional[str] = None

    model: Optional[TimingModel] = None
    toas: Optional[pint.toa.TOAs] = None
    fitter: Optional[pint.fitter.Fitter] = None

    mjds: Optional[np.ndarray] = None
    freqs: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    residual_err: Optional[np.ndarray] = None

    receiver_dict: Optional[Dict[str, Any]] = None
    masks: Dict[str, np.ndarray] = field(default_factory=dict)

    products: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, str] = field(default_factory=dict)

# =============================================================================
# Pickling Routine
# =============================================================================

@dataclass(frozen=True)
class SidecarSpec:
    """
    Declares an additional cache artifact e.g. data product required (or optional) 
    to treat a cache as valid and load previously pickled products.
    """
    suffix: str           # e.g. "_pipe.npz", "_proxy.json", ".par"
    required: bool = True # if True, cache hit requires it to exist
    description: str = "" # human-readable


@dataclass(frozen=True)
class PickleSpec:
    """
    A complete cache contract for a single pickle load (usually one fitter).
    e.g. what are the data products for a given stage of the pipeline for the
         pickle routine to hit on and load the products
    """
    stage: str                   # "basefit" | "bbx" | "noise" | ...
    prefix: str                  # full path prefix, no extension
    key: str                     # stable hash key (for logging/debug)
    fitter_required: bool = True # typically True
    sidecars: Tuple[SidecarSpec, ...] = ()

    @property
    def pkl_path(self) -> str:
        return self.prefix + ".pkl"

    def sidecar_paths(self) -> Dict[str, str]:
        return {sc.suffix: self.prefix + sc.suffix for sc in self.sidecars}

class PicklerIO:
    """
    Generic cache I/O:
      - makes stable cache keys (unique hash tag)
      - checks existence against a PickleSpec contract (what products are required?)
      - loads/saves fitter pickles
      - loads/saves simple JSON/NPZ sidecar products (generic helpers)
    """

    # -------------------------------------------------------------------------
    # hashing / normalization
    # -------------------------------------------------------------------------

    @staticmethod
    def sha256_str(s: str, n: int = 10) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]

    @staticmethod
    def sha256_file(path: str, n: int = 10) -> str:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:n]

    @staticmethod
    def normalize_value(v: Any) -> Any:
        """
        For hash key creation: normalize objects to JSON safe key
        """
        import numpy as _np
        import astropy.units as _u
        import pathlib

        # Astropy quantities: preserve unit and value(s) in that unit
        if isinstance(v, _u.Quantity):
            unit_str = str(v.unit)
            if v.isscalar:
                return {"__qty__": float(v.to_value(v.unit)), "unit": unit_str}
            else:
                return {"__qty__": _np.asarray(v.to_value(v.unit)).tolist(), "unit": unit_str}

        # Callables
        if callable(v):
            mod = getattr(v, "__module__", None)
            qn = getattr(v, "__qualname__", None)
            if mod and qn:
                return {"__callable__": f"{mod}.{qn}"}

        # Numpy 
        if isinstance(v, _np.ndarray):
            return v.tolist()
        if isinstance(v, _np.generic):
            return v.item()

        # File paths
        if isinstance(v, pathlib.Path):
            return str(v)

        # Dict/mappings/sequences
        if isinstance(v, dict):
            return {str(k): PicklerIO.normalize_value(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [PicklerIO.normalize_value(val) for val in v]

        # Sets
        if isinstance(v, set):
            norm_elems = [PicklerIO.normalize_value(x) for x in v]
            # Make sortable even if elements are dicts/lists by sorting on repr
            return sorted(norm_elems, key=lambda x: repr(x))

        # Everything else
        if isinstance(v, (float, int, str, bool)) or v is None:
            return v
        # repr as last resort for anything else passed.
        return repr(v)

    @classmethod
    def make_cache_key(cls, **kwargs: Any) -> str:
        normed = {k: cls.normalize_value(v) for k, v in (kwargs or {}).items()}
        blob = json.dumps(dict(sorted(normed.items())), sort_keys=True, default=str)
        return cls.sha256_str(blob)

    # -------------------------------------------------------------------------
    # contract checks
    # -------------------------------------------------------------------------

    @staticmethod
    def ensure_dir_for_prefix(prefix: str) -> None:
        d = os.path.dirname(prefix) or "."
        os.makedirs(d, exist_ok=True)

    @classmethod
    def have(cls, spec: PickleSpec) -> Tuple[bool, Dict[str, bool]]:
        """
        Returns (ok, detail) where detail maps artifact names to existence booleans.
        """
        detail: Dict[str, bool] = {}
        if spec.fitter_required:
            detail[".pkl"] = os.path.exists(spec.pkl_path)
        for sc in spec.sidecars:
            path = spec.prefix + sc.suffix
            detail[sc.suffix] = os.path.exists(path)
        ok = True
        if spec.fitter_required and not detail.get(".pkl", False):
            ok = False
        for sc in spec.sidecars:
            if sc.required and not detail.get(sc.suffix, False):
                ok = False
        return ok, detail

    # -------------------------------------------------------------------------
    # fitter I/O
    # -------------------------------------------------------------------------

    @classmethod
    def load_fitter(cls, spec: PickleSpec) -> pint.fitter.Fitter:
        with open(spec.pkl_path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def save_fitter(
        cls,
        fitter: pint.fitter.Fitter,
        spec: PickleSpec,
        *,
        write_par: bool = False,
        meta_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        cls.ensure_dir_for_prefix(spec.prefix)

        with open(spec.pkl_path, "wb") as f:
            pickle.dump(fitter, f)

        # Optional .par snapshot
        if write_par:
            try:
                fitter.model.as_parfile(spec.prefix + ".par")
            except Exception as e:
                print(f"[PicklerIO.save_fitter] WARNING: failed to write .par: {e}")

        # Optional small json meta
        if meta_json is not None:
            with open(spec.prefix + ".json", "w") as f:
                json.dump(cls.normalize_value(meta_json), f, indent=2)

    # -------------------------------------------------------------------------
    # generic sidecar helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def save_npz(prefix: str, suffix: str, **arrays: Any) -> None:
        path = prefix + suffix
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})

    @staticmethod
    def load_npz(prefix: str, suffix: str) -> np.lib.npyio.NpzFile:
        return np.load(prefix + suffix, allow_pickle=False)

    @staticmethod
    def save_json(prefix: str, suffix: str, payload: Mapping[str, Any]) -> None:
        with open(prefix + suffix, "w") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def load_json(prefix: str, suffix: str) -> Dict[str, Any]:
        with open(prefix + suffix, "r") as f:
            return json.load(f)

class CachePolicy:
    """
    Stage-specific cache logic (ea. stage's "contract"), centralized:
      - builds cache keys from ea. stages config and/or RunContext
      - defines prefix naming
      - declares sidecar contracts
    """

    def __init__(self, *, cfg: RunConfig, paths: OutputPaths, io: PicklerIO = PicklerIO()):
        self.cfg = cfg
        self.paths = paths
        self.io = io

    # -------------------------------------------------------------------------
    # BaseFits: f0/f1
    # -------------------------------------------------------------------------

    def cache_root(self) -> str:
        return self.cfg.pkl.cache_dir or self.paths.fig_dir
    
    def basefit_spec(self, *, which: str, ctx: RunContext) -> PickleSpec:
        """
        which: "f0" (OG par fit) | "f1" (constDM + SW fit)
        """
        if which not in ("f0", "f1"):
            raise ValueError("basefit_spec.which must be 'f0' or 'f1'.")

        par_file = ctx.par_file
        tim_file = ctx.tim_file
        if not par_file or not tim_file:
            raise ValueError("RunContext must have par_file and tim_file set before building cache specs.")

        # NOTE: basefit uses fig_dir/cache as save path
        cache_dir = os.path.join(self.cache_root(), "cache")
        os.makedirs(cache_dir, exist_ok=True)

        par_sha = self.io.sha256_str(open(par_file, "r", encoding="utf-8").read())
        tim_sha = self.io.sha256_file(tim_file)

        # Construct unique hash for fits
        key = self.io.make_cache_key(
            stage="basefit",
            which=which,
            par_sha=par_sha,
            tim_sha=tim_sha,
            solarWindModel=self.cfg.sw.model,
            ne1au=self.cfg.sw.ne1au,
            swp=self.cfg.sw.swp,
            conjunction_anchor=self.cfg.sw.conjunction_anchor,
            swx_bin_interval_days=self.cfg.sw.swx_bin_interval_days,
            swx_kwargs=dict(self.cfg.sw.swx_kwargs or {}),
            code_version=self.cfg.pkl.code_version,
            receiver_selection=dict(
                include=self.cfg.bbx.receiver_selection.include,
                exclude=self.cfg.bbx.receiver_selection.exclude,
            ),
            fitter_cls=self._stable_callable_id(self.cfg.basefit.fitter_cls) if self.cfg.basefit.fitter_cls else None,
        )

        # Unique has tag -> filename for fits
        tag = f"{self.cfg.inp.pulsar_name}_{self.cfg.sw.model}_{key}"
        suffix = "_OGfit" if which == "f0" else "_ConstDM_SWfit"
        prefix = os.path.join(cache_dir, tag + suffix)

        # basefit sidecars: optional par/json, but not required for validity
        sidecars: Tuple[SidecarSpec, ...] = ()
        return PickleSpec(stage="basefit", prefix=prefix, key=key, sidecars=sidecars)

    # -------------------------------------------------------------------------
    # BBX: fitter + required sidecars
    # -------------------------------------------------------------------------

    def bbx_spec(self, *, ctx: RunContext) -> PickleSpec:
        """
        BBX cache lives under cfg.pkl.cache_dir (or fig_dir/bb_cache).
        """
        psr = ctx.pulsar_name or self.cfg.inp.pulsar_name
        if not psr:
            raise ValueError("Need pulsar_name in RunContext or RunConfig to build cache spec.")
            
        par_file = ctx.par_file
        tim_file = ctx.tim_file
        if not par_file or not tim_file:
            raise ValueError("RunContext must have par_file and tim_file set before building cache specs.")
        
        cache_dir = os.path.join(self.cache_root(), "bb_cache")
        os.makedirs(cache_dir, exist_ok=True)

        par_sha = self.io.sha256_str(open(par_file, "r", encoding="utf-8").read()) if par_file else None
        tim_sha = self.io.sha256_file(tim_file) if tim_file else None

        proxy_cfg = CachePolicy.proxy_cfg_for_cache(self.cfg.proxy)

        key = self.io.make_cache_key(
            stage="bbx",
            psr=psr,
            par_sha=par_sha,
            tim_sha=tim_sha,
            signal_source=self.cfg.proxy.signal_source,
            include=sorted(self.cfg.bbx.receiver_selection.include or []),
            exclude=sorted(self.cfg.bbx.receiver_selection.exclude or []),
            fitness=self.cfg.bbx.fitness,
            p0=self.cfg.bbx.p0,
            min_toas=self.cfg.bbx.min_toas,
            min_time=float(self.cfg.bbx.min_time.to_value(u.d)),
            max_time=float(self.cfg.bbx.max_time.to_value(u.d)),
            gap_threshold_days=self.cfg.bbx.gap_threshold_days,
            trim_days=self.cfg.bbx.trim_days,
            proxy_cfg=proxy_cfg,
            code_version=self.cfg.pkl.code_version,
        )

        prefix = os.path.join(cache_dir, f"{psr}_{self.cfg.proxy.signal_source}_{key}")

        # BBX contract: require fitter + pipeline arrays + proxy json
        sidecars = (
            SidecarSpec("_pipe.npz", required=True, description="BBX pipeline arrays"),
            SidecarSpec("_proxy.json", required=True, description="Proxy meta used for BBX"),
        )
        return PickleSpec(stage="bbx", prefix=prefix, key=key, sidecars=sidecars)

    # -------------------------------------------------------------------------
    # Noise: fitter + (optional) noise chains  
    # -------------------------------------------------------------------------

    def noise_spec(self, *, base_noise_dir: str, using_wideband: bool) -> PickleSpec:
        psr = self.cfg.inp.pulsar_name
        prefix = os.path.join(base_noise_dir, f"{psr}_noiseFit")

        # A key is still useful for logging even if naming is fixed (for now):
        key = self.io.make_cache_key(
            stage="noise",
            psr=psr,
            using_wideband=using_wideband,
            base_noise_dir=base_noise_dir,
        )

        # Noise: fitter is required; chains are *not a single file*, so
        # treat them as “external prerequisites” rather than SidecarSpec.
        # e.g. other internal routines will require them from noise_utils
        sidecars: Tuple[SidecarSpec, ...] = ()
        return PickleSpec(stage="noise", prefix=prefix, key=key, sidecars=sidecars)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _stable_callable_id(fn: Any) -> str:
        return f"{getattr(fn, '__module__', 'unknown')}.{getattr(fn, '__qualname__', getattr(fn, '__name__', 'callable'))}"

    @staticmethod
    def proxy_cfg_for_cache(proxy: ProxyConfig) -> Dict[str, Any]:
        """
        Normalize ProxyConfig into a hash-stable dict for cache-key creation.
        Drops unstable/large keys and normalizes callables if any appear in `extra`.
        """
        d = asdict(proxy)
    
        # strip keys that should not be used for hashing
        extra = dict(proxy.extra or {})
        extra.pop("freqs", None)
        extra.pop("pdf_path", None)
    
        # For callables allowed in extra, normalize them
        for k in ("dmxparse_func"):
            if k in extra and extra[k] is not None:
                extra[k] = CachePolicy._stable_callable_id(extra[k])
    
        d["extra"] = extra
        return d

@dataclass
class PicklerBundle:
    """
    Small container to hand off Pickle configs to the "runner" classes
    """
    io: PicklerIO
    policy: CachePolicy

# =============================================================================
# BaseFits
# =============================================================================

class BaseFits:
    """
    Stage 1 runner:
      - f0: OG par fit
      - f1: constant-DM + SW model fit

    Notes
    -----
    - Inputs are taken primarily from RunContext; explicit args are used only as fallbacks.
    - Diagnostics are emitted via handle_diagnostics*() based on OutputConfig.
    - Optional caching is handled via PicklerBundle (PicklerIO + CachePolicy).
    """

    def __init__(
        self,
        *,
        pickler: Optional["PicklerBundle"] = None,
        sw_proxy: Optional["SolarWindProxy"] = None,
    ) -> None:
        self.pickler = pickler
        self.sw_proxy = sw_proxy

    # -------------------------------------------------------------------------
    # Stage 1a: par preprocessing: remove DMX, make DM constant-fit, zero absorbers
    # -------------------------------------------------------------------------

    def prepare_par_constant_dm(
        self,
        par_file: Union[str, Path],
        *,
        zero_params: Sequence[str] = ("DM1", "NE_SW", "NE_SW1"),  # param values to zero out
    ) -> str:
        """
        Make a constant-DM par (strips DMX, zeros params listed to preserve DM variations)

        Returns
        -------
        par text with DMX lines removed, constant DM inserted, and some params zeroed.
        """
        with open(par_file, "r", encoding="utf-8") as f:
            par_lines = f.readlines()

        # Remove all DMX parameters
        cleaned_lines = [line for line in par_lines if not line.lstrip().startswith("DMX")]

        for i, line in enumerate(cleaned_lines):
            stripped = line.lstrip()
            pieces = stripped.split()

            # Check 1: Set DM to a constant to be fit (e.g. normalize)
            if stripped.startswith("DM ") and len(pieces) >= 2:
                dm_val = pieces[1]
                cleaned_lines[i] = f"DM {dm_val} 1 0\n"
                print(f"[prepare_par_constant_dm] Baseline constant DM fit: {float(dm_val):0.4f} [pc·cm⁻³]")

            # Check 2: Zero parameters that would otherwise absorb SW structure
            elif any(stripped.startswith(p) for p in zero_params):
                p = pieces[0]
                cleaned_lines[i] = f"{p} 0\n"

        return "".join(cleaned_lines)

    # -------------------------------------------------------------------------
    # Stage 1b: build solar wind model par text
    # -------------------------------------------------------------------------

    def add_solar_model_par(
        self,
        *,
        par_text: str,                               # raw text from par file
        toas: Optional[pint.toa.TOAs],               # used to get the MJD range
        model_type: str = "SWM0",                    # "SWM0" | "SWM1" | "SWX"
        ne1au: float = 7.9,                          # electron density at 1 AU (SWM0/SWM1)
        swp: float = 2.0,                            # power-law index (SWM1/SWX)
        swx_bin_interval_days: Optional[float] = 365.25, # Bin width for SWX model
        conjunction_anchor: str = "center",          # "center" | "start" | "end" | "bb"
        bb_kwargs: Optional[Dict[str, Any]] = None,  # proxy inputs - build_swx_bb_edges_from_proxy
        sw_proxy: Optional["SolarWindProxy"] = None,
        return_payload: bool = False,
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """
        Modify a par text to define a solar wind model, removing any existing model first.
    
        Parameters
        ----------
        par_text : str
            Contents of the original .par file (as a single string).
        toas : pint.toa.TOAs
            PINT TOAs object for the pulsar (used to define MJD span).
        model_type : str
            One of 'SWM0', 'SWM1', or 'SWX', specifying the desired model.
        ne1au : float
            Electron density at 1 AU (used in SWM0 and SWM1).
        swp : float
            Power-law index for r^{-swp} falloff (used in SWM1 and SWX).
        conjunction_anchor : {'start','center','end','bb'}
            If 'bb', segment edges are built from the DM proxy via Bayesian Blocks.
            For 'start'/'center'/'end', segments are regular intervals of swx_bin_interval_days days.
        bb_kwargs : dict, optional
            Passed to SolarWindProxy.build_swx_bb_edges_from_proxy(...).
            Example keys: ne1au, ephem, min_theta_deg, include_Re,
                          gap_threshold_days, trim_days, p0, mindata, mintime, maxtime
    
        Returns
        -------
        str
            The modified par file content with the new solar wind model appended.
        """
        # Step 1: Remove any previous solar wind parameters from the par text
        lines = par_text.splitlines()  # parse by line
        cleaned_lines: List[str] = []
        removed_tags: List[str] = []
    
        for line in lines:
            stripped = line.strip()  # strip white space
            if not stripped or stripped.startswith("#"):
                cleaned_lines.append(line)
                continue  # skip comments and blank lines
            tag = stripped.split()[0]  # grab first entry on each line
            if (
                tag in ("NE_SW", "NE_SW1", "SWM", "SWP")
                or tag.startswith("SWXDM")
                or tag.startswith("SWXP")
                or tag.startswith("SWXR1")
                or tag.startswith("SWXR2")
            ):
                removed_tags.append(tag)
                continue  # drop these lines
            cleaned_lines.append(line)
    
        # Sanity check: confirm that removal occurred
        if removed_tags:
            unique_tags = sorted(set(removed_tags))
            print("Removed the following solar wind parameters from the par file:")
            for tag in unique_tags:
                print(f"  - {tag}")
        else:
            print("No existing solar wind parameters were found to remove.")
    
        # Step 2: Define new solar wind model block
        block: List[str] = []
        mt = str(model_type).upper()
        payload: Dict[str, Any] = {}
    
        # Model type: SWM0 - Simple r^-2 electron density profile
        if mt == "SWM0":
            block.append(f"NE_SW {ne1au} 1 0")  # Sets density at 1AU
            block.append("SWM 0")               # Choose r^-2
    
        # Model type: SWM1 - r^{-SWP} model with variable falloff index (SWP)
        elif mt == "SWM1":
            block.append(f"NE_SW {ne1au} 1 0")  # Sets density at 1AU
            block.append("SWM 1")               # r^{-SWP}
            block.append(f"SWP {swp} 1 0")      # sets SWP value as expo
    
        # Model type: SWX - advanced PINT solar wind model, reduces covariacnes, DMX-like bins
        elif mt == "SWX":
            # SWX needs toas to be implemented
            if toas is None:
                raise ValueError("toas must be provided for SWX model to define time segments")
    
            # Create multiple SWX segments over the TOA span with scaled amplitudes
            # Scaling to make consistent with SWM0: adapted to support dynamic segment scaling via SWXDM
            # Get MJD range from TOAs
            mjds = toas.get_mjds().value
            t_start, t_end = float(np.min(mjds)), float(np.max(mjds))
    
            # Step 2a: Build a clean temporary SWM0 model to compute max DM (e.g. at conjunction) for SWX scaling
            temp_cleaned_for_swm0: List[str] = []
            for line in cleaned_lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    temp_cleaned_for_swm0.append(line)
                    continue  # skip comments and blank lines
                tag = stripped.split()[0]
                if (
                    tag in ("NE_SW", "NE_SW1", "SWM", "SWP")
                    or tag.startswith("SWXDM")
                    or tag.startswith("SWXP")
                    or tag.startswith("SWXR1")
                    or tag.startswith("SWXR2")
                ):
                    continue  # ensure no SW model
                temp_cleaned_for_swm0.append(line)
    
            temp_model_text = "\n".join(temp_cleaned_for_swm0 + [f"NE_SW {ne1au} 1 0", "SWM 0"])
            base_model = pint.models.get_model(StringIO(temp_model_text))
            max_dm = base_model.get_max_dm()  # maximum DM due to solar wind at conjunction
    
            # Step 2b: Define the segmentation style for the SWX
            bb_kwargs = dict(bb_kwargs or {})
            anchor = conjunction_anchor.lower()
    
            if anchor in ("start", "center", "end"):
                if anchor in ("start", "center", "end") and swx_bin_interval_days is None:
                    raise ValueError("SWX with anchor in ('start','center','end') requires swx_bin_interval_days.")

                # Get approximate conjunction times at swx_bin_interval_days intervals
                spacing = np.arange(t_start, t_end, float(swx_bin_interval_days))
                conj_times = [
                    pint.utils.get_conjunction(
                        base_model.get_psr_coords(),
                        Time(s, format="mjd"),
                        precision="high",
                    )[0].mjd
                    for s in spacing
                ]
                # build edge list depending on conjunction_anchor
                half = float(swx_bin_interval_days) / 2.0
                conj = np.array(conj_times, dtype=float)
    
                if anchor == "start":
                    # bins start at each conjunction; ends are conj + SWX_bin_interval
                    starts = conj
                    stops = conj + float(swx_bin_interval_days)
                    raw = np.concatenate([[t_start], starts, stops, [t_end]])
                elif anchor == "end":
                    # bins end at each conjunction; starts are conj - SWX_bin_interval
                    stops = conj
                    starts = conj - float(swx_bin_interval_days)
                    raw = np.concatenate([[t_start], starts, stops, [t_end]])
                else:  # "center"
                    # each bin is centered on conjunction; starts, stops = conj -/+ half‐interval
                    starts = conj - half
                    stops = conj + half
                    raw = np.concatenate([[t_start], starts, stops, [t_end]])
    
                # Clip, drop duplicate bounds, and sort
                tol = 1.0  # days
                sorted_edges = np.sort(np.clip(raw, t_start, t_end))
                d = np.diff(sorted_edges)
                mask = np.concatenate(([True], d > tol))
                edges = sorted_edges[mask] # Drop duplicates & merge edges that are < tol day
    
                # Sanity Check: first and last bin will be off if using conjunction anchoring, check the rest
                diffs = np.diff(edges)
                interior = diffs[1:-1] if len(diffs) > 2 else diffs.copy()
                good = (np.abs(interior - float(swx_bin_interval_days)) <= tol) & (interior > 0)

                # report!
                n_total = len(interior)
                n_good = int(np.count_nonzero(good))
                pct_good = 100.0 * n_good / n_total if n_total else 100.0
                print(
                    f"The following interior SWX bins are within tolerance: {n_good}/{n_total} "
                    f"({pct_good:.1f}%) within ±{tol} d of {float(swx_bin_interval_days)} d"
                )
                if not np.all(good):
                    bad_idxs = np.nonzero(~good)[0] + 1 # +1 because we skipped the first diff
                    bad_widths = interior[~good]
                    print("The following interior SWX bins are out of tolerance:")
                    for idx, w in zip(bad_idxs, bad_widths):
                        tag = f"SWXR1_{idx:04d}"
                        print(f" → {tag}: width = {w:.6f} days")
    
            elif anchor == "bb":
                # BBX subpipeline for SWM: Build edges from DM proxy -> BB -> gaps -> refined BB
                if sw_proxy is None:
                    raise ValueError("SWX conjunction_anchor='bb' requires sw_proxy=SolarWindProxy.")
                dm_dict, pipeline_dict = sw_proxy.build_swx_bb_edges_from_proxy(
                    toas=toas,
                    model=base_model,
                    **bb_kwargs,
                )
                
                payload["swx_bb_pipeline"] = pipeline_dict
                payload["swx_dm_dict"] = dm_dict
                
                edges = np.asarray(pipeline_dict["SW_BB_refined"], dtype=float)
    
                # Ensure coverage of [t_start, t_end] and pad if necessary
                if edges.size == 0:
                    print(
                        f"Bayesian Block segmentation returned 0 intervals. "
                        f"Using a 1-bin segmentation from {t_start:.2f}, {t_end:.2f} MJD"
                    )
                    edges = np.array([t_start, t_end], dtype=float)
                else:
                    if edges[0] > t_start:
                        edges = np.insert(edges, 0, t_start)
                    if edges[-1] < t_end:
                        edges = np.append(edges, t_end)
    
            else:
                # Uniform ~1-year blocks from start to end (simple fallback)
                edges_tmp = np.arange(t_start, t_end, 365.25)
                edges = np.append(edges_tmp, t_end)
    
            # Step 2c: Build temporary SWX model to get scaling factors
            segment_texts: List[str] = []
            for i, (r1, r2) in enumerate(zip(edges[:-1], edges[1:]), start=1):
                tag = f"{i:04d}"
                segment_texts.extend(
                    [
                        f"SWXDM_{tag} 1.0",        # placeholder amplitudes
                        f"SWXP_{tag} {swp}",       # power-law index
                        f"SWXR1_{tag} {r1:.6f}",   # segment start
                        f"SWXR2_{tag} {r2:.6f}",   # segment end
                    ]
                )
    
            # Remove any solar wind lines again to avoid duplication from the temporary model
            temp_cleaned: List[str] = []
            for line in temp_cleaned_for_swm0:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                tag = stripped.split()[0]
                if (
                    tag in ("NE_SW", "NE_SW1", "SWM", "SWP")
                    or tag.startswith("SWXDM")
                    or tag.startswith("SWXP")
                    or tag.startswith("SWXR1")
                    or tag.startswith("SWXR2")
                ):
                    continue
                temp_cleaned.append(line)
    
            # Build temporary model to extract scaling values for each segment
            model_with_segments = pint.models.get_model(StringIO("\n".join(temp_cleaned + segment_texts)))
            scalings = model_with_segments.get_swscalings()  # One scale factor per SWXDM_i bin
    
            # Step 2d: Define final SWXDM block with scaled amplitudes
            for i, ((r1, r2), scale) in enumerate(zip(zip(edges[:-1], edges[1:]), scalings), start=1):
                tag = f"{i:04d}"
                dm_scaled = max_dm * scale
                dm_scaled_val = dm_scaled.to_value(u.pc / u.cm**3)
                block.extend(
                    [
                        f"SWXDM_{tag} {dm_scaled_val:.6f} 1 0",  # fit amplitude
                        f"SWXP_{tag} {swp}",                     # freeze the rest; exponent
                        f"SWXR1_{tag} {r1:.6f}",                 # Start bin
                        f"SWXR2_{tag} {r2:.6f}",                 # End bin
                    ]
                )
    
        else:
            raise ValueError(f"Unknown model_type: '{model_type}'")
    
        # Sanity Check: Confirm key parameters were added
        par_result = "\n".join(cleaned_lines + [""] + block)
        confirmed: List[str] = []
    
        if mt in ("SWM0", "SWM1"):
            if "NE_SW" in par_result:
                confirmed.append("NE_SW")
            if mt == "SWM1" and "SWP" in par_result:
                confirmed.append("SWP")
        elif mt == "SWX":
            if any("SWXDM_" in line for line in block): confirmed.append("SWXDM_i")
            if any("SWXP_"  in line for line in block): confirmed.append("SWXP_i")
            if any("SWXR1_" in line for line in block): confirmed.append("SWXR1_i")
            if any("SWXR2_" in line for line in block): confirmed.append("SWXR2_i")
    
        if confirmed:
            print(f"Successfully added solar wind model '{model_type}' with parameters:")
            for p in confirmed:
                print(f"  - {p}")
        else:
            print(f"Warning: no solar wind parameters were added for model_type '{model_type}'.")
    
        # Step 4: Return updated par text!!
        par_out = par_result + "\n"
        if return_payload:
            return par_out, payload
        return par_out

    # -------------------------------------------------------------------------
    # Stage 1c: SW insertion - handles SWX needing TOAs
    # -------------------------------------------------------------------------
    
    def insert_solar_wind(
        self,
        *,
        par_text_or_path: Union[str, Path, TextIO],
        tim_file: Union[str, Path],
        sw_model: Optional[str],
        ne1au: float,
        swp: float,
        conjunction_anchor: str,
        swx_bin_interval_days: Optional[float],
        bb_kwargs: Optional[Mapping[str, Any]] = None,
        return_payload: bool = False,
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """
        Handle solar wind model insertion based on toas dependency/chosen SWM
        - SWX needs TOAs to compute segment boundaries.
        - SWM0/SWM1 do not.
    
        Returns
        -------
        par text with a solar wind model inserted
        To feed to a PINT fitter run like StringIO(SW_par_text)
        """
        # Load par text
        par_text: str
        
        # Case 1: file-like (StringIO or open file handle)
        if hasattr(par_text_or_path, "read"):
            par_text = par_text_or_path.read()
        
        # Case 2: Path object
        elif isinstance(par_text_or_path, Path):
            par_text = par_text_or_path.read_text(encoding="utf-8")
        
        # Case 3: string path
        elif isinstance(par_text_or_path, str) and Path(par_text_or_path).exists():
            with open(par_text_or_path, "r", encoding="utf-8") as f:
                par_text = f.read()
        
        # Case 4: raw string (treat as text)
        elif isinstance(par_text_or_path, str):
            par_text = par_text_or_path
        
        else:
            raise TypeError(
                f"par_text_or_path must be a path (str/Path), par text (str), or file-like (TextIO) with .read(); "
                f"got {type(par_text_or_path)}"
            )

        # Sanity checks
        if sw_model in (None, "None"):
            print("No Solar Wind Model type was input. Is this what you meant to do?")
            return (par_text, {}) if return_payload else par_text
    
        sw_model = str(sw_model).upper()
        if sw_model == "SWX":
            if str(conjunction_anchor).lower() == "bb" and self.sw_proxy is None:
                raise ValueError("BaseFits requires sw_proxy=SolarWindProxy when conjunction_anchor='bb'.")
            print(f"[BaseFits] Preparing {sw_model} model segments...  .    .       .")
            # Temporarily load model and TOAs *before* inserting SWX model
            base_model, base_toas = pint.models.get_model_and_toas(StringIO(par_text), str(tim_file))
    
            # Now insert SWX blocks
            sw_par_text, payload = self.add_solar_model_par(
                par_text=par_text,
                toas=base_toas,  # required for SWX to compute segment edges
                model_type=sw_model,
                ne1au=ne1au,
                swp=swp,
                conjunction_anchor=conjunction_anchor,
                swx_bin_interval_days=swx_bin_interval_days,
                bb_kwargs=bb_kwargs,
                sw_proxy=self.sw_proxy,
                return_payload=True,
            )
    
            # Print status messages for SWX options
            if str(conjunction_anchor).lower() == "bb":
                print(f"→ Solar Wind Model: {sw_model} (segmentation = Bayesian Blocks)")
            else:
                print(
                    f"→ Solar Wind Model: {sw_model} (segmentation = '{conjunction_anchor}', "
                    f"interval = {swx_bin_interval_days} days)"
                )
    
            return (sw_par_text, payload) if return_payload else sw_par_text
    
        if sw_model in ("SWM0", "SWM1"):
            # SWM0/SWM1 don't need TOAs to generate their structure,
            # but we still run find_empty_masks in case custom segments were added in `fit_model`.
            print(f"[BaseFits] Preparing {sw_model} model...  .    .       .")
            sw_par_text = self.add_solar_model_par(
                par_text=par_text,
                toas=None,  # not used internally in this case
                model_type=sw_model,
                ne1au=ne1au,
                swp=swp,
                return_payload=False,
            )
    
            # Status message for SWM0/1 models
            print(f"→ Solar Wind Model: {sw_model} (static {sw_model} profile, no segmentation)")
            return (sw_par_text, {}) if return_payload else sw_par_text
    
        raise ValueError(f"Unknown solarWindModel: {sw_model}")

    # -------------------------------------------------------------------------
    # Stage 1d: model fitting setup
    # -------------------------------------------------------------------------

    def fit_model(
        self,
        par_source: Union[str, StringIO, Path],
        tim_file: Union[str, Path],
        *,
        fitter_cls: Optional[type] = None,
        freeze_DM_other: Optional[Sequence[re.Pattern]] = None,
        maxiter: Optional[int] = None,
    ) -> pint.fitter.Fitter:
        """
        Fit a model (par_source can be a path or StringIO) with (optional) parameter freezing and fitter.
        - fitter_cls: Fitter choice, if None default is pint.fitter.WLSFitter
        - freeze_DM_other: list[compiled regex]; matches are frozen (e.g., DM, DMn)
        - maxiter: int or None (None lets PINT run with its default/until convergence)

        Returns
        -------
        PINT fitter object
        """
        # Load the final model and toas for fitting
        model, toas = pint.models.get_model_and_toas(par_source, str(tim_file))
        # Custom check if any TOA in SWX segments
        model.find_empty_masks(toas, freeze=True)

        # Freeze global DM terms or all SW DM variation will be absorbed into it
        # Freeze the "global" DM
        if freeze_DM_other:
            for pn in list(model.params):
                for rx in freeze_DM_other:
                    if rx.fullmatch(pn) or rx.match(pn):
                        getattr(model, pn).frozen = True
                        break

        if fitter_cls is None:
            fitter_cls = pint.fitter.WLSFitter

        fitter = fitter_cls(toas, model)

        try:
            if maxiter is None:
                # Use PINT's default
                fitter.fit_toas()
            else:
                fitter.fit_toas(maxiter=int(maxiter))
        except pint.fitter.ConvergenceFailure as e:
            print("WARNING: fit did not converge:", e)

        return fitter

    # -------------------------------------------------------------------------
    # BaseFits: Stage 1 runner
    # -------------------------------------------------------------------------

    def run_or_load_basefit(
        self,
        *,
        cfg: "RunConfig",
        paths: "OutputPaths",
        ctx: "RunContext",
        which: Sequence[str] = ("f0", "f1"),
    ) -> Tuple[Optional[pint.fitter.Fitter], Optional[pint.fitter.Fitter], Dict[str, str]]:
        """
        Compute and/or load base fit products (Stage 1 runner).
        Products:
          - f0: OG fit pulled from the par file
          - f1: constant-DM + SW model fit

        Notes
        -----
          - Returns (f0, f1, cache_paths)
          - Writes all outputs + artifact paths into ctx.{products,paths}
          - Sets ctx.fitter/model/toas to the active product (precedence f1 > f0)
        """
        if self.pickler is None:
            raise ValueError("BaseFits.run_or_load_basefit requires pickler=PicklerBundle.")

        # validate ctx inputs
        psr = ctx.pulsar_name or cfg.inp.pulsar_name
        par_file = ctx.par_file or cfg.inp.par_file
        tim_file = ctx.tim_file or cfg.inp.tim_file
        if not psr:
            raise ValueError("Need pulsar_name in RunContext (or RunConfig as fallback).")
        if not par_file or not tim_file:
            raise ValueError("Need par_file and tim_file in RunContext (or RunConfig as fallback).")

        # unpack configs 
        sw_cfg = cfg.sw
        base_cfg = cfg.basefit
        out_cfg = cfg.out
        pkl_cfg = cfg.pkl

        io = self.pickler.io
        policy = self.pickler.policy

        # Decide which fits to make f0, f1, or both
        want = tuple(which)
        for w in want:
            if w not in ("f0", "f1"):
                raise ValueError("which must be a subset of ('f0','f1').")

        cache_paths: Dict[str, str] = {}
        f0: Optional[pint.fitter.Fitter] = None
        f1: Optional[pint.fitter.Fitter] = None

        # Enable caching?
        do_cache = bool(pkl_cfg.enabled)

        # -------------------------
        # Helper: f0
        # -------------------------
        def _load_or_refit_f0() -> pint.fitter.Fitter:
            """Load/check BaseFits cache contract and generic IO tools"""
            spec0 = policy.basefit_spec(which="f0", ctx=ctx)

            cache_paths["f0"] = spec0.pkl_path
            ctx.paths["basefit_f0_pkl"] = spec0.pkl_path  # record intended artifact path

            if do_cache:
                ok, _detail = io.have(spec0)
                if ok:
                    try:
                        print(f"[BaseFits:cache] Loading f0 from {spec0.pkl_path}")
                        return io.load_fitter(spec0)
                    except Exception as e:
                        print(f"[BaseFits:cache] f0 load failed ({type(e).__name__}): {e}. Recomputing.")

            print("[BaseFits] Refitting f0 (OG par fit)...  .    .       .")
            base_model, base_toas = pint.models.get_model_and_toas(str(par_file), str(tim_file))
            base_model.find_empty_masks(base_toas, freeze=True)

            FitterClass = base_cfg.fitter_cls or pint.fitter.WLSFitter
            f0_local = FitterClass(base_toas, base_model)

            try:
                if base_cfg.maxiter_f0 is None:
                    f0_local.fit_toas()
                else:
                    f0_local.fit_toas(maxiter=int(base_cfg.maxiter_f0))
            except pint.fitter.ConvergenceFailure as e:
                print(f"[BaseFits] WARNING: f0 did not converge: {e}")

            # Pickle fits?
            if do_cache:
                io.save_fitter(
                    f0_local,
                    spec0,
                    write_par=True,
                    meta_json=dict(
                        product="basefit_f0",
                        pulsar=psr,
                        solarWindModel=str(sw_cfg.model),
                        n_toas=int(len(f0_local.toas.table)),
                        did_refit=True,
                    ),
                )
                print("[BaseFits:cache] ✓ f0 pickled")

            return f0_local

        # -------------------------
        # Helper: f1
        # -------------------------
        def _load_or_refit_f1() -> pint.fitter.Fitter:
            """Load/check BaseFits cache contract and generic IO tools"""

            # Helper: register SWX+BB sidecar paths into ctx (used on cache-hit and after refit) 
            def _register_swx_bb_sidecars() -> None:
                # NOTE: these must match filenames used in the refit/miss branch
                edges_path = paths.artifact_path(f"{psr}_swx_bb_edges", ".npy")
                png1 = paths.fig_path(f"{psr}_swx_bb_segmentation", ".png")
                pdf1 = paths.fig_path(f"{psr}_swx_bb_segmentation", ".pdf")
                png2 = paths.fig_path(f"{psr}_swx_proxy_gaps", ".png")
                pdf2 = paths.fig_path(f"{psr}_swx_proxy_gaps", ".pdf")
                gaps_txt = paths.artifact_path(f"{psr}_swx_gaps_summary", ".txt")
        
                ctx.paths["swx_bb_edges_npy"] = edges_path
                ctx.paths["swx_bb_segmentation_png"] = png1
                ctx.paths["swx_bb_segmentation_pdf"] = pdf1
                ctx.paths["swx_proxy_gaps_png"] = png2
                ctx.paths["swx_proxy_gaps_pdf"] = pdf2
                ctx.paths["swx_gaps_summary_txt"] = gaps_txt
        
                # Optional: warn if expected sidecars are missing on cache-hit
                for k in ("swx_bb_edges_npy", "swx_bb_segmentation_png", "swx_bb_segmentation_pdf"):
                    p = ctx.paths.get(k)
                    if p and (not os.path.exists(p)):
                        print(f"[BaseFits:cache] WARNING: expected SWX sidecar missing: {k} -> {p}")
        
            spec1 = policy.basefit_spec(which="f1", ctx=ctx)
            cache_paths["f1"] = spec1.pkl_path
            ctx.paths["basefit_f1_pkl"] = spec1.pkl_path
        
            is_swx_bb = (str(sw_cfg.model).upper() == "SWX") and (str(sw_cfg.conjunction_anchor).lower() == "bb")
        
            # -------------------------
            # Cache-hit path
            # -------------------------
            if do_cache:
                ok, _detail = io.have(spec1)
                if ok:
                    try:
                        print(f"[BaseFits:cache] Loading f1 from {spec1.pkl_path}")
                        f1_cached = io.load_fitter(spec1)
        
                        # If SWX+bb, repopulate ctx.paths with expected sidecar filenames
                        if is_swx_bb:
                            _register_swx_bb_sidecars()
        
                        return f1_cached
                    except Exception as e:
                        print(f"[BaseFits:cache] f1 load failed ({type(e).__name__}): {e}. Recomputing.")
        
            # -------------------------
            # Cache-miss / recompute path
            # -------------------------
            print(f"[BaseFits] Refitting f1 (Const DM + {sw_cfg.model})...  .    .       .")
        
            # Step 1: remove DM params
            const_par_text = self.prepare_par_constant_dm(str(par_file))
        
            # Step 2: insert solar wind into par text
            # Guard against wrong bb_kwargs keywords being passed
            allowed = {
                "ne1au",
                "ephem",
                "min_theta_deg",
                "include_Re",
                "gap_threshold_days",
                "trim_days",
                "p0",
                "mindata",
                "mintime",
                "maxtime",
            }
            bb_kwargs = dict(sw_cfg.swx_kwargs or {})
            bb_kwargs = {k: v for k, v in bb_kwargs.items() if k in allowed}
        
            sw_par_text, sw_payload = self.insert_solar_wind(
                par_text_or_path=StringIO(const_par_text),
                tim_file=str(tim_file),
                sw_model=sw_cfg.model,
                ne1au=sw_cfg.ne1au,
                swp=sw_cfg.swp,
                conjunction_anchor=sw_cfg.conjunction_anchor,
                swx_bin_interval_days=sw_cfg.swx_bin_interval_days,
                bb_kwargs=bb_kwargs,  # IMPORTANT: pass filtered dict
                return_payload=True,
            )
        
            # Only present for SWX + bb (write side products/diagnostic plots)
            pipe = sw_payload.get("swx_bb_pipeline", None)
            if pipe is not None:
                #  Segmentation fig + edges artifact 
                stem1 = f"{psr}_swx_bb_segmentation"
                png1 = paths.fig_path(stem1, ".png")
                pdf1 = paths.fig_path(stem1, ".pdf")
                edges_path = paths.artifact_path(f"{psr}_swx_bb_edges", ".npy")
        
                def _save_edges() -> None:
                    np.save(edges_path, np.asarray(pipe["SW_BB_refined"], dtype=float))
        
                handle_diagnostics(
                    out=out_cfg,
                    paths=paths,
                    make_fig=lambda: plot_swx_bb_diagnostics(pipe, style=out_cfg.plot),
                    save_targets={"png": png1, "pdf": pdf1},
                    extra_savers=[_save_edges],
                    close=True,
                )
        
                # Gaps fig + text summary artifact 
                stem2 = f"{psr}_swx_proxy_gaps"
                png2 = paths.fig_path(stem2, ".png")
                pdf2 = paths.fig_path(stem2, ".pdf")
                gaps_txt = paths.artifact_path(f"{psr}_swx_gaps_summary", ".txt")
        
                def _save_gap_summary() -> None:
                    txt = format_gap_summary(gaps=pipe["sw_gaps"], mjds=np.asarray(pipe["mjd_comb"], float))
                    with open(gaps_txt, "w", encoding="utf-8") as f:
                        f.write(txt + "\n")
        
                handle_diagnostics(
                    out=out_cfg,
                    paths=paths,
                    make_fig=lambda: plot_data_gaps_diagnostics(
                        mjds=np.asarray(pipe["mjd_comb"], float),
                        resids=np.asarray(pipe["dm_comb"], float),
                        gaps=pipe["sw_gaps"],
                        mask=np.asarray(pipe.get("gap_mask", np.ones_like(pipe["mjd_comb"], bool)), bool),
                        y_label=r"DM$_{SW}$ [pc cm$^{-3}$]",
                        title="SW proxy gaps and mask",
                        style=out_cfg.plot,
                    ),
                    save_targets={"png": png2, "pdf": pdf2},
                    extra_savers=[_save_gap_summary],
                    close=True,
                )
        
                # Register sidecars consistently (avoid manual duplication)
                _register_swx_bb_sidecars()
        
            # Step 3: DM freezes - freeze global DM and any DMn (so SW carries these variations)
            dm_freezes = [re.compile(r"DM$"), re.compile(r"DM\d+$")]
        
            f1_local = self.fit_model(
                StringIO(sw_par_text),
                str(tim_file),
                fitter_cls=base_cfg.fitter_cls,
                freeze_DM_other=dm_freezes,
                maxiter=base_cfg.maxiter_f1,
            )
        
            if do_cache:
                io.save_fitter(
                    f1_local,
                    spec1,
                    write_par=False,
                    meta_json=dict(
                        product="basefit_f1",
                        pulsar=psr,
                        solarWindModel=str(sw_cfg.model),
                        n_toas=int(len(f1_local.toas.table)),
                        did_refit=True,
                    ),
                )
                print("[BaseFits:cache] ✓ f1 pickled")
        
            return f1_local

        # -------------------------
        # Execute requested products (f0, f1, or both)
        # -------------------------
        if "f0" in want:
            f0 = _load_or_refit_f0()
            ctx.products["basefit_f0"] = f0

        if "f1" in want:
            f1 = _load_or_refit_f1()
            ctx.products["basefit_f1"] = f1

        # -------------------------
        # Update RunContext with active outputs (precedence f1 > f0)
        # -------------------------
        active = f1 if f1 is not None else f0
        if active is not None:
            ctx.fitter = active
            ctx.model = active.model
            ctx.toas = active.toas

        # Keep ctx coherent
        ctx.pulsar_name = psr
        ctx.par_file = str(par_file)
        ctx.tim_file = str(tim_file)

        return f0, f1, cache_paths

# =============================================================================
# SolarWindProxy
# =============================================================================

class SolarWindProxy:
    """
    Solar-wind proxy + SW BBX segmentation utilities.
    """
    
    def __init__(self, *, bbx: "BBX", pickler: Optional["PicklerBundle"] = None) -> None:
        self.bbx = bbx
        #self.pickler = pickler
        
    # -------------------------------------------------------------------------
    # SW DM proxy: spherical 1/r^2 DM_SW(t)
    # -------------------------------------------------------------------------

    def solar_wind_dm_proxy(
        self,
        mjd: Union[np.ndarray, Sequence[float], float],   # epochs (MJD) e.g. pint.toa.TOAs.get_mjds()
        pulsar_icrs: SkyCoord,                            # Pulsar sky coord e.g. model.get_psr_coords()
        *,
        ne1au: float = 7.9,            # cm^-3 at 1 AU
        ephem: str = "DE440",
        min_theta_deg: float = 0.25,   # clip around conjunction to avoid blow up
        include_Re: bool = True,       # include 1/R_earth(t) factor (exact integral)
        return_geometry: bool = True,  # If False, return just dm_sw; If True, return dict of all params
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Solar-wind DM proxy from the spherical 1/r^2 model.

        DM_SW(t) = ne1au * (AU_in_pc [/ R_earth(t)]) * ((pi - theta)/sin(theta)),
          where theta is the Sun–pulsar elongation at time t [radians].

        IMPORTANT NOTE: This proxy is independent of any timing fit or SW model.
        It only uses: (a) the epochs 'mjd', (b) the pulsar sky direction
        'pulsar_icrs' (ICRS), and (c) the chosen solar-system ephemeris.

        Parameters
        ----------
        mjd : array-like or float
            Epochs in MJD. (Converted to TDB internally for SSB geometry.)
        pulsar_icrs : SkyCoord
            Pulsar direction in ICRS (e.g., from model.get_psr_coords()).
        ne1au : float
            Electron density at 1 AU [cm^-3].
        ephem : str
            Solar-system ephemeris to use (e.g., "DE440", "builtin").
        min_theta_deg : float
            Minimum elongation (deg) to avoid singularity at conjunction.
        include_Re : bool
            If True, include the exact 1/R_earth(t) factor; if False, assume 1 AU.
        return_geometry : bool
            If True, return a dict with DM and geometry terms; else return DM array.

        Returns
        -------
        dm_sw : ndarray [pc cm^-3] or dict
            DM series aligned to 'mjd', or a dict with keys:
            {"dm_sw", "theta_rad", "R_earth_AU", "geom_factor"}.
        """
        mjd_arr = np.atleast_1d(np.asarray(mjd, dtype=float))
        t = Time(mjd_arr, format="mjd", scale="tdb")

        AU_IN_PC = (1.0 * u.AU).to_value(u.pc)  # ~4.848e-6

        # Get Sun -> Earth barycentric vector (AU) from ephemeris
        with solar_system_ephemeris.set(ephem):
            epos, _ = get_body_barycentric_posvel("earth", t)
            spos, _ = get_body_barycentric_posvel("sun",   t)

        # Earth–Sun barycentric vector in AU; used for elongation geometry.
        Re = np.vstack(
            [
                (epos.x - spos.x).to_value(u.au),
                (epos.y - spos.y).to_value(u.au),
                (epos.z - spos.z).to_value(u.au),
            ]
        ).T
        Re_norm = np.linalg.norm(Re, axis=1)  # AU

        # Pulsar LOS unit vector (ICRS/BCRS)
        khat = pulsar_icrs.cartesian.get_xyz().value
        khat /= np.linalg.norm(khat)

        # Elongation angle theta between Sun and pulsar as seen from Earth
        # cos(theta) = -(Re \dot khat) / |Re| (incoming ray along -khat)
        cos_term = -np.einsum("ij,j->i", Re, khat) / Re_norm
        cos_term = np.clip(cos_term, -1.0, 1.0)
        theta = np.arccos(cos_term)

        # Guard against singularity at conjunction; make a floor to hit
        theta = np.maximum(theta, np.deg2rad(min_theta_deg))

        geom = (np.pi - theta) / np.sin(theta)  # = 1/sinc(1 - theta/pi)

        # Scale with AU->pc, and optionally incl. 1/R_earth(t)
        scale = AU_IN_PC / Re_norm if include_Re else AU_IN_PC
        dm_sw = ne1au * scale * geom  # pc cm^-3

        if return_geometry:
            return {
                "dm_sw": dm_sw,
                "theta_rad": theta,
                "R_earth_AU": Re_norm,
                "geom_factor": geom,
            }
        return dm_sw

    # -------------------------------------------------------------------------
    # Convert DM series to dispersive delay using PINT convention
    # -------------------------------------------------------------------------

    def sw_delay_from_dm_pint(
        self,
        dm: Union[np.ndarray, Dict[str, np.ndarray]],
        freqs_mhz: np.ndarray,
        *,
        out: str = "us",
    ) -> np.ndarray:
        """
        Convert DM series (pc cm^-3) to dispersive time delay at each frequency.
        Uses K = 4.148808e3 s MHz^2 / (pc cm^-3).

        Parameters
        ----------
        dm : ndarray or dict
            DM series [pc cm^-3]. If dict, must contain "dm_sw".
        freqs_mhz : ndarray
            Frequencies [MHz] aligned to dm.
        out : {'s','ms','us'}
            Output time units.

        Returns
        -------
        dt : ndarray
            Dispersive delay in requested units, aligned to dm/freqs.
        """
        # Check inputs
        if isinstance(dm, dict):
            if "dm_sw" not in dm:
                raise KeyError("dm dict missing key 'dm_sw'.")
            dm_arr = np.asarray(dm["dm_sw"], dtype=float)
        else:
            dm_arr = np.asarray(dm, dtype=float)

        f_MHz = np.asarray(freqs_mhz, dtype=float)
        if f_MHz.shape != dm_arr.shape:
            # broadcast if possible, else raise
            try:
                dm_arr, f_MHz = np.broadcast_arrays(dm_arr, f_MHz)
            except ValueError as e:
                raise ValueError("dm and freqs_mhz must be broadcastable to the same shape.") from e

        # Convert to time delays
        K_sec = 4.148808e3  # seconds * MHz^2 per (pc cm^-3)
        dt_sec = K_sec * dm_arr / (f_MHz**2)

        if out == "s":
            return dt_sec
        if out == "ms":
            return dt_sec * 1e3
        if out == "us":
            return dt_sec * 1e6
        raise ValueError("out must be one of: 's', 'ms', or 'us'.")

    # -------------------------------------------------------------------------
    # Bin constraint utilities on generic series (SW proxy BB bin refinement)
    # -------------------------------------------------------------------------

    def _adjust_bins_on_series(
        self,
        tbreak: Union[List[float], np.ndarray, u.Quantity, Time],
        series_mjd: np.ndarray,
        *,
        mindata: int = 8,
        mintime: u.Quantity = 0.8 * u.d,
        maxtime: u.Quantity = 80.0 * u.d,
    ) -> Tuple[List[float], np.ndarray]:
        """
        Enforce bin constraints for a generic time series.
        Generic version of adjust_dmx().
        
        Enforces:
          * minimum bin width (mintime)
          * maximum bin width (maxtime)
          * minimum number of data points per bin (mindata)

        Parameters
        ----------
        tbreak : list/ndarray/Quantity/Time
            Candidate bin edges.
        series_mjd : np.ndarray
            Time stamps (MJD) of the data series used for counts.
        mindata : int
            Minimum number of samples required per bin.
        mintime : astropy.units.Quantity
            Minimum allowed bin width.
        maxtime : astropy.units.Quantity
            Maximum allowed bin width.

        Returns
        -------
        tbreak : list[float]
            New break times (float MJD), sorted.
        n : np.ndarray
            Number of samples in each final interval.
        """
        # Standardize inputs
        if isinstance(tbreak, Time):
            tbreak_vals = tbreak.mjd
        elif isinstance(tbreak, u.Quantity):
            tbreak_vals = tbreak.to_value(u.d)
        else:
            tbreak_vals = tbreak

        edges = sorted([float(x) for x in np.atleast_1d(tbreak_vals).tolist()])
        series_mjd = np.asarray(series_mjd, dtype=float)

        if len(edges) < 2:
            raise ValueError("tbreak must contain at least two edges.")
        if series_mjd.size == 0:
            raise ValueError("series_mjd is empty.")

        # Convenience
        mint = float(mintime.to_value(u.d))
        maxt = float(maxtime.to_value(u.d))

        # Merge bins that are too short in time
        dt = np.diff(np.array(edges))
        while np.any(dt < mint):
            badindex = int(np.where(dt < mint)[0][0])
            del edges[badindex + 1]       # remove right edge -> merge with neighbor
            dt = np.diff(np.array(edges)) # recompute
            if len(edges) < 2:
                raise RuntimeError("All bins collapsed while enforcing mintime.")

        # Counts after initial merge
        dt = np.diff(np.array(edges))
        n = np.histogram(series_mjd, bins=edges)[0]

        # Split bins that are too long (but only if they already have enough data)
        while np.any((dt > maxt + mint) & (n >= mindata)):
            badindex = int(np.where((dt > maxt + mint) & (n >= mindata))[0][0])

            # Insert evenly spaced splits every 'maxt' days, leaving >= mint on both ends
            left = edges[badindex]
            right = edges[badindex + 1]
            for tsplit in np.arange(left + maxt, right - mint, maxt):
                edges.append(float(tsplit))

            edges = sorted(edges)
            dt = np.diff(np.array(edges))
            n = np.histogram(series_mjd, bins=edges)[0]

        # Merge bins with too few samples (may exceed maxtime if necessary)
        n = np.histogram(series_mjd, bins=edges)[0]
        while np.any(n < mindata):
            badindex = int(np.where(n < mindata)[0][0])
            del edges[badindex + 1]  # remove right edge -> merge
            if len(edges) < 2:
                raise RuntimeError("All bins collapsed while enforcing mindata.")
            n = np.histogram(series_mjd, bins=edges)[0]

        return edges, n

    def refine_edges_between_gaps_series(
        self,
        *,
        series_mjd: np.ndarray,
        gaps: List[Tuple[float, float]],
        tbreak: Union[List[float], np.ndarray, u.Quantity, Time],
        eps: float = 0.0,  # small inclusion fuzz for edge tests if necessary
        mindata: int = 2,
        mintime: u.Quantity = 1.0 * u.d,
        maxtime: u.Quantity = 80.0 * u.d,
    ) -> np.ndarray:
        """
        Refine a set of BB edges by enforcing constraints only within
        each contiguous data slice between gaps, using a generic time series
        (e.g., solar-wind DM proxy sample times) for counts. Gaps are excluded.

        Parameters
        ----------
        series_mjd : np.ndarray
            Time stamps (MJD) of the proxy/data series used for counting per bin.
        gaps : list of (float, float)
            Gap intervals in MJD to exclude from data segmentation.
        tbreak : list/ndarray/Quantity/Time
            Global BB edges (e.g., from BB on the full proxy series).
        eps : float
            Tiny tolerance for including global edges that lie on a slice boundary.
        mindata, mintime, maxtime
            Constraints passed through to `_adjust_bins_on_series(...)`.

        Returns
        -------
        np.ndarray
            Sorted, unique array of refined edges that cover only data slices;
            no bin corresponds to any gap.
        """
        # Standardize input edges to float MJD
        if isinstance(tbreak, Time):
            raw = tbreak.mjd
        elif isinstance(tbreak, u.Quantity):
            raw = tbreak.to_value(u.d)
        else:
            raw = np.atleast_1d(tbreak)
        global_edges = np.array(raw, dtype=float)

        # Data span from the series itself
        series_mjd = np.asarray(series_mjd, dtype=float)
        if series_mjd.size == 0:
            raise ValueError("series_mjd is empty.")
        mjd0, mjdN = float(series_mjd.min()), float(series_mjd.max())

        # Build bounds: [data_start, g0, g1, g2, g3, ..., data_end]
        bounds: List[float] = [mjd0]
        for g0, g1 in sorted(gaps, key=lambda x: x[0]):
            # Skip gaps entirely outside the data span
            if g1 <= mjd0 or g0 >= mjdN:
                continue
            # Clip to span and keep non-degenerate
            a, b = max(float(g0), mjd0), min(float(g1), mjdN)
            if b > a:
                bounds.extend([a, b])
        bounds.append(mjdN)
        bounds_arr = np.array(bounds, dtype=float)

        if len(bounds_arr) % 2 != 0:
            raise RuntimeError("Internal error: uneven number of segment boundary endpoints.")

        # For each [start, stop] data slice, refine edges against the series
        all_edges: List[float] = []
        for start, stop in bounds_arr.reshape(-1, 2):
            # Clip the global edges to this slice and force true boundaries
            inside = (global_edges >= start - eps) & (global_edges <= stop + eps)
            seg_edges = np.unique(np.concatenate([[start], global_edges[inside], [stop]]))

            # Subset the series to this slice; skip if no data
            mask = (series_mjd >= start) & (series_mjd < stop)
            if not np.any(mask):
                continue
            sub_t = series_mjd[mask]

            # Enforce constraints within this slice using the series times
            refined, _ = self._adjust_bins_on_series(
                seg_edges,
                sub_t,
                mindata=mindata,
                mintime=mintime,
                maxtime=maxtime,
            )

            # Safety clamp and collect
            refined_arr = np.clip(np.asarray(refined, dtype=float), start, stop)
            all_edges.extend(refined_arr.tolist())

        # Finally sort unique list of edges over data slices only
        edges = np.array(sorted(set(all_edges)), dtype=float)
        return edges

    # -------------------------------------------------------------------------
    # Conjunction / elongation helpers
    # -------------------------------------------------------------------------

    def compute_elongation_minima(
        self,
        *,
        toas: pint.toa.TOAs,
        model: TimingModel,
        step_days: float = 0.1,
    ) -> Tuple[np.ndarray, u.Quantity, List[float]]:
        """
        Compute solar elongation angle from pulsar coordinates and
        find minima (conjunction) through the pulsar's mjd span.
        """
        # Grab pulsar mjd span
        pulsar_mjds = toas.get_mjds()
        mjd_array = (
            np.arange(pulsar_mjds.min().value, pulsar_mjds.max().value, float(step_days)) * u.d
        )  # Create fine grid to estimate elongation on
        times = Time(mjd_array, format="mjd")

        # Get all elongations in pulsar mjd span
        elongation = get_sun(times).separation(model.get_psr_coords())

        # Find time and elongation (angular separation btwn sun and pulsar) at opposition
        # e.g. approximate times when the pulsar is closest to the Sun (conjunction)
        # If span < 1 yr, just take the global minimum
        if (pulsar_mjds.max() - pulsar_mjds.min()) < (365.25 * u.d):
            min_idx = int(np.argmin(elongation))
            min_mjds_quantity = mjd_array[min_idx : min_idx + 1]  # single value as array
            min_elongations = [float(elongation[min_idx].value)]
        else:
            # If span is multi-year, find one minimum per year
            # Bin opposition-filtered MJDs by year - need to select just 1 point per year, the true minimum
            bin_edges = np.arange(mjd_array.min().value, mjd_array.max().value + 365.25, 365.25)
            bin_indices = np.digitize(mjd_array.value, bins=bin_edges) - 1

            # Storage for one minimum per year
            min_mjds: List[u.Quantity] = []
            min_elongations: List[float] = []

            for b in np.unique(bin_indices):
                mask = bin_indices == b
                if np.any(mask):  # bin has candidates
                    elong_bin = elongation[mask]
                    min_idx = int(np.argmin(elong_bin))  # closest approach to the Sun
                    min_mjds.append(mjd_array[mask][min_idx])
                    min_elongations.append(float(elong_bin[min_idx].value))

            min_mjds_quantity = u.Quantity(min_mjds, u.day)

        # Convert MJD to year
        min_years = Time(min_mjds_quantity, format="mjd").datetime.astype("datetime64[Y]").astype(int) + 1970

        return np.asarray(min_years, dtype=int), min_mjds_quantity, min_elongations

    # -------------------------------------------------------------------------
    # SWX BB segmentation from proxy (main computation)
    # -------------------------------------------------------------------------

    def build_swx_bb_edges_from_proxy(
        self,
        *,
        toas: pint.toa.TOAs,
        model: pint.models.timing_model.TimingModel,
        ne1au: float = 7.9,
        ephem: str = "DE440",
        min_theta_deg: float = 0.25,
        include_Re: bool = True,
        gap_threshold_days: float = 200.0,
        trim_days: float = 1.0,
        p0: float = 0.14,
        mindata: int = 8,
        mintime: u.Quantity = 0.1 * u.d,
        maxtime: u.Quantity = 365.25 * u.d,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Build SWX BB segmentation from the solar-wind DM proxy and return:
          - dm_dict: solar-wind proxy dict with 'sigma_dm' added
          - pipeline_dict: intermediate arrays (incl. SW_BB_refined)

        Returns
        -------
        dm_dict : dict
            Keys include 'dm_sw' plus appended 'sigma_dm'.
        pipeline_dict : dict
            Contains MJD arrays, combined series, gaps, BB edges, and settings.

        """
        # Make SW proxy aligned with the TOAs
        pulsar_icrs = model.get_psr_coords()
        mjd = toas.get_mjds().value

        dm_dict = self.solar_wind_dm_proxy(
            mjd,
            pulsar_icrs,
            ne1au=ne1au,
            ephem=ephem,
            min_theta_deg=min_theta_deg,
            include_Re=include_Re,
            return_geometry=True,
        )
        dm_sw = dm_dict["dm_sw"]

        # Timing error -> DM error
        K_sec = 4.148808e3  # s * MHz^2 / (pc cm^-3)
        f_MHz = np.asarray(toas.table["freq"], dtype=float)            # MHz
        sig_t_s = np.asarray(toas.table["error"], dtype=float) * 1e-6  # seconds
        sigma_dm = (f_MHz**2 / K_sec) * sig_t_s                        # pc cm^-3
        dm_dict["sigma_dm"] = sigma_dm

        # -------------------------
        # Execute BBX subpipeline for SWX
        # -------------------------
        
        # Combine duplicates in DM space for BB func
        mjd_comb, dm_comb, sigdm_comb = combine_repeated_toas(mjd, dm_sw, sigma_dm)

        # Find obs gaps
        sw_gaps, gap_mask = self.bbx.find_data_gaps(
            mjd_comb,
            dm_comb,
            gap_threshold=gap_threshold_days,
            trim_days=trim_days,
        )

        # Make initial BB segmentation on proxy
        SW_BB = bayesian_blocks(
            mjd_comb,
            dm_comb,
            sigdm_comb,
            fitness="measures",
            p0=p0,
        )

        # Refine the segmentation within slices
        SW_BB_refined = self.refine_edges_between_gaps_series(
            series_mjd=mjd_comb,
            gaps=sw_gaps,
            tbreak=SW_BB,
            eps=0.0,
            mindata=mindata,
            mintime=mintime,
            maxtime=maxtime,
        )

        # Extract pulsar
        try:
            psr_param = getattr(model, "PSR", None)
            if psr_param is not None and hasattr(psr_param, "value") and psr_param.value:
                pulsar_name = psr_param.value
            else:
                pulsar_name = "Unknown"
        except Exception:
            pulsar_name = "Unknown"

        # Collect pipeline arrays for eval
        pipeline_dict: Dict[str, Any] = {
            "mjd": mjd,
            "f_MHz": f_MHz,
            "sig_t_s": sig_t_s,
            "sigma_dm": sigma_dm,
            "dm_sw": dm_sw,
            "mjd_comb": mjd_comb,
            "dm_comb": dm_comb,
            "sigdm_comb": sigdm_comb,
            "sw_gaps": sw_gaps,
            "gap_mask": gap_mask,
            "SW_BB": SW_BB,
            "SW_BB_refined": SW_BB_refined,
            "n_refined_bins": int(len(SW_BB_refined) - 1),
            "gap_threshold_days": float(gap_threshold_days),
            "trim_days": float(trim_days),
            "p0": float(p0),
            "mindata": int(mindata),
            "mintime": mintime,
            "maxtime": maxtime,
            "pulsar_name": pulsar_name,
        }

        return dm_dict, pipeline_dict


# =============================================================================
# BBX Proxy/Intermediate Containers
# =============================================================================

@dataclass(frozen=True)
class ProxySeries:
    """
    The BB input time series: t, y, yerr (the products of proxy building).
    """
    name: str                       # "chromatic" | "dmx" | "residuals"
    t: np.ndarray                   # shape (N,); time axis
    y: np.ndarray                   # shape (N,); data
    yerr: np.ndarray                # shape (N,); data err
    meta: Mapping[str, Any] = None  # lightweight origin tracking (no huge arrays)

    def validate(self) -> None:
        t = np.asarray(self.t, float)
        y = np.asarray(self.y, float)
        e = np.asarray(self.yerr, float)
        
        if t.ndim != 1 or y.ndim != 1 or e.ndim != 1:
            raise ValueError("ProxySeries t/y/yerr must be 1D arrays.")
        if not (t.size == y.size == e.size):
            raise ValueError("ProxySeries t/y/yerr must have the same length.")
        if t.size < 2:
            raise ValueError("ProxySeries must have at least 2 samples.")

        # Sanity checks:
        if not np.all(np.isfinite(t)):
            raise ValueError("ProxySeries.t contains non-finite values.")
        # Avoid NaNs
        if not np.all(np.isfinite(y)):
            raise ValueError("ProxySeries.y contains non-finite values.")
        if not np.all(np.isfinite(e)):
            raise ValueError("ProxySeries.yerr contains non-finite values.")

@dataclass(frozen=True)
class DiagnosticsPayload:
    """
    Standard payload for diagnostic plotting wrappers (plot bundle).
    """
    kind: str                         # e.g. "chromatic_suite", "dmx_dmseries", "data_gaps"
    data: Mapping[str, Any]           # only what plotters need (may include arrays)
    meta: Mapping[str, Any] = field(default_factory=dict)    # captions/labels/context
    
@dataclass
class ProxyBuildResult:
    """
    Result of building a proxy (internal return type from DispersionMeasureProxy). 
    
    Output of building a proxy:
      - `series`: the BB input time series (ProxySeries)
      - `extras`: additional intermediate objects (optional)
      - `diag`: a standardized diagnostics payload (optional)
    """
    series: ProxySeries                         
    extras: Mapping[str, Any] = field(default_factory=dict)            
    diag: Optional["DiagnosticsPayload"] = None 

@dataclass(frozen=True)
class DMXGapSliceDiagnostics:
    """Per-slice diagnostic payload for DMX segmentation-by-slice plot."""
    slice_index: int
    start: float
    stop: float
    # edges used in that slice for BB-style blocks (start + in-slice BB edges + stop)
    seg_edges: np.ndarray   
    seg_mjds: np.ndarray    # MJDs within slice
    seg_resids: np.ndarray  # residuals within slice (same indexing as seg_mjds)


@dataclass(frozen=True)
class DMXGapAdjustDiagnostics:
    """Diagnostics payload for DMX gap-aware edge adjustment."""
    mjd0: float
    mjdN: float
    bounds: np.ndarray             # shape (2*n_slices,), alternating [start0, stop0, start1, stop1, ...]
    slices: List[DMXGapSliceDiagnostics]
    global_edges: np.ndarray       # the standardized global edges used as input

# =============================================================================
# DispersionMeasureProxy
# =============================================================================

class DispersionMeasureProxy:
    def __init__(self, pickler: Optional[PicklerBundle] = None) -> None:
        self.pickler = pickler

    # -------------------------------------------------------------------------
    # Proxy orchestrator
    # -------------------------------------------------------------------------

    def build_proxy(
        self,
        *,
        cfg: ProxyConfig,
        mjds: Optional[np.ndarray] = None,
        scaled_resid: Optional[np.ndarray] = None,
        scaled_err: Optional[np.ndarray] = None,
        freqs: Optional[np.ndarray] = None,
        model: Optional[TimingModel] = None,
        toas: Optional[pint.toa.TOAs] = None,
        dmxparse_func: Optional[Callable[[Any], Dict[str, Any]]] = None,
        fitter_cls: Optional[type] = None,
        pulsar_name: Optional[str] = None,
    ) -> ProxyBuildResult:
        """
        Proxy builder switch/router: 
         - Chromatic DM Proxy:
         - DMX DM Proxy:
         - Residual DM Proxy: 
        """
        cfg.validate()

        # Validate only what the chosen path needs
        self._validate_runtime_inputs(
            cfg=cfg,
            mjds=mjds if cfg.signal_source != "dmx" else None,
            resid=scaled_resid if cfg.signal_source != "dmx" else None,
            resid_err=scaled_err if cfg.signal_source != "dmx" else None,
            freqs=freqs if cfg.signal_source == "chromatic" else None,
            model=model if cfg.signal_source == "dmx" else None,
            toas=toas if cfg.signal_source == "dmx" else None,
            dmxparse_func=dmxparse_func if cfg.signal_source == "dmx" else None,
        )
    
        # DMX path: ignore arrays entirely
        if cfg.signal_source == "dmx":
            if cfg.signal_source == "dmx":
                if model is None or toas is None or dmxparse_func is None:
                    raise RuntimeError("Internal error: DMX inputs validated but became None.")
                return self._build_proxy_dmx(
                    cfg=cfg,
                    model=model,        
                    toas=toas,
                    dmxparse_func=dmxparse_func,
                    fitter_cls=fitter_cls,
                    pulsar_name=pulsar_name,
                )
    
        # Non-DMX: arrays exist and have been validated
        mjds = np.asarray(mjds, float)
        scaled_resid = np.asarray(scaled_resid, float)
        scaled_err = np.asarray(scaled_err, float)

        if not (mjds.shape == scaled_resid.shape == scaled_err.shape):
            raise ValueError("mjds, scaled_resid, scaled_err must have identical shape.")

        if cfg.signal_source == "chromatic":
            if freqs is None:
                raise ValueError("Chromatic proxy requires freqs array.")
            freqs = np.asarray(freqs, float)
            if freqs.shape != mjds.shape:
                raise ValueError("freqs must have the same shape as mjds.")
            return self._build_proxy_chromatic(
                cfg=cfg,
                mjds=mjds,
                freqs=freqs,
                resid=scaled_resid,
                resid_err=scaled_err,
                pulsar_name=pulsar_name,
            )

        # Residuals DM Proxy: r/\nu² weighted residuals as proxy directly
        return self._build_proxy_residuals(
            cfg=cfg,
            mjds=mjds,
            resid=scaled_resid,
            resid_err=scaled_err,
            pulsar_name=pulsar_name,
        )

    # -------------------------------------------------------------------------
    # Residual proxy builder
    # -------------------------------------------------------------------------

    def _build_proxy_residuals(
        self,
        *,
        cfg: ProxyConfig,
        mjds: np.ndarray,
        resid: np.ndarray,
        resid_err: np.ndarray,
        pulsar_name: Optional[str],
    ) -> ProxyBuildResult:
        # Combine repeats for BB safety
        t, y, yerr = combine_repeated_toas(mjds, resid, resid_err)

        series = ProxySeries(
            name="residuals",
            t=t,
            y=y,
            yerr=yerr,
            meta=dict(
                pulsar_name=pulsar_name,
                signal_source="residuals",
            ),
        )
        series.validate()

        diag = DiagnosticsPayload(
            kind="proxy_residuals",
            data=dict(mjds=t, y=y, yerr=yerr),
            meta=dict(title="Residual proxy series", pulsar_name=pulsar_name),
        )
        return ProxyBuildResult(series=series, extras={}, diag=diag)

    # -------------------------------------------------------------------------
    # Chromatic proxy builder (epochwise WLS)
    # -------------------------------------------------------------------------

    def _build_proxy_chromatic(
        self,
        *,
        cfg: ProxyConfig,
        mjds: np.ndarray,
        freqs: np.ndarray,
        resid: np.ndarray,
        resid_err: np.ndarray,
        pulsar_name: Optional[str],
    ) -> ProxyBuildResult:
        out = self.build_chromatic_achromatic_series(
            mjd=mjds,
            freq=freqs,
            resid=resid,
            resid_err=resid_err,
            cfg=cfg,
        )

        # Combine repeats for BB safety
        t, y, yerr = combine_repeated_toas(out["epoch_mjd"], out["b_chrom"], out["b_err"])

        series = ProxySeries(
            name="chromatic",
            t=t,
            y=y,
            yerr=yerr,
            meta=dict(
                pulsar_name=pulsar_name,
                signal_source="chromatic",
                epoch_tol_days=float(cfg.epoch_tol_days),
                use_inv_nu2=bool(cfg.use_inv_nu2),
                ref_freq=None if cfg.ref_freq is None else float(cfg.ref_freq),
                min_channels=int(cfg.min_channels),
                min_unique_channels=int(cfg.min_unique_channels),
                min_x_span=float(cfg.min_x_span),
                min_snr_b=float(cfg.min_snr_b),
                clip_resid_outliers=bool(cfg.clip_resid_outliers),
                mad_sigma=int(cfg.mad_sigma),
                normalize_x=bool(cfg.normalize_x),
                normalize_method=str(cfg.normalize_method),
                return_dm_units=bool(cfg.return_dm_units),
            ),
        )
        series.validate()

        # Diagnostics payload (arrays only; BBX chooses wrapper + saving)
        diag = DiagnosticsPayload(
            kind="chromatic_suite",
            data=dict(
                mjd=mjds,
                freq=freqs,
                resid=resid,
                resid_err=resid_err,
                x=out['x'],
                epoch_tol_days=float(cfg.epoch_tol_days),
                groups=out["groups"],
                results=out["results"],
                epoch_mjd=out["epoch_mjd"],
                a_achrom=out["a_achrom"],
                a_err=out["a_err"],
                b_chrom=out["b_chrom"],
                b_err=out["b_err"],
                n_raw=out["n_raw"],
                n_used=out["n_used"],
                metrics_all=out["metrics_all"],
                # also include the BB input series, convenient for wrapper
                bb_t=t,
                bb_y=y,
                bb_yerr=yerr,
            ),
            meta=dict(
                title="Chromatic/achromatic epochwise WLS diagnostics",
                pulsar_name=pulsar_name,
            ),
        )

        extras = dict(
            # keep the pre-combined epoch series for downstream comparisons
            epoch_mjd=out["epoch_mjd"],
            b_chrom=out["b_chrom"],
            b_err=out["b_err"],
        )

        return ProxyBuildResult(series=series, extras=extras, diag=diag)

    # -------------------------------------------------------------------------
    # DMX dmseries proxy builder (finely binned DMX + dmxparse)
    # -------------------------------------------------------------------------

    def _build_proxy_dmx(
        self,
        *,
        cfg: ProxyConfig,
        model: TimingModel,
        toas: pint.toa.TOAs,
        dmxparse_func: Callable[[Any], Dict[str, Any]],
        fitter_cls: Optional[type],
        pulsar_name: Optional[str],
    ) -> ProxyBuildResult:
        dmx_dict = self.build_DMX_dmseries(
            model=model,
            toas=toas,
            bin_days=float(cfg.dmx_bin_days),
            dmxparse_func=dmxparse_func,
            fitter_cls=fitter_cls,
            fitter_maxiter=cfg.fitter_maxiter,
            keep_DM=bool(cfg.keep_DM),
            freeze_DM=bool(cfg.freeze_DM),
        )

        t = np.asarray(dmx_dict["dmxeps"].value, float)
        y = np.asarray(dmx_dict["dmxs"].value, float)
        yerr = np.asarray(dmx_dict["dmx_verrs"].value, float)

        series = ProxySeries(
            name="dmx",
            t=t,
            y=y,
            yerr=yerr,
            meta=dict(
                pulsar_name=pulsar_name,
                signal_source="dmx",
                bin_days=float(cfg.dmx_bin_days),
                keep_DM=bool(cfg.keep_DM),
                freeze_DM=bool(cfg.freeze_DM),
                parser_used=str(dmx_dict.get("parser_used", "dmxparse")),
            ),
        )
        series.validate()

        diag = DiagnosticsPayload(
            kind="dmx_dmseries",
            data=dict(
                edges=np.asarray(dmx_dict["edges"], float),
                counts=np.asarray(dmx_dict["counts"], int),
                dmxeps=t,
                dmxs=y,
                dmx_verrs=yerr,
            ),
            meta=dict(title="Fine DMX dmseries proxy diagnostics", pulsar_name=pulsar_name),
        )

        extras = dict(
            dmx_dict=dmx_dict,
            fitter=dmx_dict.get("fitter", None),
            fitted_model=dmx_dict.get("model", None),
        )

        return ProxyBuildResult(series=series, extras=extras, diag=diag)

    def _validate_runtime_inputs(
        self,
        *,
        cfg: "ProxyConfig",
        # arrays for chromatic/residuals
        mjds: Optional[np.ndarray] = None,
        resid: Optional[np.ndarray] = None,
        resid_err: Optional[np.ndarray] = None,
        freqs: Optional[np.ndarray] = None,
        # objects for dmx
        model: Optional["TimingModel"] = None,
        toas: Optional[pint.toa.TOAs] = None,
        dmxparse_func: Optional[Callable[[Any], Dict[str, Any]]] = None,
    ) -> None:
        """
        Validate runtime inputs required to build the requested proxy.
    
        Policy
        ------
        - chromatic/residuals: require mjds, resid, resid_err (and freqs for chromatic)
        - dmx: require model, toas, dmxparse_func (arrays are ignored)
        """
        src = str(cfg.signal_source).lower()
    
        if src == "dmx":
            if model is None:
                raise ValueError("DMX proxy requires 'model'.")
            if toas is None:
                raise ValueError("DMX proxy requires 'toas'.")
            if dmxparse_func is None:
                raise ValueError("DMX proxy requires 'dmxparse_func'.")
            return
    
        # Non-DMX: require arrays
        if mjds is None or resid is None or resid_err is None:
            raise ValueError(f"{cfg.signal_source!r} proxy requires mjds, resid, resid_err arrays.")
    
        mjds = np.asarray(mjds, float)
        resid = np.asarray(resid, float)
        resid_err = np.asarray(resid_err, float)
    
        if not (mjds.shape == resid.shape == resid_err.shape):
            raise ValueError("Proxy runtime inputs: mjds, resid, resid_err must match shapes.")
        if mjds.ndim != 1:
            raise ValueError("Proxy runtime inputs: arrays must be 1D.")
    
        if src == "chromatic":
            if freqs is None:
                raise ValueError("chromatic proxy requires freqs.")
            freqs = np.asarray(freqs, float)
            if freqs.shape != mjds.shape:
                raise ValueError("chromatic proxy: freqs must match mjds shape.")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _group_epochs(mjd: np.ndarray, *, epoch_tol_days: float) -> list[np.ndarray]:
        order = np.argsort(mjd)
        mjd_sorted = mjd[order]
        gaps = np.where(np.diff(mjd_sorted) > float(epoch_tol_days))[0] + 1
        return np.split(order, gaps)

    @staticmethod
    def _wls_line(
        x: np.ndarray, 
        y: np.ndarray, 
        yerr: Optional[np.ndarray], 
        *, 
        return_metrics: bool = False):
        
        x = np.asarray(x, float)
        y = np.asarray(y, float)

        if yerr is None:
            w = np.ones_like(y)
        else:
            yerr = np.asarray(yerr, float)
            finite_pos = np.isfinite(yerr) & (yerr > 0)
            if not np.any(finite_pos):
                w = np.ones_like(y)
            else:
                floor = np.nanmedian(yerr[finite_pos])
                yerr = np.where(yerr <= 0, floor, yerr)
                w = 1.0 / (yerr**2)

        S0 = np.sum(w)
        S1 = np.sum(w * x)
        S2 = np.sum(w * x * x)
        T0 = np.sum(w * y)
        T1 = np.sum(w * x * y)
        det = S0 * S2 - S1 * S1

        if not np.isfinite(det) or det <= 0:
            a = b = sa = sb = np.nan
            if return_metrics:
                metrics = {
                    "S0": float(S0), "S1": float(S1), "S2": float(S2), "det": float(det),
                    "snr_b": np.nan,
                    "x_span": float(np.nanmax(x) - np.nanmin(x)) if x.size else np.nan,
                    "n": int(x.size),
                }
                return a, b, sa, sb, metrics
            return a, b, sa, sb

        a = (S2 * T0 - S1 * T1) / det
        b = (S0 * T1 - S1 * T0) / det

        dof = max(1, int(len(x) - 2))
        yhat = a + b * x
        s2 = np.sum(w * (y - yhat) ** 2) / dof

        var_a = s2 * (S2 / det)
        var_b = s2 * (S0 / det)
        sa = np.sqrt(var_a) if (var_a >= 0 and np.isfinite(var_a)) else np.nan
        sb = np.sqrt(var_b) if (var_b >= 0 and np.isfinite(var_b)) else np.nan

        if return_metrics:
            with np.errstate(divide="ignore", invalid="ignore"):
                snr_b = np.abs(b) / sb if np.isfinite(sb) and sb != 0 else np.nan
            metrics = {
                "S0": float(S0), "S1": float(S1), "S2": float(S2), "det": float(det),
                "snr_b": float(snr_b) if np.isfinite(snr_b) else np.nan,
                "x_span": float(np.nanmax(x) - np.nanmin(x)) if x.size else np.nan,
                "n": int(x.size),
            }
            return a, b, sa, sb, metrics

        return a, b, sa, sb

    # -------------------------------------------------------------------------
    # Computational Core
    # -------------------------------------------------------------------------

    def build_chromatic_achromatic_series(
        self,
        *,
        mjd: np.ndarray,
        freq: np.ndarray,
        resid: np.ndarray,
        resid_err: Optional[np.ndarray],
        cfg: ProxyConfig,
    ) -> Dict[str, Any]:
        mjd = np.asarray(mjd, float)
        freq = np.asarray(freq, float)
        resid = np.asarray(resid, float)
        resid_err = None if resid_err is None else np.asarray(resid_err, float)

        if cfg.use_inv_nu2:
            x = 1.0 / (freq ** 2)
        else:
            ref = float(cfg.ref_freq) if cfg.ref_freq is not None else float(np.nanmedian(freq))
            x = (ref / freq) ** 2

        groups = self._group_epochs(mjd, epoch_tol_days=float(cfg.epoch_tol_days))

        epoch_mjd: list[float] = []
        a_list: list[float] = []
        b_list: list[float] = []
        sa_list: list[float] = []
        sb_list: list[float] = []
        n_raw_list: list[int] = []
        n_used_list: list[int] = []
        metrics_all: list[Dict[str, Any]] = []
        results: list[Dict[str, Any]] = []

        for epoch_idx, g in enumerate(groups):
            if g.size < int(cfg.min_channels):
                continue

            xg = x[g]
            yg = resid[g]
            eg = None if resid_err is None else resid_err[g]

            if eg is None:
                valid = np.isfinite(xg) & np.isfinite(yg)
            else:
                valid = np.isfinite(xg) & np.isfinite(yg) & np.isfinite(eg) & (eg > 0)

            xv, yv = xg[valid], yg[valid]
            ev = None if eg is None else eg[valid]

            if cfg.clip_resid_outliers and xv.size:
                med = np.nanmedian(yv)
                mad = 1.4826 * np.nanmedian(np.abs(yv - med))
                if np.isfinite(mad) and mad > 0:
                    keep = np.abs(yv - med) <= float(cfg.mad_sigma) * mad
                    xv, yv = xv[keep], yv[keep]
                    if ev is not None:
                        ev = ev[keep]

            if xv.size == 0:
                continue

            if np.unique(xv).size < int(cfg.min_unique_channels):
                continue
            if np.unique(xv).size < 2:
                continue
            raw_span = float(np.nanmax(xv) - np.nanmin(xv))
            if raw_span < float(cfg.min_x_span):
                continue

            if ev is None:
                wv = np.ones_like(yv)
            else:
                floor = np.nanmedian(ev[ev > 0]) if np.any(ev > 0) else np.nanmedian(ev)
                ev = np.where(ev <= 0, floor, ev)
                wv = 1.0 / (ev ** 2)

            if cfg.normalize_x:
                wsum = float(np.sum(wv))
                mu_x = float(np.sum(wv * xv) / wsum) if wsum > 0 else float(np.mean(xv))
                if cfg.normalize_method == "span":
                    sx = float(np.nanmax(xv) - np.nanmin(xv))
                else:
                    var = (np.sum(wv * (xv - mu_x) ** 2) / wsum) if wsum > 0 else float(np.var(xv))
                    sx = float(np.sqrt(var))
                if not np.isfinite(sx) or sx <= 0:
                    sx = 1.0
                x_fit = (xv - mu_x) / sx
            else:
                mu_x, sx = 0.0, 1.0
                x_fit = xv

            n_used = int(x_fit.size)

            a_p, b_p, sa_p, sb_p, metrics = self._wls_line(
                x_fit, yv, None if ev is None else ev, return_metrics=True
            )

            if not np.isfinite(b_p) or not np.isfinite(sb_p) or sb_p <= 0:
                continue

            b = b_p / sx
            sb = sb_p / sx
            a = a_p - b * mu_x
            sa = np.sqrt(max(0.0, (sa_p if np.isfinite(sa_p) else 0.0) ** 2 + (mu_x * sb) ** 2))

            snr_b = abs(b) / sb if (np.isfinite(sb) and sb > 0) else np.nan
            if np.isfinite(snr_b) and (snr_b < float(cfg.min_snr_b)):
                continue

            mid = float(np.mean(mjd[g]))
            epoch_mjd.append(mid)
            a_list.append(float(a)); b_list.append(float(b))
            sa_list.append(float(sa)); sb_list.append(float(sb))
            n_raw_list.append(int(g.size))
            n_used_list.append(int(n_used))

            if isinstance(metrics, dict):
                metrics = dict(metrics)
                metrics["n"] = int(n_used)
                metrics["x_span"] = float(raw_span)
                metrics_all.append(metrics)

            results.append(
                dict(
                    epoch_idx=int(epoch_idx),
                    mid_mjd=float(mid),
                    a=float(a), b=float(b),
                    sa=float(sa), sb=float(sb),
                    n_used=int(n_used),
                    x_span=float(raw_span),
                )
            )

        epoch_mjd = np.asarray(epoch_mjd, float)
        a_achrom = np.asarray(a_list, float)
        a_err = np.asarray(sa_list, float)
        b_chrom = np.asarray(b_list, float)
        b_err = np.asarray(sb_list, float)
        n_raw = np.asarray(n_raw_list, int)
        n_used = np.asarray(n_used_list, int)

        if bool(cfg.return_dm_units):
            K_sec = 4.148808e9  # mus * MHz^2 / (pc cm^-3)
            b_chrom = b_chrom / K_sec
            b_err = b_err / K_sec

        return dict(
            x=x,
            groups=groups,
            results=results,
            metrics_all=metrics_all,
            epoch_mjd=epoch_mjd,
            a_achrom=a_achrom,
            a_err=a_err,
            b_chrom=b_chrom,
            b_err=b_err,
            n_raw=n_raw,
            n_used=n_used,
        )

    def build_DMX_dmseries(
        self,
        *,
        model: TimingModel,
        toas: pint.toa.TOAs,
        bin_days: float,
        dmxparse_func: Callable[[Any], Dict[str, Any]],
        fitter_cls: Optional[type] = None,
        fitter_maxiter: Optional[int] = None,
        keep_DM: bool = True,
        freeze_DM: bool = False,
    ) -> Dict[str, Any]:
        if not (isinstance(bin_days, (int, float)) and float(bin_days) > 0):
            raise ValueError("bin_days must be a positive number (days).")
        if dmxparse_func is None:
            raise ValueError("dmxparse_func must be provided.")

        mjd = toas.get_mjds().value
        lo, hi = float(np.nanmin(mjd)), float(np.nanmax(mjd))
        edges = np.arange(lo, hi + float(bin_days), float(bin_days))
        if edges[-1] < hi:
            edges = np.append(edges, hi)
        edges = np.unique(edges)
        if edges.size < 2:
            raise ValueError("Not enough edges generated; check bin_days and TOA span.")

        work_model = copy.deepcopy(model)
        fine_model, counts = self.replace_dmx_model(
            work_model, 
            toas, 
            edges=edges,
            keep_DM=bool(keep_DM),
            freeze_DM=bool(freeze_DM),
        )

        if fitter_cls is None:
            fitter_cls = pint.fitter.WLSFitter

        fitter = fitter_cls(toas, fine_model)
        if fitter_maxiter is None:
            fitter.fit_toas()
        else:
            fitter.fit_toas(maxiter=int(fitter_maxiter))

        used_parser = dmxparse_func
        try:
            dmx_dict = dmxparse_func(fitter)
        except Exception:
            # FUTURE: slot covariance-free fallback parser here (?)
            raise

        dmx_dict.update(
            dict(
                edges=edges,
                counts=np.asarray(counts),
                model=fitter.model,
                fitter=fitter,
                parser_used=getattr(used_parser, "__name__", str(used_parser)),
                bin_days=float(bin_days),
            )
        )
        return dmx_dict
    

# =============================================================================
# BBX (main BB pipeline)
# =============================================================================

class BBX:
    def __init__(
        self,
        *,
        pickler: Optional["PicklerBundle"] = None,
        dm_proxy: Optional["DispersionMeasureProxy"] = None,
    ) -> None:
        self.pickler = pickler
        self.dm_proxy = dm_proxy

    # -------------------------------------------------------------------------
    # Diagnostics runners
    # -------------------------------------------------------------------------
    
    def run_proxy_diagnostics(
        self,
        *,
        diag: Optional["DiagnosticsPayload"],
        out: "OutputConfig",
        paths: "OutputPaths",
        style: "PlotStyleConfig",
        stem_prefix: str,
    ) -> bool:
        """
        Runner/bridge for proxy-building diagnostics.

        Parameters
        ----------
        diag : DiagnosticsPayload or None
            Data-only diagnostic payload returned by dm_proxy.build_proxy(...).
        out, paths, style
            Output gating + pathing + plotting style configs.
        stem_prefix : str
            A runner-defined prefix for figure stems (keeps names consistent).

        Returns
        -------
        did_any : bool
            True if any diagnostic figures were produced (and optionally saved/shown).
        """
        if diag is None:
            return False

        if diag.kind == "chromatic_suite":
            return self._run_chromatic_suite_diagnostics(
                data=diag.data,
                out=out,
                paths=paths,
                style=style,
                stem_prefix=stem_prefix,
            )

        # FUTURE: "dmx_suite", "residuals_suite", "gap_adjust_suite", etc...
        
        return False

    def _run_chromatic_suite_diagnostics(
        self,
        *,
        data: Mapping[str, Any],
        out: "OutputConfig",
        paths: "OutputPaths",
        style: "PlotStyleConfig",
        stem_prefix: str,
    ) -> bool:
        """
        Create/save/show the full chromatic diagnostics suite as a PDF bundle + optional PNGs.

        Expects `data` keys produced by DispersionMeasureProxy chromatic builder, e.g.:
          - mjd, epoch_tol_days, groups (or epoch boundaries)
          - x, resid, resid_err
          - results (per-epoch fit summaries)
          - epoch_mjd, a_achrom, a_err, b_chrom, b_err
          - n_used, metrics_all, fit_conditions, etc.
        """
        # Bundle path (one PDF for the suite)
        pdf_path = paths.artifact_path(f"{stem_prefix}_chromatic_suite", ".pdf")

        figs: List[Tuple[str, Callable[[], Figure]]] = []

        figs.append((
            f"{stem_prefix}_epoch_gap_hist",
            lambda: plot_epoch_gap_histogram(
                mjd=np.asarray(data["mjd"], float),
                epoch_tol_days=float(data["epoch_tol_days"]),
                style=style,
                x_zoom=data.get("hist_x_zoom", None),
                y_zoom=data.get("hist_y_zoom", None),
                title_suffix="  (_group_epochs)",
            ),
        ))

        if data.get("results", None) is not None:
            figs.append((
                f"{stem_prefix}_all_epoch_overlay",
                lambda: plot_all_epoch_fit_overlay(
                    x_all=np.asarray(data["x"], float),
                    y_all=np.asarray(data["resid"], float),
                    yerr_all=None if data.get("resid_err") is None else np.asarray(data["resid_err"], float),
                    mjd=np.asarray(data["mjd"], float),
                    groups=data["groups"],
                    results=data["results"],
                    style=style,
                    color_by=str(data.get("overlay_color_by", "snr")),
                    show_points=bool(data.get("overlay_show_points", True)),
                    show_lines=bool(data.get("overlay_show_lines", True)),
                    max_points_per_epoch=int(data.get("overlay_max_points_per_epoch", 200)),
                    lw=float(data.get("overlay_lw", 1.0)),
                ),
            ))

        figs.append((
            f"{stem_prefix}_points_per_epoch",
            lambda: plot_points_per_epoch_arrays(
                epoch_mjd=np.asarray(data["epoch_mjd"], float),
                n_used=np.asarray(data["n_used"], int),
                style=style,
                title_suffix="  (kept epochs; used in WLS)",
                min_points=int(data.get("min_points", 2)),
                fit_conditions=data.get("fit_conditions", None),
            ),
        ))

        figs.append((
            f"{stem_prefix}_b_snr_vs_time",
            lambda: plot_b_snr_vs_time(
                epoch_mjd=np.asarray(data["epoch_mjd"], float),
                b_chrom=np.asarray(data["b_chrom"], float),
                b_err=np.asarray(data["b_err"], float),
                style=style,
            ),
        ))

        figs.append((
            f"{stem_prefix}_a_vs_b",
            lambda: plot_a_vs_b_correlation(
                a_achrom=np.asarray(data["a_achrom"], float),
                b_chrom=np.asarray(data["b_chrom"], float),
                style=style,
            ),
        ))

        figs.append((
            f"{stem_prefix}_a_b_time_scatter",
            lambda: plot_a_b_time_scatter(
                epoch_mjd=np.asarray(data["epoch_mjd"], float),
                a_achrom=np.asarray(data["a_achrom"], float),
                b_chrom=np.asarray(data["b_chrom"], float),
                style=style,
                y_label=str(data.get("summary_y_label", "Coefficient value")),
                title=str(data.get("summary_title", "Per-epoch achromatic (a) and chromatic (b) terms")),
                legend=bool(data.get("summary_legend", True)),
            ),
        ))

        if data.get("metrics_all", None):
            figs.append((
                f"{stem_prefix}_wls_snr_summary",
                lambda: plot_wls_epoch_summaries(
                    metrics_list=list(data["metrics_all"]),
                    style=style,
                )[0],
            ))
            figs.append((
                f"{stem_prefix}_wls_condition_summary",
                lambda: plot_wls_epoch_summaries(
                    metrics_list=list(data["metrics_all"]),
                    style=style,
                )[1],
            ))

        # Main gated save/show/close logic
        return handle_diagnostics_multi(
            out=out,
            paths=paths,
            fig_factories=figs,       # list of (stem, make_fig)
            pdf_bundle_path=pdf_path,         
            close=True,
            savefig_kwargs={"dpi": style.dpi},
        )
    
    def diag_plot_data_gaps(
        self,
        *,
        gaps: Sequence[Tuple[float, float]],
        mask: np.ndarray,
        mjds: np.ndarray,
        resids: np.ndarray,
        out: "OutputConfig",
        paths: "OutputPaths",
        stem: str = "data_gaps_diagnostics",
        title: str = "Identified Data Gaps and Data Mask",
    ) -> bool:
        """
        Thin wrapper that wires style into plotters and delegates I/O to handler (`handle_diagnostics`).
        """
        def _make():
            return plot_data_gaps_diagnostics(
                mjds=mjds,
                resids=resids,
                gaps=gaps,
                mask=mask,
                style=out.plot,
                title=title,
            )

        return handle_diagnostics(
            out=out,
            paths=paths,
            make_fig=_make,
            save_targets={stem: paths.fig_path(stem, ext=".png")} if out.save_figures else None,
            close=True,
        )

    def diag_plot_gap_adjust_slices(
        self,
        *,
        diag: Optional["DMXGapAdjustDiagnostics"],
        out: "OutputConfig",
        paths: "OutputPaths",
        stem_prefix: str,
    ) -> bool:
        """
        Produce the 'segmentation-by-slice' diagnostic figure from adjust_dmx_edges_for_gaps diagnostics.
        """
        if diag is None:
            return False

        def _make():
            return plot_dmx_segmentation_by_slice_diagnostics(diag, style=out.plot)

        stem = f"{stem_prefix}_dmx_gap_adjust_slices"
        return handle_diagnostics(
            out=out,
            paths=paths,
            make_fig=_make,
            save_targets={stem: paths.fig_path(stem, ext=".png")} if out.save_figures else None,
            close=True,
        )

    def diag_plot_bchrom_vs_dmx(
        self,
        *,
        epoch_mjd: np.ndarray,
        b_chrom: np.ndarray,
        fitter: Any,
        out: "OutputConfig",
        paths: "OutputPaths",
        stem_prefix: str,
    ) -> bool:
        """
        Plot b_chrom(t) vs DMX(t) extracted from the post-refit fitter/model.
        """
        def _make():
            return plot_bchrom_vs_dmx(epoch_mjd, b_chrom, fitter, style=out.plot)

        stem = f"{stem_prefix}_bchrom_vs_dmx"
        return handle_diagnostics(
            out=out,
            paths=paths,
            make_fig=_make,
            save_targets={stem: paths.fig_path(stem, ext=".png")} if out.save_figures else None,
            close=True,
        )

    # -------------------------------------------------------------------------
    # Computational Core: gap finder + gap-adjust 
    # -------------------------------------------------------------------------

    def find_data_gaps(
        self,
        mjds: np.ndarray,
        resids: np.ndarray,
        *,
        gap_threshold: float = 100.0,
        trim_days: float = 1.0,
    ) -> Tuple[List[Tuple[float, float]], np.ndarray]:
        """
        Data-only gap finder.

        Notes
        -----
        This allows:
          - gaps are between consecutive sorted MJDs where dt > gap_threshold
          - optionally shrink edges by trim_days if there are TOAs in the raw gap
          - accept gap only if there are no finite residuals inside the proposed gap
          - Returns gaps and a boolean mask (True=keep, False=gap-masked)
        """
        mjds = np.asarray(mjds, dtype=float)
        resids = np.asarray(resids, dtype=float)

        if mjds.size == 0:
            return [], np.ones_like(mjds, dtype=bool)

        order = np.argsort(mjds)
        sorted_mjds = mjds[order]
        dt = np.diff(sorted_mjds)

        gap_indices = np.where(dt > float(gap_threshold))[0]

        gaps: List[Tuple[float, float]] = []
        mask = np.ones_like(mjds, dtype=bool)

        for idx in gap_indices:
            raw_start = float(sorted_mjds[idx])
            raw_stop = float(sorted_mjds[idx + 1])

            in_gap_mask = (mjds >= raw_start) & (mjds < raw_stop)
            toas_in_gap = int(np.sum(in_gap_mask))

            if toas_in_gap > 0:
                start = raw_start + float(trim_days)
                stop = raw_stop - float(trim_days)
            else:
                start, stop = raw_start, raw_stop

            if stop <= start:
                continue

            in_shrunk = (mjds >= start) & (mjds < stop)
            if not np.any(np.isfinite(resids[in_shrunk])):
                gaps.append((start, stop))
                mask &= ~in_shrunk

        return gaps, mask
        
    def adjust_dmx(
        self,
        tbreak: Union[list, np.ndarray, u.Quantity, Time],
        toas: pint.toa.TOAs,
        mintoas: int = 8,
        mintime: u.Quantity = 0.8 * u.d,
        maxtime: u.Quantity = 80 * u.d,
    ):
        """
        Adjust a DMX binning strategy, to allow for minimum number of TOAs and min/max time for ea. bin.

        The order of operations is:
        * Merge any bins with < ``mintime``
        * Split any bins wiht > ``maxtime``
        * Merge any bins with < ``mintoas`` TOAs

        Between each of these the bins are recomputed, so they are not done simultaneously.

        Parameters
        ----------
        tbreak : list | ndarray | Quantity | Time
            Break times / edges.
        toas : pint.toa.TOAs
            TOAs to count per interval.
        mintoas : int
            Minimum number of TOAs to allow
        mintime : astropy.units.Quantity
            Minimum time of bin to allow
        maxtime : astropy.units.Quantity
            Maximum time of bin to allow

        Returns
        -------
        tbreak : list[float]
            New break times (MJD floats, sorted)
        n : ndarray
            Number of TOAs in each interval
        """
        # Standardize inputs
        # If astropy Time convert to mjd
        if isinstance(tbreak, Time):
            tbreak = tbreak.mjd
        # If astropy Quantity covert to floats in days
        if isinstance(tbreak, u.Quantity):
            tbreak = tbreak.to_value(u.d)

        tbreak = sorted(list(np.asarray(tbreak, float)))  # Sort for binning logic later
        dt = np.diff(np.array(tbreak))  # time intervals e.g. bin widths

        # Delete the latter (right) bound of bin if dt violates mintime merging it
        while np.any(dt < mintime.to_value(u.d)):
            badindex = np.where(dt < mintime.to_value(u.d))[0][0]
            del tbreak[badindex + 1]
            dt = np.diff(np.array(tbreak))  # Update after modifying tbreak

        # Recompute widths and counts
        dt = np.diff(np.array(tbreak))
        n = np.histogram(toas.get_mjds().value, bins=tbreak)[0]  # Bin TOAs into current intervals = counts/current bins

        # Split bins that are too large, but have mintoas in them
        while np.any((dt > maxtime.to_value(u.d) + mintime.to_value(u.d)) & (n >= mintoas)):
            # Find first bin that is bigger than maxtime and contains at least mintoas
            # but must be at least mintime left to be split into valid bins
            badindex = np.where(
                (dt > maxtime.to_value(u.d) + mintime.to_value(u.d)) & (n >= mintoas)
            )[0][0]
            # Find new break points in that interval
            for tsplit in np.arange(
                tbreak[badindex] + maxtime.to_value(u.d),      # right bound - at max maxtime away
                tbreak[badindex + 1] - mintime.to_value(u.d),  # left bound  - at min mintime away
                maxtime.to_value(u.d),                         # max width   - at max maxtime
            ):
                tbreak.append(float(tsplit))  # update
            # Sort, recompute intervals, rebin into new dts
            tbreak = sorted(tbreak)
            dt = np.diff(np.array(tbreak))
            n = np.histogram(toas.get_mjds().value, bins=tbreak)[0]

        # Update counts with while loop bin results
        n = np.histogram(toas.get_mjds().value, bins=tbreak)[0]

        # Merge bins with < mintoas
        # NOTE: allows bins to exceed maxtime if there aren't enough toas
        while np.any(n < mintoas):
            badindex = np.where(n < mintoas)[0][0]
            del tbreak[badindex + 1]  # remove latter (right) bound merging it
            n = np.histogram(toas.get_mjds().value, bins=tbreak)[0]  # recompute count

        return tbreak, n

    def adjust_dmx_edges_for_gaps(
        self,
        toas: pint.toa.TOAs,
        gaps: List[Tuple[float, float]],
        tbreak: Union[list, np.ndarray, u.Quantity, Time],
        eps: float,
        mintoas: int,
        mintime: u.Quantity,
        maxtime: u.Quantity,
        *,
        residuals: Optional[np.ndarray] = None,
        return_diagnostics: bool = False,
    ) -> tuple[np.ndarray, Optional[DMXGapAdjustDiagnostics]]:
        """
        Adjust DMX edges within contiguous data segments separated by known gaps.
    
        This routine:
          1) Standardizes the input global edges.
          2) Builds gap-aware segment bounds: [data_start, gap1_start, gap1_stop, ..., data_end].
          3) For each [start, stop] data segment:
               a) clips global edges into the segment (+ eps tolerance),
               b) enforces segment boundaries,
               c) runs adjust_dmx on the segment TOAs to enforce mintoas/mintime/maxtime,
               d) clips refined edges back to [start, stop] and collects.
          4) Re-inserts all gap boundaries so no bin crosses a gap.
          5) Returns sorted unique edges.
    
        Parameters
        ----------
        toas : pint.toa.TOAs
            Full TOAs.
        gaps : list of (float, float)
            Gap intervals (MJD start, MJD stop).
        tbreak : list | np.ndarray | Quantity | Time
            Initial global edges/breakpoints.
        eps : float
            Tolerance for inclusion of edges near boundaries.
        mintoas : int
            Minimum TOAs per bin.
        mintime : Quantity
            Minimum bin width.
        maxtime : Quantity
            Maximum bin width.
        residuals : np.ndarray, optional
            Required only if return_diagnostics=True. Residuals aligned with toas order.
        return_diagnostics : bool
            If True, returns a diagnostics payload suitable for plotting.
    
        Returns
        -------
        edges : np.ndarray
            Sorted, unique DMX edges (MJD floats).
        diag : DMXGapAdjustDiagnostics or None
            Only returned when return_diagnostics=True.
        """
        # standardize global edges to float MJD array
        if isinstance(tbreak, Time):
            raw = tbreak.mjd
        elif isinstance(tbreak, u.Quantity):
            raw = tbreak.to_value(u.d)
        else:
            raw = np.atleast_1d(tbreak)
        global_edges = np.array(raw, dtype=float)
    
        mjds = np.asarray(toas.get_mjds().value, float)
        mjd0, mjdN = float(np.nanmin(mjds)), float(np.nanmax(mjds))
    
        # construct bounds [data_start, gap1_start, gap1_stop, ..., data_end]
        bounds_list: list[float] = [mjd0]
        for g0, g1 in sorted(gaps, key=lambda x: x[0]):
            if g1 <= mjd0 or g0 >= mjdN:
                continue
            bounds_list.extend([max(g0, mjd0), min(g1, mjdN)])
        bounds_list.append(mjdN)
        bounds = np.asarray(bounds_list, float)
    
        if bounds.size % 2 != 0:
            raise ValueError("Internal error: bounds should have even length (start/stop pairs).")
    
        # optional diagnostics prep
        if return_diagnostics:
            if residuals is None:
                raise ValueError("residuals must be provided when return_diagnostics=True.")
            residuals = np.asarray(residuals, float)
            if residuals.shape != mjds.shape:
                raise ValueError("residuals must match toas.get_mjds() shape.")
            diag_slices: list[DMXGapSliceDiagnostics] = []
        else:
            diag_slices = []
    
        # adjust edges per slice
        all_edges: list[float] = []
    
        for i, (start, stop) in enumerate(bounds.reshape(-1, 2)):
            start = float(start)
            stop = float(stop)
    
            in_slice = (global_edges > start - float(eps)) & (global_edges < stop + float(eps))
            seg_edges = np.unique(np.concatenate([[start], global_edges[in_slice], [stop]])).astype(float)
    
            toa_slice = (mjds >= start) & (mjds < stop)
            if not np.any(toa_slice):
                continue
    
            sub_toas = toas[toa_slice]
    
            refined, _ = self.adjust_dmx(
                seg_edges,
                sub_toas,
                mintoas=mintoas,
                mintime=mintime,
                maxtime=maxtime,
            )
    
            refined = np.clip(np.asarray(refined, float), start, stop)
            all_edges.extend(refined.tolist())
    
            if return_diagnostics:
                seg_mjds = mjds[toa_slice]
                seg_resids = residuals[toa_slice]
                diag_slices.append(
                    DMXGapSliceDiagnostics(
                        slice_index=i,
                        start=start,
                        stop=stop,
                        seg_edges=seg_edges,
                        seg_mjds=np.asarray(seg_mjds, float),
                        seg_resids=np.asarray(seg_resids, float),
                    )
                )
    
        # enforce gap boundaries
        for g0, g1 in gaps:
            all_edges.extend([float(g0), float(g1)])
    
        edges = np.array(sorted(set(all_edges)), float)
    
        diag: Optional[DMXGapAdjustDiagnostics]
        if return_diagnostics:
            diag = DMXGapAdjustDiagnostics(
                mjd0=mjd0,
                mjdN=mjdN,
                bounds=bounds,
                slices=diag_slices,
                global_edges=global_edges,
            )
        else:
            diag = None
    
        return edges, diag

    # -------------------------------------------------------------------------
    # Receiver utilities 
    # -------------------------------------------------------------------------

    def by_receiver(self, toas: pint.toa.TOAs, res: np.ndarray, err: np.ndarray):
        """
        Get residuals, residual errors, and MJD ranges per receiver.

        Returns numpy arrays with units attached: u.us, u.us, u.d.
        """
        # Pull receivers from toas table
        rcvrs = np.unique(toas.table["f"])
        # Create masks for each receiver
        rcvr_masks = [toas.table["f"] == c for c in rcvrs]

        # Put masked data into a dictionary for later use
        receiver_dict = {
            receiver: [res[mask] * u.us, err[mask] * u.us, toas[mask].get_mjds()]
            for receiver, mask in zip(rcvrs, rcvr_masks)
        }

        # Return masks to undo sort later
        rcvr_mask_dict = {
            receiver: mask for receiver, mask in zip(rcvrs, rcvr_masks)
        }

        return receiver_dict, rcvr_mask_dict

    def select_receiver_groups(
        self,
        receiver_dict,
        rcvr_mask_dict=None,
        include=None,
        exclude=None,
        return_mask=True,
    ):
        """
        Filter and return groups of receivers and their data from the full receiver dictionary.
        """
        filtered = {}
        combined_mask = None

        for rcvr in receiver_dict:
            # Skip any receiver not in the include list, but only if the include list is provided
            if include is not None and rcvr not in include:  # Whitelist filter
                continue
            # If exclude is None no exclusion, but if it is a list the receiver's are removed even if include is passed
            if exclude is not None and rcvr in exclude:  # Blacklist override
                continue
            # Stores [resids, err, mjd] for each accepted receiver
            filtered[rcvr] = receiver_dict[rcvr]

            # Optional: build a unified boolean mask for all selected receivers
            if return_mask and rcvr_mask_dict is not None:
                if combined_mask is None:
                    # Override None with a copy of first boolean mask
                    combined_mask = rcvr_mask_dict[rcvr].copy()
                else:
                    # merge True values across receivers (still indexed by OG toa.table)
                    combined_mask |= rcvr_mask_dict[rcvr]

        if return_mask:
            return filtered, combined_mask
        return filtered

    def replace_dmx_model(
        self,
        model: TimingModel,
        toas: pint.toa.TOAs,
        *,
        edges: np.ndarray,
        keep_DM: bool = True,
        freeze_DM: bool = False,
    ) -> Tuple[TimingModel, np.ndarray]:
        """
        Replace or initialize a DispersionDMX component in a PINT timing model
        using custom MJD edges — e.g., Bayesian Blocks, regular intervals, or
        any arbitrary edge array defining per-epoch DMX bins.

        Parameters
        ----------
        model : pint.models.timing_model.TimingModel
            The PINT timing model to modify.
        toas : pint.toa.TOAs
            The TOAs used to count points per DMX bin and identify empty intervals.
        edges : array-like
            Strictly increasing MJD boundaries defining DMX intervals.
            Must have length >= 2.
        keep_DM : bool, optional
            Whether to retain the global DispersionDM component (default: True).
            Set False to remove both DM and DMX components before adding a new DMX.
        freeze_DM : bool, optional
            If True and keep_DM=True, freeze the global DM term.

        Returns
        -------
        model : pint.models.timing_model.TimingModel
            Updated timing model with a fresh DispersionDMX component.
        counts : ndarray
            Number of TOAs falling in each surviving DMX interval (aligned with remaining DMX indices).
        """
        # Validate the edges
        edges = np.asarray(edges, float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("[replace_dmx_model] ERROR: `edges` must be a 1D array with at least two entries.")
        if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
            raise ValueError("[replace_dmx_model] ERROR: `edges` must be strictly increasing finite MJDs.")

        # Remove chosen DMX and/or DM components
        removed_components = []
        for cname, comp in list(model.components.items()):
            if isinstance(comp, DispersionDMX) or (not keep_DM and isinstance(comp, DispersionDM)):
                removed_components.append(cname)
                model.remove_component(cname)

        # Sanity check: print removed components
        if removed_components:
            print("[replace_dmx_model] Removed the following DM-related components:")
            for cname in removed_components:
                print(f"  - {cname}")
        else:
            print("[replace_dmx_model] No DM or DMX components found to remove.")

        # Ensure DispersionDM is or is not added
        has_DM = any(isinstance(comp, DispersionDM) for comp in model.components.values())
        if keep_DM and not has_DM:
            print("[replace_dmx_model] Adding DispersionDM...")
            model.add_component(DispersionDM(), force=True)

        if keep_DM:
            dm_par = getattr(model, "DM", None)
            if dm_par is not None:
                dm_par.frozen = bool(freeze_DM)

        if not keep_DM:
            still = [
                cname for cname, comp in model.components.items()
                if isinstance(comp, (DispersionDM, DispersionDMX))
            ]
            if still:
                print("[replace_dmx_model] WARNING: Some DM components still present:", ", ".join(still))

        # Add a fresh DispersionDMX component
        print("[replace_dmx_model] Adding new DMX component...")
        dmx = DispersionDMX()
        model.add_component(dmx, force=True)

        # Add the first DMX bin manually (index 0001)
        model.DMXR1_0001.value = float(edges[0])
        model.DMXR2_0001.value = float(edges[1])
        model.DMX_0001.frozen = False
        model.DMXR1_0001.frozen = True
        model.DMXR2_0001.frozen = True

        # Add the remaining bins (guard for 2-edge case)
        if edges.size > 2:
            _ = model.add_DMX_ranges(
                edges[1:-1],   # all remaining left edges
                edges[2:],     # all remaining right edges
                frozens=False  # Unfrozen = will be fit
            )
        print(f"→ Added {len(edges) - 1} DMX bins.")

        # Freeze empty bins and remove before fit
        frozen = model.find_empty_masks(toas, freeze=True)
        print(f"→ Frozen empty parameters: {len(frozen)} total")
        empty_dmx = [p for p in frozen if p.startswith("DMX_")]
        empty_idx = sorted({int(p.split("_")[-1]) for p in empty_dmx})

        expected_initial = len(edges) - 1

        n_empty = len(empty_idx)
        if n_empty:
            dmx_comp = model.components["DispersionDMX"]
            try:
                dmx_comp.remove_DMX_range(empty_idx)  # batch remove
            except Exception:
                for i in empty_idx:
                    dmx_comp.remove_DMX_range(i)      # 1 by 1 if batch fails

            # Sanity Check: ensure none of the empty indices remain
            remaining = dmx_comp.get_indices()
            still_here = [i for i in empty_idx if i in remaining]
            if still_here:
                print(f"[replace_dmx_model] WARNING: Failed to remove {len(still_here)} DMX bins: {still_here[:10]}...")
            else:
                print(f"→ Removed {n_empty} empty DMX bins (no TOAs).")
        else:
            print("→ No empty DMX bins detected.")

        # Get surviving DMX indices after removal
        mapping = model.get_prefix_mapping("DMX_")
        idx_keep = sorted(mapping.keys())
        n_survive = len(idx_keep)
        removed = expected_initial - n_survive

        if n_survive == 0:
            print(f"[replace_dmx_model] WARNING: All {expected_initial} DMX bins were empty and removed.")
            return model, np.zeros(0, dtype=int)

        # Compute counts aligned with remaining bins
        mjd_vals = toas.get_mjds().value
        r1s = np.array([getattr(model, f"DMXR1_{i:04d}").value for i in idx_keep], float)
        r2s = np.array([getattr(model, f"DMXR2_{i:04d}").value for i in idx_keep], float)
        counts = np.array([(mjd_vals >= r1) & (mjd_vals < r2) for r1, r2 in zip(r1s, r2s)], bool).sum(axis=1)

        # Final summary
        nz = int((counts > 0).sum())
        z = int((counts == 0).sum())
        print(f"→ ✓ Confirmed: {n_survive} DMX bins defined "
              f"(removed {removed} empty; {nz} have TOAs, {z} still empty).")

        return model, counts

    def build_dm_bb_edges_from_proxy(
        self,
        *,
        model: pint.models.timing_model.TimingModel,
        toas: pint.toa.TOAs,
        proxy_cfg: "ProxyConfig",
        bbx_cfg: "BBXConfig",
        pulsar_name: Optional[str] = None,
        fitter_cls: type = pint.fitter.WLSFitter,
        residuals_us: Optional[np.ndarray] = None,
        residual_err_us: Optional[np.ndarray] = None,
        freqs_MHz: Optional[np.ndarray] = None,
        dmxparse_func: Optional[Any] = None,
        allow_temp_fitter: bool = True,
        return_gap_diagnostics: bool = False,
    ) -> Dict[str, Any]:
        """
        Build Bayesian-Blocks DM segmentation edges from a chosen proxy series, without
        modifying the timing model (e.g. no DMX replacement and no refit).
    
        This is the DM analog of `SolarWindProxy.build_swx_bb_edges_from_proxy`: it
        constructs the BB input time series (proxy), runs Bayesian Blocks, enforces global
        DMX-like constraints, identifies observing gaps, and performs gap-aware edge
        refinement. 
        """
    
        proxy_cfg.validate()
        bbx_cfg.validate()

        src = str(proxy_cfg.signal_source).lower()
        psr = pulsar_name or getattr(getattr(model, "PSR", None), "value", None) or "PSR"
        mjds = np.asarray(toas.get_mjds().value, float)
    
        # freqs required for chromatic/residuals 
        need_freqs = src in ("chromatic", "residuals")
        if need_freqs and freqs_MHz is None:
            if "freq" not in toas.table.colnames:
                raise ValueError("Need freqs_MHz or toas.table['freq'] for chromatic/residuals proxy.")
            freqs_MHz = np.asarray(toas.table["freq"], float)
    
        # residuals/errs required for chromatic/residuals 
        if src in ("chromatic", "residuals"):
            if residuals_us is None or residual_err_us is None:
                if not allow_temp_fitter:
                    raise ValueError("Need residuals_us and residual_err_us (or allow_temp_fitter=True).")
                import copy
                f_tmp = fitter_cls(toas, copy.deepcopy(model))
                residuals_us = f_tmp.resids.calc_time_resids().to_value(u.us)
                residual_err_us = f_tmp.resids.get_data_error().to(u.us).value
    
        # shape guards 
        if freqs_MHz is not None and np.asarray(freqs_MHz).shape != mjds.shape:
            raise ValueError("freqs_MHz must have same shape as TOA mjds.")
        if residuals_us is not None and np.asarray(residuals_us).shape != mjds.shape:
            raise ValueError("residuals_us must have same shape as TOA mjds.")
        if residual_err_us is not None and np.asarray(residual_err_us).shape != mjds.shape:
            raise ValueError("residual_err_us must have same shape as TOA mjds.")
    
        # DMX proxy needs dmxparse_func 
        extra = dict(proxy_cfg.extra or {})
        if src == "dmx":
            dmxparse_func = dmxparse_func or extra.get("dmxparse_func", None)
            if dmxparse_func is None:
                raise ValueError("DMX proxy requires dmxparse_func (pass it or set proxy_cfg.extra['dmxparse_func']).")
    
        # scaling convention: residuals proxy = r/nu^2 
        if src == "residuals":
            inv_nu2 = 1.0 / (np.asarray(freqs_MHz, float) ** 2)
            scaled_resid = np.asarray(residuals_us, float) * inv_nu2
            scaled_err = np.asarray(residual_err_us, float) * inv_nu2
            freqs_for_proxy = None
        else:
            scaled_resid = None if residuals_us is None else np.asarray(residuals_us, float)
            scaled_err = None if residual_err_us is None else np.asarray(residual_err_us, float)
            freqs_for_proxy = np.asarray(freqs_MHz, float) if src == "chromatic" else None
        
        # -------------------------
        # Build proxy series
        # -------------------------
        if src == "dmx":
            proxy_result = self.dm_proxy.build_proxy(
                cfg=proxy_cfg,
                model=model,
                toas=toas,
                dmxparse_func=dmxparse_func,
                fitter_cls=fitter_cls,
                pulsar_name=psr,
            )
        else:
            proxy_result = self.dm_proxy.build_proxy(
                cfg=proxy_cfg,
                mjds=mjds,
                scaled_resid=scaled_resid,
                scaled_err=scaled_err,
                freqs=freqs_for_proxy,
                fitter_cls=fitter_cls,
                pulsar_name=psr,
            )
        
        series = proxy_result.series
        series.validate()
        
        mjds_for_bb = np.asarray(series.t, float)
        values_for_bb = np.asarray(series.y, float)
        errs_for_bb = np.asarray(series.yerr, float)

        # -------------------------
        # BB Segmentation
        # -------------------------
        # BB raw
        tBB_raw = bayesian_blocks(
            mjds_for_bb,
            values_for_bb,
            errs_for_bb,
            fitness=str(bbx_cfg.fitness),
            p0=float(bbx_cfg.p0),
        )
        
        # Enforce constraints
        tBB_adj, _ = self.adjust_dmx(
            tBB_raw,
            toas,
            mintoas=int(bbx_cfg.min_toas),
            mintime=bbx_cfg.min_time,
            maxtime=bbx_cfg.max_time,
        )
        tBB_adj = np.asarray(tBB_adj, float)
        
        # gaps
        gaps, gap_mask = self.find_data_gaps(
            mjds_for_bb,
            values_for_bb,
            gap_threshold=float(bbx_cfg.gap_threshold_days),
            trim_days=float(bbx_cfg.trim_days),
        )
        
        # final segmentation
        tBB_final, gap_diag = self.adjust_dmx_edges_for_gaps(
            toas,
            gaps,
            tBB_adj,
            1e-8,
            mintoas=int(bbx_cfg.min_toas),
            mintime=bbx_cfg.min_time,
            maxtime=bbx_cfg.max_time,
            residuals=(np.asarray(residuals_us, float) if return_gap_diagnostics and residuals_us is not None else None),
            return_diagnostics=bool(return_gap_diagnostics),
        )
        tBB_final = np.asarray(tBB_final, float)
        
        pipeline_dict: Dict[str, Any] = {
            "proxy_series": series,
            "proxy_result": proxy_result,
            "mjds_for_bb": mjds_for_bb,
            "values_for_bb": values_for_bb,
            "errs_for_bb": errs_for_bb,
            "tBB_raw": np.asarray(tBB_raw, float),
            "tBB_adj": np.asarray(tBB_adj, float),
            "tBB_final": np.asarray(tBB_final, float),
            "gaps": list(gaps),
            "gap_mask": np.asarray(gap_mask, bool),
            "pulsar_name": psr,
            "signal_source": str(src),
            "proxy_name": str(series.name),
            "proxy_meta": dict(series.meta or {}),
            "proxy_extras": dict(proxy_result.extras or {}),
        }
        if return_gap_diagnostics:
            pipeline_dict["gap_adjust_diag"] = gap_diag
        
        return pipeline_dict

    def fit_BB_pipeline(
        self,
        *,
        ctx: RunContext,
        bbx_cfg: BBXConfig,
        proxy_cfg: ProxyConfig,
        receiver_sel: ReceiverSelection,
        out_cfg: OutputConfig,
        pickle_cfg: PickleConfig,
        par_file: Optional[str] = None,
        tim_file: Optional[str] = None,
        cache: bool = True,
        code_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Stage 2 runner: BBX (Bayesian Blocks -> DMX replacement -> refit).

        Notes
        -----
        - Inputs are taken primarily from RunContext; explicit args are used only as fallbacks.
        - Diagnostics are emitted via handle_diagnostics*() based on OutputConfig.
        - Optional caching is handled via PicklerBundle (PicklerIO + CachePolicy).
    
        Expected RunContext inputs
        --------------------------
        - ctx.model, ctx.toas (or ctx.fitter providing these)
        - ctx.residuals, ctx.residual_err, ctx.mjds (or computed from fitter/toas)
        - ctx.freqs if proxy_cfg.signal_source == "chromatic" or "residuals"
          (fallback: toas.table["freq"] if present)
    
        Returns
        -------
        dict with the active BBX fitter/model/toas, BB edges, proxy metadata, and cached products.
        """
        # Sanity Checks
        if self.pickler is None:
            raise ValueError("BBX.fit_BB_pipeline requires pickler=PicklerBundle.")
        if self.dm_proxy is None:
            raise ValueError("BBX.fit_BB_pipeline requires dm_proxy=DispersionMeasureProxy.")
        
        #  Validate configs
        bbx_cfg.validate()
        proxy_cfg.validate()
        receiver_sel.validate()
        
        #  Required ctx state (Stage 1 must have run, OR ctx must already contain model+toas)
        if ctx.fitter is None and (ctx.model is None or ctx.toas is None):
            raise RuntimeError(
                "BBX.fit_BB_pipeline requires BaseFits to run first (or equivalent). "
                "Expected ctx.fitter or (ctx.model and ctx.toas)."
            )
        
        # Populate ctx.model/toas from ctx.fitter if needed
        if ctx.fitter is not None:
            ctx.model = ctx.model or ctx.fitter.model
            ctx.toas = ctx.toas or ctx.fitter.toas
        
        if ctx.model is None or ctx.toas is None:
            raise RuntimeError("BBX.fit_BB_pipeline requires ctx.model and ctx.toas (or ctx.fitter).")
        
        #  ReceiverSelection consistency check
        if receiver_sel != bbx_cfg.receiver_selection:
            raise ValueError(
                "receiver_sel must match bbx_cfg.receiver_selection. "
                "Pass receiver_sel=cfg.bbx.receiver_selection."
            )
        
        # Resolve runtime inputs from ctx (explicit args are fallbacks)
        ctx.pulsar_name = ctx.pulsar_name or getattr(getattr(ctx.model, "PSR", None), "value", None)
        ctx.par_file = ctx.par_file or (str(par_file) if par_file is not None else None)
        ctx.tim_file = ctx.tim_file or (str(tim_file) if tim_file is not None else None)

        src = str(proxy_cfg.signal_source).lower()
        model = ctx.model
        toas = ctx.toas
        
        # Proxy-specific requirements (dmxparse, freqs)
        extra = dict(proxy_cfg.extra or {})
        dmxparse_func = extra.get("dmxparse_func", None)
        
        if src == "dmx" and dmxparse_func is None:
            raise ValueError(
                "ProxyConfig.extra must include 'dmxparse_func' when signal_source='dmx'. "
                "Example: ProxyConfig(extra={'dmxparse_func': pint.dmxparse.dmxparse})"
            )
        
        # Residuals and errors (ms)
        if ctx.residuals is None and ctx.fitter is not None:
            ctx.residuals = ctx.fitter.resids.calc_time_resids().to_value(u.us)
        if ctx.residual_err is None and ctx.fitter is not None:
            ctx.residual_err = ctx.fitter.resids.get_data_error().to(u.us).value
        if ctx.residuals is None or ctx.residual_err is None:
            raise ValueError("Need ctx.residuals and ctx.residual_err (or ctx.fitter to compute them).")
        
        # MJDs
        if ctx.mjds is None:
            ctx.mjds = toas.get_mjds().value
        if ctx.mjds is None:
            raise ValueError("Need ctx.mjds or ctx.toas to compute MJDs.")
        
        # Freqs (MHz) for chromatic/residuals proxy paths
        if ctx.freqs is None and "freq" in toas.table.colnames:
            ctx.freqs = np.asarray(toas.table["freq"], dtype=float)
        
        need_freqs = src in ("chromatic", "residuals")
        if need_freqs and ctx.freqs is None:
            raise ValueError(
                f"ProxyConfig.signal_source={src!r} requires ctx.freqs or toas.table['freq']."
            )
        
        psr = ctx.pulsar_name or "PSR"

        # -------------------------
        # Fitter class: use BaseFits fitter choice
        # -------------------------
        # Fallback: First, RunConfig then WLSFitter if ctx.fitter is missing.
        FitterClass = (
            type(ctx.fitter)
            if ctx.fitter is not None
            else (getattr(self.pickler.policy.cfg.basefit, "fitter_cls", None) or pint.fitter.WLSFitter)
        )
    
        # -------------------------
        # Pickle/cache spec + fast load cache hit
        # -------------------------
        io = self.pickler.io
        policy = self.pickler.policy
    
        do_cache = bool(pickle_cfg.enabled and cache)
    
        # NOTE: CachePolicy is built from a RunConfig
        # These checks ensure that the cache key/spec aren't being 
        # built from different configs than the run
        if getattr(policy, "cfg", None) is not None:
            if policy.cfg.bbx != bbx_cfg:
                raise ValueError("pickler.policy.cfg.bbx does not match bbx_cfg passed to BBX.fit_BB_pipeline.")
            if policy.cfg.proxy != proxy_cfg:
                raise ValueError("pickler.policy.cfg.proxy does not match proxy_cfg passed to BBX.fit_BB_pipeline.")
            if policy.cfg.out != out_cfg:
                # This mismatch could mean configs are mixed.
                print("[BBX] WARNING: pickler.policy.cfg.out != out_cfg (continuing).")
            if policy.cfg.pkl != pickle_cfg:
                print("[BBX] WARNING: pickler.policy.cfg.pkl != pickle_cfg (continuing).")

        # Build cache spec and register paths into ctx (RunContext)
        spec = policy.bbx_spec(ctx=ctx)
        ctx.paths["bbx_pkl"] = spec.pkl_path
        ctx.paths["bbx_pipe_npz"] = spec.prefix + "_pipe.npz"
        ctx.paths["bbx_proxy_json"] = spec.prefix + "_proxy.json"

        # If the cache contract is satisfied (e.g. previous pickle exsists): load fitter, 
        # load sidecars, set from ctx, rebuild receiver selection from cached fitter (resid/errs), 
        # and return a dictionary of data products.
        if do_cache:
            ok, detail = io.have(spec)
            if ok:
                try:
                    print(f"[BBX:cache] Loading fitter from {spec.pkl_path}")
                    fitter_cached = io.load_fitter(spec)
    
                    pipe = io.load_npz(spec.prefix, "_pipe.npz")
                    proxy_meta = io.load_json(spec.prefix, "_proxy.json")
    
                    # Reconstruct products
                    tBB_raw = np.asarray(pipe["tBB_raw"], float)
                    tBB_adj = np.asarray(pipe["tBB_adj"], float)
                    tBB_final = np.asarray(pipe["tBB_final"], float)
                    gaps_arr = np.asarray(pipe["gaps"], float)
                    mjds_for_bb = np.asarray(pipe["mjds_for_bb"], float)
                    values_for_bb = np.asarray(pipe["values_for_bb"], float)
                    errs_for_bb = np.asarray(pipe["errs_for_bb"], float)
                    Ntoa = np.asarray(pipe["Ntoa"], int)
    
                    # Update ctx active products from cached fitter
                    ctx.fitter = fitter_cached
                    ctx.model = fitter_cached.model
                    ctx.toas = fitter_cached.toas
                    ctx.mjds = ctx.toas.get_mjds().value
                    ctx.residuals = fitter_cached.resids.calc_time_resids().to_value(u.us)
                    ctx.residual_err = fitter_cached.resids.get_data_error().to(u.us).value
                    if "freq" in ctx.toas.table.colnames:
                        ctx.freqs = np.asarray(ctx.toas.table["freq"], float)
    
                    # Receiver regrouping for downstream
                    receiver_dict, _ = self.by_receiver(ctx.toas, ctx.residuals, ctx.residual_err)
                    ctx.receiver_dict = receiver_dict
    
                    out = {
                        "model": ctx.model,
                        "toas": ctx.toas,
                        "mjds": ctx.mjds,
                        "residuals": ctx.residuals,
                        "residual_err": ctx.residual_err,
                        "receiver_dict": receiver_dict,
                        "fitter": ctx.fitter,
                        "BB_edges": tBB_final,
                        "Ntoa_per_bin": Ntoa,
                        "dmx_model": ctx.model,
                        "tBB_raw": tBB_raw,
                        "tBB_adj": tBB_adj,
                        "gaps": gaps_arr,
                        "signal_source": src,
                        "proxy": proxy_meta,
                        "bbx_cache_detail": detail,
                        "bbx_cache_key": spec.key,
                    }
                    ctx.products["bbx"] = out
                    print("[BBX:cache] ✓ cache hit")
                    return out
    
                except Exception as e:
                    print(f"[BBX:cache] LOAD FAILED ({type(e).__name__}): {e}")
                    print("[BBX:cache] Proceeding to recompute + refit.")
    
        # -------------------------
        # Receiver filtering
        # -------------------------
        mjds_all = np.asarray(ctx.mjds, float)
        resid_us_all = np.asarray(ctx.residuals, float)
        err_us_all = np.asarray(ctx.residual_err, float)
    
        rec_dict, rcvr_mask_dict = self.by_receiver(toas, resid_us_all, err_us_all)
        all_receivers = set(rec_dict.keys())
    
        include = receiver_sel.include
        exclude = receiver_sel.exclude
    
        # Validate receiver list contents against actual receivers
        if include is not None:
            invalid = [r for r in include if r not in all_receivers]
            if invalid:
                print(f"[BBX] WARNING: receivers in include not present: {invalid}")
    
        if include is not None and exclude is not None:
            overlap = set(include) & set(exclude)
            if overlap:
                print(f"[BBX] WARNING: receivers in both include and exclude: {sorted(overlap)}")
                print("[BBX] Exclude takes precedence for overlapping receivers.")
    
        if include is not None:
            selected = [r for r in include if (exclude is None or r not in exclude)]
            data_dict, combined_mask = self.select_receiver_groups(
                rec_dict, rcvr_mask_dict, include=selected, return_mask=True
            )
            print(f"[BBX] Receivers used (include \\ exclude): {list(data_dict.keys())}")
        elif exclude is not None:
            selected = [r for r in all_receivers if r not in exclude]
            data_dict, combined_mask = self.select_receiver_groups(
                rec_dict, rcvr_mask_dict, include=selected, return_mask=True
            )
            print(f"[BBX] Receivers used (after exclusion): {list(data_dict.keys())}")
        else:
            data_dict, combined_mask = self.select_receiver_groups(rec_dict, rcvr_mask_dict, return_mask=True)
            print(f"[BBX] Using all receivers: {list(data_dict.keys())}")
    
        if combined_mask is None or not np.any(combined_mask):
            raise ValueError("No TOAs selected after applying receiver masks.")
    
        ctx.masks["bbx_receiver_mask"] = combined_mask
    
        # Subselect aligned arrays/TOAs
        toas_sel = toas[combined_mask]
        mjds_sel = mjds_all[combined_mask]
        resid_us_sel = resid_us_all[combined_mask]
        err_us_sel = err_us_all[combined_mask]
        freqs_sel = None if ctx.freqs is None else np.asarray(ctx.freqs, float)[combined_mask]

        # -------------------------
        # Build DM BB segmentation (proxy -> BB -> constraints -> gaps -> refined edges)
        # -------------------------
        want_gap_adj_diag = bool(out_cfg.diag_on() or out_cfg.save_figures)

        pipe = self.build_dm_bb_edges_from_proxy(
            model=model,
            toas=toas_sel,
            proxy_cfg=proxy_cfg,
            bbx_cfg=bbx_cfg,
            pulsar_name=psr,
            fitter_cls=FitterClass,
            residuals_us=resid_us_sel,
            residual_err_us=err_us_sel,
            freqs_MHz=freqs_sel,
            dmxparse_func=dmxparse_func,
            allow_temp_fitter=False, # only needed if residuals/errs aren't fed in
            return_gap_diagnostics=want_gap_adj_diag,
        )
        
        # Unpack products for downstream usage
        tBB_raw   = np.asarray(pipe["tBB_raw"], float)
        tBB_adj   = np.asarray(pipe["tBB_adj"], float)
        tBB_final = np.asarray(pipe["tBB_final"], float)
        gaps      = list(pipe["gaps"])
        gap_mask  = np.asarray(pipe["gap_mask"], bool)
        
        # Proxy products 
        proxy_series = pipe["proxy_series"]       
        proxy_result = pipe["proxy_result"]
        
        # Record into ctx 
        ctx.products["bbx_proxy_series"] = proxy_series
        ctx.products["bbx_proxy_extras"] = dict(getattr(proxy_result, "extras", {}) or {})
        ctx.products["bbx_gaps"] = gaps
        ctx.masks["bbx_gap_mask_proxy_series"] = gap_mask
        
        # Optional slice diagnostics obj
        gap_adj_diag = pipe.get("gap_adjust_diag", None)

        # Diagnostics
        try:
            self.run_proxy_diagnostics(
                diag=proxy_result.diag,
                out=out_cfg,
                paths=OutputPaths.from_config(policy.cfg),
                style=out_cfg.plot,
                stem_prefix=f"{psr}_bbx",
            )
        except Exception as e:
            print(f"[BBX] WARNING: proxy diagnostics failed ({type(e).__name__}): {e}")
    
        # -------------------------
        # Replace DMX model with final BB edges
        # -------------------------
        work_model = copy.deepcopy(model)
        dmx_model, Ntoa = self.replace_dmx_model(
            work_model,
            toas_sel,
            edges=tBB_final,
            keep_DM=True,
            freeze_DM=False,
        )
    
        # Freeze any params that have no data in this selection (e.g. JUMPs after rcvr filtering).
        frozen_params = dmx_model.find_empty_masks(toas_sel, freeze=True)
        print(f"[BBX] → Frozen empty parameters (non-DMX may be included): {len(frozen_params)}")
    
        leftover_dmx = [p for p in frozen_params if str(p).startswith("DMX_")]
        if leftover_dmx:
            print(f"[BBX] WARNING: {len(leftover_dmx)} DMX_* still frozen after replace_dmx_model: "
                  f"{leftover_dmx[:10]}{' ...' if len(leftover_dmx) > 10 else ''}")
    
        # -------------------------
        # Refit with updated DMX
        # -------------------------
        print("[BBX] Refitting model with updated Bayesian Block DMX binning...  .  .    .")
        fitter = FitterClass(toas_sel, dmx_model)
        fitter.fit_toas()
        print("[BBX] Refit complete!")
    
        # Update residuals/errors post-refit
        residuals_post = fitter.resids.calc_time_resids().to_value(u.us)
        residual_err_post = fitter.resids.get_data_error().to(u.us).value
    
        # Receiver regrouping for downstream plotting
        receiver_dict, _ = self.by_receiver(toas_sel, residuals_post, residual_err_post)
    
        # Post-fit diagnostic: b_chrom vs DMX
        if src == "chromatic":
            try:
                epoch_mjd = np.asarray(proxy_result.extras.get("epoch_mjd", []), float)
                b_chrom = np.asarray(proxy_result.extras.get("b_chrom", []), float)
                if epoch_mjd.size and b_chrom.size:
                    self.diag_plot_bchrom_vs_dmx(
                        epoch_mjd=epoch_mjd,
                        b_chrom=b_chrom,
                        fitter=fitter,
                        out=out_cfg,
                        paths=OutputPaths.from_config(policy.cfg) if getattr(policy, "cfg", None) is not None else OutputPaths.from_config(policy.cfg),
                        stem_prefix=f"{psr}_bbx",
                    )
            except Exception as e:
                print(f"[BBX] WARNING: b_chrom vs DMX diagnostics failed ({type(e).__name__}): {e}")
    
        # -------------------------
        # Cache save: fitter + sidecars
        # -------------------------
        proxy_meta = dict(
            name=str(proxy_series.name),
            t=np.asarray(proxy_series.t, float).tolist(),
            y=np.asarray(proxy_series.y, float).tolist(),
            yerr=np.asarray(proxy_series.yerr, float).tolist(),
            meta=dict(proxy_series.meta or {}),
            signal_source=str(src),
        )
    
        if do_cache:
            io.save_fitter(
                fitter,
                spec,
                write_par=False,
                meta_json=dict(
                    product="bbx",
                    pulsar=psr,
                    signal_source=str(src),
                    n_bins=int(len(tBB_final) - 1),
                    n_toas=int(len(fitter.toas.table)),
                    code_version=code_version or pickle_cfg.code_version,
                ),
            )
            io.save_npz(
                spec.prefix,
                "_pipe.npz",
                tBB_raw=np.asarray(tBB_raw, float),
                tBB_adj=np.asarray(tBB_adj, float),
                tBB_final=np.asarray(tBB_final, float),
                gaps=np.asarray(gaps, float),
                mjds_for_bb = np.asarray(pipe["mjds_for_bb"], float),
                values_for_bb = np.asarray(pipe["values_for_bb"], float),
                errs_for_bb = np.asarray(pipe["errs_for_bb"], float),
                Ntoa=np.asarray(Ntoa, int),
            )
            io.save_json(spec.prefix, "_proxy.json", payload=proxy_meta)
            print("[BBX:cache] ✓ fitter + sidecars saved")
    
        # -------------------------
        # Update RunContext active outputs + products
        # -------------------------
        ctx.fitter = fitter
        ctx.model = fitter.model
        ctx.toas = fitter.toas
        ctx.mjds = ctx.toas.get_mjds().value
        ctx.residuals = residuals_post
        ctx.residual_err = residual_err_post
        if "freq" in ctx.toas.table.colnames:
            ctx.freqs = np.asarray(ctx.toas.table["freq"], float)
        ctx.receiver_dict = receiver_dict
    
        out = {
            "model": ctx.model,
            "toas": ctx.toas,
            "mjds": ctx.mjds,
            "residuals": ctx.residuals,
            "residual_err": ctx.residual_err,
            "receiver_dict": receiver_dict,
            "fitter": ctx.fitter,
            "BB_edges": np.asarray(tBB_final, float),
            "Ntoa_per_bin": np.asarray(Ntoa, int),
            "dmx_model": ctx.model,
            "tBB_raw": np.asarray(tBB_raw, float),
            "tBB_adj": np.asarray(tBB_adj, float),
            "gaps": np.asarray(gaps, float),
            "signal_source": str(src),
            "proxy": proxy_meta,
            "bbx_cache_key": spec.key,
        }
    
        ctx.products["bbx"] = out
        return out


# =============================================================================
# NoiseAnalysis
# =============================================================================

class NoiseAnalysis:
    """
    Stage 3 runner: enterprise-style noise analysis + (optional) refit + caching.

    Notes
    ----- 
    1. Uses PicklerBundle (PicklerIO + CachePolicy) for noise-fit pickle I/O.
    2. Chains live in {base_noise_dir}/{PSR}_nb/ or {PSR}_wb/.
    3. Provides the three modes of loading/fitting:
         - fast reload (pickle + chains)
         - rebuild from chains (no new enterprise run)
         - full run (enterprise chains + attach + refit)
    """

    def __init__(self, *, pickler: Optional["PicklerBundle"] = None) -> None:
        self.pickler = pickler

    def noise_analysis_from_fitter(
        self,
        fo: pint.fitter.Fitter,
        *,
        cfg: "NoiseAnalysisConfig",
        using_wideband: bool = False,
        compare_noise_dir: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        sampler_kwargs: Optional[dict] = None,
        save_corner: bool = False,
        no_corner_plot: bool = True,
        use_noise_point: str = "mean_large_likelihood",
        rn_bf_thres: float = 1e2,
        maxiter: Optional[int] = None,
    ) -> Tuple[pint.fitter.Fitter, TimingModel, Dict[str, Any]]:
        """
        Given an existing PINT fitter, optionally run enterprise noise analysis, attach noise
        parameters to a clean timing model, optionally refit, and optionally save/reload the
        noise-augmented fitter via PicklerIO.

        Parameters
        ----------
        fo : pint.fitter.Fitter
            Existing fitter (e.g., BBX output).
        cfg : NoiseAnalysisConfig
            Controls run/refit/save and base directory placement.
        using_wideband : bool
            Controls chain dir suffix: *_nb vs *_wb.
        compare_noise_dir : str, optional
            Optional alternate chain base dir for nu.analyze_noise / add_noise_to_model.
        model_kwargs, sampler_kwargs : dict, optional
            Passed to nu.model_noise when cfg.run_noise_analysis=True.
        save_corner, no_corner_plot, use_noise_point, rn_bf_thres :
            Passed through to pint_pal.noise_utils routines.
        maxiter : int, optional
            Max iterations for the post-noise timing refit when cfg.do_refit=True.

        Returns
        -------
        fo_noise : pint.fitter.Fitter
            Noise-augmented fitter (possibly loaded from cache).
        mo_noise : TimingModel
            Noise-augmented timing model.
        products : dict
            Lightweight products/paths for downstream use:
              {
                "psr": str,
                "base_noise_dir": str,
                "chaindir": str,
                "chaindir_compare": Optional[str],
                "noise_spec_prefix": str,
                "noise_pkl": str,
                "noise_key": str,
              }
        """
        if self.pickler is None:
            raise ValueError("NoiseAnalysis.noise_analysis_from_fitter requires pickler=PicklerBundle.")

        # -------------------------
        # Resolve basics + paths
        # -------------------------
        psr = fo.model.PSR.value
        base_noise_dir = cfg.base_noise_dir or "./"
        os.makedirs(base_noise_dir, exist_ok=True)

        io = self.pickler.io
        policy = self.pickler.policy

        chain_suffix = "wb" if using_wideband else "nb"
        chaindir = os.path.join(base_noise_dir, f"{psr}_{chain_suffix}/")
        chaindir_compare = (
            os.path.join(compare_noise_dir, f"{psr}_{chain_suffix}/")
            if compare_noise_dir is not None
            else None
        )

        spec = policy.noise_spec(base_noise_dir=base_noise_dir, using_wideband=using_wideband)
        prefix = spec.prefix  # e.g. {base_noise_dir}/{PSR}_noiseFit

        products: Dict[str, Any] = dict(
            psr=psr,
            base_noise_dir=base_noise_dir,
            chaindir=chaindir,
            chaindir_compare=chaindir_compare,
            noise_spec_prefix=prefix,
            noise_pkl=spec.pkl_path,
            noise_key=spec.key,
        )

        # -------------------------------------------------------------------------
        # FAST RELOAD MODE:
        #   cfg.run_noise_analysis=False, cfg.do_refit=False, cfg.save_pickle=False
        # -------------------------------------------------------------------------
        if (not cfg.run_noise_analysis) and (not cfg.do_refit) and (not cfg.save_pickle):
            try:
                ok, detail = io.have(spec)
                if not ok:
                    raise FileNotFoundError(f"Noise cache contract not satisfied: {detail}")

                fo_noise = io.load_fitter(spec)
                mo_noise = fo_noise.model

                # Load chains into a Core object for inspection
                noise_core, noise_dict, rn_bf = nu.analyze_noise(
                    chaindir=chaindir,
                    use_noise_point=use_noise_point,
                    likelihoods_to_average=50,
                    burn_frac=0.25,
                    save_corner=save_corner,
                    no_corner_plot=no_corner_plot,
                    chaindir_compare=chaindir_compare,
                )

                # Write noise json into chaindir
                os.makedirs(chaindir, exist_ok=True)
                with open(os.path.join(chaindir, f"{psr}_noise.json"), "w", encoding="utf-8") as f:
                    json.dump(noise_dict, f, indent=4)

                log.info(f"Reloaded noise-fit from {spec.pkl_path} and chains from {chaindir}")
                products["rn_bf"] = rn_bf
                products["noise_dict_path"] = os.path.join(chaindir, f"{psr}_noise.json")
                return fo_noise, mo_noise, products

            except Exception as e:
                log.warning(
                    f"Failed to fast-reload existing noise-fit from {spec.pkl_path} ({e!r}); "
                    "falling back to rebuilding from chains / rerun."
                )
                # fall through to standard path

        # -------------------------------------------------------------------------
        # STANDARD PATH:
        #   (optional) run enterprise -> add noise from chains -> fitter -> 
        #   (optional) refit -> (optional) save
        # -------------------------------------------------------------------------
        mo = fo.model
        to = fo.toas

        # Step 1: Clean model - strip any existing noise components
        mo_clean = copy.deepcopy(mo)
        lu.remove_noise(mo_clean)

        # Step 2: Optionally run new enterprise noise analysis
        if cfg.run_noise_analysis:
            nu.model_noise(
                mo_clean,
                to,
                using_wideband=using_wideband,
                resume=False,
                run_noise_analysis=True,
                base_op_dir=base_noise_dir,
                model_kwargs=model_kwargs or {},
                sampler_kwargs=sampler_kwargs or {},
            )

        # Step 3: Attach noise parameters from existing chains to the clean model
        mo_noise, noise_core = nu.add_noise_to_model(
            mo_clean,
            use_noise_point=use_noise_point,
            burn_frac=0.25,
            save_corner=save_corner,
            no_corner_plot=no_corner_plot,
            ignore_red_noise=False,
            using_wideband=using_wideband,
            rn_bf_thres=float(rn_bf_thres),
            base_dir=base_noise_dir,
            compare_dir=compare_noise_dir,
            return_noise_core=True,
        )

        # Step 4: Build a new fitter (same class as input)
        FitterClass = fo.__class__
        fo_noise = FitterClass(to, mo_noise)

        # Preserve the free-parameter structure from the original fitter
        fo_noise.model.free_params = list(fo.model.free_params)

        # Step 5: Optional refit
        if cfg.do_refit:
            try:
                if maxiter is None:
                    fo_noise.fit_toas()
                else:
                    fo_noise.fit_toas(maxiter=int(maxiter))
                fo_noise.model.CHI2.value = fo_noise.resids.chi2
            except pint.fitter.ConvergenceFailure:
                log.warning("Noise+timing refit failed to converge; best effort used.")
        else:
            log.info("Skipping timing refit (do_refit=False); fo_noise has noise params but no new fit.")

        # Step 6: Optional save (PicklerIO)
        if cfg.save_pickle:
            io.save_fitter(
                fo_noise,
                spec,
                write_par=True,
                meta_json=dict(
                    product="noise",
                    description="Noise-augmented timing solution",
                    use_noise_point=use_noise_point,
                    rn_bf_thres=float(rn_bf_thres),
                    n_toas=int(len(fo_noise.toas.table)),
                    free_params=list(map(str, fo_noise.model.free_params)),
                    base_noise_dir=base_noise_dir,
                    chaindir=chaindir,
                    using_wideband=bool(using_wideband),
                    did_refit=bool(cfg.do_refit),
                ),
            )
            print(f"\n✓ Noise-augmented fitter pickled at:\n  {spec.pkl_path}\n")
        else:
            log.info("save_pickle=False: not writing noise-fit pickle/par to disk.")

        products["noise_core"] = noise_core
        return fo_noise, mo_noise, products
        