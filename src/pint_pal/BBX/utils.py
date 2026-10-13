# utils.py

from __future__ import annotations

import os
import re
import glob
from datetime import datetime
from collections import Counter
from typing import Any, Callable, Iterable, Mapping, Optional, Union, Sequence, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord


# =============================================================================
# Diagnostic plot Utilities
# =============================================================================

def handle_diagnostics(
    *,
    out: "OutputConfig",
    paths: "OutputPaths",
    make_fig: Callable[[], Figure],                   # The diagnostic plotter
    save_targets: Optional[Mapping[str, str]] = None, # e.g. {"png": "/.../a.png", "pdf": "/.../a.pdf"}
    extra_savers: Optional[Iterable[Callable[[], None]]] = None, # Non-fig outputs (e.g. .npy, .txt, etc...)
    close: bool = True,
    savefig_kwargs: Optional[Mapping[str, Any]] = None, # Allow overrides/changes to savefig defaults
) -> bool:
    """
    Centralized handlers for diagnostic plot toggles/saves.

    Diagnostic plots policy this function implements:
      - Gating (diag_on/save_figures) happens here.
      - Plotters MUST build/return Figure only.
      - This function is allowed to show/save/close.
    """
    # Gatekeeper: Set toggles (diag_on = display plots; save_figures = write to disk)
    want = bool(out.diag_on() or out.save_figures)
    if not want:
        return False

    # Build figure
    fig: Optional[Figure] = None
    try:
        fig = make_fig()

        if out.save_figures:
            paths.ensure_fig_dir(enabled=True)

            kw = dict(dpi=int(out.plot.dpi), bbox_inches="tight")
            if savefig_kwargs:
                kw.update(dict(savefig_kwargs))

            # Save figure if toggled
            if save_targets:
                for _k, fpath in save_targets.items():
                    fig.savefig(fpath, **kw)

            # Save extra artifacts
            if extra_savers:
                for saver in extra_savers:
                    saver()

        # Toggle display
        if out.diag_on():
            plt.show()

        return True

    finally: # Always close
        if close and fig is not None:
            plt.close(fig)

def handle_diagnostics_multi(
    *,
    out: "OutputConfig",
    paths: "OutputPaths",
    fig_factories: Sequence[Tuple[str, Callable[[], Figure]]],
    pdf_bundle_path: Optional[str] = None,
    save_individual: bool = True,
    close: bool = True,
    savefig_kwargs: Optional[Mapping[str, Any]] = None,
) -> bool:
    """
    Centralized handler for multiple diagnostic figures.
    
    Diagnostic plots policy this function implements:
     - If neither display nor saving is enabled, do nothing.
     - Otherwise, build figures from pure (only Figure returned) plotter factories, optionally save them,
          optionally show them, and finally close them (unless close=False).
    
    Parameters
    ----------
    out : OutputConfig
        Diagnostics policy toggles (display vs saving) and default plotting settings.
    paths : OutputPaths
        Centralized path constructor and directory manager.
    fig_factories : Sequence[Tuple[str, Callable[[], Figure]]]
        Sequence of (stem, make_fig) pairs.
        - stem: filename stem (runner-defined; no extension)
        - make_fig: zero-argument callable returning a matplotlib Figure 
    pdf_bundle_path : str, optional
        If provided and saving is enabled, all figures are written into a single PDF bundle.
        Parent directories are created as needed.
    save_individual : bool
        If True and saving is enabled, also save each figure individually as PNG using
        paths.fig_path(stem, ".png").
    close : bool
        If True, close all created figures before returning.
    savefig_kwargs : Mapping[str, Any], optional
        Extra kwargs forwarded to fig.savefig(...). Defaults include dpi and bbox_inches.
    
    Returns
    -------
    did_any : bool
        True if any figures were produced (and optionally saved/shown).
    """
    # Gatekeeper: Set toggles (diag_on = display plots; save_figures = write to disk)
    want = bool(out.diag_on() or out.save_figures)
    if not want:
        return False

    # Create fig factory for multiple diagnostic plots
    figs: List[Tuple[str, Figure]] = []
    try:
        figs = []
        for stem, make_fig in fig_factories:
            try:
                fig = make_fig()
            except Exception as e:
                raise RuntimeError(f"Diagnostic figure factory failed for stem={stem!r}") from e
            figs.append((stem, fig))

        kw = dict(dpi=int(out.plot.dpi), bbox_inches="tight")
        if savefig_kwargs:
            kw.update(dict(savefig_kwargs))

        if out.save_figures:
            paths.ensure_fig_dir(enabled=True)

            # Save individual figures
            if save_individual:
                for stem, fig in figs:
                    fig.savefig(paths.fig_path(stem, ext=".png"), **kw)

            # Save bundled PDF (single artifact)
            if pdf_bundle_path:
                os.makedirs(os.path.dirname(pdf_bundle_path) or ".", exist_ok=True)
                with PdfPages(pdf_bundle_path) as pdf:
                    for _stem, fig in figs:
                        pdf.savefig(fig, bbox_inches="tight")

        if out.diag_on():
            plt.show()

        return True

    finally: # always close
        if close:
            for _stem, fig in figs:
                try:
                    plt.close(fig)
                except Exception:
                    pass

def mask_toas_from_gaps(mjd_toa: np.ndarray, gaps) -> np.ndarray:
    """
    Build a TOA-level boolean mask from (start, stop) gap intervals.

    Returns
    -------
    keep_mask : ndarray[bool]
        True for TOAs *kept* (outside gaps), False for TOAs inside any gap.
    """
    mjd_toa = np.asarray(mjd_toa, float)
    in_gap = np.zeros_like(mjd_toa, dtype=bool)
    for (g1, g2) in gaps:
        g1 = float(g1); g2 = float(g2)
        in_gap |= (mjd_toa >= g1) & (mjd_toa <= g2)
    return ~in_gap
                    
# =============================================================================
# Data manipulation utilities
# =============================================================================

def combine_repeated_toas(
    mjds: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine repeated MJDs by averaging y and combining errors in quadrature (RMS).

    Parameters
    ----------
    mjds, y, yerr : ndarray
        Arrays of equal length. mjds may be unsorted.

    Returns
    -------
    t_u, y_u, yerr_u : ndarray
        Unique MJDs and combined values/errors, sorted ascending by MJD.

    Notes
    -----
    - Sorting is internal; output is sorted by MJD.
    - MJDs are treated as exact equality groups.
    """
    mjds = np.asarray(mjds, float)
    y = np.asarray(y, float)
    yerr = np.asarray(yerr, float)

    if not (mjds.shape == y.shape == yerr.shape):
        raise ValueError("[combine_repeated_toas]: mjds, y, yerr must have identical shapes.")
    if mjds.ndim != 1:
        raise ValueError("[combine_repeated_toas]: inputs must be 1D arrays.")

    order = np.argsort(mjds)
    mjd_s = mjds[order]
    y_s = y[order]
    e_s = yerr[order]

    boundaries = np.where(np.diff(mjd_s) != 0)[0] + 1
    idx_groups = np.split(np.arange(mjd_s.size), boundaries)

    out_t, out_y, out_e = [], [], []
    for g in idx_groups:
        out_t.append(float(mjd_s[g[0]]))
        out_y.append(float(np.mean(y_s[g])))
        out_e.append(float(np.sqrt(np.mean(e_s[g] ** 2))))

    return np.asarray(out_t, float), np.asarray(out_y, float), np.asarray(out_e, float)

def remove_gap_nodes(
    basis_nodes: np.ndarray, 
    gap_nodes: np.ndarray, 
    atol: float = 1e-10):
    """
    Remove gap-node entries from a basis-node array by nearest-neighbor matching.
    
    Each gap node is matched to at most one unused basis node within a fixed
    absolute tolerance, and matched basis nodes are removed. Returns the cleaned
    array and the indices of removed entries.
    """
    basis_nodes = np.asarray(basis_nodes, dtype=float)
    gap_nodes = np.asarray(gap_nodes, dtype=float)

    # Track which indices in basis_nodes have already been assigned
    used = np.zeros(len(basis_nodes), dtype=bool)
    matched_idx = []

    for g in gap_nodes:
        d = np.abs(basis_nodes - g)

        # prevent reusing the same node for multiple gap nodes
        d[used] = np.inf

        i = np.argmin(d)
        if d[i] <= atol:
            matched_idx.append(i)
            used[i] = True

    matched_idx = np.array(sorted(matched_idx), dtype=int)
    cleaned = np.delete(basis_nodes, matched_idx)

    return cleaned, matched_idx

# =============================================================================
# Time / coordinate utilities
# =============================================================================

def days_in_year(yr: int) -> int:
    """Days of the year accounting for leaps"""
    return 366 if (yr % 4 == 0 and (yr % 100 != 0 or yr % 400 == 0)) else 365

def mjd_to_year(mjd: Union[float, np.ndarray]) -> np.ndarray:
    """Convert MJD float(s) to decimal year(s)."""
    mjd = np.asarray(mjd, float)
    t = Time(mjd, format="mjd")
    y = t.datetime
    years = np.array([dt.year for dt in y], float)
    doy = np.array([dt.timetuple().tm_yday for dt in y], float)
    secs = np.array(
        [dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6 for dt in y],
        float,
    )

    diy = np.array([days_in_year(int(yr)) for yr in years], float)
    return years + (doy - 1.0 + secs / 86400.0) / diy


def year_to_mjd(year: Union[float, np.ndarray]) -> np.ndarray:
    """Convert decimal year(s) to MJD float(s)."""
    year = np.asarray(year, float)
    y_int = np.floor(year).astype(int)
    frac = year - y_int

    diy = np.array([days_in_year(int(yr)) for yr in y_int], float)
    t0 = Time([f"{int(yr)}-01-01T00:00:00" for yr in y_int], format="isot", scale="utc")
    return (t0.mjd + frac * diy).astype(float)


def _parse_dec_ddmm(dec_str: str) -> float:
    """
    Parse Dec substring from pulsar name:
      - 'DDMM' -> degrees + minutes/60
      - 'DMM'  -> degrees + minutes/60
      - 'DD'   -> degrees only
    Returns dec degrees (absolute value) as float.
    """
    s = str(dec_str).strip()
    if len(s) == 4:               # DDMM
        d = int(s[:2])
        m = int(s[2:])
        return d + m / 60.0
    if len(s) == 3:               # DMM
        d = int(s[0])
        m = int(s[1:])
        return d + m / 60.0
    if len(s) == 2:               # DD
        return float(int(s))
    return float(int(s))          # conservative fallback


def pulsar_name_to_elat(pname: str) -> float:
    """
    Convert pulsar name (J/B) to ecliptic latitude (degrees).
    - J-names are J2000 (ICRS): JHHMM+/-DDMM
    - B-names are B1950 (FK4):  BHHMM+/-DD (sometimes +/-DDMM)
    
    Returns: barycentric true ecliptic latitude in degrees (rough estimate).
    """
    _pulsar_re = re.compile(r"^([JB])(\d{2})(\d{2})([+-])(\d{2,4})$")
    m = _pulsar_re.match(str(pname).strip())
    if not m:
        raise ValueError(f"Unrecognized pulsar name format: {pname}")

    prefix, hh, mm, sign, dd = m.groups()

    # RA in degrees from HH MM (no seconds in the name)
    ra_deg = (int(hh) + int(mm) / 60.0) * 15.0
    # Dec degrees from DD or DDMM
    dec_abs = _parse_dec_ddmm(dd)
    dec_deg = dec_abs if sign == "+" else -dec_abs

    if prefix == "J":
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    else:
        fk4 = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="fk4", equinox="B1950")
        coord = fk4.icrs

    return float(coord.barycentrictrueecliptic.lat.to_value(u.deg))


# =============================================================================
# Inspector (file + TOA lookup utilities)
# =============================================================================

def find_recent_file(path, pName, fileEnding):
    """
    Find the most recent file (e.g. par or tim) in `path` with `fileEnding` for pulsar `pName` 

    Parameters
    ----------
    path : string 
        ex: "/minish/mt00095/ng20/results/" or "~/ng20/results"
    pName : string
        ex: "J00330+0451"
    fileEnding : string
        ex: ".par", "nb.tim", "*.par", "*tim", etc...
    Returns
    -------
    found_file : string
        A string with the filepath and filename of most recent file found in "path" for "pName"
        ex: '/minish/mt00095/ng20/results/J0030+0451_PINT_20250604.nb.par'
    
    """
    # Expand ~ to full home directory
    path = os.path.expanduser(path)
    if not path.endswith("/"):
        path += "/"

    # Standardize the file ending
    if fileEnding.startswith("*"):
        ending = fileEnding               # e.g. "*.par",  "*tim"
    elif fileEnding.startswith("."):
        ending = f"*{fileEnding}"         # e.g. ".par" -> "*.par"
    else:
        ending = f"*{fileEnding}"         # e.g. "nb.tim" -> "*nb.tim"

    # Build pattern
    pattern = os.path.join(path, f"{pName}{ending}")
    files = glob.glob(pattern)

    # Extract dates and find the latest
    dated_files = [
        (f, datetime.strptime(m.group(), "%Y%m%d"))
        for f in files if (m := re.search(r"\d{8}", f))
    ]
    if dated_files:
        latest_file = max(dated_files, key=lambda x: x[1])
        found_file = latest_file[0]
        print("Most recent file found:", found_file)
    else:
        print("No file found with pattern:", pattern)
        found_file = None
    return found_file


def find_toa_by_mjd(toas: Any, target_mjd: float):
    """
    Find the closest TOA MJD to `target_mjd`, returning the subset within the
    computed minimum separation and their associated filenames.
    """
    mjds = toas.get_mjds().value
    diffs = np.abs(mjds - float(target_mjd))
    min_diff = float(np.min(diffs))
    closest_mjd = float(mjds[np.argmin(diffs)])
    print(f"Closest MJD: {closest_mjd:.6f}, difference: {min_diff:.3f}")

    tto = toas[np.isclose(toas.get_mjds().value, closest_mjd, atol=min_diff)]
    table = tto.table
    filenames = [d.get("name", "UNKNOWN") for d in table["flags"]]
    return table, filenames


def by_mjd_table(toas: Any, target_mjd: float) -> None:
    """Print a summary of TOAs at closest `target_mjd` by file + median error per file."""
    toa_table, filenames = find_toa_by_mjd(toas, target_mjd)
    file_counts = Counter(filenames)

    file_errors = {fn: [] for fn in file_counts}
    for row, fname in zip(toa_table, filenames):
        file_errors[fname].append(row["error"])

    print(f"{'Filename':<50} {'Occurrences':>12} {'Median Error (us)':>18}")
    print("-" * 85)
    for fname in file_counts:
        count = file_counts[fname]
        med_err = np.nanmedian(file_errors[fname])
        print(f"{fname:<50} {count:>12} {med_err:>18.3f}")


def find_toa_by_dmx(dmxparse_dict: Mapping[str, Any], toas: Any) -> None:
    """
    Identify the DMX bin with the largest DMX uncertainty and print the TOA file
    for the closest TOA epoch.
    """
    verrs = np.asarray(dmxparse_dict["dmx_verrs"])
    max_idx = int(np.nanargmax(verrs))

    eps = dmxparse_dict["dmxeps"][max_idx]
    big_err_dmx_date = float(getattr(eps, "value", eps))

    print(f"Largest DMX error at DMX epoch MJD: {big_err_dmx_date:.6f}")
    print(f"Max DMX uncertainty index: {max_idx}\n")

    _toa_table, filenames = find_toa_by_mjd(toas, big_err_dmx_date)
    file_counts = Counter(filenames)

    print(f"{'Filenames':<50} {'Occurrences':>10}")
    print("-" * 62)
    for file, count in file_counts.items():
        print(f"{file:<50} {count:>10}")

# =============================================================================
# Formatting helpers
# =============================================================================

def format_gap_summary(
    *,
    gaps: Sequence[Tuple[float, float]],
    mjds: np.ndarray,
) -> str:
    """
    Formatter: return a string table for logging/debug. No printing.
    """
    mjds = np.asarray(mjds, dtype=float)

    lines = []
    lines.append("Summary of Accepted Gaps:")
    lines.append("{:<5} {:>12} {:>12} {:>14} {:>10}".format("Gap#", "Start MJD", "Stop MJD", "Span (days)", "N TOAs"))
    lines.append("-" * 61)

    for i, (start, stop) in enumerate(gaps):
        n_toas = int(np.sum((mjds >= start) & (mjds < stop)))
        span = float(stop - start)
        lines.append("{:<5} {:>12.1f} {:>12.1f} {:>14.1f} {:>10}".format(i + 1, start, stop, span, n_toas))

    return "\n".join(lines)

# =============================================================================
# SW DM Proxy utilities
# =============================================================================

# -------------------------------------------------------------------------
# File readers for F10.7/L1 
# -------------------------------------------------------------------------

# F10.7 file ingestion
def read_f107_flux_file(
    path: str, 
    use_adjusted: bool = True
) -> pd.DataFrame:
    """
    Read NOAA F10.7 flux file and return a cleaned dataframe.

    Parameters
    ----------
    path : str
        Path to the F10.7 flux text file. 
        See https://www.spaceweather.gc.ca/forecast-prevision/solar-solaire/solarflux/sx-3-en.php
        
        Important columns:
         - fluxobsflux: obs flux at telescope modulated w/solar activity & dsun-earth 
         - fluxadjflux: annual dsun-earth modulation removed; flux from mean distance @1AU 
         - fluxursi:    adjusted flux multiplied by 0.9, a historical calibration scaling
    use_adjusted : bool, default True
        If True, use fluxadjflux (default). Otherwise use fluxobsflux.

    Returns
    -------
    pd.DataFrame
        Columns:
          - datetime_utc
          - mjd
          - flux
          - fluxobsflux
          - fluxadjflux
          - fluxursi
    """
    # Read fixed-width/whitespace-aligned table
    df = pd.read_fwf(path, skiprows=0)

    # Drop the dashed separator row
    if "fluxdate" in df.columns:
        df = df[df["fluxdate"].astype(str).str.contains(r"\d", regex=True)].copy()

    # Clean columns to strings first
    for col in ["fluxdate", "fluxtime", "fluxobsflux", "fluxadjflux", "fluxursi"]:
        df[col] = df[col].astype(str).str.strip()

    # Convert flux columns to a number
    for col in ["fluxobsflux", "fluxadjflux", "fluxursi"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse date + time columns
    # Example: fluxdate='20041028', fluxtime='170000'
    dt_str = df["fluxdate"] + df["fluxtime"].str.zfill(6)
    dt = pd.to_datetime(dt_str, format="%Y%m%d%H%M%S", utc=True)

    # Convert to MJD
    dt_naive = dt.dt.tz_localize(None) # Remove timezone wrapper for astropy compatibility
    t = Time(dt_naive.to_numpy(dtype="datetime64[ns]"), format="datetime64", scale="utc")
    df["datetime_utc"] = dt
    df["mjd"] = t.mjd

    # Choose the flux column to work with
    flux_col = "fluxadjflux" if use_adjusted else "fluxobsflux"
    df["flux"] = df[flux_col]

    # Keep useful columns only
    df = df[[
        "datetime_utc",
        "mjd",
        "flux",
        "fluxobsflux",
        "fluxadjflux",
        "fluxursi",
    ]].reset_index(drop=True)

    return df

# L1 file ingestion
def read_omni2_l1_file(
    path: str,
    *,
    proton_density_col: int,
    alpha_ratio_col: int,
    year_col: int = 0,
    doy_col: int = 1,
    hour_col: int = 2,
) -> pd.DataFrame:
    """
    Read OMNI2 hourly data from a local whitespace-delimited file.

    Parameters
    ----------
    path : str
        Local path to omni2_all_years.dat
    proton_density_col : int
        Column index for proton density n_p [cm^-3]
    alpha_ratio_col : int
        Column index for alpha/proton density ratio n_alpha/n_p
    year_col, doy_col, hour_col : int
        Column indices for time fields

    Returns
    -------
    pd.DataFrame
        Columns:
          - datetime_utc
          - mjd
          - proton_density
          - alpha_proton_ratio
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        comment="#",
        engine="python",
    )

    out = pd.DataFrame({
        "year": pd.to_numeric(df.iloc[:, year_col], errors="coerce"),
        "doy": pd.to_numeric(df.iloc[:, doy_col], errors="coerce"),
        "hour": pd.to_numeric(df.iloc[:, hour_col], errors="coerce"),
        "proton_density": pd.to_numeric(df.iloc[:, proton_density_col], errors="coerce"),
        "alpha_proton_ratio": pd.to_numeric(df.iloc[:, alpha_ratio_col], errors="coerce"),
    })

    # OMNI fill handling: keep only finite, non-fill physical values
    out.loc[out["proton_density"] >= 999, "proton_density"] = np.nan
    out.loc[out["alpha_proton_ratio"] >= 9.999, "alpha_proton_ratio"] = np.nan

    # Build UTC datetime from year + day-of-year + hour
    base = pd.to_datetime(out["year"].astype("Int64").astype(str), format="%Y", errors="coerce", utc=True)
    dt = base + pd.to_timedelta(out["doy"] - 1, unit="D") + pd.to_timedelta(out["hour"], unit="h")

    dt_naive = dt.dt.tz_localize(None)
    t = Time(dt_naive.to_numpy(dtype="datetime64[ns]"), format="datetime64", scale="utc")

    out["datetime_utc"] = dt
    out["mjd"] = t.mjd

    return out[["datetime_utc", "mjd", "proton_density", "alpha_proton_ratio"]]

# -------------------------------------------------------------------------
# Generic SW DM Proxy expansion helpers
# -------------------------------------------------------------------------

def select_timespan(
    df: pd.DataFrame,
    mjd_min: float,
    mjd_max: float,
    mjd_col: str = "mjd",
) -> pd.DataFrame:
    """
    Select rows within a target MJD span.
    """
    mask = (df[mjd_col] >= mjd_min) & (df[mjd_col] <= mjd_max)
    return df.loc[mask].reset_index(drop=True)

def daily_average_timeseries(
    df: pd.DataFrame,
    *,
    datetime_col: str = "datetime_utc",
    value_cols: list[str],
) -> pd.DataFrame:
    """
    Average one or more value columns to daily cadence.
    """
    out = (
        df.assign(date=df[datetime_col].dt.floor("D"))
          .groupby("date", as_index=False)[value_cols]
          .mean()
    )
    dt_naive = out["date"].dt.tz_localize(None)
    t = Time(dt_naive.to_numpy(dtype="datetime64[ns]"), format="datetime64", scale="utc")
    out["mjd"] = t.mjd
    return out[["date", "mjd"] + value_cols]

def prepare_driver_series(
    driver_daily: pd.DataFrame,
    mjd_target: np.ndarray,
    *,
    value_col: str,
    smooth_days: int = 391,
    normalize: bool = True,
) -> dict:
    """
    Smooth a daily driver series and interpolate it onto target MJDs.

    Returns
    -------
    dict
        {
          "mjd_daily",
          "value_daily",
          "value_smooth",
          "value_interp",
          "driver_norm",
          "center",
          "scale",
        }
    """
    mjd_daily = np.asarray(driver_daily["mjd"], dtype=float)
    value_daily = np.asarray(driver_daily[value_col], dtype=float)
    mjd_target = np.asarray(mjd_target, dtype=float)

    # Sanity checks
    if len(mjd_daily) == 0:
        raise ValueError("driver_daily is empty.")

    if len(mjd_target) == 0:
        raise ValueError("mjd_target is empty.")

    if smooth_days % 2 == 0:
        raise ValueError("`smooth_days` must be odd for centered running median.")

    value_smooth = running_median_1d(value_daily, smooth_days)

    value_interp = np.interp(
        mjd_target,
        mjd_daily,
        value_smooth,
        left=value_smooth[0],
        right=value_smooth[-1],
    )

    if normalize:
        center = np.nanmedian(value_interp)
        mad = np.nanmedian(np.abs(value_interp - center))
        scale = 1.4826 * mad if mad > 0 else np.nanstd(value_interp)
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        driver_norm = (value_interp - center) / scale
    else:
        center = np.nanmedian(value_interp)
        scale = 1.0
        driver_norm = value_interp

    return {
        "mjd_daily": mjd_daily,
        "value_daily": value_daily,
        "value_smooth": value_smooth,
        "value_interp": value_interp,
        "driver_norm": driver_norm,
        "center": float(center),
        "scale": float(scale),
    }

def running_median_1d(
    x: np.ndarray, 
    window: int
) -> np.ndarray:
    """
    Centered `window`-running median for 1D arrays

    Parameters
    ----------
    x : np.ndarray
        Input 1D array.
    window : int
        Window size in samples. Should be odd to be centered.

    Returns
    -------
    np.ndarray
        Smoothed array of same length as x.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1D.")
    if window < 1:
        raise ValueError("window must be >= 1.")
    if window == 1:
        print("No median taken. This is your data array.")
        return x.copy()
    if window % 2 == 0:
        raise ValueError("Window must be odd for a centered running median.")

    n = len(x)
    half = window // 2
    out = np.empty(n, dtype=float)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.nanmedian(x[lo:hi])

    return out
