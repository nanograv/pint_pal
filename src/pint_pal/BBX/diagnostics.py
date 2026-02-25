# diagnostics.py

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Mapping, Any

import numpy as np
import pandas as pd

import scipy.linalg
import scipy.stats

from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerBase

import astropy.units as u


# =============================================================================
# Shared helpers
# =============================================================================

def plot_bb_spans(
    mjd: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    *,
    ax=None,
    show_points: bool = True,
    y_label: Optional[str] = None,
) -> "plt.Axes":
    """
    Aesthetics helper: draw alternating shaded BB spans on an axis.

    Notes
    -----
     - Caller controls axis labels; this helper can optionally set y-label.
    """
    mjd = np.asarray(mjd, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.asarray(edges, dtype=float)

    if ax is None:
        _fig, ax = plt.subplots(figsize=(7, 4))

    if show_points:
        ax.plot(mjd, y, ".", alpha=0.6)

    # Draw alternating shaded spans for each [left, right) block
    for i in range(len(edges) - 1):
        ax.axvspan(edges[i], edges[i + 1], alpha=0.12, color=("C0" if i % 2 == 0 else "C1"))

    ax.set_xlabel("MJD")
    if y_label is not None:
        ax.set_ylabel(y_label)
    return ax


def select_peak_indices(
    mjd: np.ndarray,
    y: np.ndarray,
    *,
    n: int = 4,
    min_sep_days: float = 120.0,
) -> np.ndarray:
    """
    plot_swx_bb_diagnostics helper: select ~n peaks spread equally across the time span 
    with a minimum separation (`min_sep_days`).

    Strategy:
      1) Split the MJD span into n equal windows.
      2) In each window, pick the highest valid point, respecting min_sep_days to already-chosen.
      3) If some windows had no valid pick, backfill from the remaining top peaks (global),
         still enforcing min_sep_days.
    """
    mjd = np.asarray(mjd, dtype=float)
    y = np.asarray(y, dtype=float)

    # Candidates sorted by descending value
    order = np.argsort(y)[::-1]
    order = order[np.isfinite(y[order]) & (y[order] > 0)]

    if len(order) == 0 or n <= 0:
        return np.array([], dtype=int)

    # Helper for spacing check
    def far_enough(idx: int, chosen: List[int]) -> bool:
        if not chosen:
            return True
        return bool(np.all(np.abs(mjd[idx] - mjd[np.asarray(chosen)]) >= float(min_sep_days)))

    # 1. split time into n equal windows
    edges = np.linspace(mjd.min(), mjd.max(), n + 1)
    chosen: List[int] = []

    # 2. pick best peak in each window (wrt spacing)
    for k in range(n):
        a, b = edges[k], edges[k + 1]
        for idx in order:
            if a <= mjd[idx] <= b and far_enough(int(idx), chosen):
                chosen.append(int(idx))
                break

    # 3. Were enough peaks selected? If not, backfill.
    if len(chosen) < n:
        for idx in order:
            idx = int(idx)
            if idx in chosen:
                continue
            if far_enough(idx, chosen):
                chosen.append(idx)
                if len(chosen) == n:
                    break

    return np.array(chosen[:n], dtype=int)


class DoublePatchHandler(HandlerBase):
    """
    Legend handler for alternating BB-span color proxy.
    """
    def __init__(self, colors: Tuple[str, str] = ("C0", "tan"), alpha: float = 0.12, **kw):
        super().__init__(**kw)
        self.colors = colors
        self.alpha = float(alpha)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        h = height / 2
        patches = []
        for i, c in enumerate(self.colors):
            # Make ea. entry evenly spaced
            y = ydescent + i * h
            rect = Rectangle(
                (xdescent, y),
                width,
                h,
                facecolor=c,
                alpha=self.alpha,
                transform=trans,
                edgecolor="none",
            )
            patches.append(rect)
        return patches


# =============================================================================
# Diagnostic plotters - BaseFits/SolarWindProxy
# =============================================================================

def plot_data_gaps_diagnostics(
    *,
    mjds: np.ndarray,
    resids: np.ndarray,
    gaps: Sequence[Tuple[float, float]],
    mask: np.ndarray,
    y_label: str = r"Residuals ($\mu$s)",
    title: str = "Identified Data Gaps and Data Mask",
    style: Optional["PlotStyleConfig"] = None,
) -> Figure:
    """
    Gaps diagnostic plotter.

    Parameters
    ----------
    mjds, resids : arrays
        Raw (unsorted is OK).
    gaps : list of (start, stop)
        Accepted gap bounds.
    mask : bool array
        True for kept, False for masked (should be empty, ideally).
    style : PlotStyleConfig, optional
        Plot style handle. This plot currently does not require it,
        but it is accepted for future improvements.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Standardize
    mjds = np.asarray(mjds, dtype=float)
    resids = np.asarray(resids, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)

    # Plot gaps
    for i, (start, stop) in enumerate(gaps):
        ax.axvspan(float(start), float(stop), color="red", alpha=0.15, label=("Gap" if i == 0 else ""))

    ax.plot(mjds[mask], resids[mask], ".", c="grey", alpha=0.2, label="Retained data")
    ax.plot(mjds[~mask], resids[~mask], ".", c="cyan", alpha=0.9, label="Masked data (should be empty)")

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("MJD")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()

    return fig


def plot_swx_bb_diagnostics(
    segmentation_dict: Mapping[str, Any],
    *,
    title: str = "SW proxy BB segmentation",
    n_peaks: int = 4,
    cols: int = 4,
    win_days: float = 250.0,
    maxt: u.Quantity = 365.25 * u.d,
    style: Optional["PlotStyleConfig"] = None,
) -> Figure:
    """
    SW-proxy BB segmentation diagnostic plotter:
      - top-left: spans + points
      - top-right: histogram of bin widths
      - bottom: zoom panels around selected peaks

    Expects keys in segmentation_dict:
      - "SW_BB_refined", "mjd_comb", "dm_comb", "sw_gaps"
    Optional:
      - "pulsar_name"

    Parameters
    ----------
    style : PlotStyleConfig, optional
        Plot style handle (colors, dpi conventions, etc.). This plot uses
        `style.cBB` and `style.cVan` when provided; otherwise matplotlib defaults.
    """
    # Plot from dictionary arrays
    SW_plot_BB = np.asarray(segmentation_dict["SW_BB_refined"], dtype=float)
    mjd_comb = np.asarray(segmentation_dict["mjd_comb"], dtype=float)
    dm_comb = np.asarray(segmentation_dict["dm_comb"], dtype=float)
    sw_gaps = list(segmentation_dict["sw_gaps"])

    # Try to extract pulsar name
    pulsar_name = segmentation_dict.get("pulsar_name", "Unknown Pulsar")

    # Style fallbacks
    cBB = getattr(style, "cBB", "C0") if style is not None else "C0"
    cVan = getattr(style, "cVan", "C1") if style is not None else "C1"

    # Layout setup
    rows = int(math.ceil(float(n_peaks) / float(cols)))
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(
        2 + rows - 1,
        cols,
        figure=fig,
        height_ratios=[2.0] + [1.0] * rows,
        wspace=0.35,
        hspace=0.4,
    )

    ax_main = fig.add_subplot(gs[0, 0:2])
    ax_hist = fig.add_subplot(gs[0, 2:4])

    # Dynamically create bottom-row (or multi-row) zoom axes for peak plots
    axes_zoom = []
    for i in range(int(n_peaks)):
        r = 1 + (i // int(cols)) # which row (starts at 1 because row 0 = top)
        c = i % int(cols)        # which column
        axes_zoom.append(fig.add_subplot(gs[r, c]))

    # Left: spans + points (SW proxy segmentation)
    plot_bb_spans(mjd_comb, dm_comb, SW_plot_BB, ax=ax_main, show_points=False)
    ax_main.plot(mjd_comb, dm_comb, ".", ms=3, color=cVan, label="SW proxy (points)")
    ax_main.set_xlabel("MJD")
    ax_main.set_ylabel(r"DM$_{SW}$ [pc cm$^{-3}$]")
    ax_main.set_title(title)

    # Plot gaps
    for g0, g1 in sw_gaps:
        ax_main.axvspan(float(g0), float(g1), color="red", alpha=0.15)

    # Right: histogram of bin widths (spacings)
    bin_widths = np.diff(SW_plot_BB)
    mask = bin_widths < float(maxt.to_value(u.d))
    bw = bin_widths[mask]
    ax_hist.hist(
        bw,
        bins=(len(bw) if bw.size > 0 else 1),
        edgecolor="k",
        alpha=0.8,
        color=cVan,
    )
    ax_hist.set_xlabel("Bin width [days]")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Distribution of BB spacing")

    # Bottom: zoom-ins near peaks
    peak_idx = select_peak_indices(mjd_comb, dm_comb, n=int(n_peaks), min_sep_days=200.0)
    for axi, i in zip(axes_zoom, peak_idx):
        center = float(mjd_comb[int(i)])
        x0, x1 = center - 0.5 * float(win_days), center + 0.5 * float(win_days)

        plot_bb_spans(mjd_comb, dm_comb, SW_plot_BB, ax=axi, show_points=False)
        axi.plot(mjd_comb, dm_comb, ".", ms=3, color=cVan, label="_nolegend_")

        # Add red gap shading within zoom if visible
        for g0, g1 in sw_gaps:
            if float(g1) < x0 or float(g0) > x1:
                continue
            axi.axvspan(float(g0), float(g1), color="red", alpha=0.15)

        axi.set_xlim(x0, x1)
        axi.set_xlabel("MJD")
        axi.set_title(f"Zoom @ MJD ≈ {center:.0f}", fontsize=9)

        if axi is axes_zoom[0]:
            axi.set_ylabel(r"DM$_{SW}$ [pc cm$^{-3}$]")
        else:
            axi.set_ylabel("")

    # Build proxy handles for a single, figure-level legend 
    span_patch = Patch(label="BB spans")
    legend_handles = [
        Line2D([], [], linestyle="none", marker=".", color="C1", label="SW proxy (points)"),
        span_patch,
        Patch(facecolor="red", alpha=0.15, label="Gaps"),
        Patch(facecolor=cVan, edgecolor="k", label="BB spacing hist"),
    ]
    # Create custom stacked BB span colors (two colors, alternating)
    handler_map = {span_patch: DoublePatchHandler(colors=(cBB, "tan"), alpha=0.25)}

    # One key to rule them all
    fig.legend(
        handles=legend_handles,
        handler_map=handler_map,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        handlelength=1.8,
    )

    # Add suptitle with pulsar name
    if pulsar_name is not None:
        fig.suptitle(f"{pulsar_name} — {title}", fontsize=14, y=1.02)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97]) # leave space for legend + title

    return fig

def extract_swx_params(model: pint.models.timing_model.TimingModel) -> Dict[str, Any]:
    """
    Extract SWX (time-variable solar-wind) segment parameters from a PINT model.

    Returns
    -------
    swx : dict
        {
          'indices'  : np.ndarray[int],     # segment indices       (e.g. 0...N-1)
          'par_names': list[str],           # SWXDM_XXXX names
          'amps'     : np.ndarray[float],   # SWXDM amplitudes      (pc cm^-3)
          'errs'     : np.ndarray[float],   # 1-sigma uncertainties (pc cm^-3)
          'r1'       : np.ndarray[float],   # segment start MJD
          'r2'       : np.ndarray[float],   # segment end   MJD
          'mid_mjd'  : np.ndarray[float],   # (r1+r2)/2
          'bin_days' : np.ndarray[float],   # (r2-r1)
        }
    """
    # Get index mapping for SWXDM parameters: {idx: "SWXDM_XXXX"}
    mapping = model.get_prefix_mapping("SWXDM_")
    if not mapping:
        raise ValueError("No SWXDM_ parameters found in the model.")

    # Sort by numeric index to ensure consistent order
    indices = np.array(sorted(mapping.keys()), dtype=int)
    names = [mapping[i] for i in indices]

    # Segment bounds (R1=left edge/R2=right edge)
    r1_names = [f"SWXR1_{i:04d}" for i in indices]
    r2_names = [f"SWXR2_{i:04d}" for i in indices]
    r1 = np.array([getattr(model, r1n).value for r1n in r1_names], dtype=float)
    r2 = np.array([getattr(model, r2n).value for r2n in r2_names], dtype=float)

    # Mid-points and widths
    mid_mjd = 0.5 * (r1 + r2)
    bin_days = (r2 - r1)

    # Amplitudes and uncertainties
    amps = np.array([getattr(model, n).value for n in names], dtype=float)
    # Uncertainty might be None if not fit; fallback to NaN
    errs = np.array(
        [
            getattr(model, n).uncertainty_value
            if getattr(model, n).uncertainty_value is not None
            else np.nan
            for n in names
        ],
        dtype=float,
    )

    return {
        "indices": indices,
        "par_names": names,
        "amps": amps,
        "errs": errs,
        "r1": r1,
        "r2": r2,
        "mid_mjd": mid_mjd,
        "bin_days": bin_days,
    }

def plot_swx_amplitudes(fitter, min_elong_mjds, SWX_bin=None, bb=False, z_thresh=3.5):
    """
    Plot SWXDM amplitudes (`SWXDM-i` - excludes geometric factor; g_i) vs MJD using a PINT fitter:
        DM_SWX(t) = SWXDM_i x g_i
    Flags and prints outliers via modified Z-score, then plots the filtered series.

    Parameters
    ----------
    fitter : pint.fitter.Fitter
        The PINT fitter that has already been run (contains .model).
    SWX_bin : float
        Nominal bin width (days) to display in the plot title.
    z_thresh : float, default 3.5 (arbitrary)
        Modified Z-score threshold for outlier flagging.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure, for optional saving.
    data : dict
        The raw extracted SWX dict (from `extract_swx_params`), plus:
          - 'modified_z'
          - 'is_outlier'
          - 'keep_mask'
    """
    # Make NE e.g. SWX amp vs year/bin graph - Is it consistent with 0?
    mod = fitter.model
    swx = extract_swx_params(mod)

    amps = swx["amps"]
    mid_mjd = swx["mid_mjd"]

    # Outlier detection using modified Z-score (robust to outliers)
    median_amp = np.nanmedian(amps)
    mad_amp = np.nanmedian(np.abs(amps - median_amp))
    if mad_amp == 0 or not np.isfinite(mad_amp):
        modified_z = np.zeros_like(amps, dtype=float)
    else:
        modified_z = 0.6745 * (amps - median_amp) / mad_amp

    is_outlier = np.abs(modified_z) > z_thresh
    keep_mask = ~is_outlier

    # Pretty-print any outliers
    if np.any(is_outlier):
        print(f"Filtered out {np.sum(is_outlier)} outlier(s) with modified Z > {z_thresh}:")
        print("{:<12} {:>15} {:>15} {:>15}".format("SWXDM Index", "MJD_mid", "Amplitude", "Modified Z"))
        print("-" * 64)
        for idx, mjd, amp, z in zip(
            swx["indices"][is_outlier], mid_mjd[is_outlier], amps[is_outlier], modified_z[is_outlier]
        ):
            print("{:<12} {:>15.2f} {:>15.3e} {:>15.2f}".format(f"SWXDM_{idx:04d}", mjd, amp, z))
        print("")

    # Plot filtered points with error bars
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(
        swx["mid_mjd"][keep_mask],
        swx["amps"][keep_mask],
        yerr=swx["errs"][keep_mask],
        fmt="o",
        capsize=2,
        label="SWXDM amplitudes",
    )
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=1)

    # Mark outlier locations with dashed vertical ticks
    if np.any(is_outlier):
        x_out = swx["mid_mjd"][is_outlier]
        ax.scatter(
            x_out, np.zeros_like(x_out),
            marker="|", linewidths=1.6, color="red", zorder=4,
            label=f"Outlier removed (z-score≥{z_thresh})",
        )

    # Plot conjuction
    for i, mjd in enumerate(min_elong_mjds):
        plt.axvline(mjd, color='orange', linestyle=':', alpha=0.4)
        
    ax.set_xlabel("MJD")
    ax.set_ylabel(r"SWXDM amplitude (pc cm$^{-3}$)")
    # Set title
    if bb:
        ax.set_title(f"SWXDM amplitudes (BB bins)")
    else:
        ax.set_title(f"SWXDM amplitudes ({SWX_bin:.2f} d bins)")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()

    # Attach diagnostics to the returned dict
    swx["modified_z"] = modified_z
    swx["is_outlier"] = is_outlier
    swx["keep_mask"] = keep_mask

    return fig, swx

def plot_swx_dm_series(fitter, min_elong_mjds, SWX_bin=None, bb=False, z_thresh=3.5,
                       autolim=True, pad_frac=0.08):
    """
    Plot the modeled SWX DM(t) series at each TOA using the fitted PINT model:
        DM_SWX(t) = SWXDM_i x g_i
    Applies modified Z-score filtering to suppress outliers for visual clarity.

    Parameters
    ----------
    fitter : pint.fitter.Fitter
        Fitter already run (contains .model and .toas).
    min_elong_mjds : sequence of float
        Conjunction (minimum-elongation) MJDs to overplot as reference.
    SWX_bin : float or None
        Nominal bin width (days) for title annotation when bb=False.
    bb : bool
        If True, title indicates "BB bins"; else uses SWX_bin value.
    z_thresh : float
        Modified Z-score threshold for outlier flagging.
    autolim : bool
        If True, set y-limits based on filtered data with small padding.
    pad_frac : float
        Fractional vertical padding for y-limits if autolim=True.

    Returns
    -------
    fig : matplotlib.figure.Figure
    data : dict
        {
            "mjd": np.ndarray,
            "dm_sw": np.ndarray,          # modeled DM_SWX(t) [pc cm^-3]
            "modified_z": np.ndarray,
            "is_outlier": np.ndarray[bool],
            "keep_mask": np.ndarray[bool],
        }
    """
    mod  = fitter.model
    toas = fitter.toas

    # Evaluate model's SWX DM term at each TOA (returns Quantities)
    dm_sw_q = mod.components["SolarWindDispersionX"].swx_dm(toas)  # pc cm^-3
    dm_sw   = np.asarray(dm_sw_q.to(u.pc/u.cm**3).value, dtype=float)
    mjd     = toas.get_mjds().value

    # Robust modified z-score on DM(t)
    med = np.nanmedian(dm_sw)
    mad = np.nanmedian(np.abs(dm_sw - med))
    if mad == 0 or not np.isfinite(mad):
        modified_z = np.zeros_like(dm_sw, dtype=float)
    else:
        modified_z = 0.6745 * (dm_sw - med) / mad

    is_outlier = np.abs(modified_z) > z_thresh
    keep_mask  = ~is_outlier

    # Pretty-print any outliers
    if np.any(is_outlier):
        print(f"Filtered out {np.sum(is_outlier)} outlier(s) with modified Z > {z_thresh}:")
        print("{:<10} {:>15} {:>15} {:>15}".format("Index", "MJD", "DM_SWX", "Modified Z"))
        print("-" * 64)
        for i, x, y, z in zip(np.where(is_outlier)[0], mjd[is_outlier], dm_sw[is_outlier], modified_z[is_outlier]):
            print("{:<10d} {:>15.2f} {:>15.3e} {:>15.2f}".format(int(i), x, y, z))
        print("")

    # Plot: all points faint for context, filtered points emphasized
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(mjd, dm_sw, ".", ms=2.5, alpha=0.25, label="_nolegend_")  # background (all)
    ax.plot(mjd[keep_mask], dm_sw[keep_mask], ".", ms=3.0, label=r"Modeled DM$_{\rm SWX}$(t) — filtered)")

    ax.axhline(0.0, color="gray", ls="--", lw=1)

    # Mark outliers along the time axis (optional visual cue)
    if np.any(is_outlier):
        ax.scatter(mjd[is_outlier], np.zeros(np.sum(is_outlier)), marker="|",
                   linewidths=1.6, color="red", zorder=4,
                   label=f"Outliers (|Z|≥{z_thresh:g})")

    # Plot conjunction vertical lines
    for mj in min_elong_mjds:
        ax.axvline(mj, color="orange", linestyle=":", alpha=0.4)

    ax.set_xlabel("MJD")
    ax.set_ylabel(r"DM$_{\rm SWX}$ [pc cm$^{-3}$]")
    if bb:
        ax.set_title("SWX-modeled solar-wind DM(t) — BB bins")
    else:
        if SWX_bin is None:
            ax.set_title("SWX-modeled solar-wind DM(t)")
        else:
            ax.set_title(f"SWX-modeled solar-wind DM(t) — {SWX_bin:.2f} d bins")

    # Auto y-limits based on filtered data to avoid blown scale
    if autolim and np.any(keep_mask):
        yk = dm_sw[keep_mask]
        ymin, ymax = np.nanmin(yk), np.nanmax(yk)
        if np.isfinite(ymin) and np.isfinite(ymax) and ymin != ymax:
            pad = pad_frac * (ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)

    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    data = {
        "mjd": mjd,
        "dm_sw": dm_sw,
        "modified_z": modified_z,
        "is_outlier": is_outlier,
        "keep_mask": keep_mask,
    }
    return fig, data

# =============================================================================
# Diagnostic plotters - DM Proxy ("Chromatic Suite")
# =============================================================================

def _apply_style(fig: Figure, style: Optional["PlotStyleConfig"]) -> None:
    if style is None:
        return
    try:
        fig.set_dpi(int(style.dpi))
    except Exception:
        pass


def plot_epoch_gap_histogram(
    mjd,
    epoch_tol_days,
    x_zoom=None,
    y_zoom=None,
    bins="auto",
    title_suffix="",
    *,
    style: Optional["PlotStyleConfig"] = None,
) -> Figure:
    """
    Histogram of adjacent dMJDs with a vertical line at epoch_tol_days.

    Notes
    -----
     - Uses robust binning to avoid MemoryError when 'auto' explodes.
     - Counts (y-axis) shown on log scale; x-axis remains linear.
    """
    mjd = np.asarray(mjd, float)
    order = np.argsort(mjd)
    diffs = np.diff(mjd[order])

    diffs = diffs[np.isfinite(diffs) & (diffs >= 0)]
    fig, ax = plt.subplots(figsize=(9, 4))
    _apply_style(fig, style)

    if diffs.size == 0:
        ax.text(0.5, 0.5, "Not enough points for gap histogram.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    # Robust bin selection
    if isinstance(bins, str) and bins == "auto":
        n = diffs.size
        q25, q75 = np.percentile(diffs, [25, 75])
        iqr = max(q75 - q25, 0.0)
        if iqr > 0:
            bw = 2.0 * iqr / (n ** (1 / 3))
            span = diffs.max() - diffs.min()
            nbins = int(np.ceil(span / bw)) if bw > 0 else int(np.sqrt(n))
        else:
            nbins = int(np.sqrt(n))
        nbins = max(5, min(nbins, 200))
    else:
        nbins = bins

    hi = np.percentile(diffs, 99.5)
    plot_vals = diffs[diffs <= hi]

    ax.hist(plot_vals, bins=nbins, label="ΔMJD histogram", log=True)
    ax.axvline(
        epoch_tol_days,
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
        color="tab:red",
        label=f"epoch_tol_days = {epoch_tol_days:g} d",
    )

    if x_zoom is not None:
        ax.set_xlim(x_zoom[0], x_zoom[1])
    if y_zoom is not None:
        ax.set_ylim(y_zoom[0], y_zoom[1])

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("Adjacent ΔMJD (days)")
    ax.set_ylabel("Count (log)")
    ax.set_title(f"Adjacent ΔMJD Histogram{title_suffix}\n(split if Δ > tol)")

    fig.text(
        0.5,
        -0.01,
        "Shows distribution of time gaps between TOAs. A good DM proxy fit has clear clustering of small ΔMJDs "
        "with gaps larger than epoch_tol_days marking natural breaks.",
        ha="center",
        va="top",
        fontsize=9,
        wrap=True,
    )
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_all_epoch_fit_overlay(
    x_all,
    y_all,
    yerr_all,
    mjd,
    groups,
    results,
    *,
    color_by: str = "snr",         # "snr" | "n" | "none"
    show_points: bool = False,
    show_lines: bool = False,
    max_points_per_epoch: int = 200,
    lw: float = 1.0,
    style: Optional["PlotStyleConfig"] = None,
) -> Figure:
    """
    Overlay all per-epoch WLS fits y = a + b x on one axes.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    _apply_style(fig, style)

    if not results:
        ax.text(0.5, 0.5, "No fitted epochs to overlay.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    vals = None
    cbar_label = None
    if color_by == "snr":
        vals = np.array(
            [(abs(r["b"]) / r["sb"]) if (np.isfinite(r["sb"]) and r["sb"] > 0) else np.nan for r in results],
            float,
        )
        cbar_label = r"|b| / $\sigma_b$"
    elif color_by == "n":
        vals = np.array([r.get("n_used", np.nan) for r in results], float)
        cbar_label = "n_used"

    cmap = plt.get_cmap("viridis")
    norm = None
    if vals is not None and np.isfinite(vals).any():
        vmin = np.nanpercentile(vals, 5)
        vmax = np.nanpercentile(vals, 95)
        if not np.isfinite(vmin):
            vmin = np.nanmin(vals)
        if not np.isfinite(vmax):
            vmax = np.nanmax(vals)
        if vmin == vmax:
            vmin, vmax = 0.0, 1.0
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for i, r in enumerate(results):
        ei = r["epoch_idx"]
        a, b = r["a"], r["b"]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue

        g = groups[ei]
        xv = np.asarray(x_all[g], float)
        yv = np.asarray(y_all[g], float)
        ev = None if yerr_all is None else np.asarray(yerr_all[g], float)

        if ev is None:
            valid = np.isfinite(xv) & np.isfinite(yv)
        else:
            valid = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(ev) & (ev > 0)

        xv = xv[valid]
        yv = yv[valid]
        if xv.size < 2:
            continue

        if norm is None:
            color = "tab:orange"
        else:
            val_i = vals[i]
            if not np.isfinite(val_i):
                val_i = np.nanmedian(vals)
            color = cmap(norm(val_i))

        if show_points:
            if xv.size > max_points_per_epoch:
                step = int(np.ceil(xv.size / max_points_per_epoch))
                xv_plot = xv[::step]
                yv_plot = yv[::step]
            else:
                xv_plot, yv_plot = xv, yv
            ax.scatter(
                xv_plot,
                yv_plot,
                s=10,
                facecolors="tab:blue",
                edgecolors="black",
                linewidths=0.3,
                alpha=0.25,
            )

        if show_lines:
            xx = np.linspace(xv.min(), xv.max(), 100)
            yy = a + b * xx
            ax.plot(xx, yy, color=color, lw=lw, alpha=0.9)

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("x (dispersion regressor)")
    ax.set_ylabel("Residual")
    title_suffix = f" (colored by {cbar_label})" if cbar_label else ""
    ax.set_title(f"Overlay of per-epoch WLS fits{title_suffix}")

    fig.text(
        0.5,
        -0.01,
        "Each line shows a weighted least-squares fit performed within a single epoch across observing frequencies. "
        "Colors indicate per-epoch signal quality (|b|/σ_b) or number of TOAs (n_used). A well-behaved DM proxy fit has consistent slopes "
        "across epochs and a broad range of x-values, showing stable separation between achromatic (a) and chromatic (b) terms.",
        ha="center",
        va="top",
        fontsize=9,
        wrap=True,
    )

    if norm is not None:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label(cbar_label)

    fig.tight_layout()
    return fig


def plot_epoch_fit_examples(
    x_all,
    y_all,
    yerr_all,
    mjd,
    groups,
    results,
    *,
    select="worst_xspan",
    k=12,
    style: Optional["PlotStyleConfig"] = None,
) -> Figure:
    """
    Draw k example epoch fits: y vs x with WLS line.
    """
    import math

    fig, axs = plt.subplots(1, 1, figsize=(9, 4))
    _apply_style(fig, style)
    plt.close(fig)  # will recreate below

    if not results:
        fig, ax = plt.subplots(figsize=(9, 4))
        _apply_style(fig, style)
        ax.text(0.5, 0.5, "No kept epochs to visualize.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    res = results.copy()

    def keyfunc(r):
        if select == "worst_xspan":
            return (np.inf if not np.isfinite(r["x_span"]) else r["x_span"])
        if select == "lowest_n":
            return r["n_used"]
        if select == "highest_snr":
            return (-(abs(r["b"]) / r["sb"]) if (np.isfinite(r["sb"]) and r["sb"] > 0) else np.inf)
        return 0

    if select == "random":
        rng = np.random.default_rng(0)
        res = list(rng.choice(res, size=min(k, len(res)), replace=False))
    else:
        res.sort(key=keyfunc)
        res = res[: min(k, len(res))]

    n = len(res)
    ncols = 4
    nrows = math.ceil(n / ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows), squeeze=False)
    _apply_style(fig, style)

    for ax, r in zip(axs.ravel(), res):
        ei = r["epoch_idx"]
        g = groups[ei]
        xv = np.asarray(x_all[g], float)
        yv = np.asarray(y_all[g], float)
        ev = None if yerr_all is None else np.asarray(yerr_all[g], float)

        if ev is None:
            valid = np.isfinite(xv) & np.isfinite(yv)
        else:
            valid = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(ev) & (ev > 0)

        xv = xv[valid]
        yv = yv[valid]
        ev = None if ev is None else ev[valid]

        if ev is not None:
            ax.errorbar(
                xv,
                yv,
                yerr=ev,
                fmt="o",
                ms=4,
                mfc="tab:blue",
                mec="black",
                mew=0.3,
                alpha=0.8,
                lw=0,
            )
        else:
            ax.scatter(
                xv,
                yv,
                s=16,
                facecolors="tab:blue",
                edgecolors="black",
                linewidths=0.3,
                alpha=0.8,
            )

        a, b = r["a"], r["b"]
        if np.isfinite(a) and np.isfinite(b) and xv.size >= 2:
            xx = np.linspace(xv.min(), xv.max(), 100)
            ax.plot(xx, a + b * xx, color="tab:orange", lw=1.2, label="WLS fit")

        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlabel("x (dispersion regressor)")
        ax.set_ylabel("residual")
        ax.set_title(
            f"MJD~{r['mid_mjd']:.1f}  n={r['n_used']}  |b|/σ={abs(r['b'])/r['sb']:.2g}"
            if (np.isfinite(r["sb"]) and r["sb"] > 0)
            else f"MJD~{r['mid_mjd']:.1f}  n={r['n_used']}"
        )
        ax.legend(loc="best")

    for ax in axs.ravel()[n:]:
        ax.axis("off")

    fig.suptitle(f"Example per-epoch fits ({select}, k={n})", y=1.02)
    fig.text(
        0.5,
        -0.01,
        "Each panel shows a single epoch’s WLS fit of residuals vs. dispersion regressor (either x=1/ν² or x=(ν_ref/ν)²). "
        "The blue points are TOAs within that epoch; the orange line is the fitted model y = a + b·x. "
        "Epochs with few TOAs, narrow x-span, or weak slope (low |b|/σ_b) indicate poor chromatic–achromatic separation. "
        "Consistent slopes across epochs suggest a stable DM proxy; scattered or flat fits may signal overfitting or inadequate sampling.",
        ha="center",
        va="top",
        fontsize=9,
        wrap=True,
    )

    fig.tight_layout()
    return fig


def plot_points_per_epoch_arrays(
    epoch_mjd,
    n_used,
    *,
    title_suffix="",
    min_points: Optional[int] = None,
    fit_conditions: Optional[dict] = None,
    style: Optional["PlotStyleConfig"] = None,
) -> Figure:
    """
    Scatter plot of how many data points entered the WLS fit per epoch.
    """
    epoch_mjd = np.asarray(epoch_mjd, float)
    n_used = np.asarray(n_used, int)

    fig, ax = plt.subplots(figsize=(9, 4))
    _apply_style(fig, style)

    if epoch_mjd.size == 0 or n_used.size == 0:
        ax.text(0.5, 0.5, "No kept epochs to plot.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    ax.scatter(
        epoch_mjd,
        n_used,
        s=16,
        facecolors="tab:blue",
        edgecolors="black",
        linewidths=0.3,
        alpha=0.7,
        label="# points in WLS fit",
    )
    if min_points is not None:
        ax.axhline(
            min_points,
            linestyle="--",
            color="tab:red",
            linewidth=1.2,
            alpha=0.9,
            label=f"threshold = {min_points}",
        )

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch MJD")
    ax.set_ylabel("Points used in WLS")
    ax.set_title(f"Per-epoch points used in WLS{title_suffix}")
    ax.legend(loc="best")

    fc = fit_conditions if isinstance(fit_conditions, dict) else {}
    reg_label = fc.get("regressor", "1/ν²")
    min_ch = fc.get("min_channels")
    min_uniq = fc.get("min_unique_channels")
    min_span = fc.get("min_x_span")
    min_snr = fc.get("min_snr_b")
    req_err = fc.get("require_positive_errors", False)
    clip_out = fc.get("clip_resid_outliers", False)
    mad_sig = fc.get("mad_sigma", None)
    norm_x = fc.get("normalize_x", False)
    norm_meth = fc.get("normalize_method", None)

    if min_span is not None:
        min_span_str = (
            f"{float(min_span):.2e} 1/MHz²" if reg_label == "1/ν²" else f"{float(min_span):.2e} (in {reg_label})"
        )
    else:
        min_span_str = None

    bullets = [f"regressor={reg_label}"]
    if min_ch is not None:
        bullets.append(f"min_channels={min_ch}")
    if min_uniq is not None:
        bullets.append(f"min_unique_freqs={min_uniq}")
    if min_span_str:
        bullets.append(f"min_x_span={min_span_str}")
    if isinstance(min_snr, (int, float)) and min_snr > 0:
        bullets.append(f"SNR cut |b|/σ_b≥{min_snr:g}")
    if req_err:
        bullets.append("require positive finite y-errors")
    if clip_out:
        bullets.append(f"MAD clip (±{mad_sig}σ)" if mad_sig is not None else "MAD clip")
    if norm_x:
        bullets.append(f"x normalized ({norm_meth})" if norm_meth else "x normalized")

    caption = (
        "Each point shows how many TOAs actually entered the per-epoch WLS fit after all conditions were applied. "
        + ("Conditions: " + "; ".join(bullets) + "." if bullets else "")
    )

    fig.text(0.5, -0.03, caption, ha="center", va="top", fontsize=9, wrap=True)
    fig.tight_layout()
    return fig


def plot_b_snr_vs_time(
    epoch_mjd,
    b_chrom,
    b_err,
    *,
    style: Optional["PlotStyleConfig"] = None,
) -> Figure:
    """
    Plot |b|/σ_b vs epoch time on log y-axis.
    """
    epoch_mjd = np.asarray(epoch_mjd, float)
    b_chrom = np.asarray(b_chrom, float)
    b_err = np.asarray(b_err, float)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.abs(b_chrom) / b_err

    mask = np.isfinite(snr) & (snr > 0)

    fig, ax = plt.subplots(figsize=(9, 4))
    _apply_style(fig, style)

    if not np.any(mask):
        ax.text(0.5, 0.5, "No positive SNR values to plot.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    ax.scatter(
        epoch_mjd[mask],
        snr[mask],
        s=16,
        facecolors="tab:blue",
        edgecolors="black",
        linewidths=0.3,
        alpha=0.7,
        label="|b|/σ_b",
    )
    ax.axhline(2.0, linestyle="--", color="tab:red", linewidth=1.2, alpha=0.9, label="2σ")
    ax.axhline(3.0, linestyle="--", color="tab:green", linewidth=1.2, alpha=0.9, label="3σ")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch MJD")
    ax.set_ylabel("|b| / σ_b (log)")
    ax.set_title("Chromatic slope significance over time")

    fig.text(
        0.5,
        -0.01,
        "Shows how significant the chromatic slope (b) is in each epoch. Good DM proxy fits have many epochs above 2–3σ, "
        "indicating measurable dispersion trends.",
        ha="center",
        va="top",
        fontsize=9,
        wrap=True,
    )
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_a_vs_b_correlation(
    a_achrom,
    b_chrom,
    *,
    style: Optional["PlotStyleConfig"] = None,
) -> Figure:
    """
    Plot |a| vs |b| on log–log axes with optional log–log trend line.
    """
    a = np.asarray(a_achrom, float)
    b = np.asarray(b_chrom, float)

    mask = np.isfinite(a) & np.isfinite(b)

    fig, ax = plt.subplots(figsize=(9, 4))
    _apply_style(fig, style)

    if not np.any(mask):
        ax.text(0.5, 0.5, "No finite (a,b) pairs.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    a = np.abs(a[mask])
    b = np.abs(b[mask])

    ax.scatter(
        a,
        b,
        s=16,
        facecolors="tab:blue",
        edgecolors="black",
        linewidths=0.3,
        alpha=0.7,
        label="|b| vs |a|",
    )

    try:
        m, c = np.polyfit(np.log10(a), np.log10(b), 1)
        xx = np.logspace(np.log10(a.min()), np.log10(a.max()), 50)
        yy = 10 ** (m * np.log10(xx) + c)
        ax.plot(xx, yy, color="tab:orange", linewidth=1.2, alpha=0.9, label=f"log–log trend (slope={m:.2f})")
    except Exception:
        pass

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.set_xlabel(r"|a| (achromatic, log)")
    ax.set_ylabel(r"|b| (chromatic, log)")
    ax.set_title("Magnitude correlation between achromatic and chromatic terms (|a|–|b|)")

    fig.text(
        0.5,
        -0.01,
        "All finite epochs shown (absolute values). A weak log–log trend indicates independent achromatic "
        "and chromatic structures, as expected for well-conditioned fits.",
        ha="center",
        va="top",
        fontsize=9,
        wrap=True,
    )

    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_a_b_time_scatter(
    epoch_mjd,
    a_achrom,
    b_chrom,
    *,
    y_label="Coefficient value",
    title="Per-epoch achromatic (a) and chromatic (b) terms",
    legend=True,
    style: Optional["PlotStyleConfig"] = None,
) -> Figure:
    """
    Scatter plot of a(t) and b(t) vs epoch MJD on the same axes.

    Pure plotter policy:
    - Does not show or save.
    - Returns a matplotlib Figure.
    """
    epoch_mjd = np.asarray(epoch_mjd, float)
    a = np.asarray(a_achrom, float)
    b = np.asarray(b_chrom, float)

    fig, ax = plt.subplots(figsize=(9, 4))
    _apply_style(fig, style)

    ax.scatter(
        epoch_mjd,
        a,
        s=16,
        facecolors="tab:orange",
        edgecolors="black",
        linewidths=0.3,
        alpha=0.7,
        label="a (achromatic)",
    )
    ax.scatter(
        epoch_mjd,
        b,
        s=16,
        facecolors="tab:blue",
        edgecolors="black",
        linewidths=0.3,
        alpha=0.7,
        label="b (chromatic)",
    )

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch MJD")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if legend:
        ax.legend(loc="best")

    caption = (
        "Shows the achromatic (a, orange) and chromatic (b, blue) components derived from per-epoch WLS fits. "
        "Temporal structure in a(t) may reflect broadband systematics, while variations in b(t) trace "
        "frequency-dependent (DM-like) behavior."
    )
    fig.text(0.5, -0.03, caption, ha="center", va="top", fontsize=9, wrap=True)

    fig.tight_layout()
    return fig


def plot_wls_epoch_summaries(
    metrics_list,
    *,
    style: Optional["PlotStyleConfig"] = None,
) -> Tuple[Figure, Figure]:
    """
    Summarize WLS diagnostics across kept epochs.

    Returns
    -------
    (fig_snr, fig_cond)
      fig_snr  : SNR(b) vs epoch index (log y)
      fig_cond : determinant and x-span vs epoch index (two-panel)
    """
    if not metrics_list:
        fig, ax = plt.subplots(figsize=(9, 4))
        _apply_style(fig, style)
        ax.text(0.5, 0.5, "No metrics to summarize.", ha="center", va="center")
        ax.set_axis_off()
        return fig, fig

    def arr(key: str) -> np.ndarray:
        vals = [m.get(key, np.nan) for m in metrics_list]
        return np.asarray(vals, float)

    snr_b = arr("snr_b")
    dets = arr("det")
    xspan = arr("x_span")
    idx = np.arange(len(metrics_list))

    # fig 1: SNR(b)
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    _apply_style(fig1, style)

    mask = np.isfinite(snr_b) & (snr_b > 0)
    if np.any(mask):
        ax1.scatter(
            idx[mask],
            snr_b[mask],
            s=18,
            facecolors="tab:blue",
            edgecolors="black",
            linewidths=0.3,
            alpha=0.7,
            label="epochs",
        )
    ax1.axhline(2.0, linestyle="--", color="tab:red", linewidth=1.2, alpha=0.9, label="2σ")
    ax1.axhline(3.0, linestyle="--", color="tab:green", linewidth=1.2, alpha=0.9, label="3σ")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Epoch index")
    ax1.set_ylabel("|b| / σ_b (log)")
    ax1.set_title("Per-epoch slope significance")

    fig1.text(
        0.5,
        -0.01,
        "Shows how often chromatic slopes exceed 2–3σ. A good DM proxy fit has consistent epochs above these thresholds, "
        "indicating robust detection of dispersion.",
        ha="center",
        va="top",
        fontsize=9,
        wrap=True,
    )
    ax1.legend(loc="best")
    fig1.tight_layout()

    # fig 2: conditioning summaries
    fig2, axs = plt.subplots(1, 2, figsize=(11, 4))
    _apply_style(fig2, style)

    axs[0].scatter(
        idx,
        dets,
        s=18,
        facecolors="tab:blue",
        edgecolors="black",
        linewidths=0.3,
        alpha=0.7,
        label="det",
    )
    axs[0].set_xlabel("Epoch index")
    axs[0].set_ylabel("det(normal matrix)")
    axs[0].set_title("Determinant (conditioning proxy)")
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].legend(loc="best")

    axs[1].scatter(
        idx,
        xspan,
        s=18,
        facecolors="tab:blue",
        edgecolors="black",
        linewidths=0.3,
        alpha=0.7,
        label="x-span",
    )
    axs[1].set_xlabel("Epoch index")
    axs[1].set_ylabel("x-span (range of 1/ν²)")
    axs[1].set_title("Frequency coverage per epoch")
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].legend(loc="best")

    fig2.suptitle("WLS conditioning summaries")
    fig2.text(
        0.25,
        -0.02,
        "Determinant reflects regression stability:\nlarger values → more stable fits.",
        ha="center",
        va="top",
        fontsize=9,
    )
    fig2.text(
        0.75,
        -0.02,
        "x-span reflects frequency coverage:\nwide spans → better DM slope constraints.",
        ha="center",
        va="top",
        fontsize=9,
    )
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig1, fig2

def plot_dmx_segmentation_by_slice_diagnostics(
    diag: "DMXGapAdjustDiagnostics",
    *,
    style: Optional["PlotStyleConfig"] = None,
) -> Figure:
    """
    Plot DMX segmentation-by-slice diagnostics, using shaded BB spans (axvspan)
    instead of alternating +/- block lines.
    """
    c_bb0 = getattr(style, "cBB", "C0") if style is not None else "C0"
    c_bb1 = "C1"  # second alternating color
    c_pts = "k"

    fig, ax = plt.subplots(figsize=(10, 5))
    if style is not None:
        try:
            fig.set_dpi(int(style.dpi))
        except Exception:
            pass

    first_zero = True
    first_slice = True

    # Plot each slice
    for s in diag.slices:
        start, stop = float(s.start), float(s.stop)

        # Zero line over the slice
        z_lbl = "Zero line" if first_zero else "_nolegend_"
        ax.hlines(0.0, start, stop, colors="k", linewidth=1, label=z_lbl)
        first_zero = False

        # Shaded BB spans for this slice (only within [start, stop])
        edges = np.asarray(s.seg_edges, float)
        if edges.size >= 2:
            for i, (bs, be) in enumerate(zip(edges[:-1], edges[1:])):
                bs = float(max(bs, start))
                be = float(min(be, stop))
                if be <= bs:
                    continue
                color = c_bb0 if (i % 2 == 0) else c_bb1
                ax.axvspan(
                    bs,
                    be,
                    alpha=0.12,
                    color=color,
                    label=("BB spans" if first_slice and i == 0 else "_nolegend_"),
                )
            first_slice = False

        # Residuals scatter (label by slice)
        ax.scatter(
            np.asarray(s.seg_mjds, float),
            np.asarray(s.seg_resids, float),
            s=10,
            alpha=0.7,
            color=c_pts,
            edgecolors="black",
            linewidths=0.2,
            label=f"Slice {s.slice_index}: {start:.1f}–{stop:.1f}",
            zorder=3,  # keep points above spans
        )

    ax.set_xlabel("MJD")
    ax.set_ylabel("Residual (μs)")
    ax.set_title("DMX segmentation-by-slice diagnostics")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(ncol=2, fontsize="small", loc="upper right")
    fig.tight_layout()
    return fig

def _extract_dmx_series_from_fitter(fitter):
    """
    Extract (dmx_epoch_mjd, dmx_values) using pint.dmxparse.dmxparse.

    Returns
    -------
    t : np.ndarray
        DMX epoch midpoints (MJD float)
    y : np.ndarray
        DMX values (float). Defensively mean-subtracted for plotting.
    """
    try:
        from pint.dmxparse import dmxparse
        out = dmxparse(fitter)
        if not isinstance(out, dict):
            return None, None

        t = out.get("dmxeps", None)
        y = out.get("dmxs", None)
        if t is None or y is None:
            return None, None

        # handle astropy Quantity or plain arrays
        t = np.asarray(getattr(t, "value", t), dtype=float)
        y = np.asarray(getattr(y, "value", y), dtype=float)

        if t.size == 0 or y.size == 0:
            return None, None

        # mean-subtracted
        y = y - np.nanmean(y)

        return t, y

    except Exception as e:
        print(f"[diagnostics:_extract_dmx_series_from_fitter] FAILED ({type(e).__name__}): {e}")
        return None, None

def plot_bchrom_vs_dmx(
    epoch_mjd: np.ndarray,
    b_chrom: np.ndarray,
    fitter,
    *,
    style: Optional["PlotStyleConfig"] = None,
) -> Figure:
    """
    Overlay b_chrom(t) with DMX(t) extracted from fitter.
    """
    c_van = getattr(style, "cVan", "tab:blue") if style is not None else "tab:blue"
    c_bb = getattr(style, "cBB", "tab:orange") if style is not None else "tab:orange"

    fig, ax = plt.subplots(figsize=(9, 4))
    if style is not None:
        try:
            fig.set_dpi(int(style.dpi))
        except Exception:
            pass

    if epoch_mjd is None or b_chrom is None or len(epoch_mjd) == 0:
        ax.text(0.5, 0.5, "No chromatic series available to compare.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    epoch_mjd = np.asarray(epoch_mjd, float)
    b_chrom = np.asarray(b_chrom, float)

    dmx_t, dmx_v = _extract_dmx_series_from_fitter(fitter)
    if dmx_t is None or dmx_v is None:
        ax.text(0.5, 0.5, "Could not extract DMX series from fitter.", ha="center", va="center")
        ax.set_axis_off()
        return fig

    dmx_t = np.asarray(dmx_t, float)
    dmx_v = np.asarray(dmx_v, float)

    ax.scatter(
        epoch_mjd,
        b_chrom,
        s=18,
        color=c_van,
        edgecolors="black",
        linewidths=0.3,
        alpha=0.7,
        label="b_chrom (DM proxy)",
    )
    ax.scatter(
        dmx_t,
        dmx_v,
        s=18,
        color=c_bb,
        edgecolors="black",
        linewidths=0.3,
        alpha=0.7,
        label="DMX (mean-subtracted)",
    )

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("MJD")
    ax.set_ylabel("DM-equivalent units")
    ax.set_title("Chromatic proxy vs DMX")

    fig.text(
        0.5,
        -0.01,
        "Overlay of b_chrom (DM proxy) and DMX series. A good DM proxy fit tracks the DMX curve closely in both trend and amplitude.",
        ha="center",
        va="top",
        fontsize=9,
        wrap=True,
    )
    ax.legend(loc="best")
    fig.tight_layout()
    return fig

# =============================================================================
# General Diagnostics
# =============================================================================

# -------------------------------------------------------------------------
# Correlation/covariance diagnostics
# -------------------------------------------------------------------------

def plot_all_prefix_ellipses(
    fitter,
    prefix1: str,
    prefix2: str,
    nsigmas=(1, 2, 3),
    save_dir=None,
    show=True,
):
    """
    For each integer index i that appears in BOTH prefix1_i and prefix2_i,
    make a correlation-ellipse plot using plot_param_ellipses_from_fitter,
    skipping any pair that is missing from the covariance matrix.

    Examples
    --------
    # SWXDM_i vs DMX_i
    plot_all_prefix_ellipses(f2, "SWXDM_", "DMX_")

    # JUMP_i vs DMX_i (if such parameters exist)
    plot_all_prefix_ellipses(f2, "JUMP_", "DMX_")
    """
    m = fitter.model

    # Prefix mappings in the model (keys are integer indices)
    map1 = m.get_prefix_mapping(prefix1)
    map2 = m.get_prefix_mapping(prefix2)

    # Indices that exist in both model prefixes
    common_indices = sorted(set(map1.keys()) & set(map2.keys()))
    if not common_indices:
        print(f"No overlapping indices found in the *model* for "
              f"{prefix1}* and {prefix2}*.")
        return

    # Labels that actually appear in the covariance matrix
    cov = fitter.parameter_covariance_matrix
    cov_labels = [lab for (lab, _info) in cov.labels[0]]
    cov_label_set = set(cov_labels)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    psr = getattr(getattr(m, "PSR", None), "value", "PSR")
    figs_axes = []
    skipped = []

    for idx in common_indices:
        # Build parameter labels, assume 4-digit formatting as in DMX/SWXDM
        p1 = f"{prefix1}{idx:04d}"
        p2 = f"{prefix2}{idx:04d}"

        # Skip if either parameter not present in covariance labels
        if (p1 not in cov_label_set) or (p2 not in cov_label_set):
            skipped.append((idx, p1, p2))
            continue

        # Make the ellipse plot 
        fig, ax = plot_param_ellipses_from_fitter(
            fitter,
            p1=p1,
            p2=p2,
            nsigmas=nsigmas,
            ax=None,
        )

        ax.set_title(f"{psr}: correlation between {p1} and {p2}")

        if save_dir is not None:
            fname = os.path.join(save_dir, f"{psr}_ellipse_{p1}_{p2}.png")
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"Saved: {fname}")

        if not show:
            plt.close(fig)

        figs_axes.append((fig, ax))

    if skipped:
        print("Skipped indices (missing in covariance matrix):")
        for (idx, p1, p2) in skipped:
            missing = []
            if p1 not in cov_label_set:
                missing.append(p1)
            if p2 not in cov_label_set:
                missing.append(p2)
            missing_str = ", ".join(missing)
            print(f"  i={idx:04d}: missing {missing_str}")

    return figs_axes

def summarize_swx_dmx_correlations(
    fitter,
    prefix1="SWXDM_",
    prefix2="DMX_",
):
    """
    Build a table of correlation coefficients between prefix1_i and prefix2_i
    (e.g. SWXDM_i and DMX_i) for all indices i present in the model AND in
    the fitter.parameter_covariance_matrix.

    Parameters
    ----------
    fitter : pint.fitter.Fitter
        Fitter with a parameter_covariance_matrix.
    prefix1, prefix2 : str
        Parameter prefixes, default "SWXDM_" and "DMX_".

    Returns
    -------
    df : pandas.DataFrame
        Columns:
          - idx         : integer bin index
          - p1          : name of parameter 1 (e.g. "SWXDM_0001")
          - p2          : name of parameter 2 (e.g. "DMX_0001")
          - rho         : correlation coefficient between p1 and p2
          - sigma_p1    : 1σ uncertainty of p1
          - sigma_p2    : 1σ uncertainty of p2
        Sorted by descending |rho|.
    """
    m = fitter.model

    # Prefix mappings in the model (these tell us which indices exist)
    map1 = m.get_prefix_mapping(prefix1)
    map2 = m.get_prefix_mapping(prefix2)

    # Indices that exist in both model prefixes
    common_indices = sorted(set(map1.keys()) & set(map2.keys()))
    if not common_indices:
        print(f"No overlapping {prefix1}i and {prefix2}i indices found in the model.")
        return pd.DataFrame()

    # Covariance matrix and its labels
    cov = fitter.parameter_covariance_matrix
    cov_labels = [lab for (lab, _info) in cov.labels[0]]
    cov_label_set = set(cov_labels)

    # Underlying numeric matrix (strip units if present)
    mat = cov.matrix
    if hasattr(mat, "value"):
        C = mat.value
    else:
        C = np.asarray(mat, float)

    rows = []
    skipped = []

    for idx in common_indices:
        p1 = f"{prefix1}{idx:04d}"
        p2 = f"{prefix2}{idx:04d}"

        # Skip if either param is not in the covariance labels
        if (p1 not in cov_label_set) or (p2 not in cov_label_set):
            skipped.append((idx, p1, p2))
            continue

        i1 = cov_labels.index(p1)
        i2 = cov_labels.index(p2)

        var1 = C[i1, i1]
        var2 = C[i2, i2]
        cov12 = C[i1, i2]

        if var1 <= 0 or var2 <= 0:
            rho = np.nan
            sigma1 = np.nan
            sigma2 = np.nan
        else:
            sigma1 = np.sqrt(var1)
            sigma2 = np.sqrt(var2)
            rho = cov12 / (sigma1 * sigma2)
            rho = float(np.clip(rho, -1.0, 1.0))

        rows.append(
            dict(
                idx=idx,
                p1=p1,
                p2=p2,
                rho=rho,
                sigma_p1=sigma1,
                sigma_p2=sigma2,
            )
        )

    if skipped:
        print("Skipped indices (missing in covariance matrix):")
        for idx, p1, p2 in skipped:
            missing = []
            if p1 not in cov_label_set:
                missing.append(p1)
            if p2 not in cov_label_set:
                missing.append(p2)
            print(f"  i={idx:04d}: missing {', '.join(missing)}")

    df = pd.DataFrame(rows)
    # Sort by |rho| descending so the most strongly correlated bins are on top
    df["abs_rho"] = df["rho"].abs()
    df = df.sort_values("abs_rho", ascending=False).reset_index(drop=True)
    return df[["idx", "p1", "p2", "rho", "sigma_p1", "sigma_p2"]]

def plot_param_ellipses_from_fitter(
    fitter,
    p1: str,
    p2: str,
    nsigmas=(1, 2, 3),
    ax=None,
):
    """
    Plot correlation ellipses in (Δp1, Δp2) space using the parameter
    covariance matrix from a PINT fitter, following the PINT tutorials.

    Parameters
    ----------
    fitter : pint.fitter.Fitter
        Fitter with a computed parameter_covariance_matrix.
    p1, p2 : str
        Parameter names, e.g. 'SWXDM_0001', 'DMX_0001'.
    nsigmas : iterable of int
        Contours to draw (in Gaussian sigma units).
    ax : matplotlib.axes.Axes or None
        If None, create a new figure and axes.

    Returns
    -------
    fig, ax
    """
    # Extract the 2x2 covariance sub-matrix for (p1, p2)  
    cov_full = fitter.parameter_covariance_matrix
    sub = cov_full.get_label_matrix([p1, p2]).matrix  # 2x2

    # Strip units if present
    if hasattr(sub, "value"):
        C = sub.value
    else:
        C = np.asarray(sub, float)

    # Symmetrize defensively
    C = 0.5 * (C + C.T)

    var1, var2 = C[0, 0], C[1, 1]
    cov12 = C[0, 1]

    # Guard against numerical issues
    if var1 <= 0 or var2 <= 0:
        raise ValueError(f"Non-positive variance for {p1} or {p2} in covariance matrix.")

    sigma1 = np.sqrt(var1)
    sigma2 = np.sqrt(var2)

    # Correlation coefficient rho
    rho = cov12 / (sigma1 * sigma2)
    # Clip to [-1, 1] for safety
    rho = float(np.clip(rho, -1.0, 1.0))

    # 2x2 correlation matrix as in the docs example
    cor = np.array([[1.0, rho],
                    [rho, 1.0]])

    # 1D sigmas in the order [p1, p2]
    sigmas = np.array([sigma1, sigma2])

    # Eigen-decomposition of the correlation matrix
    vals, vecs = scipy.linalg.eigh(cor)

    # Set up axes 
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    angles = np.linspace(0, 2 * np.pi, 400)

    # Ellipses in Δp space (centered at 0,0) 
    for n_sigma in nsigmas:
        # Same threshold as in PINT tutorial
        thresh = np.sqrt(
            scipy.stats.chi2(2).isf(2 * scipy.stats.norm.cdf(-n_sigma))
        )
        # Unit circle mapped by correlation
        points = thresh * (
            np.sqrt(vals[0]) * np.cos(angles)[:, None] * vecs[None, :, 0]
            + np.sqrt(vals[1]) * np.sin(angles)[:, None] * vecs[None, :, 1]
        )
        # Scale by 1D sigmas to get Δp1, Δp2 units
        ax.plot(
            points[:, 0] * sigmas[0],
            points[:, 1] * sigmas[1],
            label=f"{n_sigma} sigma"
        )

    # One-sigma rectangles in each coordinate, as in PINT tut.
    ax.axvspan(-sigmas[0], sigmas[0], alpha=0.3, label="1σ in " + p1)
    ax.axhspan(-sigmas[1], sigmas[1], alpha=0.3)

    ax.set_xlabel(r"$\Delta$" + f"{p1}")
    ax.set_ylabel(r"$\Delta$" + f"{p2}")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    return fig, ax
    

# -------------------------------------------------------------------------
# White noise comparison utilities (tables + plots)
# -------------------------------------------------------------------------

def collect_white_noise_params(model, run_label):
    """
    Extract EFAC / EQUAD / ECORR maskParameters from a PINT TimingModel.

    Returns
    -------
    pd.DataFrame with columns:
        run   : label for this model/run (e.g. 'BB+SWX', 'vanilla')
        kind  : 'EFAC', 'EQUAD', or 'ECORR'
        param   : full parameter name in the model (e.g. 'EFAC1')
        backend : value of the -f key (receiver/backend name)
        value : numeric value (unitless for EFAC, us for EQUAD/ECORR)
    """
    records = []

    for pname in model.params:
        if pname.startswith(("EFAC", "EQUAD", "ECORR")):
            p = getattr(model, pname)

            # backend flag name, if present
            try:
                backend = p.key_value[0]
            except Exception:
                backend = "GLOBAL"

            records.append(
                dict(
                    run=run_label,
                    kind=p.name,        # 'EFAC', 'EQUAD', 'ECORR'
                    param=pname,        # 'EFAC1', 'EQUAD3', etc.
                    backend=backend,    # e.g. 'GUPPI', 'PUPPI', 'CHIME'
                    value=p.value,      # numeric
                )
            )

    return pd.DataFrame.from_records(records)

def plot_wn_comparison(table, kind, run_labels=("Vanilla", "BB+SWX+DMX")):
    """
    Make a side-by-side bar plot of a single `kind` (EFAC/EQUAD/ECORR)
    for any two runs.
    """
    sub = table[table["kind"] == kind].copy()
    if sub.empty:
        print(f"No {kind} parameters found.")
        return

    backends = sub["backend"].values
    y1 = sub[run_labels[0]].values
    y2 = sub[run_labels[1]].values

    x = np.arange(len(backends))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, y1, width, label=run_labels[0])
    ax.bar(x + width/2, y2, width, label=run_labels[1])

    ax.set_xticks(x)
    ax.set_xticklabels(backends, rotation=45, ha="right")
    ax.set_ylabel(kind)
    ax.set_title(f"{kind} comparison per backend")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax

def _extract_white_noise_params(model, families=("EFAC", "EQUAD", "ECORR")):
    """
    Extract white-noise maskParameters from a PINT timing model.

    Returns a list of dicts with keys:
        'family' : str  (EFAC/EQUAD/ECORR/...)
        'backend': str  (value of key_value[0], e.g. backend code)
        'param'  : str  (full par name, e.g. 'EFAC1')
        'value'  : float
    """
    rows = []
    for par_name in model.params:
        # Only consider families of interest
        fam = None
        for f in families:
            if par_name.startswith(f):
                fam = f
                break
        if fam is None:
            continue

        par = getattr(model, par_name)

        # Require maskParameter-like structure (has key_value)
        if not hasattr(par, "key_value") or not par.key_value:
            continue

        backend = par.key_value[0]
        val = float(par.value)

        rows.append(
            {
                "family": fam,
                "backend": backend,
                "param": par_name,
                "value": val,
            }
        )
    return rows

def compare_white_noise(
    fo1,
    fo2,
    label1="model1",
    label2="model2",
    families=("EFAC", "EQUAD", "ECORR"),
    make_plots=True,
):
    """
    Compare EFAC/EQUAD/ECORR (and optionally other `families`) between
    any two PINT fitters that already have noise parameters attached
    (e.g. from noise_analysis_from_fitter).

    Parameters
    ----------
    fo1, fo2 : pint.fitter.Fitter
        Fitters with white-noise parameters in fo.model.
    label1, label2 : str
        Labels used in tables/plots for the two models.
    families : tuple of str
        Parameter families to compare (default: ('EFAC', 'EQUAD', 'ECORR')).
        You can extend this, e.g. ('EFAC','EQUAD','ECORR','DMEFAC','DMEQUAD').
    make_plots : bool
        If True, make one bar plot per family.

    Returns
    -------
    table : pandas.DataFrame
        Wide table with index=(family, backend) and columns=[label1, label2].
    tidy  : pandas.DataFrame
        Long "tidy" table with columns: family, backend, label, value.
    """
    # Step 1: Extract noise params from each model
    rows1 = _extract_white_noise_params(fo1.model, families=families)
    rows2 = _extract_white_noise_params(fo2.model, families=families)

    # Tag with model label
    for r in rows1:
        r["label"] = label1
    for r in rows2:
        r["label"] = label2

    all_rows = rows1 + rows2
    if not all_rows:
        print("No matching white-noise parameters found in either model.")
        return None, None

    tidy = pd.DataFrame(all_rows)

    # Step 2: Make a wide comparison table: (family, backend) x label
    table = (
        tidy.pivot_table(
            index=["family", "backend"],
            columns="label",
            values="value",
        )
        .sort_index()
    )

    print("\n=== White-noise comparison by (family, backend) ===")
    display(table)  # print it in notebook

    # Step 3: Make a quick bar plots for each family (optional)
    if make_plots:
        families_present = sorted(tidy["family"].unique())
        for fam in families_present:
            sub = tidy[tidy["family"] == fam].copy()
            if sub.empty:
                continue

            # Pivot to backend x label for this family
            tab_fam = (
                sub.pivot_table(
                    index="backend",
                    columns="label",
                    values="value",
                )
                .sort_index()
            )

            fig, ax = plt.subplots(figsize=(8, 4))
            tab_fam.plot(kind="bar", ax=ax)

            ax.set_title(f"{fam} comparison: {label1} vs {label2}")
            ax.set_ylabel(f"{fam} value")
            ax.set_xlabel("Backend")
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()

    return table, tidy

# -------------------------------------------------------------------------
# DMX parsing + DMX time-series plotters
# -------------------------------------------------------------------------

def simple_dmxparse(fitter):
    """
    Like pint.utils.dmxparse but avoids KeyErrors if some DMX_* labels
    are missing from the covariance matrix. Returns the same dict 
    structure, with dmx_verrs = sqrt(diag(DM_cov)).
    """
    model = fitter.model

    # Which DMX indices exist?
    try:
        mapping = model.get_prefix_mapping("DMX_")
    except ValueError as e:
        raise RuntimeError("No DMX parameters in model!") from e

    # Sort by integer index
    epochs   = sorted(mapping.keys())  
    # The "bins" names (e.g. ['0001','0002', ...])
    dmx_bins = [f"{idx:04d}" for idx in epochs]

    # Pull out amplitudes, uncertainties, and R1/R2's
    DMXs = np.array([getattr(model, f"DMX_{b}").value   for b in dmx_bins]) # PINT default pc / cm3
    R1   = np.array([getattr(model, f"DMXR1_{b}").value for b in dmx_bins]) # PINT default d
    R2   = np.array([getattr(model, f"DMXR2_{b}").value for b in dmx_bins])

    # Grab units
    val_unit = getattr(model, f"DMX_{dmx_bins[0]}").units
    mjd_unit = getattr(model, f"DMXR1_{dmx_bins[0]}").units

    # Mean-subtract dmx's
    mean_dmx = DMXs.mean()
    dmxs     = (DMXs - mean_dmx) * val_unit

    # Compute midpoints/eps
    eps = (R1 + R2) / 2
    eps = eps * mjd_unit

    # Try to get the "vErrs" from the covariance matrix
    try:
        used_cov = True
        labels = [f"DMX_{b}" for b in dmx_bins]
        cov = fitter.parameter_covariance_matrix.get_label_matrix(labels).matrix # DMX submatrix in bin order
        diag = np.diag(cov)
        # If cov already carries units, sqrt keeps them; otherwise, attach val_unit.
        if hasattr(diag, "unit"):
            dmx_vErrs = np.sqrt(diag)              # 1-sigma uncertainties - NOT variances..
        else:
            dmx_vErrs = np.sqrt(diag) * val_unit   
    except Exception:
        print(f"[simple_dmxparse] Warning: failed to get DMX errors from covariance: {e}")
        used_cov = False
        dmx_vErrs = np.full_like(DMXs, np.nan, dtype=float) * val_unit

    return {
        "dmxs":       dmxs,
        "dmx_verrs":  dmx_vErrs,
        "dmxeps":     eps,
        "r1s":        R1 * mjd_unit,
        "r2s":        R2 * mjd_unit,
        "bins":       dmx_bins,
        "mean_dmx":   mean_dmx * val_unit,
        "avg_dm_err": dmx_vErrs.mean(),
        "used_cov":   used_cov,
    }

def plot_simple_dmx_time(
    fitter,
    dmxparse_func,
    **plot_kwargs,
):
    """
    Replicate plot_dmx_time() from dmxparse but spruce it up. Takes simple_dmxparse(fitter) or dmxparse(fitter).

    Parameters
    ----------
    fitter : PINT Fitter
        PINT fitter object.
    dmxparse_func : callable
        A function that takes a fitter object and returns a dict with keys
        'dmxeps', 'dmxs', 'dmx_verrs', all astropy Quantities (just like PINT dmxparse output).
    **plot_kwargs :
        Optional plotting and diagnostic settings:
            figsize        : tuple, default (10, 4)
            color          : str, default "gray"
            fmt            : str, default "s"
            alpha          : float, default 1.0
            legend         : bool, default True
            title          : bool, default True
            title_add      : str or None
            ax             : matplotlib Axes or None
            diagnostics    : bool, default False
            z_thresh       : float, default 3.5
            mark_outliers  : bool, default True
            min_elong_mjds : array-like or None
            return_data    : bool, default False
    
    Returns
    -------
    fig, ax1
        Matplotlib Figure and Axes.
    (optionally) data : dict
        If return_data=True:
            "mjd"        : Quantity [d]
            "dmx"        : Quantity [DM units]
            "err"        : Quantity [DM units]
            "years"      : np.ndarray (float)
            "modified_z" : np.ndarray (float)
            "is_outlier" : bool mask
            "keep_mask"  : bool mask
    
    """
    # Defaults
    settings = dict(
        figsize=(10, 4),
        color="gray",
        fmt="s",
        alpha=1.0,
        legend=True,
        title=True,
        title_add=None,
        ax=None,
        diagnostics=False,
        z_thresh=3.5,
        mark_outliers=True,
        min_elong_mjds=None,
        return_data=False,
    )
    settings.update(plot_kwargs)
    
    # Parse DMX series
    d = dmxparse_func(fitter)
    mjd  = d["dmxeps"]     # [d]
    dmx  = d["dmxs"]       # [pc/cm3]
    err  = d["dmx_verrs"]  # [pc/cm3]

    # Float values for internal calculations (year axis and z-scores)
    mjd_val = mjd.to_value()     # [d]
    dmx_val = dmx.to_value()     # [pc/cm3]
    err_val = err.to_value()     # [pc/cm3]

    # Convert to decimal years
    years = mjd_to_year(mjd_val)

    # Use existing axis or create a new one
    if settings["ax"] is None:
        fig, ax1 = plt.subplots(figsize=settings["figsize"])
        twin = True
    else:
        ax1 = settings["ax"]
        fig = ax1.figure # Use figure that owns the axis
        twin = False

    # Outlier detection using modified z-score (robust to outliers)
    median_amp = np.nanmedian(dmx_val)
    mad_amp = np.nanmedian(np.abs(dmx_val - median_amp))
    if mad_amp == 0 or not np.isfinite(mad_amp):
        modified_z = np.zeros_like(dmx_val, dtype=float)
    else:
        modified_z = 0.6745 * (dmx_val - median_amp) / mad_amp

    is_outlier = np.abs(modified_z) > float(settings["z_thresh"])
    keep_mask = ~is_outlier

    # Plot the DMX values
    # x is dimensionless but in years, y and yerr are u.Quantities
    ax1.errorbar(
        years[keep_mask],
        dmx[keep_mask],          
        yerr=err[keep_mask],    
        fmt=settings["fmt"],
        c=settings["color"],
        alpha=settings["alpha"],
        label="DMX",
    )
    ax1.set_xlabel("Years")
    ax1.grid(True)

    # Optionals
    if settings["legend"]:
        ax1.legend(loc="best")

    if settings["title"]:
        psr = fitter.model.PSR.value
        full_title = f"{psr}: {settings['title_add']}" if settings["title_add"] else f"{psr}: DMX vs Time"
        ax1.set_title(full_title)

    # Mark outliers along y=0 
    if settings["mark_outliers"] and np.any(is_outlier):
        x_out = years[is_outlier]
        y_out = np.zeros_like(x_out) * dmx.unit
        
        ax1.scatter(
            x_out,
            y_out,  # baseline at 0 in the same DM units
            marker="|",
            linewidths=1.6,
            color="red",
            zorder=4,
            label=f"Outlier removed (z-score ≥ {settings['z_thresh']})",
        )

    # Conjunction lines
    if settings["min_elong_mjds"] is not None:
        for k, mj in enumerate(np.asarray(settings["min_elong_mjds"], float)):
            ax1.axvline(mjd_to_year(mj), 
                        color="orange", 
                        linestyle=":", 
                        alpha=0.4,
                        label="Conjunction" if k == 0 else None)

    # Twin top x-axis with MJD scale (years to MJD floats).
    if twin:
        ax2 = ax1.twiny()
        x0, x1 = ax1.get_xlim()
        mjd0 = year_to_mjd(x0)
        mjd1 = year_to_mjd(x1)
        ax2.set_xlim(mjd0, mjd1)
        ax2.set_xlabel("MJD")

    if settings["return_data"]:
        data = dict(
            mjd=mjd, dmx=dmx, err=err, years=years,
            modified_z=modified_z, is_outlier=is_outlier, keep_mask=keep_mask
        )
        return fig, ax1, data

    fig.tight_layout()
    return fig, ax1


# -------------------------------------------------------------------------
# Generic stats/reporting utilities
# -------------------------------------------------------------------------

def zscore_filter(
    data: np.ndarray,
    *,
    z_thresh: float = 3.5,
    return_diagnostics: bool = True,
    diagnostics: str = "on",
):
    """
    Perform modified Z-score filtering on input data.

    Uses the robust estimator:
        Modified_Z_i = 0.6745 * (x_i - median(x)) / MAD(x)
    where MAD is the median absolute deviation.

    Parameters
    ----------
    data : array-like
        Input numeric data (1D array). Can include NaN values.
    z_thresh : float, default 3.5
        Threshold for defining outliers via |Z| > z_thresh.
    return_diagnostics : bool, default True
        If True, return (filtered_data, mask, diagnostics_dict).
        If False, return just filtered_data.
    diagnostics : str, default "on"
        Controls printed output. Use "off" to suppress prints.

    Returns
    -------
    filtered_data : np.ndarray
        Data with outliers removed (NaN where filtered out).
    mask : np.ndarray of bool
        Boolean mask where True = kept, False = outlier.
    diag : dict (optional)
        {
          "median": float,
          "mad": float,
          "modified_z": np.ndarray,
          "is_outlier": np.ndarray,
          "keep_mask": np.ndarray,
          "n_outliers": int
        }
    """
    data = np.asarray(data, dtype=float)
    finite_mask = np.isfinite(data)
    vals = data[finite_mask]

    if vals.size == 0:
        raise ValueError("Input data array is empty or entirely NaN.")

    median_val = np.nanmedian(vals)
    mad_val = np.nanmedian(np.abs(vals - median_val))

    if mad_val == 0 or not np.isfinite(mad_val):
        modified_z = np.zeros_like(data, dtype=float)
    else:
        modified_z = np.full_like(data, np.nan, dtype=float)
        modified_z[finite_mask] = 0.6745 * (vals - median_val) / mad_val

    is_outlier = np.abs(modified_z) > z_thresh
    keep_mask = ~is_outlier
    filtered_data = np.where(keep_mask, data, np.nan)

    n_out = int(np.nansum(is_outlier))
    if diagnostics.lower() != "off":
        print(f"→ Identified {n_out} outlier(s) via |Z| > {z_thresh}")
        if n_out > 0:
            print("{:<10} {:>15} {:>15}".format("Index", "Value", "Modified Z"))
            print("-" * 45)
            for i, (v, z) in enumerate(zip(data, modified_z)):
                if is_outlier[i]:
                    print(f"{i:<10d} {v:>15.4e} {z:>15.2f}")
            print("")

    diag = {
        "median": median_val,
        "mad": mad_val,
        "modified_z": modified_z,
        "is_outlier": is_outlier,
        "keep_mask": keep_mask,
        "n_outliers": n_out,
    }

    if return_diagnostics:
        return filtered_data, keep_mask, diag
    return filtered_data

def violation_table(toas, gaps, edges, mintoas, mintime, maxtime):
    """
    Print a summary table showing which DMX blocks violate mintoas, mintime, 
    or maxtime, and which are gaps.
    Gap blocks are allowed to violate mintoas and maxtime without being flagged.
    """
    mjds = toas.get_mjds().value
    bin_widths = np.diff(edges)
    toa_counts = np.histogram(mjds, bins=edges)[0]

    # Grab gap blocks
    gap_flags = np.zeros(len(bin_widths), dtype=bool)
    for i, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        for g0, g1 in gaps:
            if start >= g0 and end <= g1:
                gap_flags[i] = True
                break

    # Build violator rows
    violations = []
    for i, (start, end, width, count, is_gap) in enumerate(zip(edges[:-1], edges[1:], bin_widths, toa_counts, gap_flags), start=1):
        mintime_v = width < mintime.to_value(u.d)
        maxtime_v = (width > maxtime.to_value(u.d)) and not is_gap  # gaps can be long > gap_threshold spacing (~< 100 days)
        mintoas_v = (count < mintoas) and not is_gap                # gaps can have no TOAs - should be 0!

        if any([mintime_v, maxtime_v, mintoas_v, is_gap]):
            violations.append({
                "Index": f"DMX_{i:04d}",
                "Start": start,
                "End": end,
                "Width": width,
                "TOAs": count,
                "Gap": is_gap,
                "mintime": mintime_v,
                "maxtime": maxtime_v,
                "mintoas": mintoas_v
            })

    if not violations:
        print("No DMX blocks violate the specified constraints!")
        return

    # Print formatted table of violating bins
    print("Violating DMX Blocks Summary:")
    header = "{:<10} {:>12} {:>12} {:>12} {:>7} {:>6} {:>9} {:>9} {:>9}".format(
        "DMX Index", "Start MJD", "End MJD", "Width (d)", "TOAs", "Gap", "mintime", "maxtime", "mintoas"
    )
    print(header)
    print("-" * len(header))
    # Print columns w/emojis!!! ?
    for v in violations:
        print("{:<10} {:>12.2f} {:>12.2f} {:>12.2f} {:>7} {:>6} {:>7} {:>8} {:>9}".format(
            v["Index"],
            v["Start"],
            v["End"],
            v["Width"],
            v["TOAs"],
            "✓" if v["Gap"] else "❌",
            "❌" if v["mintime"] else "✓",
            "❌" if v["maxtime"] else "✓",
            "❌" if v["mintoas"] else "✓"
        ))

    # Print note
    print("\nNOTE:")
    print("- Gap blocks (Gap = ✓) are allowed to violate 'mintoas' (should be 0) and 'maxtime' (spacing > gap_threshold) without issue.")
    print("- All other ❌ values should be investigated and may indicate problems with binning.")

def summary(bins, mjds=None, frequency_table=False):
    """
    Print a formatted summary of Bayesian Block interval widths and statistics,
    and optionally a frequency table with DMX-style indexing (mjds required).
    """
    bins = sorted(list(bins))
    if len(bins) < 2:
        print("Not enough bins to compute intervals.")
        return

    bin_widths = np.diff(np.array(bins))
    n_blocks = len(bin_widths)

    # Statistical metrics
    min_width = np.min(bin_widths)
    max_width = np.max(bin_widths)
    mean_width = np.mean(bin_widths)
    median_width = np.median(bin_widths)
    std_width = np.std(bin_widths)

    # Rounded to nearest 0.5 day for mode grouping
    rounded_widths = np.round(bin_widths * 2) / 2
    width_counts = Counter(rounded_widths)
    mode_width, mode_count = width_counts.most_common(1)[0]

    def count_occurrences(value):
        return np.sum(np.isclose(bin_widths, value, atol=1e-6))

    # Print summary
    print("Summary of BB Intervals:")
    print("{:<40} {:>15} {:>15}".format("Metric", "Value", "Occurrences"))
    print("-" * 75)
    print("{:<40} {:>15} {:>15}".format("Number of Blocks", n_blocks, "—"))
    print("{:<40} {:>15.2e} {:>15}".format("Min Width (days)", min_width, count_occurrences(min_width)))
    print("{:<40} {:>15.2e} {:>15}".format("Max Width (days)", max_width, count_occurrences(max_width)))
    print("{:<40} {:>15.2e} {:>15}".format("Mean Width (days)", mean_width, count_occurrences(mean_width)))
    print("{:<40} {:>15.2e} {:>15}".format("Median Width (days)", median_width, count_occurrences(median_width)))
    print("{:<40} {:>15.2e} {:>15}".format("Std Dev Width (days)", std_width, "—"))
    print("{:<40} {:>15.2e} {:>15}".format("Most Common Width (days)", mode_width, mode_count))
    print("")

    if frequency_table:
        if mjds is None:
            raise ValueError("mjds must be provided for frequecy table to define toa counts.")
        # Bin-wise TOA counts
        toa_counts = np.histogram(mjds, bins=bins)[0] if mjds is not None else None

        print("Frequency Table of BB Intervals:")
        header = "{:<10} {:>15} {:>15} {:>15}".format("DMX Index", "Width (days)", "Rounded Width", "TOAs in Bin")
        print(header)
        print("-" * len(header))

        for i, (left, right, width) in enumerate(zip(bins[:-1], bins[1:], bin_widths), start=1):
            index = f"DMX_{i:04d}"
            rounded = np.round(width * 2) / 2
            n_toas = toa_counts[i - 1] if toa_counts is not None else "—"
            print(f"{index:<10} {width:>15.2f} {rounded:>15.1f} {n_toas:>15}")

def summarize_fitter(f, *, label=None, print_summary=True):
    """Print small fit parameter table"""
    m = f.model
    r = f.resids

    free_params = list(m.free_params)

    summary = {
        "PSR": getattr(getattr(m, "PSR", None), "value", None),
        "N_TOAs": int(len(f.toas.table)),
        "N_free_params": len(free_params),
        "chi2": float(getattr(r, "chi2", float("nan"))),
        "reduced_chi2": float(getattr(r, "reduced_chi2", float("nan"))),
    }

    if print_summary:
        if label:
            print(f"\n=== {label} ===")
        for k, v in summary.items():
            print(f"{k:15s}: {v}")

    return summary
