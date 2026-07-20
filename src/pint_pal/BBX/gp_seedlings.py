# BBX/gp_seedlings.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import astropy.units as u

from astropy.timeseries import LombScargle
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .utils import remove_gap_nodes

# Frequency grid strategies for constructing Fourier GP mode arrays
GridStrategy = Literal[
    "bb_harmonic",
    "standard",
    "custom",
    "hybrid",
]

@dataclass(frozen=True)
class FourierGPConfig:
    """
    Configuration for constructing a Fourier-GP mode array from BB products.
    """

    strategy: GridStrategy = "bb_harmonic"

    # Grid controls
    nfreqs: Optional[int] = None
    spacing_hz: Optional[float] = None
    spacing_rule: Optional[str] = None

    # BB spacing-frequency support
    central_fraction: float = 0.95
    # Central fraction used to estimate robust high-frequency support.
    # The low-frequency support remains fmin = 1 / s_max. 

    # Periodogram settings
    use_taper: bool = True
    taper_fraction: float = 0.20 # Total taper, `taper_fraction`/2 applied to each edge.
    smooth_periodogram: bool = True
    smooth_sigma_bins: float = 3.0

    peak_count: int = 3
    peak_prominence_fraction: float = 0.01
    bootstrap_fap: bool = True

    # Basis diagnostic
    evaluate_basis: bool = True

    # Sanity
    def validate(self) -> None:
        if self.strategy not in {
            "standard",
            "custom",
            "bb_harmonic",
            "hybrid",
        }:
            raise ValueError(f"Unknown strategy {self.strategy!r}.")

        if self.nfreqs is not None and self.nfreqs < 1:
            raise ValueError("nfreqs must be >= 1.")

        if self.spacing_hz is not None and self.spacing_hz <= 0:
            raise ValueError("spacing_hz must be positive.")

        if self.strategy == "custom":
            if self.spacing_hz is None:
                raise ValueError("custom requires spacing_hz.")
            if self.nfreqs is None:
                raise ValueError("custom requires nfreqs.")
            if not self.spacing_rule:
                raise ValueError("custom requires spacing_rule.")

        if self.strategy == "hybrid":
            if self.spacing_hz is None:
                raise ValueError("hybrid requires spacing_hz.")
            if not self.spacing_rule:
                raise ValueError("hybrid requires spacing_rule.")
            if self.nfreqs is not None:
                raise ValueError(
                    "hybrid derives nfreqs from the BB fmax; "
                    "use custom to specify both spacing_hz and nfreqs."
                )

        if not 0.0 < self.central_fraction < 1.0:
            raise ValueError("central_fraction must lie between 0 and 1.")

        if not 0.0 <= self.taper_fraction <= 1.0:
            raise ValueError("taper_fraction must lie between 0 and 1.")

        if self.smooth_sigma_bins < 0:
            raise ValueError("smooth_sigma_bins must be non-negative.")

        if self.peak_count < 1:
            raise ValueError("peak_count must be >= 1.")


@dataclass(frozen=True)
class BBSamplingSeries:
    """
    Gap-cleaned BB-width series and its independent-variable sampling. 

    Container for the intermediate BB-derived products.
    """

    edges: np.ndarray
    widths: np.ndarray
    midpoints: np.ndarray
    spacings: np.ndarray
    spacing_frequencies_cpd: np.ndarray

    removed_gap_indices: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )

@dataclass(frozen=True)
class BBFrequencySupport:
    """
    Raw frequency support inferred from BB midpoint-spacing statistics.

    Notes
    -----
    These values describe the BB-derived target frequency support before
    projection onto a discrete Fourier grid:

        fmin_hz = 1 / s_max
        fmax_hz = 2 / s_effective

    For the ``bb_harmonic`` strategy, the final model grid is constructed by
        projecting the raw lower-frequency scale onto the nearest integer harmonic
        of the observing span:

            N_BB = round(T / s_max)
            df_model = N_BB / T

        The number of modes is then chosen so that the final upper frequency is the
        nearest available grid frequency to the raw BB-derived fmax:

            nfreqs = round(fmax_hz / df_model)
            fmin_model = df_model
            fmax_model = nfreqs * df_model

        The raw support and final model frequencies therefore generally differ
        slightly. The integer-harmonic construction preserves the continuous-time
        integer-cycle condition over T; orthogonality under the actual irregular,
        weighted TOA sampling is evaluated with the weighted Gram matrix.

    For the ``hybrid`` strategy, the raw BB-derived lower-frequency scale is not
        used to construct the grid spacing. The user-supplied ``spacing_hz`` is retained
        without projection, and only the BB-derived upper-frequency target is mapped
        onto the nearest available grid mode:

            nfreqs = round(fmax_hz / spacing_hz)
            fmax_model = nfreqs * spacing_hz

        The final hybrid-grid fmin is therefore the user-supplied spacing, not the raw
        BB-derived ``fmin_hz``. The hybrid grid satisfies the integer-cycle condition
        only when ``spacing_hz * T`` is an integer.
    """

    s_min_days: float
    s_median_days: float
    s_max_days: float
    s_effective_days: float

    q_low_cpd: float
    q_high_cpd: float

    fmin_cpd: float
    fmax_cpd: float
    fmin_hz: float
    fmax_hz: float

    central_mask: np.ndarray


@dataclass(frozen=True)
class PeriodogramResult:
    """Lomb–Scargle diagnostic for the BB-width time series."""

    frequency_cpd: np.ndarray
    period_days: np.ndarray

    power_raw: np.ndarray
    power_smoothed: np.ndarray

    taper: np.ndarray
    values_used: np.ndarray

    peak_indices: np.ndarray
    peak_frequencies_cpd: np.ndarray
    peak_periods_days: np.ndarray
    peak_power_raw: np.ndarray
    peak_power_smoothed: np.ndarray
    peak_fap: Optional[np.ndarray] = None


@dataclass(frozen=True)
class FourierBasisResult:
    """Numerical conditioning summary for a Fourier basis."""

    gram: np.ndarray
    gram_correlation: np.ndarray

    max_offdiag: float
    median_offdiag: float

    condition_design: float
    condition_gram: float

    n_modes: int
    n_basis_columns: int


@dataclass(frozen=True)
class FourierGridResult:
    """
    Final Fourier GP mode array and associated BB-derived products.

    Notes
    -----
    ``modes_hz`` contains the actual frequencies passed to the Fourier GP.
    Therefore, ``spacing_hz``, ``fmin_model_hz``, ``fmax_model_hz``, and
    ``nfreqs`` describe the final model grid rather than the unprojected
    BB-derived frequency targets.

    For ``bb_harmonic``, both the characteristic spacing and upper-frequency
        extent are projected onto the nearest integer harmonic of the observing span:

            N_BB = round(T / s_max)
            spacing_hz = N_BB / T
            nfreqs = round(support.fmax_hz / spacing_hz)

    For ``hybrid``, the user-supplied spacing is retained exactly. Only the
        upper-frequency extent is projected:

            spacing_hz = user spacing
            nfreqs = round(support.fmax_hz / spacing_hz)

        Consequently, a hybrid grid's first frequency is determined by the user,
        not by ``support.fmin_hz``. Its final upper frequency may lie slightly
        above or below the BB-derived target. A hybrid grid satisfies the
        continuous-time integer-cycle condition only when ``spacing_hz * T`` is
        an integer.

    The raw BB-derived target values remain available through ``support``,
    while the final frequencies used by the GP are stored directly in this
    result object.
    """

    strategy: GridStrategy

    modes_hz: np.ndarray
    spacing_hz: float

    fmin_model_hz: float
    fmax_model_hz: float
    nfreqs: int

    data_span_days: float
    data_span_seconds: float

    sampling: Optional[BBSamplingSeries] = None
    support: Optional[BBFrequencySupport] = None
    periodogram: Optional[PeriodogramResult] = None
    basis: Optional[FourierBasisResult] = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

def prepare_bb_sampling_series(
    edges: np.ndarray,
    gaps: Optional[Sequence[Tuple[float, float]]] = None,
    *,
    gap_match_atol: float = 1e-8,
) -> BBSamplingSeries:
    """Construct the gap-cleaned BB-width series w(t_mid)."""
    edges = np.asarray(edges, dtype=float)

    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(
            "edges must be a one-dimensional array with >= 2 entries."
        )
    if not np.all(np.isfinite(edges)):
        raise ValueError("edges contains non-finite values.")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("edges must be strictly increasing.")

    widths0 = np.diff(edges)
    midpoints0 = 0.5 * (edges[:-1] + edges[1:])

    gap_list = [] if gaps is None else list(gaps)

    if gap_list:
        gap_nodes = np.asarray(
            [0.5 * (left + right) for left, right in gap_list],
            dtype=float,
        )

        midpoints, removed_gap_indices = remove_gap_nodes(
            midpoints0,
            gap_nodes,
            atol=float(gap_match_atol),
        )

        widths = np.delete(widths0, removed_gap_indices)
    else:
        midpoints = midpoints0.copy()
        widths = widths0.copy()
        removed_gap_indices = np.array([], dtype=int)

    valid = (
        np.isfinite(widths)
        & np.isfinite(midpoints)
        & (widths > 0)
    )

    widths = widths[valid]
    midpoints = midpoints[valid]

    if widths.size < 5:
        raise RuntimeError(
            "Too few gap-cleaned BB intervals for frequency analysis."
        )

    order = np.argsort(midpoints)
    widths = widths[order]
    midpoints = midpoints[order]

    spacings = np.diff(midpoints)
    spacings = spacings[
        np.isfinite(spacings) & (spacings > 0)
    ]

    if spacings.size < 3:
        raise RuntimeError(
            "Too few midpoint spacings to estimate BB frequency support."
        )

    return BBSamplingSeries(
        edges=edges,
        widths=widths,
        midpoints=midpoints,
        spacings=spacings,
        spacing_frequencies_cpd=1.0 / spacings,
        removed_gap_indices=removed_gap_indices,
    )

def estimate_bb_frequency_support(
    sampling: BBSamplingSeries,
    *,
    central_fraction: float = 0.95,
) -> BBFrequencySupport:
    """
    Estimate Fourier GP frequency support from BB midpoint spacings.

    The low frequency bound is set by the largest gap-cleaned
    midpoint spacing,

        fmin = 1 / s_max,

    while the upper frequency bound is set by the adopted Fourier-resolution
    convention,

        fmax = 2 / s_eff,

    where s_eff is obtained from the upper quantile of the `central_fraction` 
    percentile of the midpoint-spacing-frequency distribution e.g. frequency 
    at the 97.5th percentile for a central_fraction=0.95.

    The resulting bounds describe the BB-supported frequency range. They are
    distinct from the final model frequencies, which may be adjusted onto a
    specified harmonic grid to preserve the integer cycle/orthogonality.

    Parameters
    ----------
    sampling
        Gap-cleaned BB midpoint-spacing series.
    central_fraction
        Fraction of the midpoint-spacing-frequency distribution used to use for 
        estimating the effective frequency support. The default is 0.95,
        which uses the 2.5th and 97.5th percentiles of the spacing-frequency
        distribution to estimate the effective frequency support.

    Returns
    -------
    BBFrequencySupport
        Frequency support inferred from the BB midpoint-spacing statistics.
    """
    if not 0.0 < central_fraction < 1.0:
        raise ValueError("central_fraction must lie between 0 and 1.")

    spacings = sampling.spacings
    spacing_freq = sampling.spacing_frequencies_cpd

    # Use the central fraction of the spacing-frequency distribution
    tail = 0.5 * (1.0 - central_fraction)
    q_low_pct = 100.0 * tail
    q_high_pct = 100.0 * (1.0 - tail) # upper quartile

    # Compute the low/high quantiles of the spacing-frequency distribution
    # q_low = f_{2.5%}, q_high = f_{97.5%} for a central_fraction=0.95
    q_low, q_high = np.nanpercentile(
        spacing_freq,
        [q_low_pct, q_high_pct],
    )

    central_mask = (
        (spacing_freq >= q_low)
        & (spacing_freq <= q_high)
    )

    s_min = float(np.nanmin(spacings))
    s_median = float(np.nanmedian(spacings))
    s_max = float(np.nanmax(spacings))

    s_effective = 1.0 / float(q_high)

    # Frequency bounds [cycles per day]
    fmin_cpd = 1.0 / s_max
    fmax_cpd = 2.0 / s_effective  # aka 2*q_high

    return BBFrequencySupport(
        s_min_days=s_min,
        s_median_days=s_median,
        s_max_days=s_max,
        s_effective_days=s_effective,
        q_low_cpd=float(q_low),
        q_high_cpd=float(q_high),
        fmin_cpd=float(fmin_cpd),
        fmax_cpd=float(fmax_cpd),
        fmin_hz=float(fmin_cpd / 86400.0),
        fmax_hz=float(fmax_cpd / 86400.0),
        central_mask=central_mask,
    )

def _modes_from_spacing(
    *,
    spacing_hz: float,
    nfreqs: int,
) -> np.ndarray:
    """Construct modes f_n = n df for n = 1, ..., Nfreq."""
    spacing_hz = float(spacing_hz)
    nfreqs = int(nfreqs)

    if not np.isfinite(spacing_hz) or spacing_hz <= 0:
        raise ValueError("spacing_hz must be finite and positive.")

    if nfreqs < 1:
        raise ValueError("nfreqs must be >= 1.")

    return np.arange(1, nfreqs + 1, dtype=float) * spacing_hz


def construct_fourier_grid(
    *,
    config: FourierGPConfig,
    data_span_seconds: float,
    support: Optional[BBFrequencySupport] = None,
) -> Tuple[np.ndarray, Mapping[str, Any]]:
    """
    Construct a Fourier GP mode array using the configured strategy.
    
    Notes
    -----
    The grid strategy determines how the Fourier frequencies are constructed:
        
        - standard:    1/T spacing, default 100 modes
        - custom:      user spacing plus user mode count
        - bb_harmonic: BB spacing plus BB fmax
        - hybrid:      user spacing plus BB fmax

    BB-derived frequency bounds are treated as target values. Because the
    returned Fourier grid is discrete, the target spacing and upper frequency
    may not be represented exactly. The ``bb_harmonic`` strategy projects the
    raw BB spacing onto the nearest integer harmonic of T and selects the
    nearest number of modes to the BB-derived fmax. The ``hybrid`` strategy
    retains the user-supplied spacing and projects only the upper-frequency
    extent onto the nearest available mode.
    """
    config.validate()

    T_sec = float(data_span_seconds)
    if not np.isfinite(T_sec) or T_sec <= 0:
        raise ValueError("data_span_seconds must be finite and positive.")

    strategy = config.strategy
    metadata: dict[str, Any] = {"strategy": strategy}

    # Standard PTA-style Fourier grid: 1/T spacing, default 100/input modes
    if strategy == "standard":
        nfreqs = 100 if config.nfreqs is None else int(config.nfreqs)
        spacing_hz = 1.0 / T_sec

        modes = _modes_from_spacing(
            spacing_hz=spacing_hz,
            nfreqs=nfreqs,
        )

        metadata.update(
            spacing_rule="1/T",
            requested_nfreqs=nfreqs,
        )

    # Custom grid: user-specified spacing and mode count
    elif strategy == "custom":
        if config.spacing_hz is None or config.nfreqs is None:
            raise ValueError(
                "custom strategy requires spacing_hz and nfreqs."
            )
        spacing_hz = float(config.spacing_hz)
        nfreqs = int(config.nfreqs)

        modes = _modes_from_spacing(
            spacing_hz=spacing_hz,
            nfreqs=nfreqs,
        )

        metadata.update(
            spacing_rule=str(config.spacing_rule),
            requested_nfreqs=nfreqs,
        )

    # BB-derived grid: spacing from BB, fmax from BB
    elif strategy == "bb_harmonic":
        if support is None:
            raise ValueError(
                "bb_harmonic strategy requires BB frequency support."
            )

        T_days = T_sec / 86400.0

        # Factor to preserve integer-cycle spacing: N_BB = round(T / s_max)
        N_BB = max(
            1,
            int(np.round(T_days / support.s_max_days)), # dimensionless
        )

        # Project the BB-derived spacing onto the integer-cycle grid: df = N_BB / T
        spacing_hz = N_BB / T_sec
        nfreqs = int(np.round(support.fmax_hz / spacing_hz)) # modes

        modes = _modes_from_spacing(
            spacing_hz=spacing_hz,
            nfreqs=nfreqs,
        )

        metadata.update(
            spacing_rule="N_BB/T",
            N_BB=N_BB,
            target_fmin_hz=float(support.fmin_hz),
            target_fmax_hz=float(support.fmax_hz),
        )

    # Hybrid grid: user-specified spacing, BB-derived fmax
    elif strategy == "hybrid":
        if support is None:
            raise ValueError(
                "hybrid strategy requires BB frequency support."
            )

        if config.spacing_hz is None:
            raise ValueError(
                "hybrid strategy requires spacing_hz."
            )
        
        # Retain the user spacing and select the mode nearest to the BB-derived fmax.
        spacing_hz = float(config.spacing_hz)
        nfreqs = int(np.round(support.fmax_hz / spacing_hz))

        if nfreqs < 1:
            raise ValueError(
                "The BB-derived frequency spacing exceeds the BB-derived fmax; "
                "no Fourier modes can be constructed."
            )

        modes = _modes_from_spacing(
            spacing_hz=spacing_hz,
            nfreqs=nfreqs,
        )

        metadata.update(
            spacing_rule=str(config.spacing_rule),
            spacing_projection="none",
            fmax_projection="nearest_grid_mode",
            target_fmax_hz=float(support.fmax_hz),
            derived_nfreqs=nfreqs,
        )

        # Extra to track upper projection
        metadata["fmax_projection_offset_hz"] = float(
            modes[-1] - support.fmax_hz
        )

    else:
        raise RuntimeError(f"Unknown strategy: {strategy!r}")

    spacing_index = spacing_hz * T_sec
    metadata["spacing_index"] = float(spacing_index)
    metadata["integer_cycle_spacing"] = bool(
        np.isclose(
            spacing_index,
            np.round(spacing_index),
            rtol=0.0,
            atol=1e-10, # 9 us precision
        )
    )

    # Extras to track projection offsets from the 
    # BB-derived frequency support, if available.
    metadata.update(
        actual_spacing_hz=float(spacing_hz),
        actual_fmin_hz=float(modes[0]),
        actual_fmax_hz=float(modes[-1]),
        actual_nfreqs=int(modes.size),
    )

    if support is not None:
        metadata["fmin_projection_offset_hz"] = float(
            modes[0] - support.fmin_hz
        )
        metadata["fmax_projection_offset_hz"] = float(
            modes[-1] - support.fmax_hz
        )

    return modes, metadata

def compute_bb_width_periodogram(
    sampling: BBSamplingSeries,
    support: BBFrequencySupport,
    config: FourierGPConfig,
    *,
    samples_per_peak: float = 5.0,
    minimum_grid_size: int = 10_000,
) -> PeriodogramResult:
    """
    Compute a Lomb–Scargle diagnostic for BB width as a function of midpoint.
    
    Either with or without tapering, the BB width series is evaluated on a dense
    logarithmic frequency grid. The resulting periodogram is optionally smoothed
    and the most prominent peaks are identified. The false-alarm probability (FAP)
    of each peak is optionally estimated using bootstrap resampling.

    Notes
    -----
    Tapering:
        The tapering is applied to the BB width series to reduce edge effects in the
        Lomb–Scargle periodogram. The taper is a smooth cosine function that goes to
        zero at the edges of the time series. The taper fraction specifies the total
        fraction of the time series that is tapered, with half of the taper applied
        to each edge. For example, a taper_fraction of 0.2 applies a 10% taper to the 
        left edge and a 10% taper to the right edge, for a total taper of 20% of the 
        time series length.

    Smoothing:
        The periodogram can be optionally smoothed using a Gaussian kernel in frequency
        space. The width of the kernel is specified in units of frequency bins. Smoothing
        is applied to the raw Lomb–Scargle powers, not the tapered values. Guassian 
        smoothing can help to reduce noise in the periodogram, stabalize spectral density 
        estimates, and make prominent peaks more apparent for interpreation ease.

    False-Alarm Probability (FAP):
        The FAP of each identified peak can be estimated using bootstrap resampling.
        The FAP is the probability that a peak of equal or greater power could arise 
        from random noise. It is not a measure of significance from the true signal.
        The FAP determination for multiple peaks are independent and should be used
        purely for diagnostics.
    """
    config.validate()

    t = sampling.midpoints
    values = sampling.widths - np.nanmean(sampling.widths) # demean

    fmin = support.fmin_cpd
    fmax = support.fmax_cpd

    Tspan_days = float(np.nanmax(t) - np.nanmin(t))

    # Adopted from VanderPlas 2018, "Understanding the Lomb–Scargle Periodogram"
    n_frequency = max(
        int(samples_per_peak * Tspan_days * fmax),
        int(minimum_grid_size),
    )

    frequency = np.logspace(
        np.log10(fmin),
        np.log10(fmax),
        n_frequency,
    )

    period = 1.0 / frequency

    # Create a mask to apply tapering to reduce edge effects in the Lomb–Scargle periodogram.
    taper = np.ones_like(values)

    if config.use_taper:
        if not 0.0 <= config.taper_fraction <= 1.0:
            raise ValueError("taper_fraction must lie between 0 and 1.")

        # Normalize the time series to [0, 1] for tapering
        x = (t - np.nanmin(t)) / (
            np.nanmax(t) - np.nanmin(t)
        )

        edge_fraction = config.taper_fraction / 2.0

        if edge_fraction > 0:
            left = x < edge_fraction
            right = x > (1.0 - edge_fraction)

            # Taper is a smooth cosine function, no edge discontinuities. 
            # The taper is 1.0 in the central region and smoothly goes to 0.0 at the edges.
            taper[left] = 0.5 * (
                1.0
                - np.cos(np.pi * x[left] / edge_fraction)
            )

            taper[right] = 0.5 * (
                1.0
                - np.cos(
                    np.pi
                    * (1.0 - x[right])
                    / edge_fraction
                )
            )

    # Apply taper
    values_used = values * taper

    ls = LombScargle(t, values_used)
    power_raw = ls.power(frequency)

    # Smooth the periodogram 
    if config.smooth_periodogram and config.smooth_sigma_bins > 0:
        power_smoothed = gaussian_filter1d(
            power_raw,
            sigma=float(config.smooth_sigma_bins),
        )
    else:
        power_smoothed = power_raw.copy()

    prominence = (
        float(np.nanmax(power_smoothed))
        * float(config.peak_prominence_fraction)
    )

    peaks, _properties = find_peaks(
        power_smoothed,
        prominence=prominence,
    )

    if peaks.size == 0:
        peaks = np.array(
            [int(np.nanargmax(power_smoothed))],
            dtype=int,
        )

    # Sort peaks by descending power and select the top `peak_count` peaks
    order = np.argsort(power_smoothed[peaks])[::-1]
    peak_indices = peaks[order][: int(config.peak_count)]

    peak_fap = None

    if config.bootstrap_fap:
        # Peak positions are selected from the smoothed curve, but FAP is
        # evaluated using the corresponding raw LS powers for accurate noise assessment.
        peak_fap = np.asarray(
            ls.false_alarm_probability(
                power_raw[peak_indices],
                minimum_frequency=fmin,
                maximum_frequency=fmax,
                method="bootstrap",
            ),
            dtype=float,
        )

    return PeriodogramResult(
        frequency_cpd=frequency,
        period_days=period,
        power_raw=power_raw,
        power_smoothed=power_smoothed,
        taper=taper,
        values_used=values_used,
        peak_indices=peak_indices,
        peak_frequencies_cpd=frequency[peak_indices],
        peak_periods_days=period[peak_indices],
        peak_power_raw=power_raw[peak_indices],
        peak_power_smoothed=power_smoothed[peak_indices],
        peak_fap=peak_fap,
    )

def evaluate_fourier_basis(
    toas,
    modes_hz: np.ndarray,
    *,
    chromatic_scale: Optional[np.ndarray] = None,
) -> FourierBasisResult:
    """
    Evaluate weighted overlap and numerical conditioning of a Fourier basis.

    This function constructs the sine/cosine design matrix associated with the
    supplied Fourier frequencies,

        F = [sin(2*pi f_1 t), cos(2*pi f_1 t), ..., sin(2*pi f_N t), cos(2*pi f_N t)],

    evaluated at the actual TOA epochs. For ``N`` Fourier frequencies, the
    design matrix therefore contains ``2N`` columns.

    In continuous time with uniform coverage, Fourier harmonics that complete
    integer numbers of cycles across the data span are orthogonal. Pulsar TOAs,
    however, are irregularly sampled and have heteroscedastic uncertainties.
    The continuous-time orthogonality condition therefore does not guarantee
    that the sampled design-matrix columns remain independent.

    To quantify the sampled basis geometry, the function forms the weighted
    design matrix

        F_w = W^(1/2) F,

    where

        W = diag(1 / sigma_i^2)

    and ``sigma_i`` is the uncertainty of TOA ``i``. Weighting is important
    because TOAs with smaller uncertainties contribute more strongly to the
    timing likelihood and therefore should contribute more strongly to the
    basis-overlap diagnostic.

    The weighted Gram matrix is

        G = F_w.T @ F_w = F.T @ W @ F.

    If the sampled, weighted basis columns were perfectly orthogonal, ``G``
    would be diagonal. Its off-diagonal elements quantify overlap between
    different sine/cosine basis vectors (e.g. correlation).

    The correlation-normalized Gram matrix is

        Gcorr_ij = G_ij / sqrt(G_ii G_jj).

    Its diagonal elements are unity, while the absolute off-diagonal elements
    behave like pairwise basis correlations. The maximum off-diagonal value
    identifies the most strongly overlapping pair of columns, while the median
    off-diagonal value summarizes typical mode overlap.

    This function reports two condition numbers:

    ``condition_design``
        Condition number of ``F_w``. This directly quantifies numerical
        sensitivity of the weighted design matrix e.g. is it stable?

    ``condition_gram``
        Condition number of ``G``. Because ``G = F_w.T @ F_w``, this is
        approximately the square of the design-matrix condition number and is
        therefore more sensitive to near-degeneracy e.g. the degree of correlation.

    If ``chromatic_scale`` is provided, each row of the Fourier design matrix
    is multiplied by the corresponding per-TOA chromatic factor. For a
    dispersion-measure GP, this allows the diagnostic to more closely represent
    the actual chromatic Fourier basis used by the timing/noise model rather
    than an achromatic sine/cosine basis.

    Parameters
    ----------
    toas
        PINT TOAs object containing TOA epochs and uncertainties.
    modes_hz
        One-dimensional array of Fourier frequencies in Hz.
    chromatic_scale
        Optional one-dimensional array containing one multiplicative scaling
        factor per TOA. Its convention should match the Fourier-GP design
        matrix used by the downstream noise model.

    Returns
    -------
    FourierBasisResult
        Weighted design-matrix, Gram-matrix, correlation, and conditioning
        diagnostics.
    """
    modes_hz = np.asarray(modes_hz, dtype=float)

    if modes_hz.ndim != 1 or modes_hz.size == 0:
        raise ValueError(
            "modes_hz must be a non-empty one-dimensional array."
        )

    if not np.all(np.isfinite(modes_hz)):
        raise ValueError("modes_hz contains non-finite values.")

    if np.any(modes_hz <= 0):
        raise ValueError("All Fourier frequencies must be positive.")

    # ------------------------------------------------------------------
    # Extract and validate per-TOA inputs
    # ------------------------------------------------------------------
    mjds = np.asarray(
        toas.get_mjds().value,
        dtype=float,
    )

    sigma_seconds = np.asarray(
        toas.table["error"].quantity.to_value(u.s),
        dtype=float,
    )

    if mjds.ndim != 1 or sigma_seconds.ndim != 1:
        raise ValueError(
            "TOA epochs and uncertainties must be one-dimensional arrays."
        )

    if mjds.shape != sigma_seconds.shape:
        raise ValueError(
            "TOA epochs and uncertainties must have identical shapes."
        )

    if chromatic_scale is not None:
        chromatic_scale = np.asarray(
            chromatic_scale,
            dtype=float,
        )

        if chromatic_scale.shape != mjds.shape:
            raise ValueError(
                "chromatic_scale must have one value per TOA."
            )

    valid = (
        np.isfinite(mjds)
        & np.isfinite(sigma_seconds)
        & (sigma_seconds > 0)
    )

    if chromatic_scale is not None:
        valid &= np.isfinite(chromatic_scale)

    n_valid = int(np.count_nonzero(valid))

    if n_valid < 2:
        raise ValueError(
            "Too few valid TOAs remain for Fourier-basis evaluation."
        )

    mjds = mjds[valid]
    sigma_seconds = sigma_seconds[valid]

    if chromatic_scale is not None:
        chromatic_scale = chromatic_scale[valid]

    time_seconds = (
        mjds - np.min(mjds)
    ) * 86400.0

    weights = 1.0 / sigma_seconds**2

    # ------------------------------------------------------------------
    # Construct sampled sine/cosine design matrix
    # ------------------------------------------------------------------
    columns = []

    for frequency_hz in modes_hz:
        phase = 2.0 * np.pi * frequency_hz * time_seconds

        columns.append(np.sin(phase))
        columns.append(np.cos(phase))

    design = np.column_stack(columns)

    if chromatic_scale is not None:
        design = chromatic_scale[:, None] * design

    weighted_design = np.sqrt(weights)[:, None] * design

    gram = weighted_design.T @ weighted_design

    gram_diag = np.diag(gram)
    denominator = np.sqrt(np.outer(gram_diag, gram_diag))

    gram_correlation = np.divide(
        gram,
        denominator,
        out=np.full_like(gram, np.nan),
        where=denominator > 0,
    )

    offdiag_mask = ~np.eye(
        gram_correlation.shape[0],
        dtype=bool,
    )

    offdiag_values = np.abs(
        gram_correlation[offdiag_mask]
    )

    return FourierBasisResult(
        gram=gram,
        gram_correlation=gram_correlation,
        max_offdiag=float(np.nanmax(offdiag_values)),
        median_offdiag=float(np.nanmedian(offdiag_values)),
        condition_design=float(np.linalg.cond(weighted_design)),
        condition_gram=float(np.linalg.cond(gram)),
        n_modes=int(modes_hz.size),
        n_basis_columns=int(design.shape[1]),
    )

def build_fourier_gp_seed(
    *,
    edges: np.ndarray,
    gaps: Optional[Sequence[Tuple[float, float]]],
    data_span_seconds: float,
    config: FourierGPConfig,
    toas: Optional[Any] = None,
    compute_periodogram: bool = True,
    chromatic_scale: Optional[np.ndarray] = None,
) -> FourierGridResult:
    """Build a Fourier GP mode array and associated BB diagnostics."""
    config.validate()

    sampling: Optional[BBSamplingSeries] = None
    support: Optional[BBFrequencySupport] = None
    periodogram: Optional[PeriodogramResult] = None

    needs_bb_support = (
        config.strategy in {"bb_harmonic", "hybrid"}
        or compute_periodogram
    )

    if needs_bb_support:
        sampling = prepare_bb_sampling_series(
            edges=edges,
            gaps=gaps,
        )

        support = estimate_bb_frequency_support(
            sampling,
            central_fraction=config.central_fraction,
        )

    modes_hz, metadata = construct_fourier_grid(
        config=config,
        data_span_seconds=data_span_seconds,
        support=support,
    )

    if compute_periodogram:
        if sampling is None or support is None:
            raise RuntimeError(
                "Sampling/support products were not constructed."
            )

        periodogram = compute_bb_width_periodogram(
            sampling=sampling,
            support=support,
            config=config,
        )

    basis = None
    if config.evaluate_basis and toas is not None:
        basis = evaluate_fourier_basis(
            toas=toas,
            modes_hz=modes_hz,
            chromatic_scale=chromatic_scale,
        )

    T_sec = float(data_span_seconds)

    return FourierGridResult(
        strategy=config.strategy,
        modes_hz=modes_hz,
        spacing_hz=float(
            np.median(np.diff(modes_hz))
            if modes_hz.size > 1
            else modes_hz[0]
        ),
        fmin_model_hz=float(modes_hz.min()),
        fmax_model_hz=float(modes_hz.max()),
        nfreqs=int(modes_hz.size),
        data_span_days=T_sec / 86400.0,
        data_span_seconds=T_sec,
        sampling=sampling,
        support=support,
        periodogram=periodogram,
        basis=basis,
        metadata=metadata,
    )
