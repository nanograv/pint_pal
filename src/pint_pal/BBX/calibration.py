from __future__ import annotations

import operator
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ACFResult:
	"""Empirical autocorrelation summary for a proxy time series."""

	lags: np.ndarray
	acf: np.ndarray
	tau_int: float
	n_eff: float
	tau_1e: Optional[float] = None
	tau_zero: Optional[float] = None
	mean: float = np.nan
	variance: float = np.nan
	method: str = "pairwise_binned"
	metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationProposal:
	"""Derived Bayesian Blocks and GP calibration values from a proxy ACF."""

	p0: float
	ncp_prior: float
	n_eff: float
	tau_int: float
	tau_1e: Optional[float] = None
	tau_zero: Optional[float] = None
	min_time_days: Optional[float] = None
	max_time_days: Optional[float] = None
	gp_length_scale_days: Optional[float] = None
	gp_amplitude: Optional[float] = None
	gp_kernel_hint: str = "rbf"
	metadata: Mapping[str, Any] = field(default_factory=dict)


# Convenience functions for ACF and calibration computations
def _as_1d_float_array(values: Any, *, name: str, unit: Optional[Any] = None) -> np.ndarray:
	"""
	Convert array-like inputs to a 1D float vector e.g. Astropy Quantity or Table column, while
	rejecting (with error) any higher-dimensional inputs.

	If `unit` is supplied and the object supports Astropy-style unit
    conversion, convert to that unit first. Unitless inputs are assumed
    already to use the requested unit.
	"""
	converted = values

	# Unit conversion if selected and supported
	if unit is not None:
		if hasattr(values, "to"):
			converted = values.to(unit)
		elif hasattr(values, "to_value"):
			converted = values.to_value(unit).value
		else:
			converted = getattr(converted, "value", converted) 
	
	array = np.asarray(converted, dtype=float)
	if array.ndim != 1:
		raise ValueError(f"{name} must be a 1D array.")
	return array


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
	"""Compute weighted mean of input (e.g. proxy series) after jointly masking invalid values"""
	values = np.asarray(values, dtype=float)
	weights = np.asarray(weights, dtype=float)

	if values.shape != weights.shape:
		raise ValueError("values and weights must have the same shape.")
	
	# Create a mask for valid (finite) values and weights
	valid = (np.isfinite(values) & np.isfinite(weights) & (weights > 0))
	if not np.any(valid):
		raise ValueError("No finite values with positive finite weights to compute weighted mean.")
	
	values = values[valid]
	weights = weights[valid]

	total_weight = float(np.sum(weights))
	return float(np.sum(weights * values) / total_weight)

def estimate_effective_sample_size(sample_size: int, tau_int: float) -> float:
	"""
	Compute N_eff = N / tau_int, with a minimum of 1.0.
	
	This assumes tau_int >= 1, so the reported effective sample size is always 
	less than or equal to the actual observed sample size.
	"""
	if isinstance(sample_size, bool):
		raise ValueError("sample_size must be an integer, not a boolean.")
	
	try:
		sample_size = operator.index(sample_size)
	except TypeError as exc:
		raise ValueError("sample_size must be an integer count.") from exc
	
	tau_int = float(tau_int)

	if sample_size <= 0:
		raise ValueError("sample_size must be positive.")
	if not np.isfinite(tau_int) or tau_int <= 1.0:
		raise ValueError("tau_int must be finite and >= 1.0.")
	return float(max(1.0, min(sample_size, sample_size / tau_int)))


def first_crossing(
		    lags: Any,
    acf: Any,
    threshold: float,
    *,
    direction: Literal["down", "up", "either"] = "down",
    require_contiguous: bool = True,
) -> Optional[float]:
	"""
	Return the first positive lag where the ACF crosses a threshold, typically 1/e crossing.
	
	Generic threshold-crossing finder.
	"""
	# Convert inputs to 1D float arrays
	lag_arr = _as_1d_float_array(lags, name="lags")
	acf_arr = _as_1d_float_array(acf, name="acf")
	threshold = float(threshold)
	
	if lag_arr.size != acf_arr.size:
		raise ValueError("lags and acf must have the same length.")
	if lag_arr.size == 0:
		return None
	if not np.isfinite(threshold):
		raise ValueError("threshold must be finite.")
	if direction not in {"down", "up", "either"}:
		raise ValueError("direction must be 'down', 'up', or 'either'.")

    # Invalid lag coordinates cannot be used, but NaN ACF values are
    # deliberately retained as gaps in observational support. To avoid
	# fake crossings
	keep = np.isfinite(lag_arr) & (lag_arr >= 0)
	lag_arr = lag_arr[keep]
	acf_arr = acf_arr[keep]

	order = np.argsort(lag_arr, kind="stable")
	lag_arr = lag_arr[order]
	acf_arr = acf_arr[order]

	# Require at least two positive lags to find a crossing
	if lag_arr.size < 2:
		return None
	if np.any(np.diff(lag_arr) <= 0):
		raise ValueError("lags must unique after sorting and filtering for positive values.")
	if require_contiguous:
		if not np.isfinite(acf_arr[0]):
			return None
		
		missing = np.flatnonzero(~np.isfinite(acf_arr))
		if missing.size:
			stop = int(missing[0])
			lag_arr = lag_arr[:stop]
			acf_arr = acf_arr[:stop]

	if lag_arr.size < 2:
		return None
	
	if acf_arr[0] == threshold:
		return float(lag_arr[0])
	
	# Iterate through the ACF values to find the first crossing of the threshold
	for idx in range(lag_arr.size - 1):
		y0 = acf_arr[idx]
		y1 = acf_arr[idx + 1]
		
		if not np.isfinite(y0) or not np.isfinite(y1):
			continue

		downward = y0 > threshold and y1 <= threshold
		upward = y0 < threshold and y1 >= threshold

		crossed = (
			downward if direction == "down" else
			upward if direction == "up" else
			downward or upward
		)
		if not crossed:
			continue

		x0 = float(lag_arr[idx])
		x1 = float(lag_arr[idx + 1])

		denominator = y1 - y0
		if denominator == 0.0 or not np.isfinite(denominator):
			return float(x0)  # Avoid division by zero or non-finite values
		
		frac = (threshold - y0) / denominator
		return float(x0 + frac * (x1 - x0))
	
    # If no crossing is found, return None
	return None


def integrated_autocorrelation_time(
	lags: Any,
	acf: Any,
	*,
	reference_step: Optional[float] = None,
    truncate_at_zero_crossing: bool = True,
    require_contiguous: bool = True,
) -> float:
	"""
	Estimate the integrated autocorrelation time (IACT) from an ACF curve.

	Credit/based on: 
		https://emcee.readthedocs.io/en/stable/tutorials/autocorr/#estimating-the-integrated-autocorrelation-time
		https://mc-stan.org/docs/reference-manual/analysis.html#effective-sample-size.section
	"""
	# Normalize inputs to 1D float arrays and validate lengths
	lag_arr = _as_1d_float_array(lags, name="lags")
	acf_arr = _as_1d_float_array(acf, name="acf")
	if lag_arr.size != acf_arr.size:
		raise ValueError("lags and acf must have the same length.")

    # Sort lag and ACF together and keep only positive lags
	keep = np.isfinite(lag_arr) & (lag_arr >= 0)
	lag_arr = lag_arr[keep]
	acf_arr = acf_arr[keep]

	order = np.argsort(lag_arr, kind="stable")
	lag_arr = lag_arr[order]
	acf_arr = acf_arr[order]

	if lag_arr.size < 2:
		raise ValueError("At least two supported lag values are required.")
	if np.any(np.diff(lag_arr) <= 0):
		raise ValueError("lags must be unique after sorting.")
	if not np.isclose(lag_arr[0], 0.0):
		raise ValueError("The ACF must contain an explicit zero-lag value.")
	if not np.isfinite(acf_arr[0]) or acf_arr[0] <= 0:
		raise ValueError("The zero-lag ACF must be finite and positive.")

	if require_contiguous:
		missing = np.flatnonzero(~np.isfinite(acf_arr))
		if missing.size:
			stop = int(missing[0])
			lag_arr = lag_arr[:stop]
			acf_arr = acf_arr[:stop]

	if lag_arr.size < 2:
		raise ValueError("No contiguous nonzero-lag ACF support is available.")
	
	# Normalize the lag differences to use as a reference time step for integration
	rho = acf_arr / acf_arr[0]  # Normalize ACF to 1.0 at zero lag
	rho[0] = 1.0  # Ensure the zero-lag ACF is exactly 1.0

	# Infer a reference step only for an approximately uniform lag grid.
	if reference_step is None:
		differences = np.diff(lag_arr)
		inferred = float(np.median(differences)) if differences.size else 1.0

		if not np.isfinite(inferred) or inferred <= 0 or not np.allclose(differences, inferred, rtol=0.05, atol=0.0):
			raise ValueError(
				"Cannot infer a reference step from nonuniform lag values. " \
				"Please provide reference_step explicitly. " \
				"`reference_step` is required for nonuniform lag grids."
				)
		reference_step = inferred
	else:
		reference_step = float(reference_step)

	if not np.isfinite(reference_step) or reference_step <= 0:
		raise ValueError("reference_step must be finite and positive.")
	
	if truncate_at_zero_crossing:
		# Find the first index where the ACF crosses zero
		nonpositive = np.flatnonzero(rho[1:] <= 0)

		# If a zero-crossing is found, truncate the lag and ACF arrays 
		if nonpositive.size:
			crossing_index = int(nonpositive[0]) + 1

			if rho[crossing_index] == 0:
				# If the ACF is exactly zero at the crossing index, truncate the arrays directly
				lag_arr = lag_arr[:crossing_index + 1]
				rho = rho[:crossing_index + 1]
			else:
				# If the ACF crosses zero between two indices, perform linear interpolation 
				# to estimate the zero-crossing lag
				x0 = float(lag_arr[crossing_index - 1])
				x1 = float(lag_arr[crossing_index])
				y0 = float(rho[crossing_index - 1])
				y1 = float(rho[crossing_index])

				# Estimate the zero-crossing lag
				zero_lag = x0 - y0 * (x1 - x0) / (y1 - y0)

				lag_arr = np.concatenate((lag_arr[:crossing_index], [zero_lag]))

				rho = np.concatenate((rho[:crossing_index], [0.0]))

		else:
			warnings.warn(
				"The ACF did not cross zero within the lag range." \
				"The IACT is a truncated lower-bound estimate.",
				RuntimeWarning,
				stacklevel=2,
			)

	correlation_time = 2.0 * float(np.trapeze(rho, lag_arr))  # IACT = 2 * integral of ACF
	tau_int = correlation_time / reference_step  # Normalize by reference step to get dimensionless IACT

	if not np.isfinite(tau_int):
		raise ValueError("The integrated autocorrelation time is not finite. Check the ACF input values.")
	
	return float(max(tau_int, 1.0))


def empirical_acf(
    t: Any,
    y: Any,
    yerr: Optional[Any] = None,
    *,
    n_lags: int = 32,
    max_lag: Optional[float] = None,
    threshold: float = 1.0 / np.e,
    min_pairs: int = 8,
    reference_step: Optional[float] = None,
    pair_weighting: Literal[
        "equal",
        "inverse_product_variance",
    ] = "equal",
    correct_measurement_noise: bool = True,
) -> ACFResult:
    """
    Estimate a DCF-like ACF for an irregularly sampled time series.

    Times are converted to days when they carry compatible units.
    Unitless time inputs are assumed to already be in days.

    Empty or under-supported lag bins are returned as NaN; they are never
    interpreted as zero correlation.

	Credit/based on:
	https://www.ict.inaf.it/gitlab/stefano.covino/TimeDomainAstrophysics/-/tree/main/Lectures?ref_type=heads
	Stefano Covino, Time Domain Astrophysics - Lecture notes featuring computational examples, 2026.

	Ivezić et al., Statistics, Data Mining, and Machine Learning in Astronomy - Ch.10
    """
    t_arr = _as_1d_float_array(t, name="t", unit="day")

    # Keep y in its existing physical unit. Convert yerr to that same unit
    # when both objects carry Astropy-compatible units.
    y_unit = getattr(y, "unit", None)
    y_arr = _as_1d_float_array(y, name="y")

    if t_arr.size != y_arr.size:
        raise ValueError("t and y must have the same length.")

    if yerr is None:
        yerr_arr = None
        valid = np.isfinite(t_arr) & np.isfinite(y_arr)
    else:
        yerr_arr = _as_1d_float_array(
            yerr,
            name="yerr",
            unit=y_unit,
        )
        if yerr_arr.size != y_arr.size:
            raise ValueError("yerr must have the same length as y.")

        valid = (
            np.isfinite(t_arr)
            & np.isfinite(y_arr)
            & np.isfinite(yerr_arr)
            & (yerr_arr > 0)
        )

    removed = int(valid.size - np.count_nonzero(valid))
    if removed:
        warnings.warn(
            f"Removed {removed} rows with invalid time, value, or uncertainty.",
            RuntimeWarning,
            stacklevel=2,
        )

    t_arr = t_arr[valid]
    y_arr = y_arr[valid]
    if yerr_arr is not None:
        yerr_arr = yerr_arr[valid]

    if t_arr.size < 4:
        raise ValueError("At least four valid observations are required.")

    order = np.argsort(t_arr, kind="stable")
    t_arr = t_arr[order]
    y_arr = y_arr[order]
    if yerr_arr is not None:
        yerr_arr = yerr_arr[order]

    if yerr_arr is None:
        point_weights = np.ones_like(y_arr)
    else:
        point_weights = 1.0 / np.square(yerr_arr)

    mean_y = _weighted_mean(y_arr, point_weights)
    centered = y_arr - mean_y

    weight_sum = float(np.sum(point_weights))
    observed_variance = float(
        np.sum(point_weights * np.square(centered)) / weight_sum
    )

    if yerr_arr is not None and correct_measurement_noise:
        mean_noise_variance = float(
            np.sum(point_weights * np.square(yerr_arr)) / weight_sum
        )
    else:
        mean_noise_variance = 0.0

    intrinsic_variance = observed_variance - mean_noise_variance

    if not np.isfinite(intrinsic_variance) or intrinsic_variance <= 0:
        raise ValueError(
            "The estimated intrinsic variance is not positive. "
            "The series may be constant or unresolved relative to its "
            "measurement uncertainties."
        )

    time_span = float(t_arr[-1] - t_arr[0])
    if not np.isfinite(time_span) or time_span <= 0:
        raise ValueError("t must contain at least two distinct times.")

    # Correlations beyond half a span normally have very poor pair support.
    # The caller can request the full span explicitly.
    if max_lag is None:
        max_lag = 0.5 * time_span

    max_lag = float(max_lag)
    if not np.isfinite(max_lag) or max_lag <= 0:
        raise ValueError("max_lag must be positive and finite.")

    try:
        n_lags = operator.index(n_lags)
        min_pairs = operator.index(min_pairs)
    except TypeError as exc:
        raise TypeError("n_lags and min_pairs must be integers.") from exc

    if n_lags < 2:
        raise ValueError("n_lags must be at least 2.")
    if min_pairs < 2:
        raise ValueError("min_pairs must be at least 2.")
    if pair_weighting not in {
        "equal",
        "inverse_product_variance",
    }:
        raise ValueError("Unsupported pair_weighting value.")

    lag_edges = np.linspace(0.0, max_lag, n_lags + 1)
    nominal_centers = 0.5 * (lag_edges[:-1] + lag_edges[1:])

    # This vectorized construction is O(N^2) in memory and time. It is
    # appropriate only for the short proxy series described by the module.
    left, right = np.triu_indices(t_arr.size, k=1)
    pair_lags = t_arr[right] - t_arr[left]

    pair_valid = (
        np.isfinite(pair_lags)
        & (pair_lags > 0)
        & (pair_lags <= max_lag)
    )
    left = left[pair_valid]
    right = right[pair_valid]
    pair_lags = pair_lags[pair_valid]

    if pair_lags.size == 0:
        raise ValueError("No positive-lag pairs fall within max_lag.")

    pair_products = centered[left] * centered[right]

    if pair_weighting == "equal":
        pair_weights = np.ones_like(pair_products)
    else:
        # Approximate inverse variance of a Gaussian product. This is more
        # defensible than 1/(yerr_i^2 yerr_j^2) because it includes intrinsic
        # process variance as well as measurement variance.
        if yerr_arr is None:
            total_point_variance = np.full_like(
                y_arr,
                intrinsic_variance,
            )
        else:
            total_point_variance = (
                intrinsic_variance + np.square(yerr_arr)
            )

        product_variance = (
            total_point_variance[left]
            * total_point_variance[right]
            + intrinsic_variance**2
        )
        pair_weights = 1.0 / product_variance

    bins = np.searchsorted(
        lag_edges,
        pair_lags,
        side="right",
    ) - 1

    # A value exactly equal to max_lag belongs to the final bin.
    bins = np.minimum(bins, n_lags - 1)

    pair_count = np.bincount(
        bins,
        minlength=n_lags,
    ).astype(int)
    sum_weight = np.bincount(
        bins,
        weights=pair_weights,
        minlength=n_lags,
    )
    sum_weight_squared = np.bincount(
        bins,
        weights=np.square(pair_weights),
        minlength=n_lags,
    )
    sum_lag = np.bincount(
        bins,
        weights=pair_weights * pair_lags,
        minlength=n_lags,
    )
    sum_product = np.bincount(
        bins,
        weights=pair_weights * pair_products,
        minlength=n_lags,
    )
    sum_product_squared = np.bincount(
        bins,
        weights=pair_weights * np.square(pair_products),
        minlength=n_lags,
    )

    effective_pair_count = np.divide(
        np.square(sum_weight),
        sum_weight_squared,
        out=np.zeros_like(sum_weight),
        where=sum_weight_squared > 0,
    )

    supported = (
        (sum_weight > 0)
        & (pair_count >= min_pairs)
        & (effective_pair_count >= min_pairs)
    )

    lag_centers = np.divide(
        sum_lag,
        sum_weight,
        out=nominal_centers.copy(),
        where=sum_weight > 0,
    )
    mean_product = np.divide(
        sum_product,
        sum_weight,
        out=np.full(n_lags, np.nan),
        where=sum_weight > 0,
    )

    acf_bins = np.full(n_lags, np.nan)
    acf_bins[supported] = (
        mean_product[supported] / intrinsic_variance
    )

    # A diagnostic error estimate. Pair products are dependent because an
    # observation appears in many pairs, so bootstrap errors should replace
    # this approximation for final inference.
    second_product_moment = np.divide(
        sum_product_squared,
        sum_weight,
        out=np.full(n_lags, np.nan),
        where=sum_weight > 0,
    )
    product_variance = np.maximum(
        second_product_moment - np.square(mean_product),
        0.0,
    )
    acf_stderr = np.full(n_lags, np.nan)
    acf_stderr[supported] = (
        np.sqrt(
            product_variance[supported]
            / effective_pair_count[supported]
        )
        / intrinsic_variance
    )

    lag_values = np.concatenate(([0.0], lag_centers))
    acf_values = np.concatenate(([1.0], acf_bins))

    tau_1e = first_crossing(
        lag_values,
        acf_values,
        threshold,
        direction="down",
        require_contiguous=True,
    )
    tau_zero = first_crossing(
        lag_values,
        acf_values,
        0.0,
        direction="down",
        require_contiguous=True,
    )

    if reference_step is None:
        distinct_times = np.unique(t_arr)
        observation_gaps = np.diff(distinct_times)
        observation_gaps = observation_gaps[observation_gaps > 0]

        if observation_gaps.size == 0:
            raise ValueError("No positive observation gaps are available.")

        reference_step = float(np.median(observation_gaps))
    else:
        reference_step = float(reference_step)

    try:
        tau_int = integrated_autocorrelation_time(
            lag_values,
            acf_values,
            reference_step=reference_step,
            truncate_at_zero_crossing=True,
            require_contiguous=True,
        )
        n_eff = estimate_effective_sample_size(
            t_arr.size,
            tau_int,
        )
    except ValueError as exc:
        warnings.warn(
            f"IACT could not be identified: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        tau_int = float("nan")
        n_eff = float("nan")

    return ACFResult(
        lags=lag_values,
        acf=acf_values,
        tau_int=tau_int,
        n_eff=n_eff,
        tau_1e=tau_1e,
        tau_zero=tau_zero,
        mean=mean_y,
        variance=intrinsic_variance,
        method=f"pairwise_binned_{pair_weighting}",
        metadata={
            "n_obs": int(t_arr.size),
            "time_span_days": time_span,
            "reference_step_days": reference_step,
            "n_lags": n_lags,
            "max_lag_days": max_lag,
            "lag_edges_days": lag_edges,
            "pair_count": np.concatenate(([t_arr.size], pair_count)),
            "effective_pair_count": np.concatenate(
                ([t_arr.size], effective_pair_count)
            ),
            "acf_stderr_naive": np.concatenate(([0.0], acf_stderr)),
            "observed_variance": observed_variance,
            "mean_noise_variance": mean_noise_variance,
            "intrinsic_variance": intrinsic_variance,
            "removed_observations": removed,
            "measurement_noise_corrected": bool(
                yerr_arr is not None and correct_measurement_noise
            ),
        },
    )
		
def _supported_interpolation(
    x: float,
    xp: np.ndarray,
    fp: np.ndarray,
) -> Optional[float]:
    """Interpolate only between adjacent, finite supported values."""
    x = float(x)

    exact = np.flatnonzero(np.isclose(xp, x))
    if exact.size:
        value = fp[int(exact[0])]
        return float(value) if np.isfinite(value) else None

    upper = int(np.searchsorted(xp, x, side="right"))
    lower = upper - 1

    if lower < 0 or upper >= xp.size:
        return None
    if not np.isfinite(fp[lower]) or not np.isfinite(fp[upper]):
        return None

    fraction = (x - xp[lower]) / (xp[upper] - xp[lower])
    return float(fp[lower] + fraction * (fp[upper] - fp[lower]))


def propose_bbx_calibration(
    acf_result: ACFResult,
    *,
    n_proxy: int,
    gp_sigma_scale: float,
    alpha_total: float = 0.05,
    p0_clip: Tuple[float, float] = (1e-6, 0.5),
    ncp_prior_override: Optional[float] = None,
    guardrail_fraction: float = 0.5,
    period_days: Optional[float] = None,
    period_false_alarm_probability: Optional[float] = None,
    period_fap_max: float = 0.01,
    minimum_period_cycles: float = 2.5,
    recurrence_sigma: float = 2.0,
) -> CalibrationProposal:
    """
    Produce an analytic BB baseline and Discovery-compatible GP seeds.

    Kernel scores are diagnostics based on the empirical ACF. Final kernel
    selection should be made by fitting candidates to the original data.
    """
    try:
        n_proxy = operator.index(n_proxy)
    except TypeError as exc:
        raise TypeError("n_proxy must be the integer observation count.") from exc

    if n_proxy <= 0:
        raise ValueError("n_proxy must be positive.")

    n_eff = float(acf_result.n_eff)
    tau_int = float(acf_result.tau_int)

    if not np.isfinite(n_eff) or n_eff <= 0:
        raise ValueError("acf_result does not contain a valid n_eff.")
    if not np.isfinite(tau_int) or tau_int < 1:
        raise ValueError("acf_result does not contain a valid tau_int.")

    alpha_total = float(alpha_total)
    lower_p0, upper_p0 = map(float, p0_clip)

    if not 0 < lower_p0 < upper_p0 < 1:
        raise ValueError("p0_clip must satisfy 0 < low < high < 1.")
    if not 0 < alpha_total < 1:
        raise ValueError("alpha_total must lie strictly between zero and one.")

    # Scargle/Astropy's p0 is already a false-alarm control input. Do not
    # apply a second Bonferroni-like division by effective sample size.
    p0 = float(np.clip(alpha_total, lower_p0, upper_p0))

    analytic_ncp_prior = float(
        4.0 - np.log(73.53 * p0 * (n_proxy ** -0.478))
    )

    if ncp_prior_override is None:
        ncp_prior = analytic_ncp_prior
        ncp_source = "scargle_analytic_uncalibrated_for_correlation"
    else:
        ncp_prior = float(ncp_prior_override)
        if not np.isfinite(ncp_prior) or ncp_prior <= 0:
            raise ValueError(
                "ncp_prior_override must be positive and finite."
            )
        ncp_source = "simulation_override"

    variance_proxy = float(acf_result.variance)
    gp_sigma_scale = float(gp_sigma_scale)

    if not np.isfinite(variance_proxy) or variance_proxy <= 0:
        raise ValueError("ACF intrinsic variance must be positive.")
    if not np.isfinite(gp_sigma_scale) or gp_sigma_scale <= 0:
        raise ValueError("gp_sigma_scale must be positive and finite.")

    sigma_proxy = float(np.sqrt(variance_proxy))
    sigma_gp = sigma_proxy * gp_sigma_scale
    log10_sigma = float(np.log10(sigma_gp))

    lags = np.asarray(acf_result.lags, dtype=float)
    rho = np.asarray(acf_result.acf, dtype=float)

    supported = (
        np.isfinite(lags)
        & np.isfinite(rho)
        & (lags > 0)
    )
    if np.count_nonzero(supported) < 2:
        raise ValueError("Too few supported ACF bins for GP seeding.")

    fit_lags = lags[supported]
    fit_rho = rho[supported]

    metadata = dict(acf_result.metadata)
    stderr_all = np.asarray(
        metadata.get(
            "acf_stderr_naive",
            np.full_like(lags, np.nan),
        ),
        dtype=float,
    )
    pair_count_all = np.asarray(
        metadata.get(
            "effective_pair_count",
            np.ones_like(lags),
        ),
        dtype=float,
    )

    if stderr_all.shape != lags.shape:
        stderr_all = np.full_like(lags, np.nan)
    if pair_count_all.shape != lags.shape:
        pair_count_all = np.ones_like(lags)

    fit_stderr = stderr_all[supported]
    fit_pair_count = pair_count_all[supported]

    valid_stderr = np.isfinite(fit_stderr) & (fit_stderr > 0)
    if np.any(valid_stderr):
        score_weights = np.where(
            valid_stderr,
            1.0 / np.square(fit_stderr),
            0.0,
        )
    else:
        score_weights = np.where(
            np.isfinite(fit_pair_count) & (fit_pair_count > 0),
            fit_pair_count,
            1.0,
        )

    if np.sum(score_weights) <= 0:
        score_weights = np.ones_like(fit_rho)

    score_weights = score_weights / np.sum(score_weights)

    def score(template: np.ndarray) -> float:
        return float(
            np.sqrt(
                np.sum(
                    score_weights
                    * np.square(fit_rho - template)
                )
            )
        )

    candidate_models: dict[str, np.ndarray] = {
        "ridge": np.zeros_like(fit_lags),
    }
    candidate_seeds: dict[str, dict[str, float]] = {
        "ridge": {
            "log10_sigma_ridge": log10_sigma,
        },
    }

    tau_1e = acf_result.tau_1e

    if tau_1e is not None:
        tau_1e = float(tau_1e)
        if not np.isfinite(tau_1e) or tau_1e <= 0:
            raise ValueError("tau_1e must be positive and finite.")

        ell_sq_exp = tau_1e / np.sqrt(2.0)
        ell_matern_05 = tau_1e
        ell_matern_15 = 0.8070339571140971 * tau_1e
        ell_matern_25 = 0.7698288583132731 * tau_1e

        candidate_models["square_exponential"] = np.exp(
            -0.5 * np.square(fit_lags / ell_sq_exp)
        )
        candidate_seeds["square_exponential"] = {
            "log10_sigma_sq_exp": log10_sigma,
            "log10_ell": float(np.log10(ell_sq_exp)),
        }

        r05 = fit_lags / ell_matern_05
        candidate_models["matern_0.5"] = np.exp(-r05)
        candidate_seeds["matern_0.5"] = {
            "log10_sigma_matern": log10_sigma,
            "log10_ell": float(np.log10(ell_matern_05)),
            "nu": 0.5,
        }

        r15 = fit_lags / ell_matern_15
        c15 = np.sqrt(3.0)
        candidate_models["matern_1.5"] = (
            (1.0 + c15 * r15)
            * np.exp(-c15 * r15)
        )
        candidate_seeds["matern_1.5"] = {
            "log10_sigma_matern": log10_sigma,
            "log10_ell": float(np.log10(ell_matern_15)),
            "nu": 1.5,
        }

        r25 = fit_lags / ell_matern_25
        c25 = np.sqrt(5.0)
        candidate_models["matern_2.5"] = (
            1.0 + c25 * r25 + (5.0 / 3.0) * np.square(r25)
        ) * np.exp(-c25 * r25)
        candidate_seeds["matern_2.5"] = {
            "log10_sigma_matern": log10_sigma,
            "log10_ell": float(np.log10(ell_matern_25)),
            "nu": 2.5,
        }

    time_span_days = float(
        metadata.get("time_span_days", lags[-1])
    )
    reference_step_days = float(
        metadata.get("reference_step_days", np.nan)
    )

    quasi_periodic_checks: dict[str, Any] = {
        "requested": period_days is not None,
        "eligible": False,
    }

    if period_days is not None:
        period_days = float(period_days)

        if (
            not np.isfinite(period_days)
            or period_days <= 0
        ):
            raise ValueError("period_days must be positive and finite.")

        fap_ok = (
            period_false_alarm_probability is not None
            and np.isfinite(period_false_alarm_probability)
            and 0 <= period_false_alarm_probability <= period_fap_max
        )
        cycles_ok = (
            np.isfinite(time_span_days)
            and time_span_days / period_days >= minimum_period_cycles
        )
        sampling_ok = (
            not np.isfinite(reference_step_days)
            or period_days >= 2.0 * reference_step_days
        )

        rho_period = _supported_interpolation(
            period_days,
            lags,
            rho,
        )
        rho_half_period = _supported_interpolation(
            0.5 * period_days,
            lags,
            rho,
        )
        stderr_period = _supported_interpolation(
            period_days,
            lags,
            stderr_all,
        )
        stderr_half_period = _supported_interpolation(
            0.5 * period_days,
            lags,
            stderr_all,
        )

        positive_recurrence = (
            rho_period is not None
            and rho_half_period is not None
            and 0 < rho_half_period < rho_period < 1
        )

        if (
            positive_recurrence
            and stderr_period is not None
            and stderr_half_period is not None
            and stderr_period > 0
            and stderr_half_period > 0
        ):
            contrast_significance = (
                (rho_period - rho_half_period)
                / np.sqrt(
                    stderr_period**2
                    + stderr_half_period**2
                )
            )
            contrast_ok = contrast_significance >= recurrence_sigma
        else:
            contrast_significance = None
            contrast_ok = False

        quasi_eligible = (
            fap_ok
            and cycles_ok
            and sampling_ok
            and positive_recurrence
            and contrast_ok
        )

        quasi_periodic_checks.update({
            "eligible": quasi_eligible,
            "fap_ok": fap_ok,
            "cycles_ok": cycles_ok,
            "sampling_ok": sampling_ok,
            "positive_recurrence": positive_recurrence,
            "contrast_sigma": contrast_significance,
            "rho_period": rho_period,
            "rho_half_period": rho_half_period,
        })

        if quasi_eligible:
            ell_quasi = period_days / np.sqrt(
                -2.0 * np.log(rho_period)
            )
            gamma_p = (
                -np.log(rho_half_period)
                - period_days**2 / (8.0 * ell_quasi**2)
            )

            if gamma_p > 0 and np.isfinite(gamma_p):
                candidate_models["quasi_periodic"] = np.exp(
                    -0.5 * np.square(fit_lags / ell_quasi)
                    - gamma_p
                    * np.square(
                        np.sin(np.pi * fit_lags / period_days)
                    )
                )
                candidate_seeds["quasi_periodic"] = {
                    "log10_sigma_quasi_periodic": log10_sigma,
                    "log10_ell": float(np.log10(ell_quasi)),
                    "log10_gamma_p": float(np.log10(gamma_p)),
                    "log10_p": float(
                        np.log10(period_days / 365.25)
                    ),
                }
            else:
                quasi_periodic_checks["eligible"] = False
                quasi_periodic_checks["failure"] = (
                    "Estimated periodic damping is not positive."
                )

    kernel_scores = {
        name: score(template)
        for name, template in candidate_models.items()
    }
    ranking = sorted(
        kernel_scores,
        key=kernel_scores.get,
    )
    selected = ranking[0]

    if len(ranking) > 1:
        best_score = kernel_scores[ranking[0]]
        second_score = kernel_scores[ranking[1]]
        score_separation = (
            second_score / max(best_score, np.finfo(float).eps)
        )
    else:
        score_separation = float("nan")

    if selected == "ridge":
        discovery_kernel = "ridge_kernel"
        selected_ell = None
    elif selected == "square_exponential":
        discovery_kernel = "square_exponential_kernel"
        selected_ell = 10 ** candidate_seeds[selected]["log10_ell"]
    elif selected.startswith("matern"):
        discovery_kernel = "matern_kernel"
        selected_ell = 10 ** candidate_seeds[selected]["log10_ell"]
    else:
        discovery_kernel = "quasi_periodic_kernel"
        selected_ell = 10 ** candidate_seeds[selected]["log10_ell"]

    if tau_1e is not None:
        candidate_minimum = guardrail_fraction * tau_1e
        if np.isfinite(reference_step_days):
            min_time_days = float(
                max(reference_step_days, candidate_minimum)
            )
        else:
            min_time_days = float(candidate_minimum)
    elif np.isfinite(reference_step_days):
        min_time_days = reference_step_days
    else:
        min_time_days = None

    max_time_days = (
        time_span_days
        if np.isfinite(time_span_days) and time_span_days > 0
        else None
    )

    return CalibrationProposal(
        p0=p0,
        ncp_prior=ncp_prior,
        n_eff=n_eff,
        tau_int=tau_int,
        tau_1e=acf_result.tau_1e,
        tau_zero=acf_result.tau_zero,
        min_time_days=min_time_days,
        max_time_days=max_time_days,
        gp_length_scale_days=selected_ell,
        # This is now a standard deviation in Discovery GP units,
        # not a variance.
        gp_amplitude=sigma_gp,
        gp_kernel_hint=discovery_kernel,
        metadata={
            "alpha_total": alpha_total,
            "n_proxy": n_proxy,
            "analytic_ncp_prior": analytic_ncp_prior,
            "ncp_prior_source": ncp_source,
            "guardrail_fraction": guardrail_fraction,
            "gp_sigma_scale": gp_sigma_scale,
            "sigma_proxy": sigma_proxy,
            "sigma_gp": sigma_gp,
            "selected_candidate": selected,
            "selected_discovery_parameters": candidate_seeds[selected],
            "candidate_discovery_parameters": candidate_seeds,
            "kernel_shape_scores": kernel_scores,
            "kernel_score_ranking": ranking,
            "best_to_second_score_ratio": score_separation,
            "kernel_selection_role": (
                "initialization_hint_not_final_model_selection"
            ),
            "quasi_periodic_checks": quasi_periodic_checks,
        },
    )

def propose_bbx_calibration_old(
	acf_result: ACFResult,
	*,
	alpha_total: float = 0.05,
	p0_clip: Tuple[float, float] = (1e-6, 0.5),
	n_proxy: Optional[int] = None,
	guardrail_fraction: float = 0.5,
) -> CalibrationProposal:
	"""
	Map an ACF summary onto Bayesian Blocks and GP seed values.

	Notes:
	"""
	tau_int = float(acf_result.tau_int)
	n_eff = float(acf_result.n_eff)
	if n_proxy is None:
		n_proxy = int(max(1, acf_result.lags.size))

	p0 = float(np.clip(alpha_total / max(n_eff, 1.0), *p0_clip))
	# https://arxiv.org/abs/1207.5578 also see astropy.stats.bayesian_blocks implementation 
	ncp_prior = float(4.0 - np.log(73.53 * p0 * (max(n_proxy, 5) ** -0.478)))

    # Determine characteristic timescales and GP length scale based on the ACF results 
    # has units of the input time series (e.g., days for MJDs)
	tau_1e = acf_result.tau_1e
	tau_zero = acf_result.tau_zero
	characteristic = tau_1e if tau_1e is not None else tau_zero
	min_time_days = float(characteristic * guardrail_fraction) if characteristic is not None else None
	max_time_days = float(max(acf_result.lags[-1], characteristic or 0.0)) if acf_result.lags.size else None
	gp_length_scale_days = float(characteristic) if characteristic is not None else None

    # MAP TO DISCOVERY KERNEL NAMES AND PARAMETERS
	# https://github.com/jeremy-baier/discovery/blob/91e359a18a006186845114a12d3eea0540109b66/src/discovery/signals.py#L598
	kernel_hint = "rbf"
	if tau_1e is not None and tau_zero is not None and tau_zero > 1.5 * tau_1e:
		kernel_hint = "matern52"
	if acf_result.acf.size > 2 and np.nanmin(acf_result.acf[1:]) < -0.1:
		kernel_hint = "quasi_periodic"

	return CalibrationProposal(
		p0=p0,
		ncp_prior=ncp_prior,
		n_eff=n_eff,
		tau_int=tau_int,
		tau_1e=tau_1e,
		tau_zero=tau_zero,
		min_time_days=min_time_days,
		max_time_days=max_time_days,
		gp_length_scale_days=gp_length_scale_days,
		gp_amplitude=float(acf_result.variance),
		gp_kernel_hint=kernel_hint,
		metadata={"alpha_total": float(alpha_total), "guardrail_fraction": float(guardrail_fraction)},
	)
