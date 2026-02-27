# BBX/__init__.py

"""
BBX package: Bayesian-Blocks DMX/SWX pipeline + diagnostics + utilities.
"""

from __future__ import annotations

# Core pipeline 
from .bbx import (
    # Config Objects
    RunConfig,
    InputConfig,
    OutputConfig, 
    PickleConfig, 
    BBXConfig, 
    ProxyConfig, 
    SolarWindConfig, 
    BaseFitConfig, 
    NoiseAnalysisConfig, 
    ReceiverSelection, 
    PlotStyleConfig,
    # Filepaths/state Objects
    RunContext,
    OutputPaths,
    # Pickler
    PicklerIO,
    CachePolicy,
    PicklerBundle,
    # Pipeline runners
    BaseFits,
    SolarWindProxy,
    DispersionMeasureProxy,
    BBX,
    NoiseAnalysis,
)

# Utilities (commonly used in notebooks)
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
)

# Diagnostics / plotting 
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
    plot_all_prefix_ellipses,
    summarize_swx_dmx_correlations,
    plot_param_ellipses_from_fitter,
    collect_white_noise_params,
    plot_wn_comparison,
    compare_white_noise,
    simple_dmxparse,
    plot_simple_dmx_time,
    zscore_filter,
    violation_table,
    summary,
    summarize_fitter,
)

__all__ = [
    # configs
    "ReceiverSelection",
    "PlotStyleConfig",
    "InputConfig",
    "OutputConfig",
    "PickleConfig",
    "BaseFitConfig",
    "SolarWindConfig",
    "BBXConfig",
    "ProxyConfig",
    "NoiseAnalysisConfig",
    "RunConfig",
    # runtime/pathing
    "OutputPaths",
    "RunContext",
    # caching/pickling
    "PicklerIO",
    "CachePolicy",
    "PicklerBundle",
    # runners
    "BaseFits",
    "SolarWindProxy",
    "DispersionMeasureProxy",
    "BBX",
    "NoiseAnalysis",
    # utils
    "handle_diagnostics",
    "handle_diagnostics_multi",
    "combine_repeated_toas",
    "mjd_to_year",
    "year_to_mjd",
    "pulsar_name_to_elat",
    "find_toa_by_mjd",
    "by_mjd_table",
    "find_toa_by_dmx",
    "format_gap_summary",
    # diagnostics
    "plot_epoch_gap_histogram",
    "plot_all_epoch_fit_overlay",
    "plot_points_per_epoch_arrays",
    "plot_b_snr_vs_time",
    "plot_a_vs_b_correlation",
    "plot_a_b_time_scatter",
    "plot_wls_epoch_summaries",
    "plot_data_gaps_diagnostics",
    "plot_swx_bb_diagnostics",
    "plot_dmx_segmentation_by_slice_diagnostics",
    "plot_bchrom_vs_dmx",
    "plot_all_prefix_ellipses",
    "summarize_swx_dmx_correlations",
    "plot_param_ellipses_from_fitter",
    "collect_white_noise_params",
    "plot_wn_comparison",
    "compare_white_noise",
    "simple_dmxparse",
    "plot_simple_dmx_time",
    "zscore_filter",
    "violation_table",
    "summary",
    "summarize_fitter",
]