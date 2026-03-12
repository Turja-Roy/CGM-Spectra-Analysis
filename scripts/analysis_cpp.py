"""
CGM Analysis - C++ Implementation Wrapper

This module provides Python bindings to the C++ implementation.
"""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
from typing import Optional, Dict, Any

from _analysis_cpp import (
    compute_power_spectrum as _cpp_compute_power_spectrum,
    compute_column_density_distribution as _cpp_compute_colden,
    compute_line_width_distribution as _cpp_compute_line_width,
    compute_flux_statistics as _cpp_compute_flux_stats,
    compute_temperature_density_relation as _cpp_compute_temp_dens,
)


def compute_flux_statistics(tau: np.ndarray) -> Dict[str, float]:
    """Compute basic flux statistics from optical depth."""
    if tau.dtype != np.float32:
        tau = tau.astype(np.float32)
    result = _cpp_compute_flux_stats(tau)
    return {
        'mean_flux': result['mean_flux'],
        'median_flux': result['median_flux'],
        'std_flux': result['std_flux'],
        'min_flux': result['min_flux'],
        'max_flux': result['max_flux'],
        'mean_tau': result['mean_tau'],
        'median_tau': result['median_tau'],
        'effective_tau': result['effective_tau'],
        'deep_absorption_frac': result['deep_absorption_frac'],
        'moderate_absorption_frac': result['moderate_absorption_frac'],
        'weak_absorption_frac': result['weak_absorption_frac'],
    }


def compute_power_spectrum(
    flux: np.ndarray,
    velocity_spacing: float,
    chunk_size: int = 1000
) -> Dict[str, Any]:
    """Compute power spectrum using scipy FFT."""
    from scipy import fft
    
    n_sightlines, n_pixels = flux.shape
    
    mean_flux = float(flux.mean())
    if mean_flux <= 0:
        raise ValueError("Mean flux must be positive")
    
    n_k = n_pixels // 2 + 1
    k = 2.0 * np.pi * np.arange(n_k) / (n_pixels * velocity_spacing)
    
    power_sum = np.zeros(n_k, dtype=np.float64)
    power_sum_sq = np.zeros(n_k, dtype=np.float64)
    
    for i in range(n_sightlines):
        delta_F = flux[i] / mean_flux - 1.0
        fft_result = fft.rfft(delta_F)
        power = np.abs(fft_result) ** 2 / n_pixels
        power_sum += power[:n_k]
        power_sum_sq += power[:n_k] ** 2
    
    P_k_mean = (power_sum / n_sightlines) * velocity_spacing
    mean_power = power_sum / n_sightlines
    mean_power_sq = power_sum_sq / n_sightlines
    
    variance = np.maximum(mean_power_sq - mean_power ** 2, 0)
    P_k_std = np.sqrt(variance) * velocity_spacing
    P_k_err = P_k_std / np.sqrt(n_sightlines)
    n_modes = np.ones(n_k) * n_sightlines
    
    return {
        'k': k,
        'P_k_mean': P_k_mean,
        'P_k_std': P_k_std,
        'P_k_err': P_k_err,
        'mean_flux': mean_flux,
        'n_modes': n_modes,
        'n_sightlines': n_sightlines,
        'velocity_spacing': velocity_spacing,
    }


def compute_column_density_distribution(
    tau: np.ndarray,
    velocity_spacing: float,
    threshold: float = 0.5,
    colden: Optional[np.ndarray] = None,
    redshift: Optional[float] = None,
    box_size_ckpc_h: Optional[float] = None,
    hubble: float = 0.6774,
    omega_m: float = 0.3089
) -> Dict[str, Any]:
    """Compute column density distribution function.
    
    Uses colden (peak) method when available.
    """
    if tau.dtype != np.float32:
        tau = tau.astype(np.float32)
    
    # Make contiguous - let C++ handle the layout correctly
    if not tau.flags.c_contiguous:
        tau = np.ascontiguousarray(tau)
    
    # Handle colden - make contiguous for C++
    if colden is not None and colden.size > 0:
        if colden.dtype != np.float32:
            colden = colden.astype(np.float32)
        if not colden.flags.c_contiguous:
            colden = np.ascontiguousarray(colden)
        colden_arg = colden
    else:
        colden_arg = np.array([], dtype=np.float32).reshape(0, 0)
    
    redshift_val = redshift if redshift is not None else float('nan')
    box_size_val = box_size_ckpc_h if box_size_ckpc_h is not None else float('nan')
    
    result = _cpp_compute_colden(
        tau, velocity_spacing, threshold,
        colden_arg, redshift_val, box_size_val, hubble, omega_m
    )
    
    bins_arr = np.array(result['bins'])
    bin_centers_arr = np.array(result['bin_centers'])
    
    return {
        'N_HI': np.array(result['N_HI']),
        'counts': np.array(result['counts']),
        'bins': bins_arr,
        'bin_centers': bin_centers_arr,
        'log_bin_edges': np.log10(bins_arr),
        'log10_N_HI': np.log10(bin_centers_arr),
        'delta_log_N': np.diff(np.log10(bins_arr)),
        'f_N': np.array(result['f_N']),
        'f_N_HI': np.array(result['f_N']),
        'beta_fit': result['beta_fit'],
        'n_absorbers': result['n_absorbers'],
        'n_sightlines': result['n_sightlines'],
        'dX': result['dX'],
        'redshift': result['redshift'],
    }


def compute_line_width_distribution(
    tau: np.ndarray,
    velocity_spacing: float,
    threshold: float = 0.5,
    colden: Optional[np.ndarray] = None,
    elem: str = "H",
    ion: int = 1,
    line: int = 1215
) -> Dict[str, Any]:
    """Compute line width (b-parameter) distribution using C++ implementation."""
    if tau.dtype != np.float32:
        tau = tau.astype(np.float32)
    
    colden_arg = np.array([], dtype=np.float32).reshape(0, 0)
    
    result = _cpp_compute_line_width(tau, velocity_spacing, threshold, colden_arg)
    
    return {
        'N_HI': np.array(result['N_HI']),
        'b_params': np.array(result['b_params']),
        'temperatures': np.array(result['temperatures']),
        'b_median': result['b_median'],
        'b_mean': result['b_mean'],
        'b_std': result['b_std'],
        'n_absorbers': result['n_absorbers'],
    }


def compute_temperature_density_relation(
    temperature: np.ndarray,
    density: np.ndarray,
    tau: np.ndarray,
    min_tau: float = 0.1
) -> Dict[str, Any]:
    """Compute temperature-density relation."""
    if temperature.dtype != np.float32:
        temperature = temperature.astype(np.float32)
    if density.dtype != np.float32:
        density = density.astype(np.float32)
    if tau.dtype != np.float32:
        tau = tau.astype(np.float32)
    
    result = _cpp_compute_temp_dens(temperature, density, tau, min_tau)
    
    return {
        'temperature': np.array(result['temperature']),
        'density': np.array(result['density']),
        'log_T': np.array(result['log_T']),
        'log_rho': np.array(result['log_rho']),
        'T0': result['T0'],
        'gamma': result['gamma'],
        'gamma_err': result['gamma_err'],
        'rho_mean': result.get('rho_mean', np.nan),
        'n_pixels': result['n_pixels'],
    }
