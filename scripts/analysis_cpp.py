"""
CGM Analysis - C++ Implementation Wrapper

This module provides Python bindings to the C++ implementation
with automatic fallback to pure Python if C++ is not available.
"""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
from typing import Optional, Dict, Any

try:
    from _analysis_cpp import (
        compute_power_spectrum as _cpp_compute_power_spectrum,
        compute_column_density_distribution as _cpp_compute_colden,
        compute_line_width_distribution as _cpp_compute_line_width,
        compute_flux_statistics as _cpp_compute_flux_stats,
        compute_temperature_density_relation as _cpp_compute_temp_dens,
    )
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

from scripts import analysis as _python_analysis


def compute_flux_statistics(tau: np.ndarray, use_cpp: bool = True) -> Dict[str, float]:
    """Compute basic flux statistics from optical depth."""
    if use_cpp and CPP_AVAILABLE:
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
    else:
        return _python_analysis.compute_flux_statistics(tau)


def compute_power_spectrum(
    flux: np.ndarray,
    velocity_spacing: float,
    chunk_size: int = 1000,
    use_cpp: bool = True
) -> Dict[str, Any]:
    """Compute power spectrum with optional C++ acceleration."""
    if use_cpp and CPP_AVAILABLE:
        if flux.dtype != np.float32:
            flux = flux.astype(np.float32)
        result = _cpp_compute_power_spectrum(flux, velocity_spacing, chunk_size)
        return {
            'k': np.array(result['k']),
            'P_k_mean': np.array(result['P_k_mean']),
            'P_k_std': np.array(result['P_k_std']),
            'P_k_err': np.array(result['P_k_err']),
            'mean_flux': result['mean_flux'],
            'n_modes': np.array(result['n_modes']),
            'n_sightlines': result['n_sightlines'],
            'velocity_spacing': result['velocity_spacing'],
        }
    else:
        return _python_analysis.compute_power_spectrum(flux, velocity_spacing, chunk_size)


def compute_column_density_distribution(
    tau: np.ndarray,
    velocity_spacing: float,
    threshold: float = 0.5,
    colden: Optional[np.ndarray] = None,
    redshift: Optional[float] = None,
    box_size_ckpc_h: Optional[float] = None,
    hubble: float = 0.6774,
    omega_m: float = 0.3089,
    use_cpp: bool = True
) -> Dict[str, Any]:
    """Compute column density distribution function."""
    if use_cpp and CPP_AVAILABLE:
        if tau.dtype != np.float32:
            tau = tau.astype(np.float32)
        
        colden_arg = None
        if colden is not None:
            if colden.dtype != np.float32:
                colden = colden.astype(np.float32)
            colden_arg = colden
        
        redshift_val = redshift if redshift is not None else float('nan')
        box_size_val = box_size_ckpc_h if box_size_ckpc_h is not None else float('nan')
        
        result = _cpp_compute_colden(
            tau, velocity_spacing, threshold,
            colden_arg, redshift_val, box_size_val, hubble, omega_m
        )
        
        return {
            'N_HI': np.array(result['N_HI']),
            'counts': np.array(result['counts']),
            'bins': np.array(result['bins']),
            'bin_centers': np.array(result['bin_centers']),
            'f_N': np.array(result['f_N']),
            'beta_fit': result['beta_fit'],
            'n_absorbers': result['n_absorbers'],
            'n_sightlines': result['n_sightlines'],
            'dX': result['dX'],
            'redshift': result['redshift'],
        }
    else:
        return _python_analysis.compute_column_density_distribution(
            tau, velocity_spacing, threshold, colden,
            redshift, box_size_ckpc_h, hubble, omega_m
        )


def compute_line_width_distribution(
    tau: np.ndarray,
    velocity_spacing: float,
    threshold: float = 0.5,
    colden: Optional[np.ndarray] = None,
    use_cpp: bool = True
) -> Dict[str, Any]:
    """Compute line width (b-parameter) distribution."""
    if use_cpp and CPP_AVAILABLE:
        if tau.dtype != np.float32:
            tau = tau.astype(np.float32)
        
        colden_arg = None
        if colden is not None:
            if colden.dtype != np.float32:
                colden = colden.astype(np.float32)
            colden_arg = colden
        
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
    else:
        return _python_analysis.compute_line_width_distribution(
            tau, velocity_spacing, threshold, colden
        )


def compute_temperature_density_relation(
    temperature: np.ndarray,
    density: np.ndarray,
    tau: np.ndarray,
    min_tau: float = 0.1,
    use_cpp: bool = True
) -> Dict[str, Any]:
    """Compute temperature-density relation."""
    if use_cpp and CPP_AVAILABLE:
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
            'n_pixels': result['n_pixels'],
        }
    else:
        return _python_analysis.compute_temperature_density_relation(
            temperature, density, tau, min_tau
        )


def is_cpp_available() -> bool:
    """Check if C++ implementation is available."""
    return CPP_AVAILABLE
