"""
CGM Analysis - C++ Implementation Wrapper

This module provides Python bindings to the C++ implementation
with automatic fallback to pure Python if C++ is not available.

For Voigt profile fitting, we use fake_spectra.voigtfit which provides
exact Faddeeva function evaluation via scipy.special.wofz().
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

# Import fake_spectra.voigtfit for accurate Voigt profile fitting
try:
    from fake_spectra.voigtfit import Profiles as VoigtProfiles
    VOIGTFIT_AVAILABLE = True
except ImportError:
    VOIGTFIT_AVAILABLE = False

from scripts import analysis as _python_analysis

# Physical constants for temperature calculation
# T = (m_H / 2 k_B) * b^2 where b is in km/s
# m_H = 1.6735575e-27 kg, k_B = 1.380649e-23 J/K
# Factor = m_H / (2 * k_B) * (km/s)^2 = 1.6735575e-27 / (2 * 1.380649e-23) * 1e6
B_TO_T_FACTOR = 60.57  # K / (km/s)^2


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
        
        if colden is not None:
            if colden.dtype != np.float32:
                colden = colden.astype(np.float32)
            colden_arg = colden
        else:
            colden_arg = np.array([], dtype=np.float32).reshape(0, 0)
        
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
    use_cpp: bool = True,
    use_voigtfit: bool = True,
    elem: str = "H",
    ion: int = 1,
    line: int = 1215
) -> Dict[str, Any]:
    """Compute line width (b-parameter) distribution using Voigt profile fitting.
    
    This function uses fake_spectra.voigtfit for accurate Voigt profile fitting
    with the exact Faddeeva function (scipy.special.wofz). Falls back to C++
    approximate fitting or pure Python if fake_spectra is not available.
    
    Args:
        tau: Optical depth array, shape (n_sightlines, n_pixels)
        velocity_spacing: Velocity bin size in km/s
        threshold: Minimum optical depth for peak detection (used by C++ fallback)
        colden: Optional column density array (not used by voigtfit)
        use_cpp: If True and fake_spectra unavailable, use C++ fallback
        use_voigtfit: If True, prefer fake_spectra.voigtfit (default)
        elem: Element for line fitting (default "H" for hydrogen)
        ion: Ionization state (default 1 for HI)
        line: Line wavelength in Angstrom (default 1215 for Lyman-alpha)
    
    Returns:
        Dictionary with N_HI, b_params, temperatures, and statistics
    """
    # Primary path: use fake_spectra.voigtfit for exact Voigt fitting
    if use_voigtfit and VOIGTFIT_AVAILABLE:
        return _compute_line_width_voigtfit(
            tau, velocity_spacing, elem, ion, line
        )
    
    # Fallback 1: C++ implementation (Tepper-García approximation)
    if use_cpp and CPP_AVAILABLE:
        if tau.dtype != np.float32:
            tau = tau.astype(np.float32)
        
        if colden is not None:
            if colden.dtype != np.float32:
                colden = colden.astype(np.float32)
            colden_arg = colden
        else:
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
    
    # Fallback 2: Pure Python implementation
    return _python_analysis.compute_line_width_distribution(
        tau, velocity_spacing, threshold, colden
    )


def _compute_line_width_voigtfit(
    tau: np.ndarray,
    velocity_spacing: float,
    elem: str = "H",
    ion: int = 1,
    line: int = 1215
) -> Dict[str, Any]:
    """Compute line widths using fake_spectra.voigtfit.
    
    Uses the exact Faddeeva function via scipy.special.wofz() for
    accurate Voigt profile fitting. This is the recommended method.
    
    Algorithm from fake_spectra (based on AUTOVP by Oppenheimer & Dave):
    1. Find peaks in optical depth
    2. Iteratively fit and remove peaks
    3. Global re-fit of all peaks simultaneously
    4. Uses Nelder-Mead optimization
    """
    n_sightlines = tau.shape[0] if tau.ndim > 1 else 1
    
    # Handle 1D vs 2D input
    if tau.ndim == 1:
        tau_2d = tau.reshape(1, -1)
    else:
        tau_2d = tau
    
    all_N_HI = []
    all_b_params = []
    
    for i in range(tau_2d.shape[0]):
        tau_line = tau_2d[i]
        
        # Skip sightlines with no significant absorption
        if np.max(tau_line) < 0.01:
            continue
        
        try:
            # Create Voigt fitter using fake_spectra
            prof = VoigtProfiles(
                tau_line, 
                velocity_spacing, 
                profile="Voigt",
                elem=elem, 
                ion=ion, 
                line=line
            )
            
            # Perform the fit
            prof.do_fit()
            
            # Extract results
            b_vals = prof.get_b_params()
            N_vals = prof.get_column_densities()
            
            if len(b_vals) > 0 and len(N_vals) > 0:
                # Filter to physical range
                valid = (b_vals > 2) & (b_vals < 100) & (N_vals > 1e10) & (N_vals < 1e22)
                all_b_params.extend(b_vals[valid])
                all_N_HI.extend(N_vals[valid])
                
        except Exception as e:
            # Skip problematic sightlines
            continue
    
    # Convert to arrays
    N_HI = np.array(all_N_HI, dtype=np.float64)
    b_params = np.array(all_b_params, dtype=np.float64)
    
    # Compute derived quantities
    if len(b_params) > 0:
        temperatures = b_params**2 * B_TO_T_FACTOR
        b_sorted = np.sort(b_params)
        b_median = b_sorted[len(b_sorted) // 2]
        b_mean = np.mean(b_params)
        b_std = np.std(b_params)
    else:
        temperatures = np.array([], dtype=np.float64)
        b_median = np.nan
        b_mean = np.nan
        b_std = np.nan
    
    return {
        'N_HI': N_HI,
        'b_params': b_params,
        'temperatures': temperatures,
        'b_median': b_median,
        'b_mean': b_mean,
        'b_std': b_std,
        'n_absorbers': len(b_params),
    }


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


def is_voigtfit_available() -> bool:
    """Check if fake_spectra.voigtfit is available for exact Voigt fitting."""
    return VOIGTFIT_AVAILABLE
