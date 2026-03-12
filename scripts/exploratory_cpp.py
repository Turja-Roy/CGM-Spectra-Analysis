"""
CGM Exploratory Analysis - C++ Implementation Wrapper

This module provides Python bindings to the C++ implementation.
"""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
from typing import Optional, Dict, Any

from _exploratory_cpp import extract_spectral_features as _cpp_extract_features


def extract_spectral_features(
    tau: np.ndarray,
    velocity_spacing: float = 0.1,
    void_threshold: float = 0.9,
    line_threshold: float = 0.5,
    absorber_threshold: float = 0.5,
    max_sightlines: int = 100,
    max_separations: int = 1000,
    max_separation: float = 500.0
) -> Dict[str, Any]:
    """Extract spectral features from optical depth array.
    
    Args:
        tau: 2D array of optical depth (n_sightlines x n_pixels)
        velocity_spacing: Velocity spacing in km/s per pixel
        void_threshold: Flux threshold for void detection (default: 0.9)
        line_threshold: Flux threshold for line detection (default: 0.5)
        absorber_threshold: Tau threshold for absorber detection (default: 0.5)
        max_sightlines: Maximum sightlines to process for absorber clustering
        max_separations: Maximum number of separations to compute
        max_separation: Maximum separation in km/s
    
    Returns:
        Dictionary with spectral features
    """
    if tau.dtype != np.float32:
        tau = tau.astype(np.float32)
    
    result = _cpp_extract_features(
        tau, 
        velocity_spacing,
        void_threshold,
        line_threshold,
        absorber_threshold,
        max_sightlines,
        max_separations,
        max_separation
    )
    
    return {
        'void_sizes': np.array(result['void_sizes']),
        'line_widths': np.array(result['line_widths']),
        'absorber_separations': np.array(result['absorber_separations']),
        'mean_void_size': result['mean_void_size'],
        'median_void_size': result['median_void_size'],
        'mean_line_width': result['mean_line_width'],
        'median_line_width': result['median_line_width'],
        'saturation_fraction': result['saturation_fraction'],
        'deep_absorption_fraction': result['deep_absorption_fraction'],
        'transmission_fraction': result['transmission_fraction'],
        'flux_mean': result['flux_mean'],
        'flux_variance': result['flux_variance'],
        'flux_skewness': result['flux_skewness'],
        'flux_kurtosis': result['flux_kurtosis'],
        'mean_absorber_separation': result['mean_absorber_separation'],
        'n_voids': result['n_voids'],
        'n_lines': result['n_lines'],
        'n_absorbers': result['n_absorbers'],
    }
