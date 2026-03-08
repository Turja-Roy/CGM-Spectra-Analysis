"""
CGM Halo Analysis - C++ Implementation Wrapper

This module provides Python bindings to the C++ implementation
with automatic fallback to pure Python if C++ is not available.
"""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

try:
    from _cgm_cpp import (
        filter_isolated_halos as _cpp_filter_halos,
        compute_impact_parameters as _cpp_compute_impact
    )
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

from scripts.cgm import halos as _python_halos


def filter_isolated_halos(
    catalog: pd.DataFrame,
    isolation_factor: float = 3.0,
    radius_type: str = 'radius_vir',
    use_cpp: bool = True
) -> pd.DataFrame:
    """Filter isolated halos based on proximity to other halos.
    
    Args:
        catalog: DataFrame with halo catalog (must have position_x, position_y,
                position_z, mass_total, and radius_type columns)
        isolation_factor: Multiplier of viral radius for isolation criterion
        radius_type: Column name for halo radius
        use_cpp: Use C++ if available (default: True)
    
    Returns:
        DataFrame with isolated halos
    """
    if use_cpp and CPP_AVAILABLE:
        if len(catalog) == 0:
            return catalog
        
        positions = catalog[['position_x', 'position_y', 'position_z']].values.astype(np.float32)
        masses = catalog['mass_total'].values.astype(np.float32)
        radii = catalog[radius_type].values.astype(np.float32)
        
        # Get box size if available
        box_size = 0.0
        if 'boxsize' in catalog.columns:
            box_size = float(catalog.iloc[0]['boxsize']) if 'boxsize' in catalog else 0.0
        
        result = _cpp_filter_halos(positions, masses, radii, isolation_factor, box_size)
        
        isolated_mask = np.array(result['isolated_mask'])
        
        filtered = catalog[isolated_mask == 1].copy()
        
        print(f"Isolation filter ({isolation_factor:.1f} x R_vir): "
              f"{len(filtered):,} / {len(catalog):,} isolated halos")
        
        return filtered
    else:
        return _python_halos.filter_isolated_halos(catalog, isolation_factor, radius_type)


def compute_impact_parameters(
    sightline_origins: np.ndarray,
    sightline_dirs: np.ndarray,
    halo_positions: np.ndarray,
    halo_radii: np.ndarray,
    use_cpp: bool = True
) -> np.ndarray:
    """Compute impact parameters between sightlines and halos.
    
    Args:
        sightline_origins: (M, 3) array of sightline origin positions
        sightline_dirs: (M, 3) array of sightline direction vectors
        halo_positions: (N, 3) array of halo positions
        halo_radii: (N,) array of halo radii
        use_cpp: Use C++ if available (default: True)
    
    Returns:
        (M, N) array of impact parameters
    """
    if use_cpp and CPP_AVAILABLE:
        if sightline_origins.dtype != np.float32:
            sightline_origins = sightline_origins.astype(np.float32)
        if sightline_dirs.dtype != np.float32:
            sightline_dirs = sightline_dirs.astype(np.float32)
        if halo_positions.dtype != np.float32:
            halo_positions = halo_positions.astype(np.float32)
        if halo_radii.dtype != np.float32:
            halo_radii = halo_radii.astype(np.float32)
        
        result = _cpp_compute_impact(sightline_origins, sightline_dirs, halo_positions, halo_radii)
        return np.array(result)
    else:
        from scripts.cgm.targeted_spectra import compute_impact_parameters as python_compute_impact
        return python_compute_impact(sightline_origins, sightline_dirs, halo_positions, halo_radii)


def is_cpp_available() -> bool:
    """Check if C++ implementation is available."""
    return CPP_AVAILABLE
