"""
CGM Halo Analysis - C++ Implementation Wrapper

This module provides Python bindings to the C++ implementation.
"""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from _cgm_cpp import (
    filter_isolated_halos as _cpp_filter_halos,
    compute_impact_parameters as _cpp_compute_impact
)


def filter_isolated_halos(
    catalog: pd.DataFrame,
    isolation_factor: float = 3.0,
    radius_type: str = 'radius_vir'
) -> pd.DataFrame:
    """Filter isolated halos based on proximity to other halos.
    
    Args:
        catalog: DataFrame with halo catalog (must have position_x, position_y,
                position_z, mass_total, and radius_type columns)
        isolation_factor: Multiplier of viral radius for isolation criterion
        radius_type: Column name for halo radius
    
    Returns:
        DataFrame with isolated halos
    """
    if len(catalog) == 0:
        return catalog
    
    positions = catalog[['position_x', 'position_y', 'position_z']].values.astype(np.float32)
    masses = catalog['mass_total'].values.astype(np.float32)
    radii = catalog[radius_type].values.astype(np.float32)
    
    box_size = 0.0
    if 'boxsize' in catalog.columns:
        box_size = float(catalog.iloc[0]['boxsize']) if 'boxsize' in catalog else 0.0
    
    result = _cpp_filter_halos(positions, masses, radii, isolation_factor, box_size)
    
    isolated_mask = np.array(result['isolated_mask'])
    
    filtered = catalog[isolated_mask == 1].copy()
    
    print(f"Isolation filter ({isolation_factor:.1f} x R_vir): "
          f"{len(filtered):,} / {len(catalog):,} isolated halos")
    
    return filtered


def compute_impact_parameters(
    sightline_origins: np.ndarray,
    sightline_dirs: np.ndarray,
    halo_positions: np.ndarray,
    halo_radii: np.ndarray
) -> np.ndarray:
    """Compute impact parameters between sightlines and halos.
    
    Args:
        sightline_origins: (M, 3) array of sightline origin positions
        sightline_dirs: (M, 3) array of sightline direction vectors
        halo_positions: (N, 3) array of halo positions
        halo_radii: (N,) array of halo radii
    
    Returns:
        (M, N) array of impact parameters
    """
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
