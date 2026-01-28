"""
Sightline management utilities for CAMEL spectra analysis.

This module provides functions to generate, save, load, and validate sightlines
for consistent comparison across multiple simulations.
"""

import os
import numpy as np
import h5py
from datetime import datetime
from pathlib import Path


def generate_random_sightlines(n_sightlines, box_size, seed=None, axes_mode='random'):
    """Generate random sightline positions and projection axes."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random positions uniformly across box
    positions = np.random.uniform(0, box_size, size=(n_sightlines, 3))
    
    # Generate axes based on mode
    if axes_mode == 'random':
        axes = np.random.randint(1, 4, size=n_sightlines)
    elif axes_mode == 'all-x':
        axes = np.ones(n_sightlines, dtype=int)
    elif axes_mode == 'all-y':
        axes = np.full(n_sightlines, 2, dtype=int)
    elif axes_mode == 'all-z':
        axes = np.full(n_sightlines, 3, dtype=int)
    elif axes_mode == 'balanced':
        # Distribute evenly across three axes
        axes = np.repeat([1, 2, 3], n_sightlines // 3)
        # Add remainder randomly
        remainder = n_sightlines - len(axes)
        if remainder > 0:
            axes = np.concatenate([axes, np.random.randint(1, 4, size=remainder)])
        np.random.shuffle(axes)
    else:
        raise ValueError(f"Unknown axes_mode: {axes_mode}")
    
    return {
        'positions': positions,
        'axes': axes
    }


def save_sightlines_hdf5(sightlines_dict, output_path, metadata=None):
    """Save sightlines to HDF5 file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Create Sightlines group
        sightlines_group = f.create_group('Sightlines')
        
        # Save positions and axes
        sightlines_group.create_dataset('positions', data=sightlines_dict['positions'])
        sightlines_group.create_dataset('axes', data=sightlines_dict['axes'])
        
        # Save metadata as attributes
        if metadata is None:
            metadata = {}
        
        metadata['creation_date'] = datetime.now().isoformat()
        metadata['n_sightlines'] = len(sightlines_dict['axes'])
        
        for key, value in metadata.items():
            sightlines_group.attrs[key] = value
        
        print(f"Saved {len(sightlines_dict['axes'])} sightlines to: {output_path}")


def load_sightlines_hdf5(input_path):
    """Load sightlines from HDF5 file."""
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Sightlines file not found: {input_path}")
    
    with h5py.File(input_path, 'r') as f:
        if 'Sightlines' not in f:
            raise ValueError(f"No 'Sightlines' group found in {input_path}")
        
        sightlines_group = f['Sightlines']
        
        positions = np.array(sightlines_group['positions'])
        axes = np.array(sightlines_group['axes'])
        
        # Load metadata
        metadata = dict(sightlines_group.attrs)
        
        return {
            'positions': positions,
            'axes': axes,
            'metadata': metadata
        }


def save_sightlines_in_spectra(spectra_filepath, positions, axes, metadata=None):
    """Save sightlines inside an existing spectra HDF5 file."""
    with h5py.File(spectra_filepath, 'a') as f:
        # Remove existing Sightlines group if present
        if 'Sightlines' in f:
            del f['Sightlines']
        
        # Create new Sightlines group
        sightlines_group = f.create_group('Sightlines')
        
        # Save data
        sightlines_group.create_dataset('positions', data=positions)
        sightlines_group.create_dataset('axes', data=axes)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata['creation_date'] = datetime.now().isoformat()
        metadata['n_sightlines'] = len(axes)
        
        for key, value in metadata.items():
            sightlines_group.attrs[key] = value


def load_sightlines_from_spectra(spectra_filepath):
    """Load sightlines from a spectra HDF5 file."""
    return load_sightlines_hdf5(spectra_filepath)


def validate_sightlines(sightlines_dict, box_size=None):
    """Validate sightlines data."""
    if 'positions' not in sightlines_dict:
        raise ValueError("Sightlines dictionary missing 'positions'")
    
    if 'axes' not in sightlines_dict:
        raise ValueError("Sightlines dictionary missing 'axes'")
    
    positions = sightlines_dict['positions']
    axes = sightlines_dict['axes']
    
    # Check shapes
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Positions must be N x 3, got shape {positions.shape}")
    
    if axes.ndim != 1:
        raise ValueError(f"Axes must be 1D array, got shape {axes.shape}")
    
    if len(positions) != len(axes):
        raise ValueError(f"Positions ({len(positions)}) and axes ({len(axes)}) length mismatch")
    
    # Check axes values
    if not np.all((axes >= 1) & (axes <= 3)):
        raise ValueError("Axes must be 1, 2, or 3")
    
    # Check box size compatibility if provided
    if box_size is not None:
        if np.any(positions < 0) or np.any(positions > box_size):
            raise ValueError(f"Positions outside box [0, {box_size}]")
    
    return True


def check_sightlines_compatibility(sightlines1, sightlines2, tolerance=1e-6):
    """Check if two sightline sets are compatible."""
    if len(sightlines1['axes']) != len(sightlines2['axes']):
        return False
    
    if not np.array_equal(sightlines1['axes'], sightlines2['axes']):
        return False
    
    if not np.allclose(sightlines1['positions'], sightlines2['positions'], 
                       rtol=tolerance, atol=tolerance):
        return False
    
    return True


def get_sightline_summary(sightlines_dict):
    """Get human-readable summary of sightlines."""
    n = len(sightlines_dict['axes'])
    axes = sightlines_dict['axes']
    positions = sightlines_dict['positions']
    
    # Count axes
    n_x = np.sum(axes == 1)
    n_y = np.sum(axes == 2)
    n_z = np.sum(axes == 3)
    
    # Position statistics
    pos_min = positions.min(axis=0)
    pos_max = positions.max(axis=0)
    pos_mean = positions.mean(axis=0)
    
    summary = f"""Sightlines Summary:
  Total sightlines: {n}
  Axes distribution:
    X-axis (1): {n_x} ({100*n_x/n:.1f}%)
    Y-axis (2): {n_y} ({100*n_y/n:.1f}%)
    Z-axis (3): {n_z} ({100*n_z/n:.1f}%)
  Position range:
    X: [{pos_min[0]:.1f}, {pos_max[0]:.1f}] (mean: {pos_mean[0]:.1f})
    Y: [{pos_min[1]:.1f}, {pos_max[1]:.1f}] (mean: {pos_mean[1]:.1f})
    Z: [{pos_min[2]:.1f}, {pos_max[2]:.1f}] (mean: {pos_mean[2]:.1f})"""
    
    if 'metadata' in sightlines_dict:
        metadata = sightlines_dict['metadata']
        if 'seed' in metadata:
            summary += f"\n  Random seed: {metadata['seed']}"
        if 'creation_date' in metadata:
            summary += f"\n  Created: {metadata['creation_date']}"
    
    return summary
