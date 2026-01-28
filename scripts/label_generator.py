"""
Auto-label generation utilities for comparison plots using CAMEL parameter CSV files.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Mapping of parameter names to formatted labels
PARAMETER_LABELS = {
    'omega0': 'Ω_m',
    'omegam': 'Ω_m',
    'sigma8': 'σ_8',
    'omegab': 'Ω_b',
    'omegabaryon': 'Ω_b',
    'hubble': 'h',
    'hubbleparam': 'h',
    'ns': 'n_s',
    'n_s': 'n_s',
}

# Mapping of CSV column names to standard names
CSV_COLUMN_MAP = {
    'omega0': 'Omega0',
    'omegam': 'Omega0',
    'sigma8': 'sigma8',
    'omegab': 'OmegaBaryon',
    'omegabaryon': 'OmegaBaryon',
    'hubble': 'HubbleParam',
    'hubbleparam': 'HubbleParam',
    'ns': 'n_s',
    'n_s': 'n_s',
}


def load_parameter_table(csv_path):
    """Load CAMEL parameter CSV file."""
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Parameter CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Verify 'Name' column exists
    if 'Name' not in df.columns:
        raise ValueError(f"CSV must have 'Name' column. Found: {df.columns.tolist()}")
    
    return df


def get_parameter_value(param_table, sim_name, param_name):
    """Extract parameter value for a simulation from the parameter table."""
    # Normalize parameter name to lowercase
    param_name_lower = param_name.lower().replace('_', '')
    
    # Get CSV column name
    csv_column = CSV_COLUMN_MAP.get(param_name_lower)
    
    if csv_column is None:
        print(f"Warning: Unknown parameter '{param_name}'")
        return None
    
    if csv_column not in param_table.columns:
        print(f"Warning: Column '{csv_column}' not in parameter table")
        return None
    
    # Find row matching simulation name
    row = param_table[param_table['Name'] == sim_name]
    
    if len(row) == 0:
        print(f"Warning: Simulation '{sim_name}' not found in parameter table")
        return None
    
    value = row[csv_column].values[0]
    
    return value


def generate_labels_from_param(param_table, sim_names, param_name, include_fiducial=False, 
                                fiducial_name=None, format_str=None):
    """Generate labels for simulations based on parameter values."""
    if fiducial_name is None:
        fiducial_name = '1P_0'
    
    # Get parameter symbol for display
    param_name_lower = param_name.lower().replace('_', '')
    param_symbol = PARAMETER_LABELS.get(param_name_lower, param_name)
    
    labels = []
    
    for sim_name in sim_names:
        value = get_parameter_value(param_table, sim_name, param_name)
        
        if value is None:
            # Fallback to simulation name if parameter not found
            labels.append(sim_name)
            continue
        
        # Check if this is fiducial
        is_fiducial = (sim_name == fiducial_name)
        
        # Determine format
        if format_str is None:
            # Auto-detect precision based on value range
            if abs(value) < 0.01:
                fmt = '.4f'
            elif abs(value) < 0.1:
                fmt = '.3f'
            elif abs(value) < 1:
                fmt = '.2f'
            else:
                fmt = '.1f'
        else:
            fmt = format_str
        
        # Format label
        label = f"{param_symbol} = {value:{fmt}}"
        
        if is_fiducial and include_fiducial:
            label += " (fiducial)"
        
        labels.append(label)
    
    return labels


def detect_varying_parameter(param_table, sim_names):
    """Automatically detect which parameter varies across the given simulations."""
    # Get all cosmological parameters
    cosmo_params = ['Omega0', 'sigma8', 'OmegaBaryon', 'HubbleParam', 'n_s']
    
    varying_params = []
    
    for param_col in cosmo_params:
        if param_col not in param_table.columns:
            continue
        
        # Get values for selected simulations
        values = []
        for sim_name in sim_names:
            row = param_table[param_table['Name'] == sim_name]
            if len(row) > 0:
                values.append(row[param_col].values[0])
        
        if len(values) == 0:
            continue
        
        # Check if parameter varies (with tolerance for floating point)
        if len(set(np.round(values, decimals=6))) > 1:
            varying_params.append(param_col)
    
    if len(varying_params) == 1:
        # Convert back to standard name
        col_to_standard = {v: k for k, v in CSV_COLUMN_MAP.items()}
        standard_name = col_to_standard.get(varying_params[0], varying_params[0])
        return standard_name
    
    return None


def extract_sim_names_from_paths(file_paths):
    """Extract simulation names from file paths."""
    sim_names = []
    
    for filepath in file_paths:
        filepath = Path(filepath)
        
        # Try to extract from path structure
        # Expected: .../{suite}/{sim_set}/{sim_name}/...
        parts = filepath.parts
        
        # Look for pattern matching 1P_*, LH_*, etc.
        for part in reversed(parts):
            if '_' in part:
                # Check if it looks like a simulation name
                if any(part.startswith(prefix) for prefix in ['1P_', 'LH_', 'CV_', 'EX_']):
                    sim_names.append(part)
                    break
        else:
            # Fallback: use filename stem
            sim_names.append(filepath.stem)
    
    return sim_names


def get_fiducial_name(sim_set='1P'):
    """Get fiducial simulation name for a given set."""
    fiducial_map = {
        '1P': '1P_0',
        'LH': 'LH_0',
        'CV': 'CV_0',
        'EX': 'EX_0',
    }
    
    return fiducial_map.get(sim_set, f'{sim_set}_0')


def format_parameter_range(values, param_name):
    """Format a parameter range for display."""
    if len(values) == 0:
        return "N/A"
    
    param_name_lower = param_name.lower().replace('_', '')
    param_symbol = PARAMETER_LABELS.get(param_name_lower, param_name)
    
    min_val = np.min(values)
    max_val = np.max(values)
    
    # Auto-detect precision
    if max_val - min_val < 0.01:
        fmt = '.4f'
    elif max_val - min_val < 0.1:
        fmt = '.3f'
    elif max_val - min_val < 1:
        fmt = '.2f'
    else:
        fmt = '.1f'
    
    if min_val == max_val:
        return f"{param_symbol} = {min_val:{fmt}}"
    else:
        return f"{param_symbol} ∈ [{min_val:{fmt}}, {max_val:{fmt}}]"
