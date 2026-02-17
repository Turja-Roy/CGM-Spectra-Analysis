import os
import sys
import json
import glob
import h5py
import numpy as np
from pathlib import Path

import scripts.config as config
from scripts.label_generator import (
    load_parameter_table,
    extract_sim_names_from_paths,
    generate_labels_from_param,
    detect_varying_parameter,
    get_fiducial_name,
)
from scripts.data_export import get_analysis_output_dir
from scripts.plotting import (
    setup_plot_style,
    plot_power_spectrum_overlay,
    plot_cddf_overlay,
    plot_flux_stats_comparison,
    plot_tau_eff_comparison,
    plot_sample_spectra_comparison,
)


def parse_sightline_indices(sightlines_str):
    """
    Parse comma-separated sightline indices string into a list of integers.
    
    Parameters:
    -----------
    sightlines_str : str or None
        Comma-separated string of sightline indices (e.g., "0,5,10,25,50")
    
    Returns:
    --------
    indices : list of int or None
        List of sightline indices, or None if input is None
    
    Raises:
    -------
    ValueError : if parsing fails or indices are invalid
    """
    if sightlines_str is None:
        return None
    
    try:
        indices = [int(idx.strip()) for idx in sightlines_str.split(',')]
        
        # Validate: no negative indices
        if any(idx < 0 for idx in indices):
            raise ValueError("Sightline indices must be non-negative")
        
        # Validate: no duplicates (warn but allow)
        if len(indices) != len(set(indices)):
            print("Warning: Duplicate sightline indices detected, using unique values")
            indices = sorted(list(set(indices)))
        
        return indices
    
    except ValueError as e:
        raise ValueError(f"Invalid sightline indices format '{sightlines_str}': {e}")


def select_sightlines(n_total, user_indices=None, n_default=5, seed=42):
    """
    Select sightline indices for analysis.
    
    Parameters:
    -----------
    n_total : int
        Total number of sightlines available in the file
    user_indices : list of int or None
        User-specified sightline indices (takes precedence)
    n_default : int
        Number of sightlines to randomly sample if user_indices is None
    seed : int
        Random seed for reproducible sampling
    
    Returns:
    --------
    indices : ndarray
        Selected sightline indices (sorted)
    
    Raises:
    -------
    ValueError : if user-specified indices are out of bounds
    """
    if user_indices is not None:
        # User specified exact indices
        invalid = [idx for idx in user_indices if idx >= n_total]
        if invalid:
            raise ValueError(
                f"Sightline indices out of bounds: {invalid}. "
                f"File only has {n_total} sightlines (indices 0-{n_total-1})"
            )
        return np.array(sorted(user_indices))
    
    else:
        # Default: random sampling
        n_sample = min(n_default, n_total)
        np.random.seed(seed)
        indices = np.random.choice(n_total, n_sample, replace=False)
        return np.sort(indices)


def compute_velocity_spacing(header):
    """
    Compute velocity spacing (dvbin) from HDF5 header attributes.
    
    Tries to load dvbin directly if available (new files), otherwise
    computes from box size, Hubble parameter, and number of bins (backward compatible).
    
    Parameters:
    -----------
    header : h5py.AttributeManager
        HDF5 header attributes
    
    Returns:
    --------
    dvbin : float
        Velocity spacing in km/s per pixel
    """
    # Try to load directly (new files)
    if 'dvbin' in header:
        return float(header['dvbin'])
    
    # Compute from other attributes (backward compatible)
    try:
        nbins = header['nbins']
        box = header['box']  # ckpc/h
        Hz = header['Hz']    # km/s/Mpc
        hubble = header['hubble']  # h parameter
        
        # Compute vmax: convert box to cMpc/h, then to velocity
        vmax = (box / 1000.0) * Hz / hubble  # km/s
        
        # Compute velocity spacing
        dvbin = 2.0 * vmax / nbins
        
        return dvbin
    except KeyError as e:
        raise ValueError(f"Could not compute velocity spacing: missing header attribute {e}")


def cmd_compare(args):
    """Compare multiple spectra files with auto-labeling and overlay plots."""
    print("=" * 70)
    print("SIMULATION COMPARISON (ENHANCED)")
    print("=" * 70)
    
    # Parse sightline indices if provided
    try:
        user_sightlines = parse_sightline_indices(getattr(args, 'sightlines', None))
        if user_sightlines is not None:
            print(f"\nUser-specified sightlines: {user_sightlines}")
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Expand file patterns (glob)
    spectra_files = []
    for pattern in args.spectra_files:
        if '*' in pattern or '?' in pattern:
            expanded = glob.glob(pattern)
            if not expanded:
                print(f"Warning: Pattern '{pattern}' matched no files")
            spectra_files.extend(expanded)
        else:
            spectra_files.append(pattern)
    
    # Remove duplicates and sort
    spectra_files = sorted(list(set(spectra_files)))
    
    if len(spectra_files) < 2:
        print(f"Error: Need at least 2 files to compare. Got {len(spectra_files)}")
        if len(args.spectra_files) == 1 and ('*' in args.spectra_files[0] or '?' in args.spectra_files[0]):
            print(f"Glob pattern '{args.spectra_files[0]}' matched {len(spectra_files)} file(s)")
        return 1
    
    print(f"\nFound {len(spectra_files)} files to compare:")
    for i, f in enumerate(spectra_files, 1):
        print(f"  [{i}] {f}")
    
    # Check if files exist
    for f in spectra_files:
        if not os.path.exists(f):
            print(f"\nError: File not found: {f}")
            return 1
    
    # Generate labels
    print("\n[1/4] Generating labels...")
    
    if args.labels:
        # User-provided labels
        labels = [l.strip() for l in args.labels.split(',')]
        if len(labels) != len(spectra_files):
            print(f"Error: Number of labels ({len(labels)}) doesn't match files ({len(spectra_files)})")
            return 1
        print("Using user-provided labels")
    
    elif hasattr(args, 'param') and args.param:
        # Auto-label from parameter
        print(f"Auto-generating labels from parameter: {args.param}")
        
        # Load parameter CSV
        param_csv_path = config.DATA_DIR / 'IllustrisTNG' / '1P' / 'CosmoAstroSeed_IllustrisTNG_L25n256_1P.csv'
        
        if not param_csv_path.exists():
            print(f"Warning: Parameter CSV not found: {param_csv_path}")
            print("Falling back to simulation names")
            sim_names = extract_sim_names_from_paths(spectra_files)
            labels = sim_names
        else:
            try:
                param_table = load_parameter_table(param_csv_path)
                sim_names = extract_sim_names_from_paths(spectra_files)
                
                # Determine fiducial
                fiducial_name = args.fiducial if hasattr(args, 'fiducial') and args.fiducial else get_fiducial_name('1P')
                
                labels = generate_labels_from_param(
                    param_table, sim_names, args.param, 
                    include_fiducial=True, fiducial_name=fiducial_name
                )
                
                print(f"Generated labels based on {args.param}")
            except Exception as e:
                print(f"Warning: Could not auto-generate labels: {e}")
                print("Falling back to simulation names")
                sim_names = extract_sim_names_from_paths(spectra_files)
                labels = sim_names
    else:
        # Auto-detect parameter or use simulation names
        print("Auto-detecting varying parameter...")
        
        param_csv_path = config.DATA_DIR / 'IllustrisTNG' / '1P' / 'CosmoAstroSeed_IllustrisTNG_L25n256_1P.csv'
        sim_names = extract_sim_names_from_paths(spectra_files)
        
        if param_csv_path.exists():
            try:
                param_table = load_parameter_table(param_csv_path)
                detected_param = detect_varying_parameter(param_table, sim_names)
                
                if detected_param:
                    print(f"Detected varying parameter: {detected_param}")
                    fiducial_name = get_fiducial_name('1P')
                    labels = generate_labels_from_param(
                        param_table, sim_names, detected_param,
                        include_fiducial=True, fiducial_name=fiducial_name
                    )
                else:
                    print("Could not detect single varying parameter, using simulation names")
                    labels = sim_names
            except Exception as e:
                print(f"Warning: Auto-detection failed: {e}")
                labels = sim_names
        else:
            print("Parameter CSV not found, using simulation names")
            labels = sim_names
    
    print("\nLabels:")
    for i, label in enumerate(labels, 1):
        print(f"  [{i}] {label}")
    
    # Load analysis data from CSV files
    print("\n[2/4] Loading analysis data from CSV files...")
    
    analysis_results = []
    flux_arrays = []
    redshifts = []
    velocity_spacings = []  # Store velocity spacing for each file
    
    for i, (filepath, label) in enumerate(zip(spectra_files, labels)):
        print(f"\n  [{i+1}/{len(spectra_files)}] {label}")
        print(f"  File: {filepath}")
        
        # Load analysis data from CSV files only
        output_dir = get_analysis_output_dir(filepath)
        
        try:
            results = load_analysis_from_csv(output_dir)
            
            # Verify required fields exist
            required = ['power_spectrum', 'cddf', 'flux_stats', 'tau_eff']
            missing = [key for key in required if key not in results]
            
            if missing:
                print(f"  Error: Missing required data: {', '.join(missing)}")
                print(f"  Please run: python analyze_spectra.py analyze {filepath}")
                return 1
            
            analysis_results.append(results)
            print("  ✓ Loaded from CSV files")
            
        except Exception as e:
            print(f"  Error: Failed to load CSV files: {e}")
            print(f"  Directory: {output_dir}")
            print(f"  Please run: python analyze_spectra.py analyze {filepath}")
            return 1
        
        # Load flux data for sample spectra (subsample to save memory)
        try:
            with h5py.File(filepath, 'r') as f:
                # Find tau data (try common paths)
                tau_dataset = None
                if 'tau/H/1/1215' in f:
                    tau_dataset = f['tau/H/1/1215']
                elif 'tau' in f and isinstance(f['tau'], h5py.Dataset):
                    tau_dataset = f['tau']
                
                if tau_dataset is not None:
                    # Select sightlines: user-specified or random sampling
                    n_sightlines_total = tau_dataset.shape[0]
                    
                    try:
                        sample_indices = select_sightlines(
                            n_sightlines_total, 
                            user_indices=user_sightlines,
                            n_default=5,
                            seed=42
                        )
                    except ValueError as e:
                        print(f"  Error: {e}")
                        print(f"  File: {filepath}")
                        return 1
                    
                    # Load only the selected sightlines (huge memory savings!)
                    tau_subset = tau_dataset[sample_indices, :]
                    flux_subset = np.exp(-tau_subset)
                    flux_arrays.append(flux_subset)
                    
                    if user_sightlines is not None:
                        print(f"  Loaded {len(sample_indices)} specified sightlines: {list(sample_indices)}")
                    else:
                        print(f"  Loaded {len(sample_indices)} random sightlines (out of {n_sightlines_total})")
                else:
                    flux_arrays.append(None)
                    print("  Warning: Could not load flux data for sample spectra")
                
                # Load or compute velocity spacing
                header = f['Header'].attrs
                dvbin = compute_velocity_spacing(header)
                velocity_spacings.append(dvbin)
                print(f"  Velocity spacing: {dvbin:.4f} km/s")
                
                # Load redshift
                z = header.get('redshift', header.get('Redshift', None))
                redshifts.append(z)
                
        except Exception as e:
            print(f"  Warning: Could not load flux data: {e}")
            flux_arrays.append(None)
            redshifts.append(None)
            velocity_spacings.append(None)
    
    # Use first non-None redshift
    redshift = next((z for z in redshifts if z is not None), None)
    
    # Determine fiducial index
    fiducial_idx = None
    if hasattr(args, 'fiducial') and args.fiducial:
        fiducial_name = args.fiducial
        # Try to match fiducial name in file paths or labels
        for i, filepath in enumerate(spectra_files):
            if fiducial_name in filepath or fiducial_name in labels[i]:
                fiducial_idx = i
                print(f"\nFiducial simulation: {labels[i]} (index {i})")
                break
        
        if fiducial_idx is None:
            print(f"\nWarning: Could not find fiducial '{fiducial_name}', using first simulation")
            fiducial_idx = 0
    
    # Create output directory
    print("\n[3/4] Setting up output directory...")
    
    if hasattr(args, 'name') and args.name:
        comparison_name = args.name
    else:
        # Auto-generate name from first and last simulation
        sim_names = extract_sim_names_from_paths(spectra_files)
        comparison_name = f"comparison_{sim_names[0]}_to_{sim_names[-1]}"
    
    output_dir = config.COMPARISON_DIR / comparison_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Save comparison data
    comparison_data = {
        'files': spectra_files,
        'labels': labels,
        'redshift': redshift,
        'fiducial_idx': fiducial_idx,
        'analysis_results': analysis_results,
    }
    
    comparison_json_path = output_dir / 'comparison_data.json'
    with open(comparison_json_path, 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    print(f"Saved comparison data: {comparison_json_path}")
    
    # Generate plots
    print("\n[4/4] Generating comparison plots...")
    
    setup_plot_style()
    
    # Extract data for plotting
    power_dicts = [r['power_spectrum'] for r in analysis_results]
    cddf_dicts = [r['cddf'] for r in analysis_results]
    stats_list = [r['flux_stats'] for r in analysis_results]
    tau_eff_list = [r['tau_eff'] for r in analysis_results]
    
    # 1. Power spectrum overlay
    print("  [a] Power spectrum overlay...")
    ps_path = output_dir / 'power_spectrum_overlay.png'
    plot_power_spectrum_overlay(power_dicts, labels, ps_path, redshift, fiducial_idx)
    print(f"      Saved: {ps_path}")
    
    # Save power spectrum data
    ps_data_path = output_dir / 'power_spectrum_overlay_data.csv'
    save_power_spectrum_data_csv(power_dicts, labels, ps_data_path)
    print(f"      Data: {ps_data_path}")
    
    # 2. CDDF overlay
    print("  [b] CDDF overlay...")
    cddf_path = output_dir / 'cddf_overlay.png'
    plot_cddf_overlay(cddf_dicts, labels, cddf_path, redshift)
    print(f"      Saved: {cddf_path}")
    
    # 3. Flux statistics comparison
    print("  [c] Flux statistics comparison...")
    stats_path = output_dir / 'flux_stats_comparison.png'
    plot_flux_stats_comparison(stats_list, labels, stats_path, redshift)
    print(f"      Saved: {stats_path}")
    
    # 4. Tau_eff comparison
    print("  [d] Effective optical depth comparison...")
    tau_path = output_dir / 'tau_eff_comparison.png'
    plot_tau_eff_comparison(tau_eff_list, labels, tau_path, redshift)
    print(f"      Saved: {tau_path}")
    
    # 5. Sample spectra comparison (if data available)
    # Filter flux arrays and corresponding labels/velocity spacings
    valid_indices = [i for i, f in enumerate(flux_arrays) if f is not None and velocity_spacings[i] is not None]
    valid_flux = [flux_arrays[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]
    valid_velocity_spacings = [velocity_spacings[i] for i in valid_indices]
    
    if len(valid_flux) >= 2:
        print("  [e] Sample spectra comparison...")
        
        # Check if all files have the same pixel count
        pixel_counts = [f.shape[1] for f in valid_flux]
        unique_counts = set(pixel_counts)
        
        if len(unique_counts) > 1:
            print(f"      Warning: Files have different pixel counts: {unique_counts}")
            print(f"      Will interpolate all spectra to common velocity grid")
            
            # Find the file with the most pixels (highest resolution)
            max_pixels_idx = np.argmax(pixel_counts)
            max_pixels = pixel_counts[max_pixels_idx]
            ref_dvbin = valid_velocity_spacings[max_pixels_idx]
            
            # Create reference velocity array
            velocity_ref = np.arange(max_pixels) * ref_dvbin
            
            # Interpolate all flux arrays to reference grid
            flux_interp = []
            for j, flux in enumerate(valid_flux):
                n_pix = flux.shape[1]
                dvbin = valid_velocity_spacings[j]
                velocity_orig = np.arange(n_pix) * dvbin
                
                # Interpolate each sightline
                flux_interp_j = np.zeros((flux.shape[0], max_pixels))
                for k in range(flux.shape[0]):
                    flux_interp_j[k, :] = np.interp(velocity_ref, velocity_orig, flux[k, :])
                
                flux_interp.append(flux_interp_j)
            
            # Use interpolated data
            spectra_path = output_dir / 'sample_spectra_comparison.png'
            plot_sample_spectra_comparison(flux_interp, valid_labels, velocity_ref, spectra_path, 
                                            n_samples=5, redshift=redshift)
            print(f"      Saved: {spectra_path}")
        else:
            # All files have same pixel count - no interpolation needed
            print(f"      All files have matching pixel count: {pixel_counts[0]}")
            
            # Use velocity spacing from first valid file
            dvbin = valid_velocity_spacings[0]
            n_pixels = valid_flux[0].shape[1]
            velocity = np.arange(n_pixels) * dvbin
            
            spectra_path = output_dir / 'sample_spectra_comparison.png'
            plot_sample_spectra_comparison(valid_flux, valid_labels, velocity, spectra_path, 
                                            n_samples=5, redshift=redshift)
            print(f"      Saved: {spectra_path}")
    else:
        print("  [e] Sample spectra comparison skipped (insufficient flux data)")
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"\nCompared {len(spectra_files)} simulations:")
    for label in labels:
        print(f"  - {label}")
    
    if redshift:
        print(f"\nRedshift: z = {redshift:.3f}")
    
    if fiducial_idx is not None:
        print(f"Fiducial: {labels[fiducial_idx]}")
    
    print(f"\nOutput directory: {output_dir}")
    print("Generated plots:")
    print("  - power_spectrum_overlay.png")
    print("  - cddf_overlay.png")
    print("  - flux_stats_comparison.png")
    print("  - tau_eff_comparison.png")
    if len(valid_flux) >= 2:
        print("  - sample_spectra_comparison.png")
    
    print("\nData files:")
    print("  - comparison_data.json")
    print("  - power_spectrum_overlay_data.csv")
    
    print("=" * 70)
    
    return 0


def load_analysis_from_csv(output_dir):
    """Load analysis data from CSV files."""
    import pandas as pd
    
    output_dir = Path(output_dir)
    
    results = {
        'metadata': {
            'loaded_from': 'csv',
            'output_dir': str(output_dir)
        }
    }
    
    # Load power spectrum (required)
    ps_path = output_dir / 'power_spectrum.csv'
    if not ps_path.exists():
        raise FileNotFoundError(f"Missing power_spectrum.csv in {output_dir}")
    
    df = pd.read_csv(ps_path, sep=',', engine='python')
    results['power_spectrum'] = {
        'k': df['k_s_per_km'].values,
        'P_k_mean': df['P_k_mean_km_per_s'].values,
    }
    if 'P_k_std' in df.columns:
        results['power_spectrum']['P_k_std'] = df['P_k_std'].values
    if 'P_k_err' in df.columns:
        results['power_spectrum']['P_k_err'] = df['P_k_err'].values
    
    # Load CDDF (required)
    cddf_path = output_dir / 'cddf.csv'
    if not cddf_path.exists():
        raise FileNotFoundError(f"Missing cddf.csv in {output_dir}")
    
    df = pd.read_csv(cddf_path, sep=',', engine='python', comment='#')
    results['cddf'] = {
        'log10_N_HI': df['log10_N_HI'].values,
        'f_N_HI': df['f_N_HI'].values,
        'f_N': df['f_N_HI'].values,  # Alias for backward compatibility
    }
    if 'counts' in df.columns:
        results['cddf']['counts'] = df['counts'].values
    
    # Load flux stats (required)
    stats_path = output_dir / 'flux_stats.csv'
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing flux_stats.csv in {output_dir}")
    
    df = pd.read_csv(stats_path, sep=',', engine='python')
    # Convert from two-column format (statistic, value) to dict
    results['flux_stats'] = dict(zip(df['statistic'], df['value']))
    
    # Extract tau_eff from flux_stats (required)
    results['tau_eff'] = {
        'tau_eff': results['flux_stats'].get('effective_tau', None),
        'mean_flux': results['flux_stats'].get('mean_flux', None),
        'tau_eff_err': None,  # Not available from CSV (would need per-sightline data)
        'tau_eff_std': None,
    }
    
    return results


def save_power_spectrum_data_csv(power_dicts, labels, output_path):
    """Save power spectrum overlay data to CSV."""
    import pandas as pd
    
    # Create columns for each simulation
    data = {}
    
    # Use k from first simulation as reference
    k_ref = power_dicts[0]['k']
    data['k_s_per_km'] = k_ref
    
    for i, (power_dict, label) in enumerate(zip(power_dicts, labels)):
        k = power_dict['k']
        kPk_pi = power_dict.get('kPk_pi', k * power_dict['P_k_mean'] / np.pi)
        
        # Interpolate to reference k if needed
        if len(k) != len(k_ref) or not np.allclose(k, k_ref):
            kPk_pi_interp = np.interp(k_ref, k, kPk_pi)
        else:
            kPk_pi_interp = kPk_pi
        
        # Clean label for column name (remove special characters)
        col_name = f'kPk_pi_{label}'.replace(' ', '_').replace('=', '').replace('(', '').replace(')', '')
        data[col_name] = kPk_pi_interp
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, float_format='%.6e')
