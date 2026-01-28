"""
Enhanced comparison command with glob support, auto-labeling, and overlay plots.
"""

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
from scripts.data_export import save_analysis_results, get_analysis_output_dir
from scripts.analysis import (
    compute_flux_statistics,
    compute_effective_optical_depth,
    compute_power_spectrum,
    compute_column_density_distribution,
)
from scripts.plotting import (
    setup_plot_style,
    plot_power_spectrum_overlay,
    plot_cddf_overlay,
    plot_flux_stats_comparison,
    plot_tau_eff_comparison,
    plot_sample_spectra_comparison,
)


def cmd_compare(args):
    """Compare multiple spectra files with auto-labeling and overlay plots."""
    print("=" * 70)
    print("SIMULATION COMPARISON (ENHANCED)")
    print("=" * 70)
    
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
    
    # Load or compute analysis data
    print("\n[2/4] Loading analysis data...")
    
    analysis_results = []
    flux_arrays = []
    redshifts = []
    velocity_spacing = 0.1  # Default, will be updated from data
    
    for i, (filepath, label) in enumerate(zip(spectra_files, labels)):
        print(f"\n  [{i+1}/{len(spectra_files)}] {label}")
        print(f"  File: {filepath}")
        
        # Try to load pre-computed analysis data
        output_dir = get_analysis_output_dir(filepath)
        json_path = output_dir / 'analysis_results.json'
        
        if json_path.exists():
            print(f"  Loading pre-computed data from: {json_path}")
            try:
                with open(json_path, 'r') as f:
                    results = json.load(f)
                
                # Verify required fields exist
                required = ['power_spectrum', 'cddf', 'flux_stats', 'tau_eff']
                if all(key in results for key in required):
                    analysis_results.append(results)
                    print("  ✓ Loaded successfully")
                else:
                    print(f"  Warning: Missing required fields, computing on-the-fly...")
                    results = compute_analysis_on_the_fly(filepath)
                    analysis_results.append(results)
            except Exception as e:
                print(f"  Warning: Could not load pre-computed data: {e}")
                print("  Computing on-the-fly...")
                results = compute_analysis_on_the_fly(filepath)
                analysis_results.append(results)
        else:
            print(f"  No pre-computed data found")
            print("  Computing analysis on-the-fly...")
            results = compute_analysis_on_the_fly(filepath)
            analysis_results.append(results)
        
        # Load flux data for sample spectra comparison
        try:
            with h5py.File(filepath, 'r') as f:
                # Find tau data (try common paths)
                tau = None
                if 'tau/H/1/1215' in f:
                    tau = f['tau/H/1/1215'][:]
                elif 'tau' in f and isinstance(f['tau'], h5py.Dataset):
                    tau = f['tau'][:]
                
                if tau is not None:
                    flux = np.exp(-tau)
                    flux_arrays.append(flux)
                else:
                    flux_arrays.append(None)
                    print("  Warning: Could not load flux data for sample spectra")
                
                # Load redshift
                if 'Header' in f:
                    header = f['Header'].attrs
                    z = header.get('redshift', header.get('Redshift', None))
                    redshifts.append(z)
                else:
                    redshifts.append(None)
        except Exception as e:
            print(f"  Warning: Could not load flux data: {e}")
            flux_arrays.append(None)
            redshifts.append(None)
    
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
    valid_flux = [f for f in flux_arrays if f is not None]
    if len(valid_flux) >= 2:
        print("  [e] Sample spectra comparison...")
        
        # Create velocity array
        n_pixels = valid_flux[0].shape[1]
        velocity = np.arange(n_pixels) * velocity_spacing
        
        spectra_path = output_dir / 'sample_spectra_comparison.png'
        plot_sample_spectra_comparison(valid_flux, labels, velocity, spectra_path, 
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


def compute_analysis_on_the_fly(filepath):
    """Compute analysis results on-the-fly if pre-computed data is not available."""
    colden = None
    
    with h5py.File(filepath, 'r') as f:
        # Load tau data
        tau = None
        if 'tau/H/1/1215' in f:
            tau = f['tau/H/1/1215'][:]
        elif 'tau' in f and isinstance(f['tau'], h5py.Dataset):
            tau = f['tau'][:]
        else:
            raise ValueError("Could not find tau data in file")
        
        # Load column density if available
        try:
            if 'colden/H/1' in f:
                colden = f['colden/H/1'][:]
        except:
            pass
        
        # Load metadata
        redshift = None
        if 'Header' in f:
            header = f['Header'].attrs
            redshift = header.get('redshift', header.get('Redshift', None))
    
    # Compute flux
    flux = np.exp(-tau)
    
    # Velocity spacing (assume 0.1 km/s)
    velocity_spacing = 0.1
    
    # Compute statistics
    print("    Computing flux statistics...")
    stats = compute_flux_statistics(tau)
    
    print("    Computing tau_eff...")
    tau_eff_dict = compute_effective_optical_depth(tau)
    
    print("    Computing power spectrum...")
    power_dict = compute_power_spectrum(flux, velocity_spacing)
    
    print("    Computing CDDF...")
    cddf_dict = compute_column_density_distribution(tau, velocity_spacing, threshold=0.5, colden=colden)
    
    # Prepare results
    results = {
        'metadata': {
            'spectra_file': filepath,
            'redshift': redshift,
            'computed_on_the_fly': True,
        },
        'flux_stats': stats,
        'tau_eff': tau_eff_dict,
        'power_spectrum': power_dict,
        'cddf': cddf_dict,
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
