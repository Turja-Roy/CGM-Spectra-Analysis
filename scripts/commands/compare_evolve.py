import scripts.config as config
from scripts.comparison import compare_simulations, track_redshift_evolution, compare_simulations_comprehensive
from pathlib import Path


def cmd_compare(args):
    print("=" * 70)
    print("SIMULATION COMPARISON")
    print("=" * 70)

    labels = None
    if args.labels:
        labels = [l.strip() for l in args.labels.split(',')]
        if len(labels) != len(args.spectra_files):
            print(f"Error: Number of labels ({len(labels)}) doesn't match files ({len(args.spectra_files)})")
            return 1

    if args.mode == 'quick':
        comparisons_dir = config.PLOTS_DIR / 'comparisons'
        comparisons_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output if args.output else comparisons_dir / "simulation_comparison.png"
        
        comparison = compare_simulations(args.spectra_files, labels=labels, output_path=output_path)
        
        if comparison is None:
            return 1
        
        print("\n" + "=" * 70)
        print("COMPARISON COMPLETE")
        print(f"Plot: {output_path}")
        print("=" * 70)
        return 0
    
    else:
        output_dir = Path(args.output) if args.output else config.PLOTS_DIR / 'comparisons'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        comparison = compare_simulations_comprehensive(
            args.spectra_files,
            labels=labels,
            output_dir=output_dir,
            mode=args.mode
        )
        
        if comparison is None:
            return 1
        
        print("\n" + "=" * 70)
        print("COMPREHENSIVE COMPARISON COMPLETE")
        print(f"Output directory: {comparison['output_dir']}")
        print("=" * 70)
        return 0


def cmd_evolve(args):
    print("=" * 70)
    print("REDSHIFT EVOLUTION TRACKING")
    print("=" * 70)
    
    # Note: Sightline selection for evolve command
    if hasattr(args, 'sightlines') and args.sightlines:
        print(f"Note: --sightlines option specified, but evolve command uses CSV data")
        print(f"      Sightline selection only affects sample spectra plots (if any)")

    labels = None
    if args.labels:
        labels = [l.strip() for l in args.labels.split(',')]
        if len(labels) != len(args.spectra_files):
            print(f"Error: Number of labels ({len(labels)}) doesn't match files ({len(args.spectra_files)})")
            return 1

    comparisons_dir = config.PLOTS_DIR / 'comparisons'
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output if args.output else comparisons_dir / "redshift_evolution.png"

    evolution = track_redshift_evolution(args.spectra_files, labels=labels, output_path=output_path)

    if evolution is None:
        return 1

    print("\n" + "=" * 70)
    print("EVOLUTION TRACKING COMPLETE")
    print(f"Redshift range: z = {evolution['redshift_range'][0]:.3f} - {evolution['redshift_range'][1]:.3f}")
    print(f"Plot: {output_path}")
    print("=" * 70)

    return 0


def cmd_diagnose(args):
    """Diagnostic analysis of single spectra file."""
    import h5py
    import numpy as np
    from scripts.comparison import load_spectra_results
    from scripts.exploratory import extract_spectral_features, compare_distributions
    
    print("=" * 70)
    print("DIAGNOSTIC ANALYSIS")
    print("=" * 70)
    
    # Note: Sightline selection for diagnose command
    if hasattr(args, 'sightlines') and args.sightlines:
        print(f"Note: --sightlines option specified: {args.sightlines}")
        print(f"      This will be used for sample spectra if diagnostic plots are generated")
    
    output_dir = Path(args.output_dir) if args.output_dir else config.PLOTS_DIR / 'diagnostics'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading: {args.spectra_file}")
    results = load_spectra_results(args.spectra_file)
    
    if not results['success']:
        print(f"Error: {results.get('error', 'unknown error')}")
        return 1
    
    print(f"  z={results['redshift']:.3f}, N={results['n_sightlines']:,} sightlines")
    
    # Load tau data
    with h5py.File(args.spectra_file, 'r') as f:
        if 'tau/H/1/1215' in f:
            tau = np.array(f['tau/H/1/1215'])
        else:
            print("Error: No tau data found")
            return 1
    
    flux = np.exp(-tau)
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(f"  τ_eff:      {results['tau_eff']['tau_eff']:.4f} ± {results['tau_eff']['tau_eff_err']:.4f}")
    print(f"  <F>:        {results['flux_stats']['mean_flux']:.4f}")
    print(f"  σ_F:        {results['flux_stats']['std_flux']:.4f}")
    print(f"  N_abs:      {results['cddf']['n_absorbers']}")
    
    # Distribution analysis
    if args.distribution:
        print("\nGenerating distribution plots...")
        compare_distributions([flux], ['Spectrum'], 
                            output_dir / 'flux_distribution.png', 'Flux')
    
    # Feature extraction
    if args.features:
        print("\nExtracting spectral features...")
        features = extract_spectral_features(tau)
        
        print(f"  Mean void size:       {features['mean_void_size']:.2f} km/s")
        print(f"  Mean line width:      {features['mean_line_width']:.2f} km/s")
        print(f"  Saturation fraction:  {features['saturation_fraction']*100:.2f}%")
        print(f"  Skewness:             {features['flux_skewness']:.4f}")
        print(f"  Kurtosis:             {features['flux_kurtosis']:.4f}")
        
        # Save features to file
        with open(output_dir / 'features.txt', 'w') as f:
            f.write("SPECTRAL FEATURES\n")
            f.write("=" * 50 + "\n\n")
            for key, val in features.items():
                if not isinstance(val, np.ndarray):
                    f.write(f"{key}: {val}\n")
    
    print(f"\nDiagnostics saved to: {output_dir}/")
    print("=" * 70)
    return 0
