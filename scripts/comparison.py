import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from pathlib import Path
from .analysis import (
    compute_flux_statistics,
    compute_effective_optical_depth,
    compute_power_spectrum,
    compute_column_density_distribution,
    compute_line_width_distribution,
    compute_temperature_density_relation,
)
from .plotting import save_plot
from .statistical_tests import comprehensive_comparison, pairwise_comparison_matrix, format_test_results_table
from .exploratory import (
    extract_spectral_features,
    compare_features,
    compare_distributions,
    compute_correlation_matrix,
    spectra_clustering_analysis,
    physics_regime_analysis
)


def load_spectra_results(spectra_file, velocity_spacing=0.1):
    results = {
        'filepath': str(spectra_file),
        'success': False,
    }
    
    try:
        with h5py.File(spectra_file, 'r') as f:
            # Get redshift
            redshift = None
            if 'Header' in f:
                header = f['Header'].attrs
                redshift = header.get('redshift', header.get('Redshift', None))
            results['redshift'] = float(redshift) if redshift is not None else None
            
            # Auto-detect tau data (try HI Lya first)
            tau = None
            tau_path = None
            if 'tau/H/1/1215' in f:
                tau = np.array(f['tau/H/1/1215'])
                tau_path = 'tau/H/1/1215'
            elif 'tau' in f and isinstance(f['tau'], h5py.Dataset):
                tau = np.array(f['tau'])
                tau_path = 'tau'
            
            if tau is None:
                results['error'] = 'No tau data found'
                return results
            
            flux = np.exp(-tau)
            n_sightlines, n_pixels = tau.shape
            
            results['n_sightlines'] = n_sightlines
            results['n_pixels'] = n_pixels
            
            # Compute all analyses
            results['flux_stats'] = compute_flux_statistics(tau)
            results['tau_eff'] = compute_effective_optical_depth(tau)
            results['power_spectrum'] = compute_power_spectrum(flux, velocity_spacing)
            results['cddf'] = compute_column_density_distribution(tau, velocity_spacing)
            
            # Try line width analysis
            try:
                results['line_widths'] = compute_line_width_distribution(tau, velocity_spacing)
            except:
                results['line_widths'] = None
            
            # Try T-ρ analysis
            try:
                if tau_path and '/' in tau_path:
                    parts = tau_path.split('/')
                    temp_elem = parts[1] if len(parts) >= 2 else 'H'
                    temp_ion = parts[2] if len(parts) >= 3 else '1'
                else:
                    temp_elem, temp_ion = 'H', '1'
                
                has_temp = ('temperature' in f and temp_elem in f['temperature'] and 
                           temp_ion in f['temperature'][temp_elem])
                has_dens = ('density_weight_density' in f and temp_elem in f['density_weight_density'] and 
                           temp_ion in f['density_weight_density'][temp_elem])
                
                if has_temp and has_dens:
                    temperature = np.array(f['temperature'][temp_elem][temp_ion])
                    density = np.array(f['density_weight_density'][temp_elem][temp_ion])
                    results['temp_density'] = compute_temperature_density_relation(
                        temperature, density, tau, min_tau=0.1
                    )
                else:
                    results['temp_density'] = None
            except:
                results['temp_density'] = None
            
            results['success'] = True
            
    except Exception as e:
        results['error'] = str(e)
        return results
    
    return results


def compare_simulations(spectra_files, labels=None, output_path=None):
    if labels is None:
        labels = [f"Sim {i}" for i in range(len(spectra_files))]
    
    # Load all results
    print(f"Loading {len(spectra_files)} simulation results...")
    all_results = []
    for i, (fpath, label) in enumerate(zip(spectra_files, labels)):
        print(f"  [{i+1}/{len(spectra_files)}] {label}: {fpath}")
        results = load_spectra_results(fpath)
        if results['success']:
            results['label'] = label
            all_results.append(results)
            print(f"      OK z={results['redshift']:.3f}, N={results['n_sightlines']}")
        else:
            print(f"      Failed: {results.get('error', 'unknown error')}")
    
    if len(all_results) == 0:
        print("Error: No valid results to compare")
        return None
    
    # Create comparison figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Panel 1: Flux power spectrum
    ax = fig.add_subplot(gs[0, :])
    for i, res in enumerate(all_results):
        ps = res['power_spectrum']
        k = ps['k']
        P_k = ps['P_k_mean']
        P_k_err = ps['P_k_err']
        
        mask = k > 0
        k = k[mask]
        P_k = P_k[mask]
        P_k_err = P_k_err[mask]
        
        color = colors[i % len(colors)]
        ax.loglog(k, P_k, 'o-', color=color, linewidth=2, markersize=3,
                 label=f"{res['label']} (z={res['redshift']:.2f})", alpha=0.8)
        ax.fill_between(k, P_k - P_k_err, P_k + P_k_err, alpha=0.2, color=color)
    
    ax.set_xlabel(r'Wavenumber $k$ [s/km]', fontsize=13)
    ax.set_ylabel(r'Power Spectrum $P_F(k)$ [km/s]', fontsize=13)
    ax.set_title('Flux Power Spectrum Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10, loc='best')
    
    # Panel 2: Effective optical depth
    ax = fig.add_subplot(gs[1, 0])
    tau_effs = [res['tau_eff']['tau_eff'] for res in all_results]
    tau_errs = [res['tau_eff']['tau_eff_err'] for res in all_results]
    x_pos = np.arange(len(all_results))
    
    bars = ax.bar(x_pos, tau_effs, yerr=tau_errs, capsize=5,
                  color=[colors[i % len(colors)] for i in range(len(all_results))],
                  alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([res['label'] for res in all_results], rotation=45, ha='right')
    ax.set_ylabel(r'Effective Optical Depth $\tau_{\rm eff}$', fontsize=12)
    ax.set_title('Effective Optical Depth', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Mean flux
    ax = fig.add_subplot(gs[1, 1])
    mean_fluxes = [res['flux_stats']['mean_flux'] for res in all_results]
    
    bars = ax.bar(x_pos, mean_fluxes,
                  color=[colors[i % len(colors)] for i in range(len(all_results))],
                  alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([res['label'] for res in all_results], rotation=45, ha='right')
    ax.set_ylabel(r'Mean Transmitted Flux $\langle F \rangle$', fontsize=12)
    ax.set_title('Mean Flux', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Column density distribution
    ax = fig.add_subplot(gs[2, 0])
    for i, res in enumerate(all_results):
        cddf = res['cddf']
        if cddf['n_absorbers'] > 0 and len(cddf['counts']) > 0:
            color = colors[i % len(colors)]
            # Normalize counts to get f(N_HI) - number density per bin
            log_N = cddf['bin_centers']
            # f(N) in units of dN/dlog(N)/dz (absorbers per unit log column density per unit redshift)
            dN_dz = cddf['n_absorbers'] / res['n_sightlines'] / 0.1  # Rough dz estimate
            delta_log_N = log_N[1] - log_N[0] if len(log_N) > 1 else 1.0
            f_N = cddf['counts'] / (cddf['n_absorbers'] if cddf['n_absorbers'] > 0 else 1) / delta_log_N
            
            # Only plot non-zero bins
            mask = f_N > 0
            if np.any(mask):
                ax.scatter(log_N[mask], f_N[mask],
                          s=30, alpha=0.6, color=color, label=res['label'])
                
                # Plot fit if available
                if not np.isnan(cddf['beta_fit']):
                    N_fit = np.logspace(12, 16, 100)
                    # Simple power law for visualization
                    f_fit = f_N[mask].max() * (N_fit / 10**log_N[mask][np.argmax(f_N[mask])]) ** cddf['beta_fit']
                    ax.plot(np.log10(N_fit), f_fit, '--', color=color, alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel(r'$\log_{10}(N_{\rm HI} / {\rm cm}^{-2})$', fontsize=12)
    ax.set_ylabel(r'$f(N_{\rm HI})$', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Column Density Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Panel 5: Temperature-density parameters (if available)
    ax = fig.add_subplot(gs[2, 1])
    T0_values = []
    gamma_values = []
    valid_labels = []
    
    for i, res in enumerate(all_results):
        if res['temp_density'] is not None:
            td = res['temp_density']
            if np.isfinite(td['T0']) and np.isfinite(td['gamma']):
                T0_values.append(td['T0'])
                gamma_values.append(td['gamma'])
                valid_labels.append(res['label'])
    
    if len(T0_values) > 0:
        x_pos_td = np.arange(len(T0_values))
        
        ax2 = ax.twinx()
        
        bars1 = ax.bar(x_pos_td - 0.2, T0_values, width=0.4, 
                      color='steelblue', alpha=0.7, label=r'$T_0$ [K]')
        bars2 = ax2.bar(x_pos_td + 0.2, gamma_values, width=0.4,
                       color='coral', alpha=0.7, label=r'$\gamma$')
        
        ax.set_xticks(x_pos_td)
        ax.set_xticklabels(valid_labels, rotation=45, ha='right')
        ax.set_ylabel(r'$T_0$ at Mean Density [K]', fontsize=12, color='steelblue')
        ax2.set_ylabel(r'Polytropic Index $\gamma$', fontsize=12, color='coral')
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='coral')
        ax.set_title('IGM Equation of State', fontsize=13, fontweight='bold')
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
    else:
        ax.text(0.5, 0.5, 'No temperature-density\ndata available',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Overall title
    fig.suptitle('Simulation Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    if output_path:
        save_plot(fig, output_path)
        print(f"\nComparison plot saved: {output_path}")
    
    plt.close()
    
    # Return summary
    comparison = {
        'n_simulations': len(all_results),
        'labels': [res['label'] for res in all_results],
        'results': all_results
    }
    
    return comparison


def track_redshift_evolution(spectra_files, labels=None, output_path=None):
    if labels is None:
        labels = [f"Snap {i}" for i in range(len(spectra_files))]
    
    # Load all results
    print(f"Loading {len(spectra_files)} snapshots for evolution tracking...")
    all_results = []
    for i, (fpath, label) in enumerate(zip(spectra_files, labels)):
        print(f"  [{i+1}/{len(spectra_files)}] {label}: {fpath}")
        results = load_spectra_results(fpath)
        if results['success'] and results['redshift'] is not None:
            results['label'] = label
            all_results.append(results)
            print(f"      OK z={results['redshift']:.3f}")
        else:
            print(f"      Failed or no redshift")
    
    if len(all_results) == 0:
        print("Error: No valid results for evolution tracking")
        return None
    
    # Sort by redshift
    all_results.sort(key=lambda x: x['redshift'])
    
    # Extract evolution data
    redshifts = np.array([res['redshift'] for res in all_results])
    tau_effs = np.array([res['tau_eff']['tau_eff'] for res in all_results])
    tau_errs = np.array([res['tau_eff']['tau_eff_err'] for res in all_results])
    mean_fluxes = np.array([res['flux_stats']['mean_flux'] for res in all_results])
    n_absorbers = np.array([res['cddf']['n_absorbers'] for res in all_results])
    
    # T-ρ parameters (if available)
    T0_values = []
    gamma_values = []
    z_with_T = []
    
    for res in all_results:
        if res['temp_density'] is not None:
            td = res['temp_density']
            if np.isfinite(td['T0']) and np.isfinite(td['gamma']):
                z_with_T.append(res['redshift'])
                T0_values.append(td['T0'])
                gamma_values.append(td['gamma'])
    
    z_with_T = np.array(z_with_T)
    T0_values = np.array(T0_values)
    gamma_values = np.array(gamma_values)
    
    # Create evolution figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Effective optical depth evolution
    ax = axes[0, 0]
    ax.errorbar(redshifts, tau_effs, yerr=tau_errs, fmt='o-', 
               color='steelblue', linewidth=2, markersize=6, capsize=4, alpha=0.8)
    ax.set_xlabel('Redshift $z$', fontsize=13)
    ax.set_ylabel(r'Effective Optical Depth $\tau_{\rm eff}$', fontsize=13)
    ax.set_title(r'$\tau_{\rm eff}(z)$ Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Higher z on left
    
    # Panel 2: Mean flux evolution
    ax = axes[0, 1]
    ax.plot(redshifts, mean_fluxes, 'o-', color='coral', 
           linewidth=2, markersize=6, alpha=0.8)
    ax.set_xlabel('Redshift $z$', fontsize=13)
    ax.set_ylabel(r'Mean Transmitted Flux $\langle F \rangle$', fontsize=13)
    ax.set_title(r'$\langle F \rangle(z)$ Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # Panel 3: Number of absorbers
    ax = axes[1, 0]
    ax.plot(redshifts, n_absorbers, 'o-', color='green',
           linewidth=2, markersize=6, alpha=0.8)
    ax.set_xlabel('Redshift $z$', fontsize=13)
    ax.set_ylabel('Number of Absorbers', fontsize=13)
    ax.set_title('Absorber Count Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # Panel 4: Temperature-density evolution
    ax = axes[1, 1]
    if len(z_with_T) > 0:
        ax2 = ax.twinx()
        
        line1 = ax.plot(z_with_T, T0_values, 'o-', color='steelblue',
                       linewidth=2, markersize=6, alpha=0.8, label=r'$T_0$')
        line2 = ax2.plot(z_with_T, gamma_values, 's-', color='coral',
                        linewidth=2, markersize=6, alpha=0.8, label=r'$\gamma$')
        
        ax.set_xlabel('Redshift $z$', fontsize=13)
        ax.set_ylabel(r'$T_0$ at Mean Density [K]', fontsize=13, color='steelblue')
        ax2.set_ylabel(r'Polytropic Index $\gamma$', fontsize=13, color='coral')
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='coral')
        ax.set_title('IGM Equation of State Evolution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        # Combined legend
        lines = line1 + line2
        labels_leg = [l.get_label() for l in lines]
        ax.legend(lines, labels_leg, fontsize=11, loc='best')
    else:
        ax.text(0.5, 0.5, 'No temperature-density\ndata available',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Overall title
    z_min, z_max = redshifts.min(), redshifts.max()
    fig.suptitle(f'Redshift Evolution (z = {z_min:.2f} - {z_max:.2f})', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        save_plot(fig, output_path)
        print(f"\nEvolution plot saved: {output_path}")
    
    plt.close()
    
    # Return summary
    evolution = {
        'n_snapshots': len(all_results),
        'redshift_range': (float(redshifts.min()), float(redshifts.max())),
        'redshifts': redshifts.tolist(),
        'tau_eff': tau_effs.tolist(),
        'mean_flux': mean_fluxes.tolist(),
        'n_absorbers': n_absorbers.tolist(),
        'results': all_results
    }
    
    if len(z_with_T) > 0:
        evolution['T0'] = T0_values.tolist()
        evolution['gamma'] = gamma_values.tolist()
        evolution['z_with_T'] = z_with_T.tolist()
    
    return evolution


def compare_simulations_comprehensive(spectra_files, labels=None, output_dir=None, mode='detailed'):
    """
    Comprehensive comparison with multiple analysis modes.
    mode: 'quick' (basic plots), 'detailed' (enhanced plots), 'full' (all analyses)
    """
    if labels is None:
        labels = [f"Sim {i}" for i in range(len(spectra_files))]
    
    print(f"[COMPREHENSIVE COMPARISON - Mode: {mode}]")
    print(f"Loading {len(spectra_files)} simulations...")
    
    all_results = []
    all_tau_data = []
    
    for i, (fpath, label) in enumerate(zip(spectra_files, labels)):
        results = load_spectra_results(fpath)
        if results['success']:
            results['label'] = label
            all_results.append(results)
            
            # Load tau for deep analysis
            if mode in ['detailed', 'full']:
                with h5py.File(fpath, 'r') as f:
                    if 'tau/H/1/1215' in f:
                        all_tau_data.append(np.array(f['tau/H/1/1215']))
        else:
            print(f"  Skipping {label}: {results.get('error', 'unknown error')}")
    
    if len(all_results) == 0:
        return None
    
    if output_dir is None:
        output_dir = Path('plots/comparisons')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Basic comparison plot (always generated)
    compare_simulations(spectra_files, labels, output_path=output_dir / 'comparison_basic.png')
    
    if mode in ['detailed', 'full']:
        print("\n[ENHANCED COMPARISON PLOTS]")
        
        # Enhanced comparison with box plots and sample spectra
        _create_enhanced_comparison_plot(all_results, labels, all_tau_data, 
                                        output_dir / 'comparison_enhanced.png')
        
        # Power spectrum ratios
        _create_power_spectrum_ratio_plot(all_results, labels,
                                         output_dir / 'power_spectrum_ratios.png')
        
        # Distribution comparison for flux
        flux_data = [np.exp(-tau) for tau in all_tau_data]
        compare_distributions(flux_data, labels, 
                            output_dir / 'flux_distributions.png', 'Flux')
        
        # Statistical tests
        print("\n[STATISTICAL TESTS]")
        test_results = []
        for i in range(len(all_results)):
            for j in range(i + 1, len(all_results)):
                from scripts.statistical_tests import comprehensive_comparison as comp_test
                result = comp_test(flux_data[i], flux_data[j], 
                                 labels[i], labels[j])
                test_results.append(result)
        
        if test_results:
            from scripts.statistical_tests import format_test_results_table
            test_summary = format_test_results_table(test_results, correction='bonferroni')
            with open(output_dir / 'statistical_tests.txt', 'w') as f:
                f.write(test_summary)
            print(test_summary[:500] + "...\n(Full results in statistical_tests.txt)")
        
        # Correlation matrices
        compute_correlation_matrix(all_results, labels,
                                  output_dir / 'correlation_matrices.png')
    
    if mode == 'full':
        print("\n[FULL EXPLORATORY ANALYSIS]")
        
        # Feature extraction and comparison
        print("Extracting spectral features...")
        features_list = [extract_spectral_features(tau) for tau in all_tau_data]
        compare_features(features_list, labels,
                        output_dir / 'feature_comparison.png')
        
        # Physics regime analysis
        physics_regime_analysis(all_results, labels,
                              output_dir / 'physics_regimes.png')
        
        # Clustering analysis
        print("Running PCA/t-SNE clustering...")
        spectra_clustering_analysis(spectra_files, labels,
                                   output_dir / 'spectra_clustering.png',
                                   n_samples=500)
        
        # Pairwise comparison matrix
        print("Computing pairwise comparison matrix...")
        from scripts.statistical_tests import pairwise_comparison_matrix
        matrix_result = pairwise_comparison_matrix(flux_data, labels, metric='ks')
        _plot_pairwise_matrix(matrix_result, output_dir / 'pairwise_ks_matrix.png')
    
    print(f"\n[COMPLETE] All plots saved to {output_dir}/")
    
    return {
        'n_simulations': len(all_results),
        'labels': labels,
        'results': all_results,
        'output_dir': str(output_dir)
    }


def _create_enhanced_comparison_plot(all_results, labels, all_tau_data, output_path):
    """Enhanced comparison with box plots, ratios, sample spectra, significance."""
    n_sims = len(all_results)
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.4)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Panel 1: Power spectrum (same as before)
    ax = fig.add_subplot(gs[0, :])
    for i, res in enumerate(all_results):
        ps = res['power_spectrum']
        k, P_k = ps['k'], ps['P_k_mean']
        mask = k > 0
        ax.loglog(k[mask], P_k[mask], 'o-', color=colors[i % len(colors)], 
                 linewidth=2, markersize=3, label=f"{res['label']}", alpha=0.8)
    ax.set_xlabel(r'$k$ [s/km]', fontsize=12)
    ax.set_ylabel(r'$P_F(k)$ [km/s]', fontsize=12)
    ax.set_title('Flux Power Spectrum', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    
    # Panel 2: τ_eff box plot
    ax = fig.add_subplot(gs[1, 0])
    tau_eff_data = [res['tau_eff']['tau_eff_per_sightline'] for res in all_results]
    bp = ax.boxplot(tau_eff_data, labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(r'$\tau_{\rm eff}$', fontsize=12)
    ax.set_title('Optical Depth Distribution', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Panel 3: Flux box plot
    ax = fig.add_subplot(gs[1, 1])
    flux_data = [np.exp(-tau) for tau in all_tau_data]
    bp = ax.boxplot([f.flatten()[:10000] for f in flux_data], labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Flux', fontsize=12)
    ax.set_title('Flux Distribution', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Panel 4: Sample spectra (3 random per sim)
    ax = fig.add_subplot(gs[1, 2])
    for i, (tau, label) in enumerate(zip(all_tau_data, labels)):
        n_sightlines = tau.shape[0]
        n_pixels = tau.shape[1]
        sample_idx = np.random.choice(n_sightlines, min(3, n_sightlines), replace=False)
        
        for idx in sample_idx:
            flux = np.exp(-tau[idx])
            v = np.arange(len(flux)) * 0.1
            ax.plot(v[:2000], flux[:2000], alpha=0.4, linewidth=0.8, 
                   color=colors[i % len(colors)])
    
    ax.set_xlabel('Velocity [km/s]', fontsize=10)
    ax.set_ylabel('Flux', fontsize=10)
    ax.set_title('Sample Spectra', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 200)
    ax.grid(alpha=0.3)
    
    # Panel 5: Power spectrum ratio (ref = first sim)
    ax = fig.add_subplot(gs[2, 0])
    ref_ps = all_results[0]['power_spectrum']
    for i in range(1, n_sims):
        ps = all_results[i]['power_spectrum']
        k = ps['k']
        mask = (k > 0) & (ref_ps['P_k_mean'] > 0)
        ratio = ps['P_k_mean'][mask] / ref_ps['P_k_mean'][mask]
        ax.semilogx(k[mask], ratio, 'o-', label=f"{labels[i]}/{labels[0]}", 
                   markersize=3, linewidth=1.5, color=colors[i % len(colors)])
    ax.axhline(1, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'$k$ [s/km]', fontsize=11)
    ax.set_ylabel('Power Ratio', fontsize=11)
    ax.set_title('Power Spectrum Ratios', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Panel 6: CDDF comparison
    ax = fig.add_subplot(gs[2, 1])
    for i, res in enumerate(all_results):
        cddf = res['cddf']
        if cddf['n_absorbers'] > 0:
            log_N = np.log10(cddf['bin_centers'])
            counts = cddf['counts']
            mask = counts > 0
            if np.any(mask):
                ax.plot(log_N[mask], counts[mask], 'o-', label=labels[i],
                       color=colors[i % len(colors)], alpha=0.7, markersize=4)
    ax.set_xlabel(r'$\log_{10}(N_{\rm HI})$', fontsize=11)
    ax.set_ylabel('Counts', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('Column Density Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Panel 7: Absorption fractions
    ax = fig.add_subplot(gs[2, 2])
    x = np.arange(n_sims)
    sat_fracs = [res['flux_stats']['deep_absorption_frac'] * 100 for res in all_results]
    mod_fracs = [res['flux_stats']['moderate_absorption_frac'] * 100 for res in all_results]
    weak_fracs = [res['flux_stats']['weak_absorption_frac'] * 100 for res in all_results]
    
    width = 0.25
    ax.bar(x - width, sat_fracs, width, label='Deep', color='darkred', alpha=0.7)
    ax.bar(x, mod_fracs, width, label='Moderate', color='orange', alpha=0.7)
    ax.bar(x + width, weak_fracs, width, label='Weak', color='green', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Pixel Fraction [%]', fontsize=11)
    ax.set_title('Absorption Regimes', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    
    # Panel 8: Mean statistics comparison
    ax = fig.add_subplot(gs[3, 0])
    mean_tau = [res['flux_stats']['mean_tau'] for res in all_results]
    mean_flux = [res['flux_stats']['mean_flux'] for res in all_results]
    ax2 = ax.twinx()
    
    ax.bar(x - 0.2, mean_tau, 0.4, label=r'$\langle\tau\rangle$', color='steelblue', alpha=0.7)
    ax2.bar(x + 0.2, mean_flux, 0.4, label=r'$\langle F\rangle$', color='coral', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(r'$\langle\tau\rangle$', color='steelblue', fontsize=11)
    ax2.set_ylabel(r'$\langle F\rangle$', color='coral', fontsize=11)
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='coral')
    ax.set_title('Mean Statistics', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Panel 9: Variance comparison
    ax = fig.add_subplot(gs[3, 1])
    std_tau = [res['tau_eff']['tau_eff_std'] for res in all_results]
    std_flux = [res['flux_stats']['std_flux'] for res in all_results]
    ax2 = ax.twinx()
    
    ax.bar(x - 0.2, std_tau, 0.4, label=r'$\sigma_\tau$', color='steelblue', alpha=0.7)
    ax2.bar(x + 0.2, std_flux, 0.4, label=r'$\sigma_F$', color='coral', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(r'$\sigma_\tau$', color='steelblue', fontsize=11)
    ax2.set_ylabel(r'$\sigma_F$', color='coral', fontsize=11)
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='coral')
    ax.set_title('Variability', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # Panel 10: Summary table
    ax = fig.add_subplot(gs[3, 2])
    ax.axis('off')
    
    summary_text = "Key Statistics:\n" + "=" * 35 + "\n"
    for i, (res, label) in enumerate(zip(all_results, labels)):
        summary_text += f"\n{label}:\n"
        summary_text += f"  τ_eff: {res['tau_eff']['tau_eff']:.4f}\n"
        summary_text += f"  <F>: {res['flux_stats']['mean_flux']:.4f}\n"
        summary_text += f"  N_abs: {res['cddf']['n_absorbers']}\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
           fontsize=8, verticalalignment='top', fontfamily='monospace')
    
    fig.suptitle('Enhanced Simulation Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _create_power_spectrum_ratio_plot(all_results, labels, output_path):
    """Create detailed power spectrum ratio plot."""
    n_sims = len(all_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Absolute power spectra
    ax = axes[0]
    for i, res in enumerate(all_results):
        ps = res['power_spectrum']
        k, P_k, P_k_err = ps['k'], ps['P_k_mean'], ps['P_k_err']
        mask = k > 0
        ax.loglog(k[mask], P_k[mask], 'o-', color=colors[i % len(colors)],
                 linewidth=2, markersize=3, label=labels[i], alpha=0.8)
        ax.fill_between(k[mask], P_k[mask] - P_k_err[mask], 
                       P_k[mask] + P_k_err[mask], 
                       alpha=0.15, color=colors[i % len(colors)])
    ax.set_xlabel(r'$k$ [s/km]', fontsize=12)
    ax.set_ylabel(r'$P_F(k)$ [km/s]', fontsize=12)
    ax.set_title('Power Spectra', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Ratios relative to first
    ax = axes[1]
    ref_ps = all_results[0]['power_spectrum']
    for i in range(1, n_sims):
        ps = all_results[i]['power_spectrum']
        k = ps['k']
        mask = (k > 0) & (ref_ps['P_k_mean'] > 0)
        ratio = ps['P_k_mean'][mask] / ref_ps['P_k_mean'][mask]
        ax.semilogx(k[mask], ratio, 'o-', label=f"{labels[i]}/{labels[0]}",
                   markersize=4, linewidth=2, color=colors[i % len(colors)])
    ax.axhline(1, color='k', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xlabel(r'$k$ [s/km]', fontsize=12)
    ax.set_ylabel(r'$P_F(k) / P_F^{\rm ref}(k)$', fontsize=12)
    ax.set_title(f'Power Ratios (ref: {labels[0]})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_pairwise_matrix(matrix_result, output_path):
    """Plot pairwise comparison matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    matrix = matrix_result['matrix']
    pvalue_matrix = matrix_result['pvalue_matrix']
    labels = matrix_result['labels']
    
    # Statistic matrix
    im1 = axes[0].imshow(matrix, cmap='viridis', aspect='auto')
    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_yticks(np.arange(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].set_yticklabels(labels)
    axes[0].set_title(f'{matrix_result["metric"].upper()} Test Statistic')
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = axes[0].text(j, i, f'{matrix[i, j]:.3f}',
                              ha='center', va='center', fontsize=9)
    
    plt.colorbar(im1, ax=axes[0])
    
    # P-value matrix
    im2 = axes[1].imshow(np.log10(pvalue_matrix + 1e-100), cmap='RdYlGn', aspect='auto', vmin=-10, vmax=0)
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_yticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_yticklabels(labels)
    axes[1].set_title('p-values (log10 scale)')
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                sig = '***' if pvalue_matrix[i, j] < 0.001 else ('**' if pvalue_matrix[i, j] < 0.01 else ('*' if pvalue_matrix[i, j] < 0.05 else ''))
                text = axes[1].text(j, i, sig, ha='center', va='center', 
                                  fontsize=14, fontweight='bold', color='red')
    
    plt.colorbar(im2, ax=axes[1], label='log10(p-value)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

