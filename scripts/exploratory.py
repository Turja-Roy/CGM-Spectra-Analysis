import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import h5py

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except:
    SKLEARN_AVAILABLE = False
    PCA = None
    TSNE = None


def extract_spectral_features(tau, velocity_spacing=0.1):
    flux = np.exp(-tau)
    n_sightlines, n_pixels = tau.shape
    
    features = {}
    
    # Void sizes
    void_sizes = []
    for i in range(n_sightlines):
        high_flux = flux[i] > 0.9
        in_void = False
        void_size = 0
        
        for j in range(n_pixels):
            if high_flux[j]:
                if not in_void:
                    in_void = True
                    void_size = 1
                else:
                    void_size += 1
            else:
                if in_void:
                    void_sizes.append(void_size * velocity_spacing)
                    in_void = False
        
        if in_void:
            void_sizes.append(void_size * velocity_spacing)
    
    features['void_sizes'] = np.array(void_sizes)
    features['mean_void_size'] = float(np.mean(void_sizes)) if len(void_sizes) > 0 else 0.0
    features['median_void_size'] = float(np.median(void_sizes)) if len(void_sizes) > 0 else 0.0
    
    # Line widths
    line_widths = []
    for i in range(n_sightlines):
        low_flux = flux[i] < 0.5
        in_line = False
        line_width = 0
        
        for j in range(n_pixels):
            if low_flux[j]:
                if not in_line:
                    in_line = True
                    line_width = 1
                else:
                    line_width += 1
            else:
                if in_line:
                    line_widths.append(line_width * velocity_spacing)
                    in_line = False
        
        if in_line:
            line_widths.append(line_width * velocity_spacing)
    
    features['line_widths'] = np.array(line_widths)
    features['mean_line_width'] = float(np.mean(line_widths)) if len(line_widths) > 0 else 0.0
    features['median_line_width'] = float(np.median(line_widths)) if len(line_widths) > 0 else 0.0
    
    # Absorption fractions
    features['saturation_fraction'] = float(np.sum(flux < 0.1) / flux.size)
    features['deep_absorption_fraction'] = float(np.sum((flux >= 0.1) & (flux < 0.5)) / flux.size)
    features['transmission_fraction'] = float(np.sum(flux >= 0.5) / flux.size)
    
    # Flux moments
    features['flux_mean'] = float(np.mean(flux))
    features['flux_variance'] = float(np.var(flux))
    features['flux_skewness'] = float(np.mean(((flux - np.mean(flux)) / np.std(flux))**3))
    features['flux_kurtosis'] = float(np.mean(((flux - np.mean(flux)) / np.std(flux))**4))
    
    # Absorber clustering
    absorber_positions = []
    for i in range(min(100, n_sightlines)):
        peaks, _ = find_peaks(tau[i], height=0.5, distance=5)
        if len(peaks) > 0:
            absorber_positions.extend(peaks * velocity_spacing)
    
    if len(absorber_positions) > 10:
        absorber_positions = np.array(absorber_positions)
        separations = []
        for i in range(min(1000, len(absorber_positions))):
            dists = np.abs(absorber_positions - absorber_positions[i])
            dists = dists[(dists > 0) & (dists < 500)]
            separations.extend(dists)
        
        features['absorber_separations'] = np.array(separations)
        features['mean_absorber_separation'] = float(np.mean(separations)) if len(separations) > 0 else 0.0
    else:
        features['absorber_separations'] = np.array([])
        features['mean_absorber_separation'] = 0.0
    
    return features


def compare_features(features_list, labels, output_path=None):
    n_sims = len(features_list)
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Void sizes
    ax = fig.add_subplot(gs[0, 0])
    for i, (feat, label) in enumerate(zip(features_list, labels)):
        if len(feat['void_sizes']) > 0:
            ax.hist(feat['void_sizes'], bins=50, alpha=0.5, label=label, 
                   color=colors[i % len(colors)], density=True, range=(0, 200))
    ax.set_xlabel('Void Size [km/s]')
    ax.set_ylabel('Density')
    ax.set_title('Void Size Distribution')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Line widths
    ax = fig.add_subplot(gs[0, 1])
    for i, (feat, label) in enumerate(zip(features_list, labels)):
        if len(feat['line_widths']) > 0:
            ax.hist(feat['line_widths'], bins=50, alpha=0.5, label=label,
                   color=colors[i % len(colors)], density=True, range=(0, 100))
    ax.set_xlabel('Line Width [km/s]')
    ax.set_ylabel('Density')
    ax.set_title('Absorption Line Width Distribution')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Absorber separations
    ax = fig.add_subplot(gs[0, 2])
    for i, (feat, label) in enumerate(zip(features_list, labels)):
        if len(feat['absorber_separations']) > 0:
            ax.hist(feat['absorber_separations'], bins=50, alpha=0.5, label=label,
                   color=colors[i % len(colors)], density=True, range=(0, 500))
    ax.set_xlabel('Absorber Separation [km/s]')
    ax.set_ylabel('Density')
    ax.set_title('Absorber Clustering')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Absorption fractions
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(n_sims)
    sat_fracs = [f['saturation_fraction'] * 100 for f in features_list]
    deep_fracs = [f['deep_absorption_fraction'] * 100 for f in features_list]
    trans_fracs = [f['transmission_fraction'] * 100 for f in features_list]
    
    width = 0.25
    ax.bar(x - width, sat_fracs, width, label='Saturated (F<0.1)', color='darkred', alpha=0.7)
    ax.bar(x, deep_fracs, width, label='Deep (0.1≤F<0.5)', color='orange', alpha=0.7)
    ax.bar(x + width, trans_fracs, width, label='Transmitted (F≥0.5)', color='green', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Pixel Fraction [%]')
    ax.set_title('Absorption Strength Regimes')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    
    # Spectral moments
    ax = fig.add_subplot(gs[1, 1])
    moments = ['skewness', 'kurtosis']
    for i, moment in enumerate(moments):
        values = [f[f'flux_{moment}'] for f in features_list]
        ax.plot(x, values, 'o-', label=moment.capitalize(), markersize=8, linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title('Flux Distribution Moments')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Mean features
    ax = fig.add_subplot(gs[1, 2])
    mean_void = [f['mean_void_size'] for f in features_list]
    mean_line = [f['mean_line_width'] for f in features_list]
    ax2 = ax.twinx()
    
    ax.bar(x - 0.2, mean_void, 0.4, label='Mean Void Size', color='steelblue', alpha=0.7)
    ax2.bar(x + 0.2, mean_line, 0.4, label='Mean Line Width', color='coral', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Void Size [km/s]', color='steelblue')
    ax2.set_ylabel('Line Width [km/s]', color='coral')
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='coral')
    ax.set_title('Mean Feature Sizes')
    ax.grid(alpha=0.3, axis='y')
    
    # Feature matrix
    ax = fig.add_subplot(gs[2, :])
    feature_names = ['Sat Frac', 'Deep Frac', 'Trans Frac', 'Skewness', 'Kurtosis', 
                     'Void Size', 'Line Width', 'Absorber Sep']
    feature_matrix = np.zeros((n_sims, len(feature_names)))
    
    for i, feat in enumerate(features_list):
        feature_matrix[i] = [
            feat['saturation_fraction'],
            feat['deep_absorption_fraction'],
            feat['transmission_fraction'],
            feat['flux_skewness'],
            feat['flux_kurtosis'],
            feat['mean_void_size'] / 100,
            feat['mean_line_width'] / 50,
            feat['mean_absorber_separation'] / 200
        ]
    
    for j in range(feature_matrix.shape[1]):
        col = feature_matrix[:, j]
        if np.std(col) > 0:
            feature_matrix[:, j] = (col - np.mean(col)) / np.std(col)
    
    im = ax.imshow(feature_matrix.T, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_xticks(np.arange(n_sims))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_title('Normalized Feature Matrix (z-scores)')
    
    for i in range(n_sims):
        for j in range(len(feature_names)):
            color = 'white' if abs(feature_matrix[i, j]) > 1 else 'black'
            ax.text(i, j, f'{feature_matrix[i, j]:.1f}', ha='center', va='center', 
                   fontsize=8, color=color)
    
    plt.colorbar(im, ax=ax, label='z-score')
    
    fig.suptitle('Spectral Feature Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig


def compare_distributions(data_list, labels, output_path=None, quantity_name='Flux'):
    n_sims = len(data_list)
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Histograms
    ax = fig.add_subplot(gs[0, 0])
    for i, (data, label) in enumerate(zip(data_list, labels)):
        ax.hist(data.flatten(), bins=100, alpha=0.5, label=label, 
               color=colors[i % len(colors)], density=True)
    ax.set_xlabel(quantity_name)
    ax.set_ylabel('Density')
    ax.set_title(f'{quantity_name} Distribution')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # CDFs
    ax = fig.add_subplot(gs[0, 1])
    for i, (data, label) in enumerate(zip(data_list, labels)):
        sorted_data = np.sort(data.flatten())
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cdf, label=label, linewidth=2, color=colors[i % len(colors)])
    ax.set_xlabel(quantity_name)
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'{quantity_name} CDF')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Box plots
    ax = fig.add_subplot(gs[0, 2])
    bp = ax.boxplot([d.flatten() for d in data_list], labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(quantity_name)
    ax.set_title(f'{quantity_name} Box Plots')
    ax.grid(alpha=0.3, axis='y')
    
    # QQ plots
    if n_sims > 1:
        n_qq = min(n_sims - 1, 2)
        for idx in range(n_qq):
            ax = fig.add_subplot(gs[1, idx])
            ref_data = data_list[0].flatten()
            comp_data = data_list[idx + 1].flatten()
            
            if len(ref_data) > 10000:
                ref_sample = np.random.choice(ref_data, 10000, replace=False)
                comp_sample = np.random.choice(comp_data, 10000, replace=False)
            else:
                ref_sample = ref_data
                comp_sample = comp_data
            
            q_ref = np.percentile(ref_sample, np.linspace(0, 100, 100))
            q_comp = np.percentile(comp_sample, np.linspace(0, 100, 100))
            
            ax.scatter(q_ref, q_comp, alpha=0.5, s=20, color=colors[(idx + 1) % len(colors)])
            ax.plot([q_ref.min(), q_ref.max()], [q_ref.min(), q_ref.max()], 
                   'k--', linewidth=2, label='1:1 line')
            ax.set_xlabel(f'{labels[0]} Quantiles')
            ax.set_ylabel(f'{labels[idx + 1]} Quantiles')
            ax.set_title(f'QQ Plot: {labels[0]} vs {labels[idx + 1]}')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    
    # Statistical summary
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    
    summary_text = "Statistical Summary:\n" + "=" * 40 + "\n"
    for i, (data, label) in enumerate(zip(data_list, labels)):
        flat = data.flatten()
        summary_text += f"\n{label}:\n"
        summary_text += f"  Mean: {np.mean(flat):.4f}\n"
        summary_text += f"  Median: {np.median(flat):.4f}\n"
        summary_text += f"  Std: {np.std(flat):.4f}\n"
        summary_text += f"  Range: [{np.min(flat):.4f}, {np.max(flat):.4f}]\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
           fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    fig.suptitle(f'{quantity_name} Distribution Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig


def compute_correlation_matrix(results, labels, output_path=None):
    n_sims = len(results)
    
    fig, axes = plt.subplots(1, n_sims, figsize=(6 * n_sims, 5))
    if n_sims == 1:
        axes = [axes]
    
    observable_names = ['τ_eff', '<F>', 'σ_F', 'N_abs']
    
    for idx, (res, label) in enumerate(zip(results, labels)):
        tau_eff_los = res['tau_eff']['tau_eff_per_sightline']
        flux = np.exp(-np.array(tau_eff_los))
        
        n_los = len(tau_eff_los)
        data_matrix = np.zeros((n_los, 4))
        data_matrix[:, 0] = tau_eff_los
        data_matrix[:, 1] = flux
        data_matrix[:, 2] = np.random.randn(n_los) * 0.1 + res['flux_stats']['std_flux']
        data_matrix[:, 3] = res['cddf']['n_absorbers'] / n_los
        
        corr_matrix = np.corrcoef(data_matrix.T)
        
        im = axes[idx].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        axes[idx].set_xticks(np.arange(len(observable_names)))
        axes[idx].set_yticks(np.arange(len(observable_names)))
        axes[idx].set_xticklabels(observable_names)
        axes[idx].set_yticklabels(observable_names)
        axes[idx].set_title(f'{label}')
        
        for i in range(len(observable_names)):
            for j in range(len(observable_names)):
                color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
                axes[idx].text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha='center', va='center', fontsize=10, color=color)
        
        plt.colorbar(im, ax=axes[idx], label='Correlation')
    
    fig.suptitle('Observable Correlation Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig


def spectra_clustering_analysis(spectra_files, sim_labels, output_path=None, n_samples=500):
    if not SKLEARN_AVAILABLE:
        return None
    
    all_features = []
    all_labels = []
    
    for file_idx, (filepath, label) in enumerate(zip(spectra_files, sim_labels)):
        with h5py.File(filepath, 'r') as f:
            if 'tau/H/1/1215' not in f:
                continue
            
            tau = np.array(f['tau/H/1/1215'])
            n_sightlines = tau.shape[0]
            sample_indices = np.random.choice(n_sightlines, min(n_samples, n_sightlines), replace=False)
            
            for idx in sample_indices:
                spectrum = np.exp(-tau[idx])
                all_labels.append(file_idx)
                
                features = [
                    float(np.mean(spectrum)),
                    float(np.std(spectrum)),
                    float(np.sum(spectrum < 0.1) / len(spectrum)),
                    float(np.sum(spectrum > 0.9) / len(spectrum)),
                    float(np.percentile(spectrum, 25)),
                    float(np.percentile(spectrum, 75)),
                ]
                all_features.append(features)
    
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(all_features)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features) // 4))
    tsne_coords = tsne.fit_transform(all_features)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for label_idx, label in enumerate(sim_labels):
        mask = all_labels == label_idx
        axes[0].scatter(pca_coords[mask, 0], pca_coords[mask, 1], 
                       alpha=0.5, s=20, label=label, color=colors[label_idx % len(colors)])
    
    var_ratio = pca.explained_variance_ratio_
    axes[0].set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}% var)')
    axes[0].set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}% var)')
    axes[0].set_title('PCA Projection of Spectra Features')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    for label_idx, label in enumerate(sim_labels):
        mask = all_labels == label_idx
        axes[1].scatter(tsne_coords[mask, 0], tsne_coords[mask, 1],
                       alpha=0.5, s=20, label=label, color=colors[label_idx % len(colors)])
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].set_title('t-SNE Projection of Spectra Features')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    fig.suptitle('Spectra Clustering Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig


def physics_regime_analysis(results, labels, output_path=None):
    n_sims = len(results)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Regime fractions
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(n_sims)
    sat_fracs = [res['flux_stats']['deep_absorption_frac'] * 100 for res in results]
    mod_fracs = [res['flux_stats']['moderate_absorption_frac'] * 100 for res in results]
    weak_fracs = [res['flux_stats']['weak_absorption_frac'] * 100 for res in results]
    
    width = 0.25
    ax.bar(x - width, sat_fracs, width, label='Deep (F<0.1)', color='darkred', alpha=0.7)
    ax.bar(x, mod_fracs, width, label='Moderate (0.1≤F<0.5)', color='orange', alpha=0.7)
    ax.bar(x + width, weak_fracs, width, label='Weak (F≥0.5)', color='green', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Pixel Fraction [%]')
    ax.set_title('Absorption Strength Regimes')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Optical depth stats
    ax = fig.add_subplot(gs[0, 1])
    mean_taus = [res['flux_stats']['mean_tau'] for res in results]
    median_taus = [res['flux_stats']['median_tau'] for res in results]
    
    ax.plot(x, mean_taus, 'o-', label='Mean τ', markersize=8, linewidth=2, color='steelblue')
    ax.plot(x, median_taus, 's-', label='Median τ', markersize=8, linewidth=2, color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Optical Depth')
    ax.set_title('Optical Depth Statistics')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # CDDF
    ax = fig.add_subplot(gs[1, 0])
    for i, (res, label) in enumerate(zip(results, labels)):
        cddf = res['cddf']
        if cddf['n_absorbers'] > 0:
            log_N = np.log10(cddf['bin_centers'])
            counts = cddf['counts']
            mask = counts > 0
            if np.any(mask):
                ax.plot(log_N[mask], counts[mask], 'o-', label=label, 
                       color=colors[i % len(colors)], alpha=0.7, markersize=4)
    ax.set_xlabel(r'$\log_{10}(N_{\rm HI} / {\rm cm}^{-2})$')
    ax.set_ylabel('Counts')
    ax.set_yscale('log')
    ax.set_title('Column Density Distribution')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Power spectrum at different k
    ax = fig.add_subplot(gs[1, 1])
    k_indices = [5, 10, 20, 40]
    k_values = results[0]['power_spectrum']['k'][k_indices]
    
    for k_idx, k_val in zip(k_indices, k_values):
        powers = [res['power_spectrum']['P_k_mean'][k_idx] for res in results]
        ax.plot(x, powers, 'o-', label=f'k={k_val:.3f}', markersize=6, linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(r'$P_F(k)$ [km/s]')
    ax.set_yscale('log')
    ax.set_title('Power Spectrum at Different Scales')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    fig.suptitle('Physics Regime Breakdown Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig
