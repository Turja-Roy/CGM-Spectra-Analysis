import numpy as np
import os
from pathlib import Path
from astropy.cosmology import FlatLambdaCDM


#######################
# COSMOLOGY UTILITIES #
#######################

def get_cosmology(omega_m=0.3089, omega_lambda=0.6911, h=0.6774):
    return FlatLambdaCDM(H0=100*h, Om0=omega_m)


def compute_absorption_path_length(redshift, box_size_ckpc_h, hubble=0.6774, 
                                   omega_m=0.3089, use_cosmology=True):
    """
    Notes:

    For a periodic box, the absorption path length is simply:
        dX = box_size / h  [comoving Mpc]
    
    The proper distance would be dX / (1+z), but for CDDF we use comoving.
    """
    if use_cosmology:
        # Convert ckpc/h to Mpc (comoving)
        dX = box_size_ckpc_h / hubble / 1000.0  # Mpc
    else:
        # Simple conversion
        dX = box_size_ckpc_h / hubble / 1000.0  # Mpc
    
    return dX


#############################
# DATA PROCESSING FUNCTIONS #
#############################

def compute_flux_statistics(tau):
    flux = np.exp(-tau)

    stats = {
        'mean_flux': float(np.mean(flux)),
        'median_flux': float(np.median(flux)),
        'std_flux': float(np.std(flux)),
        'min_flux': float(np.min(flux)),
        'max_flux': float(np.max(flux)),
        'mean_tau': float(np.mean(tau)),
        'median_tau': float(np.median(tau)),
        'effective_tau': float(-np.log(np.mean(flux))),
    }

    # Absorption statistics
    total_pixels = flux.size
    stats['deep_absorption_frac'] = float((flux < 0.1).sum() / total_pixels)
    stats['moderate_absorption_frac'] = float(
        ((flux >= 0.1) & (flux < 0.5)).sum() / total_pixels)
    stats['weak_absorption_frac'] = float((flux >= 0.5).sum() / total_pixels)

    return stats


def compute_power_spectrum(flux, velocity_spacing, chunk_size=1000):
    n_sightlines, n_pixels = flux.shape

    # Normalize flux to get flux contrast
    mean_flux = np.mean(flux)

    # Wavenumber array (s/km units)
    k = np.fft.rfftfreq(n_pixels, d=velocity_spacing)
    n_k = len(k)

    # Initialize accumulators for mean and variance computation
    # Using Welford's online algorithm to avoid storing all power spectra
    power_sum = np.zeros(n_k)
    power_sum_sq = np.zeros(n_k)

    # Process in chunks to reduce memory usage
    n_chunks = int(np.ceil(n_sightlines / chunk_size))

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, n_sightlines)

        # Get flux chunk and compute contrast
        flux_chunk = flux[start_idx:end_idx]
        delta_F_chunk = flux_chunk / mean_flux - 1.0

        # Compute power for this chunk
        for i in range(delta_F_chunk.shape[0]):
            # FFT of flux contrast
            flux_fft = np.fft.rfft(delta_F_chunk[i])
            # Power spectrum (dimensionless, normalized by pixel count)
            power = np.abs(flux_fft)**2 / n_pixels

            # Update running sums
            power_sum += power
            power_sum_sq += power**2

    # Compute mean and standard deviation
    P_k_mean = (power_sum / n_sightlines) * velocity_spacing

    # Variance: Var(X) = E[X^2] - E[X]^2
    mean_power = power_sum / n_sightlines
    mean_power_sq = power_sum_sq / n_sightlines
    variance = mean_power_sq - mean_power**2
    P_k_std = np.sqrt(np.maximum(variance, 0)) * \
        velocity_spacing  # Ensure non-negative
    P_k_err = P_k_std / np.sqrt(n_sightlines)

    # Number of independent modes (useful for error estimation)
    n_modes = np.ones_like(k) * n_sightlines

    return {
        'k': k,
        'P_k_mean': P_k_mean,
        'P_k_std': P_k_std,
        'P_k_err': P_k_err,
        'mean_flux': mean_flux,
        'n_modes': n_modes,
        'n_sightlines': n_sightlines,
        'velocity_spacing': velocity_spacing
    }


def ensure_directory(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_snapshot_number(filepath):
    """Extract snapshot number from filepath."""
    basename = os.path.basename(filepath)
    if 'snap_' in basename:
        num_str = basename.split('snap_')[1].split('.')[0]
        return int(num_str)
    return None


def compute_column_density_distribution(tau, velocity_spacing, threshold=0.5, colden=None,
                                       redshift=None, box_size_ckpc_h=None, hubble=0.6774,
                                       omega_m=0.3089):
    """
    Compute column density distribution f(N_HI) with proper cosmological normalization.

    Notes:

    The differential CDDF is defined as:
        f(N) = dN/dlog10(N_HI) per unit absorption path length
        
    Units: [Mpc^-1] in comoving coordinates
    
    Normalization:
        f(N) = counts / (n_sightlines * delta_log_N * dX)
        
    where dX is the comoving absorption path length (box_size in Mpc).
    """
    # Corrected constant: 8.51e11 cm^-2 / (km/s), empirically calibrated
    TAU_TO_COLDEN_CONSTANT = 8.51e11

    column_densities = []

    for i in range(tau.shape[0]):
        tau_line = tau[i, :]
        colden_line = colden[i, :] if colden is not None else None

        # Find absorption features (contiguous pixels above threshold)
        absorbing = tau_line > threshold

        # Label connected regions
        in_feature = False
        feature_start = 0

        for j in range(len(tau_line)):
            if absorbing[j] and not in_feature:
                # Start of new feature
                in_feature = True
                feature_start = j
            elif not absorbing[j] and in_feature:
                # End of feature
                in_feature = False
                
                # Compute column density
                if colden_line is not None:
                    N_HI = np.sum(colden_line[feature_start:j])
                else:
                    feature_tau = tau_line[feature_start:j]
                    N_HI = TAU_TO_COLDEN_CONSTANT * np.sum(feature_tau) * velocity_spacing

                if N_HI > 1e12:  # Only count above sensitivity threshold
                    column_densities.append(N_HI)

        # Handle case where feature extends to edge
        if in_feature:
            if colden_line is not None:
                N_HI = np.sum(colden_line[feature_start:])
            else:
                feature_tau = tau_line[feature_start:]
                N_HI = TAU_TO_COLDEN_CONSTANT * np.sum(feature_tau) * velocity_spacing
            if N_HI > 1e12:
                column_densities.append(N_HI)

    column_densities = np.array(column_densities)
    n_sightlines = tau.shape[0]
    
    # Compute absorption path length for normalization
    if redshift is not None and box_size_ckpc_h is not None:
        dX = compute_absorption_path_length(redshift, box_size_ckpc_h, hubble, omega_m)
    else:
        # Fallback: no normalization (for backward compatibility)
        dX = 1.0
        if redshift is not None or box_size_ckpc_h is not None:
            print("Warning: Missing redshift or box_size for CDDF normalization. "
                  "Setting dX=1.0 (unnormalized)")

    # Create histogram in log space
    if len(column_densities) > 0:
        log_N_min = 12.0  # log10(N_HI)
        log_N_max = 22.0
        n_bins = 50

        bins = np.logspace(log_N_min, log_N_max, n_bins)
        counts, bin_edges = np.histogram(column_densities, bins=bins)

        # Compute log-space bin properties for proper normalization
        log_bin_edges = np.log10(bin_edges)
        delta_log_N = np.diff(log_bin_edges)  # Constant for logspace bins
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        log_bin_centers = np.log10(bin_centers)
        
        # Compute properly normalized f(N) in units of [Mpc^-1]
        # f(N) = dN/dlog10(N) per unit comoving path length
        f_N = counts / (n_sightlines * delta_log_N * dX)

        # Fit power law in range 13 < log(N) < 17 (typical Lyman-alpha forest)
        fit_mask = (log_bin_centers > 13.0) & (log_bin_centers < 17.0)

        if np.sum(fit_mask) > 5 and np.sum(counts[fit_mask]) > 0:
            # Fit f(N) = A * N^-beta using properly normalized f(N)
            f_N_fit = f_N[fit_mask]
            log_f_fit = np.log10(f_N_fit + 1e-10)  # Avoid log(0)
            log_N_fit = log_bin_centers[fit_mask]

            # Linear fit in log-log space
            valid = np.isfinite(log_f_fit) & (f_N_fit > 0)
            if np.sum(valid) > 2:
                coeffs = np.polyfit(log_N_fit[valid], log_f_fit[valid], 1)
                beta_fit = -coeffs[0]  # Negative slope
            else:
                beta_fit = np.nan
        else:
            beta_fit = np.nan
    else:
        bins = np.logspace(12, 22, 50)
        counts = np.zeros(len(bins) - 1)
        bin_edges = bins
        log_bin_edges = np.log10(bin_edges)
        delta_log_N = np.diff(log_bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        log_bin_centers = np.log10(bin_centers)
        f_N = np.zeros(len(counts))
        beta_fit = np.nan

    return {
        'N_HI': column_densities,
        'counts': counts,
        'bins': bin_edges,
        'bin_centers': bin_centers,
        'log_bin_edges': log_bin_edges,
        'log10_N_HI': log_bin_centers,
        'delta_log_N': delta_log_N,
        'beta_fit': beta_fit,
        'n_absorbers': len(column_densities),
        'n_sightlines': n_sightlines,
        'f_N': f_N,
        'f_N_HI': f_N,  # Backward compatibility
        'dX': dX,
        'redshift': redshift if redshift is not None else np.nan,
    }


def compute_effective_optical_depth(tau):
    flux = np.exp(-tau)

    # Global tau_eff
    mean_flux = np.mean(flux)
    tau_eff = -np.log(mean_flux)

    # Per-sightline tau_eff
    mean_flux_per_los = np.mean(flux, axis=1)
    tau_eff_per_los = -np.log(mean_flux_per_los)

    return {
        'tau_eff': float(tau_eff),
        'mean_flux': float(mean_flux),
        'tau_eff_per_sightline': tau_eff_per_los,
        'tau_eff_std': float(np.std(tau_eff_per_los)),
        'tau_eff_err': float(np.std(tau_eff_per_los) / np.sqrt(len(tau_eff_per_los)))
    }


def compute_line_width_distribution(tau, velocity_spacing, threshold=0.5, colden=None):
    """Compute line width (Doppler b-parameter) distribution."""
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks

    TAU_TO_COLDEN_CONSTANT = 8.51e11  # cm^-2 / (km/s)

    column_densities = []
    b_parameters = []

    def voigt_approx(v, tau_0, b, v_center):
        a = 4.7e-4  # Damping parameter for Lyman-alpha
        u = (v - v_center) / b
        tau = tau_0 * np.exp(-u**2)
        return tau

    for i in range(tau.shape[0]):
        tau_line = tau[i, :]
        colden_line = colden[i, :] if colden is not None else None

        # Find peaks in optical depth (absorption features)
        peaks, properties = find_peaks(tau_line, height=threshold, distance=5)

        for peak_idx in peaks:
            # Define feature extent (where tau drops to threshold)
            left = peak_idx
            while left > 0 and tau_line[left] > threshold * 0.3:
                left -= 1

            right = peak_idx
            while right < len(tau_line) - 1 and tau_line[right] > threshold * 0.3:
                right += 1

            if right - left < 3:  # Too narrow to fit
                continue

            # Extract feature
            feature_tau = tau_line[left:right+1]
            feature_v = np.arange(len(feature_tau)) * velocity_spacing

            # Initial guess for Voigt fit
            tau_0_guess = tau_line[peak_idx]
            v_center_guess = (peak_idx - left) * velocity_spacing
            b_guess = 20.0  # km/s, typical IGM value

            try:
                # Fit Voigt profile
                popt, _ = curve_fit(
                    voigt_approx,
                    feature_v,
                    feature_tau,
                    p0=[tau_0_guess, b_guess, v_center_guess],
                    bounds=([0, 1.0, 0], [np.inf, 100.0, feature_v[-1]]),
                    maxfev=1000
                )

                tau_0_fit, b_fit, v_center_fit = popt

                # Estimate column density
                if colden_line is not None:
                    N_HI = np.sum(colden_line[left:right+1])
                else:
                    N_HI = TAU_TO_COLDEN_CONSTANT * np.sum(feature_tau) * velocity_spacing

                # Only keep physically reasonable absorbers
                if N_HI > 1e12 and 2.0 < b_fit < 80.0:
                    column_densities.append(N_HI)
                    b_parameters.append(b_fit)

            except (RuntimeError, ValueError):
                # Fit failed, skip this feature
                continue

    column_densities = np.array(column_densities)
    b_parameters = np.array(b_parameters)

    # Convert b to temperature: T(K) = 1.28e4 * b(km/s)^2
    temperatures = 1.28e4 * b_parameters**2

    return {
        'N_HI': column_densities,
        'b_params': b_parameters,
        'temperatures': temperatures,
        'b_median': float(np.median(b_parameters)) if len(b_parameters) > 0 else np.nan,
        'b_mean': float(np.mean(b_parameters)) if len(b_parameters) > 0 else np.nan,
        'b_std': float(np.std(b_parameters)) if len(b_parameters) > 0 else np.nan,
        'n_absorbers': len(b_parameters)
    }


def compute_temperature_density_relation(temperature, density, tau, min_tau=0.1):
    # Flatten arrays and filter by optical depth
    temp_flat = temperature.flatten()
    dens_flat = density.flatten()
    tau_flat = tau.flatten()

    # Filter: only include absorbing gas (tau > min_tau) and valid values
    mask = (tau_flat > min_tau) & (temp_flat > 0) & (dens_flat > 0)
    mask &= np.isfinite(temp_flat) & np.isfinite(dens_flat)

    temp_filtered = temp_flat[mask]
    dens_filtered = dens_flat[mask]

    if len(temp_filtered) < 100:
        print(f"  Warning: Only {len(temp_filtered)} valid pixels for T-ρ fit")
        return {
            'temperature': temp_filtered,
            'density': dens_filtered,
            'log_T': np.array([]),
            'log_rho': np.array([]),
            'T0': np.nan,
            'gamma': np.nan,
            'gamma_err': np.nan,
            'n_pixels': len(temp_filtered)
        }

    # Convert density to overdensity (ρ/ρ_mean)
    rho_mean = np.median(dens_filtered)
    overdensity = dens_filtered / rho_mean

    # Take logarithms for power-law fit
    log_T = np.log10(temp_filtered)
    log_rho = np.log10(overdensity)

    # Fit T-ρ relation: log(T) = log(T0) + (gamma-1) * log(ρ/ρ_mean)
    # Robust fit using median binning
    rho_bins = np.linspace(log_rho.min(), log_rho.max(), 30)
    T_median = []
    rho_centers = []

    for i in range(len(rho_bins) - 1):
        mask_bin = (log_rho >= rho_bins[i]) & (log_rho < rho_bins[i+1])
        if np.sum(mask_bin) > 10:
            T_median.append(np.median(log_T[mask_bin]))
            rho_centers.append((rho_bins[i] + rho_bins[i+1]) / 2)

    if len(rho_centers) > 5:
        # Linear fit in log-log space
        coeffs = np.polyfit(rho_centers, T_median, 1)
        gamma_minus_1 = coeffs[0]
        log_T0 = coeffs[1]

        T0 = 10**log_T0
        gamma = gamma_minus_1 + 1.0

        # Estimate uncertainty
        T_pred = np.polyval(coeffs, rho_centers)
        residuals = np.array(T_median) - T_pred
        gamma_err = np.std(
            residuals) / np.std(rho_centers) if len(rho_centers) > 1 else np.nan
    else:
        T0 = np.nan
        gamma = np.nan
        gamma_err = np.nan

    return {
        'temperature': temp_filtered,
        'density': overdensity,
        'log_T': log_T,
        'log_rho': log_rho,
        'T0': float(T0) if np.isfinite(T0) else np.nan,
        'gamma': float(gamma) if np.isfinite(gamma) else np.nan,
        'gamma_err': float(gamma_err) if np.isfinite(gamma_err) else np.nan,
        'n_pixels': len(temp_filtered),
        'rho_mean': float(rho_mean)
    }


def compute_metal_line_statistics(tau, velocity_spacing, ion_name='Metal', threshold=0.05, colden=None):
    """Compute statistics for metal line absorption systems."""
    n_sightlines, n_pixels = tau.shape

    # Covering fraction: fraction of pixels with detectable absorption
    covering_fraction = np.sum(tau > threshold) / tau.size

    # Mean/median tau in absorbing regions
    tau_absorbing = tau[tau > threshold]
    if len(tau_absorbing) > 0:
        mean_tau_abs = np.mean(tau_absorbing)
        median_tau_abs = np.median(tau_absorbing)
    else:
        mean_tau_abs = 0.0
        median_tau_abs = 0.0

    # Count absorption systems
    column_densities = []
    n_systems = 0

    for i in range(n_sightlines):
        tau_line = tau[i, :]
        colden_line = colden[i, :] if colden is not None else None

        # Find absorption features
        absorbing = tau_line > threshold

        # Count connected regions
        in_feature = False
        feature_start = 0

        for j in range(len(tau_line)):
            if absorbing[j] and not in_feature:
                # Start of new feature
                in_feature = True
                feature_start = j
                n_systems += 1
            elif not absorbing[j] and in_feature:
                # End of feature
                in_feature = False
                
                # Estimate column density
                if colden_line is not None:
                    # Use fake_spectra's pre-computed values
                    N_ion = np.sum(colden_line[feature_start:j])
                else:
                    # Fallback: use generic constant (NEEDS VALIDATION!)
                    feature_tau = tau_line[feature_start:j]
                    N_ion = 1e13 * np.sum(feature_tau) * velocity_spacing
                column_densities.append(N_ion)

        # Handle case where feature extends to edge
        if in_feature:
            if colden_line is not None:
                N_ion = np.sum(colden_line[feature_start:])
            else:
                feature_tau = tau_line[feature_start:]
                N_ion = 1e13 * np.sum(feature_tau) * velocity_spacing
            column_densities.append(N_ion)

    column_densities = np.array(column_densities)

    # Compute dN/dz (line density per unit redshift)
    # Assuming each sightline samples ~dz = 0.1 (rough estimate)
    # For more accurate dN/dz, need actual path length
    dN_dz = n_systems / n_sightlines / 0.1 if n_sightlines > 0 else 0.0

    # Equivalent width (in velocity space, convert to Angstroms later)
    # EW ~ integral (1 - exp(-tau)) dv
    flux = np.exp(-tau)
    equivalent_width_vel = np.sum(1.0 - flux) * velocity_spacing  # km/s

    return {
        'ion_name': ion_name,
        'n_absorbers': n_systems,
        'n_sightlines': n_sightlines,
        'dN_dz': float(dN_dz),
        'covering_fraction': float(covering_fraction),
        'mean_tau': float(mean_tau_abs),
        'median_tau': float(median_tau_abs),
        # Per sightline
        'equivalent_width_vel': float(equivalent_width_vel / n_sightlines),
        'column_densities': column_densities,
        'log_N_mean': float(np.log10(np.mean(column_densities))) if len(column_densities) > 0 else np.nan,
        'log_N_median': float(np.log10(np.median(column_densities))) if len(column_densities) > 0 else np.nan
    }


def format_stats_table(stats):
    """Format flux statistics as a text table."""
    lines = [
        "Statistics:",
        "=" * 40,
        f"Mean flux:              {stats['mean_flux']:.4f}",
        f"Median flux:            {stats['median_flux']:.4f}",
        f"Std dev:                {stats['std_flux']:.4f}",
        "",
        f"Effective tau_eff:        {stats['effective_tau']:.4f}",
        f"Mean tau:                 {stats['mean_tau']:.4f}",
        f"Median tau:               {stats['median_tau']:.4f}",
        "",
        f"Deep absorption:        {stats['deep_absorption_frac']*100:.2f}%",
        f"Moderate absorption:    {
            stats['moderate_absorption_frac']*100:.2f}%",
        f"Weak absorption:        {stats['weak_absorption_frac']*100:.2f}%",
    ]
    return "\n".join(lines)


####################################
# ADVANCED ANALYSIS USING VoigtFit #
####################################

# Compute column density distribution using VoigtFit profile fitting.
def compute_column_density_distribution_vpfit(flux, wavelength, redshift, threshold=0.05,
                                              continuum_window=50, min_snr=5.0,
                                              output_dir=None):
    try:
        import VoigtFit
        from astropy import units as u
        import astropy.constants as const
    except ImportError:
        print("ERROR: VoigtFit not installed. Install with: pip install VoigtFit")
        return {'error': 'VoigtFit not available'}

    # Convert wavelength to velocity space (km/s) relative to Lyman-alpha
    lambda_lya = 1215.67  # Angstroms
    lambda_rest = lambda_lya * (1 + redshift)

    # Convert wavelength to velocity (km/s)
    c = 299792.458  # km/s
    velocity = c * (wavelength - lambda_rest) / lambda_rest

    # Convert flux to optical depth
    tau = -np.log(flux)

    column_densities = []
    b_parameters = []
    absorber_redshifts = []
    errors_N = []
    errors_b = []
    chi_squared_values = []

    # Process each sightline
    for i in range(flux.shape[0]):
        flux_line = flux[i, :]
        tau_line = tau[i, :]

        # Find absorption features
        absorbers = _find_absorption_features(tau_line, velocity, threshold)

        for absorber in absorbers:
            try:
                # Fit Voigt profile to this absorber
                result = _fit_single_absorber(flux_line, wavelength, absorber,
                                              redshift, continuum_window, min_snr)

                if result is not None:
                    column_densities.append(result['N_HI'])
                    b_parameters.append(result['b'])
                    absorber_redshifts.append(result['z_abs'])
                    errors_N.append(result['N_err'])
                    errors_b.append(result['b_err'])
                    chi_squared_values.append(result['chi2'])

            except Exception as e:
                print(f"Warning: Failed to fit absorber in sightline {i}: {e}")
                continue

    # Convert to numpy arrays
    column_densities = np.array(column_densities)
    b_parameters = np.array(b_parameters)
    absorber_redshifts = np.array(absorber_redshifts)
    errors_N = np.array(errors_N)
    errors_b = np.array(errors_b)
    chi_squared_values = np.array(chi_squared_values)

    return {
        'N_HI': column_densities,
        'b_params': b_parameters,
        'redshifts': absorber_redshifts,
        'errors_N': errors_N,
        'errors_b': errors_b,
        'chi_squared': chi_squared_values,
        'n_absorbers': len(column_densities),
        'method': 'VoigtFit'
    }


# Find absorption features in optical depth spectrum.
def _find_absorption_features(tau, velocity, threshold):
    absorbers = []

    # Find contiguous regions above threshold
    in_feature = False
    start_idx = 0

    for i in range(len(tau)):
        if tau[i] > threshold and not in_feature:
            # Start of feature
            in_feature = True
            start_idx = i
        elif tau[i] <= threshold and in_feature:
            # End of feature
            end_idx = i - 1

            # Check if feature is significant
            if end_idx - start_idx >= 3:  # At least 3 pixels
                absorbers.append({
                    'vel_start': velocity[start_idx],
                    'vel_end': velocity[end_idx],
                    'tau_max': np.max(tau[start_idx:end_idx+1]),
                    'pixel_start': start_idx,
                    'pixel_end': end_idx
                })

            in_feature = False

    # Handle feature at end
    if in_feature and len(tau) - start_idx >= 3:
        absorbers.append({
            'vel_start': velocity[start_idx],
            'vel_end': velocity[-1],
            'tau_max': np.max(tau[start_idx:]),
            'pixel_start': start_idx,
            'pixel_end': len(tau) - 1
        })

    return absorbers


# Fit a single absorber with VoigtFit.
def _fit_single_absorber(flux, wavelength, absorber, redshift, continuum_window, min_snr):
    try:
        from VoigtFit import VoigtFitter
        import astropy.units as u

        # Extract region around absorber
        pixel_start = max(0, absorber['pixel_start'] - continuum_window)
        pixel_end = min(len(flux), absorber['pixel_end'] + continuum_window)

        flux_region = flux[pixel_start:pixel_end]
        wave_region = wavelength[pixel_start:pixel_end]

        # Create VoigtFit spectrum object
        spec = VoigtFitter(wave_region, flux_region)

        # Estimate continuum (simple linear fit to edges)
        edge_size = min(10, len(flux_region)//4)
        left_flux = np.mean(flux_region[:edge_size])
        right_flux = np.mean(flux_region[-edge_size:])
        continuum = np.linspace(left_flux, right_flux, len(flux_region))
        spec.set_continuum(continuum)

        # Estimate initial parameters
        center_vel = (absorber['vel_start'] + absorber['vel_end']) / 2
        center_wave = 1215.67 * (1 + redshift) * (1 + center_vel/299792.458)

        # Rough column density estimate from optical depth
        tau_max = absorber['tau_max']
        dv = np.abs(absorber['vel_end'] - absorber['vel_start'])
        N_est = 1.13e14 * tau_max * dv  # Simplified estimate

        # Rough b estimate from width
        b_est = dv / 2.355  # FWHM to sigma conversion

        # Add HI Lyman-alpha component
        spec.add_line('HI1215', center_wave, N_est, b_est, z=redshift)

        # Fit the spectrum
        spec.fit()

        # Extract results
        if spec.lines:
            line = spec.lines[0]  # First (and only) line

            # Check fit quality
            chi2 = spec.chi2
            if chi2 > 100:  # Poor fit
                return None

            return {
                'N_HI': line.N.value,
                'b': line.b.value,
                'z_abs': line.z,
                'N_err': line.N.std if hasattr(line.N, 'std') else 0.1,
                'b_err': line.b.std if hasattr(line.b, 'std') else 5.0,
                'chi2': chi2
            }

    except Exception as e:
        print(f"VoigtFit error: {e}")
        return None
