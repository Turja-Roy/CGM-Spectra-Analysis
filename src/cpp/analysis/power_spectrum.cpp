#include "power_spectrum.h"
#include "constants.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <complex>
#include <omp.h>

namespace cgm {
namespace analysis {

// Cooley-Tukey FFT (radix-2, in-place)
static void fft_radix2(std::complex<float>* data, int n) {
    if (n <= 1) return;
    
    // Bit reversal permutation
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
        if (i < j) {
            std::swap(data[i], data[j]);
        }
        int k = n / 2;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }
    
    // Cooley-Tukey iterative FFT
    for (int len = 2; len <= n; len *= 2) {
        float angle = -2.0f * M_PI / len;
        std::complex<float> wn(std::cos(angle), std::sin(angle));
        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (int k = 0; k < len / 2; k++) {
                std::complex<float> u = data[i + k];
                std::complex<float> t = w * data[i + k + len / 2];
                data[i + k] = u + t;
                data[i + k + len / 2] = u - t;
                w *= wn;
            }
        }
    }
}

// Compute power spectrum using O(N log N) FFT
static Eigen::VectorXd compute_fft_power(const Eigen::ArrayXf& flux_contrast, int n_pixels) {
    // Pad to power of 2 if needed
    int n = 1;
    while (n < n_pixels) n *= 2;
    
    std::vector<std::complex<float>> fft_data(n);
    for (int i = 0; i < n_pixels; i++) {
        fft_data[i] = std::complex<float>(flux_contrast(i), 0.0f);
    }
    for (int i = n_pixels; i < n; i++) {
        fft_data[i] = std::complex<float>(0.0f, 0.0f);
    }
    
    fft_radix2(fft_data.data(), n);
    
    // Compute power spectrum (only first n_pixels/2 + 1 values are unique for real input)
    int n_k = n_pixels / 2 + 1;
    Eigen::VectorXd power(n_k);
    for (int i = 0; i < n_k; i++) {
        float real = fft_data[i].real();
        float imag = fft_data[i].imag();
        power(i) = (real * real + imag * imag) / n_pixels;
    }
    
    return power;
}

PowerSpectrumResult compute_power_spectrum(
    const Eigen::Ref<const Eigen::ArrayXXf>& flux,
    double velocity_spacing,
    int chunk_size) {
    
    const int n_sightlines = flux.rows();
    const int n_pixels = flux.cols();
    
    float mean_flux = flux.mean();
    if (mean_flux <= 0) {
        throw std::invalid_argument("Mean flux must be positive");
    }
    
    int n_k = n_pixels / 2 + 1;
    Eigen::VectorXd k(n_k);
    for (int i = 0; i < n_k; ++i) {
        k(i) = 2.0 * M_PI * i / (n_pixels * velocity_spacing);
    }
    
    std::vector<double> power_sum(n_k, 0.0);
    std::vector<double> power_sum_sq(n_k, 0.0);
    
    int n_chunks = (n_sightlines + chunk_size - 1) / chunk_size;
    
    #pragma omp parallel
    {
        std::vector<double> local_power_sum(n_k, 0.0);
        std::vector<double> local_power_sum_sq(n_k, 0.0);
        
        #pragma omp for schedule(dynamic)
        for (int c = 0; c < n_chunks; ++c) {
            int start = c * chunk_size;
            int end = std::min((c + 1) * chunk_size, n_sightlines);
            
            for (int i = start; i < end; ++i) {
                Eigen::ArrayXf delta_F = flux.row(i) / mean_flux - 1.0f;
                
                Eigen::VectorXd power = compute_fft_power(delta_F, n_pixels);
                
                for (int j = 0; j < n_k; ++j) {
                    double p = power(j);
                    local_power_sum[j] += p;
                    local_power_sum_sq[j] += p * p;
                }
            }
        }
        
        #pragma omp critical
        {
            for (int j = 0; j < n_k; ++j) {
                power_sum[j] += local_power_sum[j];
                power_sum_sq[j] += local_power_sum_sq[j];
            }
        }
    }
    
    Eigen::VectorXd power_sum_eigen = Eigen::VectorXd::Map(power_sum.data(), n_k);
    Eigen::VectorXd power_sum_sq_eigen = Eigen::VectorXd::Map(power_sum_sq.data(), n_k);
    
    Eigen::VectorXd P_k_mean = (power_sum_eigen / n_sightlines) * velocity_spacing;
    Eigen::VectorXd mean_power = power_sum_eigen / n_sightlines;
    Eigen::VectorXd mean_power_sq = power_sum_sq_eigen / n_sightlines;
    
    Eigen::VectorXd variance(n_k);
    for (int j = 0; j < n_k; ++j) {
        double mp = mean_power(j);
        double mp2 = mean_power_sq(j);
        double v = mp2 - mp * mp;
        variance(j) = (v > 0) ? v : 0;
    }
    
    Eigen::VectorXd P_k_std(n_k);
    for (int j = 0; j < n_k; ++j) {
        P_k_std(j) = std::sqrt(variance(j)) * velocity_spacing;
    }
    
    Eigen::VectorXd P_k_err = P_k_std / std::sqrt(n_sightlines);
    Eigen::VectorXd n_modes = Eigen::VectorXd::Ones(n_k) * n_sightlines;
    
    PowerSpectrumResult result;
    result.k = k;
    result.P_k_mean = P_k_mean;
    result.P_k_std = P_k_std;
    result.P_k_err = P_k_err;
    result.n_modes = n_modes;
    result.mean_flux = mean_flux;
    result.n_sightlines = n_sightlines;
    result.velocity_spacing = velocity_spacing;
    
    return result;
}

}
}
