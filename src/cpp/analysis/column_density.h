#ifndef CGM_ANALYSIS_COLUMN_DENSITY_H
#define CGM_ANALYSIS_COLUMN_DENSITY_H

#include <Eigen/Dense>
#include <cmath>
#include <string>

namespace cgm {
namespace analysis {

struct ColumnDensityResult {
    Eigen::VectorXd N_HI;
    Eigen::VectorXi counts;
    Eigen::VectorXd bins;
    Eigen::VectorXd bin_centers;
    Eigen::VectorXd f_N;
    double beta_fit;
    int n_absorbers;
    int n_sightlines;
    double dX;
    double redshift;
};

// New version with raw pointer for proper layout handling
ColumnDensityResult compute_column_density_distribution(
    const Eigen::Ref<const Eigen::ArrayXXf>& tau,
    double velocity_spacing,
    float threshold = 0.5f,
    const float* colden_data = nullptr,
    int colden_rows = 0,
    int colden_cols = 0,
    double redshift = std::nan(""),
    double box_size_ckpc_h = std::nan(""),
    double hubble = 0.6774,
    double omega_m = 0.3089);

// Old version - deprecated
ColumnDensityResult compute_column_density_distribution(
    const Eigen::Ref<const Eigen::ArrayXXf>& tau,
    double velocity_spacing,
    float threshold,
    const Eigen::Ref<const Eigen::ArrayXXf>* colden,
    double redshift,
    double box_size_ckpc_h,
    double hubble,
    double omega_m);

}
}

#endif
