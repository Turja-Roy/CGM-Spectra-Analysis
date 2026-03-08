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

ColumnDensityResult compute_column_density_distribution(
    const Eigen::Ref<const Eigen::ArrayXXf>& tau,
    double velocity_spacing,
    float threshold = 0.5f,
    const Eigen::Ref<const Eigen::ArrayXXf>* colden = nullptr,
    double redshift = std::nan(""),
    double box_size_ckpc_h = std::nan(""),
    double hubble = 0.6774,
    double omega_m = 0.3089);

}
}

#endif
