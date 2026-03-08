#ifndef CGM_CGM_HALOS_H
#define CGM_CGM_HALOS_H

#include <Eigen/Dense>

namespace cgm {
namespace cgm {

struct FilteredHalosResult {
    Eigen::VectorXi isolated_mask;
    int n_isolated;
    int n_non_isolated;
};

FilteredHalosResult filter_isolated_halos(
    const Eigen::Ref<const Eigen::ArrayXXf>& positions,
    const Eigen::Ref<const Eigen::VectorXf>& masses,
    const Eigen::Ref<const Eigen::VectorXf>& radii,
    float isolation_factor = 3.0f,
    float box_size = 0.0f);

Eigen::MatrixXf compute_impact_parameters(
    const Eigen::Ref<const Eigen::ArrayXXf>& sightline_origins,
    const Eigen::Ref<const Eigen::ArrayXXf>& sightline_dirs,
    const Eigen::Ref<const Eigen::ArrayXXf>& halo_positions,
    const Eigen::Ref<const Eigen::VectorXf>& halo_radii);

}
}

#endif
