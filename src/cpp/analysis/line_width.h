#ifndef CGM_ANALYSIS_LINE_WIDTH_H
#define CGM_ANALYSIS_LINE_WIDTH_H

#include <Eigen/Dense>
#include "voigt.h"

namespace cgm {
namespace analysis {

struct LineWidthResult {
    Eigen::VectorXd N_HI;
    Eigen::VectorXd b_params;
    Eigen::VectorXd temperatures;
    double b_median;
    double b_mean;
    double b_std;
    int n_absorbers;
};

LineWidthResult compute_line_width_distribution(
    const Eigen::Ref<const Eigen::ArrayXXf>& tau,
    double velocity_spacing,
    float threshold = 0.5f,
    const Eigen::Ref<const Eigen::ArrayXXf>* colden = nullptr);

}
}

#endif
