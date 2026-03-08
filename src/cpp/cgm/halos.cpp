#include "halos.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <omp.h>

namespace cgm {
namespace cgm {

FilteredHalosResult filter_isolated_halos(
    const Eigen::Ref<const Eigen::ArrayXXf>& positions,
    const Eigen::Ref<const Eigen::VectorXf>& masses,
    const Eigen::Ref<const Eigen::VectorXf>& radii,
    float isolation_factor,
    float box_size) {
    
    const int n_halos = positions.rows();
    
    std::vector<int> indices(n_halos);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), 
        [&masses](int a, int b) { return masses(a) > masses(b); });
    
    Eigen::VectorXi isolated_mask = Eigen::VectorXi::Ones(n_halos);
    
    if (n_halos <= 1000) {
        #pragma omp parallel for schedule(dynamic)
        for (int ii = 0; ii < n_halos; ++ii) {
            int i = indices[ii];
            float check_radius = isolation_factor * radii(i);
            
            for (int jj = 0; jj < n_halos; ++jj) {
                int j = indices[jj];
                if (i == j) break;
                
                if (masses(j) <= 0.5f * masses(i)) break;
                
                Eigen::Vector3f diff = positions.row(i) - positions.row(j);
                
                if (box_size > 0) {
                    if (diff(0) > box_size / 2) diff(0) -= box_size;
                    else if (diff(0) < -box_size / 2) diff(0) += box_size;
                    if (diff(1) > box_size / 2) diff(1) -= box_size;
                    else if (diff(1) < -box_size / 2) diff(1) += box_size;
                    if (diff(2) > box_size / 2) diff(2) -= box_size;
                    else if (diff(2) < -box_size / 2) diff(2) += box_size;
                }
                
                float distance = diff.norm();
                
                if (distance < check_radius + radii(j)) {
                    isolated_mask(i) = 0;
                    break;
                }
            }
        }
    } else {
        float cell_size = 10.0f;
        Eigen::Vector3f min_pos = positions.colwise().minCoeff();
        Eigen::Vector3f max_pos = positions.colwise().maxCoeff();
        Eigen::Vector3f grid_size = max_pos - min_pos;
        
        int n_cells_x = std::max(1, static_cast<int>(grid_size(0) / cell_size));
        int n_cells_y = std::max(1, static_cast<int>(grid_size(1) / cell_size));
        int n_cells_z = std::max(1, static_cast<int>(grid_size(2) / cell_size));
        
        std::vector<std::vector<int>> grid(n_cells_x * n_cells_y * n_cells_z);
        
        for (int i = 0; i < n_halos; ++i) {
            int cx = static_cast<int>((positions(i, 0) - min_pos(0)) / cell_size);
            int cy = static_cast<int>((positions(i, 1) - min_pos(1)) / cell_size);
            int cz = static_cast<int>((positions(i, 2) - min_pos(2)) / cell_size);
            
            cx = std::clamp(cx, 0, n_cells_x - 1);
            cy = std::clamp(cy, 0, n_cells_y - 1);
            cz = std::clamp(cz, 0, n_cells_z - 1);
            
            int cell_idx = cx + cy * n_cells_x + cz * n_cells_x * n_cells_y;
            grid[cell_idx].push_back(i);
        }
        
        #pragma omp parallel for schedule(dynamic)
        for (int ii = 0; ii < n_halos; ++ii) {
            int i = indices[ii];
            float check_radius = isolation_factor * radii(i);
            
            int cx = static_cast<int>((positions(i, 0) - min_pos(0)) / cell_size);
            int cy = static_cast<int>((positions(i, 1) - min_pos(1)) / cell_size);
            int cz = static_cast<int>((positions(i, 2) - min_pos(2)) / cell_size);
            
            bool is_isolated = true;
            
            for (int dx = -1; dx <= 1 && is_isolated; ++dx) {
                for (int dy = -1; dy <= 1 && is_isolated; ++dy) {
                    for (int dz = -1; dz <= 1 && is_isolated; ++dz) {
                        int nx = cx + dx;
                        int ny = cy + dy;
                        int nz = cz + dz;
                        
                        if (nx < 0 || nx >= n_cells_x || 
                            ny < 0 || ny >= n_cells_y || 
                            nz < 0 || nz >= n_cells_z) continue;
                        
                        int cell_idx = nx + ny * n_cells_x + nz * n_cells_x * n_cells_y;
                        
                        for (int j : grid[cell_idx]) {
                            if (i == j) continue;
                            if (masses(j) <= 0.5f * masses(i)) continue;
                            
                            Eigen::Vector3f diff = positions.row(i) - positions.row(j);
                            
                            if (box_size > 0) {
                                if (diff(0) > box_size / 2) diff(0) -= box_size;
                                else if (diff(0) < -box_size / 2) diff(0) += box_size;
                                if (diff(1) > box_size / 2) diff(1) -= box_size;
                                else if (diff(1) < -box_size / 2) diff(1) += box_size;
                                if (diff(2) > box_size / 2) diff(2) -= box_size;
                                else if (diff(2) < -box_size / 2) diff(2) += box_size;
                            }
                            
                            float distance = diff.norm();
                            
                            if (distance < check_radius + radii(j)) {
                                is_isolated = false;
                                break;
                            }
                        }
                    }
                }
            }
            
            if (!is_isolated) {
                isolated_mask(i) = 0;
            }
        }
    }
    
    int n_isolated = (isolated_mask.array() == 1).count();
    
    FilteredHalosResult result;
    result.isolated_mask = isolated_mask;
    result.n_isolated = n_isolated;
    result.n_non_isolated = n_halos - n_isolated;
    
    return result;
}

Eigen::MatrixXf compute_impact_parameters(
    const Eigen::Ref<const Eigen::ArrayXXf>& sightline_origins,
    const Eigen::Ref<const Eigen::ArrayXXf>& sightline_dirs,
    const Eigen::Ref<const Eigen::ArrayXXf>& halo_positions,
    const Eigen::Ref<const Eigen::VectorXf>& halo_radii) {
    
    const int n_sightlines = sightline_origins.rows();
    const int n_halos = halo_positions.rows();
    
    Eigen::MatrixXf impact_params(n_sightlines, n_halos);
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n_sightlines; ++i) {
        Eigen::Vector3f origin(sightline_origins(i, 0), sightline_origins(i, 1), sightline_origins(i, 2));
        Eigen::Vector3f dir(sightline_dirs(i, 0), sightline_dirs(i, 1), sightline_dirs(i, 2));
        
        for (int j = 0; j < n_halos; ++j) {
            Eigen::Vector3f to_halo(halo_positions(j, 0), halo_positions(j, 1), halo_positions(j, 2));
            to_halo = to_halo - origin;
            float along_ray = to_halo.dot(dir);
            Eigen::Vector3f perpendicular = to_halo - along_ray * dir;
            
            impact_params(i, j) = perpendicular.norm();
        }
    }
    
    return impact_params;
}

}
}
