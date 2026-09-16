#pragma once

#include <Eigen/Core>

namespace pose_optimization {

struct SupportLineMetrics {
  double signed_distance;
  double s;
  double length;
};

SupportLineMetrics supportLineMetrics(const Eigen::Vector3d &fl_contact,
                                      const Eigen::Vector3d &hr_contact,
                                      const Eigen::Vector3d &com);

} // namespace pose_optimization
