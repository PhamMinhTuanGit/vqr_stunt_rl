#include "pose_optimization/kinematic_geometry.hpp"

#include <cmath>
#include <stdexcept>

namespace pose_optimization {

SupportLineMetrics supportLineMetrics(const Eigen::Vector3d &fl_contact,
                                      const Eigen::Vector3d &hr_contact,
                                      const Eigen::Vector3d &com) {
  if (!fl_contact.allFinite() || !hr_contact.allFinite() || !com.allFinite()) {
    throw std::runtime_error("Support-line input is not finite");
  }
  const Eigen::Vector2d delta =
      hr_contact.head<2>() - fl_contact.head<2>();
  const double length = delta.norm();
  if (!std::isfinite(length) || length <= 1.0e-12) {
    throw std::runtime_error("FL-HR support line has zero length");
  }
  const Eigen::Vector2d direction = delta / length;
  const Eigen::Vector2d normal(-direction.y(), direction.x());
  const Eigen::Vector2d offset = com.head<2>() - fl_contact.head<2>();
  return {normal.dot(offset), direction.dot(offset), length};
}

} // namespace pose_optimization
