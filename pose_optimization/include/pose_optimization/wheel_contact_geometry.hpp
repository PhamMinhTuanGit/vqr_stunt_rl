#pragma once

#include <urdf_model/model.h>
#include <urdf_world/types.h>

#include <Eigen/Core>

#include <string>

namespace pose_optimization {

// Algebra shared by the audited double-precision helper and the CasADi
// expression used by M-TO1B. The public double overload below performs the
// runtime validation; symbolic callers constrain the wheel geometry to the
// already-audited finite, non-degenerate URDF cylinders.
template <typename Scalar>
Eigen::Matrix<Scalar, 3, 1> treadContactPointExpression(
    const Eigen::Matrix<Scalar, 3, 1> &collision_center,
    const Eigen::Matrix<Scalar, 3, 1> &wheel_axis, const Scalar &radius) {
  const Eigen::Matrix<Scalar, 3, 1> axis = wheel_axis / wheel_axis.norm();
  Eigen::Matrix<Scalar, 3, 1> ez;
  ez << Scalar(0.0), Scalar(0.0), Scalar(1.0);
  const Eigen::Matrix<Scalar, 3, 1> projected_vertical =
      ez - axis * axis.dot(ez);
  return collision_center -
         radius * projected_vertical / projected_vertical.norm();
}

struct WheelCollisionGeometry {
  std::string wheel_frame;
  Eigen::Vector3d origin_xyz;
  Eigen::Vector3d origin_rpy;
  double radius;
  double length;
  // URDF cylinders are aligned with geometry-frame +Z. This is that axis
  // rotated by the collision origin into the wheel BODY/link frame.
  Eigen::Vector3d cylinder_axis_in_wheel_frame;
};

WheelCollisionGeometry parseWheelCollisionGeometry(
    const urdf::ModelInterfaceSharedPtr &urdf_model,
    const std::string &wheel_link_name);

// center and axis must be expressed in the same gravity-aligned coordinate
// frame whose +Z is ez (normally WORLD). The axis is normalized internally. A
// vertical cylinder axis is rejected because the downward tread direction is
// then undefined by the requested projection.
Eigen::Vector3d treadContactPoint(const Eigen::Vector3d &collision_center,
                                  const Eigen::Vector3d &wheel_axis,
                                  double radius);

} // namespace pose_optimization
