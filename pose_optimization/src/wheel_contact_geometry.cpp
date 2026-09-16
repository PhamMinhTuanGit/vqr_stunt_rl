#include "pose_optimization/wheel_contact_geometry.hpp"

#include <urdf_model/link.h>

#include <Eigen/Geometry>

#include <cmath>
#include <stdexcept>

namespace pose_optimization {
namespace {

constexpr double kMinimumNorm = 1.0e-12;

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

WheelCollisionGeometry parseWheelCollisionGeometry(
    const urdf::ModelInterfaceSharedPtr &urdf_model,
    const std::string &wheel_link_name) {
  require(static_cast<bool>(urdf_model), "URDF model is null");
  const urdf::LinkConstSharedPtr link = urdf_model->getLink(wheel_link_name);
  require(static_cast<bool>(link), "Missing URDF wheel link: " + wheel_link_name);
  require(link->collision_array.size() == 1,
          "Expected exactly one collision on wheel link: " + wheel_link_name);

  const urdf::CollisionSharedPtr &collision = link->collision_array.front();
  require(static_cast<bool>(collision),
          "Null collision on wheel link: " + wheel_link_name);
  require(static_cast<bool>(collision->geometry),
          "Collision has no geometry on wheel link: " + wheel_link_name);
  require(collision->geometry->type == urdf::Geometry::CYLINDER,
          "Wheel collision is not a cylinder: " + wheel_link_name);

  const auto *cylinder =
      static_cast<const urdf::Cylinder *>(collision->geometry.get());
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
  collision->origin.rotation.getRPY(roll, pitch, yaw);
  const Eigen::Quaterniond collision_rotation(
      collision->origin.rotation.w, collision->origin.rotation.x,
      collision->origin.rotation.y, collision->origin.rotation.z);

  WheelCollisionGeometry result{
      wheel_link_name,
      {collision->origin.position.x, collision->origin.position.y,
       collision->origin.position.z},
      {roll, pitch, yaw},
      cylinder->radius,
      cylinder->length,
      collision_rotation * Eigen::Vector3d::UnitZ(),
  };

  require(result.origin_xyz.allFinite() && result.origin_rpy.allFinite() &&
              result.cylinder_axis_in_wheel_frame.allFinite() &&
              std::isfinite(result.radius) && result.radius > 0.0 &&
              std::isfinite(result.length) && result.length > 0.0,
          "Wheel collision geometry is not finite and positive: " +
              wheel_link_name);
  require(std::abs(result.cylinder_axis_in_wheel_frame.norm() - 1.0) < 1.0e-12,
          "Wheel cylinder axis is not unit length: " + wheel_link_name);
  return result;
}

Eigen::Vector3d treadContactPoint(const Eigen::Vector3d &collision_center,
                                  const Eigen::Vector3d &wheel_axis,
                                  double radius) {
  require(collision_center.allFinite(), "Collision center is not finite");
  require(wheel_axis.allFinite(), "Wheel axis is not finite");
  require(std::isfinite(radius) && radius > 0.0,
          "Wheel radius must be positive and finite");
  require(wheel_axis.norm() > kMinimumNorm, "Wheel axis has zero length");

  const Eigen::Vector3d axis = wheel_axis.normalized();
  const Eigen::Vector3d ez = Eigen::Vector3d::UnitZ();
  const Eigen::Vector3d projected_vertical = ez - axis * axis.dot(ez);
  require(projected_vertical.norm() > kMinimumNorm,
          "Wheel axis is vertical; tread contact direction is undefined");
  const Eigen::Vector3d contact = treadContactPointExpression(
      collision_center, wheel_axis, radius);
  require(contact.allFinite(), "Computed tread contact point is not finite");
  return contact;
}

} // namespace pose_optimization
