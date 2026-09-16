#include "pose_optimization/wheel_contact_geometry.hpp"

#include <urdf_model/joint.h>
#include <urdf_parser/urdf_parser.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr double kTolerance = 1.0e-10;

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool near(double lhs, double rhs, double tolerance = kTolerance) {
  return std::abs(lhs - rhs) <= tolerance;
}

} // namespace

int main() {
  try {
    const Eigen::Vector3d center(0.25, -0.20, 0.50);
    const double radius = 0.091;
    const Eigen::Vector3d horizontal_axis(0.0, 1.0, 0.0);
    const Eigen::Vector3d contact = pose_optimization::treadContactPoint(
        center, horizontal_axis, radius);
    require(contact.allFinite(), "Horizontal-axis contact point is not finite");
    require(near(contact.x(), center.x()) && near(contact.y(), center.y()) &&
                near(contact.z(), center.z() - radius),
            "Horizontal-axis test did not produce center.z - radius");

    const urdf::ModelInterfaceSharedPtr urdf_model =
        urdf::parseURDFFile(VQR_URDF_PATH);
    require(static_cast<bool>(urdf_model), "Failed to parse VQR URDF");

    const std::array<std::string, 4> wheel_names = {
        "FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL"};
    for (const std::string &name : wheel_names) {
      const auto geometry =
          pose_optimization::parseWheelCollisionGeometry(urdf_model, name);
      require(geometry.origin_xyz.allFinite() && geometry.origin_rpy.allFinite() &&
                  geometry.cylinder_axis_in_wheel_frame.allFinite(),
              "Parsed non-finite geometry for " + name);
      require(near(geometry.radius, 0.091) && near(geometry.length, 0.04),
              "Unexpected cylinder dimensions for " + name);
      require(geometry.origin_xyz.norm() < kTolerance &&
                  near(geometry.origin_rpy.x(), 1.5709) &&
                  near(geometry.origin_rpy.y(), 0.0) &&
                  near(geometry.origin_rpy.z(), 0.0),
              "Unexpected collision origin for " + name);

      const urdf::JointConstSharedPtr joint = urdf_model->getJoint(name);
      require(static_cast<bool>(joint), "Missing wheel joint: " + name);
      const Eigen::Vector3d joint_axis(joint->axis.x, joint->axis.y,
                                       joint->axis.z);
      require(joint_axis.allFinite() && joint_axis.norm() > 0.0,
              "Invalid wheel joint axis for " + name);
      // Collision rpy=1.5709 is a rounded pi/2. Its physical cylinder axis is
      // therefore close to, but intentionally checked separately from, -Y.
      require(geometry.cylinder_axis_in_wheel_frame.cross(
                  joint_axis.normalized()).norm() < 2.0e-4,
              "Cylinder and joint axes disagree for " + name);

      const Eigen::Vector3d parsed_contact =
          pose_optimization::treadContactPoint(
              Eigen::Vector3d(0.1, 0.2, 0.3),
              geometry.cylinder_axis_in_wheel_frame, geometry.radius);
      require(parsed_contact.allFinite(),
              "Parsed-geometry contact point is not finite for " + name);
    }

    std::cout << "horizontal wheel axis contact_z = center_z - radius: PASS\n";
    std::cout << "finite contact-point result: PASS\n";
    std::cout << "FL/FR/HL/HR wheel geometry parse: PASS\n";
    std::cout << "M-TO1A contact-point test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "M-TO1A contact-point test: FAIL\n";
    std::cerr << "Reason: " << error.what() << '\n';
    return 1;
  }
}
