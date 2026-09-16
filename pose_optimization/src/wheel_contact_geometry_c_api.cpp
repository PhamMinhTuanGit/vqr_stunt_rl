#include "pose_optimization/wheel_contact_geometry.hpp"

#include <urdf_parser/urdf_parser.h>

#include <Eigen/Core>

#include <exception>
#include <string>

namespace {

thread_local std::string last_error;

template <typename Callable> int guard(Callable &&callable) {
  try {
    callable();
    last_error.clear();
    return 0;
  } catch (const std::exception &error) {
    last_error = error.what();
    return 1;
  } catch (...) {
    last_error = "Unknown wheel-contact helper failure";
    return 1;
  }
}

} // namespace

extern "C" {

const char *vqr_wheel_contact_last_error() { return last_error.c_str(); }

int vqr_parse_wheel_collision(const char *urdf_path, const char *wheel_link,
                              double *origin_xyz, double *axis_xyz,
                              double *radius, double *length) {
  return guard([&]() {
    if (!urdf_path || !wheel_link || !origin_xyz || !axis_xyz || !radius ||
        !length) {
      throw std::runtime_error("Null argument to vqr_parse_wheel_collision");
    }
    const auto model = urdf::parseURDFFile(urdf_path);
    const auto geometry =
        pose_optimization::parseWheelCollisionGeometry(model, wheel_link);
    Eigen::Map<Eigen::Vector3d> origin_map(origin_xyz);
    Eigen::Map<Eigen::Vector3d> axis_map(axis_xyz);
    origin_map = geometry.origin_xyz;
    axis_map = geometry.cylinder_axis_in_wheel_frame;
    *radius = geometry.radius;
    *length = geometry.length;
  });
}

int vqr_tread_contact_point(const double *collision_center,
                            const double *wheel_axis, double radius,
                            double *contact) {
  return guard([&]() {
    if (!collision_center || !wheel_axis || !contact) {
      throw std::runtime_error("Null argument to vqr_tread_contact_point");
    }
    Eigen::Map<Eigen::Vector3d> contact_map(contact);
    contact_map = pose_optimization::treadContactPoint(
        Eigen::Map<const Eigen::Vector3d>(collision_center),
        Eigen::Map<const Eigen::Vector3d>(wheel_axis), radius);
  });
}

} // extern "C"
