#include "pose_optimization/kinematic_geometry.hpp"

#include <Eigen/Core>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool near(double lhs, double rhs, double tolerance = 1.0e-12) {
  return std::abs(lhs - rhs) <= tolerance;
}

} // namespace

int main() {
  try {
    const Eigen::Vector3d fl(1.0, 2.0, 0.0);
    const Eigen::Vector3d hr(4.0, 6.0, 0.0);
    const Eigen::Vector3d on_segment(2.5, 4.0, 3.0);
    const auto on =
        pose_optimization::supportLineMetrics(fl, hr, on_segment);
    require(near(on.length, 5.0), "Support length is incorrect");
    require(near(on.s, 2.5), "Support segment coordinate is incorrect");
    require(near(on.signed_distance, 0.0),
            "Point on support line has nonzero distance");

    const Eigen::Vector3d left_of_line(1.7, 4.6, 0.0);
    const auto left =
        pose_optimization::supportLineMetrics(fl, hr, left_of_line);
    require(near(left.s, 2.5), "Projected segment coordinate changed");
    require(near(left.signed_distance, 1.0),
            "Signed point-to-line distance is incorrect");

    std::cout << "Kinematic support geometry: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Kinematic support geometry: FAIL: " << error.what() << '\n';
    return 1;
  }
}
