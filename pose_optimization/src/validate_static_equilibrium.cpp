#include "pose_optimization/wheel_contact_geometry.hpp"

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/joint/joint-free-flyer.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <urdf_parser/urdf_parser.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kGroundTolerance = 1.0e-4;
constexpr double kSwingClearance = 0.020;
constexpr double kComTolerance = 0.002;
constexpr double kSegmentMargin = 0.020;
constexpr double kDynamicsTolerance = 1.0e-3;
constexpr double kJointSafetyMargin = 0.05;
constexpr double kLegTorqueLimit = 60.0;
constexpr double kWheelTorqueLimit = 20.0;
constexpr double kMetricTolerance = 1.0e-8;
constexpr double kNormalizationTolerance = 1.0e-12;

const std::array<std::string, 12> kLegJointNames = {
    "FL_HipX_joint", "FR_HipX_joint", "HL_HipX_joint", "HR_HipX_joint",
    "FL_HipY_joint", "FR_HipY_joint", "HL_HipY_joint", "HR_HipY_joint",
    "FL_Knee_joint", "FR_Knee_joint", "HL_Knee_joint", "HR_Knee_joint"};

const std::array<std::string, 4> kWheelJointNames = {"FL_WHEEL", "FR_WHEEL",
                                                     "HL_WHEEL", "HR_WHEEL"};

const std::array<std::pair<std::string, std::string>, 4> kWheelFrames = {{
    {"FL", "FL_WHEEL"},
    {"FR", "FR_WHEEL"},
    {"HL", "HL_WHEEL"},
    {"HR", "HR_WHEEL"},
}};

struct JointMargin {
  std::string name;
  double lower;
  double upper;
  double minimum;
};

struct SupportMetrics {
  double signed_error;
  double s;
  double length;
};

struct Result {
  Eigen::Vector3d base_position;
  Eigen::Vector3d base_rpy;
  Eigen::Vector3d com;
  std::map<std::string, Eigen::Vector3d> contacts;
  std::map<std::string, Eigen::Vector3d> forces;
  std::map<std::string, double> leg_torques;
  std::map<std::string, double> wheel_torques;
  SupportMetrics support;
  std::vector<JointMargin> margins;
  JointMargin minimum_margin;
  Eigen::VectorXd residual;
  double residual_infinity_norm;
  double friction_fl;
  double friction_hr;
  double max_leg_utilization;
  double max_wheel_utilization;
  double quaternion_norm;
};

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

const YAML::Node requireNode(const YAML::Node &parent, const std::string &key,
                             const std::string &path) {
  const YAML::Node node = parent[key];
  require(node.IsDefined() && !node.IsNull(),
          "Saved YAML is missing required state: " + path + "." + key);
  return node;
}

double scalar(const YAML::Node &parent, const std::string &key,
              const std::string &path) {
  const double value = requireNode(parent, key, path).as<double>();
  require(std::isfinite(value), "Non-finite YAML value: " + path + "." + key);
  return value;
}

Eigen::Vector3d vector3(const YAML::Node &node, const std::string &path) {
  require(node.IsSequence() && node.size() == 3,
          "Expected three-vector: " + path);
  Eigen::Vector3d value(node[0].as<double>(), node[1].as<double>(),
                        node[2].as<double>());
  require(value.allFinite(), "Non-finite YAML vector: " + path);
  return value;
}

void near(double actual, double saved, const std::string &name,
          double tolerance = kMetricTolerance) {
  require(std::isfinite(actual) && std::isfinite(saved) &&
              std::abs(actual - saved) <= tolerance,
          name + " mismatch: recomputed=" + std::to_string(actual) +
              " saved=" + std::to_string(saved));
}

void vectorNear(const Eigen::Vector3d &actual, const Eigen::Vector3d &saved,
                const std::string &name) {
  require(actual.allFinite() && saved.allFinite() &&
              (actual - saved).cwiseAbs().maxCoeff() <= kMetricTolerance,
          name + " mismatch");
}

pinocchio::FrameIndex bodyFrameId(const pinocchio::Model &model,
                                  const std::string &name) {
  require(model.existFrame(name, pinocchio::BODY),
          "Missing BODY frame: " + name);
  const auto id = model.getFrameId(name, pinocchio::BODY);
  require(id < model.frames.size() && model.frames[id].type == pinocchio::BODY,
          "Invalid BODY frame: " + name);
  return id;
}

Eigen::Quaterniond quaternionFromRpy(const Eigen::Vector3d &rpy) {
  return Eigen::Quaterniond(
      Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX()));
}

Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
  Eigen::Matrix3d result;
  result << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return result;
}

SupportMetrics supportMetrics(const Eigen::Vector3d &fl,
                              const Eigen::Vector3d &hr,
                              const Eigen::Vector3d &com) {
  const Eigen::Vector2d delta = hr.head<2>() - fl.head<2>();
  const double length = delta.norm();
  require(std::isfinite(length) && length > 1.0e-12,
          "Degenerate FL-HR support segment");
  const Eigen::Vector2d direction = delta / length;
  const Eigen::Vector2d normal(-direction.y(), direction.x());
  const Eigen::Vector2d offset = com.head<2>() - fl.head<2>();
  return {normal.dot(offset), direction.dot(offset), length};
}

void validateSet(const YAML::Node &root, const std::string &key,
                 const std::vector<std::string> &expected) {
  const auto node = requireNode(root, key, "root");
  require(node.IsSequence() && node.size() == expected.size(),
          "Invalid " + key + " set");
  for (std::size_t i = 0; i < expected.size(); ++i)
    require(node[i].as<std::string>() == expected[i],
            "Unexpected " + key + " ordering");
}

Result validate(const std::filesystem::path &yaml_path) {
  const YAML::Node root = YAML::LoadFile(yaml_path.string());
  require(requireNode(root, "milestone", "root").as<std::string>() == "M-TO2",
          "Saved result is not M-TO2");
  require(requireNode(root, "result", "root").as<std::string>() == "PASS",
          "Saved M-TO2 result is not PASS");
  validateSet(root, "stance", {"FL", "HR"});
  validateSet(root, "swing", {"FR", "HL"});

  const auto expected_urdf = std::filesystem::canonical(VQR_URDF_PATH);
  const auto model_node = requireNode(root, "model", "root");
  const auto saved_urdf = std::filesystem::canonical(
      requireNode(model_node, "urdf_path", "model").as<std::string>());
  require(saved_urdf == expected_urdf,
          "Saved YAML references a different URDF");

  pinocchio::Model model;
  pinocchio::urdf::buildModel(expected_urdf.string(),
                              pinocchio::JointModelFreeFlyer(), model);
  require(model.nq == 27 && model.nv == 22, "Unexpected Pinocchio dimensions");
  const auto urdf_model = urdf::parseURDFFile(expected_urdf);
  require(static_cast<bool>(urdf_model), "urdfdom could not load the VQR URDF");

  const auto variables = requireNode(root, "variables", "root");
  const Eigen::Vector3d base_position(scalar(variables, "base_x", "variables"),
                                      scalar(variables, "base_y", "variables"),
                                      scalar(variables, "base_z", "variables"));
  const Eigen::Vector3d base_rpy(scalar(variables, "base_roll", "variables"),
                                 scalar(variables, "base_pitch", "variables"),
                                 scalar(variables, "base_yaw", "variables"));
  require(std::abs(base_position.x()) <= kMetricTolerance &&
              std::abs(base_position.y()) <= kMetricTolerance &&
              std::abs(base_rpy.z()) <= kMetricTolerance,
          "Fixed base x/y/yaw are not zero");
  const double orientation_limit = 25.0 * std::acos(-1.0) / 180.0;
  require(std::abs(base_rpy.x()) <= orientation_limit &&
              std::abs(base_rpy.y()) <= orientation_limit,
          "Base orientation limit violated");

  Eigen::VectorXd q = pinocchio::neutral(model);
  q.head<3>() = base_position;
  const Eigen::Quaterniond quaternion = quaternionFromRpy(base_rpy);
  q.segment<4>(3) << quaternion.x(), quaternion.y(), quaternion.z(),
      quaternion.w();
  const auto legs = requireNode(variables, "leg_joints", "variables");
  std::vector<JointMargin> margins;
  for (const auto &name : kLegJointNames) {
    require(model.existJointName(name), "Missing joint: " + name);
    const auto &joint = model.joints[model.getJointId(name)];
    require(joint.nq() == 1 && joint.nv() == 1,
            "Leg joint is not scalar: " + name);
    const double angle = scalar(legs, name, "variables.leg_joints");
    q[joint.idx_q()] = angle;
    const double lower = angle - model.lowerPositionLimit[joint.idx_q()];
    const double upper = model.upperPositionLimit[joint.idx_q()] - angle;
    require(std::isfinite(lower) && std::isfinite(upper) && lower >= 0.0 &&
                upper >= 0.0,
            "Joint limit violated: " + name);
    margins.push_back({name, lower, upper, std::min(lower, upper)});
  }
  const auto wheel_angles = requireNode(variables, "wheel_angles", "variables");
  for (const auto &name : kWheelJointNames) {
    require(model.existJointName(name), "Missing wheel joint: " + name);
    const auto &joint = model.joints[model.getJointId(name)];
    require(joint.nq() == 2 && joint.nv() == 1,
            "Wheel is not a continuous joint: " + name);
    const double theta = scalar(wheel_angles, name, "variables.wheel_angles");
    require(std::abs(theta) <= kMetricTolerance,
            "Wheel angle is not fixed at zero: " + name);
    q.segment<2>(joint.idx_q()) << std::cos(theta), std::sin(theta);
  }
  require(q.allFinite(), "Reconstructed q is not finite");
  const double quaternion_norm = q.segment<4>(3).norm();
  require(std::abs(quaternion_norm - 1.0) <= kNormalizationTolerance &&
              pinocchio::isNormalized(model, q, kNormalizationTolerance),
          "Reconstructed q is not normalized");

  const auto saved_q = requireNode(root, "pinocchio_q_xyzw", "root");
  require(saved_q.IsSequence() &&
              saved_q.size() == static_cast<std::size_t>(model.nq),
          "Complete 27-element Pinocchio q is missing");
  for (Eigen::Index i = 0; i < q.size(); ++i)
    near(q[i], saved_q[static_cast<std::size_t>(i)].as<double>(),
         "pinocchio_q[" + std::to_string(i) + "]");

  pinocchio::Data data(model);
  const Eigen::VectorXd zero = Eigen::VectorXd::Zero(model.nv);
  const Eigen::VectorXd h = pinocchio::rnea(model, data, q, zero, zero);
  const Eigen::Vector3d com = pinocchio::centerOfMass(model, data, q);
  pinocchio::computeJointJacobians(model, data, q);
  pinocchio::updateFramePlacements(model, data);
  require(h.allFinite() && com.allFinite(),
          "Pinocchio returned non-finite values");

  std::map<std::string, Eigen::Vector3d> contacts;
  std::map<std::string, Eigen::MatrixXd> point_jacobians;
  for (const auto &[corner, frame_name] : kWheelFrames) {
    const auto frame_id = bodyFrameId(model, frame_name);
    const auto geometry =
        pose_optimization::parseWheelCollisionGeometry(urdf_model, frame_name);
    const auto &placement = data.oMf[frame_id];
    const Eigen::Vector3d center =
        placement.rotation() * geometry.origin_xyz + placement.translation();
    const Eigen::Vector3d axis =
        placement.rotation() * geometry.cylinder_axis_in_wheel_frame;
    const Eigen::Vector3d contact =
        pose_optimization::treadContactPoint(center, axis, geometry.radius);
    contacts.emplace(corner, contact);

    Eigen::Matrix<double, 6, Eigen::Dynamic> frame_jacobian(6, model.nv);
    frame_jacobian.setZero();
    pinocchio::getFrameJacobian(model, data, frame_id,
                                pinocchio::LOCAL_WORLD_ALIGNED, frame_jacobian);
    const Eigen::Vector3d offset = contact - placement.translation();
    point_jacobians.emplace(corner,
                            frame_jacobian.topRows(3) -
                                skew(offset) * frame_jacobian.bottomRows(3));
  }

  const auto force_node = requireNode(root, "contact_forces_world", "root");
  std::map<std::string, Eigen::Vector3d> forces;
  forces.emplace("FL",
                 vector3(requireNode(force_node, "FL", "contact_forces_world"),
                         "contact_forces_world.FL"));
  forces.emplace("HR",
                 vector3(requireNode(force_node, "HR", "contact_forces_world"),
                         "contact_forces_world.HR"));

  const auto torque_node = requireNode(root, "actuator_torques_nm", "root");
  const auto leg_torque_node =
      requireNode(torque_node, "leg", "actuator_torques_nm");
  const auto wheel_torque_node =
      requireNode(torque_node, "wheel", "actuator_torques_nm");
  std::map<std::string, double> leg_torques;
  std::map<std::string, double> wheel_torques;
  Eigen::VectorXd generalized_torque = Eigen::VectorXd::Zero(model.nv);
  double max_leg_utilization = 0.0;
  double max_wheel_utilization = 0.0;
  for (const auto &name : kLegJointNames) {
    const double torque =
        scalar(leg_torque_node, name, "actuator_torques_nm.leg");
    require(std::abs(torque) <= kLegTorqueLimit + kMetricTolerance,
            "Leg torque limit violated: " + name);
    leg_torques.emplace(name, torque);
    generalized_torque[model.joints[model.getJointId(name)].idx_v()] = torque;
    max_leg_utilization =
        std::max(max_leg_utilization, std::abs(torque) / kLegTorqueLimit);
  }
  for (const auto &name : kWheelJointNames) {
    const double torque =
        scalar(wheel_torque_node, name, "actuator_torques_nm.wheel");
    require(std::abs(torque) <= kWheelTorqueLimit + kMetricTolerance,
            "Wheel torque limit violated: " + name);
    wheel_torques.emplace(name, torque);
    generalized_torque[model.joints[model.getJointId(name)].idx_v()] = torque;
    max_wheel_utilization =
        std::max(max_wheel_utilization, std::abs(torque) / kWheelTorqueLimit);
  }

  Eigen::VectorXd residual = h - generalized_torque;
  residual.noalias() -= point_jacobians.at("FL").transpose() * forces.at("FL");
  residual.noalias() -= point_jacobians.at("HR").transpose() * forces.at("HR");
  require(residual.allFinite(), "Dynamics residual is non-finite");
  const double residual_norm = residual.lpNorm<Eigen::Infinity>();
  require(residual_norm <= kDynamicsTolerance,
          "Full dynamics residual exceeds tolerance");

  const double mu = scalar(requireNode(root, "constraints", "root"),
                           "friction_coefficient", "constraints");
  require(mu > 0.0, "Friction coefficient must be positive");
  auto friction = [mu](const Eigen::Vector3d &force,
                       const std::string &corner) {
    require(force.z() > 0.0, corner + " normal force is not strictly positive");
    require(std::abs(force.x()) <= mu * force.z() + kMetricTolerance &&
                std::abs(force.y()) <= mu * force.z() + kMetricTolerance,
            corner + " friction pyramid is violated");
    const double utilization = force.head<2>().norm() / (mu * force.z());
    require(std::isfinite(utilization) && utilization <= 1.0 + kMetricTolerance,
            corner + " friction utilization exceeds one");
    return utilization;
  };
  const double friction_fl = friction(forces.at("FL"), "FL");
  const double friction_hr = friction(forces.at("HR"), "HR");

  const SupportMetrics support =
      supportMetrics(contacts.at("FL"), contacts.at("HR"), com);
  require(std::abs(contacts.at("FL").z()) <= kGroundTolerance,
          "FL is off ground");
  require(std::abs(contacts.at("HR").z()) <= kGroundTolerance,
          "HR is off ground");
  require(contacts.at("FR").z() >= kSwingClearance,
          "FR clearance is too small");
  require(contacts.at("HL").z() >= kSwingClearance,
          "HL clearance is too small");
  require(std::abs(support.signed_error) <= kComTolerance,
          "CoM line error is too large");
  require(support.s >= kSegmentMargin &&
              support.s <= support.length - kSegmentMargin,
          "CoM projection is outside the supported segment interior");

  const auto minimum =
      std::min_element(margins.begin(), margins.end(),
                       [](const JointMargin &a, const JointMargin &b) {
                         return a.minimum < b.minimum;
                       });
  require(minimum != margins.end() &&
              minimum->minimum >= kJointSafetyMargin - kMetricTolerance,
          "Joint safety margin is violated");

  vectorNear(
      com, vector3(requireNode(root, "com_world_xyz", "root"), "com_world_xyz"),
      "CoM");
  const auto saved_contacts =
      requireNode(root, "physical_tread_contacts_world", "root");
  for (const auto &[corner, contact] : contacts)
    vectorNear(contact,
               vector3(requireNode(saved_contacts, corner,
                                   "physical_tread_contacts_world"),
                       "physical_tread_contacts_world." + corner),
               corner + " tread contact");
  const auto dynamics = requireNode(root, "dynamics", "root");
  require(scalar(dynamics, "equation_count", "dynamics") == model.nv,
          "Saved dynamics equation count is not nv=22");
  const auto saved_residual = requireNode(dynamics, "residual", "dynamics");
  require(saved_residual.IsSequence() &&
              saved_residual.size() == static_cast<std::size_t>(model.nv),
          "Complete 22-element dynamics residual is missing");
  for (Eigen::Index i = 0; i < model.nv; ++i)
    near(residual[i], saved_residual[static_cast<std::size_t>(i)].as<double>(),
         "dynamics.residual[" + std::to_string(i) + "]");
  near(residual_norm, scalar(dynamics, "residual_infinity_norm", "dynamics"),
       "dynamics residual infinity norm");

  const auto metrics = requireNode(root, "metrics", "root");
  near(contacts.at("FL").z(), scalar(metrics, "fl_contact_z_m", "metrics"),
       "FL z");
  near(contacts.at("HR").z(), scalar(metrics, "hr_contact_z_m", "metrics"),
       "HR z");
  near(contacts.at("FR").z(), scalar(metrics, "fr_clearance_m", "metrics"),
       "FR clearance");
  near(contacts.at("HL").z(), scalar(metrics, "hl_clearance_m", "metrics"),
       "HL clearance");
  near(std::abs(support.signed_error),
       scalar(metrics, "com_line_error_m", "metrics"), "CoM line error");
  near(support.s, scalar(metrics, "s_m", "metrics"), "support s");
  near(support.length, scalar(metrics, "support_length_m", "metrics"),
       "support L");
  near(support.s / support.length, scalar(metrics, "s_over_L", "metrics"),
       "s/L");
  near(minimum->minimum,
       scalar(metrics, "minimum_joint_limit_margin_rad", "metrics"),
       "minimum joint margin");
  const auto saved_friction = requireNode(root, "friction_utilization", "root");
  near(friction_fl, scalar(saved_friction, "FL", "friction_utilization"),
       "FL friction utilization");
  near(friction_hr, scalar(saved_friction, "HR", "friction_utilization"),
       "HR friction utilization");
  const auto saved_torque_utilization =
      requireNode(root, "torque_utilization", "root");
  near(max_leg_utilization,
       scalar(saved_torque_utilization, "max_leg", "torque_utilization"),
       "max leg torque utilization");
  near(max_wheel_utilization,
       scalar(saved_torque_utilization, "max_wheel", "torque_utilization"),
       "max wheel torque utilization");

  const auto continuation = requireNode(root, "continuation", "root");
  require(continuation.IsSequence() && continuation.size() == 3,
          "Continuation A/B/C records are incomplete");
  for (std::size_t i = 0; i < 3; ++i) {
    require(continuation[i]["name"].as<std::string>() ==
                    std::string(1, static_cast<char>('A' + i)) &&
                continuation[i]["success"].as<bool>(),
            "Continuation stage record is invalid");
  }

  return {base_position,
          base_rpy,
          com,
          contacts,
          forces,
          leg_torques,
          wheel_torques,
          support,
          margins,
          *minimum,
          residual,
          residual_norm,
          friction_fl,
          friction_hr,
          max_leg_utilization,
          max_wheel_utilization,
          quaternion_norm};
}

void printVector(const Eigen::Vector3d &value) {
  std::cout << '[' << value.x() << ", " << value.y() << ", " << value.z()
            << ']';
}

void printResult(const Result &result) {
  std::cout << std::fixed << std::setprecision(12);
  std::cout << "M-TO2 INDEPENDENT VALIDATION\n\nPASS/FAIL = PASS\n\n";
  std::cout << "CoM = ";
  printVector(result.com);
  std::cout << '\n';
  std::cout << "FL contact = ";
  printVector(result.contacts.at("FL"));
  std::cout << '\n';
  std::cout << "HR contact = ";
  printVector(result.contacts.at("HR"));
  std::cout << '\n';
  std::cout << "FR clearance = " << result.contacts.at("FR").z() << '\n';
  std::cout << "HL clearance = " << result.contacts.at("HL").z() << '\n';
  std::cout << "CoM signed line error = " << result.support.signed_error
            << '\n';
  std::cout << "s/L = " << result.support.s / result.support.length << '\n';
  std::cout << "f_FL = ";
  printVector(result.forces.at("FL"));
  std::cout << '\n';
  std::cout << "f_HR = ";
  printVector(result.forces.at("HR"));
  std::cout << '\n';
  std::cout << "friction utilization FL = " << result.friction_fl << '\n';
  std::cout << "friction utilization HR = " << result.friction_hr << '\n';
  std::cout << "max leg torque utilization = " << result.max_leg_utilization
            << '\n';
  std::cout << "max wheel torque utilization = " << result.max_wheel_utilization
            << '\n';
  std::cout << "minimum joint-limit margin = " << result.minimum_margin.minimum
            << " (" << result.minimum_margin.name << ")\n";
  std::cout << "quaternion norm = " << result.quaternion_norm << '\n';
  std::cout << std::scientific << "full dynamics residual infinity norm = "
            << result.residual_infinity_norm << '\n';
  std::cout
      << "saved metrics match = PASS\nall recomputed values finite = PASS\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc <= 2, "Usage: validate_static_equilibrium [saved_pose.yaml]");
    const std::filesystem::path yaml_path =
        argc == 2 ? std::filesystem::path(argv[1])
                  : std::filesystem::path(VQR_STATIC_OUTPUT_PATH);
    require(std::filesystem::exists(yaml_path),
            "Saved M-TO2 YAML does not exist: " + yaml_path.string());
    printResult(validate(yaml_path));
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "M-TO2 INDEPENDENT VALIDATION\n\nPASS/FAIL = FAIL\n";
    std::cerr << "Issue = " << error.what() << '\n';
    return 1;
  }
}
