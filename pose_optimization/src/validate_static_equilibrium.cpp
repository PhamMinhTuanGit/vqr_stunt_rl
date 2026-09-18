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
// Slack for limit comparisons of a max-iter iterate that sits exactly on an
// active bound (Python physics guard uses the same micron-scale slack).
constexpr double kLimitSlack = 2.0e-6;
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

// Equal joint values produce a point-symmetric pose in the audited URDF, so
// the FL/HR and FR/HL equal-value pairs define joint-space point symmetry.
const std::array<std::pair<std::string, std::string>, 6> kJointSymmetryPairs = {{
    {"FL_HipX_joint", "HR_HipX_joint"},
    {"FR_HipX_joint", "HL_HipX_joint"},
    {"FL_HipY_joint", "HR_HipY_joint"},
    {"FR_HipY_joint", "HL_HipY_joint"},
    {"FL_Knee_joint", "HR_Knee_joint"},
    {"FR_Knee_joint", "HL_Knee_joint"},
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
  std::string milestone;
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
  double wheel_body_angle_fl;
  double wheel_body_angle_hr;
  double wheel_parallel_angle;
  double wheel_vertical_fl;
  double wheel_vertical_hr;
  Eigen::Vector3d support_fl_body;
  Eigen::Vector3d support_hr_body;
  Eigen::Vector3d saved_support_fl_body;
  Eigen::Vector3d saved_support_hr_body;
  Eigen::Vector3d saved_swing_fr_body;
  Eigen::Vector3d saved_swing_hl_body;
  Eigen::Vector3d com_body;
  SupportMetrics support_body;
  double support_distance_3d_world;
  double support_distance_3d_body;
  double support_dx_body;
  double support_dy_body;
  double support_angle_body_x;
  double support_angle_body_y;
  double midpoint_error;
  double load_split_fraction;
  double swing_fr_tuck_radius;
  double swing_hl_tuck_radius;
  double stance_residual_x;
  double stance_residual_y;
  double swing_residual_x;
  double swing_residual_y;
  double swing_height_difference;
  double axle_line_angle_fl;
  double axle_line_angle_hr;
  double joint_symmetry_max_abs;
  Eigen::Vector3d swing_fr_body;
  Eigen::Vector3d swing_hl_body;
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

Eigen::Vector2d vector2(const YAML::Node &node, const std::string &path) {
  require(node.IsSequence() && node.size() == 2,
          "Expected two-vector: " + path);
  Eigen::Vector2d value(node[0].as<double>(), node[1].as<double>());
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
  const std::string milestone =
      requireNode(root, "milestone", "root").as<std::string>();
  const bool sideways = milestone == "M-TO2S";
  const bool aligned = milestone == "M-TO2R" || sideways;
  require(milestone == "M-TO2" || aligned,
          "Saved result is neither M-TO2, M-TO2R, nor M-TO2S");
  require(requireNode(root, "result", "root").as<std::string>() == "PASS",
          "Saved static-equilibrium result is not PASS");
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
  std::map<std::string, double> angles;
  for (const auto &name : kLegJointNames) {
    require(model.existJointName(name), "Missing joint: " + name);
    const auto &joint = model.joints[model.getJointId(name)];
    require(joint.nq() == 1 && joint.nv() == 1,
            "Leg joint is not scalar: " + name);
    const double angle = scalar(legs, name, "variables.leg_joints");
    q[joint.idx_q()] = angle;
    angles.emplace(name, angle);
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
  std::map<std::string, Eigen::Vector3d> wheel_axes_world;
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
    wheel_axes_world.emplace(corner, axis.normalized());

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

  const auto constraints = requireNode(root, "constraints", "root");
  const double mu = scalar(constraints, "friction_coefficient", "constraints");
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
  const double swing_clearance =
      sideways ? scalar(constraints, "swing_clearance_m", "constraints")
               : kSwingClearance;
  require(swing_clearance >= kSwingClearance - kMetricTolerance,
          "Invalid saved swing-clearance constraint");
  require(contacts.at("FR").z() >= swing_clearance - kMetricTolerance,
          "FR clearance is too small");
  require(contacts.at("HL").z() >= swing_clearance - kMetricTolerance,
          "HL clearance is too small");
  const double com_tolerance =
      scalar(constraints, "com_tolerance_m", "constraints");
  require(com_tolerance > 0.0 &&
              (!aligned || com_tolerance <= 0.010 + kMetricTolerance),
          "Invalid saved CoM tolerance");
  require(std::abs(support.signed_error) <= com_tolerance + kMetricTolerance,
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

  double wheel_body_angle_fl = 0.0;
  double wheel_body_angle_hr = 0.0;
  double wheel_parallel_angle = 0.0;
  double wheel_vertical_fl = 0.0;
  double wheel_vertical_hr = 0.0;
  Eigen::Vector3d support_fl_body = Eigen::Vector3d::Zero();
  Eigen::Vector3d support_hr_body = Eigen::Vector3d::Zero();
  Eigen::Vector3d saved_support_fl_body = Eigen::Vector3d::Zero();
  Eigen::Vector3d saved_support_hr_body = Eigen::Vector3d::Zero();
  Eigen::Vector3d saved_swing_fr_body = Eigen::Vector3d::Zero();
  Eigen::Vector3d saved_swing_hl_body = Eigen::Vector3d::Zero();
  Eigen::Vector3d com_body = Eigen::Vector3d::Zero();
  SupportMetrics support_body{0.0, 0.0, 0.0};
  double support_distance_3d_world = 0.0;
  double support_distance_3d_body = 0.0;
  double support_dx_body = 0.0;
  double support_dy_body = 0.0;
  double support_angle_body_x = 0.0;
  double support_angle_body_y = 0.0;
  double midpoint_error = 0.0;
  double load_split_fraction = 0.0;
  double swing_fr_tuck_radius = 0.0;
  double swing_hl_tuck_radius = 0.0;
  double stance_residual_x = 0.0;
  double stance_residual_y = 0.0;
  double swing_residual_x = 0.0;
  double swing_residual_y = 0.0;
  double swing_height_difference = 0.0;
  double axle_line_angle_fl = 0.0;
  double axle_line_angle_hr = 0.0;
  double joint_symmetry_max_abs = 0.0;
  Eigen::Vector3d swing_fr_body = Eigen::Vector3d::Zero();
  Eigen::Vector3d swing_hl_body = Eigen::Vector3d::Zero();
  if (aligned) {
    const Eigen::Matrix3d world_from_body = quaternion.toRotationMatrix();
    const Eigen::Vector3d lateral = Eigen::Vector3d::UnitY();
    const Eigen::Vector3d fl_body =
        world_from_body.transpose() * wheel_axes_world.at("FL");
    const Eigen::Vector3d hr_body =
        world_from_body.transpose() * wheel_axes_world.at("HR");
    auto signInvariantAngleDegrees = [](const Eigen::Vector3d &first,
                                        const Eigen::Vector3d &second) {
      const double cosine = std::clamp(std::abs(first.dot(second)), 0.0, 1.0);
      return std::acos(cosine) * 180.0 / std::acos(-1.0);
    };
    wheel_body_angle_fl = signInvariantAngleDegrees(fl_body, lateral);
    wheel_body_angle_hr = signInvariantAngleDegrees(hr_body, lateral);
    wheel_parallel_angle = signInvariantAngleDegrees(wheel_axes_world.at("FL"),
                                                     wheel_axes_world.at("HR"));
    wheel_vertical_fl = std::abs(wheel_axes_world.at("FL").z());
    wheel_vertical_hr = std::abs(wheel_axes_world.at("HR").z());

    const double body_limit =
        scalar(constraints, "wheel_body_alignment_limit_deg", "constraints");
    const double parallel_limit =
        scalar(constraints, "stance_wheel_parallel_limit_deg", "constraints");
    const double horizontal_limit =
        scalar(constraints, "wheel_axle_horizontal_limit_deg", "constraints");
    const double vertical_limit =
        std::sin(horizontal_limit * std::acos(-1.0) / 180.0);
    require(wheel_body_angle_fl <= body_limit + kMetricTolerance &&
                wheel_body_angle_hr <= body_limit + kMetricTolerance,
            "Independent wheel-to-body alignment check failed");
    require(wheel_parallel_angle <= parallel_limit + kMetricTolerance,
            "Independent FL-HR wheel-axis parallel check failed");
    require(wheel_vertical_fl <= vertical_limit + kMetricTolerance &&
                wheel_vertical_hr <= vertical_limit + kMetricTolerance,
            "Independent wheel-axle horizontal check failed");

    const auto saved_axes = requireNode(root, "wheel_axes", "root");
    const auto saved_world = requireNode(saved_axes, "world", "wheel_axes");
    const auto saved_body = requireNode(saved_axes, "body", "wheel_axes");
    vectorNear(wheel_axes_world.at("FL"),
               vector3(requireNode(saved_world, "FL", "wheel_axes.world"),
                       "wheel_axes.world.FL"),
               "FL WORLD wheel axis");
    vectorNear(wheel_axes_world.at("HR"),
               vector3(requireNode(saved_world, "HR", "wheel_axes.world"),
                       "wheel_axes.world.HR"),
               "HR WORLD wheel axis");
    vectorNear(fl_body,
               vector3(requireNode(saved_body, "FL", "wheel_axes.body"),
                       "wheel_axes.body.FL"),
               "FL body wheel axis");
    vectorNear(hr_body,
               vector3(requireNode(saved_body, "HR", "wheel_axes.body"),
                       "wheel_axes.body.HR"),
               "HR body wheel axis");
    const auto saved_angles =
        requireNode(saved_axes, "body_alignment_angle_deg", "wheel_axes");
    near(wheel_body_angle_fl,
         scalar(saved_angles, "FL", "wheel_axes.body_alignment_angle_deg"),
         "FL wheel-to-body angle");
    near(wheel_body_angle_hr,
         scalar(saved_angles, "HR", "wheel_axes.body_alignment_angle_deg"),
         "HR wheel-to-body angle");
    near(wheel_parallel_angle,
         scalar(saved_axes, "fl_hr_parallel_angle_deg", "wheel_axes"),
         "FL-HR wheel-axis angle");
    const auto saved_vertical =
        requireNode(saved_axes, "vertical_component_abs", "wheel_axes");
    near(wheel_vertical_fl,
         scalar(saved_vertical, "FL", "wheel_axes.vertical_component_abs"),
         "FL wheel-axis vertical component");
    near(wheel_vertical_hr,
         scalar(saved_vertical, "HR", "wheel_axes.vertical_component_abs"),
         "HR wheel-axis vertical component");
  }

  if (sideways) {
    const Eigen::Matrix3d world_from_body = quaternion.toRotationMatrix();
    support_fl_body =
        world_from_body.transpose() * (contacts.at("FL") - base_position);
    support_hr_body =
        world_from_body.transpose() * (contacts.at("HR") - base_position);
    com_body = world_from_body.transpose() * (com - base_position);
    support_body = supportMetrics(support_fl_body, support_hr_body, com_body);
    support_distance_3d_world =
        (contacts.at("HR") - contacts.at("FL")).norm();
    support_distance_3d_body = (support_hr_body - support_fl_body).norm();
    near(support_distance_3d_world, support_distance_3d_body,
         "WORLD/BODY 3D support length invariance");
    const Eigen::Vector2d delta =
        (support_hr_body - support_fl_body).head<2>();
    const double body_support_length = delta.norm();
    require(std::isfinite(body_support_length) &&
                body_support_length > kNormalizationTolerance,
            "BODY-frame support line is degenerate");
    support_dx_body = delta.x();
    support_dy_body = delta.y();
    const double direction_x = std::clamp(
        std::abs(support_dx_body) / body_support_length, 0.0, 1.0);
    const double direction_y = std::clamp(
        std::abs(support_dy_body) / body_support_length, 0.0, 1.0);
    support_angle_body_x =
        std::acos(direction_x) * 180.0 / std::acos(-1.0);
    support_angle_body_y =
        std::acos(direction_y) * 180.0 / std::acos(-1.0);

    const double dx_limit =
        scalar(constraints, "support_dx_limit_m", "constraints");
    const double y_separation =
        scalar(constraints, "support_y_separation_min_m", "constraints");
    const double direction_limit_deg =
        scalar(constraints, "support_direction_limit_deg", "constraints");
    require(dx_limit >= 0.030 - kMetricTolerance &&
                dx_limit <= 0.100 + kMetricTolerance,
            "Invalid saved BODY-frame dx limit");
    require(std::abs(support_dx_body) <= dx_limit + kLimitSlack,
            "BODY-frame support dx constraint failed");
    require(std::abs(support_dy_body) >= y_separation - kLimitSlack,
            "BODY-frame support y-separation constraint failed");
    require(direction_x <=
                std::sin(direction_limit_deg * std::acos(-1.0) / 180.0) +
                    kLimitSlack,
            "BODY-frame support direction constraint failed");

    const auto saved_contacts_body =
        requireNode(root, "support_contacts_body", "root");
    saved_support_fl_body =
        vector3(requireNode(saved_contacts_body, "FL", "support_contacts_body"),
                "support_contacts_body.FL");
    saved_support_hr_body =
        vector3(requireNode(saved_contacts_body, "HR", "support_contacts_body"),
                "support_contacts_body.HR");
    vectorNear(support_fl_body, saved_support_fl_body,
               "FL BODY support contact");
    vectorNear(support_hr_body, saved_support_hr_body,
               "HR BODY support contact");

    const auto saved_contacts_world =
        requireNode(root, "physical_tread_contacts_world", "root");
    const Eigen::Vector3d saved_fl_world =
        vector3(requireNode(saved_contacts_world, "FL",
                            "physical_tread_contacts_world"),
                "physical_tread_contacts_world.FL");
    const Eigen::Vector3d saved_hr_world =
        vector3(requireNode(saved_contacts_world, "HR",
                            "physical_tread_contacts_world"),
                "physical_tread_contacts_world.HR");
    vectorNear(world_from_body.transpose() * (saved_fl_world - base_position),
               saved_support_fl_body,
               "saved FL WORLD-to-BODY representation");
    vectorNear(world_from_body.transpose() * (saved_hr_world - base_position),
               saved_support_hr_body,
               "saved HR WORLD-to-BODY representation");
    const auto saved_line = requireNode(root, "support_line_body", "root");
    near(support_dx_body, scalar(saved_line, "dx_m", "support_line_body"),
         "BODY support dx");
    near(support_dy_body, scalar(saved_line, "dy_m", "support_line_body"),
         "BODY support dy");
    near(direction_x,
         scalar(saved_line, "direction_x_abs", "support_line_body"),
         "BODY support direction x");
    near(support_angle_body_x,
         scalar(saved_line, "angle_wrt_body_x_deg", "support_line_body"),
         "BODY support angle relative to X");
    near(support_angle_body_y,
         scalar(saved_line, "angle_wrt_body_y_deg", "support_line_body"),
         "BODY support angle relative to Y");

    // Two-wheel inverted-pendulum shaping checks.
    swing_fr_body =
        world_from_body.transpose() * (contacts.at("FR") - base_position);
    swing_hl_body =
        world_from_body.transpose() * (contacts.at("HL") - base_position);
    const auto saved_swing_body =
        requireNode(root, "swing_contacts_body", "root");
    saved_swing_fr_body =
        vector3(requireNode(saved_swing_body, "FR", "swing_contacts_body"),
                "swing_contacts_body.FR");
    saved_swing_hl_body =
        vector3(requireNode(saved_swing_body, "HL", "swing_contacts_body"),
                "swing_contacts_body.HL");
    vectorNear(swing_fr_body, saved_swing_fr_body, "FR BODY swing contact");
    vectorNear(swing_hl_body, saved_swing_hl_body, "HL BODY swing contact");

    const auto saved_symmetry = requireNode(root, "pose_symmetry", "root");
    // Use the WORLD-frame CoM projection like the Python solver; the BODY
    // frame variant matches only when the base is level.
    midpoint_error = support.s - 0.5 * support.length;
    near(midpoint_error,
         scalar(saved_symmetry, "midpoint_error_m", "pose_symmetry"),
         "support-midpoint error");
    const double midpoint_tolerance =
        scalar(constraints, "midpoint_tolerance_m", "constraints");
    require(midpoint_tolerance > 0.0 && midpoint_tolerance <= 0.020 +
                                                   kMetricTolerance,
            "Invalid saved support-midpoint tolerance");
    require(std::abs(midpoint_error) <= midpoint_tolerance + kLimitSlack,
            "CoM is not near the support-segment midpoint");
    const double fz_fl = forces.at("FL").z();
    const double fz_hr = forces.at("HR").z();
    load_split_fraction = fz_fl / (fz_fl + fz_hr);
    near(load_split_fraction,
         scalar(saved_symmetry, "load_split_fraction", "pose_symmetry"),
         "wheel load split fraction");
    require(std::abs(load_split_fraction - 0.5) <= 0.1,
            "Wheel normal loads are not balanced");

    const Eigen::Vector3d support_unit =
        (contacts.at("HR") - contacts.at("FL"))
            .normalized();
    const auto axleAngleDeg = [&support_unit](const Eigen::Vector3d &axis) {
      const double cosine =
          std::clamp(std::abs(axis.dot(support_unit)), 0.0, 1.0);
      return std::acos(cosine) * 180.0 / std::acos(-1.0);
    };
    axle_line_angle_fl = axleAngleDeg(wheel_axes_world.at("FL"));
    axle_line_angle_hr = axleAngleDeg(wheel_axes_world.at("HR"));
    const auto saved_axle =
        requireNode(saved_symmetry, "axle_line_angle_deg", "pose_symmetry");
    near(axle_line_angle_fl,
         scalar(saved_axle, "FL", "pose_symmetry.axle_line_angle_deg"),
         "FL axle-to-support-line angle");
    near(axle_line_angle_hr,
         scalar(saved_axle, "HR", "pose_symmetry.axle_line_angle_deg"),
         "HR axle-to-support-line angle");
    const double axle_limit =
        scalar(constraints, "axle_line_limit_deg", "constraints");
    require(axle_limit > 0.0 && axle_limit <= 20.0 + kMetricTolerance,
            "Invalid saved axle-line limit");
    require(axle_line_angle_fl <= axle_limit + kLimitSlack &&
                axle_line_angle_hr <= axle_limit + kLimitSlack,
            "Stance axle is not parallel to the support line");

    swing_fr_tuck_radius = swing_fr_body.head<2>().norm();
    swing_hl_tuck_radius = swing_hl_body.head<2>().norm();
    const auto saved_tuck =
        requireNode(saved_symmetry, "tuck_radius_m", "pose_symmetry");
    near(swing_fr_tuck_radius,
         scalar(saved_tuck, "FR", "pose_symmetry.tuck_radius_m"),
         "FR swing tuck radius");
    near(swing_hl_tuck_radius,
         scalar(saved_tuck, "HL", "pose_symmetry.tuck_radius_m"),
         "HL swing tuck radius");
    const double tuck_limit =
        scalar(constraints, "tuck_radius_m", "constraints");
    require(tuck_limit > 0.0, "Invalid saved tuck radius");
    require(swing_fr_tuck_radius <= tuck_limit + kLimitSlack &&
                swing_hl_tuck_radius <= tuck_limit + kLimitSlack,
            "A swing foot is not tucked within the saved radius");

    stance_residual_x = support_fl_body.x() + support_hr_body.x();
    stance_residual_y = support_fl_body.y() + support_hr_body.y();
    swing_residual_x = swing_fr_body.x() + swing_hl_body.x();
    swing_residual_y = swing_fr_body.y() + swing_hl_body.y();
    swing_height_difference = swing_fr_body.z() - swing_hl_body.z();
    const auto saved_stance_res = vector2(
        requireNode(saved_symmetry, "stance_residual_body_xy_m",
                    "pose_symmetry"),
        "pose_symmetry.stance_residual_body_xy_m");
    const auto saved_swing_res = vector2(
        requireNode(saved_symmetry, "swing_residual_body_xy_m",
                    "pose_symmetry"),
        "pose_symmetry.swing_residual_body_xy_m");
    near(stance_residual_x, saved_stance_res.x(),
         "stance point-symmetry residual x");
    near(stance_residual_y, saved_stance_res.y(),
         "stance point-symmetry residual y");
    near(swing_residual_x, saved_swing_res.x(),
         "swing point-symmetry residual x");
    near(swing_residual_y, saved_swing_res.y(),
         "swing point-symmetry residual y");
    near(swing_height_difference,
         scalar(saved_symmetry, "swing_height_difference_m", "pose_symmetry"),
         "swing height difference");
    joint_symmetry_max_abs = 0.0;
    for (const auto &[first_name, second_name] : kJointSymmetryPairs) {
      joint_symmetry_max_abs =
          std::max(joint_symmetry_max_abs,
                   std::abs(angles.at(first_name) - angles.at(second_name)));
    }
    near(joint_symmetry_max_abs,
         scalar(saved_symmetry, "joint_symmetry_max_abs_rad", "pose_symmetry"),
         "joint symmetry magnitude");
    const bool symmetry_hard = requireNode(saved_symmetry, "symmetry_hard",
                                           "pose_symmetry")
                                   .as<bool>();
    const double symmetry_tolerance = scalar(
        saved_symmetry, "symmetry_tolerance_m", "pose_symmetry");
    if (symmetry_hard) {
      const std::array<double, 5> residuals = {
          std::abs(stance_residual_x), std::abs(stance_residual_y),
          std::abs(swing_residual_x), std::abs(swing_residual_y),
          std::abs(swing_height_difference)};
      require(*std::max_element(residuals.begin(), residuals.end()) <=
                  symmetry_tolerance + kLimitSlack,
              "Hard point-symmetry constraint failed");
    }
  }

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
  if (sideways) {
    require(continuation.IsSequence() && continuation.size() > 0,
            "M-TO2S continuation records are missing");
    const double selected_eps =
        scalar(constraints, "support_dx_limit_m", "constraints");
    bool selected_attempt_found = false;
    for (std::size_t i = 0; i < continuation.size(); ++i) {
      const auto attempt = continuation[i];
      const double eps = attempt["eps_x_m"].as<double>();
      const double clearance = attempt["swing_clearance_m"].as<double>();
      require(std::isfinite(eps) && eps >= 0.030 - kMetricTolerance &&
                  eps <= 0.100 + kMetricTolerance,
              "Invalid M-TO2S eps_x continuation record");
      require(std::isfinite(clearance) &&
                  clearance >= 0.020 - kMetricTolerance &&
                  clearance <= 0.030 + kMetricTolerance,
              "Invalid M-TO2S clearance continuation record");
      if (attempt["success"].as<bool>() &&
          std::abs(eps - selected_eps) <= kMetricTolerance &&
          std::abs(clearance - swing_clearance) <= kMetricTolerance)
        selected_attempt_found = true;
    }
    require(selected_attempt_found,
            "Selected M-TO2S continuation result is not recorded as success");
  } else if (aligned) {
    require(continuation.IsSequence() && continuation.size() >= 1 &&
                continuation.size() <= 3,
            "M-TO2R CoM continuation records are incomplete");
    const auto final_attempt = continuation[continuation.size() - 1];
    require(final_attempt["success"].as<bool>(),
            "Final M-TO2R continuation attempt did not succeed");
    near(final_attempt["com_tolerance_m"].as<double>(), com_tolerance,
         "selected CoM tolerance");
  } else {
    require(continuation.IsSequence() && continuation.size() == 3,
            "Continuation A/B/C records are incomplete");
    for (std::size_t i = 0; i < 3; ++i) {
      require(continuation[i]["name"].as<std::string>() ==
                      std::string(1, static_cast<char>('A' + i)) &&
                  continuation[i]["success"].as<bool>(),
              "Continuation stage record is invalid");
    }
  }

  return {milestone,
          base_position,
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
          quaternion_norm,
          wheel_body_angle_fl,
          wheel_body_angle_hr,
          wheel_parallel_angle,
          wheel_vertical_fl,
          wheel_vertical_hr,
          support_fl_body,
          support_hr_body,
          saved_support_fl_body,
          saved_support_hr_body,
          saved_swing_fr_body,
          saved_swing_hl_body,
          com_body,
          support_body,
          support_distance_3d_world,
          support_distance_3d_body,
          support_dx_body,
          support_dy_body,
          support_angle_body_x,
          support_angle_body_y,
          midpoint_error,
          load_split_fraction,
          swing_fr_tuck_radius,
          swing_hl_tuck_radius,
          stance_residual_x,
          stance_residual_y,
          swing_residual_x,
          swing_residual_y,
          swing_height_difference,
          axle_line_angle_fl,
          axle_line_angle_hr,
          joint_symmetry_max_abs,
          swing_fr_body,
          swing_hl_body};
}

void printVector(const Eigen::Vector3d &value) {
  std::cout << '[' << value.x() << ", " << value.y() << ", " << value.z()
            << ']';
}

void printResult(const Result &result) {
  std::cout << std::fixed << std::setprecision(12);
  std::cout << result.milestone
            << " INDEPENDENT VALIDATION\n\nPASS/FAIL = PASS\n\n";
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
  if (result.milestone == "M-TO2R" || result.milestone == "M-TO2S") {
    std::cout << "wheel alignment FL vs body [deg] = "
              << result.wheel_body_angle_fl << '\n';
    std::cout << "wheel alignment HR vs body [deg] = "
              << result.wheel_body_angle_hr << '\n';
    std::cout << "FL-HR wheel-axis angle [deg] = "
              << result.wheel_parallel_angle << '\n';
    std::cout << "FL wheel-axis vertical component = "
              << result.wheel_vertical_fl << '\n';
    std::cout << "HR wheel-axis vertical component = "
              << result.wheel_vertical_hr << '\n';
    std::cout << "wheel-alignment constraints = PASS\n";
  }
  if (result.milestone == "M-TO2S") {
    std::cout << "saved FL support contact in BODY = ";
    printVector(result.saved_support_fl_body);
    std::cout << '\n';
    std::cout << "saved HR support contact in BODY = ";
    printVector(result.saved_support_hr_body);
    std::cout << '\n';
    std::cout << "WORLD-to-BODY reconstructed FL = ";
    printVector(result.support_fl_body);
    std::cout << '\n';
    std::cout << "WORLD-to-BODY reconstructed HR = ";
    printVector(result.support_hr_body);
    std::cout << '\n';
    std::cout << "support length from WORLD XY = " << result.support.length
              << '\n';
    std::cout << "support length from BODY XY = "
              << result.support_body.length << '\n';
    std::cout << "support length from WORLD 3D = "
              << result.support_distance_3d_world << '\n';
    std::cout << "support length from BODY 3D = "
              << result.support_distance_3d_body << '\n';
    std::cout << "CoM signed line error from WORLD XY = "
              << result.support.signed_error << '\n';
    std::cout << "CoM signed line error from BODY XY = "
              << result.support_body.signed_error << '\n';
    std::cout << "s/L from WORLD XY = "
              << result.support.s / result.support.length << '\n';
    std::cout << "s/L from BODY XY = "
              << result.support_body.s / result.support_body.length << '\n';
    std::cout << "BODY support dx = " << result.support_dx_body << '\n';
    std::cout << "BODY support dy = " << result.support_dy_body << '\n';
    std::cout << "support angle wrt body X [deg] = "
              << result.support_angle_body_x << '\n';
    std::cout << "support angle wrt body Y [deg] = "
              << result.support_angle_body_y << '\n';
    std::cout << "support-midpoint error [m] = " << result.midpoint_error
              << '\n';
    std::cout << "wheel load split FL/(FL+HR) = " << result.load_split_fraction
              << '\n';
    std::cout << "stance point-symmetry residual = ["
              << result.stance_residual_x << ", " << result.stance_residual_y
              << "]\n";
    std::cout << "swing point-symmetry residual = ["
              << result.swing_residual_x << ", " << result.swing_residual_y
              << "]\n";
    std::cout << "swing height difference [m] = "
              << result.swing_height_difference << '\n';
    std::cout << "FL axle vs support line [deg] = "
              << result.axle_line_angle_fl << '\n';
    std::cout << "HR axle vs support line [deg] = "
              << result.axle_line_angle_hr << '\n';
    std::cout << "FR swing tuck radius [m] = " << result.swing_fr_tuck_radius
              << '\n';
    std::cout << "HL swing tuck radius [m] = " << result.swing_hl_tuck_radius
              << '\n';
    std::cout << "joint symmetry max |delta| [rad] = "
              << result.joint_symmetry_max_abs << '\n';
    std::cout << "sideways-support constraints = PASS\n";
  }
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
