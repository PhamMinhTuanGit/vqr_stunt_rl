#include "pose_optimization/wheel_contact_geometry.hpp"

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
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
constexpr double kMetricTolerance = 1.0e-9;
constexpr double kQuaternionTolerance = 1.0e-12;

const std::array<std::string, 12> kLegJointNames = {
    "FL_HipX_joint", "FR_HipX_joint", "HL_HipX_joint", "HR_HipX_joint",
    "FL_HipY_joint", "FR_HipY_joint", "HL_HipY_joint", "HR_HipY_joint",
    "FL_Knee_joint", "FR_Knee_joint", "HL_Knee_joint", "HR_Knee_joint"};

const std::array<std::string, 4> kWheelJointNames = {
    "FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL"};

const std::array<std::pair<std::string, std::string>, 4> kWheelFrames = {{
    {"FL", "FL_WHEEL"},
    {"FR", "FR_WHEEL"},
    {"HL", "HL_WHEEL"},
    {"HR", "HR_WHEEL"},
}};

struct BaseState {
  Eigen::Vector3d position;
  Eigen::Vector3d rpy;
};

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

struct ValidationResult {
  BaseState base;
  Eigen::VectorXd q;
  Eigen::Vector3d com;
  std::map<std::string, Eigen::Vector3d> contacts;
  std::vector<JointMargin> joint_margins;
  std::map<std::string, Eigen::Vector2d> wheel_representations;
  SupportMetrics support;
  double quaternion_norm;
  JointMargin minimum_joint_margin;
};

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

const YAML::Node requireNode(const YAML::Node &parent, const std::string &key,
                             const std::string &path) {
  const YAML::Node node = parent[key];
  require(node.IsDefined() && !node.IsNull(),
          "Saved YAML is missing required state: " + path + "." + key);
  return node;
}

double readFiniteScalar(const YAML::Node &parent, const std::string &key,
                        const std::string &path) {
  const double value = requireNode(parent, key, path).as<double>();
  require(std::isfinite(value),
          "Saved YAML contains non-finite value: " + path + "." + key);
  return value;
}

Eigen::Vector3d readFiniteVector3(const YAML::Node &node,
                                 const std::string &path) {
  require(node.IsSequence() && node.size() == 3,
          "Saved YAML vector must have length 3: " + path);
  Eigen::Vector3d value;
  for (std::size_t i = 0; i < 3; ++i) {
    value[static_cast<Eigen::Index>(i)] = node[i].as<double>();
  }
  require(value.allFinite(), "Saved YAML vector is non-finite: " + path);
  return value;
}

pinocchio::FrameIndex bodyFrameId(const pinocchio::Model &model,
                                  const std::string &name) {
  require(model.existFrame(name, pinocchio::BODY),
          "Missing Pinocchio BODY frame: " + name);
  const pinocchio::FrameIndex id = model.getFrameId(name, pinocchio::BODY);
  require(id < model.frames.size() && model.frames[id].type == pinocchio::BODY,
          "Invalid Pinocchio BODY frame: " + name);
  return id;
}

Eigen::Quaterniond quaternionFromRpy(const Eigen::Vector3d &rpy) {
  const Eigen::AngleAxisd roll(rpy.x(), Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd pitch(rpy.y(), Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd yaw(rpy.z(), Eigen::Vector3d::UnitZ());
  return Eigen::Quaterniond(yaw * pitch * roll);
}

Eigen::Vector3d rpyFromQuaternion(const Eigen::Quaterniond &quaternion) {
  const double x = quaternion.x();
  const double y = quaternion.y();
  const double z = quaternion.z();
  const double w = quaternion.w();
  const double sin_pitch = std::clamp(2.0 * (w * y - z * x), -1.0, 1.0);
  return {
      std::atan2(2.0 * (w * x + y * z),
                 1.0 - 2.0 * (x * x + y * y)),
      std::asin(sin_pitch),
      std::atan2(2.0 * (w * z + x * y),
                 1.0 - 2.0 * (y * y + z * z)),
  };
}

void requireNear(double recomputed, double saved, const std::string &name,
                 double tolerance = kMetricTolerance) {
  require(std::isfinite(recomputed) && std::isfinite(saved),
          "Non-finite value while comparing " + name);
  require(std::abs(recomputed - saved) <= tolerance,
          name + " mismatch: recomputed=" + std::to_string(recomputed) +
              " saved=" + std::to_string(saved));
}

void requireVectorNear(const Eigen::Vector3d &recomputed,
                       const Eigen::Vector3d &saved,
                       const std::string &name) {
  require(recomputed.allFinite() && saved.allFinite(),
          "Non-finite vector while comparing " + name);
  require((recomputed - saved).cwiseAbs().maxCoeff() <= kMetricTolerance,
          name + " does not match saved YAML");
}

SupportMetrics computeSupportMetrics(const Eigen::Vector3d &fl,
                                     const Eigen::Vector3d &hr,
                                     const Eigen::Vector3d &com) {
  require(fl.allFinite() && hr.allFinite() && com.allFinite(),
          "Support-line inputs are not finite");
  const Eigen::Vector2d delta = hr.head<2>() - fl.head<2>();
  const double length = delta.norm();
  require(std::isfinite(length) && length > 1.0e-12,
          "FL-HR support line has zero or non-finite length");
  const Eigen::Vector2d direction = delta / length;
  const Eigen::Vector2d normal(-direction.y(), direction.x());
  const Eigen::Vector2d offset = com.head<2>() - fl.head<2>();
  return {normal.dot(offset), direction.dot(offset), length};
}

void validateCornerSequence(const YAML::Node &root, const std::string &key,
                            const std::vector<std::string> &expected) {
  const YAML::Node sequence = requireNode(root, key, "root");
  require(sequence.IsSequence() && sequence.size() == expected.size(),
          "Saved YAML has invalid " + key + " set");
  for (std::size_t i = 0; i < expected.size(); ++i) {
    require(sequence[i].as<std::string>() == expected[i],
            "Saved YAML has unexpected " + key + " ordering");
  }
}

ValidationResult validate(const std::filesystem::path &yaml_path) {
  const YAML::Node root = YAML::LoadFile(yaml_path.string());
  require(requireNode(root, "milestone", "root").as<std::string>() == "M-TO1B",
          "Saved pose milestone is not M-TO1B");
  require(requireNode(root, "result", "root").as<std::string>() == "PASS",
          "Saved M-TO1B result is not PASS");
  validateCornerSequence(root, "stance", {"FL", "HR"});
  validateCornerSequence(root, "swing", {"FR", "HL"});

  const std::filesystem::path expected_urdf =
      std::filesystem::canonical(VQR_URDF_PATH);
  const YAML::Node model_node = requireNode(root, "model", "root");
  const std::filesystem::path saved_urdf = std::filesystem::canonical(
      requireNode(model_node, "urdf_path", "model").as<std::string>());
  require(saved_urdf == expected_urdf,
          "Saved YAML does not reference the audited VQR URDF");

  pinocchio::Model model;
  pinocchio::urdf::buildModel(expected_urdf.string(),
                             pinocchio::JointModelFreeFlyer(), model);
  require(model.nq == 27 && model.nv == 22,
          "Audited VQR Pinocchio dimensions changed");
  const urdf::ModelInterfaceSharedPtr urdf_model =
      urdf::parseURDFFile(expected_urdf);
  require(static_cast<bool>(urdf_model), "urdfdom failed to load VQR URDF");

  const YAML::Node variables = requireNode(root, "variables", "root");
  BaseState base{
      {readFiniteScalar(variables, "base_x", "variables"),
       readFiniteScalar(variables, "base_y", "variables"),
       readFiniteScalar(variables, "base_z", "variables")},
      {readFiniteScalar(variables, "base_roll", "variables"),
       readFiniteScalar(variables, "base_pitch", "variables"),
       readFiniteScalar(variables, "base_yaw", "variables")},
  };
  const YAML::Node leg_joints =
      requireNode(variables, "leg_joints", "variables");
  const YAML::Node wheel_angles =
      requireNode(variables, "wheel_angles", "variables");

  Eigen::VectorXd q = pinocchio::neutral(model);
  q.head<3>() = base.position;
  const Eigen::Quaterniond quaternion = quaternionFromRpy(base.rpy);
  q.segment<4>(3) << quaternion.x(), quaternion.y(), quaternion.z(),
      quaternion.w();

  std::vector<JointMargin> joint_margins;
  joint_margins.reserve(kLegJointNames.size());
  for (const std::string &name : kLegJointNames) {
    require(model.existJointName(name), "Missing Pinocchio joint: " + name);
    const auto &joint = model.joints[model.getJointId(name)];
    require(joint.nq() == 1 && joint.nv() == 1,
            "Expected scalar leg joint: " + name);
    const double angle = readFiniteScalar(leg_joints, name, "variables.leg_joints");
    q[joint.idx_q()] = angle;
    const double lower_margin = angle - model.lowerPositionLimit[joint.idx_q()];
    const double upper_margin = model.upperPositionLimit[joint.idx_q()] - angle;
    require(std::isfinite(lower_margin) && std::isfinite(upper_margin),
            "Non-finite joint-limit margin: " + name);
    require(lower_margin >= 0.0 && upper_margin >= 0.0,
            "Joint position violates limits: " + name);
    joint_margins.push_back(
        {name, lower_margin, upper_margin,
         std::min(lower_margin, upper_margin)});
  }

  std::map<std::string, Eigen::Vector2d> wheel_representations;
  for (const std::string &name : kWheelJointNames) {
    require(model.existJointName(name), "Missing Pinocchio wheel joint: " + name);
    const auto &joint = model.joints[model.getJointId(name)];
    require(joint.nq() == 2 && joint.nv() == 1,
            "Expected continuous wheel joint: " + name);
    const double angle =
        readFiniteScalar(wheel_angles, name, "variables.wheel_angles");
    const Eigen::Vector2d representation(std::cos(angle), std::sin(angle));
    require(representation.allFinite() &&
                std::abs(representation.squaredNorm() - 1.0) <=
                    kQuaternionTolerance,
            "Invalid continuous-wheel representation: " + name);
    q.segment<2>(joint.idx_q()) = representation;
    wheel_representations.emplace(name, representation);
  }

  require(q.allFinite(), "Reconstructed Pinocchio q is not finite");
  const double quaternion_norm = q.segment<4>(3).norm();
  require(std::abs(quaternion_norm - 1.0) <= kQuaternionTolerance,
          "Reconstructed floating-base quaternion is not normalized");
  require(pinocchio::isNormalized(model, q, kQuaternionTolerance),
          "Reconstructed Pinocchio configuration is not normalized");

  const YAML::Node saved_q_node =
      requireNode(root, "pinocchio_q_xyzw", "root");
  require(saved_q_node.IsSequence() &&
              saved_q_node.size() == static_cast<std::size_t>(model.nq),
          "Saved YAML is missing the complete 27-element Pinocchio q");
  for (Eigen::Index i = 0; i < q.size(); ++i) {
    requireNear(q[i], saved_q_node[static_cast<std::size_t>(i)].as<double>(),
                "pinocchio_q_xyzw[" + std::to_string(i) + "]");
  }

  pinocchio::Data data(model);
  const Eigen::Vector3d com = pinocchio::centerOfMass(model, data, q);
  pinocchio::updateFramePlacements(model, data);
  require(com.allFinite(), "Recomputed Pinocchio CoM is not finite");

  std::map<std::string, Eigen::Vector3d> contacts;
  for (const auto &[corner, frame_name] : kWheelFrames) {
    const auto frame_id = bodyFrameId(model, frame_name);
    const auto geometry = pose_optimization::parseWheelCollisionGeometry(
        urdf_model, frame_name);
    const pinocchio::SE3 &world_from_wheel = data.oMf[frame_id];
    const Eigen::Vector3d collision_center =
        world_from_wheel.rotation() * geometry.origin_xyz +
        world_from_wheel.translation();
    const Eigen::Vector3d wheel_axis =
        world_from_wheel.rotation() * geometry.cylinder_axis_in_wheel_frame;
    const Eigen::Vector3d contact = pose_optimization::treadContactPoint(
        collision_center, wheel_axis, geometry.radius);
    require(contact.allFinite(), "Non-finite tread contact: " + corner);
    contacts.emplace(corner, contact);
  }

  const SupportMetrics support = computeSupportMetrics(
      contacts.at("FL"), contacts.at("HR"), com);
  require(std::abs(contacts.at("FL").z()) <= kGroundTolerance,
          "FL contact height fails M-TO1C tolerance");
  require(std::abs(contacts.at("HR").z()) <= kGroundTolerance,
          "HR contact height fails M-TO1C tolerance");
  require(contacts.at("FR").z() >= kSwingClearance,
          "FR clearance fails M-TO1C threshold");
  require(contacts.at("HL").z() >= kSwingClearance,
          "HL clearance fails M-TO1C threshold");
  require(std::abs(support.signed_error) <= kComTolerance,
          "CoM line error fails M-TO1C tolerance");
  require(support.s >= kSegmentMargin &&
              support.s <= support.length - kSegmentMargin,
          "CoM projection fails M-TO1C segment bounds");

  const auto minimum_iterator = std::min_element(
      joint_margins.begin(), joint_margins.end(),
      [](const JointMargin &lhs, const JointMargin &rhs) {
        return lhs.minimum < rhs.minimum;
      });
  require(minimum_iterator != joint_margins.end(),
          "No finite leg joint margins were computed");

  const YAML::Node saved_com_node =
      requireNode(root, "com_world_xyz", "root");
  requireVectorNear(com, readFiniteVector3(saved_com_node, "com_world_xyz"),
                    "CoM");
  const YAML::Node saved_contacts =
      requireNode(root, "physical_tread_contacts_world", "root");
  for (const auto &[corner, contact] : contacts) {
    requireVectorNear(
        contact,
        readFiniteVector3(requireNode(saved_contacts, corner,
                                      "physical_tread_contacts_world"),
                          "physical_tread_contacts_world." + corner),
        corner + " tread contact");
  }

  const YAML::Node metrics = requireNode(root, "metrics", "root");
  requireNear(contacts.at("FL").z(),
              readFiniteScalar(metrics, "fl_contact_z_m", "metrics"),
              "FL contact z");
  requireNear(contacts.at("HR").z(),
              readFiniteScalar(metrics, "hr_contact_z_m", "metrics"),
              "HR contact z");
  requireNear(contacts.at("FR").z(),
              readFiniteScalar(metrics, "fr_clearance_m", "metrics"),
              "FR clearance");
  requireNear(contacts.at("HL").z(),
              readFiniteScalar(metrics, "hl_clearance_m", "metrics"),
              "HL clearance");
  requireNear(std::abs(support.signed_error),
              readFiniteScalar(metrics, "com_line_error_m", "metrics"),
              "absolute CoM line error");
  requireNear(support.signed_error,
              readFiniteScalar(metrics, "com_line_signed_error_m", "metrics"),
              "signed CoM line error");
  requireNear(support.s, readFiniteScalar(metrics, "s_m", "metrics"),
              "support coordinate s");
  requireNear(support.length,
              readFiniteScalar(metrics, "support_length_m", "metrics"),
              "support length L");
  requireNear(support.s / support.length,
              readFiniteScalar(metrics, "s_over_L", "metrics"), "s/L");
  requireNear(minimum_iterator->minimum,
              readFiniteScalar(metrics, "minimum_joint_limit_margin_rad",
                               "metrics"),
              "minimum joint-limit margin");

  const Eigen::Quaterniond reconstructed_quaternion(
      q[6], q[3], q[4], q[5]);
  const Eigen::Vector3d recomputed_rpy =
      rpyFromQuaternion(reconstructed_quaternion);
  for (Eigen::Index i = 0; i < 3; ++i) {
    requireNear(recomputed_rpy[i], base.rpy[i],
                std::array<std::string, 3>{"roll", "pitch", "yaw"}[i]);
  }

  return {base,
          q,
          com,
          contacts,
          joint_margins,
          wheel_representations,
          support,
          quaternion_norm,
          *minimum_iterator};
}

void printVector(const Eigen::Vector3d &value) {
  std::cout << '[' << value.x() << ", " << value.y() << ", " << value.z()
            << ']';
}

void printResult(const ValidationResult &result) {
  std::cout << std::fixed << std::setprecision(12);
  std::cout << "M-TO1C RESULT\n\n";
  std::cout << "PASS/FAIL = PASS\n\n";
  std::cout << "CoM = ";
  printVector(result.com);
  std::cout << '\n';
  for (const auto &[corner, contact] : result.contacts) {
    std::cout << corner << " tread contact = ";
    printVector(contact);
    std::cout << '\n';
  }
  std::cout << '\n';
  std::cout << "FL contact z = " << result.contacts.at("FL").z() << '\n';
  std::cout << "HR contact z = " << result.contacts.at("HR").z() << "\n\n";
  std::cout << "FR clearance = " << result.contacts.at("FR").z() << '\n';
  std::cout << "HL clearance = " << result.contacts.at("HL").z() << "\n\n";
  std::cout << "CoM signed line error = " << result.support.signed_error << '\n';
  std::cout << "CoM line error = " << std::abs(result.support.signed_error)
            << '\n';
  std::cout << "s = " << result.support.s << '\n';
  std::cout << "L = " << result.support.length << '\n';
  std::cout << "s/L = " << result.support.s / result.support.length << "\n\n";
  std::cout << "roll = " << result.base.rpy.x() << '\n';
  std::cout << "pitch = " << result.base.rpy.y() << '\n';
  std::cout << "yaw = " << result.base.rpy.z() << "\n\n";
  std::cout << "all joint-limit margins [lower, upper, minimum] =\n";
  for (const JointMargin &margin : result.joint_margins) {
    std::cout << "  " << margin.name << " = [" << margin.lower << ", "
              << margin.upper << ", " << margin.minimum << "]\n";
  }
  for (const std::string &name : kWheelJointNames) {
    std::cout << "  " << name
              << " = unbounded continuous (no angular position limit)\n";
  }
  std::cout << "minimum joint-limit margin = "
            << result.minimum_joint_margin.minimum << '\n';
  std::cout << "joint = " << result.minimum_joint_margin.name << "\n\n";
  std::cout << "quaternion norm = " << result.quaternion_norm << '\n';
  std::cout << "wheel representation checks = PASS\n";
  for (const auto &[name, representation] : result.wheel_representations) {
    std::cout << "  " << name << " [cos(theta), sin(theta)] = ["
              << representation.x() << ", " << representation.y() << "]\n";
  }
  std::cout << "saved state completeness = PASS\n";
  std::cout << "saved metrics match = PASS\n";
  std::cout << "all recomputed values finite = PASS\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc <= 2,
            "Usage: validate_kinematic_equilibrium [saved_pose.yaml]");
    const std::filesystem::path yaml_path =
        argc == 2 ? std::filesystem::path(argv[1])
                  : std::filesystem::path(VQR_KINEMATIC_OUTPUT_PATH);
    require(std::filesystem::exists(yaml_path),
            "Saved M-TO1B YAML does not exist: " + yaml_path.string());
    const ValidationResult result = validate(yaml_path);
    printResult(result);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "M-TO1C RESULT\n\n";
    std::cerr << "PASS/FAIL = FAIL\n";
    std::cerr << "Issue = " << error.what() << '\n';
    return 1;
  }
}
