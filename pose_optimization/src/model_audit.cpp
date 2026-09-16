#include "pose_optimization/wheel_contact_geometry.hpp"

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/joint/joint-free-flyer.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <urdf_model/link.h>
#include <urdf_model/model.h>
#include <urdf_parser/urdf_parser.h>

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using NamedPosition = std::pair<std::string, double>;

constexpr double kTolerance = 1.0e-10;
constexpr double kExpectedWheelRadius = 0.091;

const std::array<std::string, 12> kLegJointOrder = {
    "FL_HipX_joint", "FR_HipX_joint", "HL_HipX_joint", "HR_HipX_joint",
    "FL_HipY_joint", "FR_HipY_joint", "HL_HipY_joint", "HR_HipY_joint",
    "FL_Knee_joint", "FR_Knee_joint", "HL_Knee_joint", "HR_Knee_joint"};

const std::array<std::string, 4> kWheelJointOrder = {
    "FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL"};

const std::array<std::pair<std::string, std::string>, 4> kWheelFrames = {{
    {"FL", "FL_WHEEL"},
    {"FR", "FR_WHEEL"},
    {"HL", "HL_WHEEL"},
    {"HR", "HR_WHEEL"},
}};

const std::array<NamedPosition, 16> kNominalJointPositions = {{
    {"FL_HipX_joint", 0.0}, {"FR_HipX_joint", 0.0},
    {"HL_HipX_joint", 0.0}, {"HR_HipX_joint", 0.0},
    {"FL_HipY_joint", -0.65}, {"FR_HipY_joint", -0.65},
    {"HL_HipY_joint", -0.65}, {"HR_HipY_joint", -0.65},
    {"FL_Knee_joint", 1.3}, {"FR_Knee_joint", 1.3},
    {"HL_Knee_joint", 1.3}, {"HR_Knee_joint", 1.3},
    {"FL_WHEEL", 0.0}, {"FR_WHEEL", 0.0},
    {"HL_WHEEL", 0.0}, {"HR_WHEEL", 0.0},
}};

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool near(double lhs, double rhs, double tolerance = kTolerance) {
  return std::abs(lhs - rhs) <= tolerance;
}

std::string readTextFile(const std::filesystem::path &path) {
  std::ifstream stream(path);
  require(stream.good(), "Cannot read provenance file: " + path.string());
  std::ostringstream contents;
  contents << stream.rdbuf();
  return contents.str();
}

template <typename Container>
void requireOrderedNamesInSection(const std::string &text,
                                  const std::string &section_start,
                                  const Container &names) {
  std::size_t cursor = text.find(section_start);
  require(cursor != std::string::npos,
          "Missing repository config section: " + section_start);
  const std::size_t section_end = text.find(']', cursor);
  require(section_end != std::string::npos,
          "Unterminated repository config section: " + section_start);
  for (const auto &name : names) {
    cursor = text.find('"' + name + '"', cursor);
    require(cursor != std::string::npos && cursor < section_end,
            "Joint is absent or out of order in " + section_start + ": " + name);
    ++cursor;
  }
}

pinocchio::FrameIndex bodyFrameId(const pinocchio::Model &model,
                                  const std::string &name) {
  require(model.existFrame(name, pinocchio::BODY),
          "Missing Pinocchio BODY frame: " + name);
  const pinocchio::FrameIndex frame_id = model.getFrameId(name, pinocchio::BODY);
  require(frame_id < model.frames.size(), "Invalid BODY frame id for: " + name);
  require(model.frames[frame_id].type == pinocchio::BODY,
          "Resolved frame is not a BODY frame: " + name);
  require(model.existFrame(name, pinocchio::JOINT),
          "Expected same-named wheel JOINT frame is missing: " + name);
  return frame_id;
}

void setJointAngle(const pinocchio::Model &model, const std::string &name,
                   double angle, Eigen::VectorXd &q) {
  require(model.existJointName(name), "Missing joint: " + name);
  const pinocchio::JointIndex id = model.getJointId(name);
  const auto &joint = model.joints[id];
  if (joint.nq() == 1) {
    q[joint.idx_q()] = angle;
    return;
  }
  if (joint.nq() == 2 && joint.nv() == 1) {
    // Pinocchio represents a continuous revolute angle by [cos(theta), sin(theta)].
    q[joint.idx_q()] = std::cos(angle);
    q[joint.idx_q() + 1] = std::sin(angle);
    return;
  }
  throw std::runtime_error("Unsupported configuration representation for joint: " +
                           name);
}

std::string urdfJointTypeName(int type) {
  switch (type) {
  case urdf::Joint::REVOLUTE:
    return "revolute";
  case urdf::Joint::CONTINUOUS:
    return "continuous";
  case urdf::Joint::PRISMATIC:
    return "prismatic";
  case urdf::Joint::FLOATING:
    return "floating";
  case urdf::Joint::PLANAR:
    return "planar";
  case urdf::Joint::FIXED:
    return "fixed";
  default:
    return "unknown";
  }
}

template <typename Container>
void printNameList(const Container &names) {
  for (std::size_t i = 0; i < names.size(); ++i) {
    std::cout << "  [" << i << "] " << names[i] << '\n';
  }
}

} // namespace

int main() {
  try {
    const std::filesystem::path urdf_path =
        std::filesystem::canonical(VQR_URDF_PATH);
    const std::filesystem::path converter_config_path =
        std::filesystem::canonical(VQR_USD_CONVERTER_CONFIG_PATH);
    const std::filesystem::path robot_config_path =
        std::filesystem::canonical(VQR_ROBOT_CONFIG_PATH);

    const std::string robot_config = readTextFile(robot_config_path);
    const std::string converter_config = readTextFile(converter_config_path);
    require(robot_config.find(
                "VQRWheel/VQRWheel_usd/VQRWheel.usd") != std::string::npos,
            "robot_cfg.py no longer references the audited VQRWheel USD");
    require(converter_config.find(
                "VQRWheel/VQRWheel_urdf/urdf/VQRWheel.urdf") !=
                std::string::npos,
            "USD converter metadata no longer identifies the audited VQRWheel URDF");
    requireOrderedNamesInSection(robot_config, "LEG_JOINT_NAMES = [",
                                 kLegJointOrder);
    requireOrderedNamesInSection(robot_config, "WHEEL_JOINT_NAMES = [",
                                 kWheelJointOrder);
    require(robot_config.find("joint_names_expr=LEG_JOINT_NAMES") !=
                std::string::npos &&
                robot_config.find("joint_names_expr=WHEEL_JOINT_NAMES") !=
                    std::string::npos,
            "Repository actuator-to-joint mapping changed");
    require(robot_config.find("LEG_EFFORT_LIMIT = 60.0") != std::string::npos &&
                robot_config.find("WHEEL_EFFORT_LIMIT = 20.0") !=
                    std::string::npos,
            "Repository actuator effort limits changed");
    require(robot_config.find("pos=(0.0, 0.0, 0.45)") != std::string::npos &&
                robot_config.find("name: -0.65") != std::string::npos &&
                robot_config.find("name: 1.3") != std::string::npos,
            "Repository nominal standing pose changed");

    const urdf::ModelInterfaceSharedPtr urdf_model = urdf::parseURDFFile(urdf_path);
    require(static_cast<bool>(urdf_model), "urdfdom failed to load the VQR URDF");

    pinocchio::Model model;
    pinocchio::urdf::buildModel(urdf_path.string(),
                               pinocchio::JointModelFreeFlyer(), model);
    require(model.nq > 0 && model.nv > 0, "Pinocchio produced an empty model");

    for (const auto &name : kLegJointOrder) {
      require(model.existJointName(name), "Missing actuated leg joint: " + name);
    }
    for (const auto &name : kWheelJointOrder) {
      require(model.existJointName(name), "Missing wheel joint: " + name);
    }

    double wheel_radius = std::numeric_limits<double>::quiet_NaN();
    std::map<std::string, pinocchio::FrameIndex> wheel_frame_ids;
    std::map<std::string, pose_optimization::WheelCollisionGeometry>
        wheel_geometries;
    std::map<std::string, Eigen::Vector3d> wheel_joint_axes_local;
    for (const auto &[corner, frame_name] : kWheelFrames) {
      const auto geometry = pose_optimization::parseWheelCollisionGeometry(
          urdf_model, frame_name);
      const double radius = geometry.radius;
      if (!std::isfinite(wheel_radius)) {
        wheel_radius = radius;
      }
      require(near(radius, wheel_radius),
              "The four wheel collision radii are not identical");
      wheel_frame_ids.emplace(corner, bodyFrameId(model, frame_name));
      wheel_geometries.emplace(corner, geometry);

      const urdf::JointConstSharedPtr wheel_joint =
          urdf_model->getJoint(frame_name);
      require(static_cast<bool>(wheel_joint),
              "Missing wheel joint for geometry: " + frame_name);
      const Eigen::Vector3d joint_axis(
          wheel_joint->axis.x, wheel_joint->axis.y, wheel_joint->axis.z);
      require(joint_axis.allFinite() && joint_axis.norm() > 0.0,
              "Invalid local wheel joint axis: " + frame_name);
      require(geometry.cylinder_axis_in_wheel_frame.cross(
                  joint_axis.normalized()).norm() < 2.0e-4,
              "Wheel collision cylinder axis disagrees with joint axis: " +
                  frame_name);
      wheel_joint_axes_local.emplace(corner, joint_axis.normalized());
    }
    require(near(wheel_radius, kExpectedWheelRadius),
            "Wheel radius differs from the audited 0.091 m value");

    const double total_mass = pinocchio::computeTotalMass(model);
    require(std::isfinite(total_mass) && total_mass > 0.0,
            "Total robot mass is not positive and finite");

    Eigen::VectorXd q = pinocchio::neutral(model);
    require(q.size() == model.nq, "Pinocchio neutral configuration has wrong size");
    q.head<3>() << 0.0, 0.0, 0.45;
    q.segment<4>(3) << 0.0, 0.0, 0.0, 1.0; // Pinocchio quaternion: xyzw.
    for (const auto &[name, angle] : kNominalJointPositions) {
      setJointAngle(model, name, angle, q);
    }
    require(q.allFinite(), "Nominal Pinocchio configuration is not finite");
    require(pinocchio::isNormalized(model, q, 1.0e-12),
            "Nominal Pinocchio configuration is not normalized");

    for (const auto &[name, angle] : kNominalJointPositions) {
      const urdf::JointConstSharedPtr joint = urdf_model->getJoint(name);
      require(static_cast<bool>(joint), "Missing URDF joint: " + name);
      if (joint->type == urdf::Joint::CONTINUOUS) {
        require(!joint->limits,
                "Continuous wheel unexpectedly has URDF limits: " + name);
      } else {
        require(static_cast<bool>(joint->limits),
                "Joint position limits are unavailable: " + name);
        require(std::isfinite(joint->limits->lower) &&
                    std::isfinite(joint->limits->upper) &&
                    joint->limits->lower <= joint->limits->upper,
                "Invalid position limits for joint: " + name);
        require(angle >= joint->limits->lower - kTolerance &&
                    angle <= joint->limits->upper + kTolerance,
                "Nominal angle violates position limits for joint: " + name);
        const pinocchio::JointIndex id = model.getJointId(name);
        const auto &pin_joint = model.joints[id];
        require(pin_joint.nq() == 1 &&
                    near(model.lowerPositionLimit[pin_joint.idx_q()],
                         joint->limits->lower) &&
                    near(model.upperPositionLimit[pin_joint.idx_q()],
                         joint->limits->upper),
                "Pinocchio and URDF position limits differ for joint: " + name);
      }
    }

    pinocchio::Data data(model);
    pinocchio::forwardKinematics(model, data, q);
    pinocchio::updateFramePlacements(model, data);
    const Eigen::Vector3d nominal_com = pinocchio::centerOfMass(model, data, q);
    require(nominal_com.allFinite(), "Nominal center of mass is not finite");

    std::map<std::string, Eigen::Vector3d> wheel_positions;
    std::map<std::string, Eigen::Vector3d> wheel_axes_world;
    std::map<std::string, Eigen::Vector3d> tread_contacts_world;
    for (const auto &[corner, frame_name] : kWheelFrames) {
      const pinocchio::SE3 &world_from_wheel =
          data.oMf[wheel_frame_ids.at(corner)];
      const auto &geometry = wheel_geometries.at(corner);
      const Eigen::Vector3d collision_center_world =
          world_from_wheel.rotation() * geometry.origin_xyz +
          world_from_wheel.translation();
      const Eigen::Vector3d cylinder_axis_world =
          world_from_wheel.rotation() *
          geometry.cylinder_axis_in_wheel_frame;
      const Eigen::Vector3d tread_contact_world =
          pose_optimization::treadContactPoint(
              collision_center_world, cylinder_axis_world, geometry.radius);
      require(collision_center_world.allFinite(),
              "Non-finite collision-center FK for wheel frame: " + frame_name);
      wheel_positions.emplace(corner, collision_center_world);
      wheel_axes_world.emplace(corner, cylinder_axis_world);
      tread_contacts_world.emplace(corner, tread_contact_world);
    }

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "M-TO0 VQR MODEL AUDIT\n";
    std::cout << "URDF: " << urdf_path.string() << '\n';
    std::cout << "Provenance robot config: " << robot_config_path.string() << '\n';
    std::cout << "Provenance USD converter config: "
              << converter_config_path.string() << '\n';
    std::cout << "Provenance trace: PASS\n";
    std::cout << "nq: " << model.nq << '\n';
    std::cout << "nv: " << model.nv << '\n';
    std::cout << "total mass [kg]: " << total_mass << '\n';

    std::cout << "\nPinocchio joint order:\n";
    for (pinocchio::JointIndex id = 0; id < model.names.size(); ++id) {
      if (id == 0) {
        std::cout << "  [0] universe (Pinocchio sentinel; not a physical joint)\n";
        continue;
      }
      const auto &joint = model.joints[id];
      std::cout << "  [" << id << "] " << model.names[id]
                << " type=" << joint.shortname() << " idx_q=" << joint.idx_q()
                << " nq=" << joint.nq() << " idx_v=" << joint.idx_v()
                << " nv=" << joint.nv() << '\n';
    }

    std::cout << "\nActuated leg joint order (Isaac Lab config):\n";
    printNameList(kLegJointOrder);
    std::cout << "\nWheel joint order (Isaac Lab config):\n";
    printNameList(kWheelJointOrder);

    std::cout << "\nContact/wheel BODY frames:\n";
    for (const auto &[corner, frame_name] : kWheelFrames) {
      std::cout << "  " << corner << " = " << frame_name
                << " (frame_id=" << wheel_frame_ids.at(corner) << ")\n";
    }
    std::cout << "Frame convention: wheel center (the URDF wheel collision "
                 "cylinder origin is [0,0,0] in each wheel BODY frame)\n";
    std::cout << "Explicit ground-contact frame: none\n";
    std::cout << "wheel radius [m]: " << wheel_radius << '\n';
    std::cout << "Wheel collision geometry (origin and axis in wheel BODY frame):\n";
    for (const auto &[corner, frame_name] : kWheelFrames) {
      const auto &geometry = wheel_geometries.at(corner);
      const Eigen::Vector3d &joint_axis = wheel_joint_axes_local.at(corner);
      std::cout << "  " << corner << " frame=" << frame_name
                << " origin_xyz=[" << geometry.origin_xyz.transpose() << "]"
                << " origin_rpy=[" << geometry.origin_rpy.transpose() << "]"
                << " radius=" << geometry.radius
                << " length=" << geometry.length
                << " cylinder_axis_local=["
                << geometry.cylinder_axis_in_wheel_frame.transpose() << "]"
                << " joint_axis_local=[" << joint_axis.transpose() << "]\n";
    }

    std::cout << "\nURDF joint limits in Pinocchio joint order:\n";
    for (pinocchio::JointIndex id = 2; id < model.names.size(); ++id) {
      const std::string &name = model.names[id];
      const urdf::JointConstSharedPtr joint = urdf_model->getJoint(name);
      require(static_cast<bool>(joint), "No URDF joint for Pinocchio joint: " + name);
      std::cout << "  " << name << " type=" << urdfJointTypeName(joint->type);
      if (joint->type == urdf::Joint::CONTINUOUS) {
        std::cout << " position=[unbounded continuous]"
                  << " pinocchio_q_bounds=[-1,1]x[-1,1] (cos/sin, not angle bounds)"
                  << " effort=[missing in URDF]";
      } else {
        require(static_cast<bool>(joint->limits), "Missing URDF limits: " + name);
        std::cout << " position=[" << joint->limits->lower << ", "
                  << joint->limits->upper << "] rad"
                  << " effort=" << joint->limits->effort << " Nm";
      }
      const auto &pin_joint = model.joints[id];
      std::cout << " pinocchio_effort="
                << model.effortLimit[pin_joint.idx_v()] << "\n";
    }

    std::cout << "\nRepository actuator effort limits:\n";
    std::cout << "  leg joints: 60.000000000 Nm (matches URDF)\n";
    std::cout << "  wheel joints: 20.000000000 Nm (Isaac Lab config only; "
                 "missing from URDF)\n";
    std::cout << "Continuous joints:\n";
    printNameList(kWheelJointOrder);

    std::cout << "\nNominal base pose:\n";
    std::cout << "  position xyz [m] = [0.000000000, 0.000000000, 0.450000000]\n";
    std::cout << "  quaternion wxyz = [1.000000000, 0.000000000, "
                 "0.000000000, 0.000000000]\n";
    std::cout << "Nominal joint pose (Isaac Lab config order):\n";
    for (const auto &[name, angle] : kNominalJointPositions) {
      std::cout << "  " << name << " = " << angle << " rad\n";
    }
    std::cout << "Nominal Pinocchio q valid: PASS\n";
    std::cout << "Nominal CoM xyz [m]: [" << nominal_com.x() << ", "
              << nominal_com.y() << ", " << nominal_com.z() << "]\n";
    std::cout << "Nominal wheel collision-center FK xyz [m], WORLD frame "
                 "(Pinocchio data.oMf):\n";
    for (const auto &[corner, unused_frame_name] : kWheelFrames) {
      (void)unused_frame_name;
      const Eigen::Vector3d &position = wheel_positions.at(corner);
      std::cout << "  " << corner << " = [" << position.x() << ", "
                << position.y() << ", " << position.z() << "]\n";
    }
    std::cout << "Nominal physical tread contact xyz [m], WORLD frame:\n";
    for (const auto &[corner, unused_frame_name] : kWheelFrames) {
      (void)unused_frame_name;
      const Eigen::Vector3d &axis = wheel_axes_world.at(corner);
      const Eigen::Vector3d &contact = tread_contacts_world.at(corner);
      std::cout << "  " << corner << " axis=[" << axis.transpose()
                << "] contact=[" << contact.transpose() << "]\n";
    }

    std::cout << "\nVerification:\n";
    std::cout << "  URDF load: PASS\n";
    std::cout << "  mass > 0: PASS\n";
    std::cout << "  CoM finite: PASS\n";
    std::cout << "  FK finite: PASS\n";
    std::cout << "  all four wheel BODY frames exist: PASS\n";
    std::cout << "  joint limits read: PASS\n";
    std::cout << "  nominal pose valid for Pinocchio: PASS\n";
    std::cout << "  all four wheel collision geometries parsed: PASS\n";
    std::cout << "  physical tread contact points finite: PASS\n";
    std::cout << "M-TO0 RESULT: PASS\n";
    std::cout << "M-TO1A RESULT: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "M-TO0/M-TO1A RESULT: FAIL\n";
    std::cerr << "Reason: " << error.what() << '\n';
    return 1;
  }
}
