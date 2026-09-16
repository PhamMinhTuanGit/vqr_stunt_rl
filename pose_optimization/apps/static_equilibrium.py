#!/usr/bin/env python3
"""M-TO2 FL-HR static-dynamics pose optimization."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np
import pinocchio as pin
import yaml

import kinematic_equilibrium as kin


ACTUATOR_NAMES = (*kin.LEG_JOINT_NAMES, *kin.WHEEL_NAMES)
N_POSE = 15
N_TORQUES = 16
N_FORCES = 6
N_VARIABLES = N_POSE + N_TORQUES + N_FORCES
N_STATIC_OUTPUTS = 15 + 22 + 2 * 3 * 22


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


class PinocchioStaticTerms(ca.Callback):
    """Numeric Pinocchio terms exposed to CasADi with central finite differences."""

    def __init__(self, urdf_path: Path, helper: kin.TreadContactLibrary) -> None:
        ca.Callback.__init__(self)
        self.model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
        if self.model.nq != 27 or self.model.nv != 22:
            raise RuntimeError("Audited VQR model dimensions changed")
        self.data = self.model.createData()
        self.helper = helper
        self.frame_ids = {
            corner: self.model.getFrameId(wheel_name, pin.BODY)
            for corner, wheel_name in zip(
                kin.WHEEL_CORNERS, kin.WHEEL_NAMES, strict=True
            )
        }
        if any(frame_id >= len(self.model.frames) for frame_id in self.frame_ids.values()):
            raise RuntimeError("A required wheel BODY frame is missing")
        self.joint_q_indices: dict[str, int] = {}
        self.joint_v_indices: dict[str, int] = {}
        for name in kin.LEG_JOINT_NAMES:
            joint = self.model.joints[self.model.getJointId(name)]
            if joint.nq != 1 or joint.nv != 1:
                raise RuntimeError(f"Leg joint is not scalar: {name}")
            self.joint_q_indices[name] = joint.idx_q
            self.joint_v_indices[name] = joint.idx_v
        self.wheel_q_indices: dict[str, int] = {}
        for name in kin.WHEEL_NAMES:
            joint = self.model.joints[self.model.getJointId(name)]
            if joint.nq != 2 or joint.nv != 1:
                raise RuntimeError(f"Wheel joint is not continuous: {name}")
            self.wheel_q_indices[name] = joint.idx_q
            self.joint_v_indices[name] = joint.idx_v
        self.construct(
            "pinocchio_static_terms",
            {"enable_fd": True, "fd_method": "central", "verbose": False},
        )

    def get_n_in(self) -> int:
        return 1

    def get_n_out(self) -> int:
        return 1

    def get_sparsity_in(self, index: int) -> ca.Sparsity:
        del index
        return ca.Sparsity.dense(N_POSE, 1)

    def get_sparsity_out(self, index: int) -> ca.Sparsity:
        del index
        return ca.Sparsity.dense(N_STATIC_OUTPUTS, 1)

    def make_q(self, pose: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        if pose.size != N_POSE:
            raise ValueError(f"Expected {N_POSE} pose variables, got {pose.size}")
        q = pin.neutral(self.model)
        q[:3] = (0.0, 0.0, pose[0])
        half_roll = 0.5 * pose[1]
        half_pitch = 0.5 * pose[2]
        q[3:7] = (
            math.sin(half_roll) * math.cos(half_pitch),
            math.cos(half_roll) * math.sin(half_pitch),
            -math.sin(half_roll) * math.sin(half_pitch),
            math.cos(half_roll) * math.cos(half_pitch),
        )
        for offset, name in enumerate(kin.LEG_JOINT_NAMES):
            q[self.joint_q_indices[name]] = pose[3 + offset]
        for name in kin.WHEEL_NAMES:
            q_index = self.wheel_q_indices[name]
            q[q_index : q_index + 2] = (1.0, 0.0)
        return q

    def evaluate_numpy(
        self, pose: np.ndarray
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        q = self.make_q(pose)
        zero_velocity = np.zeros(self.model.nv)
        h = np.asarray(
            pin.rnea(self.model, self.data, q, zero_velocity, zero_velocity),
            dtype=np.float64,
        ).copy()
        com = np.asarray(pin.centerOfMass(self.model, self.data, q), dtype=np.float64)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        contacts = np.empty((4, 3), dtype=np.float64)
        point_jacobians: dict[str, np.ndarray] = {}
        for index, corner in enumerate(kin.WHEEL_CORNERS):
            frame_id = self.frame_ids[corner]
            placement = self.data.oMf[frame_id]
            origin, local_axis, radius = self.helper.geometries[corner]
            frame_translation = np.asarray(placement.translation)
            center = np.asarray(placement.rotation @ origin + frame_translation)
            world_axis = np.asarray(placement.rotation @ local_axis)
            contact = self.helper.contact(center, world_axis, radius)
            contacts[index] = contact
            if corner in ("FL", "HR"):
                frame_jacobian = np.asarray(
                    pin.getFrameJacobian(
                        self.model,
                        self.data,
                        frame_id,
                        pin.LOCAL_WORLD_ALIGNED,
                    ),
                    dtype=np.float64,
                )
                offset = contact - frame_translation
                point_jacobians[corner] = (
                    frame_jacobian[:3, :]
                    - skew(offset) @ frame_jacobian[3:, :]
                )
        j_fl = point_jacobians["FL"]
        j_hr = point_jacobians["HR"]
        arrays = (contacts, com, q, h, j_fl, j_hr)
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise RuntimeError("Pinocchio static terms contain non-finite values")
        return arrays

    def eval(self, arguments: list[ca.DM]) -> list[ca.DM]:
        contacts, com, _, h, j_fl, j_hr = self.evaluate_numpy(
            np.asarray(arguments[0])
        )
        output = np.concatenate(
            (
                contacts.reshape(-1),
                com,
                h,
                j_fl.reshape(-1, order="F"),
                j_hr.reshape(-1, order="F"),
            )
        )
        return [ca.DM(output)]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_inputs(config_path: Path, initial_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_yaml(config_path)
    initial = load_yaml(initial_path)
    if initial["milestone"] != "M-TO1B" or initial["result"] != "PASS":
        raise RuntimeError("M-TO2 initial guess is not an M-TO1B PASS pose")
    if initial["stance"] != ["FL", "HR"] or initial["swing"] != ["FR", "HL"]:
        raise RuntimeError("M-TO1 initial guess has unexpected contact sets")
    static_config = config["static_optimization"]
    if [stage["name"] for stage in static_config["continuation"]] != ["A", "B", "C"]:
        raise RuntimeError("Expected M-TO2 continuation stages A-C")
    return config, initial


def pose_from_saved(saved: dict[str, Any]) -> np.ndarray:
    variables = saved["variables"]
    pose = np.array(
        [
            variables["base_z"],
            variables["base_roll"],
            variables["base_pitch"],
            *(variables["leg_joints"][name] for name in kin.LEG_JOINT_NAMES),
        ],
        dtype=np.float64,
    )
    if pose.size != N_POSE or not np.all(np.isfinite(pose)):
        raise RuntimeError("M-TO1 initial pose is incomplete or non-finite")
    return pose


def actuator_matrix(evaluator: PinocchioStaticTerms) -> np.ndarray:
    matrix = np.zeros((evaluator.model.nv, N_TORQUES), dtype=np.float64)
    for column, name in enumerate(ACTUATOR_NAMES):
        matrix[evaluator.joint_v_indices[name], column] = 1.0
    if np.count_nonzero(matrix) != N_TORQUES:
        raise RuntimeError("Actuator mapping is not one-to-one")
    if np.any(np.count_nonzero(matrix, axis=1) > 1):
        raise RuntimeError("Two actuators map to the same velocity row")
    return matrix


def variable_bounds(
    evaluator: PinocchioStaticTerms,
    static_config: dict[str, Any],
    stage: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(N_VARIABLES, -np.inf)
    upper = np.full(N_VARIABLES, np.inf)
    orientation_limit = math.radians(static_config["orientation_limit_deg"])
    lower[1:3] = -orientation_limit
    upper[1:3] = orientation_limit
    margin = static_config["joint_margin_rad"]
    for offset, name in enumerate(kin.LEG_JOINT_NAMES):
        q_index = evaluator.joint_q_indices[name]
        lower[3 + offset] = evaluator.model.lowerPositionLimit[q_index] + margin
        upper[3 + offset] = evaluator.model.upperPositionLimit[q_index] - margin
    if stage["torque_limits"]:
        leg_limit = static_config["torque_limits_nm"]["leg"]
        wheel_limit = static_config["torque_limits_nm"]["wheel"]
        lower[N_POSE : N_POSE + 12] = -leg_limit
        upper[N_POSE : N_POSE + 12] = leg_limit
        lower[N_POSE + 12 : N_POSE + N_TORQUES] = -wheel_limit
        upper[N_POSE + 12 : N_POSE + N_TORQUES] = wheel_limit
    if stage["positive_normal"]:
        lower[N_POSE + N_TORQUES + 2] = 0.0
        lower[N_POSE + N_TORQUES + 5] = 0.0
    return lower, upper


def initial_guess(
    evaluator: PinocchioStaticTerms,
    pose_reference: np.ndarray,
    selection: np.ndarray,
) -> np.ndarray:
    contacts, _, _, h, j_fl, j_hr = evaluator.evaluate_numpy(pose_reference)
    del contacts
    half_weight = 0.5 * pin.computeTotalMass(evaluator.model) * 9.81
    f_fl = np.array((0.0, 0.0, half_weight))
    f_hr = np.array((0.0, 0.0, half_weight))
    unactuated_residual = h - j_fl.T @ f_fl - j_hr.T @ f_hr
    tau = selection.T @ unactuated_residual
    return np.concatenate((pose_reference, tau, f_fl, f_hr))


def build_solver(
    evaluator: PinocchioStaticTerms,
    config: dict[str, Any],
    pose_reference: np.ndarray,
    selection: np.ndarray,
) -> ca.Function:
    static_config = config["static_optimization"]
    weights = static_config["weights"]
    variables = ca.MX.sym("x", N_VARIABLES)
    pose = variables[:N_POSE]
    torques = variables[N_POSE : N_POSE + N_TORQUES]
    f_fl = variables[N_POSE + N_TORQUES : N_POSE + N_TORQUES + 3]
    f_hr = variables[N_POSE + N_TORQUES + 3 :]
    terms = evaluator(pose)
    fl = terms[0:3]
    fr = terms[3:6]
    hl = terms[6:9]
    hr = terms[9:12]
    com = terms[12:15]
    h = terms[15:37]
    j_fl = ca.reshape(terms[37:103], 3, 22)
    j_hr = ca.reshape(terms[103:169], 3, 22)
    dynamics = (
        h
        - ca.DM(selection) @ torques
        - j_fl.T @ f_fl
        - j_hr.T @ f_hr
    )

    support_delta = hr[:2] - fl[:2]
    support_length = ca.norm_2(support_delta)
    com_offset = com[:2] - fl[:2]
    line_error = (
        -support_delta[1] * com_offset[0]
        + support_delta[0] * com_offset[1]
    ) / support_length
    segment_s = ca.dot(support_delta, com_offset) / support_length
    mu = static_config["friction_coefficient"]
    friction_margins = ca.vertcat(
        mu * f_fl[2] - f_fl[0],
        mu * f_fl[2] + f_fl[0],
        mu * f_fl[2] - f_fl[1],
        mu * f_fl[2] + f_fl[1],
        mu * f_hr[2] - f_hr[0],
        mu * f_hr[2] + f_hr[0],
        mu * f_hr[2] - f_hr[1],
        mu * f_hr[2] + f_hr[1],
    )
    constraints = ca.vertcat(
        dynamics,
        fl[2],
        fr[2],
        hl[2],
        hr[2],
        line_error,
        segment_s,
        support_length - segment_s,
        friction_margins,
    )
    reference = ca.DM(pose_reference)
    objective = (
        weights["pose"] * ca.sumsqr(pose[3:] - reference[3:])
        + weights["orientation"] * ca.sumsqr(pose[1:3])
        + weights["base_height"] * (pose[0] - reference[0]) ** 2
        + weights["torque"] * ca.sumsqr(torques)
        + weights["force"] * (ca.sumsqr(f_fl) + ca.sumsqr(f_hr))
    )
    ipopt = static_config["ipopt"]
    options = {
        "print_time": False,
        "error_on_fail": False,
        "ipopt.print_level": 5,
        "ipopt.sb": "yes",
        "ipopt.tol": ipopt["tolerance"],
        "ipopt.acceptable_tol": ipopt["acceptable_tolerance"],
        "ipopt.constr_viol_tol": ipopt["constraint_violation_tolerance"],
        "ipopt.bound_relax_factor": 0.0,
        "ipopt.max_iter": ipopt["max_iterations"],
        "ipopt.warm_start_init_point": "yes",
        "ipopt.hessian_approximation": "limited-memory",
    }
    return ca.nlpsol(
        "static_equilibrium",
        "ipopt",
        {"x": variables, "f": objective, "g": constraints},
        options,
    )


def constraint_bounds(
    static_config: dict[str, Any], stage: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(37, -np.inf)
    upper = np.full(37, np.inf)
    lower[:22] = 0.0
    upper[:22] = 0.0
    geometry_start = 22
    lower[geometry_start : geometry_start + 7] = (
        0.0,
        static_config["swing_clearance_m"],
        static_config["swing_clearance_m"],
        0.0,
        -static_config["com_tolerance_m"],
        static_config["segment_margin_m"],
        static_config["segment_margin_m"],
    )
    upper[geometry_start : geometry_start + 7] = (
        0.0,
        np.inf,
        np.inf,
        0.0,
        static_config["com_tolerance_m"],
        np.inf,
        np.inf,
    )
    if stage["friction"]:
        lower[29:] = 0.0
    return lower, upper


def run_continuation(
    evaluator: PinocchioStaticTerms,
    config: dict[str, Any],
    pose_reference: np.ndarray,
) -> list[dict[str, Any]]:
    static_config = config["static_optimization"]
    selection = actuator_matrix(evaluator)
    solver = build_solver(evaluator, config, pose_reference, selection)
    values = initial_guess(evaluator, pose_reference, selection)
    lambda_x = None
    lambda_g = None
    results = []
    for stage in static_config["continuation"]:
        lower_x, upper_x = variable_bounds(evaluator, static_config, stage)
        lower_g, upper_g = constraint_bounds(static_config, stage)
        arguments: dict[str, Any] = {
            "x0": values,
            "lbx": lower_x,
            "ubx": upper_x,
            "lbg": lower_g,
            "ubg": upper_g,
        }
        if lambda_x is not None:
            arguments["lam_x0"] = lambda_x
            arguments["lam_g0"] = lambda_g
        solution = solver(**arguments)
        stats = solver.stats()
        values = np.asarray(solution["x"]).reshape(-1)
        lambda_x = solution["lam_x"]
        lambda_g = solution["lam_g"]
        result = {
            "name": stage["name"],
            "status": stats["return_status"],
            "success": bool(stats["success"]),
            "iterations": int(stats["iter_count"]),
            "friction": bool(stage["friction"]),
            "positive_normal": bool(stage["positive_normal"]),
            "torque_limits": bool(stage["torque_limits"]),
            "objective": float(solution["f"]),
            "x": values.copy(),
        }
        results.append(result)
        print(
            f"Continuation {result['name']}: {result['status']}, "
            f"iterations={result['iterations']}"
        )
        if not result["success"]:
            raise RuntimeError(
                f"Continuation stage {result['name']} failed: {result['status']}"
            )
    return results


def support_metrics(
    contacts: np.ndarray, com: np.ndarray
) -> tuple[float, float, float]:
    delta = contacts[3, :2] - contacts[0, :2]
    length = float(np.linalg.norm(delta))
    direction = delta / length
    normal = np.array((-direction[1], direction[0]))
    offset = com[:2] - contacts[0, :2]
    return float(normal @ offset), float(direction @ offset), length


def evaluate_solution(
    evaluator: PinocchioStaticTerms,
    static_config: dict[str, Any],
    values: np.ndarray,
) -> dict[str, Any]:
    pose = values[:N_POSE]
    torques = values[N_POSE : N_POSE + N_TORQUES]
    f_fl = values[N_POSE + N_TORQUES : N_POSE + N_TORQUES + 3]
    f_hr = values[N_POSE + N_TORQUES + 3 :]
    contacts, com, q, h, j_fl, j_hr = evaluator.evaluate_numpy(pose)
    selection = actuator_matrix(evaluator)
    residual = h - selection @ torques - j_fl.T @ f_fl - j_hr.T @ f_hr
    signed_error, segment_s, support_length = support_metrics(contacts, com)
    mu = static_config["friction_coefficient"]
    friction_fl = float(np.linalg.norm(f_fl[:2]) / (mu * f_fl[2]))
    friction_hr = float(np.linalg.norm(f_hr[:2]) / (mu * f_hr[2]))
    leg_limit = static_config["torque_limits_nm"]["leg"]
    wheel_limit = static_config["torque_limits_nm"]["wheel"]
    torque_utilization = np.concatenate(
        (np.abs(torques[:12]) / leg_limit, np.abs(torques[12:]) / wheel_limit)
    )
    joint_margins = []
    for offset, name in enumerate(kin.LEG_JOINT_NAMES):
        q_index = evaluator.joint_q_indices[name]
        angle = pose[3 + offset]
        joint_margins.append(
            min(
                angle - evaluator.model.lowerPositionLimit[q_index],
                evaluator.model.upperPositionLimit[q_index] - angle,
            )
        )
    result = {
        "pose": pose,
        "torques": torques,
        "f_fl": f_fl,
        "f_hr": f_hr,
        "contacts": contacts,
        "com": com,
        "q": q,
        "h": h,
        "j_fl": j_fl,
        "j_hr": j_hr,
        "residual": residual,
        "residual_inf": float(np.max(np.abs(residual))),
        "signed_error": signed_error,
        "s": segment_s,
        "length": support_length,
        "friction_fl": friction_fl,
        "friction_hr": friction_hr,
        "torque_utilization": torque_utilization,
        "joint_margins": np.asarray(joint_margins),
    }
    if not all(
        np.all(np.isfinite(value))
        for value in result.values()
        if isinstance(value, np.ndarray)
    ):
        raise RuntimeError("Final static solution contains non-finite values")
    if result["residual_inf"] > 1.0e-3:
        raise RuntimeError("Full dynamics residual exceeds 1e-3")
    if abs(contacts[0, 2]) > 1.0e-4 or abs(contacts[3, 2]) > 1.0e-4:
        raise RuntimeError("Stance contact height check failed")
    if contacts[1, 2] < static_config["swing_clearance_m"]:
        raise RuntimeError("FR swing clearance check failed")
    if contacts[2, 2] < static_config["swing_clearance_m"]:
        raise RuntimeError("HL swing clearance check failed")
    if abs(signed_error) > static_config["com_tolerance_m"]:
        raise RuntimeError("CoM line error check failed")
    margin = static_config["segment_margin_m"]
    if not (segment_s >= margin and segment_s <= support_length - margin):
        raise RuntimeError("Support segment check failed")
    if f_fl[2] <= 0.0 or f_hr[2] <= 0.0:
        raise RuntimeError("A final contact normal force is not positive")
    if friction_fl > 1.0 or friction_hr > 1.0:
        raise RuntimeError("Friction utilization exceeds one")
    if np.max(torque_utilization) > 1.0:
        raise RuntimeError("Torque utilization exceeds one")
    if np.min(result["joint_margins"]) < static_config["joint_margin_rad"]:
        raise RuntimeError("Joint safety margin check failed")
    return result


def velocity_order(model: pin.Model) -> list[str]:
    labels = []
    for joint_id in range(1, len(model.names)):
        joint = model.joints[joint_id]
        for local_index in range(joint.nv):
            labels.append(f"{model.names[joint_id]}[{local_index}]")
    if len(labels) != model.nv:
        raise RuntimeError("Could not label all Pinocchio velocity rows")
    return labels


def write_output(
    output_path: Path,
    urdf_path: Path,
    initial_path: Path,
    evaluator: PinocchioStaticTerms,
    config: dict[str, Any],
    stages: list[dict[str, Any]],
) -> dict[str, Any]:
    static_config = config["static_optimization"]
    final = stages[-1]
    result = evaluate_solution(evaluator, static_config, final["x"])
    pose = result["pose"]
    torques = result["torques"]
    contacts = result["contacts"]
    torque_utilization = result["torque_utilization"]
    output = {
        "milestone": "M-TO2",
        "result": "PASS",
        "model": {"urdf_path": str(urdf_path)},
        "initial_guess": str(initial_path),
        "stance": ["FL", "HR"],
        "swing": ["FR", "HL"],
        "static_condition": {"v": [0.0] * 22, "dv": [0.0] * 22},
        "solver": {
            "name": "CasADi 3.7.2 + IPOPT",
            "status": final["status"],
            "iterations": final["iterations"],
            "total_continuation_iterations": sum(
                stage["iterations"] for stage in stages
            ),
        },
        "objective": final["objective"],
        "variables": {
            "base_x": 0.0,
            "base_y": 0.0,
            "base_z": float(pose[0]),
            "base_roll": float(pose[1]),
            "base_pitch": float(pose[2]),
            "base_yaw": 0.0,
            "leg_joints": {
                name: float(pose[3 + offset])
                for offset, name in enumerate(kin.LEG_JOINT_NAMES)
            },
            "wheel_angles": {name: 0.0 for name in kin.WHEEL_NAMES},
        },
        "pinocchio_q_xyzw": result["q"].tolist(),
        "com_world_xyz": result["com"].tolist(),
        "physical_tread_contacts_world": {
            corner: contacts[index].tolist()
            for index, corner in enumerate(kin.WHEEL_CORNERS)
        },
        "swing_clearance_m": {
            "FR": float(contacts[1, 2]),
            "HL": float(contacts[2, 2]),
        },
        "contact_forces_world": {
            "FL": result["f_fl"].tolist(),
            "HR": result["f_hr"].tolist(),
        },
        "actuator_torques_nm": {
            "leg": {
                name: float(torques[offset])
                for offset, name in enumerate(kin.LEG_JOINT_NAMES)
            },
            "wheel": {
                name: float(torques[12 + offset])
                for offset, name in enumerate(kin.WHEEL_NAMES)
            },
        },
        "dynamics": {
            "equation_count": 22,
            "pinocchio_velocity_order": velocity_order(evaluator.model),
            "residual": result["residual"].tolist(),
            "residual_infinity_norm": result["residual_inf"],
            "contact_jacobian_convention": (
                "physical tread material-point Jacobian; world-aligned linear rows"
            ),
        },
        "metrics": {
            "fl_contact_z_m": float(contacts[0, 2]),
            "hr_contact_z_m": float(contacts[3, 2]),
            "fr_clearance_m": float(contacts[1, 2]),
            "hl_clearance_m": float(contacts[2, 2]),
            "com_line_error_m": abs(result["signed_error"]),
            "com_line_signed_error_m": result["signed_error"],
            "s_m": result["s"],
            "support_length_m": result["length"],
            "s_over_L": result["s"] / result["length"],
            "minimum_joint_limit_margin_rad": float(
                np.min(result["joint_margins"])
            ),
        },
        "friction_utilization": {
            "FL": result["friction_fl"],
            "HR": result["friction_hr"],
        },
        "torque_utilization": {
            "leg": {
                name: float(torque_utilization[offset])
                for offset, name in enumerate(kin.LEG_JOINT_NAMES)
            },
            "wheel": {
                name: float(torque_utilization[12 + offset])
                for offset, name in enumerate(kin.WHEEL_NAMES)
            },
            "max_leg": float(np.max(torque_utilization[:12])),
            "max_wheel": float(np.max(torque_utilization[12:])),
        },
        "constraints": {
            "friction_coefficient": static_config["friction_coefficient"],
            "leg_torque_limit_nm": static_config["torque_limits_nm"]["leg"],
            "wheel_torque_limit_nm": static_config["torque_limits_nm"]["wheel"],
            "wheel_torque_limit_provenance": (
                "Isaac Lab actuator config; not URDF/hardware verified"
            ),
            "joint_margin_rad": static_config["joint_margin_rad"],
            "orientation_limit_deg": static_config["orientation_limit_deg"],
            "swing_clearance_m": static_config["swing_clearance_m"],
            "com_tolerance_m": static_config["com_tolerance_m"],
            "segment_margin_m": static_config["segment_margin_m"],
        },
        "continuation": [
            {key: value for key, value in stage.items() if key != "x"}
            for stage in stages
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(output, stream, sort_keys=False)
    return output


def print_vector(vector: list[float] | np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.12f}" for value in vector) + "]"


def print_result(output: dict[str, Any]) -> None:
    variables = output["variables"]
    print("\nM-TO2 RESULT\n")
    print("PASS/FAIL = PASS\n")
    print(f"Solver status = {output['solver']['status']}")
    print(f"iterations = {output['solver']['iterations']}\n")
    print("Optimized base:")
    print(f"z = {variables['base_z']:.12f}")
    print(f"roll = {variables['base_roll']:.12f}")
    print(f"pitch = {variables['base_pitch']:.12f}\n")
    print("Optimized joints:")
    for name, value in variables["leg_joints"].items():
        print(f"  {name} = {value:.12f}")
    print(f"\nCoM = {print_vector(output['com_world_xyz'])}")
    print(f"line error = {output['metrics']['com_line_error_m']:.12f}")
    print(f"s/L = {output['metrics']['s_over_L']:.12f}\n")
    print("Contacts:")
    print(f"FL = {print_vector(output['physical_tread_contacts_world']['FL'])}")
    print(f"HR = {print_vector(output['physical_tread_contacts_world']['HR'])}\n")
    print("Swing clearance:")
    print(f"FR = {output['swing_clearance_m']['FR']:.12f}")
    print(f"HL = {output['swing_clearance_m']['HL']:.12f}\n")
    print("Contact forces:")
    print(f"FL = {print_vector(output['contact_forces_world']['FL'])}")
    print(f"HR = {print_vector(output['contact_forces_world']['HR'])}\n")
    print("Leg torques:")
    for name, value in output["actuator_torques_nm"]["leg"].items():
        print(f"  {name} = {value:.12f}")
    print("\nWheel torques:")
    for name, value in output["actuator_torques_nm"]["wheel"].items():
        print(f"  {name} = {value:.12f}")
    print(
        "\nmax leg torque utilization = "
        f"{output['torque_utilization']['max_leg']:.12f}"
    )
    print(
        "max wheel torque utilization = "
        f"{output['torque_utilization']['max_wheel']:.12f}\n"
    )
    print("friction utilization:")
    print(f"FL = {output['friction_utilization']['FL']:.12f}")
    print(f"HR = {output['friction_utilization']['HR']:.12f}\n")
    print(
        "full dynamics residual infinity norm = "
        f"{output['dynamics']['residual_infinity_norm']:.12e}\n"
    )
    print("Continuation:")
    for stage in output["continuation"]:
        print(
            f"{stage['name']} = {stage['status']} "
            f"(iterations={stage['iterations']})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(os.environ["VQR_CONFIG_PATH"]))
    parser.add_argument("--urdf", type=Path, default=Path(os.environ["VQR_URDF_PATH"]))
    parser.add_argument(
        "--initial", type=Path, default=Path(os.environ["VQR_MTO1_OUTPUT_PATH"])
    )
    parser.add_argument(
        "--output", type=Path, default=Path(os.environ["VQR_STATIC_OUTPUT_PATH"])
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    config, initial = load_inputs(arguments.config.resolve(), arguments.initial.resolve())
    helper = kin.TreadContactLibrary(
        Path(os.environ["VQR_TREAD_CONTACT_LIBRARY"]).resolve(),
        arguments.urdf.resolve(),
    )
    evaluator = PinocchioStaticTerms(arguments.urdf.resolve(), helper)
    pose_reference = pose_from_saved(initial)
    stages = run_continuation(evaluator, config, pose_reference)
    output = write_output(
        arguments.output.resolve(),
        arguments.urdf.resolve(),
        arguments.initial.resolve(),
        evaluator,
        config,
        stages,
    )
    print_result(output)
    print(f"\nOutput = {arguments.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
