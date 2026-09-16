#!/usr/bin/env python3
"""M-TO1B FL-HR two-wheel kinematic equilibrium solve and verification."""

from __future__ import annotations

import argparse
import ctypes
import math
import os
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np
import pinocchio as pin
import yaml


LEG_JOINT_NAMES = (
    "FL_HipX_joint",
    "FR_HipX_joint",
    "HL_HipX_joint",
    "HR_HipX_joint",
    "FL_HipY_joint",
    "FR_HipY_joint",
    "HL_HipY_joint",
    "HR_HipY_joint",
    "FL_Knee_joint",
    "FR_Knee_joint",
    "HL_Knee_joint",
    "HR_Knee_joint",
)
WHEEL_NAMES = ("FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL")
WHEEL_CORNERS = ("FL", "FR", "HL", "HR")
N_VARIABLES = 15
N_GEOMETRY_OUTPUTS = 15
REPORT_TOLERANCE = 2.0e-8
GROUND_TOLERANCE = 1.0e-9


class TreadContactLibrary:
    """ctypes bridge that calls the exact audited M-TO1A C++ helper."""

    def __init__(self, library_path: Path, urdf_path: Path) -> None:
        self._library = ctypes.CDLL(str(library_path))
        double_pointer = ctypes.POINTER(ctypes.c_double)
        self._library.vqr_wheel_contact_last_error.restype = ctypes.c_char_p
        self._library.vqr_parse_wheel_collision.argtypes = (
            ctypes.c_char_p,
            ctypes.c_char_p,
            double_pointer,
            double_pointer,
            double_pointer,
            double_pointer,
        )
        self._library.vqr_parse_wheel_collision.restype = ctypes.c_int
        self._library.vqr_tread_contact_point.argtypes = (
            double_pointer,
            double_pointer,
            ctypes.c_double,
            double_pointer,
        )
        self._library.vqr_tread_contact_point.restype = ctypes.c_int
        self.geometries: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        for corner, wheel_name in zip(WHEEL_CORNERS, WHEEL_NAMES, strict=True):
            origin = np.empty(3, dtype=np.float64)
            axis = np.empty(3, dtype=np.float64)
            radius = ctypes.c_double()
            length = ctypes.c_double()
            code = self._library.vqr_parse_wheel_collision(
                os.fsencode(urdf_path),
                wheel_name.encode(),
                origin.ctypes.data_as(double_pointer),
                axis.ctypes.data_as(double_pointer),
                ctypes.byref(radius),
                ctypes.byref(length),
            )
            self._check(code)
            self.geometries[corner] = (origin, axis, radius.value)

    def _check(self, code: int) -> None:
        if code:
            message = self._library.vqr_wheel_contact_last_error()
            raise RuntimeError(message.decode() if message else "C++ helper failed")

    def contact(self, center: np.ndarray, axis: np.ndarray, radius: float) -> np.ndarray:
        center = np.ascontiguousarray(center, dtype=np.float64)
        axis = np.ascontiguousarray(axis, dtype=np.float64)
        output = np.empty(3, dtype=np.float64)
        pointer = ctypes.POINTER(ctypes.c_double)
        code = self._library.vqr_tread_contact_point(
            center.ctypes.data_as(pointer),
            axis.ctypes.data_as(pointer),
            radius,
            output.ctypes.data_as(pointer),
        )
        self._check(code)
        return output


class PinocchioGeometry(ca.Callback):
    """CasADi callback backed by Pinocchio FK/CoM and the M-TO1A helper."""

    def __init__(self, urdf_path: Path, helper: TreadContactLibrary) -> None:
        ca.Callback.__init__(self)
        self.model = pin.buildModelFromUrdf(str(urdf_path), pin.JointModelFreeFlyer())
        if self.model.nq != 27 or self.model.nv != 22:
            raise RuntimeError("Audited VQR model dimensions changed")
        self.data = self.model.createData()
        self.helper = helper
        self.frame_ids = {
            corner: self.model.getFrameId(wheel_name, pin.BODY)
            for corner, wheel_name in zip(WHEEL_CORNERS, WHEEL_NAMES, strict=True)
        }
        if any(frame_id >= len(self.model.frames) for frame_id in self.frame_ids.values()):
            raise RuntimeError("A required wheel BODY frame is missing")
        self.joint_q_indices = {}
        for name in LEG_JOINT_NAMES:
            joint = self.model.joints[self.model.getJointId(name)]
            if joint.nq != 1 or joint.nv != 1:
                raise RuntimeError(f"Leg joint is not scalar: {name}")
            self.joint_q_indices[name] = joint.idx_q
        self.wheel_q_indices = {}
        for name in WHEEL_NAMES:
            joint = self.model.joints[self.model.getJointId(name)]
            if joint.nq != 2 or joint.nv != 1:
                raise RuntimeError(f"Wheel joint is not continuous: {name}")
            self.wheel_q_indices[name] = joint.idx_q
        self.construct(
            "pinocchio_geometry",
            {"enable_fd": True, "fd_method": "central", "verbose": False},
        )

    def get_n_in(self) -> int:
        return 1

    def get_n_out(self) -> int:
        return 1

    def get_sparsity_in(self, index: int) -> ca.Sparsity:
        del index
        return ca.Sparsity.dense(N_VARIABLES, 1)

    def get_sparsity_out(self, index: int) -> ca.Sparsity:
        del index
        return ca.Sparsity.dense(N_GEOMETRY_OUTPUTS, 1)

    def make_q(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size != N_VARIABLES:
            raise ValueError(f"Expected {N_VARIABLES} variables, got {values.size}")
        q = pin.neutral(self.model)
        q[:3] = (0.0, 0.0, values[0])
        half_roll = 0.5 * values[1]
        half_pitch = 0.5 * values[2]
        q[3:7] = (
            math.sin(half_roll) * math.cos(half_pitch),
            math.cos(half_roll) * math.sin(half_pitch),
            -math.sin(half_roll) * math.sin(half_pitch),
            math.cos(half_roll) * math.cos(half_pitch),
        )
        for offset, name in enumerate(LEG_JOINT_NAMES):
            q[self.joint_q_indices[name]] = values[3 + offset]
        for name in WHEEL_NAMES:
            q_index = self.wheel_q_indices[name]
            q[q_index : q_index + 2] = (1.0, 0.0)
        return q

    def evaluate_numpy(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = self.make_q(values)
        com = np.asarray(pin.centerOfMass(self.model, self.data, q), dtype=np.float64)
        pin.updateFramePlacements(self.model, self.data)
        contacts = np.empty((4, 3), dtype=np.float64)
        for index, corner in enumerate(WHEEL_CORNERS):
            placement = self.data.oMf[self.frame_ids[corner]]
            origin, local_axis, radius = self.helper.geometries[corner]
            center = np.asarray(placement.rotation @ origin + placement.translation)
            world_axis = np.asarray(placement.rotation @ local_axis)
            contacts[index] = self.helper.contact(center, world_axis, radius)
        if not np.all(np.isfinite(contacts)) or not np.all(np.isfinite(com)):
            raise RuntimeError("Pinocchio geometry evaluation is not finite")
        return contacts, com, q

    def eval(self, arguments: list[ca.DM]) -> list[ca.DM]:
        contacts, com, _ = self.evaluate_numpy(np.asarray(arguments[0]))
        return [ca.DM(np.concatenate((contacts.reshape(-1), com)))]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["milestone"] != "M-TO1B":
        raise RuntimeError("Configuration milestone is not M-TO1B")
    stages = config["optimization"]["continuation"]
    if [stage["name"] for stage in stages] != ["A", "B", "C", "D"]:
        raise RuntimeError("Expected continuation stages A-D")
    return config


def support_metrics(contacts: np.ndarray, com: np.ndarray) -> tuple[float, float, float]:
    delta = contacts[3, :2] - contacts[0, :2]
    length = float(np.linalg.norm(delta))
    if not math.isfinite(length) or length <= 1.0e-12:
        raise RuntimeError("FL-HR support line has zero length")
    direction = delta / length
    normal = np.array((-direction[1], direction[0]))
    offset = com[:2] - contacts[0, :2]
    return float(normal @ offset), float(direction @ offset), length


def initial_guess(config: dict[str, Any]) -> np.ndarray:
    nominal = config["nominal"]["joint_positions"]
    values = np.zeros(N_VARIABLES, dtype=np.float64)
    values[0] = config["optimization"]["base_height_reference_m"]
    values[3:] = [nominal[name] for name in LEG_JOINT_NAMES]
    return values


def variable_bounds(
    evaluator: PinocchioGeometry, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    optimization = config["optimization"]
    lower = np.full(N_VARIABLES, -np.inf)
    upper = np.full(N_VARIABLES, np.inf)
    orientation_limit = math.radians(optimization["orientation_limit_deg"])
    lower[1:3] = -orientation_limit
    upper[1:3] = orientation_limit
    margin = optimization["joint_margin_rad"]
    for offset, name in enumerate(LEG_JOINT_NAMES):
        q_index = evaluator.joint_q_indices[name]
        lower[3 + offset] = evaluator.model.lowerPositionLimit[q_index] + margin
        upper[3 + offset] = evaluator.model.upperPositionLimit[q_index] - margin
        if lower[3 + offset] > upper[3 + offset]:
            raise RuntimeError(f"Joint margin empties bounds for {name}")
    return lower, upper


def objective_value(values: np.ndarray, config: dict[str, Any]) -> float:
    optimization = config["optimization"]
    weights = optimization["weights"]
    nominal = np.array(
        [config["nominal"]["joint_positions"][name] for name in LEG_JOINT_NAMES]
    )
    return float(
        weights["pose"] * np.sum(np.square(values[3:] - nominal))
        + weights["orientation"] * np.sum(np.square(values[1:3]))
        + weights["base_height"]
        * (values[0] - optimization["base_height_reference_m"]) ** 2
    )


def minimum_joint_margin(evaluator: PinocchioGeometry, values: np.ndarray) -> float:
    margins = []
    for offset, name in enumerate(LEG_JOINT_NAMES):
        q_index = evaluator.joint_q_indices[name]
        angle = values[3 + offset]
        margins.extend(
            (
                angle - evaluator.model.lowerPositionLimit[q_index],
                evaluator.model.upperPositionLimit[q_index] - angle,
            )
        )
    return float(min(margins))


def build_solver(
    evaluator: PinocchioGeometry, config: dict[str, Any]
) -> ca.Function:
    optimization = config["optimization"]
    weights = optimization["weights"]
    nominal = ca.DM(
        [config["nominal"]["joint_positions"][name] for name in LEG_JOINT_NAMES]
    )
    values = ca.MX.sym("x", N_VARIABLES)
    geometry = evaluator(values)
    fl = geometry[0:3]
    fr = geometry[3:6]
    hl = geometry[6:9]
    hr = geometry[9:12]
    com = geometry[12:15]
    delta = hr[:2] - fl[:2]
    length = ca.norm_2(delta)
    offset = com[:2] - fl[:2]
    line_error = (-delta[1] * offset[0] + delta[0] * offset[1]) / length
    segment_s = ca.dot(delta, offset) / length
    objective = (
        weights["pose"] * ca.sumsqr(values[3:] - nominal)
        + weights["orientation"] * ca.sumsqr(values[1:3])
        + weights["base_height"]
        * (values[0] - optimization["base_height_reference_m"]) ** 2
    )
    constraints = ca.vertcat(
        fl[2], fr[2], hl[2], hr[2], line_error, segment_s, length - segment_s
    )
    ipopt = optimization["ipopt"]
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
        "kinematic_equilibrium",
        "ipopt",
        {"x": values, "f": objective, "g": constraints},
        options,
    )


def run_continuation(
    evaluator: PinocchioGeometry, config: dict[str, Any]
) -> list[dict[str, Any]]:
    solver = build_solver(evaluator, config)
    lower_x, upper_x = variable_bounds(evaluator, config)
    x = initial_guess(config)
    lambda_x = None
    lambda_g = None
    segment_margin = config["optimization"]["segment_margin_m"]
    results = []
    for stage in config["optimization"]["continuation"]:
        arguments: dict[str, Any] = {
            "x0": x,
            "lbx": lower_x,
            "ubx": upper_x,
            "lbg": (
                0.0,
                stage["swing_clearance_m"],
                stage["swing_clearance_m"],
                0.0,
                -stage["com_tolerance_m"],
                segment_margin,
                segment_margin,
            ),
            "ubg": (
                0.0,
                np.inf,
                np.inf,
                0.0,
                stage["com_tolerance_m"],
                np.inf,
                np.inf,
            ),
        }
        if lambda_x is not None:
            arguments["lam_x0"] = lambda_x
            arguments["lam_g0"] = lambda_g
        solution = solver(**arguments)
        stats = solver.stats()
        x = np.asarray(solution["x"]).reshape(-1)
        lambda_x = solution["lam_x"]
        lambda_g = solution["lam_g"]
        result = {
            "name": stage["name"],
            "status": stats["return_status"],
            "success": bool(stats["success"]),
            "iterations": int(stats["iter_count"]),
            "swing_clearance_m": float(stage["swing_clearance_m"]),
            "com_tolerance_m": float(stage["com_tolerance_m"]),
            "objective": float(solution["f"]),
            "x": x.copy(),
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


def assert_feasible(
    evaluator: PinocchioGeometry,
    config: dict[str, Any],
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]]:
    contacts, com, q = evaluator.evaluate_numpy(values)
    signed_error, segment_s, length = support_metrics(contacts, com)
    optimization = config["optimization"]
    target = optimization["continuation"][-1]
    if abs(contacts[0, 2]) > GROUND_TOLERANCE:
        raise RuntimeError("FL tread contact is not on ground")
    if abs(contacts[3, 2]) > GROUND_TOLERANCE:
        raise RuntimeError("HR tread contact is not on ground")
    if contacts[1, 2] < target["swing_clearance_m"]:
        raise RuntimeError("FR tread clearance is insufficient")
    if contacts[2, 2] < target["swing_clearance_m"]:
        raise RuntimeError("HL tread clearance is insufficient")
    if abs(signed_error) > target["com_tolerance_m"]:
        raise RuntimeError("CoM line error exceeds tolerance")
    segment_margin = optimization["segment_margin_m"]
    if not (
        segment_s >= segment_margin
        and segment_s <= length - segment_margin
    ):
        raise RuntimeError("CoM projection violates support-segment margins")
    if minimum_joint_margin(evaluator, values) < optimization["joint_margin_rad"]:
        raise RuntimeError("Joint-limit margin is insufficient")
    orientation_limit = math.radians(optimization["orientation_limit_deg"])
    if abs(values[1]) > orientation_limit:
        raise RuntimeError("Roll exceeds orientation bound")
    if abs(values[2]) > orientation_limit:
        raise RuntimeError("Pitch exceeds orientation bound")
    return contacts, com, q, (signed_error, segment_s, length)


def write_output(
    output_path: Path,
    urdf_path: Path,
    evaluator: PinocchioGeometry,
    config: dict[str, Any],
    stages: list[dict[str, Any]],
) -> None:
    final = stages[-1]
    values = final["x"]
    contacts, com, q, support = assert_feasible(evaluator, config, values)
    signed_error, segment_s, length = support
    optimization = config["optimization"]
    target = optimization["continuation"][-1]
    output = {
        "milestone": "M-TO1B",
        "result": "PASS",
        "model": {"urdf_path": str(urdf_path)},
        "stance": ["FL", "HR"],
        "swing": ["FR", "HL"],
        "solver": {
            "name": "CasADi 3.7.2 + IPOPT",
            "status": final["status"],
            "iterations": final["iterations"],
            "total_continuation_iterations": sum(
                stage["iterations"] for stage in stages
            ),
        },
        "objective": objective_value(values, config),
        "variables": {
            "base_x": 0.0,
            "base_y": 0.0,
            "base_z": float(values[0]),
            "base_roll": float(values[1]),
            "base_pitch": float(values[2]),
            "base_yaw": 0.0,
            "leg_joints": {
                name: float(values[3 + offset])
                for offset, name in enumerate(LEG_JOINT_NAMES)
            },
            "wheel_angles": {name: 0.0 for name in WHEEL_NAMES},
        },
        "pinocchio_q_xyzw": q.tolist(),
        "com_world_xyz": com.tolist(),
        "physical_tread_contacts_world": {
            corner: contacts[index].tolist()
            for index, corner in enumerate(WHEEL_CORNERS)
        },
        "metrics": {
            "fl_contact_z_m": float(contacts[0, 2]),
            "hr_contact_z_m": float(contacts[3, 2]),
            "fr_clearance_m": float(contacts[1, 2]),
            "hl_clearance_m": float(contacts[2, 2]),
            "com_line_error_m": abs(signed_error),
            "com_line_signed_error_m": signed_error,
            "s_m": segment_s,
            "support_length_m": length,
            "s_over_L": segment_s / length,
            "minimum_joint_limit_margin_rad": minimum_joint_margin(
                evaluator, values
            ),
        },
        "constraints": {
            "joint_margin_rad": optimization["joint_margin_rad"],
            "orientation_limit_deg": optimization["orientation_limit_deg"],
            "segment_margin_m": optimization["segment_margin_m"],
            "swing_clearance_m": target["swing_clearance_m"],
            "com_tolerance_m": target["com_tolerance_m"],
        },
        "continuation": [
            {key: value for key, value in stage.items() if key != "x"}
            for stage in stages
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(output, stream, sort_keys=False)


def read_decision_vector(output: dict[str, Any]) -> np.ndarray:
    variables = output["variables"]
    return np.array(
        [
            variables["base_z"],
            variables["base_roll"],
            variables["base_pitch"],
            *(variables["leg_joints"][name] for name in LEG_JOINT_NAMES),
        ],
        dtype=np.float64,
    )


def require_near(actual: float, reported: float, name: str) -> None:
    if not (
        math.isfinite(actual)
        and math.isfinite(reported)
        and abs(actual - reported) <= REPORT_TOLERANCE
    ):
        raise RuntimeError(f"{name} mismatch: actual={actual}, reported={reported}")


def verify_output(
    output_path: Path,
    evaluator: PinocchioGeometry,
    config: dict[str, Any],
) -> None:
    with output_path.open(encoding="utf-8") as stream:
        output = yaml.safe_load(stream)
    if output["milestone"] != "M-TO1B" or output["result"] != "PASS":
        raise RuntimeError("Saved output is not an M-TO1B PASS result")
    values = read_decision_vector(output)
    contacts, com, q, support = assert_feasible(evaluator, config, values)
    signed_error, segment_s, length = support
    saved_q = np.asarray(output["pinocchio_q_xyzw"], dtype=np.float64)
    if saved_q.shape != q.shape or np.max(np.abs(saved_q - q)) > REPORT_TOLERANCE:
        raise RuntimeError("Saved Pinocchio q does not reproduce decision variables")
    saved_com = np.asarray(output["com_world_xyz"], dtype=np.float64)
    if saved_com.shape != com.shape or np.max(np.abs(saved_com - com)) > REPORT_TOLERANCE:
        raise RuntimeError("Saved Pinocchio CoM does not reproduce the pose")
    for index, corner in enumerate(WHEEL_CORNERS):
        saved_contact = np.asarray(
            output["physical_tread_contacts_world"][corner], dtype=np.float64
        )
        if (
            saved_contact.shape != contacts[index].shape
            or np.max(np.abs(saved_contact - contacts[index])) > REPORT_TOLERANCE
        ):
            raise RuntimeError(f"Saved {corner} tread contact does not reproduce")
    variables = output["variables"]
    for name in ("base_x", "base_y", "base_yaw"):
        require_near(float(variables[name]), 0.0, name)
    for name in WHEEL_NAMES:
        require_near(float(variables["wheel_angles"][name]), 0.0, name)
    metrics = output["metrics"]
    checks = (
        (contacts[0, 2], metrics["fl_contact_z_m"], "FL contact z"),
        (contacts[3, 2], metrics["hr_contact_z_m"], "HR contact z"),
        (contacts[1, 2], metrics["fr_clearance_m"], "FR clearance"),
        (contacts[2, 2], metrics["hl_clearance_m"], "HL clearance"),
        (abs(signed_error), metrics["com_line_error_m"], "CoM line error"),
        (signed_error, metrics["com_line_signed_error_m"], "signed CoM error"),
        (segment_s, metrics["s_m"], "support s"),
        (length, metrics["support_length_m"], "support length"),
        (segment_s / length, metrics["s_over_L"], "support s/L"),
        (
            minimum_joint_margin(evaluator, values),
            metrics["minimum_joint_limit_margin_rad"],
            "minimum joint margin",
        ),
        (objective_value(values, config), output["objective"], "objective"),
    )
    for actual, reported, name in checks:
        require_near(float(actual), float(reported), name)
    continuation = output["continuation"]
    if [stage["name"] for stage in continuation] != ["A", "B", "C", "D"]:
        raise RuntimeError("Saved continuation does not contain stages A-D")
    if not all(stage["success"] for stage in continuation):
        raise RuntimeError("A saved continuation stage is unsuccessful")


def print_result(
    evaluator: PinocchioGeometry,
    config: dict[str, Any],
    stages: list[dict[str, Any]],
) -> None:
    final = stages[-1]
    values = final["x"]
    contacts, _, _, support = assert_feasible(evaluator, config, values)
    signed_error, segment_s, length = support
    print("\nM-TO1B RESULT\n")
    print(f"Solver status = {final['status']}")
    print(f"iterations = {final['iterations']}\n")
    print(f"base_z = {values[0]:.12f}")
    print(f"roll = {values[1]:.12f}")
    print(f"pitch = {values[2]:.12f}\n")
    print("leg joints =")
    for offset, name in enumerate(LEG_JOINT_NAMES):
        print(f"  {name} = {values[3 + offset]:.12f}")
    print(f"\nFL contact z = {contacts[0, 2]:.12f}")
    print(f"HR contact z = {contacts[3, 2]:.12f}\n")
    print(f"FR clearance = {contacts[1, 2]:.12f}")
    print(f"HL clearance = {contacts[2, 2]:.12f}\n")
    print(f"CoM line error = {abs(signed_error):.12f}")
    print(f"s = {segment_s:.12f}")
    print(f"L = {length:.12f}")
    print(f"s/L = {segment_s / length:.12f}\n")
    print(
        "minimum joint-limit margin = "
        f"{minimum_joint_margin(evaluator, values):.12f}\n"
    )
    print("Continuation:")
    for stage in stages:
        print(
            f"{stage['name']} = {stage['status']} "
            f"(iterations={stage['iterations']})"
        )
    print("\nM-TO1B RESULT: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--config", type=Path, default=Path(os.environ["VQR_CONFIG_PATH"]))
    parser.add_argument("--urdf", type=Path, default=Path(os.environ["VQR_URDF_PATH"]))
    parser.add_argument("--output", type=Path, default=Path(os.environ["VQR_OUTPUT_PATH"]))
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    config = load_config(arguments.config.resolve())
    helper = TreadContactLibrary(
        Path(os.environ["VQR_TREAD_CONTACT_LIBRARY"]).resolve(),
        arguments.urdf.resolve(),
    )
    evaluator = PinocchioGeometry(arguments.urdf.resolve(), helper)
    if arguments.verify:
        verify_output(arguments.output.resolve(), evaluator, config)
        print("M-TO1B saved-pose verification: PASS")
        return 0
    stages = run_continuation(evaluator, config)
    write_output(arguments.output.resolve(), arguments.urdf.resolve(), evaluator, config, stages)
    verify_output(arguments.output.resolve(), evaluator, config)
    print_result(evaluator, config, stages)
    print(f"Output = {arguments.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
