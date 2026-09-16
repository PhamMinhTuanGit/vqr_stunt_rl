#!/usr/bin/env python3
"""M-TO2R static equilibrium with stance-wheel alignment constraints."""

from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np
import yaml

import kinematic_equilibrium as kin
import static_equilibrium as base


N_ALIGNED_OUTPUTS = base.N_STATIC_OUTPUTS + 12
N_CONSTRAINTS = 42


class PinocchioAlignedTerms(base.PinocchioStaticTerms):
    """M-TO2 terms plus FL/HR wheel axes in WORLD and body coordinates."""

    def get_sparsity_out(self, index: int) -> ca.Sparsity:
        del index
        return ca.Sparsity.dense(N_ALIGNED_OUTPUTS, 1)

    def stance_axes_numpy(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        root_id = self.model.getJointId("root_joint")
        world_from_body = np.asarray(self.data.oMi[root_id].rotation)
        axes: dict[str, np.ndarray] = {}
        body_axes: dict[str, np.ndarray] = {}
        for corner in ("FL", "HR"):
            placement = self.data.oMf[self.frame_ids[corner]]
            local_axis = self.helper.geometries[corner][1]
            axis = np.asarray(placement.rotation @ local_axis, dtype=np.float64)
            axis /= np.linalg.norm(axis)
            axes[corner] = axis
            body_axes[corner] = world_from_body.T @ axis
        values = (axes["FL"], axes["HR"], body_axes["FL"], body_axes["HR"])
        if not all(np.all(np.isfinite(value)) for value in values):
            raise RuntimeError("Wheel-axis evaluation contains non-finite values")
        return values

    def evaluate_aligned_numpy(
        self, pose: np.ndarray
    ) -> tuple[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        static_terms = self.evaluate_numpy(pose)
        return static_terms, self.stance_axes_numpy()

    def eval(self, arguments: list[ca.DM]) -> list[ca.DM]:
        static_terms, axes = self.evaluate_aligned_numpy(np.asarray(arguments[0]))
        contacts, com, _, h, j_fl, j_hr = static_terms
        output = np.concatenate(
            (
                contacts.reshape(-1),
                com,
                h,
                j_fl.reshape(-1, order="F"),
                j_hr.reshape(-1, order="F"),
                *axes,
            )
        )
        return [ca.DM(output)]


def load_inputs(
    config_path: Path, initial_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = base.load_yaml(config_path)
    initial = base.load_yaml(initial_path)
    if initial.get("milestone") != "M-TO2" or initial.get("result") != "PASS":
        raise RuntimeError("M-TO2R initial guess is not the saved M-TO2 PASS result")
    if initial.get("stance") != ["FL", "HR"] or initial.get("swing") != ["FR", "HL"]:
        raise RuntimeError("M-TO2 initial guess has unexpected contact sets")
    tolerances = config["aligned_static_optimization"][
        "com_tolerance_continuation_m"
    ]
    if tolerances != [0.002, 0.005, 0.010]:
        raise RuntimeError("M-TO2R CoM continuation must be exactly 2/5/10 mm")
    return config, initial


def decision_from_saved(saved: dict[str, Any]) -> np.ndarray:
    pose = base.pose_from_saved(saved)
    torque_node = saved["actuator_torques_nm"]
    torques = np.array(
        [
            *(torque_node["leg"][name] for name in kin.LEG_JOINT_NAMES),
            *(torque_node["wheel"][name] for name in kin.WHEEL_NAMES),
        ],
        dtype=np.float64,
    )
    forces = np.array(
        [
            *saved["contact_forces_world"]["FL"],
            *saved["contact_forces_world"]["HR"],
        ],
        dtype=np.float64,
    )
    decision = np.concatenate((pose, torques, forces))
    if decision.size != base.N_VARIABLES or not np.all(np.isfinite(decision)):
        raise RuntimeError("Saved M-TO2 decision vector is incomplete or non-finite")
    return decision


def build_solver(
    evaluator: PinocchioAlignedTerms,
    config: dict[str, Any],
    pose_reference: np.ndarray,
    selection: np.ndarray,
) -> ca.Function:
    static_config = config["static_optimization"]
    aligned_config = config["aligned_static_optimization"]
    weights = static_config["weights"]
    variables = ca.MX.sym("x", base.N_VARIABLES)
    pose = variables[: base.N_POSE]
    torques = variables[base.N_POSE : base.N_POSE + base.N_TORQUES]
    f_fl = variables[
        base.N_POSE + base.N_TORQUES : base.N_POSE + base.N_TORQUES + 3
    ]
    f_hr = variables[base.N_POSE + base.N_TORQUES + 3 :]

    terms = evaluator(pose)
    fl, fr, hl, hr = terms[0:3], terms[3:6], terms[6:9], terms[9:12]
    com = terms[12:15]
    h = terms[15:37]
    j_fl = ca.reshape(terms[37:103], 3, 22)
    j_hr = ca.reshape(terms[103:169], 3, 22)
    a_fl_w = terms[169:172]
    a_hr_w = terms[172:175]
    a_fl_b = terms[175:178]
    a_hr_b = terms[178:181]

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
    alignment = ca.vertcat(
        1.0 - a_fl_b[1] ** 2,
        1.0 - a_hr_b[1] ** 2,
        1.0 - ca.dot(a_fl_w, a_hr_w) ** 2,
        a_fl_w[2],
        a_hr_w[2],
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
        alignment,
    )

    reference = ca.DM(pose_reference)
    objective = (
        weights["pose"] * ca.sumsqr(pose[3:] - reference[3:])
        + weights["orientation"] * ca.sumsqr(pose[1:3])
        + weights["base_height"] * (pose[0] - reference[0]) ** 2
        + weights["torque"] * ca.sumsqr(torques)
        + weights["force"] * (ca.sumsqr(f_fl) + ca.sumsqr(f_hr))
        + aligned_config["hipx_weight"] * ca.sumsqr(pose[3:7])
    )
    ipopt = aligned_config["ipopt"]
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
        "aligned_static_equilibrium",
        "ipopt",
        {"x": variables, "f": objective, "g": constraints},
        options,
    )


def constraint_bounds(
    static_config: dict[str, Any],
    aligned_config: dict[str, Any],
    com_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    final_stage = {"friction": True, "positive_normal": True, "torque_limits": True}
    lower_base, upper_base = base.constraint_bounds(static_config, final_stage)
    lower = np.concatenate((lower_base, np.full(5, -np.inf)))
    upper = np.concatenate((upper_base, np.full(5, np.inf)))
    lower[26] = -com_tolerance
    upper[26] = com_tolerance
    upper[37] = math.sin(
        math.radians(aligned_config["wheel_body_alignment_limit_deg"])
    ) ** 2
    upper[38] = upper[37]
    upper[39] = math.sin(
        math.radians(aligned_config["stance_wheel_parallel_limit_deg"])
    ) ** 2
    vertical_limit = math.sin(
        math.radians(aligned_config["wheel_axle_horizontal_limit_deg"])
    )
    lower[40:42] = -vertical_limit
    upper[40:42] = vertical_limit
    if lower.size != N_CONSTRAINTS or upper.size != N_CONSTRAINTS:
        raise RuntimeError("Unexpected aligned constraint-vector dimension")
    return lower, upper


def axis_metrics(
    evaluator: PinocchioAlignedTerms, pose: np.ndarray
) -> dict[str, Any]:
    evaluator.evaluate_numpy(pose)
    a_fl_w, a_hr_w, a_fl_b, a_hr_b = evaluator.stance_axes_numpy()

    def sign_invariant_angle(first: np.ndarray, second: np.ndarray) -> float:
        cosine = float(np.clip(abs(first @ second), 0.0, 1.0))
        return math.degrees(math.acos(cosine))

    lateral = np.array((0.0, 1.0, 0.0))
    return {
        "world": {"FL": a_fl_w, "HR": a_hr_w},
        "body": {"FL": a_fl_b, "HR": a_hr_b},
        "body_alignment_deg": {
            "FL": sign_invariant_angle(a_fl_b, lateral),
            "HR": sign_invariant_angle(a_hr_b, lateral),
        },
        "parallel_deg": sign_invariant_angle(a_fl_w, a_hr_w),
        "vertical_component": {
            "FL": abs(float(a_fl_w[2])),
            "HR": abs(float(a_hr_w[2])),
        },
        "alignment_error_squared_sine": {
            "FL": 1.0 - float(a_fl_b[1]) ** 2,
            "HR": 1.0 - float(a_hr_b[1]) ** 2,
        },
        "parallel_error_squared_sine": 1.0 - float(a_fl_w @ a_hr_w) ** 2,
    }


def validate_candidate(
    evaluator: PinocchioAlignedTerms,
    config: dict[str, Any],
    values: np.ndarray,
    com_tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    static_config = copy.deepcopy(config["static_optimization"])
    static_config["com_tolerance_m"] = com_tolerance
    result = base.evaluate_solution(evaluator, static_config, values)
    axes = axis_metrics(evaluator, result["pose"])
    aligned_config = config["aligned_static_optimization"]
    body_limit = aligned_config["wheel_body_alignment_limit_deg"]
    parallel_limit = aligned_config["stance_wheel_parallel_limit_deg"]
    vertical_limit = math.sin(
        math.radians(aligned_config["wheel_axle_horizontal_limit_deg"])
    )
    if max(axes["body_alignment_deg"].values()) > body_limit + 1.0e-7:
        raise RuntimeError("Wheel-to-body alignment constraint failed")
    if axes["parallel_deg"] > parallel_limit + 1.0e-7:
        raise RuntimeError("FL-HR wheel-axis parallel constraint failed")
    if max(axes["vertical_component"].values()) > vertical_limit + 1.0e-9:
        raise RuntimeError("Wheel-axle horizontal constraint failed")
    return result, axes


def solve(
    evaluator: PinocchioAlignedTerms,
    config: dict[str, Any],
    initial_values: np.ndarray,
    pose_reference: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    aligned_config = config["aligned_static_optimization"]
    static_config = config["static_optimization"]
    selection = base.actuator_matrix(evaluator)
    solver = build_solver(evaluator, config, pose_reference, selection)
    final_stage = {"friction": True, "positive_normal": True, "torque_limits": True}
    lower_x, upper_x = base.variable_bounds(evaluator, static_config, final_stage)
    values = initial_values.copy()
    lambda_x = None
    lambda_g = None
    attempts: list[dict[str, Any]] = []

    for tolerance in aligned_config["com_tolerance_continuation_m"]:
        lower_g, upper_g = constraint_bounds(static_config, aligned_config, tolerance)
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
        candidate = np.asarray(solution["x"]).reshape(-1)
        if np.all(np.isfinite(candidate)):
            values = candidate
            lambda_x = solution["lam_x"]
            lambda_g = solution["lam_g"]
        attempt = {
            "name": f"{int(round(tolerance * 1000.0))}mm",
            "com_tolerance_m": tolerance,
            "status": stats["return_status"],
            "success": bool(stats["success"]),
            "iterations": int(stats["iter_count"]),
            "friction": True,
            "positive_normal": True,
            "torque_limits": True,
            "wheel_alignment_constraints": True,
            "objective": float(solution["f"]),
            "x": values.copy(),
        }
        attempts.append(attempt)
        print(
            f"CoM tolerance {attempt['name']}: {attempt['status']}, "
            f"iterations={attempt['iterations']}"
        )
        if attempt["success"]:
            try:
                result, axes = validate_candidate(
                    evaluator, config, values, tolerance
                )
            except RuntimeError as error:
                attempt["success"] = False
                attempt["validation_issue"] = str(error)
            else:
                return attempt, attempts, result, axes

    raise RuntimeError("NO STATIC EQUILIBRIUM WITH WHEEL-ALIGNMENT CONSTRAINTS")


def write_output(
    output_path: Path,
    urdf_path: Path,
    initial_path: Path,
    evaluator: PinocchioAlignedTerms,
    config: dict[str, Any],
    selected: dict[str, Any],
    attempts: list[dict[str, Any]],
    axes: dict[str, Any],
    objective_reference_path: Path,
) -> dict[str, Any]:
    effective_config = copy.deepcopy(config)
    tolerance = selected["com_tolerance_m"]
    effective_config["static_optimization"]["com_tolerance_m"] = tolerance
    output = base.write_output(
        output_path,
        urdf_path,
        initial_path,
        evaluator,
        effective_config,
        [selected],
    )
    aligned_config = config["aligned_static_optimization"]
    output["milestone"] = "M-TO2R"
    output["solver"]["total_continuation_iterations"] = sum(
        attempt["iterations"] for attempt in attempts
    )
    output["objective_terms"] = {
        "hipx_weight": aligned_config["hipx_weight"],
        "reference_pose": str(objective_reference_path),
    }
    output["wheel_axes"] = {
        "world": {
            corner: axes["world"][corner].tolist() for corner in ("FL", "HR")
        },
        "body": {
            corner: axes["body"][corner].tolist() for corner in ("FL", "HR")
        },
        "body_alignment_angle_deg": axes["body_alignment_deg"],
        "fl_hr_parallel_angle_deg": axes["parallel_deg"],
        "vertical_component_abs": axes["vertical_component"],
        "body_alignment_error": axes["alignment_error_squared_sine"],
        "fl_hr_parallel_error": axes["parallel_error_squared_sine"],
    }
    output["constraints"].update(
        {
            "wheel_body_alignment_limit_deg": aligned_config[
                "wheel_body_alignment_limit_deg"
            ],
            "stance_wheel_parallel_limit_deg": aligned_config[
                "stance_wheel_parallel_limit_deg"
            ],
            "wheel_axle_horizontal_limit_deg": aligned_config[
                "wheel_axle_horizontal_limit_deg"
            ],
            "com_tolerance_m": tolerance,
        }
    )
    output["com_tolerance_attempts"] = [
        {key: value for key, value in attempt.items() if key != "x"}
        for attempt in attempts
    ]
    output["continuation"] = output.pop("com_tolerance_attempts")
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(output, stream, sort_keys=False)
    return output


def print_result(output: dict[str, Any]) -> None:
    variables = output["variables"]
    joints = variables["leg_joints"]
    axes = output["wheel_axes"]
    leg_torques = output["actuator_torques_nm"]["leg"]
    wheel_torques = output["actuator_torques_nm"]["wheel"]
    print("\nM-TO2R RESULT\n")
    print("PASS/FAIL = PASS\n")
    print(f"Solver status = {output['solver']['status']}")
    print(f"iterations = {output['solver']['iterations']}\n")
    print("base:")
    print(f"z = {variables['base_z']:.12f}")
    print(f"roll = {variables['base_roll']:.12f}")
    print(f"pitch = {variables['base_pitch']:.12f}\n")
    print("HipX:")
    for corner in kin.WHEEL_CORNERS:
        print(f"{corner} = {joints[f'{corner}_HipX_joint']:.12f}")
    print("\nwheel alignment angle:")
    print(f"FL vs body = {axes['body_alignment_angle_deg']['FL']:.12f} deg")
    print(f"HR vs body = {axes['body_alignment_angle_deg']['HR']:.12f} deg")
    print(f"FL-HR wheel-axis angle = {axes['fl_hr_parallel_angle_deg']:.12f} deg\n")
    print("wheel axle vertical component:")
    print(f"FL = {axes['vertical_component_abs']['FL']:.12f}")
    print(f"HR = {axes['vertical_component_abs']['HR']:.12f}\n")
    print(f"CoM line error = {output['metrics']['com_line_error_m']:.12f}")
    print(f"s/L = {output['metrics']['s_over_L']:.12f}\n")
    print(f"FR clearance = {output['swing_clearance_m']['FR']:.12f}")
    print(f"HL clearance = {output['swing_clearance_m']['HL']:.12f}\n")
    print("contact forces:")
    print(f"FL = {base.print_vector(output['contact_forces_world']['FL'])}")
    print(f"HR = {base.print_vector(output['contact_forces_world']['HR'])}\n")
    print(f"max leg torque = {max(abs(value) for value in leg_torques.values()):.12f}")
    print(f"max wheel torque = {max(abs(value) for value in wheel_torques.values()):.12f}\n")
    print("friction utilization:")
    print(f"FL = {output['friction_utilization']['FL']:.12f}")
    print(f"HR = {output['friction_utilization']['HR']:.12f}\n")
    print(
        "full dynamics residual infinity norm = "
        f"{output['dynamics']['residual_infinity_norm']:.12e}\n"
    )
    print(
        "CoM tolerance used = "
        f"{int(round(output['constraints']['com_tolerance_m'] * 1000.0))}mm"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(os.environ["VQR_CONFIG_PATH"]))
    parser.add_argument("--urdf", type=Path, default=Path(os.environ["VQR_URDF_PATH"]))
    parser.add_argument(
        "--initial", type=Path, default=Path(os.environ["VQR_MTO2_OUTPUT_PATH"])
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ["VQR_ALIGNED_STATIC_OUTPUT_PATH"]),
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    config, initial = load_inputs(arguments.config.resolve(), arguments.initial.resolve())
    helper = kin.TreadContactLibrary(
        Path(os.environ["VQR_TREAD_CONTACT_LIBRARY"]).resolve(),
        arguments.urdf.resolve(),
    )
    evaluator = PinocchioAlignedTerms(arguments.urdf.resolve(), helper)
    initial_values = decision_from_saved(initial)
    objective_reference_path = Path(initial["initial_guess"]).resolve()
    objective_reference_saved = base.load_yaml(objective_reference_path)
    if (
        objective_reference_saved.get("milestone") != "M-TO1B"
        or objective_reference_saved.get("result") != "PASS"
    ):
        raise RuntimeError("Saved M-TO2 objective reference is not an M-TO1B PASS pose")
    pose_reference = base.pose_from_saved(objective_reference_saved)
    selected, attempts, _, axes = solve(
        evaluator, config, initial_values, pose_reference
    )
    output = write_output(
        arguments.output.resolve(),
        arguments.urdf.resolve(),
        arguments.initial.resolve(),
        evaluator,
        config,
        selected,
        attempts,
        axes,
        objective_reference_path,
    )
    print_result(output)
    print(f"\nOutput = {arguments.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
