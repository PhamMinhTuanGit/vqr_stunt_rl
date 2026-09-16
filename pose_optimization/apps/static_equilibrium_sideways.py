#!/usr/bin/env python3
"""M-TO2S static equilibrium with the FL-HR support line along body Y."""

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
import static_equilibrium_aligned as aligned


N_SIDEWAYS_OUTPUTS = aligned.N_ALIGNED_OUTPUTS + 6
N_CONSTRAINTS = 45


class PinocchioSidewaysTerms(aligned.PinocchioAlignedTerms):
    """M-TO2R terms plus FL/HR physical contacts expressed in BODY."""

    def get_sparsity_out(self, index: int) -> ca.Sparsity:
        del index
        return ca.Sparsity.dense(N_SIDEWAYS_OUTPUTS, 1)

    def support_contacts_body_numpy(
        self, contacts_world: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        root_id = self.model.getJointId("root_joint")
        placement = self.data.oMi[root_id]
        world_from_body = np.asarray(placement.rotation)
        body_origin_world = np.asarray(placement.translation)
        fl_body = world_from_body.T @ (contacts_world[0] - body_origin_world)
        hr_body = world_from_body.T @ (contacts_world[3] - body_origin_world)
        if not np.all(np.isfinite(fl_body)) or not np.all(np.isfinite(hr_body)):
            raise RuntimeError("BODY-frame support contacts are non-finite")
        return fl_body, hr_body

    def eval(self, arguments: list[ca.DM]) -> list[ca.DM]:
        static_terms, axes = self.evaluate_aligned_numpy(np.asarray(arguments[0]))
        contacts, com, _, h, j_fl, j_hr = static_terms
        fl_body, hr_body = self.support_contacts_body_numpy(contacts)
        output = np.concatenate(
            (
                contacts.reshape(-1),
                com,
                h,
                j_fl.reshape(-1, order="F"),
                j_hr.reshape(-1, order="F"),
                *axes,
                fl_body,
                hr_body,
            )
        )
        return [ca.DM(output)]


def load_inputs(
    config_path: Path, initial_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = base.load_yaml(config_path)
    initial = base.load_yaml(initial_path)
    if initial.get("milestone") != "M-TO2R" or initial.get("result") != "PASS":
        raise RuntimeError("M-TO2S initial guess is not the M-TO2R PASS result")
    if initial.get("stance") != ["FL", "HR"] or initial.get("swing") != ["FR", "HL"]:
        raise RuntimeError("M-TO2R initial guess has unexpected contact sets")
    sideways = config["sideways_static_optimization"]
    if sideways["eps_x_continuation_m"] != [0.10, 0.07, 0.05, 0.03]:
        raise RuntimeError("M-TO2S eps_x continuation must be 0.10/0.07/0.05/0.03 m")
    return config, initial


def build_solver(
    evaluator: PinocchioSidewaysTerms,
    config: dict[str, Any],
    pose_reference: np.ndarray,
    selection: np.ndarray,
) -> ca.Function:
    static_config = config["static_optimization"]
    aligned_config = config["aligned_static_optimization"]
    sideways_config = config["sideways_static_optimization"]
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
    p_fl_b = terms[181:184]
    p_hr_b = terms[184:187]

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
    wheel_alignment = ca.vertcat(
        1.0 - a_fl_b[1] ** 2,
        1.0 - a_hr_b[1] ** 2,
        1.0 - ca.dot(a_fl_w, a_hr_w) ** 2,
        a_fl_w[2],
        a_hr_w[2],
    )
    dx = p_hr_b[0] - p_fl_b[0]
    dy = p_hr_b[1] - p_fl_b[1]
    direction_x = dx / ca.sqrt(dx**2 + dy**2)
    sideways_constraints = ca.vertcat(dx, dy**2, direction_x)
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
        wheel_alignment,
        sideways_constraints,
    )

    reference = ca.DM(pose_reference)
    objective = (
        weights["pose"] * ca.sumsqr(pose[3:] - reference[3:])
        + weights["orientation"] * ca.sumsqr(pose[1:3])
        + weights["base_height"] * (pose[0] - reference[0]) ** 2
        + weights["torque"] * ca.sumsqr(torques)
        + weights["force"] * (ca.sumsqr(f_fl) + ca.sumsqr(f_hr))
        + aligned_config["hipx_weight"] * ca.sumsqr(pose[3:7])
        + sideways_config["dx_weight"] * dx**2
    )
    ipopt = sideways_config["ipopt"]
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
        "sideways_static_equilibrium",
        "ipopt",
        {"x": variables, "f": objective, "g": constraints},
        options,
    )


def constraint_bounds(
    config: dict[str, Any], eps_x: float, swing_clearance: float
) -> tuple[np.ndarray, np.ndarray]:
    static_config = copy.deepcopy(config["static_optimization"])
    static_config["swing_clearance_m"] = swing_clearance
    aligned_config = config["aligned_static_optimization"]
    sideways_config = config["sideways_static_optimization"]
    lower_base, upper_base = aligned.constraint_bounds(
        static_config, aligned_config, sideways_config["com_tolerance_m"]
    )
    lower = np.concatenate((lower_base, np.full(3, -np.inf)))
    upper = np.concatenate((upper_base, np.full(3, np.inf)))
    lower[42], upper[42] = -eps_x, eps_x
    lower[43] = sideways_config["y_separation_min_m"] ** 2
    direction_limit = math.sin(
        math.radians(sideways_config["support_direction_limit_deg"])
    )
    lower[44], upper[44] = -direction_limit, direction_limit
    if lower.size != N_CONSTRAINTS or upper.size != N_CONSTRAINTS:
        raise RuntimeError("Unexpected sideways constraint-vector dimension")
    return lower, upper


def support_body_metrics(
    evaluator: PinocchioSidewaysTerms, pose: np.ndarray
) -> dict[str, Any]:
    contacts, *_ = evaluator.evaluate_numpy(pose)
    fl_body, hr_body = evaluator.support_contacts_body_numpy(contacts)
    delta = hr_body[:2] - fl_body[:2]
    length = float(np.linalg.norm(delta))
    if not np.isfinite(length) or length <= 1.0e-12:
        raise RuntimeError("BODY-frame support line is degenerate")
    dx, dy = float(delta[0]), float(delta[1])
    return {
        "FL": fl_body,
        "HR": hr_body,
        "dx": dx,
        "dy": dy,
        "direction_x_abs": abs(dx) / length,
        "angle_body_x_deg": math.degrees(
            math.acos(float(np.clip(abs(dx) / length, 0.0, 1.0)))
        ),
        "angle_body_y_deg": math.degrees(
            math.acos(float(np.clip(abs(dy) / length, 0.0, 1.0)))
        ),
    }


def validate_candidate(
    evaluator: PinocchioSidewaysTerms,
    config: dict[str, Any],
    values: np.ndarray,
    eps_x: float,
    swing_clearance: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    effective_config = copy.deepcopy(config)
    effective_config["static_optimization"]["swing_clearance_m"] = swing_clearance
    com_tolerance = effective_config["sideways_static_optimization"][
        "com_tolerance_m"
    ]
    result, axes = aligned.validate_candidate(
        evaluator, effective_config, values, com_tolerance
    )
    support = support_body_metrics(evaluator, result["pose"])
    sideways_config = config["sideways_static_optimization"]
    if abs(support["dx"]) > eps_x + 1.0e-8:
        raise RuntimeError("BODY-frame support dx constraint failed")
    if abs(support["dy"]) < sideways_config["y_separation_min_m"] - 1.0e-8:
        raise RuntimeError("BODY-frame support y-separation constraint failed")
    direction_limit = math.sin(
        math.radians(sideways_config["support_direction_limit_deg"])
    )
    if support["direction_x_abs"] > direction_limit + 1.0e-8:
        raise RuntimeError("BODY-frame support direction constraint failed")
    return result, axes, support


def solve(
    evaluator: PinocchioSidewaysTerms,
    config: dict[str, Any],
    initial_values: np.ndarray,
    pose_reference: np.ndarray,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    static_config = config["static_optimization"]
    sideways_config = config["sideways_static_optimization"]
    selection = base.actuator_matrix(evaluator)
    solver = build_solver(evaluator, config, pose_reference, selection)
    final_stage = {"friction": True, "positive_normal": True, "torque_limits": True}
    lower_x, upper_x = base.variable_bounds(evaluator, static_config, final_stage)
    values = initial_values.copy()
    lambda_x = None
    lambda_g = None
    attempts: list[dict[str, Any]] = []
    best: tuple[
        dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
    ] | None = None

    clearances = [sideways_config["preferred_swing_clearance_m"]]
    fallback = sideways_config["fallback_swing_clearance_m"]
    if fallback != clearances[0]:
        clearances.append(fallback)

    for clearance_index, swing_clearance in enumerate(clearances):
        for eps_x in sideways_config["eps_x_continuation_m"]:
            lower_g, upper_g = constraint_bounds(config, eps_x, swing_clearance)
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
                "name": (
                    f"clearance_{int(round(swing_clearance * 1000.0))}mm_"
                    f"eps_{int(round(eps_x * 1000.0))}mm"
                ),
                "eps_x_m": eps_x,
                "swing_clearance_m": swing_clearance,
                "com_tolerance_m": sideways_config["com_tolerance_m"],
                "status": stats["return_status"],
                "success": bool(stats["success"]),
                "iterations": int(stats["iter_count"]),
                "friction": True,
                "positive_normal": True,
                "torque_limits": True,
                "wheel_alignment_constraints": True,
                "support_orientation_constraints": True,
                "objective": float(solution["f"]),
                "x": values.copy(),
            }
            attempts.append(attempt)
            print(
                f"Sideways {attempt['name']}: {attempt['status']}, "
                f"iterations={attempt['iterations']}"
            )
            if not attempt["success"]:
                break
            try:
                result, axes, support = validate_candidate(
                    evaluator, config, values, eps_x, swing_clearance
                )
            except RuntimeError as error:
                attempt["success"] = False
                attempt["validation_issue"] = str(error)
                break
            if (
                best is None
                or eps_x < best[0]["eps_x_m"] - 1.0e-12
                or (
                    abs(eps_x - best[0]["eps_x_m"]) <= 1.0e-12
                    and swing_clearance > best[0]["swing_clearance_m"]
                )
            ):
                best = (attempt, result, axes, support)

        if best is not None and best[0]["eps_x_m"] <= 0.03 + 1.0e-12:
            break
        if clearance_index == 0:
            print("Preferred 30 mm clearance did not reach eps_x=30 mm; trying 20 mm")

    if best is None:
        raise RuntimeError("NO STATIC EQUILIBRIUM WITH SIDEWAYS SUPPORT CONSTRAINTS")
    selected, result, axes, support = best
    return selected, attempts, result, axes, support


def write_output(
    output_path: Path,
    urdf_path: Path,
    initial_path: Path,
    objective_reference_path: Path,
    evaluator: PinocchioSidewaysTerms,
    config: dict[str, Any],
    selected: dict[str, Any],
    attempts: list[dict[str, Any]],
    axes: dict[str, Any],
    support: dict[str, Any],
) -> dict[str, Any]:
    effective_config = copy.deepcopy(config)
    effective_config["static_optimization"]["swing_clearance_m"] = selected[
        "swing_clearance_m"
    ]
    effective_config["sideways_static_optimization"]["com_tolerance_m"] = selected[
        "com_tolerance_m"
    ]
    output = aligned.write_output(
        output_path,
        urdf_path,
        initial_path,
        evaluator,
        effective_config,
        selected,
        attempts,
        axes,
        objective_reference_path,
    )
    sideways_config = config["sideways_static_optimization"]
    output["milestone"] = "M-TO2S"
    output["objective_terms"]["dx_weight"] = sideways_config["dx_weight"]
    output["support_contacts_body"] = {
        "FL": support["FL"].tolist(),
        "HR": support["HR"].tolist(),
    }
    output["support_line_body"] = {
        "dx_m": support["dx"],
        "dy_m": support["dy"],
        "direction_x_abs": support["direction_x_abs"],
        "angle_wrt_body_x_deg": support["angle_body_x_deg"],
        "angle_wrt_body_y_deg": support["angle_body_y_deg"],
    }
    output["constraints"].update(
        {
            "support_dx_limit_m": selected["eps_x_m"],
            "support_y_separation_min_m": sideways_config[
                "y_separation_min_m"
            ],
            "support_direction_limit_deg": sideways_config[
                "support_direction_limit_deg"
            ],
            "swing_clearance_m": selected["swing_clearance_m"],
            "com_tolerance_m": selected["com_tolerance_m"],
        }
    )
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(output, stream, sort_keys=False)
    return output


def print_result(output: dict[str, Any]) -> None:
    variables = output["variables"]
    joints = variables["leg_joints"]
    support_contacts = output["support_contacts_body"]
    support = output["support_line_body"]
    leg_torques = output["actuator_torques_nm"]["leg"]
    wheel_torques = output["actuator_torques_nm"]["wheel"]
    print("\nM-TO2S RESULT\n")
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
    print("\nsupport contacts in body frame:")
    print(f"FL = {base.print_vector(support_contacts['FL'])}")
    print(f"HR = {base.print_vector(support_contacts['HR'])}\n")
    print("support line:")
    print(f"dx = {support['dx_m']:.12f}")
    print(f"dy = {support['dy_m']:.12f}")
    print(f"angle wrt body X = {support['angle_wrt_body_x_deg']:.12f} deg")
    print(f"angle wrt body Y = {support['angle_wrt_body_y_deg']:.12f} deg\n")
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
    print(f"final eps_x used = {output['constraints']['support_dx_limit_m']:.12f}")
    print(
        "final y_sep_min used = "
        f"{output['constraints']['support_y_separation_min_m']:.12f}"
    )
    print(
        "final swing clearance constraint used = "
        f"{output['constraints']['swing_clearance_m']:.12f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(os.environ["VQR_CONFIG_PATH"]))
    parser.add_argument("--urdf", type=Path, default=Path(os.environ["VQR_URDF_PATH"]))
    parser.add_argument(
        "--initial", type=Path, default=Path(os.environ["VQR_MTO2R_OUTPUT_PATH"])
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ["VQR_SIDEWAYS_STATIC_OUTPUT_PATH"]),
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    config, initial = load_inputs(arguments.config.resolve(), arguments.initial.resolve())
    helper = kin.TreadContactLibrary(
        Path(os.environ["VQR_TREAD_CONTACT_LIBRARY"]).resolve(),
        arguments.urdf.resolve(),
    )
    evaluator = PinocchioSidewaysTerms(arguments.urdf.resolve(), helper)
    initial_values = aligned.decision_from_saved(initial)
    objective_reference_path = Path(initial["objective_terms"]["reference_pose"]).resolve()
    objective_reference_saved = base.load_yaml(objective_reference_path)
    if (
        objective_reference_saved.get("milestone") != "M-TO1B"
        or objective_reference_saved.get("result") != "PASS"
    ):
        raise RuntimeError("M-TO2S objective reference is not an M-TO1B PASS pose")
    pose_reference = base.pose_from_saved(objective_reference_saved)
    selected, attempts, _, axes, support = solve(
        evaluator, config, initial_values, pose_reference
    )
    output = write_output(
        arguments.output.resolve(),
        arguments.urdf.resolve(),
        arguments.initial.resolve(),
        objective_reference_path,
        evaluator,
        config,
        selected,
        attempts,
        axes,
        support,
    )
    print_result(output)
    print(f"\nOutput = {arguments.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
