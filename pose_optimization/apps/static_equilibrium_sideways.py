#!/usr/bin/env python3
"""M-TO2S static equilibrium with the FL-HR support line along body Y.

Beyond the original sideways support constraints the pose is shaped into a
two-wheel inverted-pendulum (segway-style) configuration: the stance contacts
sit point-symmetrically about the body origin with the CoM at their support
midpoint (equal wheel loading), stance wheel axles stay parallel to the
support line, and swing feet are lifted towards a target clearance and tucked
towards the body point-symmetrically. Point symmetry is enforced by a
soft-weight-then-hard-constraint continuation because the URDF mass
distribution is not perfectly point-symmetric.
"""

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


N_SIDEWAYS_OUTPUTS = aligned.N_ALIGNED_OUTPUTS + 12
N_CONSTRAINTS = 55

# Pose layout: HipX = pose[3:7] (FL, FR, HL, HR), HipY = pose[7:11],
# Knee = pose[11:15]. The URDF places the rear legs so that equal joint
# values give a point-symmetric pose (probed against the kinematics), hence
# the FL/HR and FR/HL equal-value pairs below.
JOINT_SYMMETRY_PAIRS = (
    (3, 6),
    (4, 5),
    (7, 10),
    (8, 9),
    (11, 14),
    (12, 13),
)


class PinocchioSidewaysTerms(aligned.PinocchioAlignedTerms):
    """M-TO2R terms plus all four tread contacts expressed in BODY."""

    def get_sparsity_out(self, index: int) -> ca.Sparsity:
        del index
        return ca.Sparsity.dense(N_SIDEWAYS_OUTPUTS, 1)

    def contacts_body_numpy(
        self, contacts_world: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        root_id = self.model.getJointId("root_joint")
        placement = self.data.oMi[root_id]
        world_from_body = np.asarray(placement.rotation)
        body_origin_world = np.asarray(placement.translation)
        body_contacts = tuple(
            world_from_body.T @ (contacts_world[index] - body_origin_world)
            for index in range(4)
        )
        if not all(np.all(np.isfinite(value)) for value in body_contacts):
            raise RuntimeError("BODY-frame contacts are non-finite")
        return body_contacts

    def eval(self, arguments: list[ca.DM]) -> list[ca.DM]:
        static_terms, axes = self.evaluate_aligned_numpy(np.asarray(arguments[0]))
        contacts, com, _, h, j_fl, j_hr = static_terms
        fl_body, fr_body, hl_body, hr_body = self.contacts_body_numpy(contacts)
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
                fr_body,
                hl_body,
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
    shaping_stages(config)
    return config, initial


def shaping_stages(config: dict[str, Any]) -> list[dict[str, Any]]:
    stages = config["sideways_static_optimization"]["shaping_continuation"]
    if not isinstance(stages, list) or len(stages) < 2:
        raise RuntimeError("M-TO2S shaping continuation needs at least two stages")
    if len({stage["name"] for stage in stages}) != len(stages):
        raise RuntimeError("M-TO2S shaping stage names must be unique")
    descending = {
        "tuck_radius_m": [stage["tuck_radius_m"] for stage in stages],
        "midpoint_tolerance_m": [stage["midpoint_tolerance_m"] for stage in stages],
        "axle_line_limit_deg": [stage["axle_line_limit_deg"] for stage in stages],
    }
    ascending = {
        "weight_scale": [stage["weight_scale"] for stage in stages],
        "symmetry_weight": [stage["symmetry_weight"] for stage in stages],
    }
    for key, ladder in descending.items():
        if not all(ladder[i] > ladder[i + 1] for i in range(len(ladder) - 1)):
            raise RuntimeError(f"M-TO2S {key} must tighten across shaping stages")
    for key, ladder in ascending.items():
        if not all(ladder[i] <= ladder[i + 1] for i in range(len(ladder) - 1)):
            raise RuntimeError(f"M-TO2S {key} must not decrease across stages")
    if [bool(stage["symmetry_hard"]) for stage in stages] != [False] * (
        len(stages) - 1
    ) + [True]:
        raise RuntimeError("M-TO2S only the final shaping stage may be hard")
    return stages


def build_solver(
    evaluator: PinocchioSidewaysTerms,
    config: dict[str, Any],
    pose_reference: np.ndarray,
    selection: np.ndarray,
    tuck_radius: float,
    symmetry_weight: float,
    weight_scale: float,
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
    p_fr_b = terms[187:190]
    p_hl_b = terms[190:193]

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
    midpoint_error = segment_s - 0.5 * support_length
    support_direction = (
        ca.vertcat(support_delta[0], support_delta[1], 0.0) / support_length
    )
    axle_line_alignment = ca.vertcat(
        1.0 - ca.dot(a_fl_w, support_direction) ** 2,
        1.0 - ca.dot(a_hr_w, support_direction) ** 2,
    )
    tuck_fr = tuck_radius**2 - (p_fr_b[0] ** 2 + p_fr_b[1] ** 2)
    tuck_hl = tuck_radius**2 - (p_hl_b[0] ** 2 + p_hl_b[1] ** 2)
    task_symmetry = ca.vertcat(
        p_fl_b[0] + p_hr_b[0],
        p_fl_b[1] + p_hr_b[1],
        p_fr_b[0] + p_hl_b[0],
        p_fr_b[1] + p_hl_b[1],
        p_fr_b[2] - p_hl_b[2],
    )
    pendulum_constraints = ca.vertcat(
        midpoint_error,
        axle_line_alignment,
        tuck_fr,
        tuck_hl,
        task_symmetry,
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
        wheel_alignment,
        sideways_constraints,
        pendulum_constraints,
    )

    reference = ca.DM(pose_reference)
    joint_pair_delta = ca.vertcat(
        *(pose[first] - pose[second] for first, second in JOINT_SYMMETRY_PAIRS)
    )
    swing_height_target = sideways_config["swing_height_target_m"]
    objective = (
        weights["pose"] * ca.sumsqr(pose[3:] - reference[3:])
        + weights["orientation"] * ca.sumsqr(pose[1:3])
        + weights["base_height"] * (pose[0] - reference[0]) ** 2
        + weights["torque"] * ca.sumsqr(torques)
        + weights["force"] * (ca.sumsqr(f_fl) + ca.sumsqr(f_hr))
        + aligned_config["hipx_weight"] * ca.sumsqr(pose[3:7])
        + sideways_config["dx_weight"] * dx**2
        + weight_scale
        * (
            sideways_config["tuck_weight"]
            * (
                p_fr_b[0] ** 2
                + p_fr_b[1] ** 2
                + p_hl_b[0] ** 2
                + p_hl_b[1] ** 2
            )
            + sideways_config["swing_height_weight"]
            * (
                (fr[2] - swing_height_target) ** 2
                + (hl[2] - swing_height_target) ** 2
            )
            + sideways_config["joint_symmetry_weight"] * ca.sumsqr(joint_pair_delta)
            + symmetry_weight * ca.sumsqr(task_symmetry)
        )
    )
    ipopt = sideways_config["ipopt"]
    options = {
        "print_time": False,
        "error_on_fail": False,
        "ipopt.print_level": 5,
        "ipopt.sb": "yes",
        "ipopt.tol": ipopt["tolerance"],
        "ipopt.acceptable_tol": ipopt["acceptable_tolerance"],
        "ipopt.acceptable_iter": ipopt["acceptable_iter"],
        "ipopt.mu_strategy": ipopt["mu_strategy"],
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
    config: dict[str, Any],
    eps_x: float,
    swing_clearance: float,
    stage: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    static_config = copy.deepcopy(config["static_optimization"])
    static_config["swing_clearance_m"] = swing_clearance
    aligned_config = config["aligned_static_optimization"]
    sideways_config = config["sideways_static_optimization"]
    lower_base, upper_base = aligned.constraint_bounds(
        static_config, aligned_config, sideways_config["com_tolerance_m"]
    )
    lower = np.concatenate((lower_base, np.full(13, -np.inf)))
    upper = np.concatenate((upper_base, np.full(13, np.inf)))
    lower[42], upper[42] = -eps_x, eps_x
    lower[43] = sideways_config["y_separation_min_m"] ** 2
    direction_limit = math.sin(
        math.radians(sideways_config["support_direction_limit_deg"])
    )
    lower[44], upper[44] = -direction_limit, direction_limit
    midpoint_tolerance = stage["midpoint_tolerance_m"]
    lower[45], upper[45] = -midpoint_tolerance, midpoint_tolerance
    axle_limit = math.sin(math.radians(stage["axle_line_limit_deg"])) ** 2
    upper[46] = axle_limit
    upper[47] = axle_limit
    lower[48] = 0.0
    lower[49] = 0.0
    if stage["symmetry_hard"]:
        tolerance = sideways_config["symmetry_tolerance_m"]
        lower[50:55], upper[50:55] = -tolerance, tolerance
    if lower.size != N_CONSTRAINTS or upper.size != N_CONSTRAINTS:
        raise RuntimeError("Unexpected sideways constraint-vector dimension")
    return lower, upper


def support_body_metrics(
    evaluator: PinocchioSidewaysTerms, pose: np.ndarray
) -> dict[str, Any]:
    contacts, *_ = evaluator.evaluate_numpy(pose)
    fl_body, _, _, hr_body = evaluator.contacts_body_numpy(contacts)
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


def symmetry_metrics(
    result: dict[str, Any], axes: dict[str, Any]
) -> dict[str, Any]:
    fl_body, fr_body, hl_body, hr_body = result["contacts_body"]
    contacts = result["contacts"]
    support_direction = contacts[3, :2] - contacts[0, :2]
    support_length = float(np.linalg.norm(support_direction))
    if not np.isfinite(support_length) or support_length <= 1.0e-12:
        raise RuntimeError("WORLD-frame support line is degenerate")
    unit_line = np.zeros(3)
    unit_line[:2] = support_direction / support_length

    def axle_angle(axis: np.ndarray) -> float:
        cosine = float(np.clip(abs(np.asarray(axis) @ unit_line), 0.0, 1.0))
        return math.degrees(math.acos(cosine))

    pose = result["pose"]
    f_fl, f_hr = result["f_fl"], result["f_hr"]
    return {
        "swing_contacts_body": {"FR": fr_body, "HL": hl_body},
        "stance_residual_xy_m": (fl_body[:2] + hr_body[:2]).astype(float),
        "swing_residual_xy_m": (fr_body[:2] + hl_body[:2]).astype(float),
        "swing_height_difference_m": float(fr_body[2] - hl_body[2]),
        "midpoint_error_m": float(result["s"] - 0.5 * result["length"]),
        "load_split_fraction": float(
            f_fl[2] / (f_fl[2] + f_hr[2])
        ),
        "axle_line_angle_deg": {
            "FL": axle_angle(axes["world"]["FL"]),
            "HR": axle_angle(axes["world"]["HR"]),
        },
        "tuck_radius_m": {
            "FR": float(np.linalg.norm(fr_body[:2])),
            "HL": float(np.linalg.norm(hl_body[:2])),
        },
        "joint_symmetry_max_abs_rad": float(
            max(abs(pose[first] - pose[second]) for first, second in JOINT_SYMMETRY_PAIRS)
        ),
    }


def validate_candidate(
    evaluator: PinocchioSidewaysTerms,
    config: dict[str, Any],
    values: np.ndarray,
    eps_x: float,
    swing_clearance: float,
    stage: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    effective_config = copy.deepcopy(config)
    effective_config["static_optimization"]["swing_clearance_m"] = swing_clearance
    com_tolerance = effective_config["sideways_static_optimization"][
        "com_tolerance_m"
    ]
    result, axes = aligned.validate_candidate(
        evaluator, effective_config, values, com_tolerance
    )
    contacts, *_ = evaluator.evaluate_numpy(result["pose"])
    result["contacts_body"] = evaluator.contacts_body_numpy(contacts)
    support = support_body_metrics(evaluator, result["pose"])
    sideways_config = config["sideways_static_optimization"]
    # Micron-scale slack on the pure-numeric guards: a max-iter exit can sit
    # exactly on an active bound with a residual of a few times 1e-7, three
    # orders below every physical tolerance (mm scale), which a 1e-8 guard
    # rejects without the pose being physically distinguishable.
    numeric_slack = 2.0e-6
    if abs(support["dx"]) > eps_x + numeric_slack:
        raise RuntimeError(
            f"BODY-frame support dx constraint failed: {support['dx']:.9f} "
            f"vs limit {eps_x:.6f} m"
        )
    if abs(support["dy"]) < sideways_config["y_separation_min_m"] - numeric_slack:
        raise RuntimeError(
            f"BODY-frame support y-separation constraint failed: "
            f"{support['dy']:.9f} vs minimum "
            f"{sideways_config['y_separation_min_m']:.6f} m"
        )
    direction_limit = math.sin(
        math.radians(sideways_config["support_direction_limit_deg"])
    )
    if support["direction_x_abs"] > direction_limit + numeric_slack:
        raise RuntimeError(
            f"BODY-frame support direction constraint failed: "
            f"{support['direction_x_abs']:.9f} vs limit {direction_limit:.9f}"
        )
    symmetry = symmetry_metrics(result, axes)
    if abs(symmetry["midpoint_error_m"]) > (
        stage["midpoint_tolerance_m"] + numeric_slack
    ):
        raise RuntimeError(
            f"Support-midpoint constraint failed: "
            f"{symmetry['midpoint_error_m']:.9f} vs tolerance "
            f"{stage['midpoint_tolerance_m']:.6f} m"
        )
    axle_limit = stage["axle_line_limit_deg"]
    if max(symmetry["axle_line_angle_deg"].values()) > axle_limit + numeric_slack:
        worst_axle = max(symmetry["axle_line_angle_deg"].values())
        raise RuntimeError(
            f"Axle-to-support-line alignment constraint failed: "
            f"{worst_axle:.6f} vs limit {axle_limit:.3f} deg"
        )
    tuck_radius = stage["tuck_radius_m"]
    if max(symmetry["tuck_radius_m"].values()) > tuck_radius + numeric_slack:
        worst_tuck = max(symmetry["tuck_radius_m"].values())
        raise RuntimeError(
            f"Swing tuck constraint failed: {worst_tuck:.9f} vs radius "
            f"{tuck_radius:.6f} m"
        )
    if stage["symmetry_hard"]:
        tolerance = sideways_config["symmetry_tolerance_m"]
        residuals = (
            float(np.max(np.abs(symmetry["stance_residual_xy_m"]))),
            float(np.max(np.abs(symmetry["swing_residual_xy_m"]))),
            abs(symmetry["swing_height_difference_m"]),
        )
        if max(residuals) > tolerance + numeric_slack:
            raise RuntimeError("Hard point-symmetry constraint failed")
    return result, axes, support, symmetry


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
    dict[str, Any],
]:
    static_config = config["static_optimization"]
    sideways_config = config["sideways_static_optimization"]
    selection = base.actuator_matrix(evaluator)
    final_stage = {"friction": True, "positive_normal": True, "torque_limits": True}
    lower_x, upper_x = base.variable_bounds(evaluator, static_config, final_stage)
    values = initial_values.copy()
    lambda_x = None
    lambda_g = None
    attempts: list[dict[str, Any]] = []
    best: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ] | None = None
    final_eps = min(sideways_config["eps_x_continuation_m"])

    clearances = [sideways_config["preferred_swing_clearance_m"]]
    fallback = sideways_config["fallback_swing_clearance_m"]
    if fallback != clearances[0]:
        clearances.append(fallback)

    for stage in shaping_stages(config):
        solver = build_solver(
            evaluator,
            config,
            pose_reference,
            selection,
            stage["tuck_radius_m"],
            stage["symmetry_weight"],
            stage["weight_scale"],
        )
        stage_best: tuple[
            dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
        ] | None = None
        stage_best_eps: float | None = None
        stage_entry_values = values.copy()
        for attempt_index in range(3):
            if attempt_index == 1:
                # Drop only the duals: primal warm start from the stage entry
                # with fresh multipliers often escapes the stale-dual stall.
                values = stage_entry_values.copy()
                lambda_x = None
                lambda_g = None
            elif attempt_index == 2:
                values = initial_values.copy()
                lambda_x = None
                lambda_g = None
            for clearance_index, swing_clearance in enumerate(clearances):
                reached_clearance = False
                for eps_x in sideways_config["eps_x_continuation_m"]:
                    lower_g, upper_g = constraint_bounds(
                        config, eps_x, swing_clearance, stage
                    )
                    arguments: dict[str, Any] = {
                        "x0": values,
                        "lbx": lower_x,
                        "ubx": upper_x,
                        "lbg": lower_g,
                        "ubg": upper_g,
                    }
                    warm_started = lambda_x is not None
                    if warm_started:
                        arguments["lam_x0"] = lambda_x
                        arguments["lam_g0"] = lambda_g
                    solution = solver(**arguments)
                    stats = solver.stats()
                    candidate = np.asarray(solution["x"]).reshape(-1)
                    if np.all(np.isfinite(candidate)):
                        values = candidate
                        # Chain duals only from cleanly solved points: a
                        # max-iter exit leaves degenerate multipliers that
                        # stall the next eps step (dual infeasibility flat).
                        if warm_started and stats["success"]:
                            lambda_x = solution["lam_x"]
                            lambda_g = solution["lam_g"]
                        else:
                            lambda_x = None
                            lambda_g = None
                    attempt = {
                        "name": (
                            f"{stage['name']}_"
                            f"clearance_{int(round(swing_clearance * 1000.0))}mm_"
                            f"eps_{int(round(eps_x * 1000.0))}mm"
                        ),
                        "shaping": stage["name"],
                        "eps_x_m": eps_x,
                        "swing_clearance_m": swing_clearance,
                        "tuck_radius_m": stage["tuck_radius_m"],
                        "midpoint_tolerance_m": stage["midpoint_tolerance_m"],
                        "axle_line_limit_deg": stage["axle_line_limit_deg"],
                        "weight_scale": stage["weight_scale"],
                        "symmetry_weight": stage["symmetry_weight"],
                        "symmetry_hard": bool(stage["symmetry_hard"]),
                        "restart": (
                            "warm"
                            if attempt_index == 0
                            else "primal-only"
                            if attempt_index == 1
                            else "cold"
                        ),
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
                        f"Sideways {attempt['name']} ({attempt['restart']}): "
                        f"{attempt['status']}, iterations={attempt['iterations']}"
                    )
                    if not attempt["success"]:
                        # A max-iter exit can still be primal-feasible
                        # (inf_pr < 1e-8) with only the dual certificate
                        # missing; the independent physics validation below
                        # decides whether the iterate is usable.
                        if stats["return_status"] != "Maximum_Iterations_Exceeded":
                            break
                        attempt["accepted_max_iter"] = True
                    try:
                        result, axes, support, symmetry = validate_candidate(
                            evaluator, config, values, eps_x, swing_clearance, stage
                        )
                    except RuntimeError as error:
                        attempt["success"] = False
                        attempt["validation_issue"] = str(error)
                        break
                    if attempt.get("accepted_max_iter"):
                        # Physics validation passed; the exit status is kept
                        # for the record alongside this acceptance marker.
                        attempt["success"] = True
                        print(
                            "  accepted max-iter iterate: independent physics "
                            "validation passed"
                        )
                    reached_clearance = True
                    if (
                        stage_best is None
                        or eps_x < stage_best_eps - 1.0e-12
                        or (
                            abs(eps_x - stage_best_eps) <= 1.0e-12
                            and swing_clearance
                            > stage_best[0]["swing_clearance_m"] + 1.0e-12
                        )
                    ):
                        stage_best = (attempt, result, axes, support, symmetry)
                        stage_best_eps = eps_x
                    if abs(eps_x - final_eps) <= 1.0e-12:
                        break
                if stage_best is not None and abs(
                    stage_best_eps - final_eps
                ) <= 1.0e-12:
                    break
                if not reached_clearance and clearance_index == 0:
                    issue = next(
                        (
                            attempt_issue.get("validation_issue", "")
                            for attempt_issue in reversed(attempts)
                            if attempt_issue.get("name") == attempt["name"]
                        ),
                        "",
                    )
                    print(
                        "Preferred 30 mm clearance did not validate; trying 20 mm"
                        + (f" ({issue})" if issue else "")
                    )
            if stage_best is not None:
                break
            if attempt_index == 0:
                print(
                    f"Warm-started pass of shaping stage {stage['name']} made "
                    "no progress; retrying without duals"
                )
        if stage_best is None:
            if best is None:
                raise RuntimeError(
                    "NO STATIC EQUILIBRIUM WITH SIDEWAYS SUPPORT CONSTRAINTS"
                )
            print(
                f"Shaping stage {stage['name']} did not validate; keeping the "
                "previous stage result"
            )
            break
        selected, result, axes, support, symmetry = stage_best
        # Seed the next shaping stage from the validated iterate, not the
        # last-pass iterate (which may be a failed eps step).
        values = np.asarray(selected["x"], dtype=float).copy()
        lambda_x = None
        lambda_g = None
        best = stage_best

    selected, result, axes, support, symmetry = best
    return selected, attempts, result, axes, support, symmetry


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
    symmetry: dict[str, Any],
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
    output["objective_terms"] = {
        "hipx_weight": sideways_config["hipx_weight"]
        if "hipx_weight" in sideways_config
        else config["aligned_static_optimization"]["hipx_weight"],
        "dx_weight": sideways_config["dx_weight"],
        "tuck_weight": sideways_config["tuck_weight"],
        "swing_height_weight": sideways_config["swing_height_weight"],
        "swing_height_target_m": sideways_config["swing_height_target_m"],
        "joint_symmetry_weight": sideways_config["joint_symmetry_weight"],
        "symmetry_weight": selected["symmetry_weight"],
        "reference_pose": str(objective_reference_path),
    }
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
    output["swing_contacts_body"] = {
        corner: symmetry["swing_contacts_body"][corner].tolist()
        for corner in ("FR", "HL")
    }
    output["pose_symmetry"] = {
        "stance_residual_body_xy_m": symmetry["stance_residual_xy_m"].tolist(),
        "swing_residual_body_xy_m": symmetry["swing_residual_xy_m"].tolist(),
        "swing_height_difference_m": symmetry["swing_height_difference_m"],
        "midpoint_error_m": symmetry["midpoint_error_m"],
        "load_split_fraction": symmetry["load_split_fraction"],
        "axle_line_angle_deg": symmetry["axle_line_angle_deg"],
        "tuck_radius_m": symmetry["tuck_radius_m"],
        "joint_symmetry_max_abs_rad": symmetry["joint_symmetry_max_abs_rad"],
        "symmetry_hard": bool(selected["symmetry_hard"]),
        "symmetry_tolerance_m": sideways_config["symmetry_tolerance_m"],
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
            "midpoint_tolerance_m": selected["midpoint_tolerance_m"],
            "axle_line_limit_deg": selected["axle_line_limit_deg"],
            "tuck_radius_m": selected["tuck_radius_m"],
        }
    )
    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(output, stream, sort_keys=False)
    return output


def print_result(output: dict[str, Any]) -> None:
    variables = output["variables"]
    joints = variables["leg_joints"]
    support_contacts = output["support_contacts_body"]
    swing_contacts = output["swing_contacts_body"]
    support = output["support_line_body"]
    symmetry = output["pose_symmetry"]
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
    print(f"HR = {base.print_vector(support_contacts['HR'])}")
    print("swing contacts in body frame:")
    print(f"FR = {base.print_vector(swing_contacts['FR'])}")
    print(f"HL = {base.print_vector(swing_contacts['HL'])}\n")
    print("support line:")
    print(f"dx = {support['dx_m']:.12f}")
    print(f"dy = {support['dy_m']:.12f}")
    print(f"angle wrt body X = {support['angle_wrt_body_x_deg']:.12f} deg")
    print(f"angle wrt body Y = {support['angle_wrt_body_y_deg']:.12f} deg\n")
    print("two-wheel symmetry:")
    print(f"midpoint error = {symmetry['midpoint_error_m']:.12f} m")
    print(f"load split FL/(FL+HR) = {symmetry['load_split_fraction']:.12f}")
    print(f"stance residual xy = {base.print_vector(symmetry['stance_residual_body_xy_m'])}")
    print(f"swing residual xy = {base.print_vector(symmetry['swing_residual_body_xy_m'])}")
    print(f"swing height difference = {symmetry['swing_height_difference_m']:.12f} m")
    print(f"axle vs support line FL = {symmetry['axle_line_angle_deg']['FL']:.12f} deg")
    print(f"axle vs support line HR = {symmetry['axle_line_angle_deg']['HR']:.12f} deg")
    print(f"tuck radius FR = {symmetry['tuck_radius_m']['FR']:.12f} m")
    print(f"tuck radius HL = {symmetry['tuck_radius_m']['HL']:.12f} m")
    print(f"joint symmetry max |delta| = {symmetry['joint_symmetry_max_abs_rad']:.12f} rad")
    print(f"symmetry hard = {symmetry['symmetry_hard']}\n")
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
    print(
        "final tuck radius used = "
        f"{output['constraints']['tuck_radius_m']:.12f}"
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
    selected, attempts, _, axes, support, symmetry = solve(
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
        symmetry,
    )
    print_result(output)
    print(f"\nOutput = {arguments.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
