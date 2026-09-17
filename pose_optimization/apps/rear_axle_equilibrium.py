#!/usr/bin/env python3
"""HL-HR rear-wheel kinematic pose: upright body, extended rear legs, tucked front.

Run from pose_optimization with:
    python apps/rear_axle_equilibrium.py

This solves geometry only. A PASS does not establish static force/torque balance.
"""

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


CORNERS = kin.WHEEL_CORNERS
REAR = ("HL", "HR")
FRONT = ("FL", "FR")
# 4 tread contacts + CoM + 4 hip origins + 4 knee origins + 4 wheel centres.
GEOMETRY_SIZE = 4 * 3 + 3 + 4 * 3 * 3
EPS = 1.0e-10


class RearPoseGeometry(kin.PinocchioGeometry):
    """Reuse the audited contact helper and add joint origins to the callback."""

    def __init__(self, urdf: Path, helper: kin.TreadContactLibrary) -> None:
        # Parent calls self.construct(), which queries get_sparsity_out().
        self.hip_ids: dict[str, int] = {}
        self.knee_ids: dict[str, int] = {}
        super().__init__(urdf, helper)
        for corner in CORNERS:
            for suffix, destination in (
                ("HipY_joint", self.hip_ids),
                ("Knee_joint", self.knee_ids),
            ):
                name = f"{corner}_{suffix}"
                joint_id = self.model.getJointId(name)
                if joint_id == 0 or self.model.names[joint_id] != name:
                    raise RuntimeError(f"Missing audited joint: {name}")
                destination[corner] = joint_id

    def get_sparsity_out(self, index: int) -> ca.Sparsity:
        del index
        return ca.Sparsity.dense(GEOMETRY_SIZE, 1)

    def pose(self, x: np.ndarray) -> dict[str, Any]:
        contacts, com, q = super().evaluate_numpy(x)
        # centerOfMass() above has run forward kinematics on exactly q.
        hips = np.array(
            [self.data.oMi[self.hip_ids[c]].translation.copy() for c in CORNERS]
        )
        knees = np.array(
            [self.data.oMi[self.knee_ids[c]].translation.copy() for c in CORNERS]
        )
        wheels = np.array(
            [self.data.oMf[self.frame_ids[c]].translation.copy() for c in CORNERS]
        )
        if not all(np.isfinite(a).all() for a in (contacts, com, hips, knees, wheels)):
            raise RuntimeError("Nonfinite Pinocchio frame placement")
        return {
            "contacts": contacts, "com": com, "q": q,
            "hips": hips, "knees": knees, "wheels": wheels,
        }

    def eval(self, args: list[ca.DM]) -> list[ca.DM]:
        p = self.pose(np.asarray(args[0]).reshape(-1))
        return [ca.DM(np.concatenate((
            p["contacts"].ravel(), p["com"],
            p["hips"].ravel(), p["knees"].ravel(), p["wheels"].ravel(),
        )))]


def slices(g: ca.MX) -> tuple[dict[str, ca.MX], ca.MX, dict[str, ca.MX],
                              dict[str, ca.MX], dict[str, ca.MX]]:
    contacts = {c: g[3 * i:3 * i + 3] for i, c in enumerate(CORNERS)}
    com = g[12:15]
    hips = {c: g[15 + 3 * i:18 + 3 * i] for i, c in enumerate(CORNERS)}
    knees = {c: g[27 + 3 * i:30 + 3 * i] for i, c in enumerate(CORNERS)}
    wheels = {c: g[39 + 3 * i:42 + 3 * i] for i, c in enumerate(CORNERS)}
    return contacts, com, hips, knees, wheels


def cos_between(a: ca.MX, b: ca.MX) -> ca.MX:
    return ca.dot(a, b) / (ca.norm_2(a) * ca.norm_2(b) + EPS)


def measured(p: dict[str, Any], nominal_front_length: dict[str, float]) -> dict[str, Any]:
    contacts = dict(zip(CORNERS, p["contacts"], strict=True))
    hips = dict(zip(CORNERS, p["hips"], strict=True))
    knees = dict(zip(CORNERS, p["knees"], strict=True))
    wheels = dict(zip(CORNERS, p["wheels"], strict=True))
    a, b = contacts["HL"][:2], contacts["HR"][:2]
    delta = b - a
    length = float(np.linalg.norm(delta))
    if length < 0.1:
        raise RuntimeError("Rear tread support segment is degenerate")
    d = delta / length
    offset = p["com"][:2] - a
    midpoint = (a + b) / 2.0
    m: dict[str, Any] = {
        "com_line_signed_error_m": float(np.array([-d[1], d[0]]) @ offset),
        "com_midpoint_error_xy_m": (p["com"][:2] - midpoint).tolist(),
        "s_over_L": float(d @ offset / length),
        "support_length_m": length,
        "tread_z_m": {c: float(contacts[c][2]) for c in CORNERS},
        "front_hip_wheel_length_ratio": {},
        "rear_thigh_shank_angle_deg": {},
        "rear_leg_world_vertical_angle_deg": {},
    }
    for c in REAR:
        thigh = knees[c] - hips[c]
        shank = wheels[c] - knees[c]
        leg = wheels[c] - hips[c]
        if min(np.linalg.norm(thigh), np.linalg.norm(shank), np.linalg.norm(leg)) < 1e-5:
            raise RuntimeError(f"Degenerate {c} leg geometry")
        straight_cos = float(thigh @ shank / (np.linalg.norm(thigh) * np.linalg.norm(shank)))
        vertical_cos = float(-leg[2] / np.linalg.norm(leg))
        m["rear_thigh_shank_angle_deg"][c] = math.degrees(math.acos(np.clip(straight_cos, -1, 1)))
        m["rear_leg_world_vertical_angle_deg"][c] = math.degrees(math.acos(np.clip(vertical_cos, -1, 1)))
    for c in FRONT:
        m["front_hip_wheel_length_ratio"][c] = float(
            np.linalg.norm(wheels[c] - hips[c]) / nominal_front_length[c]
        )
    return m


def nominal_front_lengths(geometry: RearPoseGeometry, cfg: dict[str, Any]) -> dict[str, float]:
    p = geometry.pose(kin.initial_guess(cfg))
    return {
        c: float(np.linalg.norm(p["wheels"][CORNERS.index(c)]
                                - p["hips"][CORNERS.index(c)]))
        for c in FRONT
    }


def stages(args: argparse.Namespace) -> list[dict[str, float | str]]:
    # Each stage is checked before its result can warm-start the next.
    return [
        dict(name="A", pitch_min=-65, pitch_max=-30, vertical=45,
             straight=75, front_ratio=1.00, clearance=0.015,
             line=0.025, midpoint=0.05),
        dict(name="B", pitch_min=-80, pitch_max=-55, vertical=25,
             straight=45, front_ratio=0.93, clearance=0.035,
             line=0.010, midpoint=0.03),
        dict(name="C", pitch_min=-89, pitch_max=-70, vertical=args.rear_vertical_deg,
             straight=args.rear_straight_deg, front_ratio=args.front_fold_ratio,
             clearance=args.clearance, line=args.com_tolerance,
             midpoint=args.midpoint_tolerance),
    ]


def build_solver(geometry: RearPoseGeometry, cfg: dict[str, Any],
                 front_lengths: dict[str, float]) -> ca.Function:
    x = ca.MX.sym("x", kin.N_VARIABLES)
    g = geometry(x)
    contacts, com, hips, knees, wheels = slices(g)
    delta = contacts["HR"][:2] - contacts["HL"][:2]
    length = ca.norm_2(delta)
    offset = com[:2] - contacts["HL"][:2]
    line_error = (-delta[1] * offset[0] + delta[0] * offset[1]) / length
    segment = ca.dot(delta, offset) / length
    midpoint_error = com[:2] - (contacts["HL"][:2] + contacts["HR"][:2]) / 2
    straight = {
        c: cos_between(knees[c] - hips[c], wheels[c] - knees[c]) for c in REAR
    }
    vertical = {
        c: -(wheels[c][2] - hips[c][2]) / (ca.norm_2(wheels[c] - hips[c]) + EPS)
        for c in REAR
    }
    fold = {
        c: ca.norm_2(wheels[c] - hips[c]) / front_lengths[c] for c in FRONT
    }
    nominal = ca.DM([cfg["nominal"]["joint_positions"][name] for name in kin.LEG_JOINT_NAMES])
    # Image-based pose preference: front rises as body x points upward.
    # Physical pose conditions remain hard constraints below.
    objective = (
        15.0 * (x[2] - math.radians(-80)) ** 2
        + 15.0 * x[1] ** 2
        + 1000.0 * ca.sumsqr(midpoint_error)
        + 400.0 * sum((1 - straight[c]) ** 2 for c in REAR)
        + 100.0 * sum((1 - vertical[c]) ** 2 for c in REAR)
        + 2.0 * sum(fold[c] ** 2 for c in FRONT)
        + 0.02 * ca.sumsqr(x[3:] - nominal)
    )
    constraints = ca.vertcat(
        contacts["HL"][2], contacts["HR"][2],
        contacts["FL"][2], contacts["FR"][2],
        line_error, midpoint_error[0], midpoint_error[1],
        segment, length - segment,
        straight["HL"], straight["HR"], vertical["HL"], vertical["HR"],
        fold["FL"], fold["FR"],
    )
    ipopt = cfg["optimization"]["ipopt"]
    return ca.nlpsol("rear_axle_pose", "ipopt",
                     {"x": x, "f": objective, "g": constraints}, {
                         "print_time": False, "error_on_fail": False,
                         "ipopt.print_level": 0, "ipopt.sb": "yes",
                         "ipopt.tol": ipopt["tolerance"],
                         "ipopt.acceptable_tol": ipopt["acceptable_tolerance"],
                         "ipopt.constr_viol_tol": ipopt["constraint_violation_tolerance"],
                         "ipopt.max_iter": max(1200, int(ipopt["max_iterations"])),
                         "ipopt.bound_relax_factor": 0.0,
                         "ipopt.hessian_approximation": "limited-memory",
                     })


def bounds(geometry: RearPoseGeometry, cfg: dict[str, Any],
           stage: dict[str, Any], roll_limit: float) -> tuple[np.ndarray, np.ndarray]:
    lbx, ubx = kin.variable_bounds(geometry, cfg)
    lbx[0], ubx[0] = 0.10, 1.30
    lbx[1], ubx[1] = -math.radians(roll_limit), math.radians(roll_limit)
    lbx[2] = math.radians(stage["pitch_min"])
    ubx[2] = math.radians(stage["pitch_max"])
    return lbx, ubx


def constraint_bounds(stage: dict[str, Any], segment_margin: float) -> tuple[list[float], list[float]]:
    c = math.cos(math.radians(stage["straight"]))
    v = math.cos(math.radians(stage["vertical"]))
    return (
        [0, 0, stage["clearance"], stage["clearance"],
         -stage["line"], -stage["midpoint"], -stage["midpoint"],
         segment_margin, segment_margin, c, c, v, v, 0, 0],
        [0, 0, np.inf, np.inf, stage["line"], stage["midpoint"],
         stage["midpoint"], np.inf, np.inf, 1, 1, 1, 1,
         stage["front_ratio"], stage["front_ratio"]],
    )


def check(geometry: RearPoseGeometry, cfg: dict[str, Any], x: np.ndarray,
          stage: dict[str, Any], args: argparse.Namespace,
          front_lengths: dict[str, float]) -> tuple[dict[str, Any], dict[str, Any]]:
    p = geometry.pose(x)
    m = measured(p, front_lengths)
    slack = 5e-5
    for c in REAR:
        if abs(m["tread_z_m"][c]) > 1e-6:
            raise RuntimeError(f"{c} tread is not on ground")
        if m["rear_thigh_shank_angle_deg"][c] > stage["straight"] + 0.02:
            raise RuntimeError(f"{c} rear leg is not sufficiently extended")
        if m["rear_leg_world_vertical_angle_deg"][c] > stage["vertical"] + 0.02:
            raise RuntimeError(f"{c} rear leg is not vertical enough")
    for c in FRONT:
        if m["tread_z_m"][c] < stage["clearance"] - slack:
            raise RuntimeError(f"{c} tread clearance too low")
        if m["front_hip_wheel_length_ratio"][c] > stage["front_ratio"] + slack:
            raise RuntimeError(f"{c} front leg is not sufficiently tucked")
    if abs(m["com_line_signed_error_m"]) > stage["line"] + slack:
        raise RuntimeError("CoM is outside the support-line tolerance")
    if max(abs(v) for v in m["com_midpoint_error_xy_m"]) > stage["midpoint"] + slack:
        raise RuntimeError("CoM is too far from rear axle midpoint")
    length = m["support_length_m"]
    s = m["s_over_L"] * length
    if not (args.segment_margin - slack <= s <= length - args.segment_margin + slack):
        raise RuntimeError("CoM projection is outside the support segment")
    lbx, ubx = bounds(geometry, cfg, stage, args.roll_limit)
    if np.any(x < lbx - slack) or np.any(x > ubx + slack):
        raise RuntimeError("Pose violates joint or base bounds")
    return p, m


def starting_points(cfg: dict[str, Any], pose_dir: Path,
                    seed_path: Path | None) -> list[np.ndarray]:
    nominal = kin.initial_guess(cfg)
    seeds = []
    if seed_path and seed_path.exists():
        with seed_path.open(encoding="utf-8") as stream:
            prior = yaml.safe_load(stream)["variables"]
        seed = nominal.copy()
        seed[0:3] = [prior["base_z"], prior["base_roll"], prior["base_pitch"]]
        seed[3:] = [prior["leg_joints"][n] for n in kin.LEG_JOINT_NAMES]
        seeds.append(seed)
    for pitch in (-40, -55, -30):
        seed = nominal.copy()
        seed[2] = math.radians(pitch)
        # Warm-start front knee flexion without prescribing a final joint pose.
        for name in ("FL_Knee_joint", "FR_Knee_joint"):
            seed[3 + kin.LEG_JOINT_NAMES.index(name)] = 1.9
        seeds.append(seed)
    return seeds


def audit_rear_extension(geometry: RearPoseGeometry, cfg: dict[str, Any],
                         args: argparse.Namespace) -> None:
    """Cheap necessary check before IPOPT; the knee may never become collinear."""
    x = kin.initial_guess(cfg)
    lb, ub = kin.variable_bounds(geometry, cfg)
    for corner in REAR:
        idx = 3 + kin.LEG_JOINT_NAMES.index(f"{corner}_Knee_joint")
        minimum = math.inf
        best_knee = math.nan
        for knee in np.linspace(lb[idx], ub[idx], 65):
            x[idx] = knee
            p = geometry.pose(x)
            hip = p["hips"][CORNERS.index(corner)]
            knee_point = p["knees"][CORNERS.index(corner)]
            wheel = p["wheels"][CORNERS.index(corner)]
            thigh = knee_point - hip
            shank = wheel - knee_point
            cosine = float(thigh @ shank / (np.linalg.norm(thigh) * np.linalg.norm(shank)))
            angle = math.degrees(math.acos(np.clip(cosine, -1, 1)))
            if angle < minimum:
                minimum, best_knee = angle, float(knee)
        print(f"{corner} audited min thigh/shank bend: {minimum:.2f} deg "
              f"at knee={best_knee:.4f} rad (sampled URDF joint range)")
        if minimum > args.rear_straight_deg + 0.1:
            raise RuntimeError(
                f"{corner}: the requested {args.rear_straight_deg:g} deg straightness "
                f"is below the sampled URDF minimum {minimum:.2f} deg. "
                "Use a reachable --rear-straight-deg only if a visibly bent "
                "rear leg meets your task; the script will not silently relax it."
            )
        x[idx] = cfg["nominal"]["joint_positions"][f"{corner}_Knee_joint"]


def solve(geometry: RearPoseGeometry, cfg: dict[str, Any],
          args: argparse.Namespace, pose_dir: Path) -> dict[str, Any]:
    front_lengths = nominal_front_lengths(geometry, cfg)
    solver = build_solver(geometry, cfg, front_lengths)
    stage_list = stages(args)
    candidates = []
    for seed_idx, start in enumerate(starting_points(cfg, pose_dir, args.seed)):
        x = start.copy()
        history = []
        for stage in stage_list:
            lbx, ubx = bounds(geometry, cfg, stage, args.roll_limit)
            x = np.clip(x, lbx + 1e-7, ubx - 1e-7)
            lbg, ubg = constraint_bounds(stage, args.segment_margin)
            try:
                result = solver(x0=x, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
                stats = solver.stats()
                x = np.array(result["x"], dtype=float).reshape(-1)
                if not bool(stats.get("success")):
                    raise RuntimeError(str(stats.get("return_status")))
                _, metrics = check(geometry, cfg, x, stage, args, front_lengths)
                history.append({
                    "stage": stage["name"], "status": stats["return_status"],
                    "iterations": int(stats["iter_count"]),
                    "pitch_deg": math.degrees(float(x[2])),
                    "rear_bend_deg": metrics["rear_thigh_shank_angle_deg"],
                })
                print(f"[seed {seed_idx}, stage {stage['name']}] "
                      f"{stats['return_status']} ({stats['iter_count']} iterations)")
            except (RuntimeError, ValueError) as exc:
                print(f"[seed {seed_idx}, stage {stage['name']}] FAIL: {exc}")
                break
        else:
            p, metrics = check(geometry, cfg, x, stage_list[-1], args, front_lengths)
            candidates.append((float(result["f"]), x.copy(), p, metrics, history))
    if not candidates:
        raise RuntimeError(
            "No feasible upright HL-HR pose. Inspect the stage and rear bend "
            "diagnostics; do not treat IPOPT's last iterate as a valid pose. "
            "A 5 degree collinearity limit may exceed the URDF knee range."
        )
    objective, x, p, metrics, history = min(candidates, key=lambda item: item[0])
    return dict(objective=objective, x=x, pose=p, metrics=metrics,
                history=history, front_lengths=front_lengths)


def save_result(path: Path, urdf: Path, solution: dict[str, Any],
                args: argparse.Namespace) -> None:
    x, p = solution["x"], solution["pose"]
    joint_values = {
        n: float(x[3 + i]) for i, n in enumerate(kin.LEG_JOINT_NAMES)
    }
    output = {
        "milestone": "HL-HR upright kinematic feasibility",
        "result": "PASS",
        "scope": "kinematic only; static torques, friction and dynamic balance not evaluated",
        "model": {"urdf_path": str(urdf)},
        "stance": list(REAR), "swing": list(FRONT),
        "objective": solution["objective"],
        "variables": {
            "base_x": 0.0, "base_y": 0.0, "base_z": float(x[0]),
            "base_roll": float(x[1]), "base_pitch": float(x[2]),
            "base_yaw": 0.0, "leg_joints": joint_values,
            "wheel_angles": {n: 0.0 for n in kin.WHEEL_NAMES},
        },
        "q_prior_isaac_lab_order": {
            "joint_names": list(kin.LEG_JOINT_NAMES),
            "joint_positions": [joint_values[n] for n in kin.LEG_JOINT_NAMES],
        },
        "pinocchio_q_xyzw": p["q"].tolist(),
        "com_world_xyz": p["com"].tolist(),
        "physical_tread_contacts_world": {
            c: p["contacts"][i].tolist() for i, c in enumerate(CORNERS)
        },
        "rear_leg_points_world": {
            c: {
                "hip": p["hips"][CORNERS.index(c)].tolist(),
                "knee": p["knees"][CORNERS.index(c)].tolist(),
                "wheel_center": p["wheels"][CORNERS.index(c)].tolist(),
            } for c in REAR
        },
        "metrics": solution["metrics"],
        "constraints": {
            "pitch_deg": [-89.0, -70.0],
            "rear_straight_angle_max_deg": args.rear_straight_deg,
            "rear_world_vertical_angle_max_deg": args.rear_vertical_deg,
            "front_length_vs_nominal_max": args.front_fold_ratio,
            "front_tread_clearance_min_m": args.clearance,
            "com_line_tolerance_m": args.com_tolerance,
            "com_midpoint_xy_tolerance_m": args.midpoint_tolerance,
            "support_segment_margin_m": args.segment_margin,
            "joint_limit_margin_rad": args.joint_margin,
        },
        "continuation": solution["history"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(output, stream, sort_keys=False)


def parse_args() -> argparse.Namespace:
    pose_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=pose_dir / "config/equilibrium.yaml")
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument("--contact-lib", type=Path,
                        default=Path(os.environ.get(
                            "VQR_TREAD_CONTACT_LIBRARY",
                            str(pose_dir / "build/libtread_contact_c_api.so"))))
    parser.add_argument("--seed", type=Path, default=pose_dir /
                        "output/kinematic_equilibrium_HL_HR.yaml")
    parser.add_argument("--output", type=Path, default=pose_dir /
                        "output/kinematic_equilibrium_HL_HR_upright.yaml")
    parser.add_argument("--clearance", type=float, default=0.05)
    parser.add_argument("--rear-straight-deg", type=float, default=5.0)
    parser.add_argument("--rear-vertical-deg", type=float, default=10.0)
    parser.add_argument("--front-fold-ratio", type=float, default=0.85)
    parser.add_argument("--com-tolerance", type=float, default=0.002)
    parser.add_argument("--midpoint-tolerance", type=float, default=0.010)
    parser.add_argument("--segment-margin", type=float, default=0.030)
    parser.add_argument("--joint-margin", type=float, default=0.05)
    parser.add_argument("--roll-limit", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = args.config.resolve()
    pose_dir = cfg_path.parent.parent
    with cfg_path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    urdf = (args.urdf or (pose_dir / cfg["model"]["urdf_path"])).resolve()
    if not urdf.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf}")
    if not args.contact_lib.is_file():
        raise FileNotFoundError(
            f"Contact library missing: {args.contact_lib}; build pose_optimization first"
        )
    if not (0 < args.rear_straight_deg < 90 and 0 < args.rear_vertical_deg < 90
            and 0 < args.front_fold_ratio <= 1 and args.clearance > 0
            and args.com_tolerance > 0 and args.midpoint_tolerance > 0):
        raise ValueError("Invalid pose tolerances")
    helper = kin.TreadContactLibrary(args.contact_lib.resolve(), urdf)
    geometry = RearPoseGeometry(urdf, helper)
    cfg["optimization"]["joint_margin_rad"] = args.joint_margin
    audit_rear_extension(geometry, cfg, args)
    result = solve(geometry, cfg, args, pose_dir)
    # A fresh FK on the final decision variables guards against stale callback data.
    p, m = check(geometry, cfg, result["x"], stages(args)[-1], args,
                 result["front_lengths"])
    result["pose"], result["metrics"] = p, m
    save_result(args.output.resolve(), urdf, result, args)
    x = result["x"]
    print("\nHL-HR UPRIGHT KINEMATIC RESULT: PASS")
    print(f"base z = {x[0]:.5f} m; pitch = {math.degrees(x[2]):.3f} deg")
    for c in REAR:
        print(f"{c}: thigh/shank = {m['rear_thigh_shank_angle_deg'][c]:.2f} deg, "
              f"world vertical = {m['rear_leg_world_vertical_angle_deg'][c]:.2f} deg")
    for c in FRONT:
        print(f"{c}: tread clearance = {m['tread_z_m'][c]:.3f} m, "
              f"fold ratio = {m['front_hip_wheel_length_ratio'][c]:.3f}")
    print(f"CoM line error = {m['com_line_signed_error_m']:.5f} m; "
          f"s/L = {m['s_over_L']:.4f}")
    print(f"Saved: {args.output.resolve()}")
    print("Kinematic result only: run a separate HL-HR static dynamics check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
