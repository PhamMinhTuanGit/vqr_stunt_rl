"""Reconstruct POS certificates and their individual blockers from audit CSVs.

Supports one or many environments. Rates count settled steps; overlapping
blockers must not be summed. This does not simulate removing a condition.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(directory):
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["cycles"] != 1:
        raise ValueError("Phase summaries require one 0 → positive → 0 cycle (--cycles 1).")
    data = np.genfromtxt(directory / "telemetry.csv", delimiter=",", names=True)
    ids = data["env_index"].astype(int) if "env_index" in data.dtype.names else np.zeros(len(data), int)
    params = metadata["differential_parameters"]
    dt = metadata["step_dt"]
    wheels = ("fl", "hr")

    def pair(suffix):
        return np.stack([data[f"audit_{wheel}_{suffix}"] for wheel in wheels], 1)

    target, measured, motor = pair("target"), pair("measured"), pair("motor_speed")
    slip = pair("rolling_residual")
    scale = np.maximum(params["speed_std"], params["relative_std"] * np.abs(target))
    fraction = params["wheel_radius"] * np.abs(motor) / np.maximum(np.abs(target), 1.e-4)
    tangent_norm = np.stack([np.hypot(data[f"audit_{w}_axle_x"], data[f"audit_{w}_axle_y"]) for w in wheels], 1)
    valid = (tangent_norm > 1.e-6) & (np.abs(target) > 1.e-4) & (data["heading_rate_valid"][:, None] > 0)
    conditions = {
        "valid_geometry": valid,
        "contact": pair("contact") > 0,
        "correct_direction": measured * target > 0.,
        "center_speed_ge_half_target": np.abs(measured) >= params["minimum_speed_fraction"] * np.abs(target),
        "motor_activity_ge_half_target": fraction >= params["minimum_speed_fraction"],
        "slip_within_scale": np.abs(slip) <= scale,
    }
    settled = np.zeros(len(data), bool)
    for env_id in np.unique(ids):
        indices = np.flatnonzero(ids == env_id)
        indices = indices[np.argsort(data["step"][indices])]
        active_steps, previous_done = 0, False
        for index in indices:
            if previous_done:
                active_steps = 0
            active_steps = active_steps + 1 if data["command"][index] > params["deadband"] else 0
            settled[index] = active_steps > 0 and active_steps * dt + 1.e-6 >= params["settle_time"]
            previous_done = bool(data["done"][index])
    all_conditions = np.stack(list(conditions.values()), axis=1).all(axis=(1, 2))
    reconstructed = settled & all_conditions
    mismatch = np.flatnonzero(reconstructed != (data["differential_pass"] > 0))
    tracking_cost = (np.sqrt(1. + ((measured - target) / scale) ** 2) - 1.).max(1)
    slip_cost = (np.sqrt(1. + (slip / scale) ** 2) - 1.).max(1)
    motor_cost = 2. * (1. - np.clip(fraction / params["minimum_speed_fraction"], 0., 1.).min(1))
    denominator = 1. + tracking_cost + slip_cost + motor_cost
    shaping = (valid & conditions["contact"]).all(1) / denominator
    shaping = np.where(data["command"] > params["deadband"], shaping, 0.)
    result = {
        "directory": str(directory), "checkpoint": metadata["checkpoint"], "policy": metadata["policy"],
        "num_envs": int(len(np.unique(ids))), "rows": int(len(data)), "resets": int(data["done"].sum()),
        "certificate_reconstruction_mismatches": int(len(mismatch)),
        "weighted_shaping_vs_trace_max_error": float(np.max(np.abs(2. * shaping - data["reward_differential_rolling"]))),
        "active": {}, "neutral": {},
    }

    def active_summary(mask):
        checks = {name: value.all(1) for name, value in conditions.items()}
        episode_ratios = [float(reconstructed[(ids == i) & mask].mean()) for i in np.unique(ids[mask])]
        counts = np.bincount(np.argmax(np.abs((measured - target) / scale)[mask], axis=1), minlength=2)
        stats = {
            "samples": int(mask.sum()), "pass_rate": float(reconstructed[mask].mean()),
            "envs_ge_90pct_steps": int(sum(value >= .90 for value in episode_ratios)),
            "env_count": len(episode_ratios), "per_env_pass_rates": episode_ratios,
            "yaw_mae": float(data["heading_error"][mask].mean()),
            "com_xy_speed": float(data["active_com_planar_speed"][mask].mean()),
            "com_forward_speed": float(data["audit_com_forward_velocity"][mask].mean()),
            "heading_rate": float(data["true_heading_rate"][mask].mean()),
            "shaping_score": float(shaping[mask].mean()),
            "tracking_cost": float(tracking_cost[mask].mean()), "slip_cost": float(slip_cost[mask].mean()),
            "motor_deficit_cost": float(motor_cost[mask].mean()),
            "tracking_bottleneck_fl_fraction": float(counts[0] / mask.sum()),
            "blockers": {}, "wheels": {},
        }
        for name, passed in checks.items():
            other = np.stack([v for key, v in checks.items() if key != name], 1).all(1)
            stats["blockers"][name] = {
                "any_wheel_failure": float((~passed[mask]).mean()),
                "only_this_condition_failed": float((~passed & other)[mask].mean()),
                "fl_failure": float((~conditions[name][mask, 0]).mean()),
                "hr_failure": float((~conditions[name][mask, 1]).mean()),
            }
        for i, wheel in enumerate(wheels):
            w = {name: float(data[f"audit_{wheel}_{name}"][mask].mean()) for name in
                 ("target", "measured", "target_ratio", "motor_fraction", "motor_target", "motor_speed",
                  "motor_torque", "motor_power", "rolling_residual")}
            w["abs_target_mae"] = float(np.abs(measured[mask, i] - target[mask, i]).mean())
            w["abs_slip"] = float(np.abs(slip[mask, i]).mean())
            w["median_action_noise_vs_abs_target"] = float(np.median(
                params["wheel_radius"] * 5. * metadata["action_std"][12 + (0 if wheel == "fl" else 3)]
                / np.maximum(np.abs(target[mask, i]), 1.e-4)))
            stats["wheels"][wheel] = w
        return stats

    result["active"]["all"] = active_summary(settled)
    for command in sorted(np.unique(data["command"][settled])):
        mask = settled & np.isclose(data["command"], command)
        result["active"][f"yaw_{command:g}"] = active_summary(mask)
    initial_end = metadata["initial_neutral_steps"] * dt
    final_start = initial_end + metadata["positive_steps"] * dt
    four_contact = np.stack([data[f"audit_{wheel}_contact"] > 0 for wheel in ("fl", "fr", "hl", "hr")], 1).all(1)
    acquired = np.zeros(len(data), bool)
    anchor_movements = []
    for env_id in np.unique(ids):
        indices = np.flatnonzero(ids == env_id)
        indices = indices[np.argsort(data["step"][indices])]
        previous_held, previous_done, anchor = False, False, None
        for index in indices:
            held = data["neutral_hold_anchored"][index] > 0
            if previous_done:
                previous_held = False
            acquired[index] = held and not previous_held
            current = np.array([data["neutral_hold_anchor_x"][index], data["neutral_hold_anchor_y"][index]])
            if acquired[index]:
                anchor = current
            if held:
                anchor_movements.append(float(np.linalg.norm(current - anchor)))
            previous_held, previous_done = held, bool(data["done"][index])
    result["max_movement_within_latched_anchor"] = max(anchor_movements, default=0.)
    for phase, mask in (
        ("initial", data["time_s"] <= initial_end),
        ("initial_settled", (data["time_s"] > 1.) & (data["time_s"] <= initial_end)),
        ("final", data["time_s"] > final_start),
        ("final_settled", data["time_s"] > final_start + 1.),
    ):
        held = mask & (data["neutral_hold_anchored"] > 0)
        if not mask.any():
            continue
        result["neutral"][phase] = {
            "root_com_xy_speed": float(data["neutral_hold_planar_speed"][mask].mean()),
            "anchor_drift": float(data["neutral_hold_drift"][held].mean()) if held.any() else None,
            "held_pass_rate": float(data["neutral_hold_pass"][held].mean()) if held.any() else None,
            "held_speed_gt_003_rate": float((data["neutral_hold_planar_speed"][held] > .03).mean()) if held.any() else None,
            "held_drift_gt_005_rate": float((data["neutral_hold_drift"][held] > .05).mean()) if held.any() else None,
            "held_missing_four_contact_rate": float((~four_contact[held]).mean()) if held.any() else None,
            "anchor_acquisitions": int((acquired & mask).sum()),
            "acquisition_speed_gt_003_rate": float((data["neutral_hold_planar_speed"][acquired & mask] > .03).mean())
                                               if (acquired & mask).any() else None,
        }
    result["initial_state"] = {key: metadata.get(key) for key in
                               ("initial_root_state", "initial_joint_pos", "body_masses")}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [analyze(directory) for directory in args.directories]
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    for result in results:
        print(result["directory"], "mismatches", result["certificate_reconstruction_mismatches"])
        for name, stats in result["active"].items():
            print(name, "pass", round(stats["pass_rate"], 4), ">=90%", stats["envs_ge_90pct_steps"], "/", stats["env_count"],
                  "failures", {k: round(v["any_wheel_failure"], 4) for k, v in stats["blockers"].items()})
