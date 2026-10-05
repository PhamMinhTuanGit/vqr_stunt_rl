"""Analyze reward-time POS rollout CSVs without launching Isaac Sim.

Usage: python analyze_yaw_pos_rollout.py TRACE_DIR [COUNTERFACTUAL_DIR ...]
Each directory must contain telemetry.csv and metadata.json.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_trace(directory):
    with (directory / "telemetry.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    columns = {name: np.array([float(row[name]) for row in rows]) for name in rows[0]}
    return columns, json.loads((directory / "metadata.json").read_text())


def vector(columns, prefix, axes="xyz"):
    return np.stack([columns[f"{prefix}_{axis}"] for axis in axes], axis=-1)


def summarize(directory):
    d, metadata = read_trace(directory)
    if metadata["cycles"] != 1:
        raise ValueError("This audit expects a single 0 → positive → 0 cycle (--cycles 1).")
    if metadata.get("num_envs", 1) != 1:
        raise ValueError("Use analyze_yaw_pos_certificate.py for parallel-environment audit traces.")
    dt = metadata["step_dt"]
    time = d["time_s"]
    initial_end = metadata["initial_neutral_steps"] * dt
    active_end = initial_end + metadata["positive_steps"] * dt
    masks = {
        "initial_neutral": time <= initial_end,
        "initial_neutral_settled": (time > 1.) & (time <= initial_end),
        "active_settled": (time > initial_end + .5) & (time <= active_end),
        "final_neutral": time > active_end,
        "final_neutral_settled": time > active_end + 1.,
    }
    com_velocity = vector(d, "audit_com_vel")
    checks = {}
    com_position = vector(d, "audit_com_pos")
    for wheel in ("fl", "hr"):
        position = vector(d, f"audit_{wheel}_pos")
        tangent = vector(d, f"audit_{wheel}_tangent")
        axle = vector(d, f"audit_{wheel}_axle")
        velocity = vector(d, f"audit_{wheel}_vel")
        offset = position - com_position
        # Independent ground-frame cross product and scalar projection.
        desired = np.cross(np.array([0., 0., 1.]), offset) * d["command"][:, None]
        projected_target = (desired * tangent).sum(-1)
        measured = (velocity * tangent).sum(-1)
        finite_difference = (position[2:] - position[:-2]) / (2. * dt)
        finite_difference = (finite_difference * tangent[1:-1]).sum(-1)
        checks[wheel] = {
            "target_max_error_m_s": float(np.max(np.abs(projected_target - d[f"audit_{wheel}_target"]))),
            "measured_max_error_m_s": float(np.max(np.abs(measured - d[f"audit_{wheel}_measured"]))),
            "center_velocity_finite_difference_mae_m_s": float(np.mean(np.abs(finite_difference - measured[1:-1]))),
            "tangent_axle_max_dot": float(np.max(np.abs((tangent * axle).sum(-1)))),
            "target_vs_reward_max_error_m_s": float(np.max(np.abs(projected_target - d[f"differential_{wheel}_target"]))),
        }
        relative = f"audit_{wheel}_relative_axial_omega"
        if relative in d:
            checks[wheel]["negative_joint_axis_max_error_rad_s"] = float(
                np.max(np.abs(d[relative] + d[f"audit_{wheel}_motor_speed"]))
            )
    root_velocity = vector(d, "audit_root_com_vel")
    root_link_velocity = vector(d, "audit_root_link_vel")
    checks["root_com_vs_link_xy_velocity_difference_mean_m_s"] = float(
        np.linalg.norm(root_velocity[:, :2] - root_link_velocity[:, :2], axis=1).mean()
    )
    result = {"directory": str(directory), "metadata": metadata, "geometry_checks": checks,
              "done_count": int(d["done"].sum()), "phases": {}}
    for phase, mask in masks.items():
        if not mask.any():
            continue
        held = mask & (d["neutral_hold_anchored"] > 0)
        stats = {
            "samples": int(mask.sum()),
            "com_xy_speed_m_s": float(np.linalg.norm(com_velocity[mask, :2], axis=1).mean()),
            "com_vx_m_s": float(com_velocity[mask, 0].mean()),
            "com_vy_m_s": float(com_velocity[mask, 1].mean()),
            "com_forward_m_s": float(d["audit_com_forward_velocity"][mask].mean()),
            "root_com_xy_speed_m_s": float(np.linalg.norm(root_velocity[mask, :2], axis=1).mean()),
            "yaw_rate_rad_s": float(d["true_heading_rate"][mask].mean()),
            "yaw_mae_rad_s": float(d["heading_error"][mask].mean()),
            "four_contacts_fraction": float(np.stack([d[f"audit_{w}_contact"][mask] for w in ("fl", "fr", "hl", "hr")], axis=1).all(1).mean()),
            "differential_pass_fraction": float(d["differential_pass"][mask].mean()),
            "neutral_hold_samples": int(held.sum()),
        }
        if held.any():
            anchor = np.stack([d["neutral_hold_anchor_x"][held], d["neutral_hold_anchor_y"][held]], -1)
            stats.update({
                "anchor_drift_mean_m": float(d["neutral_hold_drift"][held].mean()),
                "anchor_drift_max_m": float(d["neutral_hold_drift"][held].max()),
                "anchor_drift_end_m": float(d["neutral_hold_drift"][held][-1]),
                "neutral_hold_pass_fraction": float(d["neutral_hold_pass"][held].mean()),
                "anchor_max_movement_m": float(np.linalg.norm(anchor - anchor[0], axis=1).max()),
            })
        for wheel in ("fl", "fr", "hl", "hr"):
            prefix = f"audit_{wheel}_"
            stats[wheel] = {
                key: float(d[prefix + key][mask].mean())
                for key in ("target", "measured", "target_ratio", "motor_fraction", "motor_target",
                            "motor_speed", "motor_torque", "motor_power", "contact", "rolling_residual")
            }
            stats[wheel]["abs_motor_speed"] = float(np.abs(d[prefix + "motor_speed"][mask]).mean())
            stats[wheel]["abs_rolling_residual"] = float(np.abs(d[prefix + "rolling_residual"][mask]).mean())
        result["phases"][phase] = stats
    return result


def plot(directory, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, metadata = read_trace(directory)
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    time = d["time_s"]
    for wheel, color in (("fl", "tab:blue"), ("hr", "tab:orange")):
        axes[0].plot(time, d[f"audit_{wheel}_target"], "--", color=color, label=f"{wheel.upper()} target")
        axes[0].plot(time, d[f"audit_{wheel}_measured"], color=color, label=f"{wheel.upper()} measured")
        axes[1].plot(time, np.where(d[f"audit_{wheel}_ratio_valid"] > 0, d[f"audit_{wheel}_target_ratio"], np.nan), color=color, label=f"{wheel.upper()} measured/target")
        axes[2].plot(time, d[f"audit_{wheel}_motor_fraction"], color=color, label=f"{wheel.upper()} motor fraction")
    axes[1].axhline(.5, color="gray", linestyle=":")
    axes[1].axhline(1., color="gray", linestyle="--")
    axes[2].axhline(.5, color="gray", linestyle=":")
    axes[3].plot(time, d["command"], "k--", label="yaw command (rad/s)")
    axes[3].plot(time, d["true_heading_rate"], label="heading rate (rad/s)")
    axes[3].plot(time, np.linalg.norm(vector(d, "audit_com_vel")[:, :2], axis=1), label="CoM XY speed (m/s)")
    for item in results:
        trace, _ = read_trace(Path(item["directory"]))
        drift = np.where(trace["neutral_hold_anchored"] > 0, 100. * trace["neutral_hold_drift"], np.nan)
        axes[4].plot(trace["time_s"], drift, label=item["metadata"]["policy"] + ("; neutral wheels stopped" if item["metadata"]["neutral_wheel_stop"] else ""))
    for ax, label in zip(axes, ("Ground rolling (m/s)", "Measured / target", "R |qdot| / |target|", "Body motion", "Anchor drift (cm)")):
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{Path(metadata['checkpoint']).stem}: 0 → {metadata['positive_yaw']:g} → 0, "
                 f"seed {metadata['seed']}, reward-time telemetry")
    fig.tight_layout()
    output = directory.parent.parent / "rollout_audit.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    results = [summarize(directory) for directory in args.directories]
    destination = args.directories[0].parent.parent / "rollout_summary.json"
    destination.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(destination)
    print(plot(args.directories[0], results))
