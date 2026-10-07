# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Record a fixed positive-yaw / neutral command sequence for the POS task."""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


TASK_ID = "Flat-VQR-Wheel-Yaw-POS"

parser = argparse.ArgumentParser(description="Record a POS yaw-to-four-stand inspection video.")
parser.add_argument("--checkpoint", help="Checkpoint from Flat-VQR-Wheel-Yaw-POS.")
parser.add_argument("--positive-yaw", type=float, default=None, help="Positive yaw in rad/s (default: checkpoint stage limit).")
parser.add_argument("--positive-seconds", type=float, default=4.0, help="Duration of each positive-yaw phase.")
parser.add_argument("--neutral-seconds", type=float, default=4.0, help="Duration of each neutral phase.")
parser.add_argument("--initial-neutral-seconds", type=float, default=0.0, help="Neutral hold before the first positive command.")
parser.add_argument("--cycles", type=int, default=2, help="Number of positive-yaw / neutral cycles.")
parser.add_argument("--output-dir", type=Path, default=None, help="Video directory (default: next to checkpoint).")
parser.add_argument("--seed", type=int, default=None, help="Fixed rollout seed (default: runner configuration seed).")
parser.add_argument("--trace-only", action="store_true", help="Write per-step CSV telemetry without recording a video.")
parser.add_argument("--neutral-wheel-stop", action="store_true",
                    help="Audit counterfactual: zero wheel velocity actions only during neutral.")
parser.add_argument("--sample-actions", action="store_true",
                    help="Audit counterfactual: sample checkpoint Gaussian action noise, without observation noise.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.checkpoint is None:
    parser.error("--checkpoint is required")
checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve(strict=True)
output_dir_arg = args_cli.output_dir.expanduser().resolve() if args_cli.output_dir is not None else None
args_cli.enable_cameras = not args_cli.trace_only
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401 - registers TASK_ID
from rl_training.tasks.manager_based.locomotion.velocity.mdp.yaw_pos_kinematics import ground_heading_axes
from rl_training.tasks.manager_based.locomotion.velocity.mdp.wheel_contact_kinematics import rotate_vector


def _capture_rollout_audit(raw_env) -> None:
    """Freeze vector geometry, actuator commands and contacts at reward time."""
    params = raw_env.reward_manager.get_term_cfg("rolling_slip").params
    robot = raw_env.scene[params["body_asset_cfg"].name]
    data = robot.data
    body_ids = params["body_asset_cfg"].body_ids
    joint_ids = params["joint_asset_cfg"].joint_ids
    sensor_cfg = params["sensor_cfg"]
    names = [robot.body_names[i].removesuffix("_WHEEL").lower() for i in body_ids]
    masses = robot.root_physx_view.get_masses().to(raw_env.device)
    total_mass = masses.sum(1).clamp_min(1.e-6)[:, None]
    com_pos = (data.body_com_pos_w * masses[..., None]).sum(1) / total_mass
    com_vel = (data.body_com_lin_vel_w * masses[..., None]).sum(1) / total_mass
    heading_x, heading_y, _ = ground_heading_axes(data.root_quat_w)
    quat = data.body_quat_w[:, body_ids]
    axle = rotate_vector(quat, quat.new_tensor((0., 1., 0.)))
    tangent = torch.linalg.cross(axle, axle.new_tensor((0., 0., 1.)).expand_as(axle), dim=-1)
    tangent /= torch.linalg.vector_norm(tangent, dim=-1).clamp_min(1.e-6)[..., None]
    forces = raw_env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    force_norm = torch.linalg.vector_norm(forces, dim=-1)
    center_vel = data.body_link_lin_vel_w[:, body_ids]
    rolling = (center_vel * tangent).sum(-1)
    axial_omega = (data.body_ang_vel_w[:, body_ids] * axle).sum(-1)
    qdot = data.joint_vel[:, joint_ids]
    command = raw_env.command_manager.get_command("yaw_rate_cmd")[:, 0]
    offset = data.body_pos_w[:, body_ids] - com_pos[:, None]
    target_vec = torch.stack((-offset[..., 1], offset[..., 0], torch.zeros_like(offset[..., 0])), -1)
    target = (command[:, None, None] * target_vec * tangent).sum(-1)
    radius = params["wheel_radius"]
    snapshot = {}

    def vector(prefix, value, axes="xyz"):
        snapshot.update({f"audit_{prefix}_{axis}": value[:, i] for i, axis in enumerate(axes)})

    vector("com_pos", com_pos)
    vector("com_vel", com_vel)
    vector("root_link_pos", data.root_link_pos_w)
    vector("root_link_vel", data.root_link_lin_vel_w)
    vector("root_com_pos", data.root_com_pos_w)
    vector("root_com_vel", data.root_com_lin_vel_w)
    vector("root_omega", data.root_ang_vel_w)
    vector("root_quat", data.root_quat_w, "wxyz")
    vector("heading", heading_x)
    snapshot["audit_com_forward_velocity"] = (com_vel * heading_x).sum(-1)
    snapshot["audit_com_lateral_velocity"] = (com_vel * heading_y).sum(-1)
    for i, name in enumerate(names):
        prefix = f"audit_{name}"
        ratio_valid = target[:, i].abs() > 1.e-4
        denominator = torch.where(ratio_valid, target[:, i], 1.)
        snapshot.update({
            f"{prefix}_target": target[:, i], f"{prefix}_measured": rolling[:, i],
            f"{prefix}_target_ratio": torch.where(ratio_valid, rolling[:, i] / denominator, 0.),
            f"{prefix}_ratio_valid": ratio_valid.float(),
            f"{prefix}_motor_fraction": torch.where(ratio_valid, radius * qdot[:, i].abs() / denominator.abs(), 0.),
            f"{prefix}_motor_speed": qdot[:, i],
            f"{prefix}_motor_target": data.joint_vel_target[:, joint_ids[i]],
            f"{prefix}_motor_torque": data.applied_torque[:, joint_ids[i]],
            f"{prefix}_motor_power": data.applied_torque[:, joint_ids[i]] * qdot[:, i],
            f"{prefix}_axial_omega": axial_omega[:, i],
            f"{prefix}_rolling_residual": rolling[:, i] - radius * axial_omega[:, i],
            f"{prefix}_contact": (force_norm[:, i] > params["threshold"]).float(),
            f"{prefix}_contact_force": force_norm[:, i],
        })
        vector(f"{name}_pos", data.body_pos_w[:, body_ids[i]])
        vector(f"{name}_vel", center_vel[:, i])
        vector(f"{name}_axle", axle[:, i])
        vector(f"{name}_tangent", tangent[:, i])
        vector(f"{name}_quat", quat[:, i], "wxyz")
        parent_id = robot.body_names.index(f"{name.upper()}_SHANK")
        parent_omega = data.body_ang_vel_w[:, parent_id]
        snapshot[f"{prefix}_relative_axial_omega"] = (
            (data.body_ang_vel_w[:, body_ids[i]] - parent_omega) * axle[:, i]
        ).sum(-1)
    env_snapshot = {name: value.detach().clone() for name, value in snapshot.items()}
    raw_env._yaw_pos_rollout_audit_current = env_snapshot


def _command_for_step(step: int, positive_steps: int, neutral_steps: int, positive_yaw: float,
                      initial_neutral_steps: int = 0) -> float:
    """Return the scheduled command for one positive / neutral cycle."""
    if step < initial_neutral_steps:
        return 0.0
    phase_step = (step - initial_neutral_steps) % (positive_steps + neutral_steps)
    return positive_yaw if phase_step < positive_steps else 0.0


def _checkpoint_yaw_limit(checkpoint: Path, fallback: float) -> float:
    """Use the POS checkpoint's saved yaw stage for the playback command range."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    curriculum = state.get("infos", {}).get("yaw_curriculum", {})
    levels = curriculum.get("yaw_rate_levels", [])
    stage = curriculum.get("values", {}).get("_yaw_task_curriculum_yaw_stage", 0)
    if levels and 0 <= stage < len(levels):
        return float(levels[stage])
    return fallback


def _checkpoint_clearance(checkpoint: Path, clearance_levels) -> float:
    """Inspect the same landing/lift target that was trained at the saved stage."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    curriculum = (state.get("infos") or {}).get("yaw_curriculum") or {}
    stage = curriculum.get("values", {}).get("_yaw_task_curriculum_stage", 0)
    if type(stage) is not int or not 0 <= stage < len(clearance_levels):
        stage = 0
    return float(clearance_levels[stage])


def _trace_row(raw_env, step: int, dt: float, command: float, done: bool) -> dict:
    """Use reward-time snapshots so an auto-reset cannot corrupt the final row."""
    motion = raw_env._yaw_pos_motion_metrics_current
    row = {"step": step + 1, "time_s": (step + 1) * dt, "command": command, "done": int(done)}
    row.update({name: float(value[0].item()) for name, value in motion.items()})
    row["heading_error"] = abs(row["true_heading_rate"] - command)
    row.update({name: float(value[0].item()) for name, value in raw_env._yaw_pos_geometry_metrics_current.items()})
    for name, value in getattr(raw_env, "_yaw_pos_differential_metrics_current", {}).items():
        row[name] = float(value[0].item())
    for name, value in getattr(raw_env, "_yaw_pos_rollout_audit_current", {}).items():
        row[name] = float(value[0].item())
    for name, value in getattr(raw_env, "_yaw_pos_neutral_position_metrics_current", {}).items():
        row[f"neutral_hold_{name}"] = float(value[0].item())
    neutral = getattr(raw_env, "_yaw_pos_neutral_position_metrics_current", {})
    if neutral:
        row["neutral_hold_pass"] = float(neutral["passed"][0].item())
        row["neutral_position_drift"] = float(neutral["drift"][0].item()) if neutral["anchored"][0] else 0.0
    manager = raw_env.reward_manager
    for name in ("gated_yaw_tracking", "support_y_collapse", "body_angular_xy",
                 "differential_rolling", "neutral_velocity", "neutral_position"):
        if name not in manager.active_terms:
            continue
        index = manager.active_terms.index(name)
        # Isaac Lab stores weighted reward rates here (compute already divides
        # the integrated contribution by dt).
        row[f"reward_{name}"] = float(manager._step_reward[0, index].item())
    return row


@hydra_task_config(TASK_ID, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg) -> None:
    checkpoint = checkpoint_path
    if (args_cli.positive_seconds <= 0 or args_cli.neutral_seconds <= 0 or args_cli.cycles < 1
            or args_cli.initial_neutral_seconds < 0):
        raise ValueError("Phase durations must be positive and --cycles must be at least 1.")

    yaw_cfg = env_cfg.commands.yaw_rate_cmd
    yaw_limit = _checkpoint_yaw_limit(checkpoint, float(yaw_cfg.yaw_rate_range[1]))
    positive_yaw = yaw_limit if args_cli.positive_yaw is None else args_cli.positive_yaw
    if not yaw_cfg.deadband < positive_yaw <= yaw_limit:
        raise ValueError(f"--positive-yaw must be in ({yaw_cfg.deadband}, {yaw_limit}] rad/s for this checkpoint.")

    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device
    env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed
    agent_cfg.seed = env_cfg.seed
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.scene.terrain.max_init_terrain_level = None
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.curriculum = False
    if env_cfg.curriculum is not None:
        levels_cfg = env_cfg.curriculum.task_levels
        if levels_cfg is not None:
            clearance = _checkpoint_clearance(checkpoint, levels_cfg.params["clearance_levels"])
            for name in ("lift_clearance", "gated_yaw_tracking", "neutral_landing_progress"):
                getattr(env_cfg.rewards, name).params["target_clearance"] = clearance
        env_cfg.curriculum.task_levels = None
    if env_cfg.events is not None:
        for event_name in ("randomize_apply_external_force_torque", "randomize_push_robot", "push_robot",
                           "randomize_actuator_gains"):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)

    yaw_cfg.yaw_rate_range = (0.0, yaw_limit)
    yaw_cfg.external_control = True  # Prevent training resampling at reset and within the episode.
    yaw_cfg.debug_vis = False

    output_root = output_dir_arg or checkpoint.parent / "videos" / "yaw_pos_inspection"
    video_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    video_dir.mkdir(parents=True)

    env = gym.make(TASK_ID, cfg=env_cfg, render_mode=None if args_cli.trace_only else "rgb_array")
    try:
        dt = env.unwrapped.step_dt
        positive_steps = round(args_cli.positive_seconds / dt)
        neutral_steps = round(args_cli.neutral_seconds / dt)
        initial_neutral_steps = round(args_cli.initial_neutral_seconds / dt)
        if positive_steps < 1 or neutral_steps < 1:
            raise ValueError("Each phase must last at least one environment step.")
        total_steps = initial_neutral_steps + args_cli.cycles * (positive_steps + neutral_steps)
        if total_steps * dt > env_cfg.episode_length_s:
            raise ValueError("Requested video exceeds one episode; shorten phases or reduce --cycles.")

        if not args_cli.trace_only:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(video_dir),
                step_trigger=lambda step: step == 0,
                video_length=total_steps,
                name_prefix=checkpoint.stem,
                fps=round(1.0 / dt),
                disable_logger=True,
            )
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model_state = checkpoint_state["model_state_dict"]
        action_std = (model_state["std"] if "std" in model_state else model_state["log_std"].exp())
        action_std = action_std.to(env.unwrapped.device)
        env.reset()
        env.unwrapped._yaw_pos_rollout_audit_callback = _capture_rollout_audit
        command_term = env.unwrapped.command_manager.get_term("yaw_rate_cmd")
        wheel_action = env.unwrapped.action_manager.get_term("joint_vel")
        action_start = 0
        for name in env.unwrapped.action_manager.active_terms:
            if name == "joint_vel":
                break
            action_start += env.unwrapped.action_manager.get_term(name).action_dim

        print(f"[INFO] Checkpoint: {checkpoint}")
        print(f"[INFO] Sequence: ({positive_yaw:g} rad/s for {positive_steps * dt:g}s, "
              f"0 for {neutral_steps * dt:g}s) x {args_cli.cycles}")
        metadata = {
            "task": TASK_ID, "checkpoint": str(checkpoint), "seed": env_cfg.seed,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "policy": "sampled Gaussian actions" if args_cli.sample_actions else "deterministic inference",
            "observation_corruption": False,
            "interval_disturbances": False, "startup_randomization": "fixed seed",
            "neutral_wheel_stop": args_cli.neutral_wheel_stop,
            "action_std": action_std.tolist(),
            "checkpoint_curriculum": (checkpoint_state.get("infos") or {}).get("yaw_curriculum"),
            "wheel_motor_order": wheel_action._joint_names,
            "rolling_slip_parameters": {k: v for k, v in env_cfg.rewards.rolling_slip.params.items()
                                        if isinstance(v, (int, float, str))},
            "differential_parameters": {k: v for k, v in env_cfg.rewards.differential_rolling.params.items()
                                        if isinstance(v, (int, float, str))},
            "reward_source_sha256": hashlib.sha256(Path(env_cfg.rewards.differential_rolling.func.__code__.co_filename).read_bytes()).hexdigest(),
            "positive_yaw": positive_yaw, "positive_steps": positive_steps, "neutral_steps": neutral_steps,
            "cycles": args_cli.cycles, "step_dt": dt,
            "initial_neutral_steps": initial_neutral_steps,
            "target_clearance": env_cfg.rewards.lift_clearance.params["target_clearance"],
            "rate_units": "rad/s", "ground_rolling_speed_units": "m/s", "geometry_units": "m",
            "target_ratio_definition": "measured / target; undefined when ratio_valid=0 (including neutral)",
            "motor_fraction_definition": "wheel_radius * abs(joint_velocity) / abs(geometric_target)",
            "reward_units": "weighted reward per second, before step_dt multiplication",
            "body_angular_xy_weight": env_cfg.rewards.body_angular_xy.weight,
            "collapse_weight": env_cfg.rewards.support_y_collapse.weight,
            "collapse_scale": env_cfg.rewards.support_y_collapse.params["separation_scale"],
            "support_line_sigma": env_cfg.rewards.support_line.params["sigma"],
        }
        (video_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        with (video_dir / "telemetry.csv").open("w", newline="", buffering=1) as trace, torch.inference_mode():
            writer = None
            for step in range(total_steps):
                yaw_cmd = _command_for_step(step, positive_steps, neutral_steps, positive_yaw, initial_neutral_steps)
                command_term.set_external_command(yaw_cmd)
                obs = env.get_observations()  # Refresh policy input after changing the command.
                actions = policy(obs)
                if args_cli.sample_actions:
                    actions = actions + torch.randn_like(actions) * action_std
                if args_cli.neutral_wheel_stop and yaw_cmd == 0.:
                    actions[:, action_start:action_start + wheel_action.action_dim] = 0.
                _, _, dones, _ = env.step(actions)
                row = _trace_row(env.unwrapped, step, dt, yaw_cmd, bool(dones[0].item()))
                if writer is None:
                    writer = csv.DictWriter(trace, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
    finally:
        env.close()  # RecordVideo writes the MP4 here.

    print(f"[INFO] Telemetry saved: {video_dir / 'telemetry.csv'}")
    if not args_cli.trace_only:
        videos = sorted(video_dir.glob("*.mp4"))
        if not videos:
            raise RuntimeError(f"Video recorder did not write an MP4 in {video_dir}")
        print(f"[INFO] Video saved: {videos[-1]}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
