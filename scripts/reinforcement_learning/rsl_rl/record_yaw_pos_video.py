# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Record a fixed positive-yaw / neutral command sequence for the POS task."""

import argparse
import csv
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
parser.add_argument("--cycles", type=int, default=2, help="Number of positive-yaw / neutral cycles.")
parser.add_argument("--output-dir", type=Path, default=None, help="Video directory (default: next to checkpoint).")
parser.add_argument("--seed", type=int, default=None, help="Fixed rollout seed (default: runner configuration seed).")
parser.add_argument("--trace-only", action="store_true", help="Write per-step CSV telemetry without recording a video.")
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
from rl_training.tasks.manager_based.locomotion.velocity.mdp.yaw_pos_kinematics import POS_MOTION_METRICS


def _command_for_step(step: int, positive_steps: int, neutral_steps: int, positive_yaw: float) -> float:
    """Return the scheduled command for one positive / neutral cycle."""
    phase_step = step % (positive_steps + neutral_steps)
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


def _trace_row(raw_env, step: int, dt: float, command: float, done: bool) -> dict:
    """Use reward-time snapshots so an auto-reset cannot corrupt the final row."""
    motion = raw_env._yaw_pos_motion_metrics_current
    row = {"step": step + 1, "time_s": (step + 1) * dt, "command": command, "done": int(done)}
    row.update({name: float(value[0].item()) for name, value in motion.items()})
    row["heading_error"] = abs(row["true_heading_rate"] - command)
    row.update({name: float(value[0].item()) for name, value in raw_env._yaw_pos_geometry_metrics_current.items()})
    manager = raw_env.reward_manager
    for name in ("gated_yaw_tracking", "support_y_collapse", "body_angular_xy"):
        index = manager.active_terms.index(name)
        row[f"reward_{name}"] = float(manager._step_reward[0, index].item()) / dt
    return row


@hydra_task_config(TASK_ID, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg) -> None:
    checkpoint = checkpoint_path
    if args_cli.positive_seconds <= 0 or args_cli.neutral_seconds <= 0 or args_cli.cycles < 1:
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
        if positive_steps < 1 or neutral_steps < 1:
            raise ValueError("Each phase must last at least one environment step.")
        total_steps = args_cli.cycles * (positive_steps + neutral_steps)
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
        env.reset()
        command_term = env.unwrapped.command_manager.get_term("yaw_rate_cmd")

        print(f"[INFO] Checkpoint: {checkpoint}")
        print(f"[INFO] Sequence: ({positive_yaw:g} rad/s for {positive_steps * dt:g}s, "
              f"0 for {neutral_steps * dt:g}s) x {args_cli.cycles}")
        metadata = {
            "task": TASK_ID, "checkpoint": str(checkpoint), "seed": env_cfg.seed,
            "policy": "deterministic inference", "observation_corruption": False,
            "interval_disturbances": False, "startup_randomization": "fixed seed",
            "positive_yaw": positive_yaw, "positive_steps": positive_steps, "neutral_steps": neutral_steps,
            "cycles": args_cli.cycles, "step_dt": dt,
            "target_clearance": env_cfg.rewards.lift_clearance.params["target_clearance"],
            "rate_units": "rad/s", "ground_rolling_speed_units": "m/s", "geometry_units": "m",
            "reward_units": "weighted reward per second, before step_dt multiplication",
            "body_angular_xy_weight": env_cfg.rewards.body_angular_xy.weight,
            "collapse_weight": env_cfg.rewards.support_y_collapse.weight,
            "collapse_scale": env_cfg.rewards.support_y_collapse.params["separation_scale"],
            "support_line_sigma": env_cfg.rewards.support_line.params["sigma"],
        }
        (video_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        fields = ["step", "time_s", "command", "done", *POS_MOTION_METRICS, "heading_error",
                  "support_line_error", "wheel1_line_error", "wheel2_line_error", "support_y_separation",
                  "reward_gated_yaw_tracking", "reward_support_y_collapse", "reward_body_angular_xy"]
        with (video_dir / "telemetry.csv").open("w", newline="", buffering=1) as trace, torch.inference_mode():
            writer = csv.DictWriter(trace, fieldnames=fields)
            writer.writeheader()
            for step in range(total_steps):
                yaw_cmd = _command_for_step(step, positive_steps, neutral_steps, positive_yaw)
                command_term.set_external_command(yaw_cmd)
                obs = env.get_observations()  # Refresh policy input after changing the command.
                actions = policy(obs)
                _, _, dones, _ = env.step(actions)
                writer.writerow(_trace_row(env.unwrapped, step, dt, yaw_cmd, bool(dones[0].item())))
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
