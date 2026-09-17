# Copyright (c) 2026
# Debug utility derived from scripts/reinforcement_learning/rsl_rl/play.py
#
# Purpose:
#   Bypass the learned policy and directly command the task's scripted
#   standing -> TO joint reference through the *same* leg action term used
#   during training.
#
# This isolates:
#   reference generation -> normalized action mapping -> actuator tracking
#   -> contact transition
#
# If FR/HL do not lift in this script, the problem is upstream of PPO.

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

from isaaclab.app import AppLauncher

# Keep the same local import convention as play.py.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

parser = argparse.ArgumentParser(
    description="Debug the four-to-two transition by directly tracking TOPivotCommand.reference."
)
parser.add_argument("--video", action="store_true", default=False, help="Record one debug video.")
parser.add_argument(
    "--video_length",
    type=int,
    default=600,
    help="Recorded video length in environment steps. 600 steps = 12 s for this task.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=600,
    help="Maximum debug steps when not recording video.",
)
parser.add_argument(
    "--debug_interval",
    type=float,
    default=0.5,
    help="Print diagnostics every N seconds.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Use 1 for debugging.")
parser.add_argument("--task", type=str, default=None, help="Gym task name.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Agent config entry point; loaded only because the task uses the same Hydra entry mechanism as play.py.",
)
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument("--real-time", action="store_true", default=False, help="Throttle to real time.")
parser.add_argument(
    "--video_dir",
    type=str,
    default=None,
    help="Optional video output folder. Defaults to logs/debug/reference_play/<timestamp>.",
)

# Keep RSL-RL CLI args available so the script is compatible with the same task config path as play.py.
cli_args.add_rsl_rl_args(parser)

# AppLauncher arguments include --device, --headless, --enable_cameras, --kit_args, ...
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# Hydra must only see Hydra overrides.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports that require SimulationApp come after AppLauncher.
import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401


def _as_vector(value: torch.Tensor, num_envs: int, width: int) -> torch.Tensor:
    """Broadcast action scale/offset tensors to [num_envs, width]."""
    value = torch.as_tensor(value)
    if value.ndim == 0:
        value = value.expand(width)
    if value.ndim == 1:
        value = value.unsqueeze(0).expand(num_envs, -1)
    elif value.ndim == 2 and value.shape[0] == 1:
        value = value.expand(num_envs, -1)
    if value.shape != (num_envs, width):
        raise RuntimeError(
            f"Unexpected action parameter shape {tuple(value.shape)}; "
            f"expected {(num_envs, width)}."
        )
    return value


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
):
    """Run the task with scripted reference actions instead of PPO."""

    # Same basic overrides as play.py.
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Keep play deterministic with respect to policy observations and perturbations.
    if hasattr(env_cfg, "observations") and env_cfg.observations is not None:
        if hasattr(env_cfg.observations, "policy") and env_cfg.observations.policy is not None:
            env_cfg.observations.policy.enable_corruption = False

    if hasattr(env_cfg, "events") and env_cfg.events is not None:
        if hasattr(env_cfg.events, "randomize_apply_external_force_torque"):
            env_cfg.events.randomize_apply_external_force_torque = None
        if hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot = None
        # This task calls the interval event "push".
        if hasattr(env_cfg.events, "push"):
            env_cfg.events.push = None

    # Same terrain-memory reduction used by play.py.
    if hasattr(env_cfg.scene, "terrain") and env_cfg.scene.terrain is not None:
        env_cfg.scene.terrain.max_init_terrain_level = None
        if env_cfg.scene.terrain.terrain_generator is not None:
            env_cfg.scene.terrain.terrain_generator.num_rows = 1
            env_cfg.scene.terrain.terrain_generator.num_cols = 1
            env_cfg.scene.terrain.terrain_generator.curriculum = False

    print("[DEBUG] Creating environment for direct reference tracking...")
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    if isinstance(env.unwrapped, DirectMARLEnv):
        raise RuntimeError("play_debug.py expects the current single-agent manager-based pivot task.")

    raw = env.unwrapped

    # Record directly from the Gym environment. No RSL-RL wrapper is needed.
    if args_cli.video:
        if args_cli.video_dir is None:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            video_folder = os.path.abspath(
                os.path.join("logs", "debug", "reference_play", stamp)
            )
        else:
            video_folder = os.path.abspath(args_cli.video_dir)

        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[DEBUG] Recording direct-reference video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
        # RecordVideo does not change the underlying IsaacLab environment.
        raw = env.unwrapped

    obs, info = env.reset()

    pivot = raw.command_manager.get_term("pivot")
    leg_action = raw.action_manager.get_term("leg_positions")

    if len(pivot.joint_ids) != 12:
        raise RuntimeError(f"Expected 12 leg joints, got {len(pivot.joint_ids)}.")
    if raw.action_manager.total_action_dim != 16:
        raise RuntimeError(
            f"Expected 16 actions (12 leg + 4 wheel), got {raw.action_manager.total_action_dim}."
        )

    # JointPositionAction processing is:
    #     processed_target = offset + scale * raw_action
    # Our custom SoftLimitJointPositionAction only adds post-processing clamping.
    offset = _as_vector(
        leg_action._offset, raw.num_envs, len(pivot.joint_ids)
    ).to(device=raw.device, dtype=torch.float32)
    scale = _as_vector(
        leg_action._scale, raw.num_envs, len(pivot.joint_ids)
    ).to(device=raw.device, dtype=torch.float32)

    if torch.any(scale.abs() < 1.0e-8):
        raise RuntimeError("A leg action scale is zero; cannot invert reference -> action.")

    print("\n[DEBUG] Joint mapping:")
    for i, (name, joint_id) in enumerate(zip(pivot.cfg.leg_joint_names, pivot.joint_ids)):
        print(
            f"  {i:02d} {name:16s} joint_id={int(joint_id):2d} "
            f"standing={pivot.standing[i].item(): .5f} "
            f"target={pivot.target[i].item(): .5f} "
            f"offset={offset[0, i].item(): .5f} "
            f"scale={scale[0, i].item(): .5f}"
        )

    target_raw_action = (pivot.target.unsqueeze(0) - offset) / scale
    print("\n[DEBUG] Normalized action required for final TO pose:")
    print(target_raw_action[0].detach().cpu().numpy())
    print(
        "[DEBUG] final max |normalized leg action| = "
        f"{target_raw_action[0].abs().max().item():.4f}"
    )
    if target_raw_action.abs().max() > 1.0:
        print(
            "[WARNING] Final TO reference needs |action| > 1.0. "
            "The debug script will clamp to [-1, 1] to match training-time action clipping."
        )

    dt = raw.step_dt
    debug_every = max(1, int(round(args_cli.debug_interval / dt)))
    stop_after = args_cli.video_length if args_cli.video else args_cli.max_steps

    timestep = 0
    print(
        f"\n[DEBUG] Starting scripted reference tracking: dt={dt:.4f}s, "
        f"print every {debug_every} steps, stop after {stop_after} steps."
    )

    while simulation_app.is_running() and timestep < stop_after:
        start_time = time.time()

        with torch.inference_mode():
            # TOPivotCommand.reference is generated from its internal elapsed time:
            # standing for 0.5 s, then smooth transition for 2.5 s by default.
            q_ref = pivot.reference

            # Invert the actual configured action transformation.
            leg_raw = (q_ref - offset) / scale

            # Match the policy/RSL-RL action limit used in training.
            leg_raw = torch.clamp(leg_raw, -1.0, 1.0)

            actions = torch.zeros(
                (raw.num_envs, raw.action_manager.total_action_dim),
                device=raw.device,
                dtype=torch.float32,
            )
            actions[:, :12] = leg_raw
            # actions[:, 12:16] stay zero: wheels are not commanded to rotate.

            obs, reward, terminated, truncated, info = env.step(actions)

        timestep += 1

        if timestep == 1 or timestep % debug_every == 0:
            with torch.inference_mode():
                q_now = pivot.robot.data.joint_pos[:, pivot.joint_ids]
                q_ref_now = pivot.reference
                q_error = q_now - q_ref_now
                contacts = pivot.contacts()

                roll, pitch, yaw = euler_xyz_from_quat(pivot.robot.data.root_quat_w)
                roll = wrap_to_pi(roll)
                pitch = wrap_to_pi(pitch)
                yaw = wrap_to_pi(yaw)

                print("\n" + "=" * 92)
                print(
                    f"[DEBUG] step={timestep:4d}  "
                    f"t={timestep * dt:6.3f}s  "
                    f"lambda={pivot.command[0, 1].item():.4f}"
                )
                print(
                    f"[DEBUG] contacts [FL FR HL HR] = "
                    f"{contacts[0].to(torch.int32).cpu().tolist()}"
                )
                print(
                    f"[DEBUG] rpy [rad] = "
                    f"[{roll[0].item(): .4f}, {pitch[0].item(): .4f}, {yaw[0].item(): .4f}]"
                )
                print(
                    f"[DEBUG] body yaw rate = "
                    f"{pivot.robot.data.root_ang_vel_b[0, 2].item(): .4f} rad/s"
                )
                print(
                    f"[DEBUG] max|q-q_ref| = {q_error[0].abs().max().item():.5f} rad, "
                    f"RMS = {q_error[0].square().mean().sqrt().item():.5f} rad"
                )
                print("[DEBUG] leg_raw_action:")
                print(actions[0, :12].detach().cpu().numpy())
                print("[DEBUG] q_ref:")
                print(q_ref_now[0].detach().cpu().numpy())
                print("[DEBUG] q_actual:")
                print(q_now[0].detach().cpu().numpy())

                if bool(terminated[0]) or bool(truncated[0]):
                    print(
                        f"[DEBUG] episode boundary: terminated={bool(terminated[0])}, "
                        f"truncated={bool(truncated[0])}"
                    )

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("\n[DEBUG] Finished direct-reference test.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
