# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument(
    "--distance", type=float, default=100.0, help="Target forward distance (m) to travel before stopping."
)
parser.add_argument(
    "--max_time",
    type=float,
    default=None,
    help=(
        "Max simulated time (s) before stopping, regardless of distance traveled. Env time_out is disabled for"
        " distance tracking, so this is needed to bound runs where the robot doesn't move (e.g. zero velocity"
        " command)."
    ),
)
parser.add_argument(
    "--log_dir", type=str, default=None, help="Directory to save logged data. Defaults to <checkpoint_dir>/play_logs."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# import after SimulationApp is created to avoid early Omniverse/pxr imports
from rl_utils import camera_follow

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform
from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import time
import torch
import numpy as np
import csv

from rsl_rl.runners import OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
import isaaclab.utils.math as math_utils
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # default to a single environment so distance-tracking / logging is unambiguous
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1

    # handle deprecated configurations (convert old policy format to new actor/critic format)
    # agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels = None
    # disable terrain-level curriculum: mid-episode resets can otherwise reassign the
    # robot to a different terrain row, teleporting root_pos_w and corrupting distance tracking
    if getattr(env_cfg.curriculum, "terrain_levels", None) is not None:
        env_cfg.curriculum.terrain_levels = None
        if env_cfg.scene.terrain.terrain_generator is not None:
            env_cfg.scene.terrain.terrain_generator.curriculum = False
    # disable episode time-out: an auto-reset mid-run teleports root_pos_w back to the
    # spawn point, which the unsigned distance-traveled calculation below would otherwise
    # misread as forward progress
    env_cfg.terminations.time_out = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )
    else:
        # fixed forward-velocity command: vx = max lin_vel_x, vy = 0, wz = 0
        fixed_vx = env_cfg.commands.base_velocity.ranges.lin_vel_x[1]
        env_cfg.commands.base_velocity.debug_vis = False
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(
                [fixed_vx, 0.0, 0.0], dtype=torch.float32
            ).unsqueeze(0).repeat(env.num_envs, 1).to(env.device),
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    # convert config to dict and create runner
    train_cfg = agent_cfg.to_dict()
    ppo_runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # Use runner-native exporters for rsl-rl >= 4.0.0
        ppo_runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        ppo_runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        policy_nn = None
    else:
        # Fallback for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = ppo_runner.alg.policy
        else:
            policy_nn = ppo_runner.alg.actor_critic

        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        else:
            normalizer = None

        export_policy_as_onnx(
            policy=policy_nn,
            normalizer=normalizer,
            path=export_model_dir,
            filename="policy.onnx",
        )
        export_policy_as_jit(
            policy=policy_nn,
            normalizer=normalizer,
            path=export_model_dir,
            filename="policy.pt",
        )

    dt = env.unwrapped.step_dt
    # reset environment
    obs, _ = env.reset()

    # ---- logging setup ----
    robot = env.unwrapped.scene["robot"]
    joint_names = list(robot.data.joint_names)
    # ===== Debug joint order =====
    print("\n========== JOINT ORDER ==========")
    print(f"Number of joints: {len(joint_names)}")

    for i, name in enumerate(joint_names):
        print(f"{i:2d}: {name}")

    print("=================================\n")
    out_log_dir = args_cli.log_dir if args_cli.log_dir is not None else os.path.join(os.path.dirname(resume_path), "play_logs")
    os.makedirs(out_log_dir, exist_ok=True)

    has_applied_torque = hasattr(robot.data, "applied_torque")
    has_computed_torque = hasattr(robot.data, "computed_torque")
    has_root_lin_vel_b = hasattr(robot.data, "root_lin_vel_b")
    if not (has_applied_torque or has_computed_torque):
        print("[WARNING] Robot data has neither 'applied_torque' nor 'computed_torque'; torque columns will be empty.")

    # env index 0 is the one we track distance / log for
    prev_pos_xy = robot.data.root_pos_w[0, :2].clone()
    traveled = 0.0

    csv_path = os.path.join(out_log_dir, "play_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    header = ["time", "distance"]
    header += [f"joint_pos_{n}" for n in joint_names]
    header += [f"joint_vel_{n}" for n in joint_names]
    header += [f"joint_torque_{n}" for n in joint_names]
    header += ["vx_w", "vy_w", "vz_w", "vx_b", "vy_b", "vz_b"]
    csv_writer.writerow(header)

    sim_time = 0.0
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # env stepping
            obs, _, dones, _ = env.step(actions)

        # ---- record data for env 0 ----
        cur_pos_xy = robot.data.root_pos_w[0, :2]
        # skip distance accrual right after a reset: root_pos_w jumps (teleports) to the
        # new spawn pose on the same step, which is not real forward travel
        if bool(dones[0]):
            step_dist = 0.0
        else:
            step_dist = torch.norm(cur_pos_xy - prev_pos_xy).item()
        traveled += step_dist
        prev_pos_xy = cur_pos_xy.clone()
        sim_time += dt

        joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()
        joint_vel = robot.data.joint_vel[0].detach().cpu().numpy()
        if has_applied_torque:
            joint_torque = robot.data.applied_torque[0].detach().cpu().numpy()
        elif has_computed_torque:
            joint_torque = robot.data.computed_torque[0].detach().cpu().numpy()
        else:
            joint_torque = np.zeros_like(joint_pos)

        lin_vel_w = robot.data.root_lin_vel_w[0].detach().cpu().numpy()
        if has_root_lin_vel_b:
            lin_vel_b = robot.data.root_lin_vel_b[0].detach().cpu().numpy()
        else:
            # manual fallback: rotate world-frame linear velocity into the root/body frame
            lin_vel_b_t = math_utils.quat_rotate_inverse(
                robot.data.root_quat_w[0:1], robot.data.root_lin_vel_w[0:1]
            )
            lin_vel_b = lin_vel_b_t[0].detach().cpu().numpy()

        row = [sim_time, traveled]
        row += joint_pos.tolist()
        row += joint_vel.tolist()
        row += joint_torque.tolist()
        row += lin_vel_w.tolist()
        row += lin_vel_b.tolist()
        csv_writer.writerow(row)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        # stop once the robot has covered the target distance
        if traveled >= args_cli.distance:
            print(f"[INFO] Reached target distance {args_cli.distance} m (traveled {traveled:.2f} m). Stopping.")
            break

        # stop once the max simulated time is reached (needed when the robot isn't expected
        # to move, e.g. a zero velocity command, since traveled would never reach args_cli.distance)
        if args_cli.max_time is not None and sim_time >= args_cli.max_time:
            print(f"[INFO] Reached max simulated time {args_cli.max_time} s. Stopping.")
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    csv_file.close()
    print(f"[INFO] Saved full logged data to: {csv_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()