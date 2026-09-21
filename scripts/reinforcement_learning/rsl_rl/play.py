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
parser.add_argument("--max_steps", type=int, default=None, help="Stop playback after this many environment steps.")
parser.add_argument(
    "--reward_diagnostics",
    action="store_true",
    default=False,
    help="Print the measured weighted reward decomposition during playback.",
)
parser.add_argument(
    "--reward_diagnostics_interval",
    type=int,
    default=250,
    help="Playback steps per reward decomposition report.",
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
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 50
    if args_cli.reward_diagnostics_interval <= 0:
        raise ValueError("--reward_diagnostics_interval must be positive.")
    if args_cli.max_steps is not None and args_cli.max_steps <= 0:
        raise ValueError("--max_steps must be positive.")

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

    # disable randomization and training-only curricula for play
    env_cfg.observations.policy.enable_corruption = False
    if env_cfg.events is not None:
        for event_name in (
            "randomize_apply_external_force_torque",
            "randomize_push_robot",
            "push_robot",
        ):
            if hasattr(env_cfg.events, event_name):
                setattr(env_cfg.events, event_name, None)

    if env_cfg.curriculum is not None:
        # A task-level curriculum initializes the yaw range at its easiest
        # stage.  Playback disables that curriculum, so first promote the
        # command to the final trained range instead of leaving it at ±0.25.
        task_levels = getattr(env_cfg.curriculum, "task_levels", None)
        if task_levels is not None:
            command_name = task_levels.params.get("command_name")
            final_yaw_limit = task_levels.params.get("maximum_yaw_limit")
            if final_yaw_limit is None:
                yaw_rate_levels = task_levels.params.get("yaw_rate_levels")
                if yaw_rate_levels:
                    final_yaw_limit = yaw_rate_levels[-1]
            if command_name is not None and final_yaw_limit is not None:
                command_cfg = getattr(env_cfg.commands, command_name)
                final_yaw_limit = float(final_yaw_limit)
                command_cfg.yaw_rate_range = (-final_yaw_limit, final_yaw_limit)

        for curriculum_name in ("command_levels", "task_levels"):
            if hasattr(env_cfg.curriculum, curriculum_name):
                setattr(env_cfg.curriculum, curriculum_name, None)

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1]/2,
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
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

    reward_diagnostic_sum = None
    reward_diagnostic_samples = 0
    gate_open_sum = 0.0
    mean_abs_yaw_command_sum = 0.0
    yaw_rate_error_sum = 0.0
    if args_cli.reward_diagnostics:
        reward_manager = env.unwrapped.reward_manager
        reward_diagnostic_sum = torch.zeros(len(reward_manager.active_terms), device=env.unwrapped.device)
        print("[INFO] Reward diagnostics enabled; values are weighted reward rates averaged across environments.")
        if "yaw_rate_cmd" in env.unwrapped.command_manager.active_terms:
            yaw_command_term = env.unwrapped.command_manager.get_term("yaw_rate_cmd")
            initial_abs_command = env.unwrapped.command_manager.get_command("yaw_rate_cmd")[:, 0].abs().mean()
            print(f"[INFO] yaw_rate_cmd range: {yaw_command_term.cfg.yaw_rate_range}")
            print(f"[INFO] initial mean |yaw_rate_cmd|: {initial_abs_command.item():.6f}")
    
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.reward_diagnostics:
            reward_manager = env.unwrapped.reward_manager
            reward_diagnostic_sum += reward_manager._step_reward.mean(dim=0)
            reward_diagnostic_samples += 1
            unwrapped_env = env.unwrapped
            if hasattr(unwrapped_env, "_yaw_gate_open_current"):
                gate_open_sum += float(unwrapped_env._yaw_gate_open_current.mean().item())
            if hasattr(unwrapped_env, "_yaw_command_abs_current"):
                mean_abs_yaw_command_sum += float(unwrapped_env._yaw_command_abs_current.mean().item())
                yaw_rate_error_sum += float(unwrapped_env._yaw_rate_abs_error_current.mean().item())
            if reward_diagnostic_samples == args_cli.reward_diagnostics_interval:
                mean_terms = reward_diagnostic_sum / reward_diagnostic_samples
                positive_total = mean_terms.clamp_min(0.0).sum()
                yaw_index = (
                    reward_manager.active_terms.index("gated_yaw_tracking")
                    if "gated_yaw_tracking" in reward_manager.active_terms
                    else None
                )
                print("\n[REWARD DIAGNOSTICS] weighted rate by term")
                for name, value in zip(reward_manager.active_terms, mean_terms.tolist()):
                    print(f"  {name:32s} {value: .6f}")
                print(f"  {'positive_total':32s} {positive_total.item(): .6f}")
                if yaw_index is not None and positive_total.item() > 0.0:
                    yaw_share = mean_terms[yaw_index].clamp_min(0.0) / positive_total
                    print(f"  {'gated_yaw_positive_share':32s} {yaw_share.item(): .3%}")
                gate_open_rate = gate_open_sum / reward_diagnostic_samples
                mean_abs_yaw_command = mean_abs_yaw_command_sum / reward_diagnostic_samples
                mean_yaw_rate_error = yaw_rate_error_sum / reward_diagnostic_samples
                tracking_ratio = (
                    1.0 - mean_yaw_rate_error / mean_abs_yaw_command
                    if mean_abs_yaw_command > 1.0e-6
                    else 0.0
                )
                print(f"  {'gate_open_rate':32s} {gate_open_rate: .6f}")
                print(f"  {'mean_abs_yaw_cmd':32s} {mean_abs_yaw_command: .6f}")
                print(f"  {'mean_error_yaw_rate':32s} {mean_yaw_rate_error: .6f}")
                print(f"  {'tracking_ratio':32s} {tracking_ratio: .6f}")
                reward_diagnostic_sum.zero_()
                reward_diagnostic_samples = 0
                gate_open_sum = 0.0
                mean_abs_yaw_command_sum = 0.0
                yaw_rate_error_sum = 0.0
        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
            break

        if args_cli.keyboard:
            camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
