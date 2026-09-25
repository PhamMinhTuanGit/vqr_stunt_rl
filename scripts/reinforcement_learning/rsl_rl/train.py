# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--critic_warmup_iterations",
    type=int,
    default=None,
    help="Critic-only PPO updates after resume. Flat-VQR-Wheel-Yaw defaults to 150; use 0 to disable.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# suppress noisy omni.usd warnings (e.g. unresolved visual prim references)
import carb
carb.logging.acquire_logging().set_level_threshold_for_source(
    "omni.usd", carb.logging.LogSettingBehavior.OVERRIDE, carb.logging.LEVEL_ERROR
)

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import inspect
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
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401
from rl_training.tasks.manager_based.locomotion.velocity.mdp.fsm import YawFSMCommand

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

_YAW_CURRICULUM_PREFIX = "_yaw_task_curriculum_"
_YAW_CURRICULUM_CHECKPOINT_KEY = "yaw_curriculum"
_YAW_CURRICULUM_PERSISTENT_FIELDS = (
    "_yaw_task_curriculum_stage",
    "_yaw_task_curriculum_yaw_stage",
    "_yaw_task_curriculum_consecutive_passes",
)

_YAW_FSM_CURRICULUM_CHECKPOINT_KEY = "yaw_fsm_curriculum"
_YAW_FSM_CURRICULUM_PERSISTENT_FIELDS = (
    "_yaw_fsm_task_curriculum_phase",
    "_yaw_task_curriculum_stage",
    "_yaw_task_curriculum_yaw_stage",
    "_yaw_task_curriculum_consecutive_passes",
    "_yaw_fsm_task_curriculum_ramp_start_fsm_gated_tracking",
    "_yaw_fsm_task_curriculum_ramp_start_transition_progress",
    "_yaw_fsm_task_curriculum_ramp_start_spin_center_drift",
    "_yaw_fsm_task_curriculum_ramp_start_safe_recovery_entry",
)


def _export_yaw_curriculum_state(task_env) -> dict:
    """Serialize scalar yaw-curriculum state without simulator tensors."""
    values = {
        name: getattr(task_env, name)
        for name in _YAW_CURRICULUM_PERSISTENT_FIELDS
        if hasattr(task_env, name) and isinstance(getattr(task_env, name), (bool, int, float))
    }
    current_step = int(task_env.common_step_counter)
    stage_start_step = int(getattr(task_env, "_yaw_task_curriculum_stage_start_step", current_step))
    term_params = task_env.cfg.curriculum.task_levels.params
    return {
        "version": 1,
        "values": values,
        "stage_elapsed_steps": max(0, current_step - stage_start_step),
        "yaw_rate_levels": list(term_params["yaw_rate_levels"]),
        "dr_scale_levels": list(term_params["dr_scale_levels"]),
    }


def _restore_yaw_curriculum_state(task_env, checkpoint_infos) -> bool:
    """Restore a saved yaw curriculum and immediately reapply its difficulty."""
    if not isinstance(checkpoint_infos, dict):
        return False
    payload = checkpoint_infos.get(_YAW_CURRICULUM_CHECKPOINT_KEY)
    if not isinstance(payload, dict) or payload.get("version") != 1:
        return False
    term_params = task_env.cfg.curriculum.task_levels.params
    if payload.get("yaw_rate_levels") != list(term_params["yaw_rate_levels"]) or payload.get(
        "dr_scale_levels"
    ) != list(term_params["dr_scale_levels"]):
        print("[WARN] Saved yaw curriculum stage table differs from the active config; ignoring saved state.")
        return False

    values = payload.get("values")
    if not isinstance(values, dict):
        return False
    for name, value in values.items():
        if name.startswith(_YAW_CURRICULUM_PREFIX) and isinstance(value, (bool, int, float)):
            setattr(task_env, name, value)

    elapsed_steps = max(0, int(payload.get("stage_elapsed_steps", 0)))
    task_env._yaw_task_curriculum_stage_start_step = int(task_env.common_step_counter) - elapsed_steps

    # Calling the term with no completed environments clamps restored indices
    # and reapplies command, reward-target and online-DR configuration.
    term_cfg = task_env.cfg.curriculum.task_levels
    term_cfg.func(task_env, [], **term_cfg.params)
    command_name = term_cfg.params["command_name"]
    command_term = task_env.command_manager.get_term(command_name)
    all_env_ids = torch.arange(task_env.num_envs, device=task_env.device)
    command_term.reset(all_env_ids)
    return True


def _install_yaw_curriculum_checkpointing(runner: OnPolicyRunner, task_env) -> None:
    """Inject yaw curriculum state into every RSL-RL checkpoint."""
    original_save = runner.save

    def save_with_yaw_curriculum(path: str, infos: dict | None = None) -> None:
        checkpoint_infos = dict(infos) if isinstance(infos, dict) else {}
        if infos is not None and not isinstance(infos, dict):
            checkpoint_infos["runner_infos"] = infos
        checkpoint_infos[_YAW_CURRICULUM_CHECKPOINT_KEY] = _export_yaw_curriculum_state(task_env)
        original_save(path, checkpoint_infos)

    runner.save = save_with_yaw_curriculum


def _export_yaw_fsm_curriculum_state(task_env) -> dict:
    """Serialize only scalar FSM curriculum state and relative timing."""
    values = {
        name: getattr(task_env, name)
        for name in _YAW_FSM_CURRICULUM_PERSISTENT_FIELDS
        if hasattr(task_env, name) and isinstance(getattr(task_env, name), (bool, int, float))
    }
    current_step = int(task_env.common_step_counter)
    stage_start_step = int(
        getattr(task_env, "_yaw_fsm_task_curriculum_stage_start_step", current_step)
    )
    reward_ramp_start_step = int(
        getattr(task_env, "_yaw_fsm_task_curriculum_reward_ramp_start_step", current_step)
    )
    term_params = task_env.cfg.curriculum.task_levels.params
    return {
        "version": 1,
        "values": values,
        "stage_elapsed_steps": max(0, current_step - stage_start_step),
        "reward_ramp_elapsed_steps": max(0, current_step - reward_ramp_start_step),
        "clearance_levels": list(term_params["clearance_levels"]),
        "yaw_rate_levels": list(term_params["yaw_rate_levels"]),
        "dr_scale_levels": list(term_params["dr_scale_levels"]),
    }


def _restore_yaw_fsm_curriculum_state(task_env, checkpoint_infos) -> bool:
    """Restore FSM phase/stages and reapply all phase-dependent configuration."""
    if not isinstance(checkpoint_infos, dict):
        return False
    payload = checkpoint_infos.get(_YAW_FSM_CURRICULUM_CHECKPOINT_KEY)
    if not isinstance(payload, dict) or payload.get("version") != 1:
        return False

    term_cfg = task_env.cfg.curriculum.task_levels
    term_params = term_cfg.params
    saved_tables = (
        payload.get("clearance_levels"),
        payload.get("yaw_rate_levels"),
        payload.get("dr_scale_levels"),
    )
    active_tables = (
        list(term_params["clearance_levels"]),
        list(term_params["yaw_rate_levels"]),
        list(term_params["dr_scale_levels"]),
    )
    if saved_tables != active_tables:
        print("[WARN] Saved FSM curriculum stage table differs from the active config; ignoring saved state.")
        return False

    values = payload.get("values")
    if not isinstance(values, dict):
        return False
    allowed_fields = set(_YAW_FSM_CURRICULUM_PERSISTENT_FIELDS)
    for name, value in values.items():
        if name in allowed_fields and isinstance(value, (bool, int, float)):
            setattr(task_env, name, value)

    current_step = int(task_env.common_step_counter)
    stage_elapsed_steps = max(0, int(payload.get("stage_elapsed_steps", 0)))
    reward_ramp_elapsed_steps = max(0, int(payload.get("reward_ramp_elapsed_steps", 0)))
    task_env._yaw_fsm_task_curriculum_stage_start_step = current_step - stage_elapsed_steps
    task_env._yaw_fsm_task_curriculum_reward_ramp_start_step = (
        current_step - reward_ramp_elapsed_steps
    )

    # If a checkpoint is taken exactly on a phase boundary, the curriculum
    # refreshes these start values on its next call. Seed the active configs
    # first so a Phase-C restore cannot accidentally restart from Phase A.
    phase_reward_names = (
        term_params["yaw_reward_name"],
        term_params["transition_reward_name"],
        term_params.get("spin_center_drift_reward_name", "spin_center_drift"),
        term_params.get("safe_recovery_reward_name", "safe_recovery_entry"),
    )
    for reward_name in phase_reward_names:
        start_name = f"_yaw_fsm_task_curriculum_ramp_start_{reward_name}"
        if hasattr(task_env, start_name):
            reward_cfg = task_env.reward_manager.get_term_cfg(reward_name)
            reward_cfg.weight = float(getattr(task_env, start_name))
            task_env.reward_manager.set_term_cfg(reward_name, reward_cfg)

    # An empty completion list prevents accumulator consumption while the
    # curriculum reapplies clearance, command range, DR, gates, and weights.
    term_cfg.func(task_env, [], **term_params)
    command_term = task_env.command_manager.get_term(term_params["command_name"])
    all_env_ids = torch.arange(task_env.num_envs, device=task_env.device)
    command_term.reset(all_env_ids)
    return True


def _install_yaw_fsm_curriculum_checkpointing(runner: OnPolicyRunner, task_env) -> None:
    """Inject the distinct FSM curriculum payload into every checkpoint."""
    original_save = runner.save

    def save_with_yaw_fsm_curriculum(path: str, infos: dict | None = None) -> None:
        checkpoint_infos = dict(infos) if isinstance(infos, dict) else {}
        if infos is not None and not isinstance(infos, dict):
            checkpoint_infos["runner_infos"] = infos
        checkpoint_infos[_YAW_FSM_CURRICULUM_CHECKPOINT_KEY] = (
            _export_yaw_fsm_curriculum_state(task_env)
        )
        original_save(path, checkpoint_infos)

    runner.save = save_with_yaw_fsm_curriculum


def _verify_yaw_reward_config(env, env_cfg) -> None:
    """Print and enforce the source reward contract before any PPO update."""
    reward_manager = env.unwrapped.reward_manager
    term_names = list(reward_manager.active_terms)
    yaw_weight = (
        float(reward_manager.get_term_cfg("gated_yaw_tracking").weight)
        if "gated_yaw_tracking" in term_names
        else float("nan")
    )
    print(f"[INFO] Flat-VQR-Wheel-Yaw env config source: {inspect.getfile(type(env_cfg))}")
    print(f"[INFO] Flat-VQR-Wheel-Yaw reward terms ({len(term_names)}): {term_names}")
    print(f"[INFO] Flat-VQR-Wheel-Yaw gated_yaw_tracking weight: {yaw_weight}")
    if "support_contact" in term_names:
        raise RuntimeError("Stale yaw reward config detected: support_contact must not be an active reward term.")
    if len(term_names) != 18 or yaw_weight != 8.0:
        raise RuntimeError(
            "Unexpected Flat-VQR-Wheel-Yaw reward config: expected 18 terms and gated_yaw_tracking weight 8.0."
        )


def _verify_yaw_fsm_contract(env, env_cfg) -> None:
    """Enforce the FSM task contract before constructing the PPO runner."""
    task_env = env.unwrapped
    reward_manager = task_env.reward_manager
    term_names = list(reward_manager.active_terms)
    required_reward_terms = {
        "fsm_gated_tracking",
        "transition_progress",
        "transition_support_load",
        "return_to_four_landing",
        "four_stand_ready_bonus",
        "spin_center_drift",
        "safe_recovery_entry",
    }
    missing_reward_terms = sorted(required_reward_terms.difference(term_names))
    tracking_weight = (
        float(reward_manager.get_term_cfg("fsm_gated_tracking").weight)
        if "fsm_gated_tracking" in term_names
        else float("nan")
    )
    command = task_env.command_manager.get_term("yaw_rate_cmd")
    required_command_buffers = (
        "fsm_state",
        "support_diagonal",
        "state_time",
        "just_switched",
        "just_returned_to_four",
        "positive_pose_ready",
        "negative_pose_ready",
        "four_stand_ready",
        "unsafe",
        "yaw_entry_pos",
    )
    missing_command_buffers = [
        name for name in required_command_buffers if not hasattr(command, name)
    ]

    print(f"[INFO] Flat-VQR-Wheel-Yaw-FSM env config source: {inspect.getfile(type(env_cfg))}")
    print(f"[INFO] Flat-VQR-Wheel-Yaw-FSM reward terms ({len(term_names)}): {term_names}")
    print(f"[INFO] Flat-VQR-Wheel-Yaw-FSM fsm_gated_tracking weight: {tracking_weight}")
    if len(term_names) != 25 or missing_reward_terms or tracking_weight != 8.0:
        raise RuntimeError(
            "Unexpected Flat-VQR-Wheel-Yaw-FSM reward config: expected 25 terms, "
            "fsm_gated_tracking weight 8.0, and all required FSM reward terms; "
            f"missing={missing_reward_terms}."
        )
    if not isinstance(command, YawFSMCommand) or missing_command_buffers:
        raise RuntimeError(
            "Unexpected Flat-VQR-Wheel-Yaw-FSM command contract: expected YawFSMCommand "
            f"with all public FSM buffers; missing={missing_command_buffers}."
        )


def _install_critic_warmup(runner: OnPolicyRunner, num_iterations: int) -> None:
    """Freeze actor/distribution parameters for the first resumed PPO updates."""
    if num_iterations <= 0:
        return

    frozen_parameters = []
    for name, parameter in runner.alg.policy.named_parameters():
        if not name.startswith("critic."):
            frozen_parameters.append((parameter, parameter.requires_grad))
            parameter.requires_grad_(False)
    if not frozen_parameters:
        raise RuntimeError("Critic warm-up found no actor/distribution parameters to freeze.")

    original_update = runner.alg.update
    original_schedule = runner.alg.schedule
    # With a frozen actor the measured KL is approximately zero. Leaving the
    # adaptive schedule enabled would therefore drive the shared optimizer LR
    # to its maximum during critic-only updates.
    runner.alg.schedule = "fixed"
    completed_updates = 0

    def update_with_critic_warmup():
        nonlocal completed_updates
        warmup_active = completed_updates < num_iterations
        loss_dict = original_update()
        completed_updates += 1
        loss_dict["critic_warmup_active"] = float(warmup_active)
        if completed_updates == num_iterations:
            for parameter, original_requires_grad in frozen_parameters:
                parameter.requires_grad_(original_requires_grad)
            runner.alg.schedule = original_schedule
            print(f"[INFO] Critic warm-up complete after {num_iterations} PPO updates; actor unfrozen.")
        return loss_dict

    runner.alg.update = update_with_critic_warmup
    print(f"[INFO] Critic-only warm-up enabled for {num_iterations} PPO updates.")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # handle deprecated configurations (convert old policy format to new actor/critic format)
    # agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    yaw_task_env = env.unwrapped if task_name == "Flat-VQR-Wheel-Yaw" else None
    yaw_fsm_task_env = env.unwrapped if task_name == "Flat-VQR-Wheel-Yaw-FSM" else None
    if task_name == "Flat-VQR-Wheel-Yaw":
        _verify_yaw_reward_config(env, env_cfg)
    elif task_name == "Flat-VQR-Wheel-Yaw-FSM":
        _verify_yaw_fsm_contract(env, env_cfg)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # convert config to dict and create runner
    train_cfg = agent_cfg.to_dict()
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
    
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    checkpoint_infos = None
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        checkpoint_infos = runner.load(resume_path)

    if yaw_task_env is not None:
        restored = _restore_yaw_curriculum_state(yaw_task_env, checkpoint_infos)
        if restored:
            print(
                "[INFO] Restored Flat-VQR-Wheel-Yaw curriculum: "
                f"clearance_stage={yaw_task_env._yaw_task_curriculum_stage}, "
                f"yaw_stage={yaw_task_env._yaw_task_curriculum_yaw_stage}."
            )
        elif agent_cfg.resume:
            print("[WARN] Checkpoint has no yaw curriculum state; starting curriculum from stage zero.")
        _install_yaw_curriculum_checkpointing(runner, yaw_task_env)

    if yaw_fsm_task_env is not None:
        restored = _restore_yaw_fsm_curriculum_state(yaw_fsm_task_env, checkpoint_infos)
        if restored:
            print(
                "[INFO] Restored Flat-VQR-Wheel-Yaw-FSM curriculum: "
                f"phase={yaw_fsm_task_env._yaw_fsm_task_curriculum_phase}, "
                f"lift_stage={yaw_fsm_task_env._yaw_task_curriculum_stage}, "
                f"yaw_stage={yaw_fsm_task_env._yaw_task_curriculum_yaw_stage}."
            )
        elif agent_cfg.resume:
            print(
                "[WARN] Checkpoint has no FSM yaw curriculum state; "
                "starting FSM curriculum from Phase A."
            )
        _install_yaw_fsm_curriculum_checkpointing(runner, yaw_fsm_task_env)

    critic_warmup_iterations = args_cli.critic_warmup_iterations
    if critic_warmup_iterations is None:
        critic_warmup_iterations = 150 if agent_cfg.resume and task_name == "Flat-VQR-Wheel-Yaw" else 0
    if critic_warmup_iterations < 0:
        raise ValueError("--critic_warmup_iterations must be non-negative.")
    if critic_warmup_iterations > 0 and not agent_cfg.resume:
        print("[WARN] Critic warm-up was requested without --resume; disabling it for fresh training.")
        critic_warmup_iterations = 0
    if agent_cfg.resume:
        start_iteration = runner.current_learning_iteration
        final_update_label = start_iteration + agent_cfg.max_iterations - 1
        print(
            "[INFO] --max_iterations is relative on resume: "
            f"{agent_cfg.max_iterations} updates from checkpoint iteration {start_iteration}; "
            f"expected final checkpoint label {final_update_label}."
        )
    _install_critic_warmup(runner, critic_warmup_iterations)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    
    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
