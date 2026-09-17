"""Temporary runtime validation for four-wheel reset and contact timing."""

import argparse
import json
import statistics

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--case", choices=("grace", "zero", "ppo"), required=True)
parser.add_argument("--steps", type=int, default=250)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app


def emit(label, value):
    print(label, json.dumps(value), flush=True)


try:
    import gymnasium as gym
    import torch

    import rl_training.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    task = "Pivot-VQR-FourWheel-Rotate-v0"
    emit("MARK", {"case": args.case, "stage": "imports_complete"})

    if args.case == "grace":
        cfg = parse_env_cfg(task, device=args.device, num_envs=2)
        emit("MARK", {"case": args.case, "stage": "before_make"})
        env = gym.make(task, cfg=cfg)
        emit("MARK", {"case": args.case, "stage": "after_make"})
        try:
            env.reset()
            raw = env.unwrapped
            raw.episode_length_buf[:] = torch.tensor([73, 327], device=raw.device)
            zero = torch.zeros((2, raw.action_manager.total_action_dim), device=raw.device)
            trace = []
            for step in range(5):
                _, _, terminated, truncated, _ = env.step(zero)
                emit("MARK", {"case": args.case, "stage": "step", "step": step + 1})
                trace.append(
                    {
                        "physical_time_s": (step + 1) * raw.step_dt,
                        "episode_counters": raw.episode_length_buf.detach().cpu().tolist(),
                        "physical_age_s": raw._lost_wheel_contact_age.detach().cpu().tolist(),
                        "lost_time_s": raw._lost_wheel_contact_time.detach().cpu().tolist(),
                        "lost_term": raw.termination_manager.get_term("lost_wheel_contact").detach().cpu().tolist(),
                        "terminated": terminated.detach().cpu().tolist(),
                        "truncated": truncated.detach().cpu().tolist(),
                    }
                )
            emit("GRACE_TRACE", trace)
        finally:
            env.close()

    elif args.case == "zero":
        from rl_training.tasks.manager_based.locomotion.pivot.config.wheeled.vqr.robot_cfg import (
            WHEEL_BODY_NAMES,
        )

        cfg = parse_env_cfg(task, device=args.device, num_envs=16)
        env = gym.make(task, cfg=cfg)
        try:
            env.reset()
            raw = env.unwrapped
            sensor = raw.scene.sensors["contact_forces"]
            wheel_ids, wheel_names = sensor.find_bodies(WHEEL_BODY_NAMES, preserve_order=True)
            zero = torch.zeros((16, raw.action_manager.total_action_dim), device=raw.device)
            lost_resets = 0
            timeout_resets = 0
            invalid_after_transient = torch.zeros(16, dtype=torch.bool, device=raw.device)
            minimum_forces = torch.full((16, 4), torch.inf, device=raw.device)
            for step in range(args.steps):
                _, reward, _, _, _ = env.step(zero)
                lost_resets += int(raw.termination_manager.get_term("lost_wheel_contact").sum().item())
                timeout_resets += int(raw.termination_manager.get_term("time_out").sum().item())
                forces = torch.linalg.vector_norm(sensor.data.net_forces_w[:, wheel_ids], dim=-1)
                if step >= 9:
                    minimum_forces = torch.minimum(minimum_forces, forces)
                    invalid_after_transient |= torch.any(forces <= 2.0, dim=1)
                assert torch.isfinite(reward).all()
            emit(
                "ZERO_ACTION",
                {
                    "num_envs": 16,
                    "steps": args.steps,
                    "seconds_per_env": args.steps * raw.step_dt,
                    "lost_contact_resets": lost_resets,
                    "timeout_resets": timeout_resets,
                    "envs_with_invalid_contact_after_0_2_s": int(invalid_after_transient.sum().item()),
                    "wheel_names": wheel_names,
                    "minimum_force_norms_after_0_2_s": minimum_forces.detach().cpu().tolist(),
                },
            )
        finally:
            env.close()

    else:
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        cfg = parse_env_cfg(task, device=args.device, num_envs=16)
        agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        agent_cfg.device = args.device
        base_env = gym.make(task, cfg=cfg)
        env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
        try:
            env.reset()
            raw = base_env.unwrapped
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
            runner.train_mode()
            env.episode_length_buf = torch.randint_like(
                env.episode_length_buf, high=int(env.max_episode_length)
            )
            initial_counters = env.episode_length_buf.detach().cpu().tolist()
            obs = env.get_observations().to(args.device)
            physical_lengths = torch.zeros(16, dtype=torch.long, device=raw.device)
            all_completed = []
            lost_completed = []
            timeout_completed = []
            lost_resets = 0
            timeout_resets = 0
            for _ in range(args.steps):
                with torch.inference_mode():
                    actions = runner.alg.policy.act(obs)
                    assert torch.isfinite(actions).all()
                    obs, rewards, dones, _ = env.step(actions.to(env.device))
                assert torch.isfinite(rewards).all()
                physical_lengths += 1
                lost = raw.termination_manager.get_term("lost_wheel_contact").clone()
                timeout = raw.termination_manager.get_term("time_out").clone()
                for env_id in torch.nonzero(dones, as_tuple=False).flatten().tolist():
                    length = int(physical_lengths[env_id].item())
                    all_completed.append(length)
                    if lost[env_id]:
                        lost_completed.append(length)
                    if timeout[env_id]:
                        timeout_completed.append(length)
                lost_resets += int(lost.sum().item())
                timeout_resets += int(timeout.sum().item())
                physical_lengths[dones] = 0

            def stats(values):
                if not values:
                    return {"count": 0, "mean_steps": None, "median_steps": None}
                return {
                    "count": len(values),
                    "mean_steps": statistics.mean(values),
                    "median_steps": statistics.median(values),
                    "mean_seconds": statistics.mean(values) * raw.step_dt,
                    "median_seconds": statistics.median(values) * raw.step_dt,
                    "min_steps": min(values),
                    "max_steps": max(values),
                }

            emit(
                "UNTRAINED_PPO",
                {
                    "num_envs": 16,
                    "steps": args.steps,
                    "initial_episode_counters": initial_counters,
                    "init_noise_std": agent_cfg.policy.init_noise_std,
                    "lost_contact_resets": lost_resets,
                    "timeout_resets": timeout_resets,
                    "all_completed": stats(all_completed),
                    "lost_contact_completed": stats(lost_completed),
                    "timeout_completed": stats(timeout_completed),
                },
            )
        finally:
            env.close()
except BaseException as exc:
    import traceback

    traceback.print_exc()
    emit("DIAGNOSTIC_ERROR", {"type": type(exc).__name__, "message": str(exc)})
    raise
finally:
    app.close()
