"""Runtime smoke test for ``Pivot-VQR-Train-v0`` on ManagerBasedRLEnv."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4, choices=range(1, 5))
parser.add_argument("--steps", type=int, default=100)
parser.add_argument(
    "--config_only",
    action="store_true",
    help="Validate registration/config without creating a simulator scene.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 100:
    parser.error("--steps must be at least 100 for the acceptance smoke test")
launcher = AppLauncher(args)
simulation_app = launcher.app

try:
    import gymnasium as gym
    import torch

    import rl_training.tasks  # noqa: F401 -- registers repository tasks
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.envs.mdp.actions import JointEffortAction
    from isaaclab_tasks.utils import parse_env_cfg

    task_id = "Pivot-VQR-Train-v0"
    assert task_id in gym.registry
    assert gym.registry[task_id].entry_point == "isaaclab.envs:ManagerBasedRLEnv"

    cfg = parse_env_cfg(task_id, device=args.device, num_envs=args.num_envs)
    assert cfg.actions.wheel_torques.class_type is JointEffortAction
    assert cfg.actions.wheel_torques.scale == 24.0
    assert cfg.actions.wheel_torques.clip == {".*": (-24.0, 24.0)}
    assert cfg.actions.leg_residuals.clip == {".*": (-1.0, 1.0)}
    durations_cfg = (
        cfg.commands.pivot_mode.ground_duration_s,
        cfg.commands.pivot_mode.rear_up_duration_s,
        cfg.commands.pivot_mode.balance_duration_s,
        cfg.commands.pivot_mode.land_duration_s,
    )
    assert sum(durations_cfg) == cfg.episode_length_s
    assert cfg.commands.pivot_mode.resampling_time_range == (1.0e6, 1.0e6)
    assert cfg.scene.robot.actuators["wheels"].damping == 0.0
    assert cfg.scene.robot.actuators["wheels"].effort_limit == 24.0
    assert cfg.observations.policy.balance_history.history_length == 5
    assert cfg.observations.policy.balance_history.flatten_history_dim is True
    if args.config_only:
        print(
            "CONFIG PASS",
            {
                "entry_point": gym.registry[task_id].entry_point,
                "wheel_action": cfg.actions.wheel_torques.class_type.__name__,
                "wheel_peak_nm": cfg.actions.wheel_torques.scale,
                "episode_s": cfg.episode_length_s,
            },
            flush=True,
        )
        raise SystemExit(0)
    env = gym.make(task_id, cfg=cfg)
    try:
        observations, _ = env.reset()
        raw = env.unwrapped
        assert isinstance(raw, ManagerBasedRLEnv)
        assert type(raw) is ManagerBasedRLEnv
        assert all(torch.isfinite(value).all() for value in observations.values())

        leg_action = raw.action_manager.get_term("leg_residuals")
        wheel_action = raw.action_manager.get_term("wheel_torques")
        assert raw.action_manager.total_action_dim == 16
        assert leg_action.action_dim == 12
        assert wheel_action.action_dim == 4
        assert isinstance(wheel_action, JointEffortAction)

        command_term = raw.command_manager.get_term("pivot_mode")
        command = raw.command_manager.get_command("pivot_mode")
        assert command.shape == (raw.num_envs, 7)

        # Exercise the four time boundaries directly without relying on the
        # robot surviving long enough under random actions to reach LAND.
        durations = (
            command_term.cfg.ground_duration_s,
            command_term.cfg.rear_up_duration_s,
            command_term.cfg.balance_duration_s,
            command_term.cfg.land_duration_s,
        )
        sample_times = (
            0.0,
            durations[0] + raw.step_dt,
            sum(durations[:2]) + raw.step_dt,
            sum(durations[:3]) + raw.step_dt,
        )
        observed_modes = []
        saved_episode_length = raw.episode_length_buf.clone()
        for sample_time in sample_times:
            raw.episode_length_buf[:] = round(sample_time / raw.step_dt)
            command_term._update_command()
            observed_modes.append(int(command_term.mode[0].item()))
        assert observed_modes == [0, 1, 2, 3]

        # The prior phase is continuous at all three mode boundaries.
        eps = raw.step_dt
        for boundary in (
            durations[0],
            sum(durations[:2]),
            sum(durations[:3]),
        ):
            phases = []
            for sample_time in (boundary - eps, boundary, boundary + eps):
                raw.episode_length_buf[:] = round(sample_time / raw.step_dt)
                command_term._update_command()
                phases.append(command_term.pose_phase.clone())
            assert torch.max(torch.abs(phases[2] - phases[0])) < 0.01
        raw.episode_length_buf.copy_(saved_episode_length)
        env.reset()

        for _ in range(args.steps):
            action = 2.0 * torch.rand(
                raw.num_envs,
                raw.action_manager.total_action_dim,
                device=raw.device,
            ) - 1.0
            observations, reward, _, _, _ = env.step(action)
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(value).all() for value in observations.values())
            wheel_effort = raw.scene["robot"].data.applied_torque[:, wheel_action._joint_ids]
            assert torch.max(torch.abs(wheel_effort)) <= 24.0 + 1.0e-5

        history = raw.observation_manager._group_obs_term_history_buffer["policy"][
            "balance_history"
        ].buffer
        assert history.shape == (raw.num_envs, 5, 3)
        assert history.reshape(raw.num_envs, -1).shape == (raw.num_envs, 15)

        # A subset reset must leave every non-selected articulation state intact.
        if raw.num_envs > 1:
            subset = torch.tensor([0], device=raw.device, dtype=torch.long)
            untouched = torch.arange(1, raw.num_envs, device=raw.device)
            root_before = raw.scene["robot"].data.root_state_w[untouched].clone()
            joints_before = raw.scene["robot"].data.joint_pos[untouched].clone()
            raw._reset_idx(subset)
            torch.testing.assert_close(
                raw.scene["robot"].data.root_state_w[untouched], root_before
            )
            torch.testing.assert_close(
                raw.scene["robot"].data.joint_pos[untouched], joints_before
            )
            assert torch.all(command_term.mode[subset] == 0)

        print(
            "SMOKE PASS",
            {
                "env_type": type(raw).__name__,
                "num_envs": raw.num_envs,
                "action_dim": raw.action_manager.total_action_dim,
                "wheel_action_type": type(wheel_action).__name__,
                "command_shape": tuple(command.shape),
                "mode_sequence": observed_modes,
                "history_shape": tuple(history.reshape(raw.num_envs, -1).shape),
                "steps": args.steps,
            },
            flush=True,
        )
    finally:
        env.close()
except Exception:
    import traceback

    traceback.print_exc()
    raise
finally:
    simulation_app.close()
