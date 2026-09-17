"""Runtime smoke validation for the minimal four-wheel rotation task."""

import argparse
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

try:
    import gymnasium as gym
    import torch

    import rl_training.tasks  # noqa: F401
    import rl_training.tasks.manager_based.locomotion.pivot.mdp as mdp
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.utils import parse_env_cfg
    from rl_training.tasks.manager_based.locomotion.pivot.config.wheeled.vqr.robot_cfg import (
        WHEEL_BODY_NAMES,
    )

    task = "Pivot-VQR-FourWheel-Rotate-v0"
    assert task in gym.registry
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env = gym.make(task, cfg=cfg)
    try:
        observations, _ = env.reset()
        raw = env.unwrapped
        robot = raw.scene["robot"]
        contact_sensor = raw.scene.sensors["contact_forces"]

        wheel_body_ids, wheel_body_names = robot.find_bodies(WHEEL_BODY_NAMES, preserve_order=True)
        contact_body_ids, contact_body_names = contact_sensor.find_bodies(
            WHEEL_BODY_NAMES, preserve_order=True
        )
        assert wheel_body_names == WHEEL_BODY_NAMES
        assert contact_body_names == WHEEL_BODY_NAMES
        assert len(wheel_body_ids) == len(contact_body_ids) == 4

        leg_action = raw.action_manager.get_term("leg_positions")
        wheel_action = raw.action_manager.get_term("wheel_velocities")
        assert raw.action_manager.total_action_dim == 16
        assert leg_action.action_dim == 12
        assert wheel_action.action_dim == 4

        asset_cfg = SceneEntityCfg(
            "robot", body_names=WHEEL_BODY_NAMES, preserve_order=True
        )
        asset_cfg.resolve(raw.scene)
        sensor_cfg = SceneEntityCfg(
            "contact_forces", body_names=WHEEL_BODY_NAMES, preserve_order=True
        )
        sensor_cfg.resolve(raw.scene)

        command = raw.command_manager.get_command("yaw_rate_cmd")
        assert command.shape == (raw.num_envs, 3)
        assert torch.all(command[:, :2] == 0.0)
        assert torch.all(command[:, 2].abs() <= 0.3)

        initial_forces = contact_sensor.data.net_forces_w[:, contact_body_ids]
        initial_force_norms = torch.linalg.vector_norm(initial_forces, dim=-1)
        initial_lost = mdp.lost_wheel_contact(raw, sensor_cfg, threshold=2.0)
        assert initial_lost.dtype == torch.bool
        assert initial_lost.shape == (raw.num_envs,)

        print("SPAWN PASS", robot.cfg.prim_path)
        print(
            "ACTIONS",
            {
                "total": raw.action_manager.total_action_dim,
                "leg_positions": leg_action.action_dim,
                "wheel_velocities": wheel_action.action_dim,
            },
        )
        print("ACTION_JOINTS", {"legs": leg_action._joint_names, "wheels": wheel_action._joint_names})
        print("WHEEL_BODIES", list(zip(wheel_body_names, wheel_body_ids)))
        print("CONTACT_BODIES", list(zip(contact_body_names, contact_body_ids)))
        print("INITIAL_COMMAND", command.detach().cpu().tolist())
        print("INITIAL_CONTACT_FORCE_NORMS", initial_force_norms.detach().cpu().tolist())
        print("INITIAL_LOST_CONTACT", initial_lost.detach().cpu().tolist())

        zero_action = torch.zeros(
            (raw.num_envs, raw.action_manager.total_action_dim), device=raw.device
        )
        false_reset_termination = False
        reward_samples = []
        for step in range(args.steps):
            observations, reward, terminated, truncated, _ = env.step(zero_action)
            command = raw.command_manager.get_command("yaw_rate_cmd")
            forces = contact_sensor.data.net_forces_w[:, contact_body_ids]
            force_norms = torch.linalg.vector_norm(forces, dim=-1)
            contact_count = (force_norms > 2.0).sum(dim=1)
            slip = mdp.lateral_wheel_slip(raw, sensor_cfg, asset_cfg, threshold=2.0)
            lost = mdp.lost_wheel_contact(raw, sensor_cfg, threshold=2.0)
            yaw_tracking = mdp.track_ang_vel_z_exp(
                raw, std=math.sqrt(0.5), command_name="yaw_rate_cmd"
            )
            xy_tracking = mdp.track_lin_vel_xy_exp(
                raw, std=math.sqrt(0.5), command_name="yaw_rate_cmd"
            )
            upright = mdp.flat_orientation_l2(raw)

            assert torch.isfinite(slip).all()
            assert slip.shape == (raw.num_envs,)
            assert lost.dtype == torch.bool and lost.shape == (raw.num_envs,)
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(value).all() for value in observations.values())
            if step == 0 and torch.any(terminated & lost):
                false_reset_termination = bool(torch.any(contact_count < 4).item())

            reward_samples.append(
                {
                    "total": reward.detach().cpu().tolist(),
                    "yaw_tracking": yaw_tracking.detach().cpu().tolist(),
                    "xy_tracking": xy_tracking.detach().cpu().tolist(),
                    "upright_cost": upright.detach().cpu().tolist(),
                    "lateral_slip_cost": slip.detach().cpu().tolist(),
                }
            )
            print(
                "STEP",
                step,
                {
                    "yaw_command": command[:, 2].detach().cpu().tolist(),
                    "actual_yaw_rate": robot.data.root_ang_vel_b[:, 2].detach().cpu().tolist(),
                    "base_vx": robot.data.root_lin_vel_b[:, 0].detach().cpu().tolist(),
                    "base_vy": robot.data.root_lin_vel_b[:, 1].detach().cpu().tolist(),
                    "wheel_force_norms": force_norms.detach().cpu().tolist(),
                    "contact_count": contact_count.detach().cpu().tolist(),
                    "lateral_wheel_slip": slip.detach().cpu().tolist(),
                    "reward": reward.detach().cpu().tolist(),
                    "lost_wheel_contact": lost.detach().cpu().tolist(),
                    "terminated": terminated.detach().cpu().tolist(),
                    "truncated": truncated.detach().cpu().tolist(),
                },
            )

        print("FALSE_RESET_CONTACT_TERMINATION", false_reset_termination)
        print("REWARD_FIRST", reward_samples[0])
        print("REWARD_LAST", reward_samples[-1])
        print("RUNTIME PASS", flush=True)
    finally:
        env.close()
except Exception:
    import traceback

    traceback.print_exc()
    raise
finally:
    app.close()
