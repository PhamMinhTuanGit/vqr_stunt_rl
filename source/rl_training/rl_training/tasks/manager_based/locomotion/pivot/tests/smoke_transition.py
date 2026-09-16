"""Small simulator check, then optional PPO through the canonical train script."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

try:
    import gymnasium as gym
    import torch
    import rl_training.tasks
    from isaaclab_tasks.utils import parse_env_cfg
    from rl_training.tasks.manager_based.locomotion.pivot.mdp.to_reference import pose_reference, contact_targets

    task = "Pivot-FourToTwoWheelRotate-v0"
    assert task in gym.registry
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env = gym.make(task, cfg=cfg)
    try:
        env.reset()
        raw = env.unwrapped
        term = raw.command_manager.get_term("pivot")
        print("JOINT_MAPPING", list(zip(term.cfg.leg_joint_names, term.joint_ids)))
        assert len(term.joint_ids) == 12
        assert raw.action_manager.total_action_dim == 16
        phase = torch.tensor([0., .5, 1.], device=raw.device)
        ref = pose_reference(term.standing, term.target, phase)
        torch.testing.assert_close(ref[0], term.standing)
        torch.testing.assert_close(ref[1], (term.standing + term.target) / 2)
        torch.testing.assert_close(ref[2], term.target)
        assert torch.all(term.command[:, 1] == 0)
        assert torch.max((term.robot.data.joint_pos[:, term.joint_ids] - term.standing).abs()) < .04
        # Exercise all levels and timings without requiring random actions to balance.
        for level in range(5):
            term.level = level
            env.reset()
            term.elapsed[:] = term.cfg.standing_time_s + term.cfg.transition_duration_s
            term._update_command()
            torch.testing.assert_close(term.reference, term.target.expand(raw.num_envs, -1))
            assert torch.all(term.command[:, 0].abs() <= term.cfg.yaw_limits[level])
            assert torch.isfinite(raw.reward_manager.compute(raw.step_dt)).all()
        term.level = 0
        env.reset()
        for _ in range(args.steps):
            action = torch.rand(raw.num_envs, 16, device=raw.device) * 2 - 1
            obs, reward, terminated, truncated, info = env.step(action)
            assert torch.isfinite(reward).all()
            for value in obs.values():
                assert torch.isfinite(value).all()
        print("SMOKE PASS", {key: tuple(value.shape) for key, value in obs.items()})
        print("METRICS", info.get("log", {}))
    finally:
        env.close()
except Exception:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
