"""32-env headless contract, checkpoint, rollout and one-update PPO verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", default="outputs/yaw_pos_transfer_smoke")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry
import rl_training.tasks  # noqa: F401

from yaw_pos_transfer import TASK_ID, load_base_policy, policy_contract, _canonical
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.vqr_wheel.yaw_env_cfg import VQRWheelFlatEnvCfg
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.vqr_wheel.yaw_env_pos_cfg import VQRWheelFlatEnvPOSCfg
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.vqr_wheel.yaw_env_pos_transfer_cfg import (
    POSITIVE_SPECIFIC, NEUTRAL_SPECIFIC, GLOBAL, REWARD_CLASSIFICATION,
)


def main():
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    regression_path = output / "regression_before.json"
    if not regression_path.exists():
        mdp = Path(__file__).resolve().parents[3] / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity"
        protected = [
            mdp / "config/wheeled/vqr_wheel/yaw_env_cfg.py",
            mdp / "config/wheeled/vqr_wheel/yaw_env_pos_cfg.py",
            mdp / "mdp/rewards.py", mdp / "mdp/curriculums.py",
            mdp / "mdp/yaw_pos_commands.py", mdp / "mdp/yaw_pos_rewards.py", mdp / "mdp/yaw_pos_curriculums.py",
            mdp / "velocity_yaw_env_cfg.py",
        ]
        regression_path.write_text(json.dumps({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}, indent=2) + "\n")
    base_cfg, pos_cfg = VQRWheelFlatEnvCfg(), VQRWheelFlatEnvPOSCfg()
    old_snapshots = [cfg.to_dict() for cfg in (base_cfg, pos_cfg)]
    cfg = load_cfg_from_registry(TASK_ID, "env_cfg_entry_point")
    runner_cfg = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")
    assert old_snapshots == [cfg.to_dict() for cfg in (VQRWheelFlatEnvCfg(), VQRWheelFlatEnvPOSCfg())]
    assert policy_contract(base_cfg.to_dict()) == policy_contract(cfg.to_dict())
    for name in POSITIVE_SPECIFIC + GLOBAL:
        base_term, term = getattr(base_cfg.rewards, name), getattr(cfg.rewards, name)
        assert base_term.weight == term.weight
        assert _canonical(base_term.params) == _canonical(term.params)
    cfg.scene.num_envs = 32
    cfg.sim.device = args.device
    cfg.seed = 42
    runner_cfg.device = args.device
    runner_cfg.seed = 42
    env = gym.make(TASK_ID, cfg=cfg)
    task_env = env.unwrapped
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=str(output / "ppo"), device=args.device)
    report = load_base_policy(runner, task_env, args.checkpoint)
    assert set(REWARD_CLASSIFICATION) == set(task_env.reward_manager.active_terms)
    report["reward_classification"] = REWARD_CLASSIFICATION
    report["reward_weights"] = {name: task_env.reward_manager.get_term_cfg(name).weight for name in task_env.reward_manager.active_terms}
    report["observation_dimensions"] = {name: list(dim) for name, dim in task_env.observation_manager.group_obs_dim.items()}
    action_terms = task_env.action_manager.active_terms
    report["runtime_action_order"] = [joint for name in action_terms for joint in task_env.action_manager.get_term(name)._joint_names]
    assert report["runtime_action_order"] == report["action_order"]
    assert report["observation_dimensions"] == {"policy": [55], "critic": [83]}
    assert report["action_dimension"] == 16
    with torch.inference_mode():
        obs = wrapped.get_observations()
        action, value = runner.alg.policy.act_inference(obs), runner.alg.policy.evaluate(obs)
        assert action.shape == (32, 16) and value.shape == (32, 1)
        assert torch.isfinite(action).all() and torch.isfinite(value).all()
    report["actor_critic_forward"] = "PASS"

    command = task_env.command_manager.get_term("yaw_rate_cmd")
    # All modes in the same live simulator state, with exact boundary values.
    boundary = [0.0, 0.05, 0.10, 0.100001, 0.25]
    report["boundary_commands"] = boundary
    with torch.inference_mode():
        for yaw in boundary:
            command._command[:, 0] = yaw
            active = yaw > 0.10
            for name in POSITIVE_SPECIFIC:
                term = task_env.reward_manager.get_term_cfg(name)
                actual = term.func(task_env, **term.params)
                if active:
                    expected = term.func.__wrapped__(task_env, **term.params)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                else:
                    assert torch.count_nonzero(actual) == 0, name
            for name in NEUTRAL_SPECIFIC:
                term = task_env.reward_manager.get_term_cfg(name)
                actual = term.func(task_env, **term.params)
                assert torch.isfinite(actual).all()
                if active:
                    assert torch.count_nonzero(actual) == 0, name
            for name in GLOBAL:
                term = task_env.reward_manager.get_term_cfg(name)
                actual = term.func(task_env, **term.params)
                assert torch.isfinite(actual).all(), name
                torch.testing.assert_close(actual, getattr(base_cfg.rewards, name).func(task_env, **term.params), atol=0, rtol=0)
        # A real safety penalty and regularizer remain nonzero in both modes.
        original_action = task_env.action_manager.action.clone()
        task_env.action_manager.action.add_(1.0)
        for yaw in (0.0, 0.25):
            command._command[:, 0] = yaw
            term = task_env.reward_manager.get_term_cfg("action_rate")
            assert (term.func(task_env, **term.params) > 0).all()
        task_env.action_manager.action.copy_(original_action)
        root_position = task_env.scene["robot"].data.root_pos_w
        original_position = root_position.clone()
        root_position[:, 2] = task_env.scene.env_origins[:, 2] + 0.30
        for yaw in (0.0, 0.25):
            command._command[:, 0] = yaw
            term = task_env.reward_manager.get_term_cfg("low_base_height")
            assert (term.func(task_env, **term.params) > 0).all()
        root_position.copy_(original_position)
        task_env.reset()
    report["simulator_reward_equivalence_and_masks"] = "PASS (bit-exact)"
    # Roll out 240 steps (4.8 s), enough to observe command transitions and
    # early safety resets without starting another PPO update.
    seen_metrics = set()
    with torch.inference_mode():
        for _ in range(240):
            obs = wrapped.get_observations()
            action = runner.alg.policy.act_inference(obs)
            obs, reward, done, extras = wrapped.step(action)
            assert all(torch.isfinite(obs[group]).all() for group in ("policy", "critic"))
            assert torch.isfinite(reward).all()
            seen_metrics.update(extras["log"])
    report["headless_rollout"] = {"num_envs": 32, "steps": 240, "status": "PASS"}
    required_metrics = {
        "mode_positive_fraction", "mode_neutral_fraction", "mean_abs_yaw_cmd",
        "positive_yaw_score", "positive_yaw_rate_error", "positive_tracking_ratio",
        "support_score", "lift_progress", "balance_score", "positive_success", "neutral_success",
        "neutral_four_contact_rate", "neutral_pose_error", "neutral_abs_yaw_rate", "neutral_planar_speed",
        "torso_contact", "terrain_out_of_bounds", "time_out",
    }
    assert {f"Transfer/{name}" for name in required_metrics} <= seen_metrics
    report["diagnostics"] = sorted(seen_metrics)
    assert not runner.alg.optimizer.state
    before = {key: value.clone() for key, value in runner.alg.policy.state_dict().items()}
    initial_optimizer = runner.alg.optimizer
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
    assert runner.alg.optimizer is initial_optimizer and runner.alg.optimizer.state
    after = runner.alg.policy.state_dict()
    report["updated_policy_keys"] = [key for key in before if not torch.equal(before[key], after[key])]
    assert any(key.startswith("actor.") for key in report["updated_policy_keys"])
    assert any(key.startswith("critic.") for key in report["updated_policy_keys"])
    assert task_env._yaw_task_curriculum_stage == task_env._yaw_task_curriculum_yaw_stage == 0
    runner.save(str(output / "ppo_smoke.pt"), infos={"task": TASK_ID, "transfer": report})
    assert (output / "ppo_smoke.pt").is_file()
    report["ppo_update"] = "PASS (one update, fresh optimizer, actor and critic changed, checkpoint saved)"
    regression = json.loads(regression_path.read_text())
    for name, expected_hash in regression.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected_hash, name
    report["base_and_pos_sources_unchanged"] = "PASS"
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("POS_TRANSFER_SMOKE_PASS", json.dumps(report), flush=True)
    env.close()


try:
    main()
except Exception:
    import traceback
    traceback.print_exc()
    print("POS_TRANSFER_SMOKE_FAILED", flush=True)
    raise
finally:
    app.close()
