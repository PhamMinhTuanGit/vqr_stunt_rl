"""Regression checks for command-change settling and faithful Skill resumes."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from test_yaw_pos_differential import _rolling_env
from test_yaw_pos_task import EntityCfg, _reward_module


ROOT = Path(__file__).resolve().parents[1]


def _step(reward, env, settle_time=.5):
    support = EntityCfg("robot", body_names=["FL_WHEEL", "HR_WHEEL"])
    joints = EntityCfg("robot", joint_names=["FL_WHEEL", "HR_WHEEL"])
    joints.joint_ids = [0, 3]
    sensor = EntityCfg("contact_forces", body_names=["FL_WHEEL", "HR_WHEEL"])
    result = reward.yaw_pos_rolling_tracking(
        env, support, joints, sensor, "yaw_rate_cmd", .1, .091, settle_time=settle_time,
    )
    env.episode_length_buf += 1
    return result


def test_positive_command_change_restarts_full_settle_per_environment(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _rolling_env((1., 1.))
    for _ in range(25):
        _step(reward, env)
    assert env._yaw_pos_differential_pass_samples.tolist() == [1, 1]
    env.command[0] = .5
    assert env._yaw_pos_previous_command[0] == 1.  # No alias to the command buffer.
    _step(reward, env)
    assert env._yaw_pos_active_age.tolist() == pytest.approx([.02, .52])
    assert env._yaw_pos_differential_pass_samples.tolist() == [1, 2]
    assert env._yaw_pos_differential_legacy_samples.tolist() == [2, 2]
    for _ in range(23):
        _step(reward, env)
    assert env._yaw_pos_differential_pass_samples[0] == 1
    _step(reward, env)
    assert env._yaw_pos_differential_pass_samples[0] == 2


def test_neutral_and_episode_reset_restart_both_timers(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _rolling_env((1., 1.))
    for _ in range(30):
        _step(reward, env)
    env.command[0] = 0.
    env.episode_length_buf[1] = 1
    _step(reward, env)
    assert env._yaw_pos_active_age.tolist() == pytest.approx([0., .02])
    assert env._yaw_pos_legacy_active_age.tolist() == pytest.approx([0., .02])
    env.command[0] = 1.
    _step(reward, env)
    assert env._yaw_pos_active_age[0] == pytest.approx(.02)


def test_settle_only_changes_certificate_sampling_not_dense_reward(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _rolling_env((1.,))
    for _ in range(25):
        _step(reward, env)
    env.command[:] = .5
    before = env._yaw_pos_differential_pass_samples.clone()
    during = _step(reward, env)
    for _ in range(24):
        after = _step(reward, env)
    torch.testing.assert_close(during, after, rtol=0, atol=0)
    assert before.item() + 1 == env._yaw_pos_differential_pass_samples.item()
    assert env._yaw_pos_differential_legacy_samples.item() == 26


def _verify_resume():
    path = ROOT / "scripts/reinforcement_learning/rsl_rl/yaw_pos_skill_training.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "verify_skill_resume")
    import math
    namespace = {"torch": torch, "Path": Path, "math": math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["verify_skill_resume"]


def test_real_checkpoint_4000_resume_preserves_model_std_optimizer_and_lr():
    checkpoint = (ROOT / "logs/rsl_rl/vqr_wheel_yaw_flat_pos_skill/"
                  "2026-10-05_15-46-34_skill_phases_scratch/model_4000.pt")
    if not checkpoint.exists():
        pytest.skip("The local checkpoint 4000 is unavailable.")
    from rsl_rl.modules import ActorCritic
    from rsl_rl.runners import OnPolicyRunner
    from tensordict import TensorDict
    observations = TensorDict({"policy": torch.zeros(1, 55), "critic": torch.zeros(1, 83)}, batch_size=[1])
    policy = ActorCritic(observations, {"policy": ["policy"], "critic": ["critic"]}, 16,
                         actor_hidden_dims=[512, 256, 128],
                         critic_hidden_dims=[512, 256, 128], activation="elu", noise_std_type="log")
    optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
    runner = NS(alg=NS(policy=policy, optimizer=optimizer, learning_rate=.001), current_learning_iteration=0)
    infos = OnPolicyRunner.load(runner, str(checkpoint), load_optimizer=True, map_location="cpu")
    audit = _verify_resume()(runner, checkpoint)
    assert runner.current_learning_iteration == 4000
    assert infos["yaw_pos_skill_curriculum"]["values"]["phase"] == 2
    assert runner.alg.learning_rate == optimizer.param_groups[0]["lr"]
    assert runner.alg.learning_rate == pytest.approx(2.25e-5)
    assert audit["model_preserved"] and audit["optimizer_preserved"]
    with torch.no_grad():
        policy.log_std[12] += .01
    with pytest.raises(RuntimeError, match="model.log_std"):
        _verify_resume()(runner, checkpoint)
