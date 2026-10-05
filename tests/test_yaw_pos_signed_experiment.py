"""POS experiment contracts: selective noise/resume, diagnostics and strict gates."""

import ast
import copy
import math
import os
from types import SimpleNamespace as NS

import pytest
import torch

from test_yaw_pos_task import MDP, ROOT, _load, _reward_module
from test_yaw_pos_differential import _rolling_env, _differential, _curriculum_env, _evidence


def _resume_helpers():
    path = ROOT / "scripts/reinforcement_learning/rsl_rl/train.py"
    names = {"_configure_pos_resume_support_wheel_std", "_verify_pos_resume_state"}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"torch": torch, "math": math, "os": os, "OnPolicyRunner": object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return NS(**{name: namespace[name] for name in names})


def test_wheel_noise_bounds_and_leg_samples_are_preserved(monkeypatch):
    module = _load(monkeypatch, "yaw_pos_noise", MDP / "yaw_pos_noise.py")
    names = tuple(f"{side}_{joint}" for side in ("FL", "FR", "HL", "HR")
                  for joint in ("HipX_joint", "HipY_joint", "Knee_joint"))
    names += ("FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL")
    cfg = module.YawPosJointVelocityNoiseCfg(joint_names=names)
    data = torch.zeros(20000, 16)
    torch.manual_seed(71)
    result = cfg.func(data, cfg)
    torch.manual_seed(71)
    original = 6.0 * torch.rand_like(data) - 3.0
    torch.testing.assert_close(result[:, :12], original[:, :12], atol=3.e-7, rtol=1.e-6)
    assert result.shape == data.shape
    assert result[:, 12:].abs().max() <= .5
    assert result[:, :12].abs().max() > 2.99
    assert result[:, 12:].std().item() == pytest.approx(.5 / math.sqrt(3), rel=.02)
    assert data.count_nonzero() == 0
    with pytest.raises(ValueError, match="joint order"):
        cfg.func(torch.zeros(2, 15), cfg)


def test_signed_diagnostics_measure_each_wheel_and_report_overspeed(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _rolling_env((1., 1., 1.))
    data = env.scene["robot"].data
    data.body_link_lin_vel_w[:, 0, 0] = torch.tensor([-.15, .3, -.6])
    data.body_link_lin_vel_w[:, 3, 0] = torch.tensor([.6, .3, .3])
    _differential(reward, env)
    metrics = env._yaw_pos_differential_metrics_current
    assert metrics["differential_fl_signed_ratio"].tolist() == pytest.approx([.5, -1., 2.])
    assert metrics["differential_hr_signed_ratio"].tolist() == pytest.approx([2., 1., 1.])
    assert metrics["differential_any_wrong_sign_pct"].tolist() == [0., 100., 0.]
    assert metrics["differential_fl_overspeed"].tolist() == pytest.approx([0., 0., 1.])
    assert metrics["differential_hr_overspeed_pct"].tolist() == [100., 0., 0.]


def test_pos_push_is_reduced_without_relaxing_behavior_gates(monkeypatch):
    module, env, params = _curriculum_env(monkeypatch)
    _evidence(env, differential=89., neutral=100.)
    result = module.yaw_pos_task_levels(env, [0, 1], **params)
    assert not result["window_passed"]
    assert env._yaw_task_curriculum_yaw_stage == 5
    assert env.event_manager.get_term_cfg("randomize_push_robot").params["velocity_range"] == {
        "x": (-.15, .15), "y": (-.15, .15),
    }
    assert env.event_manager.get_term_cfg("randomize_apply_external_force_torque").params["force_range"] == (-10., 10.)


@pytest.mark.parametrize("wheel_first", [False, True])
def test_resume_changes_only_named_support_std_and_its_moments(wheel_first):
    helpers = _resume_helpers()
    legs = [f"{side}_{joint}" for side in ("FL", "FR", "HL", "HR")
            for joint in ("HipX_joint", "HipY_joint", "Knee_joint")]
    wheels = ["HR_WHEEL", "FR_WHEEL", "FL_WHEEL", "HL_WHEEL"]
    terms = ["joint_vel", "joint_pos"] if wheel_first else ["joint_pos", "joint_vel"]
    names = wheels + legs if wheel_first else legs + wheels
    policy = NS(log_std=torch.nn.Parameter(torch.linspace(.02, .5, 16).log()),
                noise_std_type="log", state_dependent_std=False, distribution=object())
    state = {"step": torch.tensor(77.), "exp_avg": torch.arange(16.).clone(),
             "exp_avg_sq": torch.arange(16.).clone() + 1.}
    runner = NS(alg=NS(policy=policy, optimizer=NS(state={policy.log_std: state})))
    env = NS(action_manager=NS(active_terms=terms,
                             get_term=lambda term: NS(_joint_names=legs if term == "joint_pos" else wheels)))
    before = policy.log_std.detach().clone()
    helpers._configure_pos_resume_support_wheel_std(runner, env, .12)
    changed = [names.index(name) for name in ("FL_WHEEL", "HR_WHEEL")]
    kept = [i for i in range(16) if i not in changed]
    assert torch.equal(policy.log_std[kept], before[kept])
    torch.testing.assert_close(policy.log_std[changed].exp(), torch.full((2,), .12))
    assert state["step"] == 77.
    assert torch.equal(state["exp_avg"][kept], torch.arange(16.)[kept])
    assert torch.equal(state["exp_avg_sq"][kept], (torch.arange(16.) + 1.)[kept])
    assert state["exp_avg"][changed].count_nonzero() == 0
    assert state["exp_avg_sq"][changed].count_nonzero() == 0
    assert policy.distribution is None


def test_latest_real_checkpoint_preserves_networks_optimizer_and_other_std(tmp_path, monkeypatch):
    import yaml
    from rsl_rl.runners import OnPolicyRunner
    from tensordict import TensorDict
    monkeypatch.syspath_prepend(str(ROOT / "scripts/reinforcement_learning/rsl_rl"))
    from yaw_pos_adaptation import _ConfigLoader

    checkpoint = ROOT / "logs/rsl_rl/vqr_wheel_yaw_flat_pos/2026-10-02_15-05-31/model_36999.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = yaml.load((checkpoint.parent / "params/agent.yaml").read_text(), Loader=_ConfigLoader)
    config["device"] = "cpu"
    obs = TensorDict({"policy": torch.zeros(2, 55), "critic": torch.zeros(2, 83)}, batch_size=[2])
    vec = NS(num_envs=2, num_actions=16, device="cpu", get_observations=lambda: obs)
    runner = OnPolicyRunner(vec, config, log_dir=str(tmp_path), device="cpu")
    runner.load(str(checkpoint), load_optimizer=True, map_location="cpu")
    before_optimizer = copy.deepcopy(runner.alg.optimizer.state_dict())
    legs = [f"{side}_{joint}" for side in ("FL", "FR", "HL", "HR")
            for joint in ("HipX_joint", "HipY_joint", "Knee_joint")]
    wheels = ["FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL"]
    env = NS(action_manager=NS(active_terms=["joint_pos", "joint_vel"],
                             get_term=lambda term: NS(_joint_names=legs if term == "joint_pos" else wheels)))
    helpers = _resume_helpers()
    helpers._configure_pos_resume_support_wheel_std(runner, env, .12)
    helpers._verify_pos_resume_state(runner, env, str(checkpoint), 1000, 500, support_wheel_std=.12)
    changed = [12, 15]
    kept = [i for i in range(16) if i not in changed]
    model = runner.alg.policy.state_dict()
    for name, tensor in saved["model_state_dict"].items():
        if name == "log_std":
            assert torch.equal(model[name][kept], tensor[kept])
        else:
            assert torch.equal(model[name], tensor), name
    assert runner.current_learning_iteration == 36999
    after = runner.alg.optimizer.state_dict()
    assert after["param_groups"] == before_optimizer["param_groups"]
    std_state_index = next(index for group, live_group in zip(after["param_groups"], runner.alg.optimizer.param_groups)
                           for index, parameter in zip(group["params"], live_group["params"])
                           if parameter is runner.alg.policy.log_std)
    for index, state in before_optimizer["state"].items():
        for name, value in state.items():
            actual = after["state"][index][name]
            if index == std_state_index and isinstance(value, torch.Tensor) and value.shape == (16,):
                assert torch.equal(actual[kept], value[kept])
                assert actual[changed].count_nonzero() == 0
            elif isinstance(value, torch.Tensor):
                assert torch.equal(actual, value)
            else:
                assert actual == value
