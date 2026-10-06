"""CPU regressions for Phase C shaping, exact certification and selective resume."""

import ast
import copy
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from test_yaw_pos_differential import _rolling_env, _differential
from test_yaw_pos_task import MDP, CONFIG, EntityCfg, _load, _reward_module
from test_yaw_pos_skill_state import skill, episode


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "scripts/reinforcement_learning/rsl_rl/yaw_pos_skill_training.py"


def _skill_rewards(monkeypatch):
    _reward_module(monkeypatch)
    return _load(monkeypatch, "yaw_pos_skill_rewards", MDP / "yaw_pos_skill_rewards.py")


def _configs():
    wheels = ["FL_WHEEL", "HR_WHEEL"]
    joints = EntityCfg("robot", joint_names=wheels)
    joints.joint_ids = [0, 3]
    return dict(support_asset_cfg=EntityCfg("robot", body_names=wheels),
                support_joint_cfg=joints,
                support_sensor_cfg=EntityCfg("contact_forces", body_names=wheels),
                command_name="yaw_rate_cmd", deadband=.1)


@pytest.mark.parametrize("fractions,expected", [
    ((0., 1.), 0.), ((.25, 1.), .5), ((.5, .5), 1.), ((2., .1), .2),
])
def test_motor_formula_uses_magnitude_and_worse_wheel(monkeypatch, fractions, expected):
    rewards = _skill_rewards(monkeypatch)
    target = torch.tensor([[-.2, .4]])
    qdot = torch.tensor([fractions]) * target.abs() / .091
    qdot[:, 0] *= -1
    score = rewards.motor_participation_scores(qdot, target).amin(1)
    assert score.item() == pytest.approx(expected)


@pytest.mark.parametrize("ratios,expected", [
    ((0., 1.), 0.), ((.25, 1.), .5), ((.5, .5), 1.), ((2., .1), .2),
    ((-.5, 1.), 0.), ((1., -.5), 0.),
])
def test_ground_formula_requires_sign_and_worse_wheel(monkeypatch, ratios, expected):
    rewards = _skill_rewards(monkeypatch)
    target = torch.tensor([[-.2, .4]])
    scores = rewards.ground_speed_coverage_scores(torch.tensor([ratios]) * target, target)
    assert scores.amin(1).item() == pytest.approx(expected)
    for index, ratio in enumerate(ratios):
        if ratio <= 0:
            assert scores[0, index].item() == 0.


def test_small_target_denominators_are_finite_and_exact(monkeypatch):
    rewards = _skill_rewards(monkeypatch)
    target = torch.tensor([[0., 1.e-5]])
    qdot = torch.tensor([[0., 1.e-4 / .091]])
    motor = rewards.motor_participation_scores(qdot, target)
    ground = rewards.ground_speed_coverage_scores(torch.tensor([[1., 5.e-5]]), target)
    assert motor.tolist()[0] == pytest.approx([0., 1.])
    assert ground.tolist()[0] == pytest.approx([0., .5])
    assert torch.isfinite(motor).all() and torch.isfinite(ground).all()


@pytest.mark.parametrize("failure", [
    "neutral", "deadband", "negative", "contact", "geometry", "heading", "target", "nan_measured",
])
def test_active_geometry_and_support_masks(monkeypatch, failure):
    rewards = _skill_rewards(monkeypatch)
    env = _rolling_env()
    data = env.scene["robot"].data
    if failure in ("neutral", "deadband", "negative"):
        env.command[:] = {"neutral": 0., "deadband": .1, "negative": -.5}[failure]
    elif failure == "contact":
        env.contacts[:, 3] = 0.
    elif failure == "geometry":
        data.body_quat_w[:, 0] = torch.tensor([2.**-.5, 2.**-.5, 0., 0.])
    elif failure == "heading":
        data.root_quat_w[:] = torch.tensor([2.**-.5, 0., 2.**-.5, 0.])
    elif failure == "target":
        data.body_pos_w[:, 0, 1] = 0.
    else:
        data.body_link_lin_vel_w[:, 0] = torch.nan
    for function in (rewards.skill_motor_participation, rewards.skill_ground_speed_coverage):
        value = function(env, **_configs())
        assert value.item() == 0. and torch.isfinite(value).all()


def test_support_order_is_checked(monkeypatch):
    rewards = _skill_rewards(monkeypatch)
    configs = _configs()
    configs["support_sensor_cfg"].body_names.reverse()
    with pytest.raises(ValueError, match="ordered FL, HR"):
        rewards.skill_motor_participation(_rolling_env(), **configs)


def test_dense_rewards_do_not_change_certificate_or_tracking(monkeypatch):
    baseline = _reward_module(monkeypatch)
    rewards = _load(monkeypatch, "yaw_pos_skill_rewards", MDP / "yaw_pos_skill_rewards.py")
    env = _rolling_env()
    env.scene["robot"].data.joint_vel[:, 0] *= .25
    env.scene["robot"].data.body_link_lin_vel_w[:, 3] *= .25
    before = _differential(baseline, env)
    counters = {k: v.clone() for k, v in vars(env).items()
                if k.startswith("_yaw_pos_differential") and isinstance(v, torch.Tensor)}
    motor = rewards.skill_motor_participation(env, **_configs())
    ground = rewards.skill_ground_speed_coverage(env, **_configs())
    assert motor.item() == pytest.approx(.5)
    assert ground.item() == pytest.approx(.5)
    for name, value in counters.items():
        assert torch.equal(value, getattr(env, name))
    assert env._yaw_pos_differential_pass_sum.item() == 0.
    after = _differential(baseline, env)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert env._yaw_pos_skill_support_fl_motor_fraction_min.item() == pytest.approx(.25)
    assert env._yaw_pos_skill_support_hr_ground_speed_coverage_min.item() == pytest.approx(.5)


def _curriculum_functions(names, rewards=None):
    path = MDP / "yaw_pos_skill_curriculums.py"
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    from fractions import Fraction
    namespace = {"torch": torch, "math": math, "Fraction": Fraction,
                 "get_state": lambda env: env._yaw_pos_skill_state,
                 "SUPPORT_COVERAGE_METRICS": rewards.SUPPORT_COVERAGE_METRICS if rewards else ()}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("passed,count,expected", [(9, 10, True), (8, 10, False), (0, 0, False)])
def test_exact_boundary_through_episode_and_window(passed, count, expected):
    state = skill.SkillState(phase=2, settings=skill.SkillSettings(
        min_episodes=1, required_windows=1, pose_steps=0, skill_steps=0))
    values, samples = episode()
    values["differential_episode"] = (torch.tensor(passed, dtype=torch.float32) / max(count, 1)).item()
    samples["differential"] = (passed, count)
    state.record(values, samples)
    assert state.window["passes"]["differential_episode"] == int(expected)
    assert state.window["successes"] == int(expected)
    assert state.evaluate(10) is expected
    assert ("differential" not in state.last["blockers"]) is expected


def test_adapter_uses_raw_counts_for_paired_boundary():
    env = _rolling_env((.25, .25, .25))
    env._yaw_pos_skill_state = skill.SkillState(phase=2)
    env.reward_manager = NS(get_term_cfg=lambda _: NS(weight=1.),
                            _episode_sums={"balance": torch.ones(3) * .04})
    env.termination_manager = NS(terminated=torch.zeros(3, dtype=torch.bool))
    env._yaw_pos_differential_pass_sum = torch.tensor([9., 8., 0.])
    env._yaw_pos_differential_pass_samples = torch.tensor([10, 10, 0])
    env._yaw_pos_differential_legacy_sum = torch.tensor([9., 8., 0.])
    env._yaw_pos_differential_legacy_samples = torch.tensor([10, 10, 0])
    values, metrics, _ = _curriculum_functions({"_values"})["_values"](env, torch.arange(3))
    assert values["differential_episode"][0].item() < .90  # Reproduce the original float32 bug.
    for name in ("differential_fixed_episode", "differential_legacy_episode"):
        assert metrics[name][0].tolist() == [1., 0., 0.]
        assert metrics[name][1].tolist() == [1, 1, 0]


def test_mean_min_aggregation_and_episode_reset(monkeypatch):
    rewards = _skill_rewards(monkeypatch)
    env = _rolling_env((.25,))
    env._yaw_pos_skill_state = skill.SkillState(phase=2)
    env.reward_manager = NS(get_term_cfg=lambda _: NS(weight=1.),
                            _episode_sums={"balance": torch.tensor([.04])})
    env.termination_manager = NS(terminated=torch.zeros(1, dtype=torch.bool))
    rewards.skill_motor_participation(env, **_configs())
    rewards.skill_ground_speed_coverage(env, **_configs())
    env.scene["robot"].data.joint_vel[:, 0] *= .25
    rewards.skill_motor_participation(env, **_configs())
    funcs = _curriculum_functions({"_values", "_clear_episode_buffers"}, rewards)
    _, metrics, _ = funcs["_values"](env, torch.tensor([0]))
    state = env._yaw_pos_skill_state
    values, samples = episode()
    samples.update({k: (v[0].item(), v[1].item()) for k, v in metrics.items()
                    if k.startswith("support_")})
    state.record(values, samples)
    assert state.rates()["support_fl_motor_fraction_mean"] == pytest.approx(.625)
    assert state.rates()["support_fl_motor_fraction_min"] == pytest.approx(.25)
    funcs["_clear_episode_buffers"](env, torch.tensor([0]))
    assert env._yaw_pos_skill_support_fl_motor_fraction_samples.item() == 0
    assert torch.isinf(env._yaw_pos_skill_support_fl_motor_fraction_min).all()
    samples["support_fl_motor_fraction_min"] = (.75, 1)
    state.record(values, samples)
    assert state.rates()["support_fl_motor_fraction_min"] == pytest.approx(.25)


def _training_functions():
    names = {"action_names", "initialize_wheel_std", "override_support_wheel_std", "reset_resume_window",
             "log_active_yaw_shaping",
             "restore_skill_state", "verify_skill_resume", "install_skill_hooks", "export_skill_state"}
    nodes = [n for n in ast.parse(TRAINING.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"torch": torch, "math": math, "Path": Path,
                 "SkillState": skill.SkillState, "SkillSettings": skill.SkillSettings,
                 "empty_window": skill.empty_window, "CHECKPOINT_KEY": skill.CHECKPOINT_KEY,
                 "EPISODE_GATES": skill.EPISODE_GATES, "WINDOW_GATES": skill.WINDOW_GATES}
    import json
    namespace["json"] = json
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAINING), "exec"), namespace)
    return namespace


def _runner_env(wheel_first=False):
    legs = [f"{side}_{joint}" for side in ("FL", "FR", "HL", "HR")
            for joint in ("HipX_joint", "HipY_joint", "Knee_joint")]
    wheels = ["HR_WHEEL", "FL_WHEEL", "HL_WHEEL", "FR_WHEEL"]
    terms = ["joint_vel", "joint_pos"] if wheel_first else ["joint_pos", "joint_vel"]
    names = wheels + legs if wheel_first else legs + wheels
    policy = torch.nn.Module()
    policy.log_std = torch.nn.Parameter(torch.linspace(.1, .8, 16).log())
    policy.actor = torch.nn.Linear(3, 2)
    policy.critic = torch.nn.Linear(4, 1)
    policy.noise_std_type, policy.state_dependent_std, policy.distribution = "log", False, object()
    optimizer = torch.optim.Adam(policy.parameters(), lr=.000225, amsgrad=True)
    sum(p.sum() for p in policy.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad()
    runner = NS(alg=NS(policy=policy, optimizer=optimizer, learning_rate=.001),
                current_learning_iteration=8500, save=lambda *_: None, log=lambda *_: None)
    env = NS(action_manager=NS(active_terms=terms, get_term=lambda name: NS(
        _joint_names=wheels if name == "joint_vel" else legs)))
    return runner, env, names


@pytest.mark.parametrize("wheel_first", [False, True])
def test_support_std_and_only_selected_adam_moments_change(wheel_first):
    runner, env, names = _runner_env(wheel_first)
    policy, optimizer = runner.alg.policy, runner.alg.optimizer
    model_before = copy.deepcopy(policy.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    indices = [names.index(name) for name in ("FL_WHEEL", "HR_WHEEL")]
    kept = [i for i in range(16) if i not in indices]
    audit = _training_functions()["override_support_wheel_std"](runner, env)
    assert policy.distribution is None and policy.log_std.requires_grad
    assert policy.log_std.detach()[indices].exp().tolist() == pytest.approx([.09, .09])
    assert torch.equal(policy.log_std.detach()[kept], model_before["log_std"][kept])
    for name in model_before.keys() - {"log_std"}:
        assert torch.equal(policy.state_dict()[name], model_before[name])
    after = optimizer.state_dict()
    assert after["param_groups"] == optimizer_before["param_groups"]
    log_std_id = after["param_groups"][0]["params"][0]
    for param_id, before in optimizer_before["state"].items():
        for key, value in before.items():
            actual = after["state"][param_id][key]
            if param_id == log_std_id and key in audit["cleared_adam_moments"]:
                assert torch.count_nonzero(actual[indices]) == 0
                assert torch.equal(actual[kept], value[kept])
            else:
                assert torch.equal(actual, value)
    assert runner.current_learning_iteration == 8500


def test_install_hooks_preserves_phase_c_and_discards_pending_before_reset(monkeypatch, tmp_path):
    runner, env, names = _runner_env()
    state = skill.SkillState(phase=2, stage_start_step=100, consecutive_passes=2)
    state.record(*episode())
    state.last = {"phase": 2, "rates": {"differential": .8}}
    state.pending = {"phase": 3, "yaw_stage": 0, "robustness_stage": 0}
    payload = state.export(250)
    original_payload = copy.deepcopy(payload)
    checkpoint = tmp_path / "model_8500.pt"
    torch.save({"model_state_dict": runner.alg.policy.state_dict(),
                "optimizer_state_dict": runner.alg.optimizer.state_dict(), "iter": 8500}, checkpoint)
    env.common_step_counter = 0
    env.observation_manager = NS(group_obs_dim={"policy": (55,), "critic": (83,)})
    env.reward_manager = NS(get_term_cfg=lambda name: NS(
        weight=1., params={"settle_time": .5, "normalized_excess": True}))
    env.command = NS(cfg=NS(yaw_rate_range=(0., .25)))
    reset_seen = []
    def reset():
        current = env._yaw_pos_skill_state
        reset_seen.append(current.pending)
        current.commit(env.common_step_counter)  # Would wrongly advance if pending was retained.
    env.reset = reset
    def apply_difficulty(task_env):
        current = task_env._yaw_pos_skill_state
        task_env.command.cfg.yaw_rate_range = (0., current.settings.yaw_levels[current.yaw_stage])
    module = types.ModuleType("rl_training.tasks.manager_based.locomotion.velocity.mdp.yaw_pos_skill_curriculums")
    module.get_state = lambda task_env: task_env._yaw_pos_skill_state
    module.apply_difficulty = apply_difficulty
    monkeypatch.setitem(sys.modules, module.__name__, module)
    audit = _training_functions()["install_skill_hooks"](
        runner, env, tmp_path / "run", checkpoint, {skill.CHECKPOINT_KEY: payload})
    resumed = env._yaw_pos_skill_state
    assert reset_seen == [None]
    assert (resumed.phase, resumed.yaw_stage, resumed.robustness_stage) == (2, 0, 0)
    assert resumed.stage_start_step == -150
    assert resumed.window == skill.empty_window()
    assert resumed.pending is None and resumed.consecutive_passes == 0 and resumed.last == {}
    assert resumed.settings.contract() == original_payload["settings"]
    assert payload == original_payload
    assert env.command.cfg.yaw_rate_range == (0., .25)
    assert audit["starting_iteration"] == 8500 and audit["yaw_limit"] == .25
    assert audit["resume"]["model_preserved"] and audit["resume"]["optimizer_preserved"]
    assert audit["action_std"]["FL_WHEEL"] == pytest.approx(.09)
    assert audit["action_std"]["HR_WHEEL"] == pytest.approx(.09)
    assert audit["reward_weights"] == {"motor_participation": 1., "ground_speed_coverage": 1.}
    assert (tmp_path / "run" / "skill_prestart.json").exists()
    assert runner.alg.learning_rate == .000225


def test_new_rewards_are_skill_only_with_unit_weights():
    tree = ast.parse((CONFIG / "yaw_env_pos_skill_cfg.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "VQRWheelYawPosSkillRewardsCfg")
    for name in ("motor_participation", "ground_speed_coverage"):
        assignment = next(n for n in cls.body if isinstance(n, ast.Assign)
                          and n.targets[0].id == name)
        assert next(k.value.value for k in assignment.value.keywords if k.arg == "weight") == 1.
    baseline = (CONFIG / "yaw_env_pos_cfg.py").read_text()
    assert "skill_motor_participation" not in baseline and "skill_ground_speed_coverage" not in baseline


def test_live_logging_before_completed_episodes():
    logged = {}
    runner = NS(writer=NS(add_scalar=lambda name, value, step: logged.update({name: (value, step)})))
    env = NS(reward_manager=NS(active_terms=["motor_participation", "ground_speed_coverage"],
             _step_reward=torch.tensor([[.2, .3], [.6, .1]]), get_term_cfg=lambda _: NS(weight=1.)))
    for wheel in ("fl", "hr"):
        for quantity in ("motor_fraction", "ground_speed_coverage"):
            base = f"_yaw_pos_skill_support_{wheel}_{quantity}"
            setattr(env, base + "_samples", torch.tensor([2, 0]))
            setattr(env, base + "_sum", torch.tensor([1.5, 0.]))
            setattr(env, base + "_min", torch.tensor([.5, torch.inf]))
    _training_functions()["log_active_yaw_shaping"](runner, env, 8500)
    assert logged["Live/skill/reward/motor_participation"] == pytest.approx((.4, 8500))
    assert logged["Live/skill/reward/ground_speed_coverage"] == pytest.approx((.2, 8500))
    assert logged["Live/skill/metric/support_fl_motor_fraction_mean"] == (.75, 8500)
    assert logged["Live/skill/metric/support_hr_ground_speed_coverage_min"] == (.5, 8500)


def test_live_logging_rejects_non_finite_reward():
    env = NS(reward_manager=NS(active_terms=["motor_participation", "ground_speed_coverage"],
             _step_reward=torch.tensor([[torch.nan, .3]]), get_term_cfg=lambda _: NS(weight=1.)))
    runner = NS(writer=NS(add_scalar=lambda *_: None))
    with pytest.raises(RuntimeError, match="Non-finite Skill shaping reward"):
        _training_functions()["log_active_yaw_shaping"](runner, env, 8500)
