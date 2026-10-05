"""CPU-only tests for the isolated POS skill promotion contract."""

import copy
import ast
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

MODULE = (Path(__file__).resolve().parents[1] /
          "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/yaw_pos_skill_state.py")
spec = importlib.util.spec_from_file_location("yaw_pos_skill_state_for_test", MODULE)
skill = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = skill
spec.loader.exec_module(skill)


def settings():
    return skill.SkillSettings(min_episodes=1, required_windows=1, pose_steps=0, skill_steps=0)


def episode(yaw=1.0, differential=1.0, neutral=1.0, contact=1.0, anchor=1.0,
            tracking_error=0.0, unsafe=False):
    values = {"support": 1.0, "lift": 1.0, "balance": 1.0, "height": 0.49,
              "safe": 0.0 if unsafe else 1.0, "yaw": yaw,
              "differential_episode": differential}
    samples = {"command": (0.25, 1), "error": (tracking_error, 1),
               "edge_command": (0.25, 1), "edge_error": (tracking_error, 1),
               "differential": (differential, 1), "neutral": (neutral, 1),
               "four_contact": (contact, 1), "anchor_coverage": (anchor, 1)}
    return values, samples


def pass_window(state, **kwargs):
    state.record(*episode(**kwargs))
    return state.evaluate(10)


def advance_to(phase):
    state = skill.SkillState(settings=settings())
    while state.phase < phase:
        assert pass_window(state)
        assert state.commit(10)
    return state


def test_a_pose_ignores_yaw_differential_neutral_and_tracking():
    state = skill.SkillState(settings=settings())
    values, samples = episode(yaw=0.0, differential=0.0, neutral=0.0,
                              contact=0.0, anchor=0.0, tracking_error=1.0)
    state.record(values, samples, active=False)
    assert state.evaluate(10)
    assert state.pending["phase"] == 1
    assert state.last["enabled"]["yaw"] is False
    assert state.last["enabled"]["differential"] is False
    assert state.last["enabled"]["neutral"] is False


def test_b_yaw_blocks_without_differential_or_neutral():
    state = advance_to(1)
    assert not pass_window(state, yaw=0.0, differential=0.0, neutral=0.0)
    assert "yaw" in state.last["blockers"]
    assert "differential" not in state.last["blockers"]
    assert "neutral" not in state.last["blockers"]
    assert pass_window(state, differential=0.0, neutral=0.0)


def test_c_differential_blocks_and_d_neutral_coverage_blocks():
    state = advance_to(2)
    assert not pass_window(state, differential=0.89, neutral=0.0)
    assert "differential" in state.last["blockers"]
    assert "neutral" not in state.last["blockers"]
    assert pass_window(state, neutral=0.0)
    state.commit(10)
    assert state.phase == 3
    assert not pass_window(state, neutral=0.89)
    assert "neutral" in state.last["blockers"]
    assert not pass_window(state, contact=0.89)
    assert "four_contact" in state.last["blockers"]
    assert not pass_window(state, anchor=0.89)
    assert "anchor_coverage" in state.last["blockers"]
    assert pass_window(state)


def test_e_yaw_and_robustness_require_both_final_certificates():
    state = advance_to(4)
    assert not pass_window(state, differential=0.89)
    assert state.yaw_stage == 0
    assert not pass_window(state, neutral=0.89)
    assert state.yaw_stage == 0
    assert pass_window(state)
    state.commit(10)
    assert state.yaw_stage == 1 and state.dr_scale == 0
    state.yaw_stage = len(state.settings.yaw_levels) - 1
    assert not pass_window(state, differential=0.89)
    assert state.robustness_stage == 0
    assert pass_window(state)
    state.commit(10)
    assert state.robustness_stage == 1


def test_checkpoint_roundtrip_preserves_partial_window_and_pending_boundary():
    state = skill.SkillState(settings=settings())
    state.record(*episode())
    payload = json.loads(json.dumps(state.export(12)))
    restored = skill.SkillState.restore(payload, settings(), 30)
    assert restored.window == state.window
    assert restored.stage_start_step == 18
    assert restored.evaluate(30)
    pending = json.loads(json.dumps(restored.export(31)))
    restored = skill.SkillState.restore(pending, settings(), 100)
    assert restored.pending == {"phase": 1, "yaw_stage": 0, "robustness_stage": 0}
    assert restored.commit(100) and restored.phase == 1
    changed = copy.deepcopy(pending)
    changed["settings"]["certificate"] = 0.8
    with pytest.raises(ValueError, match="mismatch"):
        skill.SkillState.restore(changed, settings(), 100)


def test_window_conjunction_safety_and_stability():
    cfg = skill.SkillSettings(min_episodes=1, required_windows=3, pose_steps=100)
    state = skill.SkillState(settings=cfg)
    for step in (10, 20):
        state.record(*episode())
        assert not state.evaluate(step)
    state.record(*episode(unsafe=True))
    assert not state.evaluate(30)
    assert "safe" in state.last["blockers"]
    assert state.consecutive_passes == 0
    for step in (40, 50, 60):
        state.record(*episode())
        assert not state.evaluate(step)
    assert state.consecutive_passes == 3
    assert "minimum_duration" in state.last["blockers"]
    state.record(*episode())
    assert state.evaluate(100)


@pytest.mark.parametrize("wheel_first", [False, True])
def test_initial_wheel_std_uses_joint_names_and_remains_learnable(wheel_first):
    path = Path(__file__).resolve().parents[1] / "scripts/reinforcement_learning/rsl_rl/yaw_pos_skill_training.py"
    names = {"action_names", "initialize_wheel_std"}
    functions = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    legs = [f"{side}_{joint}" for side in ("FL", "FR", "HL", "HR")
            for joint in ("HipX_joint", "HipY_joint", "Knee_joint")]
    wheels = [f"{side}_WHEEL" for side in ("HR", "FL", "HL", "FR")]
    terms = ["joint_vel", "joint_pos"] if wheel_first else ["joint_pos", "joint_vel"]
    manager = NS(active_terms=terms, get_term=lambda name: NS(_joint_names=wheels if name == "joint_vel" else legs))
    policy = NS(log_std=torch.nn.Parameter(torch.zeros(16)), noise_std_type="log",
                state_dependent_std=False, distribution=object())
    mapping = namespace["initialize_wheel_std"](policy, NS(action_manager=manager))
    assert policy.distribution is None
    assert set(mapping) == set(legs + wheels)
    assert all(mapping[name] == pytest.approx(1.0) for name in legs)
    assert all(mapping[name] == pytest.approx(0.15) for name in wheels)
    optimizer = torch.optim.Adam([policy.log_std], lr=0.1)
    policy.log_std.sum().backward()
    optimizer.step()
    ordered = wheels + legs if wheel_first else legs + wheels
    assert any(policy.log_std.detach()[i].exp().item() != pytest.approx(0.15)
               for i, name in enumerate(ordered)
               if name in wheels)


def test_skill_noise_keeps_wheel_bound_while_scaling_other_noise():
    path = MODULE.with_name("yaw_pos_skill_noise.py")
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name == "SkillNoiseModel")
    class BaseNoise:
        def __init__(self, cfg, num_envs, device):
            self._noise_model_cfg = cfg
    namespace = {"torch": torch, "NoiseModel": BaseNoise}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    names = tuple(f"{side}_{joint}" for side in ("FL", "FR", "HL", "HR")
                  for joint in ("HipX_joint", "HipY_joint", "Knee_joint"))
    names += ("FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL")
    cfg = NS(bound=3.0, bias_bound=0.0, joint_names=names)
    model = namespace["SkillNoiseModel"](cfg, 5000, "cpu")
    data = torch.zeros(5000, 16)
    nominal = model(data)
    assert torch.count_nonzero(nominal[:, :12]) == 0
    assert nominal[:, 12:].abs().max() <= .5
    model.set_scale(1.0)
    robust = model(data)
    assert robust[:, :12].abs().max() <= 3.0
    assert robust[:, :12].abs().max() > 2.9
    assert robust[:, 12:].abs().max() <= .5


def test_physical_randomization_uses_nominal_properties_each_stage():
    path = MODULE.with_name("yaw_pos_skill_dr.py")
    spec = importlib.util.spec_from_file_location("yaw_pos_skill_dr_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class View:
        def __init__(self):
            self.masses = torch.ones(3, 4)
            self.inertias = torch.zeros(3, 4, 9)
            self.inertias[..., (0, 4, 8)] = 1.0
            self.coms = torch.zeros(3, 4, 7)
            self.materials = torch.zeros(3, 4, 3)
        def get_masses(self): return self.masses
        def get_inertias(self): return self.inertias
        def get_coms(self): return self.coms
        def get_material_properties(self): return self.materials
        def set_masses(self, value, ids): self.masses = value.clone()
        def set_inertias(self, value, ids): self.inertias = value.clone()
        def set_coms(self, value, ids): self.coms = value.clone()
        def set_material_properties(self, value, ids): self.materials = value.clone()
    view = View()
    robot = NS(root_physx_view=view, body_names=["FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "TORSO"])
    state = skill.SkillState(settings=settings())
    env = NS(num_envs=3, scene={"robot": robot}, _yaw_pos_skill_state=state)
    module.reset_physical(env, [0, 1, 2])
    torch.testing.assert_close(view.masses, torch.ones_like(view.masses))
    torch.testing.assert_close(view.coms, torch.zeros_like(view.coms))
    torch.testing.assert_close(view.materials[..., :2], torch.ones_like(view.materials[..., :2]))
    state.phase, state.yaw_stage, state.robustness_stage = 4, len(state.settings.yaw_levels) - 1, 5
    module.reset_physical(env, [0, 1, 2])
    robust_mass = view.masses.clone()
    assert torch.all((view.materials[..., :2] >= .35) & (view.materials[..., :2] <= 1.5))
    assert torch.any(view.coms[..., :3] != 0)
    assert len(env._yaw_pos_skill_material_buckets[5]) == 1024
    module.reset_physical(env, [0, 1, 2])
    assert not torch.allclose(view.masses, robust_mass)
    assert len(env._yaw_pos_skill_material_buckets[5]) == 1024
    state.robustness_stage = 0
    module.reset_physical(env, [0, 1, 2])
    torch.testing.assert_close(view.masses, torch.ones_like(view.masses))
    torch.testing.assert_close(view.coms, torch.zeros_like(view.coms))


def test_neutral_telemetry_wrapper_preserves_old_pos_reward(monkeypatch):
    from test_yaw_pos_task import MDP, EntityCfg, _env, _load, _reward_module
    old = _reward_module(monkeypatch)
    new = _load(monkeypatch, "yaw_pos_skill_rewards", MDP / "yaw_pos_skill_rewards.py")

    def setup():
        env = _env([0.0, 0.25])
        env.num_envs, env.device, env.step_dt = 2, "cpu", 0.02
        env.episode_length_buf = torch.full((2,), 2, dtype=torch.long)
        data = env.scene["robot"].data
        data.root_pos_w = torch.zeros(2, 3)
        data.root_lin_vel_w = torch.zeros(2, 3)
        return env

    old_env, new_env = setup(), setup()
    sensor = EntityCfg("contact_forces", body_names=["FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL"])
    for _ in range(12):
        original = old.yaw_pos_neutral_position(old_env, sensor, "yaw_rate_cmd", .1)
        observed = new.skill_neutral_position(new_env, sensor, "yaw_rate_cmd", .1)
        torch.testing.assert_close(observed, original, rtol=0, atol=0)
    assert new_env._yaw_pos_skill_neutral_samples.tolist() == [12, 0]
    assert new_env._yaw_pos_skill_anchor_sum.tolist() == [3., 0.]
