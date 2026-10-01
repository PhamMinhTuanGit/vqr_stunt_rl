"""CPU contracts for neutral/positive yaw without starting Isaac Sim."""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).parents[1]
MDP = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp"
CONFIG = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel"


class EntityCfg:
    def __init__(self, name, body_names=None, joint_names=None, **_):
        self.name = name
        self.body_names = body_names
        wheel_index = {"FL_WHEEL": 0, "FR_WHEEL": 1, "HL_WHEEL": 2, "HR_WHEEL": 3}
        self.body_ids = [wheel_index[body] for body in body_names] if isinstance(body_names, list) else []
        self.joint_names = joint_names
        self.joint_ids = list(range(12)) if joint_names else []


def _load(monkeypatch, name: str, path: Path):
    package_name = "yaw_pos_test_package"
    package = types.ModuleType(package_name)
    package.__path__ = [str(MDP)]
    monkeypatch.setitem(sys.modules, package_name, package)
    spec = importlib.util.spec_from_file_location(f"{package_name}.{name}", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _reward_module(monkeypatch):
    managers = types.ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = EntityCfg
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    baseline = types.ModuleType("yaw_pos_test_package.rewards")
    index = {"FL_WHEEL": 0, "FR_WHEEL": 1, "HL_WHEEL": 2, "HR_WHEEL": 3}
    baseline._yaw_wheel_contacts = lambda env, cfg, threshold: env.contacts[:, [index[x] for x in cfg.body_names]] >= threshold
    baseline.yaw_support_contact = lambda env, cfg, threshold: baseline._yaw_wheel_contacts(env, cfg, threshold).float().prod(dim=1)
    baseline._yaw_lift_progress = lambda env, cfg, *_: env.clearance[:, [index[x] for x in cfg.body_names]]
    baseline._yaw_support_geometry = lambda env, cfg: (env.distance, env.projection, env.length)
    baseline._yaw_whole_body_com_xy = lambda asset: asset.data.whole_body_com_xy
    baseline._yaw_command_penalty_scale = lambda env, *_: torch.ones(env.command.shape[0])
    monkeypatch.setitem(sys.modules, baseline.__name__, baseline)
    return _load(monkeypatch, "yaw_pos_rewards", MDP / "yaw_pos_rewards.py")


def _env(commands):
    commands = torch.tensor(commands, dtype=torch.float32).reshape(-1, 1)
    n = len(commands)
    data = SimpleNamespace(
        root_ang_vel_b=torch.zeros(n, 3), root_lin_vel_b=torch.zeros(n, 3),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        body_pos_w=torch.zeros(n, 4, 3), whole_body_com_xy=torch.zeros(n, 2),
        joint_pos=torch.zeros(n, 12), default_joint_pos=torch.zeros(n, 12),
        joint_vel=torch.zeros(n, 12),
    )
    env = SimpleNamespace(
        command=commands, contacts=torch.ones(n, 4), clearance=torch.zeros(n, 4),
        distance=torch.zeros(n), projection=torch.full((n,), 0.5), length=torch.ones(n),
        scene={"robot": SimpleNamespace(data=data)},
    )
    env.command_manager = SimpleNamespace(
        get_command=lambda _: env.command,
        get_term=lambda _: SimpleNamespace(cfg=SimpleNamespace(yaw_rate_range=(0.0, 0.6))),
    )
    return env


def test_masks_and_neutral_rewards(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _env([0.0, 0.05, 0.1, 0.5])
    _, active, neutral = reward.yaw_pos_masks(env, "yaw_rate_cmd", 0.1)
    assert active.tolist() == [False, False, False, True]
    assert neutral.tolist() == [True, True, True, False]

    support = EntityCfg("contact_forces", body_names=["FL_WHEEL", "HR_WHEEL"])
    lifted = EntityCfg("robot", body_names=["FR_WHEEL", "HL_WHEEL"])
    all_wheels = EntityCfg("contact_forces", body_names=["FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL"])
    legs = EntityCfg("robot", joint_names=["leg"])
    env.clearance[3, [1, 2]] = 1.0
    assert reward.yaw_pos_com_support(env, EntityCfg("robot"), 0.08, "yaw_rate_cmd", 0.1).tolist() == [0, 0, 0, 1]
    assert reward.yaw_pos_lift_clearance(env, lifted, support, 0.091, 0.05, "yaw_rate_cmd", 0.1).tolist() == [0, 0, 0, 1]
    assert reward.yaw_pos_neutral_landing_progress(env, lifted, 0.091, 0.05, "yaw_rate_cmd", 0.1).tolist() == [1, 1, 1, 0]
    assert reward.yaw_pos_four_stand_pose(env, legs, "yaw_rate_cmd", 0.1).tolist() == [1, 1, 1, 0]
    assert reward.yaw_pos_four_wheel_contact(env, all_wheels, "yaw_rate_cmd", 0.1).tolist() == [1, 1, 1, 0]
    env.scene["robot"].data.joint_pos[0] = 0.5
    env.contacts[0] = torch.tensor([1., 0., 0., 1.])
    assert reward.yaw_pos_four_stand_pose(env, legs, "yaw_rate_cmd", 0.1)[0] < 1
    assert reward.yaw_pos_four_wheel_contact(env, all_wheels, "yaw_rate_cmd", 0.1)[0] < 0.2


def test_diagonal_line_through_com_does_not_enforce_body_y_alignment(monkeypatch):
    reward = _reward_module(monkeypatch)
    baseline_path = MDP / "rewards.py"
    geometry_node = next(node for node in ast.parse(baseline_path.read_text()).body
                         if isinstance(node, ast.FunctionDef) and node.name == "_yaw_support_geometry")
    geometry_namespace = {"torch": torch, "_yaw_whole_body_com_xy": lambda asset: asset.data.whole_body_com_xy}
    future_annotations = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    geometry_module = ast.fix_missing_locations(ast.Module(body=[future_annotations, geometry_node], type_ignores=[]))
    exec(compile(geometry_module, str(baseline_path), "exec"),
         geometry_namespace)
    monkeypatch.setattr(reward, "_yaw_support_geometry", geometry_namespace["_yaw_support_geometry"])
    env = _env([0.25, 0.25, 0.25, 0.0])
    support = EntityCfg("robot", body_names=["FL_WHEEL", "HR_WHEEL"])
    data = env.scene["robot"].data
    # Both lines pass through CoM and stay within the support segment.
    fl = torch.tensor([0.15, 0.25])
    hr = -fl
    data.body_pos_w[0, 0, :2], data.body_pos_w[0, 3, :2] = fl, hr
    data.body_pos_w[1, 0, :2] = torch.tensor([0.0, 0.29])
    data.body_pos_w[1, 3, :2] = torch.tensor([0.0, -0.29])
    # Rotate the first pose by 90 degrees and translate it in world space.
    data.whole_body_com_xy[2] = torch.tensor([1.0, 2.0])
    data.root_quat_w[2] = torch.tensor([2.0**-0.5, 0.0, 0.0, 2.0**-0.5])
    data.body_pos_w[2, 0, :2] = data.whole_body_com_xy[2] + torch.tensor([-0.25, 0.15])
    data.body_pos_w[2, 3, :2] = data.whole_body_com_xy[2] + torch.tensor([0.25, -0.15])
    data.body_pos_w[3] = data.body_pos_w[0]

    distance, projection, _ = reward._yaw_support_geometry(env, support)
    assert distance[:3].tolist() == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert projection[:3].tolist() == pytest.approx([0.5, 0.5, 0.5])
    assert reward.yaw_pos_com_support(env, support, 0.08, "yaw_rate_cmd", 0.1).tolist() == pytest.approx(
        [1, 1, 1, 0], abs=1e-6
    )
    assert reward.yaw_pos_com_inside_segment(env, support, 0.05, "yaw_rate_cmd", 0.1).tolist() == [1, 1, 1, 0]
    shaped = reward.yaw_pos_heading_support(env, support, 0.27, "yaw_rate_cmd", 0.1)
    assert shaped.tolist() == pytest.approx([1.0 - 0.15 / 0.27, 1.0, 1.0 - 0.15 / 0.27, 0.0])
    current = env._yaw_pos_heading_metrics_current
    assert current[:3, 0].tolist() == pytest.approx([0.15, 0.0, 0.15])
    assert current[:3, 1].tolist() == pytest.approx([-0.15, 0.0, -0.15])
    assert current[:3, 2].tolist() == pytest.approx([0.15, 0.0, 0.15])
    assert current[0, 3].item() < 0.9
    assert current[1, 3].item() == pytest.approx(1.0)
    assert current[2, 3].item() == pytest.approx(current[0, 3].item())
    assert env._yaw_pos_heading_support_x_rms_samples.tolist() == [1, 1, 1, 0]


def test_command_sequence_changes_reward_target_without_reset(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _env([0.5])
    support = EntityCfg("contact_forces", body_names=["FL_WHEEL", "HR_WHEEL"])
    lifted = EntityCfg("robot", body_names=["FR_WHEEL", "HL_WHEEL"])
    all_wheels = EntityCfg("contact_forces", body_names=["FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL"])
    legs = EntityCfg("robot", joint_names=["leg"])
    scores = []
    for command in (0.5, 0.0, 0.5, 0.0):
        env.command[0, 0] = command
        if command:
            env.contacts[0] = torch.tensor([1., 0., 0., 1.])
            env.clearance[0] = torch.tensor([0., 1., 1., 0.])
            env.scene["robot"].data.root_ang_vel_b[0, 2] = command
        else:
            env.contacts[0] = 1.0
            env.clearance[0] = 0.0
            env.scene["robot"].data.root_ang_vel_b[0, 2] = 0.0
        yaw = reward.yaw_pos_gated_tracking(
            env, "yaw_rate_cmd", support, lifted, 0.091, 0.05, 0.30, 0.1,
        )[0]
        reward.yaw_pos_lift_clearance(env, lifted, support, 0.091, 0.05, "yaw_rate_cmd", 0.1)
        landing = reward.yaw_pos_neutral_landing_progress(env, lifted, 0.091, 0.05, "yaw_rate_cmd", 0.1)[0]
        pose = reward.yaw_pos_four_stand_pose(env, legs, "yaw_rate_cmd", 0.1)[0]
        contact = reward.yaw_pos_four_wheel_contact(env, all_wheels, "yaw_rate_cmd", 0.1)[0]
        scores.append((yaw.item(), landing.item(), pose.item(), contact.item()))
    assert scores[0] == scores[2] == (1.0, 0.0, 0.0, 0.0)
    assert scores[1] == scores[3] == (1.0, 1.0, 1.0, 1.0)
    assert env._yaw_tracking_metric_samples.item() == 2
    assert env._yaw_pos_neutral_samples.item() == 2
    assert env._yaw_lift_min_progress_samples.item() == 2


def test_neutral_yaw_and_landing_improve_before_four_wheel_contact(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _env([0.0])
    env.contacts[0] = torch.tensor([1., 0., 0., 1.])  # FL+HR support; FR+HL still airborne.
    support = EntityCfg("contact_forces", body_names=["FL_WHEEL", "HR_WHEEL"])
    lifted = EntityCfg("robot", body_names=["FR_WHEEL", "HL_WHEEL"])
    all_wheels = EntityCfg("contact_forces", body_names=["FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL"])
    index = {"FR_WHEEL": 1, "HL_WHEEL": 2}
    monkeypatch.setattr(
        reward, "_yaw_lift_progress",
        lambda env, cfg, radius, target: (env.clearance[:, [index[name] for name in cfg.body_names]] / target).clamp(0, 1),
    )
    env.clearance[0, [1, 2]] = 0.05
    yaw_scores = []
    for yaw_rate in (0.7, 0.35, 0.0):
        env.scene["robot"].data.root_ang_vel_b[0, 2] = yaw_rate
        yaw_scores.append(reward.yaw_pos_gated_tracking(
            env, "yaw_rate_cmd", support, lifted, 0.091, 0.05, 0.30, 0.1,
        )[0].item())
    assert 0 < yaw_scores[0] < yaw_scores[1] < yaw_scores[2] == 1.0
    env.contacts[0] = 1.0
    env.scene["robot"].data.root_ang_vel_b[0, 2] = 0.35
    full_contact_yaw_score = reward.yaw_pos_gated_tracking(
        env, "yaw_rate_cmd", support, lifted, 0.091, 0.05, 0.30, 0.1,
    )[0].item()
    assert full_contact_yaw_score == pytest.approx(yaw_scores[1])
    env.contacts[0] = torch.tensor([1., 0., 0., 1.])

    landing_scores = []
    for fr_clearance, hl_clearance in ((0.05, 0.05), (0.025, 0.05), (0.025, 0.025),
                                       (0.0, 0.025), (0.0, 0.0)):
        env.clearance[0, [1, 2]] = torch.tensor([fr_clearance, hl_clearance])
        landing_scores.append(reward.yaw_pos_neutral_landing_progress(
            env, lifted, 0.091, 0.05, "yaw_rate_cmd", 0.1,
        )[0].item())
    assert landing_scores == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert reward.yaw_pos_four_wheel_contact(env, all_wheels, "yaw_rate_cmd", 0.1)[0].item() == pytest.approx(0.1)
    assert env._yaw_pos_neutral_four_contact_sum.item() == 0.0


def test_pos_active_yaw_std_rejects_zero_yaw_without_changing_neutral(monkeypatch):
    config = ast.parse((CONFIG / "yaw_env_pos_cfg.py").read_text())
    rewards_cfg = next(node for node in config.body
                       if isinstance(node, ast.ClassDef) and node.name == "VQRWheelYawPosRewardsCfg")
    yaw_cfg = next(node.value for node in rewards_cfg.body
                   if isinstance(node, ast.Assign) and node.targets[0].id == "gated_yaw_tracking")
    params = next(keyword.value for keyword in yaw_cfg.keywords if keyword.arg == "params")
    values = {ast.literal_eval(key): ast.literal_eval(value)
              for key, value in zip(params.keys, params.values)
              if isinstance(value, ast.Constant)}
    assert values["std"] == 0.20
    assert values["neutral_std"] == 0.30

    reward = _reward_module(monkeypatch)
    env = _env([0.15, 0.25, 0.0])
    env.contacts[:] = torch.tensor([1.0, 0.0, 0.0, 1.0])
    env.clearance[:, [1, 2]] = 1.0
    support = EntityCfg("contact_forces", body_names=["FL_WHEEL", "HR_WHEEL"])
    lifted = EntityCfg("robot", body_names=["FR_WHEEL", "HL_WHEEL"])

    def yaw_scores():
        return reward.yaw_pos_gated_tracking(
            env, "yaw_rate_cmd", support, lifted, 0.091, 0.05,
            values["std"], 0.1, neutral_std=values["neutral_std"],
        ).tolist()

    zero_yaw = yaw_scores()
    assert zero_yaw == pytest.approx([math.exp(-(0.15 / 0.20) ** 2),
                                      math.exp(-(0.25 / 0.20) ** 2), 1.0])
    assert zero_yaw[0] < 0.6
    assert zero_yaw[1] < 0.25

    env.scene["robot"].data.root_ang_vel_b[:, 2] = torch.tensor([0.15, 0.245, 0.30])
    matched_yaw = yaw_scores()
    assert matched_yaw[0] == pytest.approx(1.0)
    assert matched_yaw[1] > 0.99
    assert matched_yaw[2] == pytest.approx(math.exp(-1.0))


def test_sampler_and_external_command(monkeypatch):
    utils = types.ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils)
    baseline = types.ModuleType("yaw_pos_test_package.commands")

    class BaseCommand:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self.device = "cpu"
            self.num_envs = env.num_envs
            self._command = torch.zeros(env.num_envs, 1)

    baseline.YawRateCommand = BaseCommand
    baseline.YawRateCommandCfg = type("YawRateCommandCfg", (), {})
    monkeypatch.setitem(sys.modules, baseline.__name__, baseline)
    module = _load(monkeypatch, "yaw_pos_commands", MDP / "yaw_pos_commands.py")
    cfg = SimpleNamespace(neutral_probability=0.30, deadband=0.1, yaw_rate_range=(0.0, 0.6), external_control=False)
    term = module.YawPosCommand(cfg, SimpleNamespace(num_envs=10000))
    torch.manual_seed(7)
    term._resample_command(slice(None))
    values = term._command[:, 0]
    assert 0.27 < (values == 0).float().mean() < 0.33
    assert torch.all((values == 0) | ((values > 0.1) & (values <= 0.6)))
    term.set_external_command(torch.tensor([0.0, 0.05, 0.5]), [0, 1, 2])
    assert term._command[:3, 0].tolist() == [0.0, 0.0, 0.5]
    term._resample_command([0, 1, 2])
    assert term._command[:3, 0].tolist() == [0.0, 0.0, 0.5]
    term.set_external_command(-0.5, [2])
    assert term._command[2, 0] == 0
    term.use_training_sampler()
    term._resample_command([2])
    assert term._command[2, 0] >= 0


def test_new_registration_runner_and_existing_config_are_isolated():
    registration = (CONFIG / "__init__.py").read_text()
    runner = (CONFIG / "agents/rsl_rl_ppo_cfg.py").read_text()
    pos = ast.parse((CONFIG / "yaw_env_pos_cfg.py").read_text())
    assert 'id="Flat-VQR-Wheel-Yaw-POS"' in registration
    assert "VQRWheelFlatEnvPOSCfg" in registration
    assert "VQRWheelYawFlatPOSPPORunnerCfg" in registration
    assert 'self.experiment_name = "vqr_wheel_yaw_flat_pos"' in runner
    classes = {node.name: node for node in pos.body if isinstance(node, ast.ClassDef)}
    assert "VQRWheelFlatEnvPOSCfg" in classes
    reward_names = {node.targets[0].id for node in classes["VQRWheelYawPosRewardsCfg"].body if isinstance(node, ast.Assign)}
    assert reward_names == {"com_support", "support_span_band", "lift_clearance", "com_inside_segment",
                            "heading_support",
                            "gated_yaw_tracking", "lifted_wheel_spin", "neutral_landing_progress",
                            "four_stand_pose", "four_wheel_contact"}


def test_curriculum_uses_active_metrics_and_logs_neutral_separately(monkeypatch):
    baseline = types.ModuleType("yaw_pos_test_package.curriculums")
    received = {}

    def fake_yaw_task_levels(*args, **kwargs):
        received.update(kwargs)
        assert args[0]._yaw_tracking_metric_samples.tolist() == [2]
        assert args[0]._yaw_pos_neutral_samples.tolist() == [2]
        return {"yaw_limit": torch.tensor(0.25)}

    baseline.yaw_task_levels = fake_yaw_task_levels
    monkeypatch.setitem(sys.modules, baseline.__name__, baseline)
    module = _load(monkeypatch, "yaw_pos_curriculums", MDP / "yaw_pos_curriculums.py")
    reward_configs = {
        "lift_clearance": SimpleNamespace(params={"target_clearance": 0.10}),
        "neutral_landing_progress": SimpleNamespace(params={"target_clearance": 0.05}),
    }
    env = SimpleNamespace(
        num_envs=1, device="cpu", _yaw_tracking_metric_samples=torch.tensor([2]),
        _yaw_pos_neutral_samples=torch.tensor([2]),
        reward_manager=SimpleNamespace(
            get_term_cfg=lambda name: reward_configs[name],
            set_term_cfg=lambda name, cfg: reward_configs.__setitem__(name, cfg),
        ),
        _yaw_pos_neutral_four_contact_sum=torch.tensor([2.]),
        _yaw_pos_neutral_four_contact_samples=torch.tensor([2]),
        _yaw_pos_neutral_pose_error_sum=torch.tensor([0.2]),
        _yaw_pos_neutral_pose_error_samples=torch.tensor([2]),
        _yaw_pos_neutral_abs_yaw_rate_sum=torch.tensor([0.4]),
        _yaw_pos_neutral_abs_yaw_rate_samples=torch.tensor([2]),
        _yaw_pos_neutral_planar_speed_sum=torch.tensor([0.6]),
        _yaw_pos_neutral_planar_speed_samples=torch.tensor([2]),
        _yaw_pos_heading_fl_x_from_com_sum=torch.tensor([0.12]),
        _yaw_pos_heading_fl_x_from_com_samples=torch.tensor([2]),
        _yaw_pos_heading_hr_x_from_com_sum=torch.tensor([-0.08]),
        _yaw_pos_heading_hr_x_from_com_samples=torch.tensor([2]),
        _yaw_pos_heading_support_x_rms_sum=torch.tensor([0.14]),
        _yaw_pos_heading_support_x_rms_samples=torch.tensor([2]),
        _yaw_pos_heading_support_line_body_y_alignment_sum=torch.tensor([1.8]),
        _yaw_pos_heading_support_line_body_y_alignment_samples=torch.tensor([2]),
    )
    result = module.yaw_pos_task_levels(
        env, [0], "yaw_rate_cmd", (0.05,), (0.25,), (0.30,), (0.30,), (0.20,),
        "lift_clearance", "balance", "gated_yaw_tracking", "torso_contact", 0.35,
        0.85, 0.8, 0.75, 0.65, 2048, 0.85, 3, 1000, 6000,
    )
    assert received["active_only"] is True
    assert result["mode_positive_fraction"] == pytest.approx(0.5)
    assert result["neutral_four_contact_rate"] == pytest.approx(1.0)
    assert result["neutral_pose_error"] == pytest.approx(0.1)
    assert result["neutral_abs_yaw_rate"] == pytest.approx(0.2)
    assert result["neutral_planar_speed"] == pytest.approx(0.3)
    assert result["heading_fl_x_from_com_m"] == pytest.approx(0.06)
    assert result["heading_hr_x_from_com_m"] == pytest.approx(-0.04)
    assert result["heading_support_x_rms_m"] == pytest.approx(0.07)
    assert result["heading_support_line_body_y_alignment"] == pytest.approx(0.9)
    assert reward_configs["neutral_landing_progress"].params["target_clearance"] == pytest.approx(0.10)
    assert env._yaw_pos_neutral_four_contact_sum.item() == 0
    assert env._yaw_pos_heading_support_x_rms_sum.item() == 0


def test_neutral_only_episode_cannot_promote_yaw_ladder(monkeypatch):
    assets = types.ModuleType("isaaclab.assets")
    assets.Articulation = object
    managers = types.ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = EntityCfg
    terrains = types.ModuleType("isaaclab.terrains")
    terrains.TerrainImporter = object
    monkeypatch.setitem(sys.modules, "isaaclab.assets", assets)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    monkeypatch.setitem(sys.modules, "isaaclab.terrains", terrains)
    module = _load(monkeypatch, "curriculums", MDP / "curriculums.py")

    class ConfigManager:
        def __init__(self, configs):
            self.configs = configs

        def get_term_cfg(self, name):
            return self.configs[name]

        def set_term_cfg(self, name, cfg):
            self.configs[name] = cfg

    command_term = SimpleNamespace(cfg=SimpleNamespace(yaw_rate_range=(0.0, 0.25)))
    reward_manager = ConfigManager({
        "lift_clearance": SimpleNamespace(params={"target_clearance": 0.05}),
        "gated_yaw_tracking": SimpleNamespace(params={"target_clearance": 0.05}),
    })
    event_manager = ConfigManager({
        "randomize_apply_external_force_torque": SimpleNamespace(params={}),
        "randomize_actuator_gains": SimpleNamespace(params={}),
        "randomize_push_robot": SimpleNamespace(params={}),
        "randomize_reset_base": SimpleNamespace(params={"pose_range": {}, "velocity_range": {}}),
    })
    env = SimpleNamespace(
        common_step_counter=100, num_envs=1, device="cpu", step_dt=0.02,
        episode_length_buf=torch.tensor([100]),
        _yaw_tracking_metric_samples=torch.tensor([0]),
        command_manager=SimpleNamespace(get_term=lambda _: command_term),
        reward_manager=reward_manager, event_manager=event_manager,
    )
    result = module.yaw_task_levels(
        env, [0], "yaw_rate_cmd", (0.05,), (0.25, 0.40), (0.30, 0.40),
        (0.30, 0.35), (0.20, 0.25), "lift_clearance", "balance",
        "gated_yaw_tracking", "torso_contact", 0.35, 0.85, 0.80,
        0.75, 0.65, 1, 0.85, 1, 0, 0, active_only=True,
    )
    assert result["pending_evaluated_episodes"].item() == 0
    assert result["yaw_stage"].item() == 0
    assert command_term.cfg.yaw_rate_range == (0.0, 0.25)


def test_pos_resume_restores_and_installs_yaw_curriculum_checkpoint_hook():
    train = ROOT / "scripts/reinforcement_learning/rsl_rl/train.py"
    main = next(node for node in ast.parse(train.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    selection = next(node.value for node in main.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "yaw_task_env"
                             for target in node.targets))
    task_env = SimpleNamespace(_yaw_task_curriculum_stage=2, _yaw_task_curriculum_yaw_stage=1)
    env = SimpleNamespace(unwrapped=task_env)
    task_name = "Flat-VQR-Wheel-Yaw-POS"
    selected = eval(compile(ast.Expression(selection), str(train), "eval"),
                    {"task_name": task_name, "env": env})
    assert selected is task_env

    resume_block = next(node for node in main.body if isinstance(node, ast.If)
                        and ast.unparse(node.test) == "yaw_task_env is not None")
    calls = []
    runner = SimpleNamespace()
    checkpoint_infos = {"yaw_curriculum": {"version": 1}}
    namespace = {
        "task_name": task_name, "yaw_task_env": selected,
        "checkpoint_infos": checkpoint_infos, "runner": runner,
        "agent_cfg": SimpleNamespace(resume=True),
        "_restore_yaw_curriculum_state": lambda task, info: calls.append(("restore", task, info)) or True,
        "_install_yaw_curriculum_checkpointing": lambda run, task: calls.append(("hook", run, task)),
    }
    exec(compile(ast.Module(body=[resume_block], type_ignores=[]), str(train), "exec"), namespace)
    assert calls == [("restore", task_env, checkpoint_infos), ("hook", runner, task_env)]

    reward_guard = next(node for node in main.body if isinstance(node, ast.If)
                        and any(isinstance(child, ast.Name) and child.id == "_verify_yaw_reward_config"
                                for child in ast.walk(node)))
    assert "Flat-VQR-Wheel-Yaw-POS" not in ast.unparse(reward_guard.test)
