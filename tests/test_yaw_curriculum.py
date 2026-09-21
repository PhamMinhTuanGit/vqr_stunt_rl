"""CPU-only tests for Flat-VQR-Wheel-Yaw curriculum gates."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).parents[1]
CURRICULUM_PATH = (
    REPO_ROOT
    / "source"
    / "rl_training"
    / "rl_training"
    / "tasks"
    / "manager_based"
    / "locomotion"
    / "velocity"
    / "mdp"
    / "curriculums.py"
)
YAW_CONFIG_PATH = (
    REPO_ROOT
    / "source"
    / "rl_training"
    / "rl_training"
    / "tasks"
    / "manager_based"
    / "locomotion"
    / "velocity"
    / "config"
    / "wheeled"
    / "vqr_wheel"
    / "yaw_env_cfg.py"
)


def _load_curriculums_module(monkeypatch: pytest.MonkeyPatch):
    """Load the module without starting Isaac Sim."""

    class SceneEntityCfg:
        def __init__(self, name: str, *args, **kwargs):
            self.name = name

    isaaclab = types.ModuleType("isaaclab")
    assets = types.ModuleType("isaaclab.assets")
    managers = types.ModuleType("isaaclab.managers")
    terrains = types.ModuleType("isaaclab.terrains")
    assets.Articulation = object
    managers.SceneEntityCfg = SceneEntityCfg
    terrains.TerrainImporter = object
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.assets", assets)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    monkeypatch.setitem(sys.modules, "isaaclab.terrains", terrains)

    spec = importlib.util.spec_from_file_location("yaw_curriculum_under_test", CURRICULUM_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _curriculum_param_keys_from_config() -> set[str]:
    tree = ast.parse(YAW_CONFIG_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "VQRWheelYawCurriculumCfg":
            assignment = next(
                item for item in node.body if isinstance(item, ast.Assign) and item.targets[0].id == "task_levels"
            )
            call = assignment.value
            params_keyword = next(keyword for keyword in call.keywords if keyword.arg == "params")
            return {ast.literal_eval(key) for key in params_keyword.value.keys}
    raise AssertionError("VQRWheelYawCurriculumCfg.task_levels was not found")


def _literal_config_assignment(name: str):
    tree = ast.parse(YAW_CONFIG_PATH.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    )
    return ast.literal_eval(assignment.value)


def test_yaw_tracking_ratio_rejects_zero_yaw_policy(monkeypatch: pytest.MonkeyPatch):
    curriculum = _load_curriculums_module(monkeypatch)

    command = torch.tensor(0.5)
    assert curriculum.yaw_tracking_ratio(command, command).item() == pytest.approx(0.0)
    assert curriculum.yaw_tracking_ratio(torch.tensor(0.1), command).item() == pytest.approx(0.8)


def test_stage_tables_must_be_aligned_and_monotonic(monkeypatch: pytest.MonkeyPatch):
    curriculum = _load_curriculums_module(monkeypatch)

    curriculum.validate_yaw_curriculum_levels(
        (0.05, 0.10, 0.20),
        (0.25, 0.40, 1.00),
        (0.30, 0.50, 1.00),
        (0.30, 0.40, 0.55),
        (0.20, 0.30, 0.45),
    )

    with pytest.raises(ValueError, match="equal non-zero lengths"):
        curriculum.validate_yaw_curriculum_levels(
            (0.05,),
            (0.25, 1.00),
            (0.30,),
            (0.30, 0.55),
            (0.20, 0.45),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        curriculum.validate_yaw_curriculum_levels(
            (0.05,),
            (0.25, 0.25),
            (0.30, 0.50),
            (0.30, 0.40),
            (0.20, 0.30),
        )


def test_config_supplies_every_required_curriculum_parameter(monkeypatch: pytest.MonkeyPatch):
    curriculum = _load_curriculums_module(monkeypatch)
    required = set(inspect.signature(curriculum.yaw_task_levels).parameters) - {"env", "env_ids"}

    assert _curriculum_param_keys_from_config() == required


def test_config_reaches_one_rad_per_second_and_full_online_dr():
    yaw_levels = _literal_config_assignment("YAW_RATE_LEVELS")
    dr_levels = _literal_config_assignment("ONLINE_DR_SCALE_LEVELS")

    assert yaw_levels == (0.25, 0.40, 0.55, 0.70, 0.85, 1.00)
    assert len(dr_levels) == len(yaw_levels)
    assert dr_levels[0] == pytest.approx(0.30)
    assert dr_levels[-1] == pytest.approx(1.00)


def test_restored_final_stage_reapplies_command_and_online_dr(monkeypatch: pytest.MonkeyPatch):
    curriculum = _load_curriculums_module(monkeypatch)

    class ConfigManager:
        def __init__(self, configs):
            self.configs = configs

        def get_term_cfg(self, name):
            return self.configs[name]

        def set_term_cfg(self, name, cfg):
            self.configs[name] = cfg

    command_term = SimpleNamespace(cfg=SimpleNamespace(yaw_rate_range=(-0.25, 0.25)))
    command_manager = SimpleNamespace(get_term=lambda name: command_term)
    reward_manager = ConfigManager(
        {
            "lift_clearance": SimpleNamespace(params={"target_clearance": 0.05}),
            "gated_yaw_tracking": SimpleNamespace(params={"target_clearance": 0.05}),
        }
    )
    event_manager = ConfigManager(
        {
            "randomize_apply_external_force_torque": SimpleNamespace(
                params={"force_range": (0.0, 0.0), "torque_range": (0.0, 0.0)}
            ),
            "randomize_actuator_gains": SimpleNamespace(
                params={
                    "stiffness_distribution_params": (1.0, 1.0),
                    "damping_distribution_params": (1.0, 1.0),
                }
            ),
            "randomize_push_robot": SimpleNamespace(params={"velocity_range": {}}),
            "randomize_reset_base": SimpleNamespace(
                params={"pose_range": {}, "velocity_range": {}}
            ),
        }
    )
    env = SimpleNamespace(
        common_step_counter=0,
        num_envs=2,
        device="cpu",
        episode_length_buf=torch.zeros(2, dtype=torch.long),
        step_dt=0.02,
        command_manager=command_manager,
        reward_manager=reward_manager,
        event_manager=event_manager,
        _yaw_task_curriculum_stage=3,
        _yaw_task_curriculum_yaw_stage=5,
    )

    state = curriculum.yaw_task_levels(
        env,
        [],
        command_name="yaw_rate_cmd",
        clearance_levels=(0.05, 0.10, 0.15, 0.20),
        yaw_rate_levels=(0.25, 0.40, 0.55, 0.70, 0.85, 1.00),
        dr_scale_levels=(0.30, 0.40, 0.55, 0.70, 0.85, 1.00),
        tracking_ratio_thresholds=(0.30, 0.35, 0.40, 0.45, 0.50, 0.55),
        edge_tracking_ratio_thresholds=(0.20, 0.25, 0.30, 0.35, 0.40, 0.45),
        lift_reward_name="lift_clearance",
        balance_reward_name="balance",
        yaw_reward_name="gated_yaw_tracking",
        torso_contact_termination_name="torso_contact",
        minimum_base_height=0.35,
        support_threshold=0.85,
        lift_progress_threshold=0.80,
        balance_threshold=0.75,
        yaw_threshold=0.65,
        min_evaluated_episodes=2048,
        required_success_rate=0.85,
        required_consecutive_windows=3,
        min_clearance_stage_steps=1000,
        min_yaw_stage_steps=6000,
    )

    assert state["yaw_limit"].item() == pytest.approx(1.0)
    assert state["online_dr_scale"].item() == pytest.approx(1.0)
    assert command_term.cfg.yaw_rate_range == (-1.0, 1.0)
    force_cfg = event_manager.get_term_cfg("randomize_apply_external_force_torque")
    assert force_cfg.params["force_range"] == (-10.0, 10.0)
    gain_cfg = event_manager.get_term_cfg("randomize_actuator_gains")
    assert gain_cfg.params["stiffness_distribution_params"] == pytest.approx((0.85, 1.15))
    assert reward_manager.get_term_cfg("lift_clearance").params["target_clearance"] == pytest.approx(0.20)


@pytest.mark.parametrize(
    ("success_rate", "tracking_ratio", "edge_tracking_ratio", "expected"),
    [
        (0.85, 0.30, 0.20, True),
        (0.84, 0.60, 0.60, False),
        (0.95, 0.29, 0.60, False),
        (0.95, 0.60, 0.19, False),
    ],
)
def test_yaw_window_requires_every_gate(
    monkeypatch: pytest.MonkeyPatch,
    success_rate: float,
    tracking_ratio: float,
    edge_tracking_ratio: float,
    expected: bool,
):
    curriculum = _load_curriculums_module(monkeypatch)

    assert curriculum.yaw_curriculum_window_passes(
        success_rate,
        tracking_ratio,
        edge_tracking_ratio,
        required_success_rate=0.85,
        tracking_ratio_threshold=0.30,
        edge_tracking_ratio_threshold=0.20,
    ) is expected
