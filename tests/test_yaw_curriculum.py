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
TRAIN_PATH = REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"


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


def test_fsm_curriculum_requires_both_diagonals_before_phase_promotion(
    monkeypatch: pytest.MonkeyPatch,
):
    """A perfect POS lift score must not certify a failed NEG diagonal."""
    curriculum = _load_curriculums_module(monkeypatch)

    class ConfigManager:
        def __init__(self, configs):
            self.configs = configs

        def get_term_cfg(self, name):
            return self.configs[name]

        def set_term_cfg(self, name, cfg):
            self.configs[name] = cfg

    def reward_cfg(weight=1.0):
        return SimpleNamespace(weight=weight, params={"target_clearance": 0.05})

    command = SimpleNamespace(cfg=SimpleNamespace(yaw_rate_range=(-0.25, 0.25), target_clearance=0.05))
    rewards = ConfigManager(
        {
            "lift_clearance": reward_cfg(3.0),
            "fsm_gated_tracking": reward_cfg(8.0),
            "transition_progress": reward_cfg(2.0),
            "spin_center_drift": reward_cfg(0.0),
            "safe_recovery_entry": reward_cfg(0.0),
        }
    )
    events = ConfigManager(
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
            "randomize_reset_base": SimpleNamespace(params={"pose_range": {}, "velocity_range": {}}),
        }
    )
    env = SimpleNamespace(
        common_step_counter=1,
        num_envs=2,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.ones(2, dtype=torch.long),
        command_manager=SimpleNamespace(get_term=lambda _: command),
        reward_manager=rewards,
        event_manager=events,
        # Both branches entered YAW, but only POS achieved lift/support.
        _yaw_fsm_pos_yaw_samples=torch.tensor([1, 0]),
        _yaw_fsm_neg_yaw_samples=torch.tensor([0, 1]),
        _yaw_fsm_pos_lift_sum=torch.tensor([1.0, 0.0]),
        _yaw_fsm_neg_lift_sum=torch.tensor([0.0, 0.0]),
        _yaw_fsm_pos_support_sum=torch.tensor([1.0, 0.0]),
        _yaw_fsm_neg_support_sum=torch.tensor([0.0, 1.0]),
        _yaw_fsm_state_steps=torch.tensor(
            [[2, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0]], dtype=torch.long
        ),
        _yaw_fsm_switches=torch.tensor([1, 2], dtype=torch.long),
        _yaw_fsm_episode_steps=torch.tensor([3, 2], dtype=torch.long),
    )
    params = dict(
        command_name="yaw_rate_cmd",
        clearance_levels=(0.05,),
        yaw_rate_levels=(0.25,),
        dr_scale_levels=(0.30,),
        tracking_ratio_thresholds=(0.30,),
        edge_tracking_ratio_thresholds=(0.20,),
        lift_reward_name="lift_clearance",
        balance_reward_name="balance",
        yaw_reward_name="fsm_gated_tracking",
        torso_contact_termination_name="torso_contact",
        minimum_base_height=0.35,
        support_threshold=0.85,
        lift_progress_threshold=0.80,
        balance_threshold=0.75,
        yaw_threshold=0.65,
        min_evaluated_episodes=2,
        required_success_rate=0.85,
        required_consecutive_windows=1,
        min_clearance_stage_steps=0,
        min_yaw_stage_steps=0,
        transition_reward_name="transition_progress",
        min_directional_episodes=1,
        reward_ramp_steps=0,
    )

    state = curriculum.yaw_fsm_task_levels(env, torch.tensor([0, 1]), **params)

    assert state["phase"].item() == 0.0
    assert state["pos/score"].item() == pytest.approx(1.0)
    assert state["neg/score"].item() == pytest.approx(0.0)
    assert state["state_fraction/0"].item() == pytest.approx(3.0 / 5.0)
    assert state["state_fraction/1"].item() == pytest.approx(1.0 / 5.0)
    assert state["state_fraction/4"].item() == pytest.approx(1.0 / 5.0)
    assert state["switch_rate"].item() == pytest.approx(3.0 / 5.0)
    assert torch.equal(env._yaw_fsm_episode_steps, torch.zeros(2, dtype=torch.long))


def _load_train_curriculum_helpers():
    """Load checkpoint helpers without importing train.py and launching Isaac Sim."""
    tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"))
    names = {
        "_YAW_CURRICULUM_PREFIX",
        "_YAW_CURRICULUM_CHECKPOINT_KEY",
        "_YAW_CURRICULUM_PERSISTENT_FIELDS",
        "_YAW_FSM_CURRICULUM_CHECKPOINT_KEY",
        "_YAW_FSM_CURRICULUM_PERSISTENT_FIELDS",
        "_export_yaw_curriculum_state",
        "_restore_yaw_curriculum_state",
        "_export_yaw_fsm_curriculum_state",
        "_restore_yaw_fsm_curriculum_state",
    }
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(target, ast.Name) and target.id in names:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), TRAIN_PATH, "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


class _ConfigManager:
    def __init__(self, configs):
        self.configs = configs

    def get_term_cfg(self, name):
        return self.configs[name]

    def set_term_cfg(self, name, cfg):
        self.configs[name] = cfg


class _ResettableCommand:
    def __init__(self):
        self.cfg = SimpleNamespace(yaw_rate_range=(-0.25, 0.25), target_clearance=0.05)
        self.reset_env_ids = None

    def reset(self, env_ids):
        self.reset_env_ids = env_ids.clone()


def _online_dr_manager():
    return _ConfigManager(
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


def _fsm_curriculum_params():
    return {
        "command_name": "yaw_rate_cmd",
        "clearance_levels": (0.05, 0.10, 0.15, 0.20),
        "yaw_rate_levels": (0.25, 0.40, 0.55),
        "dr_scale_levels": (0.30, 0.40, 0.55),
        "tracking_ratio_thresholds": (0.30, 0.35, 0.40),
        "edge_tracking_ratio_thresholds": (0.20, 0.25, 0.30),
        "lift_reward_name": "lift_clearance",
        "balance_reward_name": "balance",
        "yaw_reward_name": "fsm_gated_tracking",
        "torso_contact_termination_name": "torso_contact",
        "minimum_base_height": 0.35,
        "support_threshold": 0.85,
        "lift_progress_threshold": 0.80,
        "balance_threshold": 0.75,
        "yaw_threshold": 0.65,
        "min_evaluated_episodes": 8,
        "required_success_rate": 0.85,
        "required_consecutive_windows": 3,
        "min_clearance_stage_steps": 1000,
        "min_yaw_stage_steps": 6000,
        "transition_reward_name": "transition_progress",
        "spin_center_drift_reward_name": "spin_center_drift",
        "safe_recovery_reward_name": "safe_recovery_entry",
        "min_directional_episodes": 4,
        "transition_success_threshold": 0.85,
        "drift_thresholds": (0.08, 0.08, 0.08),
        "reward_ramp_steps": 3600,
    }


def _fsm_checkpoint_env(curriculum, common_step_counter: int):
    command = _ResettableCommand()
    rewards = _ConfigManager(
        {
            "lift_clearance": SimpleNamespace(weight=3.0, params={"target_clearance": 0.05}),
            "fsm_gated_tracking": SimpleNamespace(
                weight=8.0,
                params={
                    "target_clearance": 0.05,
                    "clearance_gate_floor": 0.0,
                    "clearance_gate_floor_decay_s": 0.0,
                },
            ),
            "transition_progress": SimpleNamespace(
                weight=2.0, params={"target_clearance": 0.05}
            ),
            "spin_center_drift": SimpleNamespace(weight=0.0, params={}),
            "safe_recovery_entry": SimpleNamespace(weight=0.0, params={}),
        }
    )
    params = _fsm_curriculum_params()
    term_cfg = SimpleNamespace(func=curriculum.yaw_fsm_task_levels, params=params)
    return SimpleNamespace(
        common_step_counter=common_step_counter,
        num_envs=3,
        device="cpu",
        step_dt=0.02,
        episode_length_buf=torch.zeros(3, dtype=torch.long),
        command_manager=SimpleNamespace(get_term=lambda _: command),
        reward_manager=rewards,
        event_manager=_online_dr_manager(),
        cfg=SimpleNamespace(curriculum=SimpleNamespace(task_levels=term_cfg)),
    )


def test_fsm_checkpoint_restores_phase_c_timing_and_reapplies_config(
    monkeypatch: pytest.MonkeyPatch,
):
    curriculum = _load_curriculums_module(monkeypatch)
    helpers = _load_train_curriculum_helpers()
    saved_env = _fsm_checkpoint_env(curriculum, common_step_counter=10_000)
    saved_values = {
        "_yaw_fsm_task_curriculum_phase": 2,
        "_yaw_task_curriculum_stage": 3,
        "_yaw_task_curriculum_yaw_stage": 2,
        "_yaw_task_curriculum_consecutive_passes": 2,
        "_yaw_fsm_task_curriculum_stage_start_step": 9_000,
        "_yaw_fsm_task_curriculum_reward_ramp_start_step": 9_400,
        "_yaw_fsm_task_curriculum_ramp_start_fsm_gated_tracking": 8.0,
        "_yaw_fsm_task_curriculum_ramp_start_transition_progress": 2.0,
        "_yaw_fsm_task_curriculum_ramp_start_spin_center_drift": -0.5,
        "_yaw_fsm_task_curriculum_ramp_start_safe_recovery_entry": 0.0,
    }
    for name, value in saved_values.items():
        setattr(saved_env, name, value)

    payload = helpers._export_yaw_fsm_curriculum_state(saved_env)
    assert payload["stage_elapsed_steps"] == 1_000
    assert payload["reward_ramp_elapsed_steps"] == 600
    assert not any(isinstance(value, torch.Tensor) for value in payload["values"].values())

    resumed_env = _fsm_checkpoint_env(curriculum, common_step_counter=200)
    restored = helpers._restore_yaw_fsm_curriculum_state(
        resumed_env, {"yaw_fsm_curriculum": payload}
    )

    assert restored is True
    assert resumed_env._yaw_fsm_task_curriculum_phase == 2
    assert resumed_env._yaw_task_curriculum_stage == 3
    assert resumed_env._yaw_task_curriculum_yaw_stage == 2
    assert resumed_env._yaw_task_curriculum_consecutive_passes == 2
    assert resumed_env._yaw_fsm_task_curriculum_stage_start_step == -800
    assert resumed_env._yaw_fsm_task_curriculum_reward_ramp_start_step == -400

    command = resumed_env.command_manager.get_term("yaw_rate_cmd")
    assert command.cfg.yaw_rate_range == pytest.approx((-0.55, 0.55))
    assert command.cfg.target_clearance == pytest.approx(0.20)
    assert torch.equal(command.reset_env_ids, torch.arange(3))
    for reward_name in ("lift_clearance", "fsm_gated_tracking", "transition_progress"):
        assert resumed_env.reward_manager.get_term_cfg(reward_name).params[
            "target_clearance"
        ] == pytest.approx(0.20)

    assert resumed_env.reward_manager.get_term_cfg("fsm_gated_tracking").weight == pytest.approx(8.0)
    assert resumed_env.reward_manager.get_term_cfg("transition_progress").weight == pytest.approx(2.0)
    assert resumed_env.reward_manager.get_term_cfg("spin_center_drift").weight == pytest.approx(-0.75)
    assert resumed_env.reward_manager.get_term_cfg("safe_recovery_entry").weight == pytest.approx(
        -1.0 / 3.0
    )
    tracking_params = resumed_env.reward_manager.get_term_cfg("fsm_gated_tracking").params
    assert tracking_params["clearance_gate_floor"] == 0.0
    assert tracking_params["clearance_gate_floor_decay_s"] == 0.0

    force_cfg = resumed_env.event_manager.get_term_cfg("randomize_apply_external_force_torque")
    assert force_cfg.params["force_range"] == pytest.approx((-5.5, 5.5))
    gain_cfg = resumed_env.event_manager.get_term_cfg("randomize_actuator_gains")
    assert gain_cfg.params["stiffness_distribution_params"] == pytest.approx((0.9175, 1.0825))
    push_cfg = resumed_env.event_manager.get_term_cfg("randomize_push_robot")
    assert push_cfg.params["velocity_range"]["x"] == pytest.approx((-0.275, 0.275))


def test_baseline_checkpoint_format_and_restore_remain_backward_compatible(
    monkeypatch: pytest.MonkeyPatch,
):
    curriculum = _load_curriculums_module(monkeypatch)
    helpers = _load_train_curriculum_helpers()
    command = _ResettableCommand()
    rewards = _ConfigManager(
        {
            "lift_clearance": SimpleNamespace(weight=3.0, params={"target_clearance": 0.05}),
            "gated_yaw_tracking": SimpleNamespace(
                weight=8.0, params={"target_clearance": 0.05}
            ),
        }
    )
    params = {
        "command_name": "yaw_rate_cmd",
        "clearance_levels": (0.05, 0.10, 0.15, 0.20),
        "yaw_rate_levels": (0.25, 0.40, 0.55),
        "dr_scale_levels": (0.30, 0.40, 0.55),
        "tracking_ratio_thresholds": (0.30, 0.35, 0.40),
        "edge_tracking_ratio_thresholds": (0.20, 0.25, 0.30),
        "lift_reward_name": "lift_clearance",
        "balance_reward_name": "balance",
        "yaw_reward_name": "gated_yaw_tracking",
        "torso_contact_termination_name": "torso_contact",
        "minimum_base_height": 0.35,
        "support_threshold": 0.85,
        "lift_progress_threshold": 0.80,
        "balance_threshold": 0.75,
        "yaw_threshold": 0.65,
        "min_evaluated_episodes": 8,
        "required_success_rate": 0.85,
        "required_consecutive_windows": 3,
        "min_clearance_stage_steps": 1000,
        "min_yaw_stage_steps": 6000,
    }
    term_cfg = SimpleNamespace(func=curriculum.yaw_task_levels, params=params)

    def make_env(step):
        return SimpleNamespace(
            common_step_counter=step,
            num_envs=2,
            device="cpu",
            step_dt=0.02,
            episode_length_buf=torch.zeros(2, dtype=torch.long),
            command_manager=SimpleNamespace(get_term=lambda _: command),
            reward_manager=rewards,
            event_manager=_online_dr_manager(),
            cfg=SimpleNamespace(curriculum=SimpleNamespace(task_levels=term_cfg)),
        )

    saved_env = make_env(500)
    saved_env._yaw_task_curriculum_stage = 2
    saved_env._yaw_task_curriculum_yaw_stage = 1
    saved_env._yaw_task_curriculum_consecutive_passes = 2
    saved_env._yaw_task_curriculum_stage_start_step = 300
    payload = helpers._export_yaw_curriculum_state(saved_env)
    assert set(payload) == {
        "version",
        "values",
        "stage_elapsed_steps",
        "yaw_rate_levels",
        "dr_scale_levels",
    }

    resumed_env = make_env(50)
    assert helpers._restore_yaw_curriculum_state(
        resumed_env, {"yaw_curriculum": payload}
    )
    assert resumed_env._yaw_task_curriculum_stage == 2
    assert resumed_env._yaw_task_curriculum_yaw_stage == 1
    assert resumed_env._yaw_task_curriculum_consecutive_passes == 2
    assert resumed_env._yaw_task_curriculum_stage_start_step == -150
    assert command.cfg.yaw_rate_range == pytest.approx((-0.40, 0.40))
    assert torch.equal(command.reset_env_ids, torch.arange(2))
