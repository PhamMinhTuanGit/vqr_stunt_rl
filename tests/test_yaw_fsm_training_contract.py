"""Static CPU-only checks for the yaw-FSM training contract."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).parents[1]
CONFIG_DIR = (
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
)
BASELINE_CONFIG = CONFIG_DIR / "yaw_env_cfg.py"
FSM_CONFIG = CONFIG_DIR / "yaw_env_fsm_cfg.py"
RUNNER_CONFIG = CONFIG_DIR / "agents" / "rsl_rl_ppo_cfg.py"
REGISTRATION = CONFIG_DIR / "__init__.py"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"


def _class_node(path: Path, name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
    )


def _reward_assignments(path: Path, class_name: str) -> dict[str, ast.Call]:
    node = _class_node(path, class_name)
    return {
        item.targets[0].id: item.value
        for item in node.body
        if isinstance(item, ast.Assign)
        and isinstance(item.targets[0], ast.Name)
        and isinstance(item.value, ast.Call)
        and isinstance(item.value.func, ast.Name)
        and item.value.func.id == "RewTerm"
    }


def _keyword(call: ast.Call, name: str) -> ast.expr:
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


def test_fsm_balance_is_gated_and_reward_counts_are_isolated():
    baseline = _reward_assignments(BASELINE_CONFIG, "VQRWheelRewardsCfg")
    fsm = _reward_assignments(FSM_CONFIG, "VQRWheelFSMRewardsCfg")

    assert len(baseline) == 18
    assert len(fsm) == 24
    assert "support_contact" not in fsm
    assert "return_to_four_landing" in fsm
    assert "four_stand_ready_bonus" in fsm

    fsm_balance_params = ast.literal_eval(_keyword(fsm["balance"], "params"))
    baseline_balance_params = ast.literal_eval(_keyword(baseline["balance"], "params"))
    assert fsm_balance_params["fsm_command_name"] == "yaw_rate_cmd"
    assert "fsm_command_name" not in baseline_balance_params


def test_fsm_registration_uses_its_own_runner_and_experiment_directory():
    runner = _class_node(RUNNER_CONFIG, "VQRWheelYawFlatFSMPPORunnerCfg")
    assert [base.id for base in runner.bases if isinstance(base, ast.Name)] == [
        "VQRWheelYawFlatPPORunnerCfg"
    ]
    post_init = next(
        item for item in runner.body if isinstance(item, ast.FunctionDef) and item.name == "__post_init__"
    )
    experiment_assignment = next(
        item
        for item in post_init.body
        if isinstance(item, ast.Assign)
        and isinstance(item.targets[0], ast.Attribute)
        and item.targets[0].attr == "experiment_name"
    )
    assert ast.literal_eval(experiment_assignment.value) == "vqr_wheel_yaw_flat_fsm"

    source = REGISTRATION.read_text(encoding="utf-8")
    tree = ast.parse(source)
    registrations: dict[str, dict[str, str]] = {}
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not isinstance(call.func, ast.Attribute) or call.func.attr != "register":
            continue
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        task_id = ast.literal_eval(keywords["id"])
        kwargs = keywords["kwargs"]
        registrations[task_id] = {
            ast.literal_eval(key): ast.unparse(value)
            for key, value in zip(kwargs.keys, kwargs.values)
        }

    assert "VQRWheelYawFlatFSMPPORunnerCfg" in registrations[
        "Flat-VQR-Wheel-Yaw-FSM"
    ]["rsl_rl_cfg_entry_point"]
    assert "VQRWheelYawFlatPPORunnerCfg" in registrations["Flat-VQR-Wheel-Yaw"][
        "rsl_rl_cfg_entry_point"
    ]
    assert "FSMPPORunnerCfg" not in registrations["Flat-VQR-Wheel-Yaw"][
        "rsl_rl_cfg_entry_point"
    ]


def test_fsm_startup_contract_checks_reward_and_command_runtime_objects():
    class YawFSMCommand:
        pass

    tree = ast.parse(TRAIN_SCRIPT.read_text(encoding="utf-8"))
    verifier = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_verify_yaw_fsm_contract"
    )
    namespace = {"inspect": inspect, "YawFSMCommand": YawFSMCommand}
    exec(compile(ast.Module(body=[verifier], type_ignores=[]), TRAIN_SCRIPT, "exec"), namespace)

    command = YawFSMCommand()
    for name in (
        "fsm_state",
        "support_diagonal",
        "state_time",
        "just_switched",
        "just_returned_to_four",
        "positive_pose_ready",
        "negative_pose_ready",
        "four_stand_ready",
        "unsafe",
        "yaw_entry_pos",
    ):
        setattr(command, name, object())
    required = {
        "fsm_gated_tracking",
        "transition_progress",
        "return_to_four_landing",
        "four_stand_ready_bonus",
        "spin_center_drift",
        "safe_recovery_entry",
    }
    active_terms = sorted(required | {f"term_{index}" for index in range(18)})
    reward_manager = SimpleNamespace(
        active_terms=active_terms,
        get_term_cfg=lambda _: SimpleNamespace(weight=8.0),
    )
    task_env = SimpleNamespace(
        reward_manager=reward_manager,
        command_manager=SimpleNamespace(get_term=lambda _: command),
    )
    env = SimpleNamespace(unwrapped=task_env)

    namespace["_verify_yaw_fsm_contract"](env, SimpleNamespace())

    reward_manager.active_terms = active_terms[:-1]
    with pytest.raises(RuntimeError, match="expected 24 terms"):
        namespace["_verify_yaw_fsm_contract"](env, SimpleNamespace())
