"""Static startup and registration checks for the independent staged task."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel"
MDP = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp"
TRAIN = ROOT / "scripts/reinforcement_learning/rsl_rl/train.py"


def _class(path, name):
    return next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name == name)


def test_registration_and_runner_are_isolated():
    registry = ast.parse((CONFIG / "__init__.py").read_text())
    registrations = {}
    for node in ast.walk(registry):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "register":
            arguments = {item.arg: item.value for item in node.keywords}
            registrations[ast.literal_eval(arguments["id"])] = ast.unparse(arguments["kwargs"])
    assert "Flat-VQR-Wheel-Yaw-FSM-Staged" in registrations
    assert "yaw_env_fsm_staged_cfg:VQRWheelFlatEnvFSMStagedCfg" in registrations["Flat-VQR-Wheel-Yaw-FSM-Staged"]
    assert "VQRWheelYawFlatFSMStagedPPORunnerCfg" in registrations["Flat-VQR-Wheel-Yaw-FSM-Staged"]
    assert "yaw_env_fsm_cfg:VQRWheelFlatEnvFSMCfg" in registrations["Flat-VQR-Wheel-Yaw-FSM"]
    assert "VQRWheelYawFlatFSMPPORunnerCfg" in registrations["Flat-VQR-Wheel-Yaw-FSM"]
    runner = _class(CONFIG / "agents/rsl_rl_ppo_cfg.py", "VQRWheelYawFlatFSMStagedPPORunnerCfg")
    assert "vqr_wheel_yaw_flat_fsm_staged" in ast.unparse(runner)


def test_config_inherits_actor_rewards_and_nominal_rough_pose():
    staged = CONFIG / "yaw_env_fsm_staged_cfg.py"
    source = staged.read_text()
    node = _class(staged, "VQRWheelFlatEnvFSMStagedCfg")
    assert [base.id for base in node.bases] == ["VQRWheelFlatEnvFSMCfg"]
    assert "self.events.randomize_reset_joints.params[\"position_range\"] = (1.0, 1.0)" in source
    assert "self.events.randomize_reset_joints.params[\"velocity_range\"] = (0.0, 0.0)" in source
    assert '"roll": (0.0, 0.0), "pitch": (0.0, 0.0)' in source
    assert "axis: (0.0, 0.0)" in source
    assert "CLEARANCE_LEVELS = (0.02, 0.03, 0.05)" in source
    assert "YAW_RATE_LEVELS = (0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00)" in source
    assert "hold_window\": 2048" in source and "directional_window\": 1024" in source
    asset = (ROOT / "source/rl_training/rl_training/assets/deeprobotics.py").read_text()
    assert "pos=(0.0, 0.0, 0.45)" in asset
    for joint, value in (("HipX", "0.0"), ("HipY", "-0.65"), ("Knee", "1.3")):
        assert f'".*_{joint}_joint": {value}' in asset
    assert 'joint_vel={".*": 0.0}' in asset
    # The staged config inherits the FSM observation group unchanged. Its
    # actor contains the command and FSM state; contact/readiness stay critic-only.
    observations = _class(CONFIG / "yaw_env_fsm_cfg.py", "VQRWheelFSMObservationsCfg")
    policy = next(item for item in observations.body
                  if isinstance(item, ast.ClassDef) and item.name == "PolicyCfg")
    assert [item.targets[0].id for item in policy.body if isinstance(item, ast.Assign)] == ["fsm_state"]
    assert "observations:" not in source


def test_training_has_distinct_startup_and_checkpoint_state():
    tree = ast.parse(TRAIN.read_text())
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {
        "_verify_staged_yaw_contract", "_export_staged_yaw_state",
        "_restore_staged_yaw_state", "_install_staged_yaw_checkpointing",
    } <= names
    source = TRAIN.read_text()
    assert '"yaw_fsm_staged_curriculum"' in source
    assert 'task_name == "Flat-VQR-Wheel-Yaw-FSM-Staged"' in source
    assert "_verify_staged_yaw_contract(env, env_cfg)" in source
    assert "_install_staged_yaw_checkpointing(runner, staged_yaw_task_env)" in source
    mdp_source = (MDP / "__init__.py").read_text()
    assert "from .staged_yaw_curriculum import *" in mdp_source


def test_checkpoint_round_trip_preserves_partial_directional_window():
    spec = importlib.util.spec_from_file_location("staged_curriculum_contract", MDP / "staged_yaw_curriculum.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = ast.parse(TRAIN.read_text())
    names = {"_export_staged_yaw_state", "_restore_staged_yaw_state", "_install_staged_yaw_checkpointing"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"StagedPromotion": module.StagedPromotion, "torch": torch,
                 "_STAGED_YAW_CHECKPOINT_KEY": "yaw_fsm_staged_curriculum", "OnPolicyRunner": object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAIN), "exec"), namespace)

    params = {
        "command_name": "yaw_rate_cmd", "clearance_levels": (.02, .03, .05),
        "yaw_rate_levels": (.15, .25), "dr_scale_levels": (.30, 1.0),
        "tracking_ratio_thresholds": (.30, .35),
        "hold_window": 2048, "directional_window": 1024, "required_windows": 3,
    }
    promotion = module.StagedPromotion(phase=2, clearance_index=1)
    promotion.consecutive_passes = 2
    promotion.record("pos", yaw=1, support=1, pose=1)
    promotion.record("neg", yaw=0, support=1, pose=0)
    reapplied = []
    resets = []
    term_cfg = SimpleNamespace(params=params, func=lambda env, ids, **kwargs: reapplied.append(ids))
    command = SimpleNamespace(reset=lambda ids: resets.append(ids.tolist()))
    env = SimpleNamespace(
        cfg=SimpleNamespace(curriculum=SimpleNamespace(task_levels=term_cfg)),
        _staged_yaw_promotion=promotion, num_envs=2, device="cpu",
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )
    payload = namespace["_export_staged_yaw_state"](env)
    env._staged_yaw_promotion = module.StagedPromotion()
    assert namespace["_restore_staged_yaw_state"](env, {"yaw_fsm_staged_curriculum": payload})
    assert env._staged_yaw_promotion.export() == promotion.export()
    assert reapplied == [[]]
    assert resets == [[0, 1]]

    saved = []
    runner = SimpleNamespace(save=lambda path, infos: saved.append((path, infos)))
    namespace["_install_staged_yaw_checkpointing"](runner, env)
    runner.save("checkpoint.pt", {"other": 1})
    assert saved[0][1]["yaw_fsm_staged_curriculum"]["promotion"] == promotion.export()
    assert saved[0][1]["other"] == 1
