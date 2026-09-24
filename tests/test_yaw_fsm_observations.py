"""CPU-only checks for the FSM critic's mirrored support geometry."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).parents[1]
OBSERVATIONS_PATH = (
    REPO_ROOT
    / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/observations.py"
)
FSM_CONFIG_PATH = (
    REPO_ROOT
    / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py"
)


def _quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., 1:]
    doubled_cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + quat[..., :1] * doubled_cross + torch.cross(xyz, doubled_cross, dim=-1)


def _load_geometry_functions():
    names = {
        "com_support_coordinate",
        "support_wheel_alignment",
        "_fsm_select_support_geometry",
        "fsm_com_support_coordinate",
        "fsm_support_wheel_alignment",
    }
    tree = ast.parse(OBSERVATIONS_PATH.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    namespace = {
        "torch": torch,
        "quat_apply": _quat_apply,
        "ManagerBasedRLEnv": object,
        "SceneEntityCfg": object,
        "Articulation": object,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), OBSERVATIONS_PATH, "exec"), namespace)
    return namespace


def _yaw_quat(angle_deg: float, dtype: torch.dtype) -> torch.Tensor:
    angle = math.radians(angle_deg) / 2.0
    return torch.tensor([math.cos(angle), 0.0, 0.0, math.sin(angle)], dtype=dtype)


def _make_env(diagonal: list[int], *, mirrored_com: bool = False):
    count = len(diagonal)
    dtype = torch.float64
    # Body order is FL, FR, HL, HR. Both support pairs are front -> hind.
    positions = torch.tensor(
        [[-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=dtype,
    ).unsqueeze(0).expand(count, -1, -1).clone()
    center = torch.tensor([[0.4, 0.2, 0.0]], dtype=dtype).expand(count, -1).clone()
    if mirrored_com:
        center[1, 1] = -center[1, 1]
    com_positions = center.unsqueeze(1).expand(-1, 4, -1).clone()
    wheel_quats = torch.stack(
        [_yaw_quat(0, dtype), _yaw_quat(60, dtype), _yaw_quat(-60, dtype), _yaw_quat(60, dtype)]
    ).unsqueeze(0).expand(count, -1, -1).clone()
    if mirrored_com:
        wheel_quats[:, 1] = _yaw_quat(0, dtype)
    root_quat = _yaw_quat(0, dtype).unsqueeze(0).expand(count, -1).clone()
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_pos_w=positions,
            body_com_pos_w=com_positions,
            body_quat_w=wheel_quats,
            root_quat_w=root_quat,
        ),
        root_physx_view=SimpleNamespace(get_masses=lambda: torch.ones(count, 4, dtype=dtype)),
        device="cpu",
    )
    command = SimpleNamespace(support_diagonal=torch.tensor(diagonal))
    env = SimpleNamespace(
        scene={"robot": robot},
        command_manager=SimpleNamespace(get_term=lambda name: command),
        device="cpu",
    )
    pos_cfg = SimpleNamespace(name="robot", body_ids=torch.tensor([0, 3]))
    neg_cfg = SimpleNamespace(name="robot", body_ids=torch.tensor([1, 2]))
    return env, pos_cfg, neg_cfg


def test_fsm_geometry_selects_pos_neg_and_neutral_without_shape_or_dtype_change():
    functions = _load_geometry_functions()
    env, pos_cfg, neg_cfg = _make_env([1, -1, 0])
    for legacy_name, fsm_name in (
        ("support_wheel_alignment", "fsm_support_wheel_alignment"),
        ("com_support_coordinate", "fsm_com_support_coordinate"),
    ):
        positive = functions[legacy_name](env, pos_cfg)
        negative = functions[legacy_name](env, neg_cfg)
        observed = functions[fsm_name](env, pos_cfg, neg_cfg, "yaw_rate_cmd")
        assert observed.shape == positive.shape == negative.shape == (3, 2)
        assert observed.dtype == positive.dtype == negative.dtype == torch.float64
        torch.testing.assert_close(observed[0], positive[0])
        torch.testing.assert_close(observed[1], negative[1])
        assert torch.equal(observed[2], torch.zeros_like(observed[2]))
        assert not torch.allclose(positive[1], negative[1])


def test_fsm_com_coordinate_uses_front_to_hind_order_for_both_diagonals():
    functions = _load_geometry_functions()
    env, pos_cfg, neg_cfg = _make_env([1, -1])
    robot = env.scene["robot"]
    robot.data.body_pos_w[:, 0, :2] = torch.tensor([0.0, 0.0])  # FL
    robot.data.body_pos_w[:, 3, :2] = torch.tensor([2.0, 0.0])  # HR
    robot.data.body_pos_w[:, 1, :2] = torch.tensor([0.0, 0.0])  # FR
    robot.data.body_pos_w[:, 2, :2] = torch.tensor([2.0, 0.0])  # HL
    robot.data.body_com_pos_w[:, :, :2] = torch.tensor([1.5, 0.25])
    observed = functions["fsm_com_support_coordinate"](env, pos_cfg, neg_cfg, "yaw_rate_cmd")
    expected = torch.tensor([[0.5, 0.25], [0.5, 0.25]], dtype=torch.float64)
    torch.testing.assert_close(observed, expected)


def test_fsm_geometry_respects_reflected_pose():
    functions = _load_geometry_functions()
    env, pos_cfg, neg_cfg = _make_env([1, -1], mirrored_com=True)
    alignment = functions["fsm_support_wheel_alignment"](env, pos_cfg, neg_cfg, "yaw_rate_cmd")
    coordinates = functions["fsm_com_support_coordinate"](env, pos_cfg, neg_cfg, "yaw_rate_cmd")
    torch.testing.assert_close(alignment[0], alignment[1])
    torch.testing.assert_close(coordinates[0, 0], coordinates[1, 0])
    torch.testing.assert_close(coordinates[0, 1], -coordinates[1, 1])


def test_only_fsm_critic_overrides_the_two_geometry_terms():
    tree = ast.parse(FSM_CONFIG_PATH.read_text(encoding="utf-8"))
    observations = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VQRWheelFSMObservationsCfg"
    )
    critic = next(
        node for node in observations.body
        if isinstance(node, ast.ClassDef) and node.name == "CriticCfg"
    )
    for name, function_name in (
        ("support_wheel_alignment", "fsm_support_wheel_alignment"),
        ("com_support_coordinate", "fsm_com_support_coordinate"),
    ):
        term = next(
            node.value for node in critic.body
            if isinstance(node, ast.Assign) and node.targets[0].id == name
        )
        assert isinstance(term, ast.Call) and term.func.id == "ObsTerm"
        assert next(keyword.value.attr for keyword in term.keywords if keyword.arg == "func") == function_name
        params = next(keyword.value for keyword in term.keywords if keyword.arg == "params")
        values = {key.value: value for key, value in zip(params.keys, params.values)}
        assert values["asset_cfg"].keywords[0].value.id == "SUPPORT_WHEEL_NAMES"
        assert values["asset_cfg_mirror"].keywords[0].value.id == "SUPPORT_WHEEL_NAMES_MIRROR"
        assert values["command_name"].value == "yaw_rate_cmd"
