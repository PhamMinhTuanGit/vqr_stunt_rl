"""Crouching, HipX abduction and high-yaw slip regressions on CPU."""

import ast
from types import SimpleNamespace as NS

import pytest
import torch

from test_yaw_pos_task import MDP, _env, _reward_module


class Scene(dict):
    pass


def height_env(commands, heights):
    env = _env(commands)
    env.scene = Scene(env.scene)
    # Separate local base height from world origin, including a translated env.
    env.scene.env_origins = torch.zeros(len(commands), 3, dtype=torch.float64)
    env.scene.env_origins[:, 2] = torch.arange(len(commands)) * 10.
    env.scene["robot"].data.root_pos_w = env.scene.env_origins.clone()
    env.scene["robot"].data.root_pos_w[:, 2] += torch.tensor(heights, dtype=torch.float64)
    return env


def test_crouching_penalty_starts_above_safety_floor_and_increases_at_low_yaw(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = height_env([0., 0.1, 0.25, 0.5, 1.], [0.40] * 5)
    cost = reward.yaw_pos_base_height_deficit_l2(env, "yaw_rate_cmd", 0.46)
    assert cost.tolist() == pytest.approx([2.88, 2.592, 2.16, 1.44, 1.44], abs=5e-5)
    assert env._yaw_pos_low_speed_base_height_samples.tolist() == [1, 1, 1, 0, 0]
    # Crossing the neutral/positive deadband does not turn the cost off.
    assert cost[0] > cost[1] > cost[2] > cost[3] > 0.
    assert cost[3].item() == pytest.approx(cost[4].item())


def test_height_cost_is_zero_above_floor_and_pushes_a_crouched_base_up(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = height_env([0.25] * 4, [0.40, 0.45, 0.49, 0.55])
    data = env.scene["robot"].data
    data.root_pos_w.requires_grad_()
    cost = reward.yaw_pos_base_height_deficit_l2(env, "yaw_rate_cmd", 0.46)
    assert cost[0] > cost[1] > 0.
    assert torch.count_nonzero(cost[2:]) == 0
    cost.sum().backward()
    assert torch.all(data.root_pos_w.grad[:2, 2] < 0.)
    assert torch.count_nonzero(data.root_pos_w.grad[2:]) == 0


def test_hipx_penalty_is_symmetric_in_both_modes_and_ignores_other_joints(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _env([0., 0.25, 1.])
    data = env.scene["robot"].data
    ids = [0, 3, 6, 9]
    data.joint_pos[0, ids] = 0.15
    data.joint_pos[1, ids] = -0.15
    data.joint_pos[2, [1, 2, 4, 5, 7, 8, 10, 11]] = 2.
    data.joint_pos.requires_grad_()
    cost = reward.yaw_pos_hipx_deviation_l2(env, NS(name="robot", joint_ids=ids))
    assert cost.tolist() == pytest.approx([1., 1., 0.])
    cost.sum().backward()
    assert torch.all(data.joint_pos.grad[0, ids] > 0.)
    assert torch.all(data.joint_pos.grad[1, ids] < 0.)
    assert torch.count_nonzero(data.joint_pos.grad[2]) == 0
    assert env._yaw_pos_hipx_abs_error_sum.tolist() == pytest.approx([0.15, 0.15, 0.])


def test_hipx_penalty_tracks_the_nominal_offset(monkeypatch):
    reward = _reward_module(monkeypatch)
    env = _env([0.25])
    data = env.scene["robot"].data
    data.default_joint_pos[0, 0] = 0.05
    data.joint_pos[0, 0] = 0.05
    assert reward.yaw_pos_hipx_deviation_l2(env, NS(name="robot", joint_ids=[0])).item() == 0.


def test_slip_cost_remains_active_at_high_yaw_and_ignores_airborne_wheels(monkeypatch):
    reward = _reward_module(monkeypatch)
    # Exercise the actual shared command-relief function, including its floor.
    tree = ast.parse((MDP / "rewards.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_yaw_command_penalty_scale")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[])), "command relief", "exec"), namespace)
    monkeypatch.setattr(reward, "_yaw_command_penalty_scale", namespace[function.name])
    env = _env([0., 0.25, 1., 2.])
    velocities = torch.zeros(4, 2, 2)
    velocities[:, 0, 0] = 0.2
    velocities[:, 1, 0] = 10.  # No contact, so this wheel must contribute nothing.
    contacts = torch.tensor([[True, False]]).repeat(4, 1)
    monkeypatch.setattr(reward, "contacted_wheel_velocities", lambda *args: (velocities, contacts))
    cfg = dict(sensor_cfg=None, body_asset_cfg=None, joint_asset_cfg=None, wheel_radius=0.091,
               command_name="yaw_rate_cmd", yaw_reference=1., deadband=0.1)
    cost = reward.yaw_pos_rolling_wheel_slip(env, **cfg, minimum_scale=0.25)
    assert cost.tolist() == pytest.approx([0.04, 0.03, 0.01, 0.01])
    # Legacy callers retain the previous behavior unless the POS config opts in.
    assert reward.yaw_pos_rolling_wheel_slip(env, **cfg).tolist() == pytest.approx([0.04, 0.03, 0., 0.])
    contacts.zero_()
    assert torch.count_nonzero(reward.yaw_pos_rolling_wheel_slip(env, **cfg, minimum_scale=0.25)) == 0


@pytest.mark.parametrize("params", [{"error_scale": 0.}, {"low_speed_yaw": 0.}, {"low_speed_multiplier": 0.5}])
def test_height_penalty_rejects_invalid_scales(monkeypatch, params):
    reward = _reward_module(monkeypatch)
    with pytest.raises(ValueError):
        reward.yaw_pos_base_height_deficit_l2(height_env([0.25], [0.49]), "yaw_rate_cmd", 0.46, **params)


def test_hipx_penalty_rejects_zero_scale(monkeypatch):
    reward = _reward_module(monkeypatch)
    with pytest.raises(ValueError, match="std"):
        reward.yaw_pos_hipx_deviation_l2(_env([0.25]), NS(name="robot", joint_ids=[0]), std=0.)
