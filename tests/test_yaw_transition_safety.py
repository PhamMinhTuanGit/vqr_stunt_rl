"""CPU checks for the transition height warning and support-gated lift credit."""

import ast
from collections.abc import Sequence
from types import SimpleNamespace

import torch

from test_yaw_fsm import REWARDS_PATH
from test_yaw_fsm_systemic import _reward_scene

TERMINATIONS_PATH = REWARDS_PATH.with_name("terminations.py")
OBSERVATIONS_PATH = REWARDS_PATH.with_name("observations.py")


def _transition_scene():
    rewards, state, env, command = _reward_scene()
    names = {"yaw_transition_low_base_height", "yaw_downward_low_base_velocity_l2"}
    nodes = [
        node for node in ast.parse(REWARDS_PATH.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(REWARDS_PATH), "exec"), rewards)
    env.scene["robot"].data.root_lin_vel_w = torch.zeros(2, 3)
    return rewards, state, env, command


def test_transition_height_warning_is_monotone_and_state_gated():
    rewards, state, env, command = _transition_scene()
    cfg = SimpleNamespace(name="robot")
    args = dict(warning_height=.43, minimum_height=.35, fsm_command_name="yaw_rate_cmd", asset_cfg=cfg)
    samples = []
    for height in (.44, .43, .41, .39, .36, .351):
        env.scene["robot"].data.root_pos_w[:, 2] = height
        samples.append(rewards["yaw_transition_low_base_height"](env, **args))
    assert torch.equal(samples[0], torch.zeros(2))
    assert torch.equal(samples[1], torch.zeros(2))
    assert all((later > earlier).all() for earlier, later in zip(samples[1:-1], samples[2:]))

    command.fsm_state[:] = torch.tensor([state.YAW_POS, state.FOUR_STAND])
    env.common_step_counter += 1
    assert torch.equal(rewards["yaw_transition_low_base_height"](env, **args), torch.zeros(2))


def test_transition_stability_does_not_track_standing_height():
    rewards, _, env, _ = _transition_scene()
    robot = env.scene["robot"].data
    robot.root_quat_w[:, 0] = .1  # The test double interprets this as roll.
    values = []
    for height in (.44, .49, .55):
        robot.root_pos_w[:, 2] = height
        values.append(rewards["four_stand_stability"](
            env, target_height=.49, fsm_command_name="yaw_rate_cmd"
        ))
    assert torch.allclose(values[0], values[1])
    assert torch.allclose(values[1], values[2])
    assert (values[0] < 0).all()


def test_downward_speed_is_penalized_before_hard_height_failure():
    rewards, state, env, _ = _transition_scene()
    cfg = SimpleNamespace(name="robot")
    env.scene["robot"].data.root_lin_vel_w[:, 2] = -.5
    args = dict(minimum_height=.35, height_margin=.10, warning_height=.43,
                fsm_command_name="yaw_rate_cmd", asset_cfg=cfg)
    env.scene["robot"].data.root_pos_w[:, 2] = .43
    assert torch.equal(rewards["yaw_downward_low_base_velocity_l2"](env, **args), torch.zeros(2))
    env.scene["robot"].data.root_pos_w[:, 2] = .39
    middle = rewards["yaw_downward_low_base_velocity_l2"](env, **args)
    env.scene["robot"].data.root_pos_w[:, 2] = .351
    near_boundary = rewards["yaw_downward_low_base_velocity_l2"](env, **args)
    assert torch.allclose(middle, torch.full((2,), .125), atol=1e-6)
    assert (near_boundary > middle).all()
    command = env.command_manager.get_term("yaw_rate_cmd")
    command.fsm_state[:] = torch.tensor([state.YAW_POS, state.FOUR_STAND])
    env.common_step_counter += 1
    assert torch.equal(rewards["yaw_downward_low_base_velocity_l2"](env, **args), torch.zeros(2))


def test_lift_credit_uses_selected_support_diagonal_and_keeps_exploration():
    rewards, _, env, _ = _transition_scene()
    forces = env.scene.sensors["contact_forces"].data.net_forces_w
    # POS is FL/HR and NEG is FR/HL. Swap which pair is loaded in each env.
    forces.zero_()
    forces[0, [1, 2], 2] = 80.0
    forces[1, [0, 3], 2] = 80.0
    for progress in env.lift_progress.values():
        progress.fill_(1.0)
    support = dict(
        support_sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=[0, 3]),
        support_sensor_cfg_mirror=SimpleNamespace(name="contact_forces", body_ids=[1, 2]),
    )
    args = dict(asset_cfg=SimpleNamespace(name="pos"), asset_cfg_mirror=SimpleNamespace(name="neg"),
                wheel_radius=.091, target_clearance=.05, fsm_command_name="yaw_rate_cmd", **support)
    clearance = rewards["yaw_lift_clearance"](env, **args, transition_ungated_fraction=.25)
    assert torch.allclose(clearance, torch.full((2,), -.75))

    progress_term = rewards["TransitionProgress"](None, env)
    credit = progress_term(env, **args, transition_ungated_fraction=.25)
    assert torch.allclose(credit, torch.full((2,), .2475))
    progress_term.reset()
    forces.zero_()
    forces[0, [0, 3], 2] = 80.0
    forces[1, [1, 2], 2] = 80.0
    env.common_step_counter += 1
    assert torch.equal(rewards["yaw_lift_clearance"](env, **args), torch.zeros(2))
    full_credit = progress_term(env, **args, transition_ungated_fraction=.25)
    assert torch.allclose(full_credit, torch.full((2,), .99))


def test_height_and_tilt_failure_are_separate_termination_masks():
    node = next(
        node for node in ast.parse(TERMINATIONS_PATH.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "FSMUnsafeWithGrace"
    )
    class ManagerTermBase:
        def __init__(self, cfg, env):
            pass

    masks = (torch.tensor([False, False]), torch.tensor([True, False]),
             torch.tensor([False, True]))
    namespace = dict(torch=torch, ManagerTermBase=ManagerTermBase,
                     ManagerBasedRLEnv=object, SceneEntityCfg=object, Sequence=Sequence,
                     yaw_fsm_unsafe_components=lambda *args, **kwargs: masks)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(TERMINATIONS_PATH), "exec"), namespace)
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=.05)
    args = dict(robot_name="robot", torso_sensor_cfg=object(), threshold=1.0,
                grace_period_s=0.0, minimum_base_height=.35, unsafe_angle_limit=.8)
    term = namespace["FSMUnsafeWithGrace"](None, env)
    assert torch.equal(term(env, **args, failure_kind="height"), masks[1])
    assert torch.equal(term(env, **args, failure_kind="tilt"), masks[2])
    assert torch.equal(term(env, **args), torch.tensor([True, True]))


def test_unsafe_components_keep_height_and_tilt_thresholds_distinct():
    names = {"yaw_fsm_unsafe", "yaw_fsm_unsafe_components"}
    nodes = [
        node for node in ast.parse(OBSERVATIONS_PATH.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = dict(
        torch=torch, ManagerBasedEnv=object, SceneEntityCfg=object, Articulation=object,
        wheel_contact=lambda *args, **kwargs: torch.zeros(3, 1),
        euler_xyz_from_quat=lambda q: (q[:, 0], q[:, 1], q[:, 2]),
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(OBSERVATIONS_PATH), "exec"), namespace)
    class Scene(dict):
        env_origins = torch.zeros(3, 3)

    env = SimpleNamespace(scene=Scene(robot=SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.tensor([[0., 0., .349], [0., 0., .43], [0., 0., .35]]),
            root_quat_w=torch.tensor([[0., 0., 0., 0.], [.81, 0., 0., 0.], [0., 0., 0., 0.]]),
        ))))
    args = dict(robot_name="robot", torso_sensor_cfg=object(),
                minimum_base_height=.35, unsafe_angle_limit=.8)
    torso, height, tilt = namespace["yaw_fsm_unsafe_components"](env, **args)
    assert torch.equal(torso, torch.zeros(3, dtype=torch.bool))
    assert torch.equal(height, torch.tensor([True, False, False]))
    assert torch.equal(tilt, torch.tensor([False, True, False]))
    assert torch.equal(namespace["yaw_fsm_unsafe"](env, **args), height | tilt)
