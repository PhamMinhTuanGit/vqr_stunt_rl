"""CPU-only ownership tests for the yaw-FSM position anchor."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).parents[1]
FSM_PATH = (
    REPO_ROOT
    / "source"
    / "rl_training"
    / "rl_training"
    / "tasks"
    / "manager_based"
    / "locomotion"
    / "velocity"
    / "mdp"
    / "fsm.py"
)


def _load_fsm_module():
    """Load the FSM module through its Isaac-Lab-independent fallback."""
    spec = importlib.util.spec_from_file_location("yaw_fsm_under_test", FSM_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_command_owns_yaw_entry_position_and_reward_has_no_private_anchor():
    fsm_module = _load_fsm_module()
    vector_fsm = fsm_module.YawFSMVectorized(num_envs=2)

    # Build only the command state consumed by _step_fsm; no Isaac simulator is
    # needed to validate the buffer ownership and transition timing.
    command = object.__new__(fsm_module.YawFSMCommand)
    command._fsm = vector_fsm
    command._fsm_state = vector_fsm.fsm_state
    command.support_diagonal = vector_fsm.support_diagonal
    command.state_time = vector_fsm.state_time
    command.just_switched = vector_fsm.just_switched
    command.yaw_entry_pos = torch.zeros(2, 2)
    command._command = torch.tensor([[0.20], [0.20]])
    command.positive_pose_ready = torch.zeros(2, dtype=torch.bool)
    command.negative_pose_ready = torch.zeros(2, dtype=torch.bool)
    command.four_stand_ready = torch.zeros(2, dtype=torch.bool)
    command.unsafe = torch.zeros(2, dtype=torch.bool)
    command._fsm_predicates_enabled = False
    command._env = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(2, 3)))
    command.robot = SimpleNamespace(
        data=SimpleNamespace(root_pos_w=torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]))
    )

    # FOUR_STAND -> TRANSITION must not record an anchor yet.
    command._step_fsm()
    assert torch.equal(command.yaw_entry_pos, torch.zeros(2, 2))

    # Record precisely on TRANSITION -> YAW, and preserve the entry point
    # while the command remains in YAW.
    command.positive_pose_ready.fill_(True)
    command.robot.data.root_pos_w[:, :2] = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    command._step_fsm()
    assert torch.equal(command.yaw_entry_pos, torch.tensor([[5.0, 6.0], [7.0, 8.0]]))

    command.robot.data.root_pos_w[:, :2] = torch.tensor([[9.0, 10.0], [11.0, 12.0]])
    command._step_fsm()
    assert torch.equal(command.yaw_entry_pos, torch.tensor([[5.0, 6.0], [7.0, 8.0]]))

    # The helper contains only transition state; resetting a command clears its
    # own anchor rather than a second hidden buffer.
    assert not hasattr(vector_fsm, "yaw_entry_pos")
    command._reset_fsm(torch.tensor([0]))
    assert torch.equal(command.yaw_entry_pos[0], torch.zeros(2))
    assert torch.equal(command.yaw_entry_pos[1], torch.tensor([7.0, 8.0]))
    assert "_fsm_spin_entry_pos" not in FSM_PATH.with_name("rewards.py").read_text(encoding="utf-8")


def test_vector_fsm_holds_yaw_for_minimum_dwell_but_unsafe_bypasses_it():
    fsm_module = _load_fsm_module()
    fsm = fsm_module.YawFSMVectorized(
        num_envs=1, dt=0.05, yaw_min_dwell=0.20, recovery_dwell=0.50
    )
    true = torch.tensor([True])
    false = torch.tensor([False])
    yaw = torch.tensor([0.20])

    fsm.update(yaw, false, false, false, false)  # FOUR -> TRANSITION_POS
    fsm.update(yaw, true, false, false, false)   # TRANSITION_POS -> YAW_POS
    assert fsm.state.item() == fsm_module.VQRFsmState.YAW_POS

    # A zero command cannot leave YAW for its first four 50 ms frames.
    for _ in range(4):
        fsm.update(torch.tensor([0.0]), false, false, false, false)
        assert fsm.state.item() == fsm_module.VQRFsmState.YAW_POS
    fsm.update(torch.tensor([0.0]), false, false, false, false)
    assert fsm.state.item() == fsm_module.VQRFsmState.RETURN_TO_4

    # Safety has absolute priority and does not wait for any command dwell.
    fsm.reset()
    fsm.update(yaw, false, false, false, false)
    fsm.update(yaw, true, false, false, false)
    fsm.update(yaw, false, false, false, true)
    assert fsm.state.item() == fsm_module.VQRFsmState.SAFE_RECOVERY


def test_recovery_needs_continuous_safe_four_stand_and_sign_flip_visits_four():
    fsm_module = _load_fsm_module()
    fsm = fsm_module.YawFSMVectorized(
        num_envs=1, dt=0.10, yaw_min_dwell=0.20, recovery_dwell=0.50
    )
    true = torch.tensor([True])
    false = torch.tensor([False])

    fsm.update(torch.tensor([0.2]), false, false, false, true)
    assert fsm.state.item() == fsm_module.VQRFsmState.SAFE_RECOVERY
    for _ in range(3):
        fsm.update(torch.tensor([0.2]), false, false, true, false)
    fsm.update(torch.tensor([0.2]), false, false, false, false)  # interrupts dwell
    for _ in range(4):
        fsm.update(torch.tensor([0.2]), false, false, true, false)
        assert fsm.state.item() == fsm_module.VQRFsmState.SAFE_RECOVERY
    fsm.update(torch.tensor([0.2]), false, false, true, false)
    assert fsm.state.item() == fsm_module.VQRFsmState.FOUR_STAND

    # A reversed command is considered only on the frame after RETURN lands
    # in FOUR_STAND; it cannot take a direct sign-flip edge.
    fsm.update(torch.tensor([0.2]), false, false, false, false)
    fsm.update(torch.tensor([0.2]), true, false, false, false)
    for _ in range(3):
        fsm.update(torch.tensor([0.0]), false, false, false, false)
    assert fsm.state.item() == fsm_module.VQRFsmState.RETURN_TO_4
    fsm.update(torch.tensor([-0.2]), false, false, true, false)
    assert fsm.state.item() == fsm_module.VQRFsmState.FOUR_STAND
    fsm.update(torch.tensor([-0.2]), false, false, true, false)
    assert fsm.state.item() == fsm_module.VQRFsmState.TRANSITION_NEG


def test_scalar_and_vector_fsm_are_equivalent_for_batched_trajectories():
    fsm_module = _load_fsm_module()
    num_envs = 8
    vector = fsm_module.YawFSMVectorized(
        num_envs=num_envs, dt=0.02, yaw_min_dwell=0.20, recovery_dwell=0.50
    )
    scalar = [
        fsm_module.VQRYawFSM(dt=0.02, yaw_min_dwell=0.20, recovery_dwell=0.50)
        for _ in range(num_envs)
    ]
    generator = torch.Generator().manual_seed(11)
    for _ in range(60):
        yaw = torch.empty(num_envs).uniform_(-0.4, 0.4, generator=generator)
        pos = torch.rand(num_envs, generator=generator) > 0.75
        neg = torch.rand(num_envs, generator=generator) > 0.75
        four = torch.rand(num_envs, generator=generator) > 0.20
        unsafe = torch.rand(num_envs, generator=generator) > 0.98
        observed = vector.update(yaw, pos, neg, four, unsafe)
        expected = torch.tensor(
            [
                int(item.update(float(yaw[i]), bool(pos[i]), bool(neg[i]), bool(four[i]), bool(unsafe[i])))
                for i, item in enumerate(scalar)
            ]
        )
        assert torch.equal(observed.cpu(), expected)


def test_swing_contact_selection_uses_any_wheel_and_mirrors_exactly():
    fsm_module = _load_fsm_module()
    positive_support_contact = torch.tensor(
        [[True, False], [False, True], [True, True], [True, True]]
    )
    negative_support_contact = torch.tensor(
        [[False, True], [False, False], [True, False], [True, True]]
    )
    diagonal = torch.tensor([1, 1, -1, 0])

    selected = fsm_module.select_swing_wheel_contact(
        positive_support_contact,
        negative_support_contact,
        diagonal,
    )

    # POS swings the NEG pair, NEG swings the POS pair, and FOUR has no
    # selected swing pair. A single contacted wheel is sufficient.
    assert torch.equal(selected, torch.tensor([True, False, True, False]))

    mirrored = fsm_module.select_swing_wheel_contact(
        negative_support_contact,
        positive_support_contact,
        -diagonal,
    )
    assert torch.equal(mirrored, selected)
