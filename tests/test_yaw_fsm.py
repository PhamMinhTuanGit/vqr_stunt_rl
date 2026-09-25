"""CPU-only ownership tests for the yaw-FSM position anchor."""

from __future__ import annotations

import ast
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
FSM_GATES_PATH = FSM_PATH.with_name("fsm_gates.py")
REWARDS_PATH = FSM_PATH.with_name("rewards.py")


def _load_fsm_module():
    """Load the FSM module through its Isaac-Lab-independent fallback."""
    spec = importlib.util.spec_from_file_location("yaw_fsm_under_test", FSM_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_fsm_tracking_reward():
    """Load the real FSM pose rewards and gates without importing Isaac Sim."""
    fsm_module = _load_fsm_module()
    gate_tree = ast.parse(FSM_GATES_PATH.read_text(encoding="utf-8"))
    gate_names = {"_CACHE_STEP_ATTR", "_CACHE_ATTR", "_step_key", "_required_tensor", "fsm_gates"}
    gate_nodes = []
    for node in gate_tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id in gate_names:
            gate_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in gate_names:
            gate_nodes.append(node)
    namespace = {
        "torch": torch,
        "VQRFsmState": fsm_module.VQRFsmState,
        "select_swing_wheel_contact": fsm_module.select_swing_wheel_contact,
    }
    exec(compile(ast.Module(body=gate_nodes, type_ignores=[]), FSM_GATES_PATH, "exec"), namespace)

    reward_names = {
        "_yaw_wheel_contacts",
        "_yaw_support_shape",
        "yaw_com_support",
        "yaw_com_inside_support_segment",
        "yaw_lift_clearance",
        "fsm_gated_tracking",
        "four_stand_ready_bonus",
    }
    reward_tree = ast.parse(REWARDS_PATH.read_text(encoding="utf-8"))
    reward_nodes = [
        node
        for node in reward_tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in reward_names)
        or (isinstance(node, ast.ClassDef) and node.name == "ReturnToFourLanding")
    ]
    class FakeManagerTermBase:
        def __init__(self, cfg, env):
            pass

    class FakeSceneEntityCfg(SimpleNamespace):
        def __init__(self, name):
            super().__init__(name=name)

    namespace.update(
        {
            "ManagerBasedRLEnv": object,
            "SceneEntityCfg": FakeSceneEntityCfg,
            "ContactSensor": object,
            "RigidObject": object,
            "ManagerTermBase": FakeManagerTermBase,
            "RewTerm": object,
            "_fsm_step_telemetry": lambda *args: None,
            "_fsm_transition_telemetry": lambda *args: None,
            "_fsm_masked_accumulate": lambda *args: None,
            "_fsm_masked_accumulate_pair": lambda *args: None,
            "_fsm_episode_or": lambda *args: None,
            "_fsm_record_positive_budget": lambda *args: None,
        }
    )
    exec(compile(ast.Module(body=reward_nodes, type_ignores=[]), REWARDS_PATH, "exec"), namespace)
    namespace["_yaw_lift_progress"] = lambda env, cfg, *args: env.lift_progress[cfg.name]
    namespace["_yaw_support_geometry"] = lambda env, cfg: (
        torch.zeros(env.num_envs),
        torch.full((env.num_envs,), 0.5),
        torch.ones(env.num_envs),
    )
    return namespace, fsm_module.VQRFsmState


def test_transition_telemetry_uses_active_support_lift_and_pose():
    tree = ast.parse(REWARDS_PATH.read_text(encoding="utf-8"))
    names = {"_fsm_step_telemetry", "_fsm_transition_telemetry"}
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_TRANSITION_TELEMETRY_FIELDS"
        )
    ]
    namespace = {
        "torch": torch,
        "euler_xyz_from_quat": lambda q: (q[:, 0], q[:, 1], q[:, 2]),
        "_yaw_wheel_contacts": lambda env, cfg, threshold: env.torso_contact,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), REWARDS_PATH, "exec"), namespace)

    state = torch.tensor([1, 3, 1, 2])
    gates = {
        "fsm_state": state,
        "b_trans": torch.tensor([True, True, True, False]),
        "just_switched": torch.tensor([True, True, False, True]),
    }
    env = SimpleNamespace(
        num_envs=4,
        torso_contact=torch.tensor([[False], [True], [True], [True]]),
    )
    robot = SimpleNamespace(data=SimpleNamespace(root_quat_w=torch.tensor([
        [0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.0, 0.0],
    ])))
    command = SimpleNamespace(cfg=SimpleNamespace(
        torso_sensor_cfg=SimpleNamespace(name="torso"),
        contact_threshold=1.0,
        pose_angle_limit=0.35,
        clearance_fraction=0.8,
    ))
    namespace["_fsm_step_telemetry"](env, gates)
    namespace["_fsm_transition_telemetry"](
        env, gates,
        torch.tensor([[True, True], [True, False], [True, True], [True, True]]),
        torch.tensor([[0.9, 0.8], [0.9, 0.9], [0.2, 0.8], [1.0, 1.0]]),
        robot, command,
    )
    assert env._yaw_fsm_transition_duration_steps.tolist() == [1, 1, 1, 0]
    assert env._yaw_fsm_transition_attempts.tolist() == [1, 1, 1, 0]
    expected = {
        "support_ready": [1.0, 0.0, 1.0, 0.0],
        "lift_wheel_1_progress": [0.9, 0.9, 0.2, 0.0],
        "lift_wheel_2_progress": [0.8, 0.9, 0.8, 0.0],
        "clearance_ready": [1.0, 1.0, 0.0, 0.0],
        "attitude_ready": [1.0, 1.0, 0.0, 0.0],
        "pose_ready": [1.0, 0.0, 0.0, 0.0],
        "torso_contact": [0.0, 1.0, 1.0, 0.0],
    }
    for field, values in expected.items():
        assert torch.allclose(
            getattr(env, f"_yaw_fsm_transition_{field}_sum"), torch.tensor(values)
        )


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
    for _ in range(5):
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


def test_vector_fsm_command_exit_and_unsafe_bypass_pose_loss_grace():
    fsm_module = _load_fsm_module()
    fsm = fsm_module.YawFSMVectorized(
        num_envs=1, dt=0.05, yaw_min_dwell=0.20, recovery_dwell=0.50
    )
    true = torch.tensor([True])
    false = torch.tensor([False])
    yaw = torch.tensor([0.20])

    fsm.update(yaw, false, false, false, false)  # FOUR -> TRANSITION_POS
    for _ in range(2):
        fsm.update(yaw, true, false, false, false)  # TRANSITION_POS -> YAW_POS
    assert fsm.state.item() == fsm_module.VQRFsmState.YAW_POS

    # A zero command exits immediately, even when pose readiness is lost.
    fsm.update(torch.tensor([0.0]), false, false, false, false)
    assert fsm.state.item() == fsm_module.VQRFsmState.RETURN_TO_4
    assert fsm._yaw_pose_invalid_time.item() == 0.0

    # Safety has absolute priority and does not wait for any command dwell.
    fsm.reset()
    fsm.update(yaw, false, false, false, false)
    for _ in range(2):
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
    fsm.update(torch.tensor([0.0]), false, false, false, false)
    assert fsm.state.item() == fsm_module.VQRFsmState.RETURN_TO_4
    fsm.update(torch.tensor([-0.2]), false, false, true, false)
    assert fsm.state.item() == fsm_module.VQRFsmState.FOUR_STAND
    fsm.update(torch.tensor([-0.2]), false, false, true, false)
    assert fsm.state.item() == fsm_module.VQRFsmState.TRANSITION_NEG


def test_scalar_and_vector_fsm_are_equivalent_for_batched_trajectories():
    fsm_module = _load_fsm_module()
    # 8,192 independently randomized transitions gives the vectorized path
    # meaningful coverage while keeping this test comfortably CPU-only.
    num_envs = 32
    vector = fsm_module.YawFSMVectorized(
        num_envs=num_envs, dt=0.02, yaw_min_dwell=0.20, recovery_dwell=0.50
    )
    scalar = [
        fsm_module.VQRYawFSM(dt=0.02, yaw_min_dwell=0.20, recovery_dwell=0.50)
        for _ in range(num_envs)
    ]
    generator = torch.Generator().manual_seed(11)
    for _ in range(256):
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


def test_vector_fsm_reset_restores_the_canonical_four_stand_state():
    fsm_module = _load_fsm_module()
    fsm = fsm_module.YawFSMVectorized(num_envs=4, dt=0.02)
    fsm.fsm_state[:] = torch.tensor([1, 2, 4, 6])
    fsm.support_diagonal[:] = torch.tensor([1, 1, -1, -1])
    fsm.state_time[:] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    fsm.just_returned_to_four[:] = True
    fsm._yaw_pose_invalid_time[:] = torch.tensor([0.02, 0.04, 0.06, 0.08])
    fsm._yaw_pose_ready_time[:] = torch.tensor([0.02, 0.04, 0.06, 0.08])

    fsm.reset(torch.tensor([0, 2, 3]))

    assert torch.equal(fsm.fsm_state, torch.tensor([0, 2, 0, 0]))
    assert torch.equal(fsm.support_diagonal, torch.tensor([0, 1, 0, 0]))
    assert torch.equal(fsm.state_time, torch.tensor([0.0, 0.2, 0.0, 0.0]))
    assert torch.equal(fsm.just_returned_to_four, torch.tensor([False, True, False, False]))
    assert torch.equal(fsm._yaw_pose_invalid_time, torch.tensor([0.0, 0.04, 0.0, 0.0]))
    assert torch.equal(fsm._yaw_pose_ready_time, torch.tensor([0.0, 0.04, 0.0, 0.0]))


def test_return_completion_pulse_requires_four_ready_for_both_signs():
    module = _load_fsm_module()
    for direction in (1, -1):
        for fsm in (module.VQRYawFSM(yaw_pose_ready_dwell=0.0),
                    module.YawFSMVectorized(1, yaw_pose_ready_dwell=0.0)):
            def step(command, ready=False):
                pos = direction == 1 and ready
                neg = direction == -1 and ready
                if isinstance(fsm, module.YawFSMVectorized):
                    fsm.update(torch.tensor([command]), torch.tensor([pos]),
                               torch.tensor([neg]), torch.tensor([ready]), torch.tensor([False]))
                    return int(fsm.state.item()), bool(fsm.just_returned_to_four.item())
                fsm.update(command, pos, neg, ready, False)
                return int(fsm.state), fsm.just_returned_to_four

            step(0.2 * direction)
            step(0.2 * direction, ready=True)
            assert step(0.0) == (module.VQRFsmState.RETURN_TO_4, False)
            assert step(0.0) == (module.VQRFsmState.RETURN_TO_4, False)
            assert step(0.0, ready=True) == (module.VQRFsmState.FOUR_STAND, True)
            assert step(0.0) == (module.VQRFsmState.FOUR_STAND, False)


def test_yaw_pose_loss_grace_reacquires_the_same_diagonal_after_continuous_loss():
    fsm_module = _load_fsm_module()
    state = fsm_module.VQRFsmState
    false = torch.tensor([False])
    true = torch.tensor([True])
    for direction, yaw_state, transition_state in (
        (1, state.YAW_POS, state.TRANSITION_POS),
        (-1, state.YAW_NEG, state.TRANSITION_NEG),
    ):
        fsm = fsm_module.YawFSMVectorized(num_envs=1, dt=0.02, yaw_pose_loss_grace=0.10)
        command = torch.tensor([0.2 * direction])
        positive = true if direction == 1 else false
        negative = true if direction == -1 else false
        fsm.update(command, false, false, false, false)
        for _ in range(5):
            fsm.update(command, positive, negative, false, false)
        assert fsm.state.item() == yaw_state
        assert fsm.support_diagonal.item() == direction

        for step in range(4):
            fsm.update(command, false, false, false, false)
            assert fsm.state.item() == yaw_state
            assert torch.allclose(fsm._yaw_pose_invalid_time, torch.tensor([0.02 * (step + 1)]))

        # A recovered pose breaks the loss streak; the next four invalid
        # frames must still be shorter than the five-step grace period.
        fsm.update(command, positive, negative, false, false)
        assert fsm._yaw_pose_invalid_time.item() == 0.0
        for _ in range(4):
            fsm.update(command, false, false, false, false)
            assert fsm.state.item() == yaw_state
        fsm.update(command, false, false, false, false)
        assert fsm.state.item() == transition_state
        assert fsm.support_diagonal.item() == direction
        assert fsm._yaw_pose_invalid_time.item() == 0.0

        for step in range(4):
            fsm.update(command, positive, negative, false, false)
            assert fsm.state.item() == transition_state
            assert torch.allclose(fsm._yaw_pose_ready_time, torch.tensor([0.02 * (step + 1)]))
        fsm.update(command, positive, negative, false, false)
        assert fsm.state.item() == yaw_state
        assert fsm.support_diagonal.item() == direction
        assert fsm._yaw_pose_ready_time.item() == 0.0


def test_yaw_pose_loss_priority_unsafe_then_command_exit():
    fsm_module = _load_fsm_module()
    state = fsm_module.VQRFsmState
    false = torch.tensor([False])
    true = torch.tensor([True])
    for direction, exit_command in ((1, 0.0), (-1, 0.2)):
        fsm = fsm_module.YawFSMVectorized(num_envs=1, dt=0.02, yaw_pose_loss_grace=0.10)
        command = torch.tensor([0.2 * direction])
        positive = true if direction == 1 else false
        negative = true if direction == -1 else false
        fsm.update(command, false, false, false, false)
        for _ in range(5):
            fsm.update(command, positive, negative, false, false)
        for _ in range(4):
            fsm.update(command, false, false, false, false)
        fsm.update(torch.tensor([exit_command]), false, false, false, false)
        assert fsm.state.item() == state.RETURN_TO_4
        assert fsm._yaw_pose_invalid_time.item() == 0.0

        fsm.reset()
        fsm.update(command, false, false, false, false)
        for _ in range(5):
            fsm.update(command, positive, negative, false, false)
        fsm.update(torch.tensor([exit_command]), false, false, false, true)
        assert fsm.state.item() == state.SAFE_RECOVERY
        assert fsm._yaw_pose_invalid_time.item() == 0.0


def test_yaw_pose_loss_grace_is_exposed_on_command_config():
    fsm_module = _load_fsm_module()
    assert fsm_module.YawFSMCommandCfg.yaw_pose_loss_grace == 0.10
    assert fsm_module.YawFSMCommandCfg.yaw_pose_ready_dwell == 0.10


def test_pose_ready_requires_five_continuous_steps_for_both_fsm_implementations_and_signs():
    module = _load_fsm_module()
    state = module.VQRFsmState
    for vectorized in (False, True):
        for direction, transition, yaw_state in (
            (1, state.TRANSITION_POS, state.YAW_POS),
            (-1, state.TRANSITION_NEG, state.YAW_NEG),
        ):
            fsm = (
                module.YawFSMVectorized(num_envs=1, dt=0.02)
                if vectorized else module.VQRYawFSM(dt=0.02)
            )

            def step(command, ready=False, unsafe=False):
                pos = ready and direction == 1
                neg = ready and direction == -1
                if vectorized:
                    return int(fsm.update(command, pos, neg, False, unsafe).item())
                return int(fsm.update(command, pos, neg, False, unsafe))

            def ready_time():
                value = fsm._yaw_pose_ready_time
                return float(value.item()) if vectorized else value

            command = 0.2 * direction
            assert step(command) == transition
            for index in range(4):
                assert step(command, ready=True) == transition
                assert abs(ready_time() - 0.02 * (index + 1)) < 1.0e-6
            assert step(command) == transition
            assert ready_time() == 0.0
            for _ in range(4):
                assert step(command, ready=True) == transition
            assert step(command, ready=True) == yaw_state
            assert ready_time() == 0.0


def test_pose_ready_dwell_yields_to_unsafe_and_command_exit_for_both_signs():
    module = _load_fsm_module()
    state = module.VQRFsmState
    for vectorized in (False, True):
        for direction in (1, -1):
            for exit_command, unsafe, expected in (
                (0.0, False, state.RETURN_TO_4),
                (-0.2 * direction, False, state.RETURN_TO_4),
                (0.0, True, state.SAFE_RECOVERY),
            ):
                fsm = (
                    module.YawFSMVectorized(num_envs=1, dt=0.02)
                    if vectorized else module.VQRYawFSM(dt=0.02)
                )
                command = 0.2 * direction
                pos = direction == 1
                neg = direction == -1
                fsm.update(command, False, False, False, False)
                for _ in range(4):
                    fsm.update(command, pos, neg, False, False)
                ready_time = fsm._yaw_pose_ready_time
                if vectorized:
                    ready_time = ready_time.item()
                assert abs(ready_time - 0.08) < 1.0e-6
                fsm.update(exit_command, pos, neg, False, unsafe)
                actual_state = fsm.state.item() if vectorized else fsm.state
                ready_time = fsm._yaw_pose_ready_time
                if vectorized:
                    ready_time = ready_time.item()
                assert actual_state == expected
                assert ready_time == 0.0


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


def test_fsm_tracking_support_bonus_requires_lift_and_is_independent_of_yaw_error():
    rewards, state = _load_fsm_tracking_reward()
    tracking_reward = rewards["fsm_gated_tracking"]
    states = torch.tensor(
        [
            state.TRANSITION_POS,
            state.TRANSITION_POS,
            state.YAW_POS,
            state.YAW_NEG,
            state.FOUR_STAND,
            state.RETURN_TO_4,
            state.SAFE_RECOVERY,
            state.TRANSITION_NEG,
            state.YAW_POS,
            state.YAW_POS,
            state.YAW_NEG,
            state.YAW_NEG,
        ]
    )
    diagonal = torch.tensor([1, 1, 1, -1, 0, 1, 0, -1, 1, 1, -1, -1])
    forces = torch.zeros(12, 4, 3)
    forces[:, :, 2] = 2.0
    forces[2, 3, 2] = 0.0  # POS loses HR: partial bonus, no tracking.
    forces[9, [0, 3], 2] = 0.0  # POS loses both support wheels.
    forces[10, 2, 2] = 0.0  # NEG loses HL: mirror of case 2.
    forces[11, [1, 2], 2] = 0.0  # NEG loses both support wheels.
    lift = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    yaw_velocity = torch.tensor([0.0, 10.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    command = SimpleNamespace(
        fsm_state=states,
        support_diagonal=diagonal,
        state_time=torch.zeros(12),
        just_switched=torch.zeros(12, dtype=torch.bool),
        cfg=SimpleNamespace(yaw_rate_range=(-1.0, 1.0)),
    )
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))

    class Scene(dict):
        sensors = {"contact_forces": sensor}

    root_ang_vel_b = torch.stack([torch.zeros(12), torch.zeros(12), yaw_velocity], dim=1)
    env = SimpleNamespace(
        num_envs=12,
        device="cpu",
        common_step_counter=0,
        scene=Scene(robot=SimpleNamespace(data=SimpleNamespace(root_ang_vel_b=root_ang_vel_b))),
        lift_progress={
            "lift_pos": lift[:, None].expand(-1, 2),
            "lift_neg": lift[:, None].expand(-1, 2),
        },
        command_manager=SimpleNamespace(
            get_term=lambda _: command,
            get_command=lambda _: torch.zeros(12, 1),
        ),
    )
    positive_cfg = SimpleNamespace(name="contact_forces", body_ids=torch.tensor([0, 3]))
    negative_cfg = SimpleNamespace(name="contact_forces", body_ids=torch.tensor([1, 2]))

    observed = tracking_reward(
        env,
        command_name="yaw_rate_cmd",
        fsm_command_name="yaw_rate_cmd",
        support_sensor_cfg=positive_cfg,
        support_sensor_cfg_mirror=negative_cfg,
        lifted_asset_cfg=SimpleNamespace(name="lift_pos"),
        lifted_asset_cfg_mirror=SimpleNamespace(name="lift_neg"),
        wheel_radius=0.091,
        target_clearance=0.05,
        std=0.30,
    )
    expected = torch.tensor([
        0.0, 0.25, 0.0125, 0.25, 0.0, 0.0,
        0.0, 0.0, 1.25, 0.0, 0.0125, 0.0,
    ])
    assert torch.allclose(observed, expected, atol=1e-6)
    # With lift=1 and tracking either blocked or negligible, the +2 maximum
    # support bonus maps 0/1/2 support contacts to 0/0.05/1 respectively.
    assert torch.allclose(observed[[9, 2, 1]] * 4.0, torch.tensor([0.0, 0.05, 1.0]))
    assert torch.allclose(observed[[11, 10, 3]] * 4.0, torch.tensor([0.0, 0.05, 1.0]))
    assert env._yaw_fsm_pos_support_loss_max_steps[[2, 9]].tolist() == [1, 1]
    assert env._yaw_fsm_neg_support_loss_max_steps[[10, 11]].tolist() == [1, 1]


def test_fsm_pose_rewards_keep_dense_yaw_signal_without_support_contact():
    rewards, state = _load_fsm_tracking_reward()
    n = 8
    command = SimpleNamespace(
        fsm_state=torch.tensor([
            state.YAW_POS, state.YAW_POS, state.YAW_POS,
            state.YAW_NEG, state.YAW_NEG, state.YAW_NEG,
            state.TRANSITION_POS, state.RETURN_TO_4,
        ]),
        support_diagonal=torch.tensor([1, 1, 1, -1, -1, -1, 1, 1]),
        state_time=torch.zeros(n),
        just_switched=torch.zeros(n, dtype=torch.bool),
    )
    lift_pos = torch.ones(n, 2)
    lift_neg = torch.ones(n, 2)
    lift_pos[2] = 0.0
    lift_neg[5] = 0.0
    env = SimpleNamespace(
        num_envs=n,
        device="cpu",
        common_step_counter=0,
        scene={},
        command_manager=SimpleNamespace(get_term=lambda _: command),
        lift_progress={"lift_pos": lift_pos, "lift_neg": lift_neg},
    )
    fsm_args = {"fsm_command_name": "yaw_rate_cmd"}
    geom_args = {
        "asset_cfg": SimpleNamespace(name="robot"),
        "asset_cfg_mirror": SimpleNamespace(name="robot"),
    }
    com = rewards["yaw_com_support"](env, std=0.08, **geom_args, **fsm_args)
    inside = rewards["yaw_com_inside_support_segment"](
        env, std=0.05, **geom_args, **fsm_args
    )
    expected_geom = torch.ones(n)
    expected_geom[-1] = 0.0  # RETURN no longer pays diagonal geometry.
    assert torch.allclose(com, expected_geom)
    assert torch.allclose(inside, expected_geom)

    lift = rewards["yaw_lift_clearance"](
        env,
        asset_cfg=SimpleNamespace(name="lift_pos"),
        asset_cfg_mirror=SimpleNamespace(name="lift_neg"),
        wheel_radius=0.091,
        target_clearance=0.05,
        **fsm_args,
    )
    expected_lift = torch.tensor([1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 0.0])
    assert torch.allclose(lift, expected_lift)

    # The legacy branch does not inspect FSM contact and retains exact values.
    legacy_com = rewards["yaw_com_support"](env, geom_args["asset_cfg"], std=0.08)
    legacy_inside = rewards["yaw_com_inside_support_segment"](
        env, geom_args["asset_cfg"], std=0.05
    )
    legacy_lift = rewards["yaw_lift_clearance"](
        env, SimpleNamespace(name="lift_pos"), wheel_radius=0.091, target_clearance=0.05
    )
    assert torch.equal(legacy_com, torch.ones(n))
    assert torch.equal(legacy_inside, torch.ones(n))
    assert torch.equal(legacy_lift, 2.0 * lift_pos.mean(dim=1) - 1.0)


def test_return_landing_and_completion_bonus_are_mirrored_one_time_events():
    rewards, state = _load_fsm_tracking_reward()
    command = SimpleNamespace(
        fsm_state=torch.tensor([state.RETURN_TO_4, state.RETURN_TO_4]),
        support_diagonal=torch.tensor([1, -1]),
        state_time=torch.zeros(2),
        just_switched=torch.ones(2, dtype=torch.bool),
        just_returned_to_four=torch.zeros(2, dtype=torch.bool),
        cfg=SimpleNamespace(target_clearance=0.05),
    )
    forces = torch.zeros(2, 4, 3)
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        common_step_counter=0,
        scene=SimpleNamespace(sensors={"contact_forces": sensor}),
        command_manager=SimpleNamespace(get_term=lambda _: command),
        lift_progress={"pos": torch.ones(2, 2), "neg": torch.ones(2, 2)},
    )
    term = rewards["ReturnToFourLanding"](None, env)
    args = {
        "asset_cfg": SimpleNamespace(name="pos"),
        "asset_cfg_mirror": SimpleNamespace(name="neg"),
        "sensor_cfg": SimpleNamespace(name="contact_forces", body_ids=[1, 2]),
        "sensor_cfg_mirror": SimpleNamespace(name="contact_forces", body_ids=[0, 3]),
        "wheel_radius": 0.091,
        "fsm_command_name": "yaw_rate_cmd",
    }
    assert torch.equal(term(env, **args), torch.zeros(2))

    command.just_switched.fill_(False)
    env.common_step_counter += 1
    env.lift_progress["pos"][:] = 0.5
    env.lift_progress["neg"][:] = 0.5
    assert torch.allclose(term(env, **args), torch.full((2,), 0.5))

    env.common_step_counter += 1
    forces[0, 1, 2] = 2.0  # POS swing FR
    forces[1, 0, 2] = 2.0  # NEG swing FL
    assert torch.allclose(term(env, **args), torch.ones(2))
    env.common_step_counter += 1
    assert torch.equal(term(env, **args), torch.zeros(2))

    env.common_step_counter += 1
    forces[0, 2, 2] = 2.0  # POS swing HL
    forces[1, 3, 2] = 2.0  # NEG swing HR
    assert torch.allclose(term(env, **args), torch.ones(2))

    command.fsm_state[:] = state.FOUR_STAND
    command.support_diagonal.zero_()
    command.just_switched.fill_(True)
    command.just_returned_to_four.fill_(True)
    env.common_step_counter += 1
    assert torch.equal(term(env, **args), torch.zeros(2))
    assert torch.equal(
        rewards["four_stand_ready_bonus"](env, "yaw_rate_cmd"), torch.ones(2)
    )
    command.just_switched.fill_(False)
    command.just_returned_to_four.fill_(False)
    env.common_step_counter += 1
    assert torch.equal(
        rewards["four_stand_ready_bonus"](env, "yaw_rate_cmd"), torch.zeros(2)
    )

    # A wheel that was already in contact on return entry was not regained.
    term.reset()
    command.fsm_state[:] = state.RETURN_TO_4
    command.support_diagonal[:] = torch.tensor([1, -1])
    command.just_switched.fill_(True)
    env.common_step_counter += 1
    assert torch.equal(term(env, **args), torch.zeros(2))
    command.just_switched.fill_(False)
    env.common_step_counter += 1
    assert torch.equal(term(env, **args), torch.zeros(2))


def test_return_timeout_only_fires_in_return_after_2_5_seconds():
    path = FSM_PATH.with_name("terminations.py")
    function = next(
        node for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == "fsm_return_timeout"
    )
    namespace = {"torch": torch, "VQRFsmState": _load_fsm_module().VQRFsmState,
                 "ManagerBasedRLEnv": object}
    exec(compile(ast.Module(body=[function], type_ignores=[]), path, "exec"), namespace)
    command = SimpleNamespace(
        fsm_state=torch.tensor([5, 5, 2, 0]),
        state_time=torch.tensor([2.48, 2.50, 4.0, 4.0]),
    )
    env = SimpleNamespace(
        num_envs=4, device="cpu",
        command_manager=SimpleNamespace(get_term=lambda _: command),
    )
    assert torch.equal(namespace["fsm_return_timeout"](env), torch.tensor([False, True, False, False]))
