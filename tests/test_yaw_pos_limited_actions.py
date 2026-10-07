"""Exercise the POS action terms across process/apply/reset boundaries on CPU."""

import sys
import types
from types import SimpleNamespace as NS

import torch

from test_yaw_pos_task import MDP, _load


def action_module(monkeypatch):
    actions = types.ModuleType("isaaclab.envs.mdp.actions")
    actions.JointPositionActionCfg = type("JointPositionActionCfg", (), {})
    actions.JointVelocityActionCfg = type("JointVelocityActionCfg", (), {})
    base = types.ModuleType("isaaclab.envs.mdp.actions.joint_actions")
    # Emulate only the installed Isaac Lab parent's affine process and target writes.
    class JointAction:
        def process_actions(self, actions):
            self._raw_actions[:] = actions
            self._processed_actions = self._raw_actions * self._scale + self._offset

        def apply_actions(self):
            self.written_target = self._processed_actions.clone()

        def reset(self, env_ids=None):
            self._raw_actions[env_ids] = 0.
    base.JointPositionAction = JointAction
    base.JointVelocityAction = JointAction
    utils = types.ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    monkeypatch.setitem(sys.modules, actions.__name__, actions)
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    return _load(monkeypatch, "yaw_pos_actions", MDP / "yaw_pos_actions.py")


def term_state(term, position):
    term.cfg = NS(max_target_velocity=2. if position else 58.9,
                  max_target_acceleration=10. if position else 20.)
    term._dt = .005
    term._target = torch.zeros(2, 12 if position else 4, dtype=torch.float64)
    term._raw_actions = torch.zeros_like(term._target)
    term._processed_actions = torch.zeros_like(term._target)
    term._desired_target = torch.zeros_like(term._target)
    term._previous_applied = torch.zeros_like(term._target)
    term._scale = .5 if position else 5.
    term._offset = torch.zeros_like(term._target)
    if position:
        term._lower = torch.full_like(term._target, -.785)
        term._upper = -term._lower
        term._target_velocity = torch.zeros_like(term._target)
        term._joint_ids = list(range(12))
        term._asset = NS(data=NS(joint_pos=torch.zeros_like(term._target)))
    return term


def test_large_first_action_does_not_jump_target_and_history_is_applied_target(monkeypatch):
    module = action_module(monkeypatch)
    legs = term_state(module.LimitedJointPositionAction.__new__(module.LimitedJointPositionAction), True)
    wheels = term_state(module.LimitedJointVelocityAction.__new__(module.LimitedJointVelocityAction), False)
    legs.process_actions(torch.full_like(legs._target, 100.))
    wheels.process_actions(torch.full_like(wheels._target, 100.))
    assert legs._processed_actions.count_nonzero() == wheels._processed_actions.count_nonzero() == 0
    for _ in range(4):
        legs.apply_actions()
        wheels.apply_actions()
    torch.testing.assert_close(legs.written_target, torch.full_like(legs._target, .0025))
    torch.testing.assert_close(wheels.written_target, torch.full_like(wheels._target, .4))
    env = NS(action_manager=NS(get_term=lambda name: legs if name == "joint_pos" else wheels))
    history = module.applied_target_history(env)
    assert history.shape == (2, 16)
    torch.testing.assert_close(history[:, :12], legs.written_target / .5)
    torch.testing.assert_close(history[:, 12:], wheels.written_target / 5.)
    assert history.abs().max() < .1


def test_partial_reset_starts_from_measured_pose_and_preserves_other_env(monkeypatch):
    module = action_module(monkeypatch)
    legs = term_state(module.LimitedJointPositionAction.__new__(module.LimitedJointPositionAction), True)
    legs._target.fill_(.4)
    legs._target_velocity.fill_(1.)
    legs._asset.data.joint_pos[0] = -.2
    legs.reset([0])
    torch.testing.assert_close(legs._target[0], torch.full_like(legs._target[0], -.2))
    torch.testing.assert_close(legs._target[1], torch.full_like(legs._target[1], .4))
    assert legs._target_velocity[0].count_nonzero() == 0
    torch.testing.assert_close(legs._target_velocity[1], torch.ones_like(legs._target_velocity[1]))
    assert legs.applied_action_delta[0].count_nonzero() == 0
    legs.process_actions(torch.full_like(legs._target, 100.))
    legs.apply_actions()
    assert (legs.written_target[0] + .2).abs().max() <= .00025 + 1e-12
    wheels = term_state(module.LimitedJointVelocityAction.__new__(module.LimitedJointVelocityAction), False)
    wheels._target.fill_(20.)
    wheels.reset([0])
    assert wheels._target[0].count_nonzero() == 0
    torch.testing.assert_close(wheels._target[1], torch.full_like(wheels._target[1], 20.))


def test_action_rate_measures_applied_change_across_policy_steps(monkeypatch):
    module = action_module(monkeypatch)
    legs = term_state(module.LimitedJointPositionAction.__new__(module.LimitedJointPositionAction), True)
    wheels = term_state(module.LimitedJointVelocityAction.__new__(module.LimitedJointVelocityAction), False)
    env = NS(action_manager=NS(get_term=lambda name: legs if name == "joint_pos" else wheels))
    for term in (legs, wheels):
        term.process_actions(torch.ones_like(term._target))
        for _ in range(4):
            term.apply_actions()
    before = module.applied_target_history(env).clone()
    for term in (legs, wheels):
        term.process_actions(-torch.ones_like(term._target))
        for _ in range(4):
            term.apply_actions()
    expected = (module.applied_target_history(env) - before).square().sum(-1)
    torch.testing.assert_close(module.applied_target_rate_l2(env), expected)
