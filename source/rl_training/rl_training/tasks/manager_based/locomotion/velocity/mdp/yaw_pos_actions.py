"""POS action terms: limit physical targets at every 200 Hz physics tick."""

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction, JointVelocityAction
from isaaclab.utils import configclass

from .yaw_pos_action_limits import limit_position_target, limit_velocity_target


class LimitedJointPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._dt = float(env.cfg.sim.dt)
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self._lower, self._upper = limits[..., 0], limits[..., 1]
        self._target = self._asset.data.joint_pos[:, self._joint_ids].clone().clamp(min=self._lower, max=self._upper)
        self._target_velocity = torch.zeros_like(self._target)
        self._desired_target = self._target.clone()
        self._previous_applied = torch.zeros_like(self._target)
        self._processed_actions.copy_(self._target)
        # Install collider geometry once. Other tasks keep their original link-center behavior.
        centers = torch.zeros(self._asset.num_bodies, 3, device=self.device)
        signs = torch.zeros(self._asset.num_bodies, device=self.device)
        widths = torch.zeros(self._asset.num_bodies, device=self.device)
        for name, offset in cfg.wheel_center_offsets.items():
            ids, _ = self._asset.find_bodies([name], preserve_order=True)
            centers[ids] = centers.new_tensor(offset)
            signs[ids] = cfg.wheel_motor_axis_signs[name]
            widths[ids] = cfg.wheel_half_widths[name]
        self._asset._yaw_pos_wheel_center_offsets = centers
        self._asset._yaw_pos_wheel_motor_axis_signs = signs
        self._asset._yaw_pos_wheel_half_widths = widths

    @property
    def applied_actions(self):
        return (self._target - self._offset) / self._scale

    @property
    def applied_action_delta(self):
        return self.applied_actions - self._previous_applied

    def process_actions(self, actions):
        self._previous_applied.copy_(self.applied_actions)
        super().process_actions(actions)
        self._desired_target.copy_(self._processed_actions.clamp(min=self._lower, max=self._upper))
        self._processed_actions.copy_(self._target)

    def apply_actions(self):
        self._target, self._target_velocity = limit_position_target(
            self._desired_target, self._target, self._target_velocity, self._lower, self._upper,
            self.cfg.max_target_velocity, self.cfg.max_target_acceleration, self._dt,
        )
        self._processed_actions.copy_(self._target)
        super().apply_actions()

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._target[ids] = self._asset.data.joint_pos[ids][:, self._joint_ids].clamp(min=self._lower[ids], max=self._upper[ids])
        self._target_velocity[ids] = 0.
        self._desired_target[ids] = self._target[ids]
        self._processed_actions[ids] = self._target[ids]
        self._previous_applied[ids] = self.applied_actions[ids]


class LimitedJointVelocityAction(JointVelocityAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._dt = float(env.cfg.sim.dt)
        self._target = torch.zeros_like(self._processed_actions)
        self._desired_target = self._target.clone()
        self._previous_applied = self._target.clone()

    @property
    def applied_actions(self):
        return (self._target - self._offset) / self._scale

    @property
    def applied_action_delta(self):
        return self.applied_actions - self._previous_applied

    def process_actions(self, actions):
        self._previous_applied.copy_(self.applied_actions)
        super().process_actions(actions)
        self._desired_target.copy_(self._processed_actions.clamp(-self.cfg.max_target_velocity, self.cfg.max_target_velocity))
        self._processed_actions.copy_(self._target)

    def apply_actions(self):
        self._target = limit_velocity_target(self._desired_target, self._target, self.cfg.max_target_velocity,
                                             self.cfg.max_target_acceleration, self._dt)
        self._processed_actions.copy_(self._target)
        super().apply_actions()

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._target[ids] = 0.
        self._desired_target[ids] = 0.
        self._processed_actions[ids] = 0.
        self._previous_applied[ids] = self.applied_actions[ids]


@configclass
class LimitedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = LimitedJointPositionAction
    max_target_velocity: float = 2.0
    max_target_acceleration: float = 10.0
    wheel_center_offsets: dict = {}
    wheel_motor_axis_signs: dict = {}
    wheel_half_widths: dict = {}


@configclass
class LimitedJointVelocityActionCfg(JointVelocityActionCfg):
    class_type: type = LimitedJointVelocityAction
    max_target_velocity: float = 58.90
    max_target_acceleration: float = 20.0


def applied_target_history(env, position_action_name: str = "joint_pos", velocity_action_name: str = "joint_vel"):
    """55D policy channels 39:55, in the same normalized leg/wheel order as actions."""
    return torch.cat([env.action_manager.get_term(name).applied_actions
                      for name in (position_action_name, velocity_action_name)], dim=-1)


def applied_target_rate_l2(env, position_action_name: str = "joint_pos", velocity_action_name: str = "joint_vel"):
    return torch.cat([env.action_manager.get_term(name).applied_action_delta
                      for name in (position_action_name, velocity_action_name)], dim=-1).square().sum(dim=-1)
