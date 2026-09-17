"""Standing-to-diagonal bootstrap using the existing manager-based task."""

from dataclasses import MISSING

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from .to_reference import REFERENCE_PATH, contact_targets, load_leg_reference, pose_reference, smoothstep, unload_fraction
from .events import reset_four_wheel_standing


class TOPivotCommand(CommandTerm):
    """One nonperiodic command [yaw rate, lambda], with episode-local timing.

    Do not use episode_length_buf as a clock: RSL-RL randomizes it at startup.
    Levels are latched per reset so active episodes keep their own task.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.transition_duration_s <= 0 or cfg.standing_time_s < 0:
            raise ValueError("Invalid transition timing")
        if not 0 <= cfg.initial_level <= 4:
            raise ValueError("initial_level must be 0..4")
        self.robot = env.scene[cfg.asset_name]
        self.sensor = env.scene.sensors[cfg.sensor_name]
        self.joint_ids, names = self.robot.find_joints(cfg.leg_joint_names, preserve_order=True)
        self.contact_ids, wheel_names = self.sensor.find_bodies(cfg.wheel_body_names, preserve_order=True)
        if names != cfg.leg_joint_names or wheel_names != cfg.wheel_body_names:
            raise ValueError("Runtime joint/contact name mapping differs from configuration")
        reference = load_leg_reference(names, cfg.reference_path)
        self.target = torch.tensor([reference[name] for name in names], device=self.device)
        self.standing = torch.tensor([cfg.standing_positions[name] for name in names], device=self.device)
        limits = self.robot.data.joint_pos_limits[:, self.joint_ids]
        margin = env.cfg.actions.leg_positions.joint_limit_margin
        if margin is None or margin < 0:
            raise ValueError("TO transition requires a nonnegative action joint-limit margin")
        if torch.any((self.target < limits[..., 0] + margin) | (self.target > limits[..., 1] - margin)):
            raise ValueError("TO reference outside runtime joint limits plus safety margin")
        self.level = cfg.initial_level
        self.episode_level = torch.full((self.num_envs,), self.level, device=self.device, dtype=torch.long)
        self.elapsed = torch.zeros(self.num_envs, device=self.device)
        self._command = torch.zeros(self.num_envs, 2, device=self.device)
        self.sampled_yaw = torch.zeros(self.num_envs, device=self.device)
        self.weights = torch.tensor(cfg.pose_weights, device=self.device)
        self.yaw_limits = torch.tensor(cfg.yaw_limits, device=self.device)
        self.required_hold = torch.tensor(cfg.hold_success_s, device=self.device)
        self.hold = torch.zeros_like(self.elapsed)
        self.best_hold = torch.zeros_like(self.elapsed)
        self.final_steps = torch.zeros_like(self.elapsed)
        self.final_good = torch.zeros_like(self.elapsed)
        self.final_yaw_error = torch.zeros_like(self.elapsed)
        self.steps = torch.zeros_like(self.elapsed)
        self.window_count = 0
        self.window_success = 0
        self.last_success_rate = 0.0
        for key in ("FL_contact_ratio", "HR_contact_ratio", "FR_unwanted_contact_ratio",
                    "HL_unwanted_contact_ratio", "pose_reference_error", "yaw_tracking_error",
                    "roll", "pitch", "lambda", "actual_yaw_rate", "yaw_rate_cmd"):
            self.metrics[key] = torch.zeros_like(self.elapsed)

    @property
    def command(self):
        return self._command

    @property
    def reference(self):
        return pose_reference(self.standing, self.target, self.command[:, 1])

    def contacts(self):
        return (self.sensor.data.net_forces_w[:, self.contact_ids].norm(dim=-1) > self.cfg.contact_threshold).float()

    def _resample_command(self, env_ids):
        self.episode_level[env_ids] = self.level
        self.elapsed[env_ids] = 0.0
        self._command[env_ids] = 0.0
        magnitude = self.yaw_limits[self.episode_level[env_ids]]
        # Nonzero commands at rotation levels make tracking success meaningful.
        random = torch.rand(len(env_ids), device=self.device)
        sign = torch.where(torch.rand_like(random) < 0.5, -1.0, 1.0)
        self.sampled_yaw[env_ids] = sign * magnitude * (0.25 + 0.75 * random)

    def _update_command(self):
        self.elapsed += self._env.step_dt
        phase = ((self.elapsed - self.cfg.standing_time_s) / self.cfg.transition_duration_s).clamp(0, 1)
        self._command[:, 1] = phase
        self._command[:, 0] = self.sampled_yaw * smoothstep((phase - 0.8) / 0.2)

    def _update_metrics(self):
        contact = self.contacts()
        phase = self.command[:, 1]
        final = phase >= 1.0
        roll, pitch, _ = euler_xyz_from_quat(self.robot.data.root_quat_w)
        roll, pitch = wrap_to_pi(roll), wrap_to_pi(pitch)
        good = (contact[:, 0] > 0) & (contact[:, 3] > 0) & (contact[:, 1] == 0) & (contact[:, 2] == 0)
        good &= (roll.abs() < 0.5) & (pitch.abs() < 0.5)
        self.hold = torch.where(final & good, self.hold + self._env.step_dt, 0.0)
        self.best_hold = torch.maximum(self.best_hold, self.hold)
        yaw_error = (self.robot.data.root_ang_vel_b[:, 2] - self.command[:, 0]).abs()
        self.final_steps += final
        self.final_good += final & good
        self.final_yaw_error += final * yaw_error
        self.steps += 1
        values = {
            "FL_contact_ratio": contact[:, 0], "HR_contact_ratio": contact[:, 3],
            "FR_unwanted_contact_ratio": final * contact[:, 1],
            "HL_unwanted_contact_ratio": final * contact[:, 2],
            "pose_reference_error": (self.robot.data.joint_pos[:, self.joint_ids] - self.reference).square().sum(-1),
            "yaw_tracking_error": yaw_error, "roll": roll, "pitch": pitch, "lambda": phase,
            "actual_yaw_rate": self.robot.data.root_ang_vel_b[:, 2], "yaw_rate_cmd": self.command[:, 0],
        }
        for key, value in values.items():
            self.metrics[key] += value

    def successful(self, ids):
        level = self.episode_level[ids]
        good = self.best_hold[ids] >= self.required_hold[level]
        contact_ok = self.final_good[ids] / self.final_steps[ids].clamp_min(1) >= self.cfg.final_contact_success_ratio
        good &= (level == 0) | contact_ok
        yaw_ok = self.final_yaw_error[ids] / self.final_steps[ids].clamp_min(1) < self.cfg.yaw_success_error
        good &= (level < 2) | (yaw_ok & (self.sampled_yaw[ids].abs() >= 0.049))
        return good & ~self._env.termination_manager.terminated[ids]

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None or isinstance(env_ids, slice) else env_ids
        extra = {}
        for key, value in self.metrics.items():
            denominator = self.final_steps if "unwanted" in key else self.steps
            extra[key] = (value[ids] / denominator[ids].clamp_min(1)).mean().item()
            value[ids] = 0
        extra["transition_success_rate"] = (self.best_hold[ids] >= self.cfg.hold_success_s[0]).float().mean().item()
        extra["two_wheel_hold_duration"] = self.best_hold[ids].mean().item()
        extra["final_phase_contact_ratio"] = (self.final_good[ids] / self.final_steps[ids].clamp_min(1)).mean().item()
        extra["final_phase_yaw_error"] = (self.final_yaw_error[ids] / self.final_steps[ids].clamp_min(1)).mean().item()
        extra["task_success_rate"] = self.successful(ids).float().mean().item()
        extra["fall_rate"] = self._env.termination_manager.terminated[ids].float().mean().item()
        extra["curriculum_level"] = float(self.level)
        for buffer in (self.steps, self.hold, self.best_hold, self.final_steps, self.final_good, self.final_yaw_error):
            buffer[ids] = 0
        self.command_counter[ids] = 0
        self._resample(ids)
        return extra


@configclass
class TOPivotCommandCfg(CommandTermCfg):
    class_type: type = TOPivotCommand
    asset_name: str = "robot"
    sensor_name: str = "contact_forces"
    leg_joint_names: list[str] = MISSING
    wheel_body_names: list[str] = MISSING
    standing_positions: dict = MISSING
    reference_path: str = REFERENCE_PATH
    resampling_time_range: tuple = (1.0e6, 1.0e6)
    standing_time_s: float = 0.5
    transition_duration_s: float = 2.5
    initial_level: int = 0
    pose_weights: tuple = (3.0, 2.0, 1.0, 0.4, 0.15)
    yaw_limits: tuple = (0.0, 0.0, 0.2, 0.5, 1.0)
    hold_success_s: tuple = (0.3, 2.0, 3.0, 3.0, 4.0)
    contact_threshold: float = 1.0
    final_contact_success_ratio: float = 0.7
    yaw_success_error: float = 0.15
    # New: explicitly freeze curriculum advancement when requested by a run.
    freeze_level: bool = False


def transition_levels(env, env_ids, command_name="pivot", min_episodes=256, success_rate=0.8):
    term = env.command_manager.get_term(command_name)
    ids = torch.arange(env.num_envs, device=env.device) if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=env.device)
    ids = ids[(term.steps[ids] > 0) & (term.episode_level[ids] == term.level)]
    term.window_count += len(ids)
    term.window_success += int(term.successful(ids).sum().item())
    # Old behavior (kept here as a reference): every successful window could
    # advance the global level automatically.
    # if term.window_count >= min_episodes:
    #     term.last_success_rate = term.window_success / term.window_count
    #     if term.last_success_rate >= success_rate:
    #         term.level = min(term.level + 1, 4)
    #     term.window_count = term.window_success = 0

    # New behavior: an explicit freeze keeps the configured level fixed while
    # still reporting the observed window success rate for diagnostics. The
    # fallback keeps older externally-created command configs runnable.
    if term.window_count >= min_episodes:
        term.last_success_rate = term.window_success / term.window_count
        freeze_level = getattr(term.cfg, "freeze_level", False)
        if not freeze_level and term.last_success_rate >= success_rate:
            term.level = min(term.level + 1, 4)
        term.window_count = term.window_success = 0
    return {"level": float(term.level), "window_success_rate": term.last_success_rate}


def to_pose_reward(env, k_pose=0.8):
    term = env.command_manager.get_term("pivot")
    error = (term.robot.data.joint_pos[:, term.joint_ids] - term.reference).square().sum(-1)
    return term.weights[term.episode_level] * torch.exp(-k_pose * error)


def scheduled_contact_reward(env):
    term = env.command_manager.get_term("pivot")

    contact = term.contacts()
    phase = term.command[:, 1]
    unload = unload_fraction(phase)

    # FL, FR, HL, HR
    four_wheel = contact.mean(dim=-1)

    # Desired final support: FL + HR
    diagonal = 0.5 * (
        contact[:, 0]
        + contact[:, 3]
        - contact[:, 1]
        - contact[:, 2]
    )

    return (1.0 - unload) * four_wheel + unload * diagonal

def transition_yaw_reward(env, k_yaw=11.11):
    term = env.command_manager.get_term("pivot")
    error = term.robot.data.root_ang_vel_b[:, 2] - term.command[:, 0]
    return smoothstep((term.command[:, 1] - 0.8) / 0.2) * torch.exp(-k_yaw * error.square())


def reset_transition_standing(
    env, env_ids, asset_cfg, nominal_joint_positions, leg_joint_count, root_height,
    joint_position_noise, leg_velocity_noise, wheel_velocity_noise, roll_noise,
    pitch_noise, yaw_range, angular_velocity_noise, root_xy_noise=(-0.01, 0.01),
):
    """Same normal standing reset; larger but modest noise only at level 3+."""
    kwargs = dict(asset_cfg=asset_cfg, nominal_joint_positions=nominal_joint_positions,
                  leg_joint_count=leg_joint_count, root_height=root_height,
                  joint_position_noise=joint_position_noise, leg_velocity_noise=leg_velocity_noise,
                  wheel_velocity_noise=wheel_velocity_noise, roll_noise=roll_noise,
                  pitch_noise=pitch_noise, yaw_range=yaw_range,
                  angular_velocity_noise=angular_velocity_noise, root_xy_noise=root_xy_noise)
    level = env.command_manager.get_term("pivot").level
    if level >= 3:
        kwargs.update(joint_position_noise=(-0.035, 0.035), leg_velocity_noise=(-0.10, 0.10),
                      angular_velocity_noise=(-0.15, 0.15), roll_noise=(-0.025, 0.025), pitch_noise=(-0.025, 0.025))
    reset_four_wheel_standing(env, env_ids, **kwargs)


def transition_push(env, env_ids, max_velocity=0.15):
    term = env.command_manager.get_term("pivot")
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    ids = ids[(term.episode_level[ids] >= 3) & (term.command[ids, 1] >= 1)]
    velocity = term.robot.data.root_vel_w[ids].clone()
    velocity[:, :2] += torch.empty(len(ids), 2, device=env.device).uniform_(-max_velocity, max_velocity)
    if len(ids):
        term.robot.write_root_velocity_to_sim(velocity, env_ids=ids)
