"""Environment adapter for the isolated POS skill state machine."""

from __future__ import annotations

import math
from fractions import Fraction

import torch

from .yaw_pos_skill_state import EPISODE_GATES, WINDOW_GATES, SkillSettings, SkillState
from .yaw_pos_skill_noise import SkillNoiseModel
from .yaw_pos_skill_rewards import SUPPORT_COVERAGE_METRICS


def get_state(env) -> SkillState:
    if not hasattr(env, "_yaw_pos_skill_state"):
        env._yaw_pos_skill_state = SkillState(stage_start_step=int(env.common_step_counter))
    return env._yaw_pos_skill_state


def promotion_reset(env) -> torch.Tensor:
    pending = get_state(env).pending is not None
    return torch.full((env.num_envs,), pending, device=env.device, dtype=torch.bool)


def _observations_noise_scale(env, scale: float):
    manager = env.observation_manager
    found = set()
    for name, cfg in zip(manager._group_obs_term_names["policy"], manager._group_obs_term_cfgs["policy"]):
        model = getattr(cfg.noise, "func", None) if cfg.noise is not None else None
        if isinstance(model, SkillNoiseModel):
            model.set_scale(scale)
            found.add(name)
    expected = {"base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"}
    if found != expected:
        raise RuntimeError(f"Skill noise terms differ from config: {found} != {expected}")


def apply_difficulty(env):
    state = get_state(env)
    scale = state.dr_scale
    command = env.command_manager.get_term("yaw_rate_cmd")
    command.cfg.yaw_rate_range = (0.0, state.settings.yaw_levels[state.yaw_stage])
    for reward_name in ("lift_clearance", "gated_yaw_tracking", "neutral_landing_progress"):
        cfg = env.reward_manager.get_term_cfg(reward_name)
        cfg.params["target_clearance"] = state.settings.clearance
        env.reward_manager.set_term_cfg(reward_name, cfg)
    force = env.event_manager.get_term_cfg("randomize_apply_external_force_torque")
    force.params["force_range"] = (-10 * scale, 10 * scale)
    force.params["torque_range"] = (-10 * scale, 10 * scale)
    env.event_manager.set_term_cfg("randomize_apply_external_force_torque", force)
    gains = env.event_manager.get_term_cfg("randomize_actuator_gains")
    gains.params["stiffness_distribution_params"] = (1 - .15 * scale, 1 + .15 * scale)
    gains.params["damping_distribution_params"] = (1 - .15 * scale, 1 + .15 * scale)
    env.event_manager.set_term_cfg("randomize_actuator_gains", gains)
    push = env.event_manager.get_term_cfg("randomize_push_robot")
    push.params["velocity_range"] = {"x": (-.15 * scale, .15 * scale), "y": (-.15 * scale, .15 * scale)}
    env.event_manager.set_term_cfg("randomize_push_robot", push)
    reset = env.event_manager.get_term_cfg("randomize_reset_base")
    reset.params["pose_range"].update({"roll": (-.3 * scale, .3 * scale),
                                      "pitch": (-.3 * scale, .3 * scale)})
    reset.params["velocity_range"].update({
        "x": (-.2 * scale, .2 * scale), "y": (-.2 * scale, .2 * scale),
        "z": (-.2 * scale, .2 * scale), "roll": (-.05 * scale, .05 * scale),
        "pitch": (-.05 * scale, .05 * scale), "yaw": (0.0, 0.0),
    })
    env.event_manager.set_term_cfg("randomize_reset_base", reset)
    _observations_noise_scale(env, scale)


def _values(env, ids):
    n = len(ids)
    def sample(name, default=0.0):
        value = getattr(env, name, None)
        return value[ids] if value is not None else torch.full((n,), default, device=env.device)

    def average(base):
        counts = sample(base + "_samples")
        totals = sample(base + "_sum")
        return torch.where(counts > 0, totals / counts.clamp_min(1), torch.nan)

    duration = env.episode_length_buf[ids].float() * env.step_dt
    balance_cfg = env.reward_manager.get_term_cfg("balance")
    if balance_cfg.weight <= 0:
        raise ValueError("Skill balance reward weight must be positive.")
    balance = env.reward_manager._episode_sums["balance"][ids] / (duration * balance_cfg.weight)
    diff_samples = sample("_yaw_pos_differential_pass_samples")
    diff = torch.where(diff_samples > 0, sample("_yaw_pos_differential_pass_sum") / diff_samples.clamp_min(1), torch.nan)
    values = {
        "support": average("_yaw_support_score"), "lift": average("_yaw_lift_min_progress"),
        "balance": balance, "height": sample("_yaw_base_height_min", math.inf),
        "safe": (~env.termination_manager.terminated[ids]).float(),
        "yaw": average("_yaw_active_yaw_score"), "differential_episode": diff,
    }
    metrics = {
        "command": (sample("_yaw_command_abs_sum"), sample("_yaw_tracking_metric_samples")),
        "error": (sample("_yaw_rate_abs_error_sum"), sample("_yaw_tracking_metric_samples")),
        "edge_command": (sample("_yaw_edge_command_abs_sum"), sample("_yaw_edge_tracking_samples")),
        "edge_error": (sample("_yaw_edge_rate_abs_error_sum"), sample("_yaw_edge_tracking_samples")),
        "differential": (sample("_yaw_pos_differential_pass_sum"), diff_samples),
        "neutral": (sample("_yaw_pos_neutral_hold_pass_sum"), sample("_yaw_pos_neutral_hold_pass_samples")),
        "four_contact": (sample("_yaw_pos_neutral_four_contact_sum"), sample("_yaw_pos_neutral_samples")),
        "anchor_coverage": (sample("_yaw_pos_skill_anchor_sum"), sample("_yaw_pos_skill_neutral_samples")),
    }
    legacy_count = sample("_yaw_pos_differential_legacy_samples")
    eligible_pair = (diff_samples > 0) & (legacy_count > 0)
    certificate = Fraction(str(get_state(env).settings.certificate))
    fixed_pass = (sample("_yaw_pos_differential_pass_sum").long() * certificate.denominator
                  >= diff_samples.long() * certificate.numerator)
    legacy_pass = (sample("_yaw_pos_differential_legacy_sum").long() * certificate.denominator
                   >= legacy_count.long() * certificate.numerator)
    metrics["differential_fixed_episode"] = (
        (fixed_pass & eligible_pair).float(), eligible_pair.long()
    )
    metrics["differential_legacy_episode"] = (
        (legacy_pass & eligible_pair).float(), eligible_pair.long()
    )
    for metric in SUPPORT_COVERAGE_METRICS:
        base = f"_yaw_pos_skill_{metric}"
        counts = sample(base + "_samples")
        metrics[metric + "_mean"] = (sample(base + "_sum"), counts)
        metrics[metric + "_min"] = (
            torch.where(counts > 0, sample(base + "_min"), 0.), (counts > 0).long()
        )
    for metric in ("differential_fl_signed_ratio", "differential_hr_signed_ratio",
                   "differential_any_wrong_sign_pct", "neutral_position_drift", "neutral_planar_speed",
                   "differential_legacy", "differential_settled", "differential_legacy_settled",
                   "active_command_change", "differential_fail_invalid", "differential_fail_contact",
                   "differential_fail_sign", "differential_fail_ground_speed",
                   "differential_fail_motor_speed", "differential_fail_residual"):
        metrics[metric] = (sample(f"_yaw_pos_{metric}_sum"), sample(f"_yaw_pos_{metric}_samples"))
    return values, metrics, sample("_yaw_tracking_metric_samples") > 0


def _clear_episode_buffers(env, ids):
    for name, value in vars(env).items():
        if (name.startswith("_yaw_") and (name.endswith("_sum") or name.endswith("_samples"))
                and isinstance(value, torch.Tensor) and value.ndim >= 1 and len(value) == env.num_envs):
            value[ids] = 0
    if hasattr(env, "_yaw_base_height_min"):
        env._yaw_base_height_min[ids] = torch.inf
    for metric in SUPPORT_COVERAGE_METRICS:
        name = f"_yaw_pos_skill_{metric}_min"
        if hasattr(env, name):
            getattr(env, name)[ids] = torch.inf
    for name in ("_yaw_pos_active_age", "_yaw_pos_legacy_active_age", "_yaw_pos_neutral_contact_age",
                 "_yaw_pos_neutral_anchored", "_yaw_pos_neutral_anchor"):
        if hasattr(env, name):
            getattr(env, name)[ids] = 0


def skill_task_levels(env, env_ids) -> dict[str, torch.Tensor]:
    state = get_state(env)
    ids = torch.arange(env.num_envs, device=env.device) if isinstance(env_ids, slice) else torch.as_tensor(
        env_ids, dtype=torch.long, device=env.device)
    step = int(env.common_step_counter)
    if state.pending is not None:
        if len(ids) != env.num_envs:
            raise RuntimeError("Skill phase boundary must reset every environment.")
        state.commit(step)
    else:
        completed = ids[env.episode_length_buf[ids] > 0]
        if len(completed):
            values, metrics, active = _values(env, completed)
            active = active.detach().cpu().tolist()
            values = {k: v.detach().cpu().tolist() for k, v in values.items()}
            metrics = {k: (v[0].detach().cpu().tolist(), v[1].detach().cpu().tolist())
                       for k, v in metrics.items()}
            for row in range(len(completed)):
                state.record({k: value[row] for k, value in values.items()},
                             {k: (value[0][row], value[1][row]) for k, value in metrics.items()},
                             bool(active[row]))
            state.evaluate(step)
    _clear_episode_buffers(env, ids)
    apply_difficulty(env)
    def scalar(x):
        return torch.tensor(float(x), device=env.device)
    result = {"skill/phase": scalar(state.phase), "skill/clearance_stage": scalar(0),
              "skill/yaw_stage": scalar(state.yaw_stage),
              "skill/yaw_limit": scalar(state.settings.yaw_levels[state.yaw_stage]),
              "skill/robustness_stage": scalar(state.robustness_stage),
              "skill/robustness_scale": scalar(state.dr_scale),
              "skill/pending_transition": scalar(state.pending is not None),
              "skill/window_episodes": scalar(state.window["episodes"]),
              "skill/consecutive_passes": scalar(state.consecutive_passes)}
    for gate, enabled in state.enabled().items():
        result[f"skill/gate_enabled/{gate}"] = scalar(enabled)
        result[f"skill/gate_threshold/{gate}"] = scalar(state.thresholds()[gate])
        result[f"skill/blocker/{gate}"] = scalar(gate in state.last.get("blockers", ()))
    for name in ("joint_episode_success", "consecutive_windows", "minimum_duration", "insufficient_episodes"):
        blocked = name in state.last.get("blockers", ()) or (name == "insufficient_episodes" and not state.last)
        result[f"skill/blocker/{name}"] = scalar(blocked)
    for name, value in state.last.get("rates", {}).items():
        result[f"skill/last_window/{name}"] = scalar(value)
    for name, value in state.last.get("gate_rates", {}).items():
        result[f"skill/last_window/gate_pass_rate/{name}"] = scalar(value)
    for name, value in state.last.get("counts", {}).items():
        result[f"skill/last_window/samples/{name}"] = scalar(value)
    return result
