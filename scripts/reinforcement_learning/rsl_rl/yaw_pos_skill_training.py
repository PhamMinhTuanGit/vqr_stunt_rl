"""Scratch-only runner hooks for the isolated POS skill experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from rl_training.tasks.manager_based.locomotion.velocity.mdp.yaw_pos_skill_state import (
    CHECKPOINT_KEY, EPISODE_GATES, WINDOW_GATES, SkillSettings, SkillState,
)


def action_names(task_env):
    manager = task_env.action_manager
    if manager.active_terms.count("joint_pos") != 1 or manager.active_terms.count("joint_vel") != 1:
        raise RuntimeError("Skill experiment requires one leg and one wheel action term.")
    names = [joint for name in manager.active_terms for joint in manager.get_term(name)._joint_names]
    legs = {f"{side}_{joint}" for side in ("FL", "FR", "HL", "HR")
            for joint in ("HipX_joint", "HipY_joint", "Knee_joint")}
    wheels = {f"{side}_WHEEL" for side in ("FL", "FR", "HL", "HR")}
    if len(names) != 16 or len(set(names)) != 16 or set(names) != legs | wheels:
        raise RuntimeError(f"Unexpected POS skill action order: {names}")
    return names


def initialize_wheel_std(policy, task_env, target=0.15):
    names = action_names(task_env)
    if (not isinstance(getattr(policy, "log_std", None), torch.nn.Parameter)
            or getattr(policy, "noise_std_type", None) != "log"
            or getattr(policy, "state_dependent_std", False)):
        raise RuntimeError("Skill policy requires independent learnable log_std.")
    if not math.isfinite(target) or target <= 0 or policy.log_std.numel() != len(names):
        raise ValueError("Invalid wheel std or action dimensions.")
    leg_indices = [i for i, name in enumerate(names) if not name.endswith("_WHEEL")]
    if not torch.allclose(policy.log_std.detach()[leg_indices].exp(),
                          torch.ones(len(leg_indices), device=policy.log_std.device), atol=1.e-6):
        raise RuntimeError("Fresh POS skill legs must start at baseline std 1.0.")
    with torch.no_grad():
        for index, name in enumerate(names):
            if name.endswith("_WHEEL"):
                policy.log_std[index] = math.log(target)
    policy.distribution = None
    return dict(zip(names, policy.log_std.detach().exp().cpu().tolist()))


def export_skill_state(task_env):
    from rl_training.tasks.manager_based.locomotion.velocity.mdp.yaw_pos_skill_curriculums import get_state
    return get_state(task_env).export(int(task_env.common_step_counter))


def restore_skill_state(task_env, payload):
    """Explicit utility for checkpoint validation; fresh training never calls it."""
    from rl_training.tasks.manager_based.locomotion.velocity.mdp.yaw_pos_skill_curriculums import apply_difficulty
    next_state = SkillState.restore(payload, SkillSettings(), int(task_env.common_step_counter))
    task_env._yaw_pos_skill_state = next_state
    apply_difficulty(task_env)
    return next_state


def install_skill_hooks(runner, task_env, log_dir):
    from rl_training.tasks.manager_based.locomotion.velocity.mdp.yaw_pos_skill_curriculums import get_state, apply_difficulty
    state = get_state(task_env)
    apply_difficulty(task_env)
    if runner.current_learning_iteration != 0:
        raise RuntimeError("POS skill experiment must start at iteration zero.")
    dimensions = {name: list(shape) for name, shape in task_env.observation_manager.group_obs_dim.items()}
    if dimensions != {"policy": [55], "critic": [83]}:
        raise RuntimeError(f"POS skill observation dimensions changed: {dimensions}")
    std = initialize_wheel_std(runner.alg.policy, task_env)
    initial = {
        "task": "Flat-VQR-Wheel-Yaw-POS-Skill", "starting_iteration": 0,
        "phase": state.phase, "clearance_stage": 0, "yaw_stage": state.yaw_stage,
        "yaw_limit": state.settings.yaw_levels[state.yaw_stage], "robustness_scale": state.dr_scale,
        "action_std": std, "observations": dimensions,
        "gate_enabled": state.enabled(), "gate_threshold": state.thresholds(),
    }
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    (Path(log_dir) / "skill_prestart.json").write_text(json.dumps(initial, indent=2) + "\n")
    print("[PRESTART] POS skill: " + json.dumps(initial, sort_keys=True))

    original_save = runner.save
    def save_with_skill(path, infos=None):
        data = dict(infos) if isinstance(infos, dict) else {}
        if infos is not None and not isinstance(infos, dict):
            data["runner_infos"] = infos
        data[CHECKPOINT_KEY] = export_skill_state(task_env)
        original_save(path, data)
    runner.save = save_with_skill

    original_log = runner.log
    names = action_names(task_env)
    def log_with_skill(locs, *args, **kwargs):
        result = original_log(locs, *args, **kwargs)
        step = locs["it"]
        writer = runner.writer
        state = get_state(task_env)
        scalar = writer.add_scalar
        scalar("Curriculum/skill/current_phase", state.phase, step)
        scalar("Curriculum/skill/clearance_stage", 0, step)
        scalar("Curriculum/skill/yaw_stage", state.yaw_stage, step)
        scalar("Curriculum/skill/yaw_limit", state.settings.yaw_levels[state.yaw_stage], step)
        scalar("Curriculum/skill/robustness_stage", state.robustness_stage, step)
        scalar("Curriculum/skill/robustness_scale", state.dr_scale, step)
        scalar("Curriculum/skill/pending_transition", int(state.pending is not None), step)
        scalar("Curriculum/skill/complete", int(state.complete), step)
        scalar("Curriculum/skill/window_active_episodes", state.window["episodes"], step)
        scalar("Curriculum/skill/consecutive_windows", state.consecutive_passes, step)
        scalar("Curriculum/skill/last_window_phase", state.last.get("phase", state.phase), step)
        scalar("Curriculum/skill/last_window_active_episodes", state.last.get("episodes", 0), step)
        stamp = getattr(task_env, "_yaw_pos_skill_applied_physics_stage", None)
        if stamp is None or not bool((stamp == state.robustness_stage).all()):
            raise RuntimeError("Physical DR stage has not been applied to every environment.")
        lags = [actuator.positions_delay_buffer.time_lags
                for actuator in task_env.scene["robot"].actuators.values()]
        actual_min = min(int(lag.min().item()) for lag in lags)
        actual_max = max(int(lag.max().item()) for lag in lags)
        permitted_max = 2 + math.floor(6 * state.dr_scale)
        if actual_min < 2 or actual_max > permitted_max:
            raise RuntimeError(f"Actual actuator delay {actual_min}–{actual_max} exceeds [2,{permitted_max}].")
        scalar("Curriculum/skill/actual_delay_min_steps", actual_min, step)
        scalar("Curriculum/skill/actual_delay_max_steps", actual_max, step)
        rates = state.rates() if state.window["episodes"] else state.last.get("rates", {})
        gate_rates = ({name: state.window["passes"][name] / state.window["episodes"]
                       for name in EPISODE_GATES} if state.window["episodes"] else state.last.get("gate_rates", {}))
        for name in EPISODE_GATES + WINDOW_GATES:
            scalar(f"Curriculum/skill/gate_enabled/{name}", int(state.enabled()[name]), step)
            scalar(f"Curriculum/skill/gate_threshold/{name}", state.thresholds()[name], step)
            scalar(f"Curriculum/skill/blocker/{name}", int(name in state.last.get("blockers", ())), step)
            scalar(f"Curriculum/skill/gate_pass_rate/{name}",
                   gate_rates.get(name, rates.get(name, 0.0)), step)
        for name in ("joint_episode_success", "consecutive_windows", "minimum_duration"):
            scalar(f"Curriculum/skill/blocker/{name}", int(name in state.last.get("blockers", ())), step)
        scalar("Curriculum/skill/blocker/insufficient_episodes",
               int(state.window["episodes"] < state.settings.min_episodes), step)
        for name in ("support_score", "lift_score", "balance_score", "height_score", "yaw_score",
                     "tracking", "edge_tracking",
                     "differential", "neutral", "four_contact", "anchor_coverage",
                     "differential_fl_signed_ratio", "differential_hr_signed_ratio",
                     "differential_any_wrong_sign_pct", "neutral_position_drift", "neutral_planar_speed"):
            scalar(f"Curriculum/skill/metric/{name}",
                   gate_rates.get(name, rates.get(name, 0.0)), step)
        current_std = runner.alg.policy.log_std.detach().exp().cpu().tolist()
        for name, value in zip(names, current_std):
            scalar(f"Policy/action_std/{name}", value, step)
        return result
    runner.log = log_with_skill
    return initial
