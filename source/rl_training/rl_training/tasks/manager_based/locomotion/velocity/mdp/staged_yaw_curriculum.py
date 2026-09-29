"""Independent, episode-based curriculum for the flat staged yaw cycle."""

from __future__ import annotations

from collections.abc import Sequence

import torch

__all__ = [
    "staged_complete", "staged_failure_cost", "staged_four_contact",
    "staged_four_dwell_bonus", "staged_milestone_bonus", "StagedPromotion",
    "staged_promotion_reset", "staged_yaw_task_levels",
]


def staged_complete(env, command_name: str = "yaw_rate_cmd") -> torch.Tensor:
    return env.command_manager.get_term(command_name).staged_complete


def staged_failure_cost(env) -> torch.Tensor:
    failed = _staged_failure_mask(env)
    return failed.float() / env.step_dt


def _staged_failure_mask(env) -> torch.Tensor:
    """One failure for safety or an incomplete maneuver timeout."""
    done = env.termination_manager
    phase = env.command_manager.get_term("yaw_rate_cmd").episode_phase
    complete = done.get_term("staged_complete")
    names = getattr(done, "active_terms", None)
    if names is None:
        safety = done.terminated & ~complete
    else:
        safety = torch.zeros_like(done.terminated)
        for name in names:
            if name not in ("staged_complete", "staged_promotion_reset", "time_out"):
                safety |= done.get_term(name)
    incomplete_timeout = done.get_term("time_out") & (phase > 0) & ~complete
    return safety | incomplete_timeout


def staged_promotion_reset(env) -> torch.Tensor:
    """End all old-stage episodes before the promoted global settings apply."""
    pending = bool(getattr(env, "_staged_yaw_promotion_reset_pending", False))
    return torch.full((env.num_envs,), pending, device=env.device, dtype=torch.bool)


def staged_four_contact(env, command_name: str = "yaw_rate_cmd") -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    return ((command.fsm_state == 0) & command.four_stand_ready).float() * command.four_reward_gate


def staged_four_dwell_bonus(env, command_name: str = "yaw_rate_cmd") -> torch.Tensor:
    return env.command_manager.get_term(command_name).cycle.just_four_dwell.float() / env.step_dt


def staged_milestone_bonus(env, command_name: str = "yaw_rate_cmd") -> torch.Tensor:
    valid = env.command_manager.get_term(command_name).staged_complete
    valid = valid & ~_staged_failure_mask(env)
    valid = valid & ~env.termination_manager.get_term("staged_promotion_reset")
    return valid.float() / env.step_dt


_FIELDS = (
    "attempts", "four_hold", "entry", "yaw", "return", "cycle", "support", "pose",
    "command_sum", "error_sum", "yaw_samples", "drift_sum", "drift_samples",
)


def _empty_counts() -> dict[str, int | float]:
    return {name: 0.0 if name.endswith("sum") else 0 for name in _FIELDS}


class StagedPromotion:
    """Window accumulator with independent POS and NEG certification."""

    def __init__(self, phase: int = 0, clearance_index: int = 0, yaw_index: int = 0):
        self.phase = phase
        self.clearance_index = clearance_index
        self.yaw_index = yaw_index
        self.consecutive_passes = 0
        self.window = {"pos": _empty_counts(), "neg": _empty_counts()}
        self.last_counts = {"pos": _empty_counts(), "neg": _empty_counts()}
        self.last_rates = {"pos": {}, "neg": {}, "all": {}}
        self.last_window_passed = False
        self.last_window_phase = phase

    def record(self, direction: str, **values: int | float) -> None:
        if direction not in self.window:
            raise ValueError(f"Unknown direction {direction!r}.")
        row = self.window[direction]
        row["attempts"] += 1
        for name, value in values.items():
            if name not in row or name == "attempts":
                raise ValueError(f"Unknown staged outcome {name!r}.")
            row[name] += value

    def ready(self, hold_window: int, directional_window: int) -> bool:
        rows = self.window
        if self.phase == 0:
            return sum(row["attempts"] for row in rows.values()) >= hold_window
        return all(row["attempts"] >= directional_window for row in rows.values())

    @staticmethod
    def rates(row: dict[str, int | float]) -> dict[str, float]:
        attempts = max(row["attempts"], 1)
        command = row["command_sum"]
        return {
            "four_hold_success": row["four_hold"] / attempts,
            "four_to_transition_success": row["entry"] / attempts,
            "transition_to_yaw_success": row["yaw"] / attempts,
            "return_to_four_success": row["return"] / attempts,
            "full_cycle_success": row["cycle"] / attempts,
            "support_ready_rate": row["support"] / attempts,
            "pose_ready_rate": row["pose"] / attempts,
            "tracking_ratio": 1.0 - row["error_sum"] / command if command > 1.0e-6 else 0.0,
            "mean_yaw_drift": row["drift_sum"] / max(row["drift_samples"], 1),
        }

    def evaluate(
        self, clearance_levels: Sequence[float], yaw_levels: Sequence[float],
        tracking_thresholds: Sequence[float], hold_window: int = 2048,
        directional_window: int = 1024, required_windows: int = 3,
    ) -> bool:
        if not self.ready(hold_window, directional_window):
            return False
        rates = {direction: self.rates(row) for direction, row in self.window.items()}
        self.last_window_phase = self.phase
        self.last_counts = {direction: row.copy() for direction, row in self.window.items()}
        totals = {name: sum(row[name] for row in self.window.values()) for name in _FIELDS}
        rates["all"] = self.rates(totals)
        self.last_rates = rates
        if self.phase == 0:
            passed = rates["all"]["four_hold_success"] >= 0.90
        elif self.phase == 1:
            passed = all(rates[d]["four_to_transition_success"] >= 0.85 for d in ("pos", "neg"))
        elif self.phase == 2:
            passed = all(
                rates[d]["transition_to_yaw_success"] >= 0.85
                and rates[d]["support_ready_rate"] >= 0.85
                and rates[d]["pose_ready_rate"] >= 0.80
                for d in ("pos", "neg")
            )
        else:
            passed = all(
                rates[d]["full_cycle_success"] >= 0.85
                and rates[d]["tracking_ratio"] >= tracking_thresholds[self.yaw_index]
                and rates[d]["mean_yaw_drift"] <= 0.08
                for d in ("pos", "neg")
            )
        self.last_window_passed = passed
        self.consecutive_passes = min(self.consecutive_passes + 1, required_windows) if passed else 0
        promoted = False
        if self.consecutive_passes >= required_windows:
            if self.phase == 2 and self.clearance_index + 1 < len(clearance_levels):
                self.clearance_index += 1
                promoted = True
            elif self.phase < 3:
                self.phase += 1
                promoted = True
            elif self.yaw_index + 1 < len(yaw_levels):
                self.yaw_index += 1
                promoted = True
            if promoted:
                self.consecutive_passes = 0
        self.window = {"pos": _empty_counts(), "neg": _empty_counts()}
        return promoted

    def export(self) -> dict:
        return {
            "phase": self.phase, "clearance_index": self.clearance_index,
            "yaw_index": self.yaw_index, "consecutive_passes": self.consecutive_passes,
            "window": self.window, "last_rates": self.last_rates,
            "last_counts": self.last_counts,
            "last_window_passed": self.last_window_passed,
            "last_window_phase": self.last_window_phase,
        }

    @classmethod
    def restore(cls, data: dict) -> "StagedPromotion":
        state = cls(int(data["phase"]), int(data["clearance_index"]), int(data["yaw_index"]))
        state.consecutive_passes = int(data["consecutive_passes"])
        state.window = {
            direction: {field: data["window"][direction][field] for field in _FIELDS}
            for direction in ("pos", "neg")
        }
        state.last_rates = data.get("last_rates", state.last_rates)
        state.last_counts = data.get("last_counts", state.last_counts)
        state.last_window_passed = bool(data.get("last_window_passed", False))
        state.last_window_phase = int(data.get("last_window_phase", state.phase))
        return state


def staged_yaw_task_levels(
    env, env_ids: Sequence[int], command_name: str, clearance_levels: Sequence[float],
    yaw_rate_levels: Sequence[float], dr_scale_levels: Sequence[float],
    tracking_ratio_thresholds: Sequence[float], hold_window: int = 2048,
    directional_window: int = 1024, required_windows: int = 3,
) -> dict[str, torch.Tensor]:
    if not (len(yaw_rate_levels) == len(dr_scale_levels) == len(tracking_ratio_thresholds)):
        raise ValueError("Yaw, DR, and tracking tables must have equal length.")
    if not clearance_levels or any(x <= 0 for x in clearance_levels):
        raise ValueError("Clearance levels must be positive.")
    state = getattr(env, "_staged_yaw_promotion", None)
    if state is None:
        state = StagedPromotion()
        env._staged_yaw_promotion = state
    ids = (torch.arange(env.num_envs, device=env.device) if isinstance(env_ids, slice)
           else torch.as_tensor(env_ids, device=env.device, dtype=torch.long))
    completed = ids[env.episode_length_buf[ids] > 0]
    command = env.command_manager.get_term(command_name)
    pending_reset = bool(getattr(env, "_staged_yaw_promotion_reset_pending", False))
    if pending_reset:
        # The promotion termination makes the next _reset_idx contain every
        # environment. Its episodes were deliberately cut short, so they do
        # not enter the new stage's evaluation window.
        all_ids = torch.arange(env.num_envs, device=env.device)
        if ids.numel() != env.num_envs or not torch.equal(torch.sort(ids).values, all_ids):
            raise RuntimeError("Staged promotion reset must include all environments.")
        completed = completed[:0]
        env._staged_yaw_promotion_reset_pending = False
    if completed.numel():
        # Command computation follows termination/reward computation in Isaac
        # Lab. Refresh the live predicates before certifying a terminal frame.
        refresh = getattr(command, "_update_fsm_predicates", None)
        if refresh is not None:
            refresh()
        current_unsafe = getattr(command, "unsafe", torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
        current_four_ready = getattr(
            command, "four_stand_ready", torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        )
        done = env.termination_manager
        failed = _staged_failure_mask(env)
        cycle = command.cycle
        episode_phase = getattr(command, "episode_phase", None)
        episode_clearance_index = getattr(command, "episode_clearance_index", None)
        episode_yaw_index = getattr(command, "episode_yaw_index", None)
        eligible = torch.ones_like(completed, dtype=torch.bool)
        if episode_phase is not None:
            eligible &= episode_phase[completed] == state.phase
        if state.phase == 2 and episode_clearance_index is not None:
            eligible &= episode_clearance_index[completed] == state.clearance_index
        if state.phase == 3 and episode_yaw_index is not None:
            eligible &= episode_yaw_index[completed] == state.yaw_index
        clean = ~(failed[completed] | current_unsafe[completed])
        hold_ok = torch.zeros_like(completed, dtype=torch.bool)
        if state.phase == 0:
            hold_ok = cycle.phase0_hold_success[completed] & clean
        returned = (
            cycle.landed_four[completed] & ~failed[completed]
            & ~cycle.invalid[completed] & ~current_unsafe[completed]
        )
        cycled = clean & cycle.full_cycle_complete[completed] & ~cycle.invalid[completed]
        cycled &= current_four_ready[completed]
        for direction, direction_mask in (
            ("pos", eligible & (command.schedule.sign[completed] > 0)),
            ("neg", eligible & (command.schedule.sign[completed] < 0)),
        ):
            row = state.window[direction]
            row["attempts"] += int(direction_mask.sum().item())
            outcomes = {
                "four_hold": hold_ok & (state.phase == 0),
                "entry": clean & cycle.entered_transition[completed],
                "yaw": clean & cycle.entered_yaw[completed],
                "return": returned,
                "cycle": cycled,
                "support": cycle.support_ready_seen[completed],
                "pose": cycle.pose_ready_seen[completed],
            }
            for field, value in outcomes.items():
                row[field] += int((value & direction_mask).sum().item())
            for field in ("command_sum", "error_sum", "yaw_samples", "drift_sum", "drift_samples"):
                row[field] += getattr(cycle, field)[completed][direction_mask].sum().item()
        promoted = state.evaluate(clearance_levels, yaw_rate_levels, tracking_ratio_thresholds,
                                  hold_window, directional_window, required_windows)
        if promoted:
            env._staged_yaw_promotion_reset_pending = True

    if getattr(env, "_staged_yaw_promotion_reset_pending", False):
        # Curriculum runs inside _reset_idx after Isaac Lab has selected the
        # current reset IDs. Keep every global reward/DR parameter at the old
        # level until the queued all-environment reset is processed next step.
        return {"promotion_reset_pending": torch.tensor(1.0, device=env.device)}

    phase = state.phase
    clearance = float(clearance_levels[state.clearance_index] if phase <= 2 else clearance_levels[-1])
    yaw_limit = float(yaw_rate_levels[state.yaw_index] if phase == 3 else yaw_rate_levels[0])
    dr_scale = float(dr_scale_levels[state.yaw_index] if phase == 3 else 0.0)
    command.cfg.staged_phase = phase
    command.cfg.staged_clearance_index = state.clearance_index
    command.cfg.staged_yaw_index = state.yaw_index
    command.cfg.yaw_rate_limit = yaw_limit
    command.cfg.yaw_rate_range = (-yaw_limit, yaw_limit)
    command.cfg.target_clearance = clearance

    weights = {
        "fsm_gated_tracking": (0.0, 0.0, 2.0, 8.0)[phase],
        "transition_progress": (0.0, 1.0, 1.0, 1.0)[phase],
        "return_to_four_landing": (0.0, 0.0, 0.0, 20.0)[phase],
        "four_stand_ready_bonus": (0.0, 0.0, 0.0, 20.0)[phase],
        "spin_center_drift": (0.0, 0.0, 0.0, -2.0)[phase],
        "safe_recovery_entry": (0.0, 0.0, 0.0, -60.0)[phase],
        "staged_four_contact": (2.0, 2.0, 1.0, 1.0)[phase],
        "staged_four_dwell_bonus": (5.0, 5.0, 0.0, 0.0)[phase],
        "staged_milestone_bonus": (0.0, 20.0, 20.0, 20.0)[phase],
    }
    for name in env.reward_manager.active_terms:
        cfg = env.reward_manager.get_term_cfg(name)
        changed = False
        if name in weights:
            cfg.weight = weights[name]
            changed = True
        if "target_clearance" in cfg.params:
            cfg.params["target_clearance"] = clearance
            changed = True
        if changed:
            env.reward_manager.set_term_cfg(name, cfg)

    force = env.event_manager.get_term_cfg("randomize_apply_external_force_torque")
    force.params["force_range"] = (-10.0 * dr_scale, 10.0 * dr_scale)
    force.params["torque_range"] = (-10.0 * dr_scale, 10.0 * dr_scale)
    env.event_manager.set_term_cfg("randomize_apply_external_force_torque", force)
    gains = env.event_manager.get_term_cfg("randomize_actuator_gains")
    gains.params["stiffness_distribution_params"] = (1.0 - .15 * dr_scale, 1.0 + .15 * dr_scale)
    gains.params["damping_distribution_params"] = (1.0 - .15 * dr_scale, 1.0 + .15 * dr_scale)
    env.event_manager.set_term_cfg("randomize_actuator_gains", gains)
    push = env.event_manager.get_term_cfg("randomize_push_robot")
    push.params["velocity_range"] = {
        "x": (-.5 * dr_scale, .5 * dr_scale), "y": (-.5 * dr_scale, .5 * dr_scale),
    }
    env.event_manager.set_term_cfg("randomize_push_robot", push)

    def scalar(value):
        return torch.tensor(float(value), device=env.device)

    result = {
        "promotion_reset_pending": scalar(0.0),
        "phase": scalar(phase), "clearance_index": scalar(state.clearance_index),
        "yaw_index": scalar(state.yaw_index), "target_clearance": scalar(clearance),
        "yaw_limit": scalar(yaw_limit), "online_dr_scale": scalar(dr_scale),
        "consecutive_pass_windows": scalar(state.consecutive_passes),
        "last_window_passed": scalar(state.last_window_passed),
        "last_window_phase": scalar(state.last_window_phase),
    }
    for direction in ("pos", "neg"):
        result[f"{direction}/attempts"] = scalar(state.window[direction]["attempts"])
        result[f"{direction}/last_attempts"] = scalar(state.last_counts[direction]["attempts"])
        for field in ("four_hold", "entry", "yaw", "return", "cycle"):
            result[f"{direction}/last_{field}_successes"] = scalar(state.last_counts[direction][field])
        current_rates = state.rates(state.window[direction])
        for name, value in current_rates.items():
            result[f"{direction}/{name}"] = scalar(
                value if state.window[direction]["attempts"] else state.last_rates[direction].get(name, value)
            )
        for name, value in state.last_rates[direction].items():
            result[f"{direction}/last_window/{name}"] = scalar(value)
    totals = {name: sum(row[name] for row in state.window.values()) for name in _FIELDS}
    current_rates = state.rates(totals)
    for name, value in current_rates.items():
        result[name] = scalar(value if totals["attempts"] else state.last_rates["all"].get(name, value))
    for name, value in state.last_rates["all"].items():
        result[f"last_window/{name}"] = scalar(value)
    return result
