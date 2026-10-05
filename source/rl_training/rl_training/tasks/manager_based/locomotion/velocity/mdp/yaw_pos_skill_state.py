"""Simulator-independent promotion rules for the scratch POS skill experiment."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field

TASK_ID = "Flat-VQR-Wheel-Yaw-POS-Skill"
CHECKPOINT_KEY = "yaw_pos_skill_curriculum"
PHASES = ("A_pose", "B_yaw", "C_differential", "D_neutral", "E_speed")


@dataclass(frozen=True)
class SkillSettings:
    clearance: float = 0.05
    yaw_levels: tuple = (0.25, 0.40, 0.55, 0.70, 0.85, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00)
    tracking_thresholds: tuple = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55) + (0.55,) * 6
    edge_thresholds: tuple = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45) + (0.45,) * 6
    robustness_scales: tuple = (0.0, 0.10, 0.25, 0.50, 0.75, 1.00)
    support: float = 0.85
    lift: float = 0.80
    balance: float = 0.75
    height: float = 0.35
    yaw: float = 0.65
    certificate: float = 0.90
    coverage: float = 0.90
    success_rate: float = 0.85
    min_episodes: int = 2048
    required_windows: int = 3
    pose_steps: int = 1000
    skill_steps: int = 6000

    def __post_init__(self):
        n = len(self.yaw_levels)
        if n == 0 or len(self.tracking_thresholds) != n or len(self.edge_thresholds) != n:
            raise ValueError("Yaw and tracking tables must have equal nonzero lengths.")
        if any(not math.isfinite(x) or x <= 0 for x in self.yaw_levels) or any(
            a >= b for a, b in zip(self.yaw_levels, self.yaw_levels[1:])
        ):
            raise ValueError("Yaw levels must be finite, positive and increasing.")
        if self.robustness_scales[0] != 0 or self.robustness_scales[-1] != 1 or any(
            a >= b for a, b in zip(self.robustness_scales, self.robustness_scales[1:])
        ):
            raise ValueError("Robustness must increase from zero to one.")
        rates = (self.support, self.lift, self.balance, self.yaw, self.certificate,
                 self.coverage, self.success_rate, *self.tracking_thresholds, *self.edge_thresholds)
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in rates):
            raise ValueError("Gate rates must be finite and in [0, 1].")
        if self.min_episodes < 1 or self.required_windows < 1 or min(self.pose_steps, self.skill_steps) < 0:
            raise ValueError("Invalid window counts or minimum stage duration.")
        if not math.isfinite(self.height) or self.height <= 0 or self.clearance <= 0:
            raise ValueError("Height and clearance must be positive.")

    def contract(self):
        # Lists also survive YAML/JSON round trips without tuple coercion.
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


EPISODE_GATES = ("support", "lift", "balance", "height", "safe", "yaw", "differential_episode")
WINDOW_GATES = ("tracking", "edge_tracking", "differential", "neutral", "four_contact", "anchor_coverage")


def empty_window():
    return {"episodes": 0, "successes": 0, "passes": dict.fromkeys(EPISODE_GATES, 0), "metrics": {}}


@dataclass
class SkillState:
    settings: SkillSettings = field(default_factory=SkillSettings)
    phase: int = 0
    yaw_stage: int = 0
    robustness_stage: int = 0
    stage_start_step: int = 0
    consecutive_passes: int = 0
    complete: bool = False
    pending: dict | None = None
    window: dict = field(default_factory=empty_window)
    last: dict = field(default_factory=dict)

    @property
    def dr_scale(self):
        return self.settings.robustness_scales[self.robustness_stage]

    def enabled(self):
        return {**dict.fromkeys(EPISODE_GATES[:5], True), "yaw": self.phase >= 1,
                "differential_episode": self.phase >= 2, "tracking": self.phase >= 1,
                "edge_tracking": self.phase >= 1, "differential": self.phase >= 2,
                **dict.fromkeys(("neutral", "four_contact", "anchor_coverage"), self.phase >= 3)}

    def thresholds(self):
        c = self.settings
        return {"support": c.support, "lift": c.lift, "balance": c.balance, "height": c.height,
                "safe": 1.0, "yaw": c.yaw, "differential_episode": c.certificate,
                "tracking": c.tracking_thresholds[self.yaw_stage],
                "edge_tracking": c.edge_thresholds[self.yaw_stage], "differential": c.certificate,
                "neutral": c.certificate, "four_contact": c.coverage, "anchor_coverage": c.coverage}

    def record(self, values: dict, samples: dict, active: bool = True):
        """Add one completed episode; each metric contains its raw (sum, count)."""
        if self.pending is not None or self.complete:
            return
        for name, (total, count) in samples.items():
            row = self.window["metrics"].setdefault(name, [0.0, 0])
            row[0] += float(total)
            row[1] += int(count)
        # Phase A tests pose acquisition for every completed episode; later
        # phases require active yaw samples before an episode is eligible.
        if not active and self.phase > 0:
            return
        passed = {name: math.isfinite(values.get(name, float("nan")))
                  and values[name] >= threshold
                  for name, threshold in self.thresholds().items() if name in EPISODE_GATES}
        self.window["episodes"] += 1
        for name, value in passed.items():
            self.window["passes"][name] += int(value)
        self.window["successes"] += int(all(passed[k] for k in EPISODE_GATES if self.enabled()[k]))
        for name in ("support", "lift", "balance", "height", "yaw"):
            value = values.get(name, float("nan"))
            if math.isfinite(value):
                row = self.window["metrics"].setdefault(name + "_score", [0.0, 0])
                row[0] += float(value)
                row[1] += 1

    def rates(self):
        metrics = self.window["metrics"]
        result = {k: v[0] / v[1] if v[1] > 0 else 0.0 for k, v in metrics.items()}
        for name, command, error in (("tracking", "command", "error"),
                                     ("edge_tracking", "edge_command", "edge_error")):
            cmd = metrics.get(command, [0, 0])[0]
            result[name] = 1.0 - metrics.get(error, [0, 0])[0] / cmd if cmd > 1.e-6 else 0.0
        result["success_rate"] = self.window["successes"] / max(self.window["episodes"], 1)
        return result

    def evaluate(self, step: int):
        """Evaluate a disjoint completed window and queue, rather than apply, promotion."""
        if self.pending is not None or self.complete or self.window["episodes"] < self.settings.min_episodes:
            return False
        enabled, thresholds, rates = self.enabled(), self.thresholds(), self.rates()
        blockers = []
        gate_rates = {k: self.window["passes"][k] / self.window["episodes"] for k in EPISODE_GATES}
        for name in EPISODE_GATES:
            if enabled[name] and gate_rates[name] < self.settings.success_rate:
                blockers.append(name)
        if rates["success_rate"] < self.settings.success_rate:
            blockers.append("joint_episode_success")
        counts = {name: self.window["metrics"].get(name, [0, 0])[1] for name in WINDOW_GATES}
        counts["tracking"] = self.window["metrics"].get("command", [0, 0])[1]
        counts["edge_tracking"] = self.window["metrics"].get("edge_command", [0, 0])[1]
        for name in WINDOW_GATES:
            value = rates.get(name, 0.0)
            if enabled[name] and (counts[name] <= 0 or not math.isfinite(value) or value < thresholds[name]):
                blockers.append(name)
        passed = not blockers
        self.consecutive_passes = min(self.consecutive_passes + 1, self.settings.required_windows) if passed else 0
        elapsed = step - self.stage_start_step
        minimum = self.settings.pose_steps if self.phase == 0 else self.settings.skill_steps
        if passed and self.consecutive_passes < self.settings.required_windows:
            blockers.append("consecutive_windows")
        if passed and elapsed < minimum:
            blockers.append("minimum_duration")
        self.last = {"phase": self.phase, "yaw_stage": self.yaw_stage,
                     "robustness_stage": self.robustness_stage, "enabled": enabled,
                     "thresholds": thresholds, "rates": rates, "gate_rates": gate_rates,
                     "counts": counts, "episodes": self.window["episodes"],
                     "passed": passed, "blockers": blockers}
        if not blockers:
            if self.phase < 4:
                self.pending = {"phase": self.phase + 1, "yaw_stage": 0, "robustness_stage": 0}
            elif self.yaw_stage + 1 < len(self.settings.yaw_levels):
                self.pending = {"phase": 4, "yaw_stage": self.yaw_stage + 1, "robustness_stage": 0}
            elif self.robustness_stage + 1 < len(self.settings.robustness_scales):
                self.pending = {"phase": 4, "yaw_stage": self.yaw_stage,
                                "robustness_stage": self.robustness_stage + 1}
            else:
                self.complete = True
        self.window = empty_window()
        return self.pending is not None

    def commit(self, step: int):
        if self.pending is None:
            return False
        for name, value in self.pending.items():
            setattr(self, name, value)
        self.pending = None
        self.window = empty_window()
        self.consecutive_passes = 0
        self.stage_start_step = step
        return True

    def export(self, step: int):
        values = {k: copy.deepcopy(v) for k, v in vars(self).items()
                  if k not in ("settings", "stage_start_step")}
        return {"version": 1, "settings": self.settings.contract(), "values": values,
                "stage_elapsed_steps": max(0, step - self.stage_start_step)}

    @classmethod
    def restore(cls, payload: dict, settings: SkillSettings, step: int):
        if payload.get("version") != 1 or payload.get("settings") != settings.contract():
            raise ValueError("Skill checkpoint version/settings mismatch.")
        values = copy.deepcopy(payload.get("values", {}))
        state = cls(settings=settings)
        allowed = set(vars(state)) - {"settings", "stage_start_step"}
        if set(values) != allowed:
            raise ValueError("Skill checkpoint fields mismatch.")
        for row in (values, values.get("pending")):
            if row is None:
                continue
            p, y, r = (row.get(k) for k in ("phase", "yaw_stage", "robustness_stage"))
            if (type(p) is not int or not 0 <= p <= 4 or type(y) is not int
                    or not 0 <= y < len(settings.yaw_levels) or type(r) is not int
                    or not 0 <= r < len(settings.robustness_scales)
                    or (p < 4 and (y != 0 or r != 0)) or (r > 0 and y != len(settings.yaw_levels) - 1)):
                raise ValueError("Invalid skill checkpoint stage indices.")
        if type(values["consecutive_passes"]) is not int or not 0 <= values["consecutive_passes"] <= settings.required_windows:
            raise ValueError("Invalid skill checkpoint streak.")
        window = values["window"]
        if (set(window) != set(empty_window()) or set(window["passes"]) != set(EPISODE_GATES)
                or type(window["episodes"]) is not int or window["episodes"] < 0
                or not 0 <= window["successes"] <= window["episodes"]
                or any(not 0 <= v <= window["episodes"] for v in window["passes"].values())
                or any(len(v) != 2 or not math.isfinite(v[0]) or type(v[1]) is not int or v[1] < 0
                       for v in window["metrics"].values())):
            raise ValueError("Invalid skill checkpoint window.")
        elapsed = payload.get("stage_elapsed_steps")
        if type(elapsed) is not int or elapsed < 0:
            raise ValueError("Invalid skill checkpoint elapsed steps.")
        for name, value in values.items():
            setattr(state, name, value)
        state.stage_start_step = step - elapsed
        return state
