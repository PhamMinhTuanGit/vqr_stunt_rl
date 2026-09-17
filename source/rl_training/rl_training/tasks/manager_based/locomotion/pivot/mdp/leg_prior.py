# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Continuous leg-position prior for the four-mode HL-HR pivot task.

The PD residual scale (0.3) and joint-limit clamping live in the action term
(``actions.py``); this module only interpolates reference poses.
"""

from __future__ import annotations

import torch

from .pivot_math import blend_joint_prior
from .to_reference import load_leg_reference

# Cached module-level reference (12 joints), loaded once per process.
_REFERENCE_CACHE: dict[str, dict[str, float]] = {}


def standing_pose(joint_names: list[str]) -> torch.Tensor:
    """Audited four-wheel standing pose from the robot configuration."""
    from ..config.wheeled.vqr.robot_cfg import DEFAULT_JOINT_POS

    values = [DEFAULT_JOINT_POS[name] for name in joint_names]
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


def balance_pose(joint_names: list[str], reference_path: str | None = None) -> torch.Tensor:
    """Return a validated HL-HR pose, or the safe standing fallback.

    The repository currently contains only FL-HR TO results.  Those are not
    silently reused for this task.  Until an HL-HR result is supplied through
    ``reference_path``, the residual controller is centred on the standing
    pose; this is conservative, continuous, and directly checkable in Isaac
    Sim, but leaves the policy residual responsible for discovering rear-up.
    """
    if reference_path is None:
        return standing_pose(joint_names)

    key = str(reference_path)
    if key not in _REFERENCE_CACHE:
        _REFERENCE_CACHE[key] = load_leg_reference(
            joint_names,
            reference_path,
            expected_stance=("HL", "HR"),
            expected_swing=("FL", "FR"),
        )
    values = [_REFERENCE_CACHE[key][name] for name in joint_names]
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


# Mode ids (mirror supervisor_gate to stay import-free in either direction).
GROUND = 0
REAR_UP = 1
BALANCE = 2
LAND = 3


def q_prior(
    mode: torch.Tensor,
    tuck_command: torch.Tensor,
    joint_names: list[str],
    reference_path: str | None = None,
) -> torch.Tensor:
    """Continuous prior driven by the command term's deterministic pose phase.

    mode: int tensor ``(N,)`` with GROUND=0, REAR_UP=1, BALANCE=2, LAND=3.
    The second argument keeps the existing API name for compatibility.  It is
    no longer randomly sampled: 0=standing and 1=balance, with a C1 schedule
    GROUND(0) -> REAR_UP(0..1) -> BALANCE(1) -> LAND(1..0).
    """
    standing = standing_pose(joint_names).to(
        device=mode.device, dtype=tuck_command.dtype
    )  # (1, 12)
    balance = balance_pose(joint_names, reference_path).to(
        device=mode.device, dtype=tuck_command.dtype
    )  # (1, 12)

    n = mode.shape[0]
    return blend_joint_prior(
        standing.expand(n, -1),
        balance.expand(n, -1),
        tuck_command,
    )
