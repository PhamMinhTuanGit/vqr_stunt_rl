# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""q_prior(mode, tuck*): leg priors centred on the audited TO reference (I3).

The PD residual scale (0.3) and joint-limit clamping live in the action term
(``actions.py``); this module only interpolates reference poses.
"""

from __future__ import annotations

import torch

from .to_reference import load_leg_reference, smoothstep

# Cached module-level reference (12 joints), loaded once per process.
_REFERENCE_CACHE: dict[str, dict[str, float]] = {}


def standing_pose(joint_names: list[str]) -> torch.Tensor:
    """Audited four-wheel standing pose from the robot configuration."""
    from ..config.wheeled.vqr.robot_cfg import DEFAULT_JOINT_POS

    values = [DEFAULT_JOINT_POS[name] for name in joint_names]
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


def balance_pose(joint_names: list[str], reference_path: str | None = None) -> torch.Tensor:
    """Two-wheel balance pose from the validated FL-HR TO solution."""
    # TODO: Replace this only after an HL-HR trajectory-optimization reference
    # has been validated.  The four-mode task currently uses HL-HR support, so
    # this existing FL-HR reference is a known bootstrap approximation.
    key = reference_path or "default"
    if key not in _REFERENCE_CACHE:
        if reference_path is None:
            _REFERENCE_CACHE[key] = load_leg_reference(joint_names)
        else:
            _REFERENCE_CACHE[key] = load_leg_reference(joint_names, reference_path)
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
    """Mode-interpolated leg prior with the tuck blend (I3).

    mode: int tensor ``(N,)`` with GROUND=0, REAR_UP=1, BALANCE=2, LAND=3.
    tuck_command: ``(N,)`` in [0, 1]; 0 = open stance, 1 = fully tucked.

    Per-mode logic:
      GROUND  : standing pose, tuck does not apply.
      REAR_UP : blend standing -> balance pose driven by tuck (the lift).
      BALANCE : balance pose.
      LAND    : blend balance -> standing driven by tuck (the set-down).
    """
    standing = standing_pose(joint_names).to(
        device=mode.device, dtype=tuck_command.dtype
    )  # (1, 12)
    balance = balance_pose(joint_names, reference_path).to(
        device=mode.device, dtype=tuck_command.dtype
    )  # (1, 12)

    n = mode.shape[0]
    tuck = smoothstep(tuck_command.clamp(0.0, 1.0)).unsqueeze(-1)  # (N, 1)

    standing = standing.expand(n, -1)
    balance = balance.expand(n, -1)
    rear_up = standing + tuck * (balance - standing)
    land = balance + tuck * (standing - balance)

    prior = standing
    prior = torch.where((mode == REAR_UP).unsqueeze(-1), rear_up, prior)
    prior = torch.where((mode == BALANCE).unsqueeze(-1), balance, prior)
    prior = torch.where((mode == LAND).unsqueeze(-1), land, prior)
    return prior
