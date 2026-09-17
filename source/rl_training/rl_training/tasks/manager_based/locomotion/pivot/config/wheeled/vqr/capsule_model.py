# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Virtual capsule model for the four-mode pivot task (invariants I6/I11).

The URDF carries no collision geometry for ``*_HIP`` bodies; this module
declares analytic capsules for them plus the torso, and pairs whose
clearance the self-collision reward watches.  Wheel collision stays on the
cylinder primitive (I11).
"""

from __future__ import annotations

# Capsule radii [m] keyed by body name; hip capsules are virtual (section 12).
CAPSULE_RADII: dict[str, float] = {
    "TORSO": 0.14,
    "FL_HIP": 0.05,
    "FR_HIP": 0.05,
    "HL_HIP": 0.05,
    "HR_HIP": 0.05,
    "FL_THIGH": 0.04,
    "FR_THIGH": 0.04,
    "HL_THIGH": 0.04,
    "HR_THIGH": 0.04,
    "FL_SHANK": 0.035,
    "FR_SHANK": 0.035,
    "HL_SHANK": 0.035,
    "HR_SHANK": 0.035,
}

# Capsule axis endpoints in each body's local frame (start, end) [m].
CAPSULE_SEGMENTS_LOCAL: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "TORSO": ((0.0, 0.0, -0.14), (0.0, 0.0, 0.14)),
    "FL_HIP": ((0.0, -0.05, 0.0), (0.0, 0.05, 0.0)),
    "FR_HIP": ((0.0, -0.05, 0.0), (0.0, 0.05, 0.0)),
    "HL_HIP": ((0.0, -0.05, 0.0), (0.0, 0.05, 0.0)),
    "HR_HIP": ((0.0, -0.05, 0.0), (0.0, 0.05, 0.0)),
    "FL_THIGH": ((0.0, 0.0, 0.0), (0.0, 0.0, -0.14)),
    "FR_THIGH": ((0.0, 0.0, 0.0), (0.0, 0.0, -0.14)),
    "HL_THIGH": ((0.0, 0.0, 0.0), (0.0, 0.0, -0.14)),
    "HR_THIGH": ((0.0, 0.0, 0.0), (0.0, 0.0, -0.14)),
    "FL_SHANK": ((0.0, 0.0, 0.0), (0.0, 0.0, -0.12)),
    "FR_SHANK": ((0.0, 0.0, 0.0), (0.0, 0.0, -0.12)),
    "HL_SHANK": ((0.0, 0.0, 0.0), (0.0, 0.0, -0.12)),
    "HR_SHANK": ((0.0, 0.0, 0.0), (0.0, 0.0, -0.12)),
}

# Cross-body pairs watched by the analytic self-collision reward (reward-only,
# PhysX ``enabled_self_collisions`` stays False).  Left/right hip pairs are the
# only pairs the torso cannot occlude; front legs also cross behind the torso
# during the BALANCE tuck.
SELFCOLLISION_PAIRS: list[tuple[str, str]] = [
    ("FL_HIP", "HR_HIP"),
    ("FR_HIP", "HL_HIP"),
    ("FL_THIGH", "HR_THIGH"),
    ("FR_THIGH", "HL_THIGH"),
    ("FL_SHANK", "HR_SHANK"),
    ("FR_SHANK", "HL_SHANK"),
    ("FL_HIP", "HR_THIGH"),
    ("FR_HIP", "HL_THIGH"),
]

__all__ = ["CAPSULE_RADII", "CAPSULE_SEGMENTS_LOCAL", "SELFCOLLISION_PAIRS"]
