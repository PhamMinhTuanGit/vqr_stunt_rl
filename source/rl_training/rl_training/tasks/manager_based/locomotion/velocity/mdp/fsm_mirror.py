"""Canonical POS/NEG wheel-diagonal mapping for the yaw FSM rewards."""

from __future__ import annotations

# The first endpoint is always the front wheel.  Keeping that order is
# important for the finite support-segment terms.
SUPPORT_POS = ("FL_WHEEL", "HR_WHEEL")
SUPPORT_NEG = ("FR_WHEEL", "HL_WHEEL")
SWING_POS = SUPPORT_NEG
SWING_NEG = SUPPORT_POS

MIRROR = {
    "com_support": (SUPPORT_POS, SUPPORT_NEG),
    "com_inside_segment": (SUPPORT_POS, SUPPORT_NEG),
    "support_span_band": (SUPPORT_POS, SUPPORT_NEG),
    "rolling_slip": (SUPPORT_POS, SUPPORT_NEG),
    "lift_clearance": (SWING_POS, SWING_NEG),
    "lifted_wheel_spin": (SWING_POS, SWING_NEG),
    "transition_progress": (SWING_POS, SWING_NEG),
}

__all__ = [
    "SUPPORT_POS",
    "SUPPORT_NEG",
    "SWING_POS",
    "SWING_NEG",
    "MIRROR",
]
