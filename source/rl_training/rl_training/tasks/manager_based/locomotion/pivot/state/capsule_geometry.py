# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Vectorised segment-segment closest distance for analytic collision (I6).

Implements the clamped closest-point algorithm (Ericson, Real-Time Collision
Detection) as a batched torch kernel: segments are given by their endpoints
``(p1, q1)`` and ``(p2, q2)`` with leading batch dimension only.
"""

from __future__ import annotations

import torch

_EPS = 1.0e-9


def point_segment_distance(
    point: torch.Tensor, seg_start: torch.Tensor, seg_end: torch.Tensor
) -> torch.Tensor:
    """Distance from each point to each segment (batched, shape ``(N,)``)."""
    direction = seg_end - seg_start
    denom = torch.clamp(torch.sum(direction.square(), dim=-1), min=_EPS)
    t = torch.clamp(
        torch.sum((point - seg_start) * direction, dim=-1) / denom, min=0.0, max=1.0
    ).unsqueeze(-1)
    closest = seg_start + t * direction
    return torch.norm(point - closest, dim=-1)


def segment_segment_distance(
    p1: torch.Tensor, q1: torch.Tensor, p2: torch.Tensor, q2: torch.Tensor
) -> torch.Tensor:
    """Closest distance between two line segments, batched over ``N`` pairs."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2

    a = torch.sum(d1 * d1, dim=-1)
    e = torch.sum(d2 * d2, dim=-1)
    f = torch.sum(d2 * r, dim=-1)

    # Degenerate segment guards keep the divisions finite.
    a_safe = torch.clamp(a, min=_EPS)
    e_safe = torch.clamp(e, min=_EPS)

    c = torch.sum(d1 * r, dim=-1)
    b = torch.sum(d1 * d2, dim=-1)
    denom = torch.clamp(a_safe * e_safe - b * b, min=_EPS)

    s = torch.clamp((b * f - c * e_safe) / denom, min=0.0, max=1.0)
    t = (b * s + f) / e_safe
    t_high = t > 1.0
    t = torch.where(t_high, torch.ones_like(t), t)
    s = torch.where(
        t_high, torch.clamp((b + f) / a_safe, min=0.0, max=1.0), s
    )
    t = torch.where(t_high, torch.ones_like(t), (b * s + f) / e_safe)
    # ``t`` was clamped via masking above; re-clamp for the low branch.
    t = torch.clamp(t, min=0.0, max=1.0)

    closest1 = p1 + s.unsqueeze(-1) * d1
    closest2 = p2 + t.unsqueeze(-1) * d2
    return torch.norm(closest1 - closest2, dim=-1)
