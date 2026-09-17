# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Analytic capsule clearance (invariant I6): reward-only self-collision.

PhysX self-collisions stay disabled; the MDP penalises capsule pairs from
``capsule_model.py`` through this pure-torch helper.
"""

from __future__ import annotations

import torch

from .capsule_geometry import segment_segment_distance


class CapsuleClearance:
    """Minimum distance between configured capsule pairs, world framed."""

    def __init__(self, pair_names: list[tuple[str, str]], radius: dict[str, float]):
        self._pair_names = list(pair_names)
        self._radii = dict(radius)
        self._pair_indices: list[tuple[int, int]] | None = None

    def bind(self, name_to_index: dict[str, int]) -> None:
        """Resolve capsule names to body indices once assets are known."""
        self._pair_indices = [
            (name_to_index[a], name_to_index[b]) for a, b in self._pair_names
        ]

    @property
    def bound(self) -> bool:
        return self._pair_indices is not None

    def update(self, capsule_endpoints: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Return the per-env minimum surface distance across all pairs.

        ``capsule_endpoints`` maps capsule name -> ``(start_w, end_w)`` with
        each tensor shaped ``(N, 3)``.  Result shape ``(N,)``; may be
        negative when capsules overlap.
        """
        if self._pair_indices is None:
            raise RuntimeError("CapsuleClearance.bind() must be called before update()")

        distances = []
        for i, (name_a, name_b) in enumerate(self._pair_indices):
            start_a, end_a = capsule_endpoints[self._pair_names[i][0]]
            start_b, end_b = capsule_endpoints[self._pair_names[i][1]]
            core = segment_segment_distance(start_a, end_a, start_b, end_b)
            radii = self._radii[self._pair_names[i][0]] + self._radii[self._pair_names[i][1]]
            distances.append(core - radii)
        return torch.stack(distances, dim=0).amin(dim=0)
