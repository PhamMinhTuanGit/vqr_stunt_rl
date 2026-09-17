"""CPU unit tests for the four-mode pivot learning formulas."""

import importlib.util
from pathlib import Path
import unittest

import torch


PIVOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pivot_math", PIVOT / "mdp/pivot_math.py")
pivot_math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pivot_math)


class PivotLearningLogicTest(unittest.TestCase):
    def test_ground_quarter_turn_about_body_center_has_no_drift(self):
        mode = torch.tensor([pivot_math.GROUND, pivot_math.BALANCE])
        body_center = torch.zeros(2, 2)
        body_anchor = torch.zeros(2, 2)
        # Rear midpoint moved only because the chassis yawed 90 degrees about
        # its center.  GROUND must ignore that motion; BALANCE must not.
        support_midpoint = torch.tensor([[0.0, -0.25], [0.0, -0.25]])
        support_anchor = torch.tensor([[-0.25, 0.0], [-0.25, 0.0]])
        drift = pivot_math.mode_planar_drift(
            mode, body_center, body_anchor, support_midpoint, support_anchor
        )
        self.assertEqual(drift[0].item(), 0.0)
        self.assertLess(drift[0].item(), 0.20)
        self.assertGreater(drift[1].item(), 0.20)

    def test_joint_target_is_continuous_at_all_mode_boundaries(self):
        durations = (2.0, 3.0, 10.0, 5.0)
        boundaries = (2.0, 5.0, 15.0)
        standing = torch.tensor([[0.0, -0.65, 1.30]])
        balance = torch.tensor([[0.2, -1.10, 1.80]])
        eps = 1.0e-5

        expected_at_boundary = (standing[0], balance[0], balance[0])
        for boundary, expected in zip(boundaries, expected_at_boundary, strict=True):
            times = torch.tensor([boundary - eps, boundary, boundary + eps])
            phase = pivot_math.pose_phase_from_time(times, *durations)
            targets = pivot_math.blend_joint_prior(
                standing.expand(3, -1), balance.expand(3, -1), phase
            )
            torch.testing.assert_close(targets[1], expected)
            self.assertLess(torch.max(torch.abs(targets[2] - targets[0])).item(), 1.0e-4)

    def test_front_lift_requires_both_rear_contacts(self):
        contacts = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        score = pivot_math.supported_front_lift_score(contacts)
        torch.testing.assert_close(score, torch.tensor([1.0, 0.0, 0.0]))

    def test_landing_force_below_budget_has_no_penalty(self):
        forces = torch.tensor([[100.0, 320.0, 644.9, 645.0], [0.0, 646.0, 0.0, 0.0]])
        excess = pivot_math.force_budget_excess(forces, 645.0)
        self.assertEqual(excess[0].item(), 0.0)
        self.assertGreater(excess[1].item(), 0.0)

    def test_zeta_dot_matches_finite_difference(self):
        dtype = torch.float64
        dt = 1.0e-6
        heading = torch.nn.functional.normalize(
            torch.tensor([[0.85, 0.30, -0.43]], dtype=dtype), dim=-1
        )
        angular_velocity = torch.tensor([[0.20, -0.40, 1.20]], dtype=dtype)
        com_pos = torch.tensor([[0.31, -0.14]], dtype=dtype)
        support_pos = torch.tensor([[-0.08, 0.05]], dtype=dtype)
        com_vel = torch.tensor([[0.23, -0.11]], dtype=dtype)
        support_vel = torch.tensor([[0.04, 0.07]], dtype=dtype)

        signals = pivot_math.balance_coordinates(
            com_pos,
            com_vel,
            support_pos,
            support_vel,
            heading,
            angular_velocity,
            omega0=4.118,
        )
        heading_next_w = torch.nn.functional.normalize(
            heading + torch.linalg.cross(angular_velocity, heading, dim=-1) * dt,
            dim=-1,
        )
        heading_next = torch.nn.functional.normalize(heading_next_w[:, :2], dim=-1)
        zeta_next = torch.sum(
            ((com_pos + com_vel * dt) - (support_pos + support_vel * dt))
            * heading_next,
            dim=-1,
        )
        finite_difference = (zeta_next - signals[:, 0]) / dt
        torch.testing.assert_close(signals[:, 1], finite_difference, rtol=2.0e-5, atol=2.0e-6)


if __name__ == "__main__":
    unittest.main()
