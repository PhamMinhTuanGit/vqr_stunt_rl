"""CPU checks runnable without starting Isaac Sim."""

import ast
import importlib.util
from pathlib import Path
import unittest

import torch

PIVOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("to_reference", PIVOT / "mdp/to_reference.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
# Read the authoritative constants without importing Isaac Sim.
tree = ast.parse((PIVOT / "config/wheeled/vqr/robot_cfg.py").read_text())
NAMES = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "LEG_JOINT_NAMES" for t in n.targets))


class ReferenceTest(unittest.TestCase):
    def test_names_and_interpolation(self):
        values = reference.load_leg_reference(NAMES)
        reverse = reference.load_leg_reference(list(reversed(NAMES)))
        self.assertEqual(values, reverse)
        self.assertEqual(list(reverse), list(reversed(NAMES)))
        target = torch.tensor(list(values.values()))
        standing = torch.tensor([0.0 if "HipX" in n else -0.65 if "HipY" in n else 1.3 for n in NAMES])
        phase = torch.tensor([0.0, 0.25, 0.5, 1.0])
        result = reference.pose_reference(standing, target, phase)
        torch.testing.assert_close(result[0], standing)
        torch.testing.assert_close(result[-1], target)
        torch.testing.assert_close(result[1], standing + 0.15625 * (target - standing))
        torch.testing.assert_close(result[2], (standing + target) / 2)
        self.assertTrue(torch.isfinite(result).all())

    def test_contact_schedule(self):
        phase = torch.tensor([0.0, 0.39, 0.4, 0.6, 0.8, 1.0])
        targets = reference.contact_targets(phase)
        torch.testing.assert_close(targets[:3], torch.ones(3, 4))
        torch.testing.assert_close(targets[3], torch.tensor([1., .5, .5, 1.]))
        torch.testing.assert_close(targets[4:], torch.tensor([[1., 0., 0., 1.]]).expand(2, 4))

    def test_bad_mapping_rejected(self):
        with self.assertRaises(ValueError):
            reference.load_leg_reference(NAMES[:-1])


if __name__ == "__main__":
    unittest.main()
