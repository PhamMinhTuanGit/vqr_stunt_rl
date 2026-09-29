"""CPU checks for signed yaw keyboard playback without starting Isaac Sim."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pytest
import torch


ROOT = Path(__file__).parents[1]
PLAY = ROOT / "scripts/reinforcement_learning/rsl_rl/play.py"
COMMANDS = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/commands.py"


def _source_definition(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_keyboard_yaw_keeps_both_directions_and_refreshes_observation():
    received = []
    term = SimpleNamespace(set_external_command=received.append)
    env = SimpleNamespace(unwrapped=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: term)))
    writer = _source_definition(PLAY, "_write_yaw_keyboard_command", {})
    writer(env, SimpleNamespace(advance=lambda: (0.0, 0.0, -0.6)))
    writer(env, SimpleNamespace(advance=lambda: (0.0, 0.0, 0.6)))
    assert received == [-0.6, 0.6]

    tree = ast.parse(PLAY.read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    loop = next(node for node in ast.walk(main) if isinstance(node, ast.While)
                and ast.unparse(node.test) == "simulation_app.is_running()")
    source = ast.unparse(loop)
    assert source.index("_write_yaw_keyboard_command(env, controller)") < source.index("actions = policy(obs)")
    assert source.index("obs = env.get_observations()", source.index("_write_yaw_keyboard_command")) < source.index("actions = policy(obs)")


def test_external_yaw_command_survives_resampling():
    class BaseCommand:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self.num_envs = 2
            self.device = "cpu"
            self.metrics = {}

    command_cls = _source_definition(
        COMMANDS, "YawRateCommand", {"CommandTerm": BaseCommand, "torch": torch, "Sequence": Sequence}
    )
    cfg = SimpleNamespace(asset_name="robot", yaw_rate_range=(-1.0, 1.0))
    term = command_cls(cfg, SimpleNamespace(scene={"robot": object()}))
    term.set_external_command(-0.6)
    term._resample_command([0, 1])
    assert term.command[:, 0].tolist() == pytest.approx([-0.6, -0.6])
    term.set_external_command(2.0, env_ids=[1])
    assert term.command[:, 0].tolist() == pytest.approx([-0.6, 1.0])
