"""Small CPU contract for POS keyboard playback without launching Isaac Sim."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


PLAY = Path(__file__).parents[1] / "scripts/reinforcement_learning/rsl_rl/play.py"


def _play_tree():
    return ast.parse(PLAY.read_text(encoding="utf-8"))


def _keyboard_writer():
    node = next(node for node in _play_tree().body if isinstance(node, ast.FunctionDef)
                and node.name == "_write_pos_keyboard_command")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(PLAY), "exec"), namespace)
    return namespace[node.name]


def _checkpoint_limit_reader(checkpoint):
    node = next(node for node in _play_tree().body if isinstance(node, ast.FunctionDef)
                and node.name == "_checkpoint_pos_yaw_limit")
    torch = SimpleNamespace(load=lambda path, **kwargs: checkpoint)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(PLAY), "exec"), namespace)
    return namespace[node.name]


@pytest.mark.parametrize(("stage", "expected"), [(0, 0.25), (2, 0.55)])
def test_pos_keyboard_limit_uses_checkpoint_yaw_stage(stage, expected):
    levels = [0.25, 0.40, 0.55, 1.0]
    checkpoint = {"infos": {"yaw_curriculum": {
        "yaw_rate_levels": levels,
        "values": {"_yaw_task_curriculum_yaw_stage": stage},
    }}}
    assert _checkpoint_limit_reader(checkpoint)("model.pt", levels) == expected


@pytest.mark.parametrize("checkpoint", [
    {"infos": None},
    {"infos": {"yaw_curriculum": {
        "yaw_rate_levels": [0.25, 0.45],
        "values": {"_yaw_task_curriculum_yaw_stage": 1},
    }}},
])
def test_pos_keyboard_limit_falls_back_to_first_stage_without_matching_metadata(checkpoint):
    assert _checkpoint_limit_reader(checkpoint)("model.pt", [0.25, 0.40, 1.0]) == 0.25


def test_pos_keyboard_limit_accepts_appended_stages_without_raising_trained_limit():
    saved = [0.25, 0.40, 0.55, 0.70, 0.85, 1.0]
    checkpoint = {"infos": {"yaw_curriculum": {
        "yaw_rate_levels": saved, "values": {"_yaw_task_curriculum_yaw_stage": 5},
    }}}
    read = _checkpoint_limit_reader(checkpoint)
    assert read("model.pt", saved + [1.25, 1.5, 1.75, 2.0, 2.5, 3.0]) == 1.0
    checkpoint["infos"]["yaw_curriculum"]["values"]["_yaw_task_curriculum_yaw_stage"] = 6
    assert read("model.pt", saved + [1.25]) == 0.25


def test_pos_keyboard_branch_isolated_from_base_velocity():
    main = next(node for node in _play_tree().body if isinstance(node, ast.FunctionDef) and node.name == "main")
    keyboard = next(node for node in ast.walk(main) if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "args_cli.keyboard")
    assert "env_cfg.scene.num_envs = 1" in ast.unparse(keyboard)
    pos_branch = next(node for node in keyboard.body if isinstance(node, ast.If)
                      and "Flat-VQR-Wheel-Yaw-POS" in ast.unparse(node.test))
    pos_source = "\n".join(map(ast.unparse, pos_branch.body))
    assert "base_velocity" not in pos_source
    assert "velocity_commands" not in pos_source
    assert "yaw_cfg.external_control = True" in pos_source
    assert "Se2Keyboard" in pos_source
    old_source = "\n".join(map(ast.unparse, pos_branch.orelse))
    assert "base_velocity" in old_source
    assert "velocity_commands" in old_source
    loop = next(node for node in ast.walk(main) if isinstance(node, ast.While)
                and ast.unparse(node.test) == "simulation_app.is_running()")
    loop_source = ast.unparse(loop)
    assert loop_source.index("_write_pos_keyboard_command(env, controller)") < loop_source.index("actions = policy(obs)")
    assert loop_source.index("obs = env.get_observations()") < loop_source.index("actions = policy(obs)")


@pytest.mark.parametrize(
    ("omega_z", "expected"),
    [(0.5, 0.5), (2.0, 1.0), (0.0, 0.0), (-0.5, 0.0), (0.05, 0.0)],
)
def test_keyboard_yaw_written_through_command_term(omega_z, expected):
    calls = []
    term = SimpleNamespace(
        cfg=SimpleNamespace(yaw_rate_range=(0.0, 1.0), deadband=0.1),
        set_external_command=lambda value: calls.append(value),
    )

    def get_term(name):
        assert name == "yaw_rate_cmd"  # POS must never request base_velocity.
        return term

    env = SimpleNamespace(unwrapped=SimpleNamespace(command_manager=SimpleNamespace(get_term=get_term)))
    controller = SimpleNamespace(advance=lambda: (99.0, -99.0, omega_z))
    _keyboard_writer()(env, controller)
    assert calls == [expected]
