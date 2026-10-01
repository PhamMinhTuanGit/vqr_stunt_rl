"""CPU checks for the POS inspection video script."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/reinforcement_learning/rsl_rl/record_yaw_pos_video.py"


def _tree():
    return ast.parse(SCRIPT.read_text(encoding="utf-8"))


def _function(name, namespace=None):
    node = next(node for node in _tree().body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {} if namespace is None else namespace
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, 0.25), (3, 0.25), (4, 0.0), (7, 0.0),
     (8, 0.25), (11, 0.25), (12, 0.0), (15, 0.0)],
)
def test_two_positive_neutral_cycles(step, expected):
    assert _function("_command_for_step")(step, 4, 4, 0.25) == expected


def test_checkpoint_stage_sets_default_yaw_limit():
    state = {"infos": {"yaw_curriculum": {
        "yaw_rate_levels": [0.25, 0.4],
        "values": {"_yaw_task_curriculum_yaw_stage": 1},
    }}}
    fake_torch = SimpleNamespace(load=lambda *args, **kwargs: state)
    assert _function("_checkpoint_yaw_limit", {"torch": fake_torch, "Path": Path})(Path("model.pt"), 0.25) == 0.4


def test_video_uses_external_command_and_refreshes_observation():
    source = SCRIPT.read_text(encoding="utf-8")
    main = next(node for node in _tree().body if isinstance(node, ast.FunctionDef) and node.name == "main")
    loop = next(node for node in ast.walk(main) if isinstance(node, ast.For)
                and ast.unparse(node.target) == "step")
    loop_source = ast.unparse(loop)
    assert "yaw_cfg.external_control = True" in source
    assert "env_cfg.curriculum.task_levels = None" in source
    assert "gym.wrappers.RecordVideo" in source
    assert loop_source.index("command_term.set_external_command(yaw_cmd)") < loop_source.index("obs = env.get_observations()")
    assert loop_source.index("obs = env.get_observations()") < loop_source.index("actions = policy(obs)")
    assert "base_velocity" not in source
    assert "velocity_commands" not in source


def test_trace_row_uses_pre_reset_snapshots_and_converts_reward_rate():
    import torch
    raw = SimpleNamespace(
        _yaw_pos_motion_metrics_current={"true_heading_rate": torch.tensor([.3]),
                                         "support_fl_motor_speed": torch.tensor([1.7])},
        _yaw_pos_geometry_metrics_current={"support_y_separation": torch.tensor([.36])},
        # Deliberately different reset state: the writer must not read it.
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_ang_vel_w=torch.tensor([[0., 0., 99.]])))},
        reward_manager=SimpleNamespace(
            active_terms=["gated_yaw_tracking", "support_y_collapse", "body_angular_xy"],
            _step_reward=torch.tensor([[.08, -.00648, -.002]]),
        ),
    )
    row = _function("_trace_row")(raw, 4, .02, .4, True)
    assert row["step"] == 5 and row["time_s"] == pytest.approx(.10) and row["done"] == 1
    assert row["heading_error"] == pytest.approx(.1)
    assert row["support_fl_motor_speed"] == pytest.approx(1.7)
    assert row["support_y_separation"] == pytest.approx(.36)
    assert row["reward_gated_yaw_tracking"] == pytest.approx(4.)
    assert row["reward_support_y_collapse"] == pytest.approx(-.324)
    assert row["reward_body_angular_xy"] == pytest.approx(-.1)
