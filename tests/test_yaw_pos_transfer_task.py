"""CPU contracts for base reward equivalence, isolation, and transfer certification."""

from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).parents[1]
MDP = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp"
SCRIPT = ROOT / "scripts/reinforcement_learning/rsl_rl/yaw_pos_transfer.py"
CHECKPOINT = ROOT / "logs/rsl_rl/vqr_wheel_yaw_flat/2026-09-21_15-32-03_yaw_dr_curriculum/model_29998.pt"


class Entity:
    def __init__(self, name, body_names=None, joint_names=None, **kwargs):
        self.name = name
        self.body_names = body_names
        self.joint_names = joint_names
        wheels = ["FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL"]
        self.body_ids = [wheels.index(name) for name in body_names] if body_names else slice(None)
        self.joint_ids = kwargs.get("joint_ids", list(range(12)) if joint_names else slice(None))


def load_module(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def command_modules(monkeypatch):
    package = types.ModuleType("transfer_command_test_mdp")
    package.__path__ = [str(MDP)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    utils = types.ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    baseline = types.ModuleType(f"{package.__name__}.commands")

    class BaseCommand:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self.device = "cpu"
            self.num_envs = env.num_envs
            self._command = torch.zeros(env.num_envs, 1)

    baseline.YawRateCommand = BaseCommand
    baseline.YawRateCommandCfg = type("YawRateCommandCfg", (), {})
    monkeypatch.setitem(sys.modules, baseline.__name__, baseline)
    pos = load_module(monkeypatch, f"{package.__name__}.yaw_pos_commands", MDP / "yaw_pos_commands.py")
    transfer = load_module(
        monkeypatch, f"{package.__name__}.yaw_pos_transfer_commands", MDP / "yaw_pos_transfer_commands.py"
    )
    return NS(pos=pos, transfer=transfer)


@pytest.mark.parametrize("yaw_limit", [0.25, 1.00])
def test_transfer_sampler_covers_deadband_preserves_ratio_and_positive_samples(command_modules, yaw_limit):
    cfg = NS(neutral_probability=0.30, neutral_uniform_fraction=0.50,
             deadband=0.10, yaw_rate_range=(0.0, yaw_limit), external_control=False)
    env = NS(num_envs=100000)
    original = command_modules.pos.YawPosCommand(cfg, env)
    transfer = command_modules.transfer.YawPosTransferCommand(cfg, env)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        original._resample_command(slice(None))
        torch.manual_seed(7)
        transfer._resample_command(slice(None))
    old_values, values = original._command[:, 0], transfer._command[:, 0]
    neutral = values <= cfg.deadband
    positive = values > cfg.deadband
    nonzero_neutral = values[(values > 0) & neutral]
    assert torch.isfinite(values).all() and (values >= 0).all() and (values <= yaw_limit).all()
    assert (values == 0).float().mean().item() == pytest.approx(0.15, abs=0.005)
    assert len(nonzero_neutral) / env.num_envs == pytest.approx(0.15, abs=0.005)
    assert neutral.float().mean().item() == pytest.approx(0.30, abs=0.005)
    assert positive.float().mean().item() == pytest.approx(0.70, abs=0.005)
    assert torch.equal(neutral, old_values <= cfg.deadband)
    torch.testing.assert_close(values[positive], old_values[positive], atol=0, rtol=0)
    assert nonzero_neutral.min() < 0.001 and nonzero_neutral.max() > 0.099
    assert nonzero_neutral.mean().item() == pytest.approx(0.05, abs=0.001)
    histogram = torch.histc(nonzero_neutral, bins=10, min=0.0, max=cfg.deadband)
    torch.testing.assert_close(histogram / len(nonzero_neutral), torch.full((10,), 0.10), atol=0.02, rtol=0)
    assert command_modules.transfer.YawPosTransferCommandCfg.neutral_probability == 0.30
    assert command_modules.transfer.YawPosTransferCommandCfg.neutral_uniform_fraction == 0.50


def test_transfer_sampler_respects_selected_envs_empty_selection_and_external_override(command_modules):
    cfg = NS(neutral_probability=0.30, neutral_uniform_fraction=0.50,
             deadband=0.10, yaw_rate_range=(0.0, 0.25), external_control=False)
    term = command_modules.transfer.YawPosTransferCommand(cfg, NS(num_envs=16))
    term._command.fill_(0.20)
    selected = [0, 2, 4, 6]
    term._resample_command(selected)
    unselected = [index for index in range(16) if index not in selected]
    torch.testing.assert_close(term._command[unselected], torch.full((12, 1), 0.20))
    before = term._command.clone()
    term._resample_command([])
    torch.testing.assert_close(term._command, before)
    term.set_external_command(0.25)
    term._resample_command(slice(None))
    torch.testing.assert_close(term._command, torch.full((16, 1), 0.25))
    term.use_training_sampler()
    term._resample_command(slice(None))
    assert (term._command >= 0).all() and (term._command <= 0.25).all()


@pytest.fixture
def modules(monkeypatch):
    package = types.ModuleType("transfer_test_mdp")
    package.__path__ = [str(MDP)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    managers = types.ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = Entity
    monkeypatch.setitem(sys.modules, managers.__name__, managers)
    assets, terrains = types.ModuleType("isaaclab.assets"), types.ModuleType("isaaclab.terrains")
    assets.Articulation = object
    terrains.TerrainImporter = object
    monkeypatch.setitem(sys.modules, assets.__name__, assets)
    monkeypatch.setitem(sys.modules, terrains.__name__, terrains)
    # Execute the real reward functions without importing simulator extensions.
    baseline = types.ModuleType("transfer_test_mdp.rewards")
    baseline.__dict__.update(torch=torch, math=math, SceneEntityCfg=Entity)
    baseline.math_utils = NS(
        euler_xyz_from_quat=lambda q: (torch.zeros(len(q)), torch.zeros(len(q)), torch.zeros(len(q))),
        wrap_to_pi=lambda x: x,
    )
    tree = ast.parse((MDP / "rewards.py").read_text())
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes += [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(MDP / "rewards.py"), "exec"), baseline.__dict__)
    monkeypatch.setitem(sys.modules, baseline.__name__, baseline)
    neutral = load_module(monkeypatch, "transfer_test_mdp.yaw_pos_rewards", MDP / "yaw_pos_rewards.py")
    reward = load_module(monkeypatch, "transfer_test_mdp.yaw_pos_transfer_rewards", MDP / "yaw_pos_transfer_rewards.py")
    # Full diagnostic emissions are covered in the simulator smoke test.
    monkeypatch.setattr(reward, "_record_diagnostics", lambda *args: None)
    curriculum = load_module(monkeypatch, "transfer_test_mdp.curriculums", MDP / "curriculums.py")
    transfer_curriculum = load_module(monkeypatch, "transfer_test_mdp.yaw_pos_transfer_curriculums", MDP / "yaw_pos_transfer_curriculums.py")
    return NS(base=baseline, neutral=neutral, reward=reward, curriculum=transfer_curriculum)


def reward_env(commands):
    command = torch.tensor(commands).reshape(-1, 1)
    n = len(commands)
    torch.manual_seed(9)
    data = NS(
        root_ang_vel_b=torch.randn(n, 3) * 0.20,
        root_lin_vel_b=torch.randn(n, 3) * 0.10,
        root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
        root_pos_w=torch.tensor([[0., 0., 0.49]]).repeat(n, 1),
        body_pos_w=torch.randn(n, 4, 3) * 0.10,
        body_com_pos_w=torch.randn(n, 4, 3) * 0.05,
        joint_pos=torch.randn(n, 16) * 0.25,
        default_joint_pos=torch.zeros(n, 16),
        joint_vel=torch.randn(n, 16),
    )
    data.body_pos_w[:, 0, :2] = torch.tensor([0.25, 0.25])
    data.body_pos_w[:, 3, :2] = torch.tensor([-0.25, -0.25])
    data.body_pos_w[:, :, 2] += 0.10
    robot = NS(data=data, device="cpu", root_physx_view=NS(get_masses=lambda: torch.ones(n, 4)))
    sensor = NS(data=NS(net_forces_w=torch.zeros(n, 4, 3)))
    sensor.data.net_forces_w[:, :, 2] = torch.tensor([20., 0., 0., 20.])

    class Scene(dict):
        env_origins = torch.zeros(n, 3)
        sensors = {"contact_forces": sensor}

    return NS(command_manager=NS(get_command=lambda _: command, get_term=lambda _: NS(cfg=NS(yaw_rate_range=(0., 0.25)))),
              scene=Scene(robot=robot), action_manager=NS(action=torch.ones(n, 16), prev_action=torch.zeros(n, 16)))


def test_deadband_boundary_and_negative_commands(modules):
    env = reward_env([0.0, 0.05, 0.10, 0.100001, 0.25, -0.2])
    _, positive, neutral = modules.reward.mode_masks(env)
    assert positive.tolist() == [False, False, False, True, True, False]
    assert neutral.tolist() == [True, True, True, False, False, True]
    assert not (positive & neutral).any() and (positive | neutral).all()


@pytest.mark.parametrize("function,params", [
    ("yaw_com_support", {"asset_cfg": Entity("robot", body_names=["FL_WHEEL", "HR_WHEEL"]), "std": 0.08}),
    ("yaw_base_height_tracking", {"target_height": 0.49, "error_scale": 0.10}),
    ("yaw_support_span_band_l2", {"asset_cfg": Entity("robot", body_names=["FL_WHEEL", "HR_WHEEL"]), "minimum_span": 0.5, "maximum_span": 0.7, "std": 0.05}),
    ("yaw_lift_clearance", {"asset_cfg": Entity("robot", body_names=["FR_WHEEL", "HL_WHEEL"]), "wheel_radius": 0.091, "target_clearance": 0.05}),
    ("yaw_com_inside_support_segment", {"asset_cfg": Entity("robot", body_names=["FL_WHEEL", "HR_WHEEL"]), "std": 0.05}),
    ("yaw_balance", {"std": 0.25, "nominal_roll": 0., "nominal_pitch": 0.}),
    ("yaw_gated_tracking", {"command_name": "yaw_rate_cmd", "support_sensor_cfg": Entity("contact_forces", body_names=["FL_WHEEL", "HR_WHEEL"]), "lifted_asset_cfg": Entity("robot", body_names=["FR_WHEEL", "HL_WHEEL"]), "wheel_radius": 0.091, "target_clearance": 0.05, "std": 0.30}),
    ("yaw_lifted_wheel_spin_l2", {"asset_cfg": Entity("robot", joint_ids=[13, 14]), "command_name": "yaw_rate_cmd", "yaw_reference": 1.0}),
])
def test_every_positive_reward_is_exact_base_arithmetic(modules, function, params):
    env = reward_env([0., 0.05, 0.10, 0.100001, 0.25] * 4)
    actual = getattr(modules.reward, f"positive_{function}")(env, **params)
    expected = getattr(modules.base, function)(copy.deepcopy(env), **params)
    mask = modules.reward.mode_masks(env)[1]
    torch.testing.assert_close(actual[mask], expected[mask], atol=0, rtol=0)
    assert torch.count_nonzero(actual[~mask]) == 0
    assert inspect.signature(getattr(modules.reward, f"positive_{function}")) == inspect.signature(getattr(modules.base, function))


def test_neutral_rewards_zero_in_positive_and_reward_standing(modules):
    env = reward_env([0., 0.05, 0.10, 0.100001, 0.25])
    env.scene["robot"].data.joint_pos.zero_()
    env.scene["robot"].data.root_ang_vel_b.zero_()
    env.scene["robot"].data.body_pos_w[:, :, 2] = 0.091
    env.scene.sensors["contact_forces"].data.net_forces_w[:, :, 2] = 20
    params = {"command_name": "yaw_rate_cmd", "deadband": 0.1}
    for value in (
        modules.neutral.yaw_pos_four_wheel_contact(env, Entity("contact_forces", body_names=["FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL"]), **params),
        modules.neutral.yaw_pos_four_stand_pose(env, Entity("robot", joint_names=["leg"]), **params),
        modules.neutral.yaw_pos_neutral_landing_progress(env, Entity("robot", body_names=["FR_WHEEL", "HL_WHEEL"]), wheel_radius=0.091, target_clearance=0.05, **params),
        modules.reward.neutral_yaw_tracking(env, std=0.30, **params),
    ):
        assert value.tolist() == pytest.approx([1, 1, 1, 0, 0])


def test_statistics_are_pos_only_across_command_transitions(modules):
    env = reward_env([0., 0.25])
    params = dict(command_name="yaw_rate_cmd", support_sensor_cfg=Entity("contact_forces", body_names=["FL_WHEEL", "HR_WHEEL"]),
                  lifted_asset_cfg=Entity("robot", body_names=["FR_WHEEL", "HL_WHEEL"]), wheel_radius=0.091, target_clearance=0.05, std=0.30)
    modules.reward.positive_yaw_gated_tracking(env, **params)
    assert env._yaw_tracking_metric_samples.tolist() == [0, 1]
    assert env._yaw_support_score_samples.tolist() == [0, 1]
    env.command_manager.get_command("yaw_rate_cmd")[:, 0] = torch.tensor([0.25, 0.])
    modules.reward.positive_yaw_gated_tracking(env, **params)
    assert env._yaw_tracking_metric_samples.tolist() == [1, 1]
    assert env._yaw_active_yaw_score_samples.tolist() == [1, 1]


class Manager:
    def __init__(self, configs):
        self.configs = configs
    def get_term_cfg(self, name):
        return self.configs[name]
    def set_term_cfg(self, name, cfg):
        self.configs[name] = cfg


def curriculum_env():
    rewards = Manager({name: NS(params={"target_clearance": 0.05}, weight=2.) for name in ("lift_clearance", "gated_yaw_tracking", "balance", "neutral_landing_progress")})
    rewards._episode_sums = {"balance": torch.tensor([4., 4.])}
    events = Manager({
        "randomize_apply_external_force_torque": NS(params={}), "randomize_actuator_gains": NS(params={}),
        "randomize_push_robot": NS(params={}), "randomize_reset_base": NS(params={"pose_range": {}, "velocity_range": {}}),
    })
    return NS(num_envs=2, device="cpu", step_dt=0.02, common_step_counter=2000,
              episode_length_buf=torch.tensor([200, 200]), reward_manager=rewards, event_manager=events,
              command_manager=NS(get_term=lambda _: NS(cfg=NS(yaw_rate_range=(0., 0.25)))),
              termination_manager=NS(get_term=lambda _: torch.zeros(2, dtype=torch.bool)),
              _yaw_task_curriculum_stage_start_step=0)


def fill_good_evidence(env, neutral_good=True, active=True, neutral=True):
    for name in ("_yaw_support_score", "_yaw_gate_open", "_yaw_lift_min_progress", "_yaw_active_yaw_score", "_yaw_pos_transfer_balance"):
        setattr(env, name + "_sum", torch.full((2,), 100. if active else 0.))
        setattr(env, name + "_samples", torch.full((2,), 100 if active else 0, dtype=torch.long))
    env._yaw_tracking_metric_samples = torch.full((2,), 100 if active else 0, dtype=torch.long)
    env._yaw_command_abs_sum = torch.full((2,), 25. if active else 0.)
    env._yaw_rate_abs_error_sum = torch.zeros(2)
    env._yaw_edge_tracking_samples = env._yaw_tracking_metric_samples.clone()
    env._yaw_edge_command_abs_sum = env._yaw_command_abs_sum.clone()
    env._yaw_edge_rate_abs_error_sum = torch.zeros(2)
    env._yaw_base_height_min = torch.full((2,), 0.49)
    env._yaw_pos_neutral_samples = torch.full((2,), 100 if neutral else 0, dtype=torch.long)
    env._yaw_pos_neutral_four_contact_sum = torch.full((2,), 100. if neutral_good and neutral else 0.)
    env._yaw_pos_transfer_neutral_pose_score_sum = torch.full((2,), 100. if neutral else 0.)
    env._yaw_pos_neutral_abs_yaw_rate_sum = torch.zeros(2)
    env._yaw_pos_neutral_planar_speed_sum = torch.zeros(2)


def run_curriculum(modules, env, ids=slice(None)):
    return modules.curriculum.yaw_pos_transfer_task_levels(
        env, ids, "yaw_rate_cmd", (0.05, 0.10, 0.15, 0.20), (0.25, 0.4), (0.3, 0.4),
        (0.3, 0.35), (0.2, 0.25), "lift_clearance", "balance", "gated_yaw_tracking", "torso_contact",
        0.35, 0.85, 0.80, 0.75, 0.65, 2, 0.85, 3, 1000, 6000,
    )


@pytest.mark.parametrize("neutral_good", [False, True])
def test_promotion_requires_three_joint_success_windows(modules, neutral_good):
    env = curriculum_env()
    for _ in range(3):
        fill_good_evidence(env, neutral_good=neutral_good)
        before = env.reward_manager._episode_sums["balance"].clone()
        result = run_curriculum(modules, env)
        torch.testing.assert_close(env.reward_manager._episode_sums["balance"], before)
        assert result["balance_score"] == 1  # Half the steps are neutral; POS normalization remains 1.
    assert env._yaw_task_curriculum_stage == (1 if neutral_good else 0)
    assert env.reward_manager.get_term_cfg("neutral_landing_progress").params["target_clearance"] == (0.1 if neutral_good else 0.05)
    assert result["neutral_window_passed"].item() == float(neutral_good)


def test_late_neutral_failure_undoes_base_promotion_and_resets_streak(modules):
    env = curriculum_env()
    for _ in range(2):
        fill_good_evidence(env)
        run_curriculum(modules, env)
    fill_good_evidence(env, neutral_good=False)
    result = run_curriculum(modules, env)
    assert env._yaw_task_curriculum_stage == 0
    assert result["stage_advanced"].item() == result["consecutive_pass_windows"].item() == 0
    assert result["target_clearance"].item() == pytest.approx(0.05)
    assert env._yaw_task_curriculum_stage_start_step == 0


@pytest.mark.parametrize("active,neutral", [(True, False), (False, True)])
def test_missing_mode_evidence_cannot_promote(modules, active, neutral):
    env = curriculum_env()
    for _ in range(4):
        fill_good_evidence(env, active=active, neutral=neutral)
        run_curriculum(modules, env)
    assert env._yaw_task_curriculum_stage == env._yaw_task_curriculum_yaw_stage == 0


def test_curriculum_clears_only_reset_rows(modules):
    env = curriculum_env()
    fill_good_evidence(env)
    run_curriculum(modules, env, [0])
    assert env._yaw_pos_transfer_balance_sum.tolist() == [0., 100.]
    assert env._yaw_pos_neutral_samples.tolist() == [0, 100]


@pytest.mark.parametrize("failed_gate", ["pose", "yaw", "planar", "height", "torso"])
def test_each_neutral_objective_blocks_promotion(modules, failed_gate):
    env = curriculum_env()
    for _ in range(3):
        fill_good_evidence(env)
        if failed_gate == "pose":
            env._yaw_pos_transfer_neutral_pose_score_sum.zero_()
        elif failed_gate == "yaw":
            env._yaw_pos_neutral_abs_yaw_rate_sum.fill_(11.)
        elif failed_gate == "planar":
            env._yaw_pos_neutral_planar_speed_sum.fill_(11.)
        elif failed_gate == "height":
            env._yaw_base_height_min.fill_(0.30)
        else:
            env.termination_manager.get_term = lambda _: torch.ones(2, dtype=torch.bool)
        result = run_curriculum(modules, env)
    assert env._yaw_task_curriculum_stage == 0 and result["neutral_success"].item() == 0


@pytest.mark.parametrize("neutral_good", [True, False])
def test_yaw_and_dr_promotion_preserves_base_stage_order_and_neutral_gate(modules, neutral_good):
    env = curriculum_env()
    env._yaw_task_curriculum_stage = 3
    env.common_step_counter = 6000
    for _ in range(3):
        fill_good_evidence(env, neutral_good=neutral_good)
        result = run_curriculum(modules, env)
    assert env._yaw_task_curriculum_stage == 3
    assert env._yaw_task_curriculum_yaw_stage == (1 if neutral_good else 0)
    assert result["yaw_limit"].item() == pytest.approx(0.4 if neutral_good else 0.25)
    assert result["dr_scale"].item() == pytest.approx(0.4 if neutral_good else 0.3)
    assert result["target_clearance"].item() == pytest.approx(0.20)


def test_registered_old_task_entry_points_are_unchanged():
    import subprocess
    path = "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/__init__.py"
    previous = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, check=True, capture_output=True, text=True).stdout
    def registrations(source):
        return {next(keyword.value.value for keyword in node.value.keywords if keyword.arg == "id"): ast.dump(node)
                for node in ast.parse(source).body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "register"}
    before, after = registrations(previous), registrations((ROOT / path).read_text())
    for name in ("Flat-VQR-Wheel-Yaw", "Flat-VQR-Wheel-Yaw-POS"):
        assert before[name] == after[name]
    assert "Flat-VQR-Wheel-Yaw-POS-Transfer" in after


@pytest.fixture
def loader(monkeypatch):
    return load_module(monkeypatch, "transfer_loader_test", SCRIPT)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Local successful checkpoint required")
def test_loader_rejects_same_dimensions_with_changed_order_or_scale(loader):
    with (CHECKPOINT.parent / "params/env.yaml").open() as f:
        source = loader.yaml.load(f, Loader=loader._ConfigLoader)
    loader.audit_checkpoint_contract(CHECKPOINT, source)
    changed = copy.deepcopy(source)
    changed["actions"]["joint_vel"]["joint_names"].reverse()
    with pytest.raises(RuntimeError, match="actions"):
        loader.audit_checkpoint_contract(CHECKPOINT, changed)
    changed = copy.deepcopy(source)
    changed["observations"]["policy"]["joint_vel"]["scale"] = 0.5
    with pytest.raises(RuntimeError, match="policy_terms"):
        loader.audit_checkpoint_contract(CHECKPOINT, changed)
    changed = copy.deepcopy(source)
    changed["observations"]["policy"]["base_ang_vel"] = changed["observations"]["policy"].pop("base_ang_vel")
    with pytest.raises(RuntimeError, match="policy_terms"):
        loader.audit_checkpoint_contract(CHECKPOINT, changed)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Local successful checkpoint required")
@pytest.mark.parametrize("corruption", ["shape", "keys", "nonfinite"])
def test_loader_refuses_partial_or_corrupt_policy_before_runner_load(loader, tmp_path, corruption):
    import shutil
    (tmp_path / "params").mkdir()
    shutil.copyfile(CHECKPOINT.parent / "params/env.yaml", tmp_path / "params/env.yaml")
    with (tmp_path / "params/env.yaml").open() as f:
        config = loader.yaml.load(f, Loader=loader._ConfigLoader)
    saved = torch.load(CHECKPOINT, weights_only=False, map_location="cpu")
    target_state = copy.deepcopy(saved["model_state_dict"])
    if corruption == "shape":
        saved["model_state_dict"]["actor.0.weight"] = saved["model_state_dict"]["actor.0.weight"][:, :-1]
    elif corruption == "keys":
        del saved["model_state_dict"]["critic.0.bias"]
    else:
        saved["model_state_dict"]["log_std"][0] = float("nan")
    torch.save(saved, tmp_path / "model.pt")
    runner = NS(alg=NS(optimizer=NS(state={}), policy=NS(state_dict=lambda: target_state)))
    task = NS(common_step_counter=0, cfg=config)
    with pytest.raises(RuntimeError, match="mismatch|partial|non-finite"):
        loader.load_base_policy(runner, task, tmp_path / "model.pt")


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Local successful checkpoint required")
def test_real_runner_load_preserves_full_policy_and_resets_optimizer_iteration_curriculum(loader):
    pytest.importorskip("rsl_rl")
    from rsl_rl.runners import OnPolicyRunner
    from tensordict import TensorDict
    with (CHECKPOINT.parent / "params/env.yaml").open() as f:
        config = loader.yaml.load(f, Loader=loader._ConfigLoader)
    with (CHECKPOINT.parent / "params/agent.yaml").open() as f:
        runner_cfg = loader.yaml.load(f, Loader=loader._ConfigLoader)
    runner_cfg["device"] = "cpu"
    obs = TensorDict({"policy": torch.randn(32, 55), "critic": torch.randn(32, 83)}, batch_size=[32])
    vec = NS(num_envs=32, num_actions=16, device="cpu", get_observations=lambda: obs)
    runner = OnPolicyRunner(vec, runner_cfg, device="cpu")
    task = NS(common_step_counter=0, _yaw_task_curriculum_stage=3, _yaw_task_curriculum_yaw_stage=4)
    def curriculum(env, ids):
        env._yaw_task_curriculum_stage = env._yaw_task_curriculum_yaw_stage = 0
    task.cfg = NS(to_dict=lambda: config, curriculum=NS(task_levels=NS(func=curriculum, params={"dr_scale_levels": [0.3]})))
    task.cfg.curriculum.task_levels.func = lambda env, ids, **kwargs: curriculum(env, ids)
    task.command_manager = NS(get_term=lambda _: NS(cfg=NS(yaw_rate_range=(0., 0.25))))
    task.reward_manager = NS(get_term_cfg=lambda _: NS(params={"target_clearance": 0.05}))
    task.reset = lambda: None
    report = loader.load_base_policy(runner, task, CHECKPOINT)
    assert report["source_iteration"] == 29998 and runner.current_learning_iteration == 0
    assert report["optimizer_fresh"] and not runner.alg.optimizer.state
    assert report["learned_noise_loaded"]
    assert runner.alg.policy.act_inference(obs).shape == (32, 16)
    assert runner.alg.policy.evaluate(obs).shape == (32, 1)
