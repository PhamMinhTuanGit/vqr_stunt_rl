"""Static startup and registration checks for the independent staged task."""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel"
MDP = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp"
TRAIN = ROOT / "scripts/reinforcement_learning/rsl_rl/train.py"


def _class(path, name):
    return next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name == name)


def test_registration_and_runner_are_isolated():
    registry = ast.parse((CONFIG / "__init__.py").read_text())
    registrations = {}
    for node in ast.walk(registry):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "register":
            arguments = {item.arg: item.value for item in node.keywords}
            registrations[ast.literal_eval(arguments["id"])] = ast.unparse(arguments["kwargs"])
    assert "Flat-VQR-Wheel-Yaw-FSM-Staged" in registrations
    assert "yaw_env_fsm_staged_cfg:VQRWheelFlatEnvFSMStagedCfg" in registrations["Flat-VQR-Wheel-Yaw-FSM-Staged"]
    assert "VQRWheelYawFlatFSMStagedPPORunnerCfg" in registrations["Flat-VQR-Wheel-Yaw-FSM-Staged"]
    assert "yaw_env_fsm_cfg:VQRWheelFlatEnvFSMCfg" in registrations["Flat-VQR-Wheel-Yaw-FSM"]
    assert "VQRWheelYawFlatFSMPPORunnerCfg" in registrations["Flat-VQR-Wheel-Yaw-FSM"]
    runner = _class(CONFIG / "agents/rsl_rl_ppo_cfg.py", "VQRWheelYawFlatFSMStagedPPORunnerCfg")
    assert "vqr_wheel_yaw_flat_fsm_staged" in ast.unparse(runner)


def test_staged_runner_exploration_overrides_do_not_change_fsm_defaults():
    runner_cfg = CONFIG / "agents/rsl_rl_ppo_cfg.py"
    rough = _class(runner_cfg, "VQRWheelRoughPPORunnerCfg")
    defaults = {
        item.targets[0].id: item.value for item in rough.body
        if isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Name)
    }
    policy_args = {item.arg: item.value for item in defaults["policy"].keywords}
    algorithm_args = {item.arg: item.value for item in defaults["algorithm"].keywords}
    assert ast.literal_eval(policy_args["init_noise_std"]) == 1.0
    assert ast.literal_eval(algorithm_args["entropy_coef"]) == 0.01

    for name in (
        "VQRWheelFlatPPORunnerCfg", "VQRWheelYawFlatPPORunnerCfg",
        "VQRWheelYawFlatFSMPPORunnerCfg",
    ):
        parent_source = ast.unparse(_class(runner_cfg, name))
        assert "self.policy.init_noise_std" not in parent_source
        assert "self.algorithm.entropy_coef" not in parent_source

    staged = _class(runner_cfg, "VQRWheelYawFlatFSMStagedPPORunnerCfg")
    assert [base.id for base in staged.bases] == ["VQRWheelYawFlatFSMPPORunnerCfg"]
    overrides = {
        ast.unparse(item.targets[0]): ast.literal_eval(item.value)
        for item in ast.walk(staged) if isinstance(item, ast.Assign)
    }
    assert overrides["self.policy.init_noise_std"] == 0.10
    assert overrides["self.algorithm.entropy_coef"] == 0.002


def _staged_exploration_helpers():
    tree = ast.parse(TRAIN.read_text())
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_STAGED_YAW_NOISE_STD_BY_PHASE"
                    for target in node.targets))
        or (isinstance(node, ast.FunctionDef) and node.name == "_install_staged_yaw_exploration")
    ]
    namespace = {"torch": torch, "math": math, "StagedPromotion": lambda: SimpleNamespace(phase=0),
                 "OnPolicyRunner": object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAIN), "exec"), namespace)
    return namespace


def _fake_staged_runner(initial_std=0.8):
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.noise_std_type = "log"
            self.state_dependent_std = False
            self.log_std = torch.nn.Parameter(torch.full((2,), math.log(initial_std)))

    policy = Policy()
    updates = []
    algorithm = SimpleNamespace(
        policy=policy, optimizer=torch.optim.Adam(policy.parameters()),
        update=lambda: updates.append(1) or {"loss": 1.0},
    )
    scalars = []
    writer = SimpleNamespace(add_scalar=lambda tag, value, step: scalars.append((tag, value, step)))
    return SimpleNamespace(alg=algorithm, log=lambda locs: "logged", writer=writer, scalars=scalars), updates


def test_staged_exploration_applies_phase_zero_and_each_later_target():
    install = _staged_exploration_helpers()["_install_staged_yaw_exploration"]
    runner, _ = _fake_staged_runner()
    promotion = SimpleNamespace(phase=0, clearance_index=0, yaw_index=0)
    install(runner, SimpleNamespace(_staged_yaw_promotion=promotion))
    assert not runner.alg.policy.log_std.requires_grad
    assert torch.exp(runner.alg.policy.log_std).tolist() == pytest.approx([0.10, 0.10])
    for phase, target in ((1, 0.15), (2, 0.25), (3, 0.20)):
        promotion.phase = phase
        runner.alg.update()
        assert torch.exp(runner.alg.policy.log_std).tolist() == pytest.approx([target, target])


def test_staged_exploration_updates_once_per_phase_promotion():
    install = _staged_exploration_helpers()["_install_staged_yaw_exploration"]
    runner, updates = _fake_staged_runner()
    promotion = SimpleNamespace(phase=0, clearance_index=0, yaw_index=0)
    install(runner, SimpleNamespace(_staged_yaw_promotion=promotion))
    parameter = runner.alg.policy.log_std
    runner.alg.optimizer.state[parameter] = {"marker": "same phase"}
    runner.alg.update()
    assert runner.alg.optimizer.state[parameter]["marker"] == "same phase"
    promotion.phase = 1
    runner.alg.update()
    assert parameter not in runner.alg.optimizer.state
    assert torch.exp(parameter).tolist() == pytest.approx([0.15, 0.15])
    runner.alg.optimizer.state[parameter] = {"marker": "after promotion"}
    runner.alg.update()
    assert runner.alg.optimizer.state[parameter]["marker"] == "after promotion"
    promotion.phase = 2
    runner.alg.update()
    runner.alg.optimizer.state[parameter] = {"marker": "clearance change"}
    promotion.clearance_index = 1
    runner.alg.update()
    assert runner.alg.optimizer.state[parameter]["marker"] == "clearance change"
    assert len(updates) == 5


def test_staged_resume_repairs_loaded_noise_to_restored_phase():
    install = _staged_exploration_helpers()["_install_staged_yaw_exploration"]
    checkpoint_runner, _ = _fake_staged_runner(initial_std=0.8)
    saved_model = checkpoint_runner.alg.policy.state_dict()
    resumed_runner, _ = _fake_staged_runner(initial_std=0.1)
    resumed_runner.alg.policy.load_state_dict(saved_model)
    assert torch.exp(resumed_runner.alg.policy.log_std).tolist() == pytest.approx([0.8, 0.8])
    restored_curriculum = SimpleNamespace(_staged_yaw_promotion=SimpleNamespace(phase=2))
    install(resumed_runner, restored_curriculum)
    assert torch.exp(resumed_runner.alg.policy.log_std).tolist() == pytest.approx([0.25, 0.25])
    assert not resumed_runner.alg.policy.log_std.requires_grad
    assert torch.exp(resumed_runner.alg.policy.state_dict()["log_std"]).tolist() == pytest.approx([0.25, 0.25])


def test_staged_exploration_telemetry_reads_current_parameter():
    install = _staged_exploration_helpers()["_install_staged_yaw_exploration"]
    runner, _ = _fake_staged_runner()
    promotion = SimpleNamespace(phase=0)
    install(runner, SimpleNamespace(_staged_yaw_promotion=promotion))
    assert runner.log({"it": 7}) == "logged"
    assert runner.scalars == [
        ("Curriculum/staged_phase", 0, 7),
        ("Policy/staged_target_exploration_std", 0.10, 7),
        ("Policy/staged_actual_mean_actor_std", pytest.approx(0.10), 7),
    ]
    promotion.phase = 1
    runner.alg.update()
    runner.log({"it": 8})
    assert runner.scalars[-3:] == [
        ("Curriculum/staged_phase", 1, 8),
        ("Policy/staged_target_exploration_std", 0.15, 8),
        ("Policy/staged_actual_mean_actor_std", pytest.approx(0.15), 8),
    ]


def test_rsl_rl_actor_samples_with_staged_phase_zero_std():
    pytest.importorskip("rsl_rl")
    from tensordict import TensorDict
    from rsl_rl.modules import ActorCritic

    obs = TensorDict({"policy": torch.zeros(4, 3), "critic": torch.zeros(4, 3)}, batch_size=[4])
    policy = ActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, num_actions=2,
                         actor_hidden_dims=[8], critic_hidden_dims=[8], init_noise_std=0.10,
                         noise_std_type="log")
    optimizer = torch.optim.Adam(policy.parameters())
    runner = SimpleNamespace(
        alg=SimpleNamespace(policy=policy, optimizer=optimizer, update=lambda: {}),
        log=lambda locs: None, writer=SimpleNamespace(add_scalar=lambda *args: None),
    )
    promotion = SimpleNamespace(phase=0)
    _staged_exploration_helpers()["_install_staged_yaw_exploration"](
        runner, SimpleNamespace(_staged_yaw_promotion=promotion)
    )
    policy.act(obs)
    torch.testing.assert_close(policy.action_std, torch.full((4, 2), 0.10))
    optimizer.zero_grad()
    policy.act_inference(obs).square().mean().backward()
    optimizer.step()
    policy.act(obs)
    torch.testing.assert_close(policy.action_std, torch.full((4, 2), 0.10))
    promotion.phase = 1
    runner.alg.update()
    policy.act(obs)
    torch.testing.assert_close(policy.action_std, torch.full((4, 2), 0.15))


def test_rsl_rl_rollout_promotion_keeps_stored_action_probabilities():
    pytest.importorskip("rsl_rl")
    from tensordict import TensorDict
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic

    obs = TensorDict({"policy": torch.zeros(4, 3), "critic": torch.zeros(4, 3)}, batch_size=[4])
    policy = ActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, num_actions=2,
                         actor_hidden_dims=[8], critic_hidden_dims=[8], init_noise_std=0.10,
                         noise_std_type="log")
    algorithm = PPO(policy, num_learning_epochs=1, num_mini_batches=1, schedule="fixed",
                    desired_kl=None, device="cpu")
    algorithm.init_storage("rl", 4, 2, obs, (2,))
    std_during_update = []
    original_update = algorithm.update

    def monitored_update():
        std_during_update.append(policy.log_std.detach().exp().clone())
        result = original_update()
        std_during_update.append(policy.log_std.detach().exp().clone())
        return result

    algorithm.update = monitored_update
    runner = SimpleNamespace(alg=algorithm, log=lambda locs: None,
                             writer=SimpleNamespace(add_scalar=lambda *args: None))
    promotion = SimpleNamespace(phase=0)
    _staged_exploration_helpers()["_install_staged_yaw_exploration"](
        runner, SimpleNamespace(_staged_yaw_promotion=promotion)
    )

    with torch.inference_mode():
        for step in range(2):
            algorithm.act(obs)
            if step == 0:
                # Curriculum promotion happens inside env.step, after sampling.
                promotion.phase = 1
            algorithm.process_env_step(obs, torch.ones(4), torch.zeros(4, dtype=torch.bool), {})
            torch.testing.assert_close(policy.log_std.exp(), torch.full((2,), 0.10))
        stored_log_prob = algorithm.storage.actions_log_prob.clone()
        stored_sigma = algorithm.storage.sigma.clone()
        expected_log_prob = torch.distributions.Normal(
            algorithm.storage.mu, stored_sigma
        ).log_prob(algorithm.storage.actions).sum(dim=-1, keepdim=True)
        torch.testing.assert_close(stored_log_prob, expected_log_prob)
        algorithm.compute_returns(obs)

    runner.alg.update()
    for value in std_during_update:
        torch.testing.assert_close(value, torch.full((2,), 0.10))
    assert policy.log_std.grad is None
    torch.testing.assert_close(algorithm.storage.actions_log_prob, stored_log_prob)
    torch.testing.assert_close(algorithm.storage.sigma, stored_sigma)
    torch.testing.assert_close(policy.log_std.exp(), torch.full((2,), 0.15))
    algorithm.act(obs)
    torch.testing.assert_close(algorithm.transition.action_sigma, torch.full((4, 2), 0.15))


def test_config_inherits_actor_rewards_and_nominal_rough_pose():
    staged = CONFIG / "yaw_env_fsm_staged_cfg.py"
    source = staged.read_text()
    node = _class(staged, "VQRWheelFlatEnvFSMStagedCfg")
    assert [base.id for base in node.bases] == ["VQRWheelFlatEnvFSMCfg"]
    assert "self.events.randomize_reset_joints.params[\"position_range\"] = (1.0, 1.0)" in source
    assert "self.events.randomize_reset_joints.params[\"velocity_range\"] = (0.0, 0.0)" in source
    assert '"roll": (0.0, 0.0), "pitch": (0.0, 0.0)' in source
    assert "axis: (0.0, 0.0)" in source
    assert "CLEARANCE_LEVELS = (0.02, 0.03, 0.05)" in source
    assert "YAW_RATE_LEVELS = (0.15, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00)" in source
    assert "hold_window\": 2048" in source and "directional_window\": 1024" in source
    asset = (ROOT / "source/rl_training/rl_training/assets/deeprobotics.py").read_text()
    assert "pos=(0.0, 0.0, 0.45)" in asset
    for joint, value in (("HipX", "0.0"), ("HipY", "-0.65"), ("Knee", "1.3")):
        assert f'".*_{joint}_joint": {value}' in asset
    assert 'joint_vel={".*": 0.0}' in asset
    # The staged config inherits the FSM observation group unchanged. Its
    # actor contains the command and FSM state; contact/readiness stay critic-only.
    observations = _class(CONFIG / "yaw_env_fsm_cfg.py", "VQRWheelFSMObservationsCfg")
    policy = next(item for item in observations.body
                  if isinstance(item, ast.ClassDef) and item.name == "PolicyCfg")
    assert [item.targets[0].id for item in policy.body if isinstance(item, ast.Assign)] == ["fsm_state"]
    assert "observations:" not in source


def test_promotion_reset_is_a_staged_only_truncation():
    staged = CONFIG / "yaw_env_fsm_staged_cfg.py"
    terminations = _class(staged, "VQRWheelStagedTerminationsCfg")
    terms = {item.targets[0].id: item.value for item in terminations.body
             if isinstance(item, ast.Assign) and isinstance(item.value, ast.Call)}
    assert set(terms) == {"staged_complete", "staged_promotion_reset"}
    promotion = terms["staged_promotion_reset"]
    arguments = {keyword.arg: keyword.value for keyword in promotion.keywords}
    assert ast.unparse(arguments["func"]) == "mdp.staged_promotion_reset"
    assert ast.literal_eval(arguments["time_out"]) is True
    original = _class(CONFIG / "yaw_env_fsm_cfg.py", "VQRWheelFSMTerminationsCfg")
    assert "staged_promotion_reset" not in ast.unparse(original)


def test_training_has_distinct_startup_and_checkpoint_state():
    tree = ast.parse(TRAIN.read_text())
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {
        "_verify_staged_yaw_contract", "_export_staged_yaw_state",
        "_restore_staged_yaw_state", "_install_staged_yaw_checkpointing",
    } <= names
    source = TRAIN.read_text()
    assert '"yaw_fsm_staged_curriculum"' in source
    assert 'task_name == "Flat-VQR-Wheel-Yaw-FSM-Staged"' in source
    assert "_verify_staged_yaw_contract(env, env_cfg)" in source
    assert "_install_staged_yaw_checkpointing(runner, staged_yaw_task_env)" in source
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    staged_branch = next(node for node in ast.walk(main)
                         if isinstance(node, ast.If) and ast.unparse(node.test) == "staged_yaw_task_env is not None")
    exploration_calls = [node for node in ast.walk(main)
                         if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name)
                         and node.func.id == "_install_staged_yaw_exploration"]
    assert len(exploration_calls) == 1
    assert exploration_calls[0] in list(ast.walk(staged_branch))
    main_source = ast.get_source_segment(source, main)
    assert (main_source.index("checkpoint_infos = runner.load(resume_path)")
            < main_source.index("_restore_staged_yaw_state(staged_yaw_task_env, checkpoint_infos)")
            < main_source.index("_install_staged_yaw_exploration(runner, staged_yaw_task_env)")
            < main_source.index("runner.learn("))
    mdp_source = (MDP / "__init__.py").read_text()
    assert "from .staged_yaw_curriculum import *" in mdp_source


def test_checkpoint_round_trip_preserves_partial_directional_window():
    spec = importlib.util.spec_from_file_location("staged_curriculum_contract", MDP / "staged_yaw_curriculum.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = ast.parse(TRAIN.read_text())
    names = {"_export_staged_yaw_state", "_restore_staged_yaw_state", "_install_staged_yaw_checkpointing"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"StagedPromotion": module.StagedPromotion, "torch": torch,
                 "_STAGED_YAW_CHECKPOINT_KEY": "yaw_fsm_staged_curriculum", "OnPolicyRunner": object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAIN), "exec"), namespace)

    params = {
        "command_name": "yaw_rate_cmd", "clearance_levels": (.02, .03, .05),
        "yaw_rate_levels": (.15, .25), "dr_scale_levels": (.30, 1.0),
        "tracking_ratio_thresholds": (.30, .35),
        "hold_window": 2048, "directional_window": 1024, "required_windows": 3,
    }
    promotion = module.StagedPromotion(phase=2, clearance_index=1)
    promotion.consecutive_passes = 2
    promotion.record("pos", yaw=1, support=1, pose=1)
    promotion.record("neg", yaw=0, support=1, pose=0)
    reapplied = []
    resets = []
    term_cfg = SimpleNamespace(params=params, func=lambda env, ids, **kwargs: reapplied.append(ids))
    command = SimpleNamespace(reset=lambda ids: resets.append(ids.tolist()))
    env = SimpleNamespace(
        cfg=SimpleNamespace(curriculum=SimpleNamespace(task_levels=term_cfg)),
        _staged_yaw_promotion=promotion, num_envs=2, device="cpu",
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )
    payload = namespace["_export_staged_yaw_state"](env)
    env._staged_yaw_promotion = module.StagedPromotion()
    assert namespace["_restore_staged_yaw_state"](env, {"yaw_fsm_staged_curriculum": payload})
    assert env._staged_yaw_promotion.export() == promotion.export()
    assert reapplied == [[]]
    assert resets == [[0, 1]]

    resumed_runner, _ = _fake_staged_runner(initial_std=0.8)
    _staged_exploration_helpers()["_install_staged_yaw_exploration"](resumed_runner, env)
    assert torch.exp(resumed_runner.alg.policy.log_std).tolist() == pytest.approx([0.25, 0.25])

    saved = []
    runner = SimpleNamespace(save=lambda path, infos: saved.append((path, infos)))
    namespace["_install_staged_yaw_checkpointing"](runner, env)
    runner.save("checkpoint.pt", {"other": 1})
    assert saved[0][1]["yaw_fsm_staged_curriculum"]["promotion"] == promotion.export()
    assert saved[0][1]["other"] == 1
