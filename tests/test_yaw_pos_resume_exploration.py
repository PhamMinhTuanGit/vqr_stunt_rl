"""POS checkpoint exploration and RSL-RL rollout probability contract."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).parents[1]
TRAIN = ROOT / "scripts/reinforcement_learning/rsl_rl/train.py"
RUNNER_CFG = (
    ROOT
    / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/agents/rsl_rl_ppo_cfg.py"
)


def _pos_resume_helper():
    node = next(
        node for node in ast.parse(TRAIN.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_configure_pos_resume_leg_std"
    )
    namespace = {"torch": torch, "math": math, "OnPolicyRunner": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(TRAIN), "exec"), namespace)
    return namespace[node.name]


def test_pos_entropy_is_unchanged_and_resume_adjustment_follows_load():
    tree = ast.parse(RUNNER_CFG.read_text())
    pos = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "VQRWheelYawFlatPOSPPORunnerCfg")
    assignments = {
        ast.unparse(node.targets[0]): ast.literal_eval(node.value)
        for node in ast.walk(pos) if isinstance(node, ast.Assign)
    }
    assert assignments["self.algorithm.entropy_coef"] == 0.0025
    assert "self.policy.init_noise_std" not in assignments
    assert "self.algorithm.entropy_coef" not in ast.unparse(next(
        node for node in tree.body if isinstance(node, ast.ClassDef)
        and node.name == "VQRWheelYawFlatPPORunnerCfg"
    ))

    main = next(node for node in ast.parse(TRAIN.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    source = ast.unparse(main)
    assert source.index("checkpoint_infos = runner.load(resume_path)") < source.index(
        "_configure_pos_resume_leg_std(runner, env.unwrapped)"
    ) < source.index("runner.learn(")
    assert "task_name == 'Flat-VQR-Wheel-Yaw-POS' and agent_cfg.resume" in source


@pytest.mark.parametrize("wheel_first", [False, True])
def test_loaded_pos_leg_std_changes_only_legs_and_ppo_uses_new_distribution(wheel_first):
    pytest.importorskip("rsl_rl")
    from tensordict import TensorDict
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic

    torch.manual_seed(7)
    obs = TensorDict({"policy": torch.zeros(4, 3), "critic": torch.zeros(4, 3)}, batch_size=[4])
    policy = ActorCritic(
        obs, {"policy": ["policy"], "critic": ["critic"]}, num_actions=16,
        actor_hidden_dims=[8], critic_hidden_dims=[8], init_noise_std=1.0, noise_std_type="log",
    )
    algorithm = PPO(policy, num_learning_epochs=1, num_mini_batches=1, schedule="fixed",
                    desired_kl=None, device="cpu")
    algorithm.init_storage("rl", 4, 1, obs, (16,))

    wheel_names = ("FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL")
    leg_names = tuple(
        f"{side}_{joint}" for side in ("FL", "FR", "HL", "HR")
        for joint in ("HipX_joint", "HipY_joint", "Knee_joint")
    )
    wheel_slice = slice(0, 4) if wheel_first else slice(12, 16)
    leg_slice = slice(4, 16) if wheel_first else slice(0, 12)
    checkpoint_std = torch.full((16,), 0.18)
    checkpoint_std[wheel_slice] = torch.tensor([0.81, 0.62, 0.57, 0.86])
    saved_model = policy.state_dict()
    saved_model["log_std"] = checkpoint_std.log()
    policy.load_state_dict(saved_model)  # Same restore path as runner.load().
    policy.act(obs)  # A previously cached distribution must be discarded.
    assert policy.distribution is not None

    moments = {
        "step": torch.tensor(9.0),
        "exp_avg": torch.ones_like(policy.log_std),
        "exp_avg_sq": torch.ones_like(policy.log_std),
    }
    algorithm.optimizer.state[policy.log_std] = moments
    action_manager = SimpleNamespace(
        active_terms=["joint_vel", "joint_pos"] if wheel_first else ["joint_pos", "joint_vel"],
        action_term_dim=[4, 12] if wheel_first else [12, 4],
        get_term=lambda name: SimpleNamespace(_joint_names=wheel_names if name == "joint_vel" else leg_names),
    )
    scalars = []
    runner = SimpleNamespace(
        alg=algorithm,
        writer=SimpleNamespace(add_scalar=lambda tag, value, step: scalars.append((tag, value, step))),
        log=lambda locs: "logged",
    )

    _pos_resume_helper()(runner, SimpleNamespace(action_manager=action_manager))
    assert policy.distribution is None
    torch.testing.assert_close(policy.log_std[leg_slice].exp(), torch.full((12,), 0.25))
    torch.testing.assert_close(policy.log_std[wheel_slice].exp(), checkpoint_std[wheel_slice])
    assert moments["step"] == 9
    for key in ("exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(moments[key][leg_slice], torch.zeros(12))
        torch.testing.assert_close(moments[key][wheel_slice], torch.ones(4))

    with torch.inference_mode():
        actions = algorithm.act(obs)
        transition = algorithm.transition
        torch.testing.assert_close(transition.action_sigma[:, leg_slice], torch.full((4, 12), 0.25))
        torch.testing.assert_close(transition.action_sigma[:, wheel_slice], checkpoint_std[wheel_slice].expand(4, 4))
        expected_log_prob = torch.distributions.Normal(
            transition.action_mean, transition.action_sigma
        ).log_prob(actions).sum(dim=-1)
        torch.testing.assert_close(transition.actions_log_prob, expected_log_prob)
        algorithm.process_env_step(obs, torch.ones(4), torch.zeros(4, dtype=torch.bool), {})
        torch.testing.assert_close(algorithm.storage.sigma[0, :, leg_slice], torch.full((4, 12), 0.25))
        torch.testing.assert_close(algorithm.storage.sigma[0, :, wheel_slice], checkpoint_std[wheel_slice].expand(4, 4))
        torch.testing.assert_close(algorithm.storage.actions_log_prob[0], expected_log_prob.unsqueeze(-1))

    assert runner.log({"it": 7}) == "logged"
    wheel_std = checkpoint_std[wheel_slice]
    assert scalars == [
        ("Policy/action_std/leg_mean", pytest.approx(0.25), 7),
        ("Policy/action_std/leg_std", pytest.approx(0.0), 7),
        ("Policy/action_std/wheel_mean", pytest.approx(wheel_std.mean().item()), 7),
        ("Policy/action_std/wheel_std", pytest.approx(wheel_std.std(unbiased=False).item()), 7),
    ]
