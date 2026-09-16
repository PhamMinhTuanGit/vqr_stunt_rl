# Standing-to-two-wheel pivot training

Task: `Pivot-FourToTwoWheelRotate-v0`. This extends the existing manager-based
four-to-two task, not the legacy periodic `PivotCommand` or the diagonal-reset
balance/rotate tasks. Robot assets, actuator models, and TO outputs are unchanged.

## Reference and actions

`mdp/to_reference.py` loads only the twelve named leg positions from
`pose_optimization/output/static_equilibrium_FL_HR_sideways.yaml`. Missing,
duplicate, unexpected, or nonfinite joint entries fail early. TO forces and
torques are never policy targets. Keep this YAML available with the checkout.

The command is `[yaw_rate_cmd, lambda]`. After 0.5 s of standing, lambda ramps
over a configurable 2.5 s; the 12 s episode leaves about 9 s for stabilization.
Its clock is independent of RSL-RL's randomized initial episode-length buffer.
The pose reward uses `q_stand + (3*lambda**2 - 2*lambda**3)*(q_TO-q_stand)`.
This never changes the action offset or writes a reference to the robot.

Actions remain twelve standing-centered leg position targets and four wheel
velocity targets. Per-joint scales include the TO pose plus 0.20 rad exploration
room (minimum scale 0.25). For this task only, action targets are clamped to
physical joint limits with 0.05 rad margin. The previous 90% soft-limit clamp
would exclude the TO FL knee even though it satisfies physical limits plus the
margin. Other tasks retain their existing soft-limit behavior.

Actor observations are 52-dimensional; critic observations are 61-dimensional.
Both see yaw/phase once. The critic retains privileged linear velocity, wheel
contacts, and clearance observations. Actions remain 16-dimensional.

## Rewards and curriculum

Pose shaping is `weight[level]*exp(-2*sum(leg_error**2))`. Contact targets in
FL/FR/HL/HR order are `[1,1,1,1]` up to lambda 0.4, interpolate smoothly to
`[1,0,0,1]` by 0.8, and stay there. The contact reward is a soft mean-square
matching score with weight 3. Yaw reward has weight 1.5 and is gated smoothly
over lambda 0.8--1. Broad upright shaping, survival, and moderate effort,
joint-velocity, action-rate, angular-motion and drift penalties remain; there
is no exact TO height/orientation tracking.

| Level | Yaw range (rad/s) | Pose weight | Required continuous final hold (s) |
| --- | --- | --- | --- |
| 0 | 0 | 3.0 | 0.3 |
| 1 | 0 | 2.0 | 2.0 |
| 2 | +/-0.2 | 1.0 | 3.0 |
| 3 | +/-0.5 | 0.4 | 3.0 |
| 4 | +/-1.0 | 0.15 | 4.0 |

Rotation-level command magnitudes are sampled from 25% to 100% of the maximum,
with either sign. Advancement requires at least 256 completed current-level
episodes with success rate >=80%, not a return threshold. Success requires
FL/HR contact, FR/HL no contact, moderate attitude and the required final-phase
hold, without terminal failure. Levels 1+ also require correct contact for 70%
of final-phase samples; levels 2+ require mean absolute final yaw error <0.15
rad/s with a nonzero commanded yaw. Levels are latched at episode reset.

All resets use normal four-wheel standing with small noise. Levels 3+ increase
reset noise and enable modest horizontal velocity pushes only in final phase.
No TO-reset debug mode is enabled. Existing base-contact, inversion, tilt,
height, drift and timeout terminations remain; FR/HL ground contact is not an
early-transition termination.

Metrics include episode return (RSL-RL), transition success, best continuous
two-wheel hold, final-phase contact quality/yaw error, task success, FL/HR
contact ratios, final-phase FR/HL unwanted ratios, pose error, yaw error,
roll/pitch, phase and curriculum level. `fall_rate` counts all non-timeout
failure terminations, including excessive drift. Final-phase ratios of zero
can mean the phase was never reached: interpret them together with phase and
hold duration, not as evidence of clean swing clearance.

## Tests and training

From the repository root, using its Isaac Lab Python environment:

```bash
python source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/tests/test_to_reference.py
python source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/tests/smoke_transition.py --num_envs=4 --steps=200 --headless --device cuda:1
python scripts/reinforcement_learning/rsl_rl/train.py --task=Pivot-FourToTwoWheelRotate-v0 --num_envs=16 --max_iterations=2 --headless --device cuda:1
```

Validated: 3 pure tensor/YAML tests; 4-environment, 200-step random rollout
with finite actor/critic observations and rewards; all five command levels;
2 PPO iterations with 16 environments (768 transitions). This proves pipeline
execution, not learned balance. Observed two-wheel success and hold were zero.
No long training was run. PPO architecture/hyperparameters are unchanged.

For the current machine, the installed editable package points to another
checkout; prepend `PYTHONPATH="$PWD/source/rl_training"` to select this one.
Use `/home/robotics/miniconda3/envs/env_isaaclab/bin/python` if that environment
is not activated. GPU 0 was occupied and caused PhysX OOM; checks succeeded
with `env -u CUDA_VISIBLE_DEVICES` and `--device cuda:1`.

Suggested next full-training command (not executed):

```bash
env -u CUDA_VISIBLE_DEVICES PYTHONPATH="$PWD/source/rl_training" /home/robotics/miniconda3/envs/env_isaaclab/bin/python scripts/reinforcement_learning/rsl_rl/train.py --task=Pivot-FourToTwoWheelRotate-v0 --num_envs=256 --headless --device cuda:1
```

Tune timing via `env.commands.pivot.transition_duration_s=3.0`, for example.
The curriculum manager state is not part of the standard RSL-RL policy
checkpoint: on resume explicitly set `env.commands.pivot.initial_level=N` to
the previously achieved level. Window counters restart. Old four-to-two
checkpoints have different command/reward semantics and should not be treated
as equivalent to new training runs. Reference path overrides should be paired
with an audit of the standing-centered action scales for the new pose.
