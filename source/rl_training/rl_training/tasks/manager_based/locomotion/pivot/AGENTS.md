# AGENTS.md

## Scope

You are working on the Pivot reinforcement-learning task for a wheeled quadruped robot.

All implementation work MUST stay inside:

```text
source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/
```

Do not modify files outside this directory.

In particular, DO NOT modify:

```text
tasks/manager_based/locomotion/velocity/
assets/
scripts/
train.py
play.py
shared Isaac Lab infrastructure
```

Existing code outside `pivot/` may be imported and reused, but must not be edited.

---

# Project Goal

Implement RL environments for a wheeled quadruped that performs in-place rotation while balancing on two diagonal wheels.

Development sequence:

```text
Phase A:
Two-wheel diagonal balance

Phase B:
Two-wheel diagonal balance + yaw rotation

Future Phase C:
4-wheel stance
→ CoM shift
→ unload diagonal
→ lift diagonal
→ 2-wheel balance
→ rotate
```

Only Phase A and Phase B are currently in scope.

Do NOT implement the 4-wheel → 2-wheel transition unless explicitly requested.

---

# Target Contact Configuration

Initial support pair:

```text
FL + HR = support wheels
FR + HL = lifted wheels
```

Do not automatically alternate diagonal pairs in the initial implementation.

Robot names and ordering must come from:

```text
pivot/config/wheeled/vqr/robot_cfg.py
```

Use its constants rather than inventing or duplicating names.

---

# Required Structure

```text
pivot/
├── __init__.py
├── mdp/
│   ├── __init__.py
│   ├── rewards.py
│   ├── observations.py
│   ├── events.py
│   └── terminations.py
└── config/
    ├── __init__.py
    └── wheeled/
        ├── __init__.py
        └── vqr/
            ├── __init__.py
            ├── robot_cfg.py
            ├── balance_env_cfg.py
            ├── rotate_env_cfg.py
            └── agents/
                ├── __init__.py
                └── rsl_rl_ppo_cfg.py
```

Avoid unnecessary files and unnecessary duplication.

---

# Framework

Keep the existing stack:

```text
Isaac Sim
Isaac Lab
ManagerBasedRLEnv
RSL-RL
PPO
```

Do not introduce another RL framework or a custom PPO implementation unless explicitly requested.

---

# CRITICAL: Command Execution Contract

This section is authoritative for every agent working in this directory.

Agents MUST NOT invent launch commands, absolute environment paths, Python executables, task IDs, or script paths.

## 1. Repository root

All project commands must be executed from the repository root: the directory containing both:

```text
scripts/
source/
```

Before running repository scripts, verify the current directory with:

```bash
pwd
ls scripts source
```

If `scripts` and `source` are not both present, do not run training or play commands until the repository root is located.

Do NOT assume a fixed repository path such as:

```text
/home/robotics/...
/home/tuanpm/...
~/IsaacLab/...
```

The checkout path differs between local and server machines.

## 2. Python environment

Use the `python` executable from the environment that is already activated by the user/session.

Before simulator or training commands, inspect it with:

```bash
which python
python --version
```

This repository expects Python 3.11 with the installed Isaac Lab environment.

Do NOT guess or automatically run a machine-specific activation command such as:

```bash
conda activate <invented-env-name>
source /home/.../conda.sh
```

unless that exact environment/path is already provided by the user or current shell context.

Do NOT use system `/usr/bin/python` when the active Isaac Lab environment provides another interpreter.

## 3. Package import

The normal installation method is editable installation:

```bash
python -m pip install -e source/rl_training
```

Do not reinstall the package on every validation run.

Do not add arbitrary `PYTHONPATH` overrides by default.

Only use a temporary `PYTHONPATH` override when diagnosing an existing editable-install mismatch, and state clearly why it is needed.

## 4. Canonical task-registry check

Before training a newly added task, first verify that Gym registration is visible with the repository's existing tool:

```bash
python scripts/tools/list_envs.py
```

Confirm the exact expected task ID appears.

Current Pivot task IDs are:

```text
Pivot-TwoWheelBalance-v0
Pivot-TwoWheelRotate-v0
```

Do NOT silently substitute old experimental IDs such as:

```text
Pivot-VQR-Wheel-v0
```

unless the user explicitly asks to work on that legacy task.

## 5. Canonical RSL-RL training command

The only normal single-GPU training entrypoint for this repository is:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=<TASK_ID> \
  --headless
```

For M1 smoke testing:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Pivot-TwoWheelBalance-v0 \
  --num_envs=16 \
  --max_iterations=2 \
  --headless
```

For M2 smoke testing:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Pivot-TwoWheelRotate-v0 \
  --num_envs=16 \
  --max_iterations=2 \
  --headless
```

Only add:

```text
--device cuda:0
```

when the target machine/device is known or the user explicitly requests it.

Do NOT hard-code `cuda:0` inside environment or PPO config merely because a validation machine uses GPU 0.

Do NOT launch 1024/4096 environments before the small smoke test passes.

## 6. Canonical play command

Use:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=<TASK_ID> \
  --num_envs=1
```

Add checkpoint arguments only when a real checkpoint path/run is known.

Do not invent checkpoint files or log directories.

## 7. Canonical resume pattern

When the user explicitly asks to resume training, use the repository-supported form:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=<TASK_ID> \
  --resume \
  --load_run=<RUN_NAME> \
  --checkpoint=<CHECKPOINT> \
  --headless
```

Do not infer `<RUN_NAME>` or `<CHECKPOINT>` if they have not been discovered from the actual logs.

## 8. Multi-GPU

Do not use distributed training unless explicitly requested.

When it is requested, follow the repository pattern:

```bash
python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node=<NUM_GPUS> \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task=<TASK_ID> \
  --headless \
  --distributed
```

Do not invent `torchrun` arguments or assume the number of GPUs.

## 9. Isaac Lab launcher usage

For repository training/play, prefer the repository scripts above.

Do NOT replace them with guessed commands such as:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py ...
python -m <guessed.module> ...
isaac-sim python ...
```

unless the user explicitly asks for an IsaacLab-source-tree launch or the repository's standard `python` invocation has been proven unavailable.

The repository README uses direct `python` invocation from the activated Isaac Lab environment; follow that convention.

## 10. One-off validation scripts

Short inline Python diagnostics are allowed, but they must obey these rules:

1. launch `AppLauncher` before importing modules that require the running Isaac Sim application;
2. import `rl_training.tasks` before querying Gym task registration when needed;
3. close the environment and simulation app cleanly;
4. do not present an unexecuted diagnostic as a passed validation;
5. do not claim CUDA/PhysX runtime success if the current machine cannot create a CUDA scene.

## 11. Never claim a command passed unless it actually ran

Reports must distinguish:

```text
STRUCTURAL PASS
IMPORT PASS
CPU/STATIC CHECK PASS
SIMULATOR RUNTIME PASS
PPO SMOKE PASS
```

If the current machine lacks GPU/driver access, report the runtime test as BLOCKED rather than PASS.

## 12. Before giving the user a command

Verify all of the following:

```text
script path exists in this repository
exact task ID is registered
CLI flag is supported by that script
command assumes the repository root as cwd
no invented absolute path is embedded
environment/device assumptions are stated rather than guessed
```

If uncertain, inspect the repository first. Do not guess.

---

# Robot Model and Configuration

Prefer the existing USD directly. Do not perform runtime URDF conversion unless necessary.

The robot-specific articulation configuration belongs in:

```text
config/wheeled/vqr/robot_cfg.py
```

The robot is floating-base.

Actuators are separated into:

```text
leg actuators
wheel actuators
```

Legs use position-based PD control.
Wheels use velocity control.

Use real/existing actuator limits when available. Do not invent aggressive torque or velocity limits.

Before changing robot names, inspect the actual USD/URDF and current `robot_cfg.py`.

Never invent joint/body names.

---

# Action Space

Policy action:

```text
12 leg position residuals
+
4 wheel velocity targets
=
16 actions
```

Use:

```text
JointPositionActionCfg
```

for legs and:

```text
JointVelocityActionCfg
```

for wheels.

Initial conservative scales:

```text
leg position: about 0.15–0.20 rad
wheel velocity: about 3–5 rad/s
```

Do not use direct torque actions for the initial tasks.

---

# Actor Observation

Actor observations must be deployable on the physical robot.

Recommended terms:

```text
base angular velocity
projected gravity
yaw-rate command (Rotate only; Balance may omit/zero it)
leg joint position
leg joint velocity
wheel velocity
previous action
```

Do NOT put simulator-only privileged quantities in the actor, including:

```text
absolute world position
ground-truth contact force
perfect base linear velocity
future state
reward information
```

unless explicitly justified.

---

# Critic Observation

The critic may use privileged simulation information, for example:

```text
base linear velocity
contact state
contact forces
base height
wheel clearance
additional simulator state
```

Keep actor and critic observation groups separate.
Prefer asymmetric actor-critic.

---

# Phase A — Two-Wheel Balance

Task:

```text
Pivot-TwoWheelBalance-v0
```

File:

```text
balance_env_cfg.py
```

Target:

```text
FL = support/contact
HR = support/contact
FR = lifted
HL = lifted
```

The Balance task focuses on balance/contact maintenance, not yaw rotation and not the 4-wheel → 2-wheel transition.

Reset near a valid two-wheel state using small perturbations only:

```text
small joint noise
small roll/pitch noise
random yaw
small angular velocity
near-zero wheel velocity
```

Do not initially add large mass/CoM randomization, pushes, motor-strength randomization, large delay, or strong terrain randomization.

Balance rewards should remain small and purposeful, roughly 5–7 terms:

```text
support contact
lifted diagonal / clearance
balance around nominal equilibrium
roll/pitch angular stability
planar drift
small effort penalty
action-rate penalty
```

Do not force roll=0 and pitch=0 if the physical diagonal equilibrium requires lean.

Terminate only clearly unrecoverable states such as torso contact, inversion, excessive attitude error, collapsed height, or excessive drift.

Do not immediately terminate a brief lifted-wheel touch.

---

# Phase B — Two-Wheel Rotate

Task:

```text
Pivot-TwoWheelRotate-v0
```

File:

```text
rotate_env_cfg.py
```

Rotate MUST inherit from Balance rather than duplicate it.

Preferred pattern:

```python
class VQRTwoWheelRotateEnvCfg(VQRTwoWheelBalanceEnvCfg):
    ...
```

Keep:

```text
FL + HR support
FR + HL lifted
16D action space
Balance reset
Balance safety terminations
asymmetric actor-critic
```

Add only the behavior needed for rotation.

Use yaw-only command input.
Do not add pitch command, gesture phase, or sin/cos phase clock unless explicitly justified.

Start with a conservative yaw-rate range and increase gradually, for example:

```text
±0.3
→ ±0.6
→ ±1.0
→ ±1.5 rad/s
```

Add a yaw-rate tracking reward, for example:

```text
exp(-(actual_yaw_rate - commanded_yaw_rate)^2 / sigma^2)
```

Balance/contact reward must remain strong enough that spinning while falling is not profitable.

A curriculum must not increase difficulty based only on episode survival. Use balance/contact/tracking quality where practical.

---

# PPO Configuration

Use RSL-RL PPO in:

```text
config/wheeled/vqr/agents/rsl_rl_ppo_cfg.py
```

Reasonable baseline:

```text
num_steps_per_env = 24
gamma = 0.99
lambda = 0.95
clip_param = 0.2
learning_rate = 1e-3
actor hidden dims = [512, 256, 128]
critic hidden dims = [512, 256, 128]
activation = ELU
```

Do not tune PPO to hide incorrect physics, indexing, reset, contacts, observations, or rewards.

---

# Environment Registration

Register Pivot tasks under:

```text
config/wheeled/vqr/__init__.py
```

using:

```text
isaaclab.envs:ManagerBasedRLEnv
```

Current task IDs:

```text
Pivot-TwoWheelBalance-v0
Pivot-TwoWheelRotate-v0
```

Ensure parent `__init__.py` files import the modules required for registration.

---

# Required Validation Order

Never jump directly to large-scale training.

Use this order:

```text
1. task registry
2. robot load
3. joint/body mapping
4. 16D action mapping
5. wheel contact mapping
6. reset pose
7. random-action finite-value test
8. 16-env PPO smoke test
9. 256 envs
10. 1024+ envs
```

For robot load verify:

```text
USD loads
floating articulation valid
joint count correct
wheel count correct
no NaN
no invalid-inertia failure
```

For contact mapping explicitly resolve and verify:

```text
FL
FR
HL
HR
```

Never assume simulator body ordering.

For reset validation check:

```text
FL + HR support contacts
FR + HL lifted initially
finite state/reward/observations
reasonable penetration
no explosive reset impulse
no immediate unrecoverable pose
```

For Rotate additionally check:

```text
yaw command sampling
yaw-rate tracking signal
actor observation dimension after command insertion
```

---

# Required Metrics

Do not evaluate training from total reward alone.

Log or expose at least:

```text
actual yaw rate
commanded yaw rate
yaw-rate tracking error
base vx
base vy
roll
pitch
roll rate
pitch rate
FL contact status
FR contact status
HL contact status
HR contact status
FR clearance
HL clearance
episode length
termination reason
```

For Balance, yaw metrics may remain zero/unused.
For Rotate, yaw tracking must be explicitly evaluated.

---

# Debugging Priority

When behavior is wrong, debug in this order:

```text
1. robot model
2. joint mapping
3. action mapping
4. contact mapping
5. reset pose
6. termination
7. observation
8. reward
9. PPO
```

Do not tune PPO to compensate for an environment bug.

---

# Code Reuse

Reuse compatible generic MDP functions from the existing locomotion implementation where appropriate.

Pivot-specific logic such as:

```text
diagonal support reward
lifted-wheel clearance
pivot termination
two-wheel reset
```

should live under `pivot/mdp/`.

The existing velocity task is reference code only and must not be modified.

---

# Out of Scope

Unless explicitly requested, do NOT implement:

```text
4-wheel → 2-wheel transition
automatic diagonal switching
rough terrain
jumping
external disturbance recovery
large domain randomization
recurrent policies
transformers
imitation learning
AMP
direct torque policies
custom RL algorithms
hardware deployment
ONNX export
```

---

# Definition of Done — TwoWheelBalance

`Pivot-TwoWheelBalance-v0` is complete only when:

```text
task appears in scripts/tools/list_envs.py
environment registration works
USD robot spawns correctly
16 actions map correctly
actor observations are valid
critic observations are valid
wheel contact mapping is verified
two-wheel reset is valid
reward values remain finite
termination behaves correctly
16-env PPO smoke test actually runs
```

Do not report runtime/PPO PASS if the required simulator/GPU test was not executed.

---

# Definition of Done — TwoWheelRotate

`Pivot-TwoWheelRotate-v0` is complete only when:

```text
it inherits from Balance
yaw-only command is implemented
yaw tracking reward is implemented
FL-HR support remains enforced
FR-HL lifted behavior remains enforced
planar drift remains controlled
task appears in scripts/tools/list_envs.py
16-env PPO smoke test actually runs
yaw tracking metrics are logged
training runs without NaN or invalid simulation state
```

---

# Implementation Style

Prefer:

```text
small changes
clear names
explicit constants
inheritance
reusable functions
minimal duplication
```

Avoid:

```text
magic indices
magic joint ordering
duplicated configs
huge reward lists
silent exception handling
hard-coded simulator body IDs
hard-coded machine paths
hard-coded cuda:0 in task config
invented task IDs
invented CLI flags
invented launch scripts
```

When an assumption is unavoidable, document it next to the code.

---

# Agent Working Rule

Implement one validated milestone at a time.

For each milestone:

```text
inspect existing code
→ implement
→ verify registry/import
→ run minimal simulator validation when available
→ run PPO smoke test when available
→ fix failures
→ report exactly what was and was not executed
→ continue
```

If a model name, task ID, script path, CLI flag, actuator property, contact mapping, environment, device, or checkpoint is uncertain, inspect the repository/current runtime instead of guessing.

Most importantly:

```text
NEVER invent a command.
NEVER invent an absolute path.
NEVER invent a task ID.
NEVER call a validation PASS if it was not executed successfully.
```
