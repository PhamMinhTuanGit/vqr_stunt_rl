# AGENTS.md

## Scope

You are working on a new reinforcement-learning task for a wheeled quadruped robot.

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

The development sequence is:

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

Only Phase A and Phase B are in scope now.

Do NOT implement the 4-wheel → 2-wheel transition yet.

---

# Target Contact Configuration

Initial target support pair:

```text
FL + HR = support wheels

FR + HL = lifted wheels
```

The task must keep this pairing explicit.

Do not alternate diagonal pairs automatically in the first implementation.

Use constants rather than scattering wheel names throughout the code.

---

# Required Directory Structure

Target structure:

```text
pivot/
├── __init__.py
│
├── mdp/
│   ├── __init__.py
│   ├── rewards.py
│   ├── observations.py
│   ├── events.py
│   └── terminations.py
│
└── config/
    ├── __init__.py
    │
    └── wheeled/
        ├── __init__.py
        │
        └── vqr/
            ├── __init__.py
            ├── robot_cfg.py
            ├── balance_env_cfg.py
            ├── rotate_env_cfg.py
            │
            └── agents/
                ├── __init__.py
                └── rsl_rl_ppo_cfg.py
```

Avoid unnecessary files.

Do not duplicate generic implementation that can safely be imported from the existing velocity task.

---

# Existing Framework

The repository already uses:

```text
Isaac Sim
Isaac Lab
ManagerBasedRLEnv
RSL-RL
PPO
```

Keep this stack.

Do not introduce another RL framework.

Do not implement custom PPO unless explicitly requested.

---

# Robot Model

The user already has:

```text
URDF
USD
```

Prefer loading the USD directly.

Do not perform runtime URDF conversion unless necessary.

Because files outside `pivot/` cannot be modified, define the robot-specific `ArticulationCfg` inside:

```text
pivot/config/wheeled/vqr/robot_cfg.py
```

---

# Phase 0 — Model Audit

Before implementing RL logic, inspect the robot model and determine the exact names of:

```text
base link

12 leg joints

4 wheel joints

FL wheel body
FR wheel body
HL wheel body
HR wheel body
```

The implementation must use the actual names from the model.

Never invent joint names if they can be obtained from the USD or URDF.

Centralize them in `robot_cfg.py`.

Example structure:

```python
BASE_LINK_NAME = "..."

LEG_JOINT_NAMES = [
    ...
]

WHEEL_JOINT_NAMES = [
    ...
]

WHEEL_BODY_NAMES = [
    ...
]

SUPPORT_WHEEL_NAMES = [
    "FL wheel body",
    "HR wheel body",
]

LIFTED_WHEEL_NAMES = [
    "FR wheel body",
    "HL wheel body",
]
```

Do not hard-code these values repeatedly elsewhere.

---

# Robot Configuration

Create the robot configuration in:

```text
config/wheeled/vqr/robot_cfg.py
```

The robot should be a floating-base articulation.

Required actuator separation:

```text
leg actuators
wheel actuators
```

Leg joints should use position-based PD control.

Wheel joints should use velocity control.

Conceptually:

```text
Legs:
    position command
    stiffness > 0
    damping > 0

Wheels:
    velocity command
    stiffness = 0
    damping > 0
```

Use the robot's real or existing model actuator limits whenever available.

Do not invent aggressive torque or velocity limits.

---

# Action Space

The policy action must contain:

```text
12 leg position residuals
+
4 wheel velocity targets
```

Total:

```text
16 actions
```

Architecture:

```text
Policy
   │
   ├── 12 leg q targets
   │
   └── 4 wheel velocity targets
```

Use:

```text
JointPositionActionCfg
```

for leg joints and:

```text
JointVelocityActionCfg
```

for wheel joints.

Initial conservative action scaling:

```text
leg position scale:
approximately 0.15–0.20 rad

wheel velocity scale:
approximately 3–5 rad/s
```

These values may later be tuned.

Do not use direct torque action in the initial task.

---

# Actor Observation

Actor observations should contain only quantities that are realistically deployable on the physical robot.

Recommended actor observations:

```text
base angular velocity

projected gravity

yaw-rate command

leg joint position

leg joint velocity

wheel velocity

previous action
```

The actor should NOT directly receive simulator-only quantities such as:

```text
absolute world position

ground-truth world orientation

ground-truth contact force

perfect base linear velocity

future state

reward information
```

unless later explicitly justified for deployment.

---

# Critic Observation

The critic may use privileged simulation information.

Examples:

```text
base linear velocity

contact state

contact forces

base height

additional simulator state
```

Keep actor and critic observation groups separate.

Prefer an asymmetric actor-critic setup.

---

# Phase A — Two-Wheel Balance

Implement:

```text
Pivot-TwoWheelBalance-v0
```

in:

```text
balance_env_cfg.py
```

This environment must focus only on balancing on the selected diagonal support pair.

Commands:

```text
vx_cmd = 0

vy_cmd = 0

yaw_rate_cmd = 0
```

Do not train rotation yet.

---

# Balance Initial State

The Balance environment should reset close to a valid two-wheel configuration.

Target:

```text
FL = support/contact

HR = support/contact

FR = lifted

HL = lifted
```

The policy should initially learn:

```text
balance
+
contact maintenance
```

It should NOT initially learn the entire transition from four-wheel stance.

The two-wheel reset state can include a small nominal body lean if required by the robot geometry.

---

# Reset Event

Implement robot reset logic in:

```text
mdp/events.py
```

Reset close to the nominal two-wheel state.

Apply only small randomization initially.

Recommended reset noise:

```text
small joint position perturbation

small roll perturbation

small pitch perturbation

random yaw

small angular velocity

near-zero wheel velocity
```

Avoid large perturbations.

Initial training should NOT use:

```text
large mass randomization

large CoM randomization

external pushes

large motor-strength randomization

large control delay

strong terrain randomization
```

First make the nominal task learnable.

---

# Balance Rewards

Keep the initial reward set small.

Prefer approximately 5–7 meaningful reward terms.

Do not add dozens of weak or conflicting rewards.

Required categories follow.

## 1. Support contact

Reward maintaining contact on:

```text
FL
HR
```

The reward should encourage stable support, not excessive impact force.

---

## 2. Lifted diagonal

Encourage:

```text
FR not in contact

HL not in contact
```

Also include minimum wheel clearance where appropriate.

Contact state alone is insufficient because a policy may exploit it by keeping a wheel almost touching the ground.

---

## 3. Balance

Penalize excessive:

```text
roll

pitch

roll rate

pitch rate
```

Do not force:

```text
roll = 0

pitch = 0
```

with an extremely strong penalty.

The physically valid two-wheel equilibrium may require body lean.

---

## 4. Translational drift

The robot should rotate or balance in place rather than drive away.

For Balance, penalize:

```text
vx² + vy²
```

or equivalent planar linear velocity.

---

## 5. Torque regularization

Apply a small penalty to leg effort.

Do not make torque minimization dominate balance.

---

## 6. Action rate

Penalize rapid changes:

```text
||a_t - a_(t-1)||²
```

Use this mainly for smoother and more deployable behavior.

---

# Reward Design Principles

Avoid sparse binary-only reward when a smooth formulation is easy to construct.

Prefer continuous signals for:

```text
orientation error

velocity error

wheel clearance

contact quality
```

Do not introduce reward terms simply because they exist in the velocity task.

Every enabled reward must have a clear reason related to the pivot task.

---

# Termination

Implement custom terminations in:

```text
mdp/terminations.py
```

Terminate when clearly unrecoverable.

Examples:

```text
base/body touches ground

robot is inverted

roll exceeds a large safety threshold

pitch exceeds a large safety threshold

base height collapses
```

Do NOT immediately terminate because a lifted wheel briefly touches the ground.

Allow short recovery events.

If required, only terminate lifted-wheel contact after it persists for a meaningful duration.

---

# Contact Sensors

Use Isaac Lab contact sensors to identify wheel support state.

Verify body IDs before training.

Never assume the wheel ordering returned by the simulator.

Explicitly map:

```text
FL
FR
HL
HR
```

to the corresponding body IDs.

All contact-related rewards must be validated with a single robot before large-scale training.

---

# Phase B — Two-Wheel Rotate

Implement:

```text
Pivot-TwoWheelRotate-v0
```

in:

```text
rotate_env_cfg.py
```

The Rotate environment should inherit from the Balance environment.

Do not duplicate the entire Balance configuration.

Preferred pattern:

```python
class VQRTwoWheelRotateEnvCfg(VQRTwoWheelBalanceEnvCfg):
    ...
```

The Rotate task should change only what is required for rotation.

---

# Rotate Command

Use yaw-only commands.

Set:

```text
vx_cmd = 0

vy_cmd = 0
```

and:

```text
yaw_rate_cmd ≠ 0
```

Initial command range:

```text
[-1.0, 1.0] rad/s
```

A curriculum is recommended.

Example:

```text
Stage 1:
|yaw_rate| <= 0.2 rad/s

Stage 2:
|yaw_rate| <= 0.5 rad/s

Stage 3:
|yaw_rate| <= 1.0 rad/s
```

Do not start with unnecessarily aggressive yaw commands.

---

# Rotate Reward

Reuse all valid Balance rewards.

Add a yaw-rate tracking reward such as:

```text
exp(
    -(actual_yaw_rate - commanded_yaw_rate)^2
    / sigma^2
)
```

Yaw tracking should become one of the primary positive rewards.

Keep planar drift penalty enabled.

The desired behavior is:

```text
maintain two-wheel balance
+
maintain FL-HR support
+
keep FR-HL lifted
+
track yaw rate
+
avoid translating away
```

---

# PPO Configuration

Use RSL-RL PPO.

Create configuration in:

```text
config/wheeled/vqr/agents/rsl_rl_ppo_cfg.py
```

Use the existing wheeled M20 PPO configuration as a reference.

Reasonable initial settings:

```text
num_steps_per_env = 24

gamma = 0.99

lambda = 0.95

clip_param = 0.2

learning_rate = 1e-3

actor hidden dimensions:
[512, 256, 128]

critic hidden dimensions:
[512, 256, 128]

activation:
ELU
```

Do not perform extensive PPO hyperparameter tuning before verifying the environment.

---

# Environment Registration

Register:

```text
Pivot-TwoWheelBalance-v0

Pivot-TwoWheelRotate-v0
```

under:

```text
config/wheeled/vqr/__init__.py
```

Use:

```text
isaaclab.envs:ManagerBasedRLEnv
```

as the entry point.

Ensure the parent `__init__.py` files import the required modules so Gym registration is executed.

---

# Required Validation Order

Do not immediately launch a large training run.

Validate in this order.

## Step 1 — Robot load

Run one robot.

Verify:

```text
USD loads

articulation valid

base is floating

joint count correct

wheel joint count correct

no NaN

no invalid inertia failure
```

---

## Step 2 — Joint mapping

Print or inspect:

```text
joint names

joint IDs

body names

body IDs
```

Confirm all 16 controlled joints.

---

## Step 3 — Action mapping

Manually verify:

```text
each leg action moves the intended joint

each wheel velocity action rotates the intended wheel

sign convention is correct
```

---

## Step 4 — Contact mapping

Verify contact signals individually for:

```text
FL

FR

HL

HR
```

Do not continue if wheel contact IDs are incorrect.

---

## Step 5 — Reset

Verify the Balance environment repeatedly resets near the intended two-wheel configuration.

No reset may produce:

```text
NaN

invalid joint state

extreme penetration

explosive contact forces

immediate unrecoverable pose
```

---

## Step 6 — Random-action test

Run a small number of environments using random actions.

Check:

```text
reward finite

observations finite

actions finite

termination finite

reset works
```

---

## Step 7 — PPO smoke test

Run approximately:

```text
16 environments

1–2 iterations
```

before large-scale training.

The purpose is only to verify the training pipeline.

---

## Step 8 — Scale gradually

Recommended sequence:

```text
16 envs
↓
256 envs
↓
1024+ envs
```

Only scale after the previous level is stable.

---

# Required Metrics

Do not judge training only from total episode reward.

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

FR wheel clearance

HL wheel clearance

episode length

termination reason
```

For Balance, yaw-related metrics can remain near zero.

For Rotate, yaw tracking must be explicitly evaluated.

---

# Debugging Priorities

When training behaves incorrectly, debug in this order:

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

Do not tune PPO to compensate for incorrect physics or incorrect indexing.

---

# Code Reuse

Reuse existing generic MDP functions where appropriate.

For example, generic functions for:

```text
projected gravity

angular velocity

joint states

previous action

action-rate penalty

yaw-rate tracking
```

may be imported from the existing locomotion implementation if compatible.

Do not copy large amounts of generic code into `pivot/` without reason.

However, pivot-specific logic such as:

```text
diagonal support reward

lifted-wheel clearance

pivot termination

two-wheel reset
```

should live under `pivot/mdp/`.

---

# Do Not Modify Existing Velocity Task

The existing:

```text
tasks/manager_based/locomotion/velocity/
```

task is reference code only.

Never modify it while implementing Pivot.

If functionality needs to differ from Velocity, implement a Pivot-specific version.

---

# Out of Scope

Do NOT implement the following yet:

```text
4-wheel → 2-wheel transition

automatic diagonal switching

rough terrain

jumping

external disturbance recovery

large domain randomization

recurrent policies

transformer policies

imitation learning

AMP

direct torque policies

custom RL algorithms

hardware deployment

ONNX export
```

These are later phases.

---

# Future Phase

After both initial environments work, the next task will be approximately:

```text
Pivot-FourToTwoWheelRotate-v0
```

It will train:

```text
4-wheel stance

→ CoM shift

→ unload FR-HL

→ lift FR-HL

→ stabilize FL-HR support

→ rotate
```

Do not preemptively implement this now.

---

# Definition of Done — TwoWheelBalance

`Pivot-TwoWheelBalance-v0` is complete only when:

```text
environment registration works

USD robot spawns correctly

16 actions map correctly

actor observations are valid

critic observations are valid

wheel contact mapping is verified

two-wheel reset is stable

reward values remain finite

termination behaves correctly

16-env PPO smoke test passes

training shows increasing ability to maintain FL-HR support
```

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

PPO smoke test passes

yaw tracking metrics are logged

the task can train without NaN or invalid simulation state
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

unnecessary abstractions
```

When assumptions are unavoidable, document them directly next to the code.

---

# Agent Working Rule

Implement one validated milestone at a time.

For every milestone:

```text
implement
→ run minimal validation
→ fix failures
→ report result
→ continue
```

Do not implement all phases before testing the early ones.

If a model, joint name, actuator property, or contact mapping is uncertain, inspect the existing URDF/USD or repository configuration instead of guessing.

The immediate priority is:

```text
correct model
→ correct action mapping
→ correct two-wheel reset
→ correct contact reward
→ Balance task
→ Rotate task
```

Optimization and tuning come later.
