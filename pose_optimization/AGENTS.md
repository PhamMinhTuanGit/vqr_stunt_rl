# AGENTS.md — VQR Two-Wheel Pose Optimization

## Scope

All work for this task MUST stay inside:

```text
rl_training/pose_optimization/
```

Do not modify files outside this directory unless explicitly instructed by the user.

The goal is to build a small, reproducible optimization tool that finds a physically feasible static two-wheel equilibrium pose for the VQR wheeled quadruped.

For the first implementation, the active stance contacts are:

```text
FL = stance
HR = stance
FR = swing
HL = swing
```

The solver output will later be used as an initialization/reference state for reinforcement learning in the `pivot` task.

---

## Repository/model rules

### URDF

Reuse the exact VQR URDF path already used by the existing repository configuration.

Before writing optimization code:

1. Inspect the current VQR robot configuration, especially the existing file:

```text
source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/robot_cfg.py
```

2. Trace the robot asset definition until the actual URDF source/path used by the project is identified.
3. Reuse that existing URDF path unchanged.
4. Do NOT:
   - create a duplicate URDF;
   - move the URDF;
   - silently switch to another model;
   - hard-code a guessed absolute path;
   - modify the source URDF.

If the current asset configuration points to a USD but the original URDF path exists elsewhere in the repository, locate the same original VQR URDF previously used to generate/audit that USD and reuse it.

Document the resolved URDF path in:

```text
pose_optimization/README.md
```

---

# Objective

Find a static configuration

```text
q*
```

such that the robot can theoretically balance using only the FL and HR wheel contacts while FR and HL are lifted from the ground.

The final optimization variables will be:

```text
q
tau
f_FL
f_HR
```

where:

- `q` = floating-base robot configuration;
- `tau` = actuated joint torques;
- `f_FL` = FL contact force;
- `f_HR` = HR contact force.

At the static equilibrium:

```text
v = 0
dv = 0
```

The initial implementation MUST be built incrementally. Do not implement the complete NLP in one step.

---

# Coordinate/contact conventions

Before optimization, inspect the URDF and establish:

- floating-base convention;
- joint names and ordering;
- FL wheel/contact frame;
- FR wheel/contact frame;
- HL wheel/contact frame;
- HR wheel/contact frame;
- wheel radius;
- actuated leg joints;
- wheel joints;
- continuous joints;
- joint position limits;
- effort/torque limits where available.

Never infer contact frames only from names if they can be verified from the model.

Create a model audit executable/test that prints these values.

The optimization assumes:

```text
stance set = {FL, HR}
swing set  = {FR, HL}
```

---

# Mathematical problem

Let the ground plane be:

```text
z = 0
```

Let the two active contact points be:

```text
p_FL(q)
p_HR(q)
```

and the center of mass be:

```text
p_C(q)
```

Use the XY projections:

```text
p_FL_xy
p_HR_xy
p_C_xy
```

Define the support-line direction:

```text
d = (p_HR_xy - p_FL_xy)
    / ||p_HR_xy - p_FL_xy||
```

Define the perpendicular direction:

```text
n = [-d_y, d_x]^T
```

The signed perpendicular CoM error is:

```text
e_com = n^T (p_C_xy - p_FL_xy)
```

The static-pose target is:

```text
|e_com| <= epsilon_com
```

with an initial default:

```text
epsilon_com = 0.002 m
```

The CoM projection must also lie inside the FL-HR support segment, not merely on the infinite support line.

Define:

```text
s = d^T (p_C_xy - p_FL_xy)
L = ||p_HR_xy - p_FL_xy||
```

Require:

```text
segment_margin <= s <= L - segment_margin
```

Initial default:

```text
segment_margin = 0.02 m
```

---

# Contact constraints

If the selected wheel frame is located at the wheel center:

```text
p_FL.z = wheel_radius
p_HR.z = wheel_radius
```

If an actual ground-contact frame exists, use:

```text
p_FL.z = 0
p_HR.z = 0
```

Do not assume which convention is correct. Verify it from the model.

FR and HL must be clear of the ground:

```text
p_FR.z >= contact_height + swing_clearance
p_HL.z >= contact_height + swing_clearance
```

Initial default:

```text
swing_clearance = 0.02 m
```

---

# Static dynamics

Use Pinocchio rigid-body dynamics.

The full dynamics are:

```text
M(q) dv + h(q, v)
    = S^T tau
    + J_FL(q)^T f_FL
    + J_HR(q)^T f_HR
```

At the static equilibrium:

```text
v  = 0
dv = 0
```

therefore require:

```text
h(q, 0)
    - S^T tau
    - J_FL(q)^T f_FL
    - J_HR(q)^T f_HR
    = 0
```

Prefer computing:

```text
h(q, 0) = RNEA(q, 0, 0)
```

with Pinocchio.

The infinity norm of the final dynamics residual should be reported.

Target acceptance threshold:

```text
||r_dyn||_inf <= 1e-3
```

Prefer:

```text
<= 1e-4
```

when numerically achievable.

---

# Contact-force constraints

For:

```text
i in {FL, HR}
```

require:

```text
f_z_i >= 0
```

Use a friction pyramid first:

```text
-mu * f_z_i <= f_x_i <= mu * f_z_i
-mu * f_z_i <= f_y_i <= mu * f_z_i
```

Do not start with a nonlinear friction cone unless needed later.

Put `mu` in a YAML configuration file.

Initial default may be:

```text
mu: 0.6
```

but it must remain configurable.

Report friction utilization:

```text
rho_i = sqrt(f_x_i^2 + f_y_i^2) / (mu * f_z_i)
```

for both FL and HR.

---

# Joint and torque constraints

Respect joint position limits from the robot model.

Prefer a configurable safety margin:

```text
q_min + joint_margin
    <= q_j
    <= q_max - joint_margin
```

Initial default:

```text
joint_margin = 0.05 rad
```

Respect known effort limits.

For the final physically feasible solve, target:

```text
|tau_j| <= torque_usage_limit * tau_max_j
```

with default:

```text
torque_usage_limit = 0.8
```

If effort limits are missing for any relevant joint, do NOT invent values.

Instead:

1. report exactly which limits are unavailable;
2. allow the solver to run without those torque bounds only for diagnostic milestones;
3. mark the final solution as not hardware-qualified until real limits are supplied.

---

# Floating-base symmetry

Avoid unnecessary global symmetries.

For static pose optimization:

- fix global yaw to zero;
- constrain/remove arbitrary XY translation using a simple consistent convention;
- do not over-constrain roll and pitch.

Roll and pitch may move because the optimizer may need body inclination to move the CoM onto the FL-HR support line.

Use configurable bounds such as:

```text
abs(roll)  <= 25 deg
abs(pitch) <= 25 deg
```

Prefer penalizing roll/pitch in the objective rather than forcing:

```text
roll = 0
pitch = 0
```

---

# Wheel joints

Wheel rotation angles are not meaningful decision variables for a static circular-wheel equilibrium unless the model geometry proves otherwise.

Prefer to fix wheel angles to their nominal value.

At this static milestone:

```text
wheel velocity = 0
```

Do not add wheel-speed/yaw-rate optimization yet.

---

# Cost function

Once the required feasibility constraints are working, use a simple objective such as:

```text
J =
    w_orientation * (roll^2 + pitch^2)
  + w_pose        * ||q_leg - q_nominal||^2
  + w_tau         * ||tau||^2
  + w_force       * (
        ||f_FL - f_FL_ref||^2
      + ||f_HR - f_HR_ref||^2
    )
```

where a reasonable initial contact-force reference is:

```text
f_FL_ref = [0, 0, m*g/2]
f_HR_ref = [0, 0, m*g/2]
```

Do not use arbitrary large weights without documenting them.

Put all weights in:

```text
pose_optimization/config/equilibrium.yaml
```

Suggested initial values:

```yaml
weights:
  orientation: 100.0
  pose: 1.0
  torque: 1.0e-3
  force: 1.0e-4
```

These are initialization values, not authoritative tuned values.

---

# Required milestone order

## M-TO0 — Model audit

Goal: prove the model used by the optimizer is the same VQR robot model used by the project.

Required:

- load the existing VQR URDF with Pinocchio;
- print `nq` and `nv`;
- print joint names/order;
- identify FL/FR/HL/HR wheel/contact frames;
- report wheel radius;
- report position limits;
- report available effort limits;
- report total mass;
- report nominal CoM;
- verify FK can be evaluated at the nominal pose.

No optimization yet.

Deliverables:

```text
pose_optimization/
  README.md
  config/equilibrium.yaml
  src/model_audit.cpp
```

or an equivalent clean structure.

M-TO0 must PASS before M-TO1.

---

## M-TO1 — Kinematic feasibility

Goal: determine whether robot geometry can produce a valid FL-HR two-wheel support pose.

Optimize configuration only.

Required constraints:

```text
FL contact on ground
HR contact on ground

FR above ground
HL above ground

CoM projection close to FL-HR support line
CoM projection inside FL-HR support segment

joint limits
roll/pitch bounds
yaw fixed
```

Do NOT add:

```text
tau
contact forces
RNEA equilibrium
friction
```

yet.

Primary output:

```text
q_kinematic*
```

Required metrics:

```text
CoM-to-support-line distance
CoM support-segment coordinate
FL contact height error
HR contact height error
FR clearance
HL clearance
minimum joint-limit margin
roll
pitch
```

Save the solution to:

```text
pose_optimization/output/kinematic_equilibrium.yaml
```

M-TO1 must PASS before M-TO2.

---

## M-TO2 — Static dynamics feasibility

Starting from the M-TO1 solution, add:

```text
tau
f_FL
f_HR
RNEA static equilibrium
```

Require:

```text
h(q,0)
  - S^T tau
  - J_FL^T f_FL
  - J_HR^T f_HR
  = 0
```

Do not add friction constraints until this unconstrained/static-force problem is debugged.

Initialize forces approximately with:

```text
f_FL ~= [0, 0, m*g/2]
f_HR ~= [0, 0, m*g/2]
```

Required metrics:

```text
dynamics residual
FL force
HR force
joint torques
CoM-to-support-line distance
```

Save:

```text
pose_optimization/output/static_equilibrium.yaml
```

M-TO2 must PASS before M-TO3.

---

## M-TO3 — Physical feasibility

Add:

```text
friction pyramid
positive normal force
real torque limits
joint safety margins
```

Optimize the pose rather than freezing the M-TO1 pose.

Required final metrics:

```text
solver status
objective value
dynamics residual
CoM-to-support-line distance
CoM position on support segment

FL contact position
HR contact position
FR clearance
HL clearance

f_FL
f_HR

FL friction utilization
HR friction utilization

all joint positions
all joint torques
maximum torque utilization
minimum joint-limit margin
base position
base orientation
```

Final output:

```text
pose_optimization/output/two_wheel_equilibrium_FL_HR.yaml
```

The output must contain enough information to reproduce the state in Isaac Lab later.

---

# Solver stack

Preferred stack:

```text
C++17
Pinocchio
CasADi
IPOPT
Eigen
yaml-cpp
```

Use the Pinocchio model for:

```text
forward kinematics
frame positions
center of mass
Jacobians
RNEA
```

Use CasADi + IPOPT for the nonlinear optimization if the currently installed Pinocchio/CasADi integration supports the required symbolic operations cleanly.

Do not add another optimization framework unless necessary.

Avoid ROS dependencies.

This tool should remain standalone from Isaac Sim / Isaac Lab.

---

# Build expectations

Provide a local build path such as:

```bash
cd rl_training/pose_optimization
cmake -S . -B build
cmake --build build -j
```

Executables should be runnable independently.

Example target structure:

```text
pose_optimization/
├── AGENTS.md
├── CMakeLists.txt
├── README.md
├── config/
│   └── equilibrium.yaml
├── include/
│   └── pose_optimization/
├── src/
│   ├── model_audit.cpp
│   ├── kinematic_equilibrium.cpp
│   └── static_equilibrium.cpp
├── apps/
│   ├── solve_kinematic_equilibrium.cpp
│   └── solve_static_equilibrium.cpp
├── tests/
└── output/
```

This structure is recommended but may be adjusted if there is a clear technical reason.

---

# Tests

At minimum add tests/checks for:

1. the URDF loads successfully;
2. expected FL/FR/HL/HR frames exist;
3. total robot mass is positive;
4. nominal CoM is finite;
5. frame FK results are finite;
6. support-line calculation is correct;
7. signed point-to-line distance is correct;
8. support-segment coordinate is correct;
9. M-TO1 saved pose reproduces the reported geometry;
10. M-TO2/M-TO3 saved pose reproduces the reported dynamics residual.

No test should depend on Isaac Sim being installed/running.

---

# Initial guess strategy

Never initialize the floating-base robot with all-zero `q`.

Use the existing nominal/default VQR standing pose from the repository as the starting configuration.

For M-TO1:

- begin from the normal standing pose;
- lift FR and HL modestly;
- allow FL/HR leg configuration and body pose to move;
- use the previous successful solution as the initial guess for later stages.

Use continuation:

```text
M-TO0
  ↓
M-TO1
  ↓
M-TO2
  ↓
M-TO3
```

Do not solve the hardest problem from a random initialization.

---

# Acceptance criteria

A solver reporting `Solve_Succeeded` is not enough.

The final M-TO3 result should ideally satisfy:

```text
CoM line error:
  <= 0.002 m

dynamics infinity-norm residual:
  <= 1e-3

FR clearance:
  >= configured swing_clearance

HL clearance:
  >= configured swing_clearance

FL normal force:
  > 0

HR normal force:
  > 0

friction utilization:
  < 1.0
```

Prefer additional margin:

```text
friction utilization < 0.8
torque utilization   < 0.8
```

when feasible.

If the problem is infeasible, do not hide that result.

Report whether failure first appears at:

```text
kinematic feasibility
static dynamics
friction feasibility
torque feasibility
```

This diagnosis is a required result.

---

# Important interpretation

A failure to find a static FL-HR equilibrium does NOT automatically mean the robot cannot perform the stunt.

If no static pose exists, the physical task may require dynamic active balancing.

That would imply later optimization/training should target a dynamic state or trajectory rather than:

```text
v* = 0
```

Do not weaken physical constraints merely to force a successful static solution.

---

# Explicitly out of scope for this task

Do NOT implement yet:

```text
full 4-wheel -> 2-wheel trajectory optimization
contact switching optimization
complementarity constraints
MPC
LQR
PPO/RL training
Isaac Lab reward changes
Isaac Lab curriculum
yaw-rate optimization
wheel-speed trajectory optimization
hardware deployment
```

The current objective is only:

```text
Find and validate the best physically feasible static FL-HR
two-wheel equilibrium pose.
```

---

# Agent working rules

1. Work only under `rl_training/pose_optimization/`.
2. Reuse the project's existing VQR URDF path.
3. Do not modify the URDF.
4. Do not modify the existing `pivot` training task.
5. Do not make silent assumptions about frame names, wheel radius, torque limits, or joint order.
6. Validate model data before optimization.
7. Implement milestones sequentially.
8. Keep every milestone independently runnable.
9. Save solver inputs/configuration and outputs.
10. Print concise quantitative diagnostics after every solve.
11. Prefer simple formulations that are easy to audit.
12. Do not introduce ROS.
13. Do not add unnecessary heavy dependencies.
14. Do not proceed to a later milestone if the current one has an unexplained failure.
15. Document any repository/model assumption in `README.md`.

---

# First task for the agent

Implement **M-TO0 only**.

Do not start M-TO1 yet.

The first response after implementation must report:

```text
1. exact reused URDF path
2. nq / nv
3. total mass
4. actuated joint names/order
5. wheel joint names/order
6. FL/FR/HL/HR contact or wheel frame names
7. wheel radius
8. position limits
9. available effort limits
10. nominal base pose
11. nominal CoM
12. files created/modified
13. build command
14. test/run result
15. any unresolved model ambiguity
```

Stop after M-TO0 and wait for the next instruction before implementing M-TO1.
