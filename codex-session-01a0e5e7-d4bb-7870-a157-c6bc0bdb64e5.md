# Codex conversation

## User

kiểm tra cho tôTRANSITION vẫn có incentive “fail sớm”. Đây là lỗi quan trọng nhất. Sau khi loại reward farming, reward trong TRANSITION chuyển quá mạnh sang deficit: com_support, transition_support_load, fsm_gated_tracking, stability... đều có thể âm trong lúc robot đang học acquisition. Nếu robot chưa giữ được support, tiếp tục cố thêm 1–2 giây sẽ tích lũy penalty; terminate ở ~0.18 s lại cắt được phần penalty tương lai. Điều này khớp với run trước: transition_duration≈0.184 s, gần như zero YAW và unsafe termination rất cao. Audit cũng xác định đây là một trong ba blocker chính.

## Assistant

Mình sẽ lần theo reward và điều kiện kết thúc của TRANSITION, rồi kiểm tra xem các penalty tích lũy có khiến terminate sớm có lợi hơn tiếp tục acquisition hay không. Nếu xác nhận được, mình sẽ sửa cơ chế đó và chạy kiểm tra phù hợp.

## Activity

    $ pwd; rg --files -g 'AGENTS.md' -g '!**/.git/**' -g '.codegraph/**' | head -80
    /home/robotics/tuanpm48/vqr/rl_training
    docs/document_yaw_fsm/AGENTS.md
    pose_optimization/AGENTS.md
    source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/AGENTS.md
    ✓ • 0ms

## Activity

    $ test -d .codegraph && echo codegraph-present || echo codegraph-absent
    codegraph-present
    ✓ • 0ms

## Activity

    $ rg -n 'transition_support_load|fsm_gated_tracking|com_support|transition_duration|TRANSITION' source/rl_training/rl_training/tasks/manager_based/locomotion/pivot -g '*.py' | head -160
    source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/tests/smoke_transition.py:60:            term.elapsed[:] = term.cfg.standing_time_s + term.cfg.transition_duration_s
    source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/mdp/to_transition.py:23:        if cfg.transition_duration_s <= 0 or cfg.standing_time_s < 0:
    source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/mdp/to_transition.py:87:        phase = ((self.elapsed - self.cfg.standing_time_s) / self.cfg.transition_duration_s).clamp(0, 1)
    source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/mdp/to_transition.py:158:    transition_duration_s: float = 2.5
    ✓ • 0ms

## Activity

    $ git status --short
    ✓ • 0ms

## Activity

    $ cat source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/AGENTS.md
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
    ✓ • 0ms

## Activity

    $ codegraph explore "TRANSITION reward terms com_support transition_support_load fsm_gated_tracking stability termination unsafe early failure in pivot"
    **Dynamic boundaries (the static path ends at runtime dispatch)**
    
    - `test_downward_speed_is_penalized_before_hard_height_failure` (tests/test_yaw_transition_safety.py:67) — computed member call: `assert torch.equal(rewards["yaw_downward_low_base_velocity_l2"](env, **args), torch.zeros(2))` (+3 more such sites in this body)
      candidates for key `yaw_downward_low_base_velocity_l2`: `yaw_downward_low_base_velocity_l2` (source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:338)
    
    > These sites choose their call target at runtime (registry / bus / reflection) — the site shown IS where the flow continues. To follow it, run codegraph_explore or codegraph_node on a candidate; source for the sites above is included below.
    
    > Full source for these symbols is below — the call flow among them, followed by their bodies.
    **Exploration: TRANSITION reward terms com_support transition_support_load fsm_gated_tracking stability termination unsafe early failure in pivot**
    
    Found 41 symbols across 5 files.
    
    **Blast radius — what depends on these (update/verify before editing)**
    
    - `TerminationsCfg` (source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/balance_env_cfg.py:283) — 4 callers in `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/four_to_two_rotate_env_cfg.py`, `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_cfg.py`, `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py`, `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/balance_env_cfg.py`; no tests found within 3 caller hops
    - `TerminationsCfg` (source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/velocity_env_cfg.py:793) — 1 caller in `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/velocity_env_cfg.py`; no tests found within 3 caller hops
    - `PivotRewardsCfg` (source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/pivot_env_cfg.py:287) — 1 caller in `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/pivot_env_cfg.py`; no tests found within 3 caller hops
    - `PivotCommandsCfg` (source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/pivot_env_cfg.py:117) — 1 caller in `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/pivot_env_cfg.py`; no tests found within 3 caller hops
    
    **Source Code**
    
    > The code below is the **verbatim, current on-disk source** of these files — re-read from disk on this call and line-numbered, byte-for-byte identical to what the Read tool returns. It is NOT a summary, outline, or stale cache. Treat each block as a Read you have already performed: do not Read a file shown here.
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py`** — _fsm_masked_accumulate(calls), _yaw_wheel_contacts(calls), _fsm_masked_accumulate_pair(calls), fsm_gates(calls), _yaw_lift_progress(calls), _fsm_record_positive_budget(calls), +10 more
    
    ```python
    866        env._yaw_fsm_positive_budget_neg += positive * gates["diag_neg"].to(positive.dtype)
    867
    868
    869    def fsm_gated_tracking(
    870        env: ManagerBasedRLEnv,
    871        command_name: str,
    872        fsm_command_name: str,
    873        support_sensor_cfg: SceneEntityCfg,
    874        support_sensor_cfg_mirror: SceneEntityCfg,
    875        lifted_asset_cfg: SceneEntityCfg,
    876        lifted_asset_cfg_mirror: SceneEntityCfg,
    877        wheel_radius: float,
    878        target_clearance: float,
    879        std: float,
    880        contact_threshold: float = 1.0,
    881        clearance_gate_floor: float = 0.0,
    882        clearance_gate_floor_decay_s: float = 0.0,
    883        edge_command_fraction: float = 0.80,
    884        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    885        lift_progress_threshold: float = 0.80,
    886        support_threshold: float = 0.85,
    887    ) -> torch.Tensor:
    888        """Track signed yaw commands during TRANSITION and YAW states.
    889
    890        Tracking and lift-scaled support contact are available during transition;
    891        curriculum tracking accumulators update only in ``YAW_POS``/``YAW_NEG``.
    892        The signed command preserves the direction constraint for either branch.
    893        """
    894        if std <= 0.0:
    895            raise ValueError("std must be positive.")
    896        if target_clearance <= 0.0:
    897            raise ValueError("target_clearance must be positive.")
    898        if not 0.0 <= clearance_gate_floor <= 1.0:
    899            raise ValueError("clearance_gate_floor must be in [0, 1].")
    900        if clearance_gate_floor_decay_s < 0.0:
    901            raise ValueError("clearance_gate_floor_decay_s must be non-negative.")
    902        if not 0.0 < edge_command_fraction <= 1.0:
    903            raise ValueError("edge_command_fraction must be in (0, 1].")
    904
    905        gates = fsm_gates(env, fsm_command_name)
    906        _fsm_step_telemetry(env, gates)
    907        support_pos_contact = _yaw_wheel_contacts(env, support_sensor_cfg, contact_threshold)
    908        support_neg_contact = _yaw_wheel_contacts(env, support_sensor_cfg_mirror, contact_threshold)
    909        support_pos = support_pos_contact.to(dtype=torch.float32).prod(dim=1)
    910        support_neg = support_neg_contact.to(dtype=torch.float32).prod(dim=1)
    911        support_gate = torch.where(gates["diag_pos"], support_pos, support_neg)
    912        lost_support = gates["b_yaw"] & (support_gate == 0.0)
    913        if not hasattr(env, "_yaw_fsm_support_loss_run_steps"):
    914            env._yaw_fsm_support_loss_run_steps = torch.zeros(
    915                env.num_envs, device=support_gate.device, dtype=torch.long
    916            )
    917        env._yaw_fsm_support_loss_run_steps = torch.where(
    918            lost_support,
    919            env._yaw_fsm_support_loss_run_steps + 1,
    920            torch.zeros_like(env._yaw_fsm_support_loss_run_steps),
    921        )
    922        for suffix, direction in (("pos", gates["diag_pos"]), ("neg", gates["diag_neg"])):
    923            name = f"_yaw_fsm_{suffix}_support_loss_max_steps"
    924            if not hasattr(env, name):
    925                setattr(env, name, torch.zeros_like(env._yaw_fsm_support_loss_run_steps))
    926            previous_max = getattr(env, name)
    927            previous_max.copy_(torch.where(
    928                direction & lost_support,
    929                torch.maximum(previous_max, env._yaw_fsm_support_loss_run_steps),
    930                previous_max,
    931            ))
    932        support_contacts = torch.where(
    933            gates["diag_pos"].unsqueeze(1), support_pos_contact, support_neg_contact
    934        )
    935        support_shape = _yaw_support_shape(support_contacts)
    936        # The opposite support diagonal is exactly the active swing diagonal.
    937        swing_contact = select_swing_wheel_contact(
    938            support_pos_contact,
    939            support_neg_contact,
    940            gates["support_diagonal"],
    941        )
    942
    943        lift_pos = _yaw_lift_progress(env, lifted_asset_cfg, wheel_radius, target_clearance)
    944        lift_neg = _yaw_lift_progress(env, lifted_asset_cfg_mirror, wheel_radius, target_clearance)
    945        selected_lift = torch.where(gates["diag_pos"].unsqueeze(1), lift_pos, lift_neg)
    946        lift_progress = selected_lift.mean(dim=1)
    947
    948        asset: RigidObject = env.scene[asset_cfg.name]
    949        _fsm_transition_telemetry(
    950            env, gates, support_contacts, selected_lift, asset,
    951            env.command_manager.get_term(fsm_command_name),
    952        )
    953        yaw_command = env.command_manager.get_command(command_name)[:, 0]
    954        yaw_error = torch.abs(yaw_command - asset.data.root_ang_vel_b[:, 2])
    955        yaw_tracking = torch.exp(-yaw_error.square() / std**2)
    956
    957        # Metrics are intentionally YAW-only; otherwise transition/FOUR steps
    958        # dilute the curriculum score and make the scale ladder stall.
    959        yaw_mask = gates["b_yaw"].to(dtype=yaw_command.dtype)
    960        # Reuse the legacy yaw-curriculum accumulator names.  This keeps the
    961        # existing checkpoint/export path valid while changing only the sample
    962        # mask: FSM tracking statistics are normalized by YAW-state steps.
    963        _fsm_masked_accumulate(
    964            env,
    965            support_gate,
    966            yaw_mask,
    967            "_yaw_support_score_sum",
    968            "_yaw_support_score_samples",
    969        )
    970        _fsm_masked_accumulate(
    971            env,
    972            support_gate,
    973            yaw_mask,
    974            "_yaw_gate_open_sum",
    975            "_yaw_gate_open_samples",
    976        )
    977        _fsm_masked_accumulate_pair(
    978            env,
    979            torch.abs(yaw_command),
    980            yaw_error,
    981            yaw_mask,
    982            "_yaw_command_abs_sum",
    983            "_yaw_rate_abs_error_sum",
    984            "_yaw_tracking_metric_samples",
    985        )
    986
    987        # The legacy accumulators above deliberately remain direction-agnostic:
    988        # the original yaw curriculum and checkpoint export consume them.  The
    989        # FSM curriculum additionally needs independent evidence for each
    990        # diagonal; a strong POS branch must never promote a weak NEG branch.
    991        yaw_pos_mask = gates["b_yaw"] & gates["diag_pos"]
    992        yaw_neg_mask = gates["b_yaw"] & gates["diag_neg"]
    993        for suffix, contacts, direction_mask, wheel_names in (
    994            ("pos", support_pos_contact, yaw_pos_mask, ("FL", "HR")),
    995            ("neg", support_neg_contact, yaw_neg_mask, ("FR", "HL")),
    996        ):
    997            for index, wheel_name in enumerate(wheel_names):
    998                _fsm_masked_accumulate(
    999                    env,
    1000                    contacts[:, index].to(dtype=yaw_command.dtype),
    1001                    direction_mask.to(dtype=yaw_command.dtype),
    1002                    f"_yaw_fsm_{suffix}_support_{wheel_name}_sum",
    1003                    f"_yaw_fsm_{suffix}_support_{wheel_name}_samples",
    1004                )
    1005        for suffix, direction_mask in (("pos", yaw_pos_mask), ("neg", yaw_neg_mask)):
    1006            mask = direction_mask.to(dtype=yaw_command.dtype)
    1007            _fsm_masked_accumulate(
    1008                env,
    1009                lift_progress,
    1010                mask,
    1011                f"_yaw_fsm_{suffix}_lift_sum",
    1012                f"_yaw_fsm_{suffix}_yaw_samples",
    1013            )
    1014            _fsm_masked_accumulate(
    1015                env,
    1016                support_gate,
    1017                mask,
    1018                f"_yaw_fsm_{suffix}_support_sum",
    1019                f"_yaw_fsm_{suffix}_support_samples",
    1020            )
    1021            _fsm_masked_accumulate_pair(
    1022                env,
    1023                torch.abs(yaw_command),
    1024                yaw_error,
    1025                mask,
    1026                f"_yaw_fsm_{suffix}_command_abs_sum",
    1027                f"_yaw_fsm_{suffix}_yaw_abs_error_sum",
    1028                f"_yaw_fsm_{suffix}_tracking_samples",
    1029            )
    1030            _fsm_masked_accumulate(
    1031                env,
    1032                swing_contact.to(dtype=yaw_command.dtype),
    1033                mask,
    1034                f"_yaw_fsm_{suffix}_swing_contact_sum",
    1035                f"_yaw_fsm_{suffix}_swing_contact_samples",
    1036            )
    1037
    1038        _fsm_attempt_telemetry(
    1039            env, gates, lift_progress, support_gate, lift_progress_threshold, support_threshold
    1040        )
    1041        env._yaw_support_score_current = support_gate
    1042        env._yaw_gate_open_current = support_gate
    1043        env._yaw_command_abs_current = torch.abs(yaw_command)
    1044        env._yaw_rate_abs_error_current = yaw_error
    1045
    1046        command_term = env.command_manager.get_term(command_name)
    1047        yaw_limits = torch.as_tensor(
    1048            command_term.cfg.yaw_rate_range,
    1049            device=yaw_command.device,
    1050            dtype=yaw_command.dtype,
    1051        )
    1052        yaw_limit = yaw_limits.abs().amax()
    1053        edge_mask = (torch.abs(yaw_command) >= edge_command_fraction * yaw_limit) & gates["b_yaw"]
    1054        _fsm_masked_accumulate_pair(
    1055            env,
    1056            torch.abs(yaw_command),
    1057            yaw_error,
    1058            edge_mask.to(dtype=yaw_command.dtype),
    1059            "_yaw_edge_command_abs_sum",
    1060            "_yaw_edge_rate_abs_error_sum",
    1061            "_yaw_edge_tracking_samples",
    1062        )
    1063
    1064        # The phase-B floor provides a short exploration bridge at transition
    1065        # entry, then decays to zero so the policy cannot keep collecting reward
    1066        # without lifting.  A zero decay keeps the legacy/static-floor behavior.
    1067        if clearance_gate_floor_decay_s > 0.0:
    1068            floor = clearance_gate_floor * torch.clamp(
    1069                1.0 - gates["state_time"] / clearance_gate_floor_decay_s,
    1070                min=0.0,
    1071                max=1.0,
    1072            )
    1073        else:
    1074            floor = clearance_gate_floor
    1075        clearance_weight = floor + (1.0 - floor) * lift_progress
    1076        state_gate = gates["f_trans"] + gates["f_yaw"]
    1077        tracking = support_gate * clearance_weight * yaw_tracking * state_gate
    1078        # Contact alone gives no bonus while the swing pair is still on the ground.
    1079        support_bonus = 0.25 * support_shape * lift_progress * state_gate
    1080        # Center TRANSITION at its maximum: holding pose/tracking is never
    1081        # positive income. YAW retains the original tracking and support bonus.
    1082        reward = tracking + support_bonus - 1.25 * gates["f_trans"]
    1083        _fsm_record_positive_budget(env, gates, "fsm_gated_tracking", reward)
    1084        return reward
    1085
    1086
    1087    def four_stand_stability(
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/pivot_env_cfg.py`** — SceneEntityCfg(instantiates), configclass(decorates), PivotCommandsCfg(class), PivotRewardsCfg(class)
    
    ```python
    113    ##
    114
    115
    116    @configclass
    117    class PivotCommandsCfg:
    118        pivot_mode = mdp.PivotModeCommandCfg(
    119            asset_name="robot",
    120            support_body_names=REAR_SUPPORT_WHEEL_NAMES,
    121            # Sample once per episode; mid-episode resampling would move the
    122            # frozen support anchor and invalidate the time-based FSM.
    123            resampling_time_range=(1.0e6, 1.0e6),
    124            omega_z_range=(-PHYS.omega_z_limit_balance, PHYS.omega_z_limit_balance),
    125            omega_z_limit=PHYS.omega_z_limit_balance,
    126            ground_omega_z_limit=PHYS.omega_z_limit_ground,
    127            delta_theta_range=(-PHYS.delta_theta_command_limit, PHYS.delta_theta_command_limit),
    128            delta_theta_limit=PHYS.delta_theta_command_limit,
    129            rel_standing_envs=0.1,
    130            ground_duration_s=2.0,
    131            rear_up_duration_s=3.0,
    132            balance_duration_s=10.0,
    133            land_duration_s=5.0,
    134        )
    135
    136
    137    ##
    
    ... (gap) ...
    
    284
    285
    286    @configclass
    287    class PivotRewardsCfg:
    288        # omega_z tracking: GROUND 1.0, BALANCE 1.5
    289        omega_z = RewTerm(
    290            func=mdp.omega_z_tracking,
    291            weight=1.0,
    292            params={"command_name": "pivot_mode", "std": 0.5},
    293        )
    294        # exp(-xi^2/sigma^2): REAR_UP 1.5, BALANCE 2.5
    295        capture_point = RewTerm(
    296            func=mdp.capture_point_reward,
    297            weight=1.0,
    298            params={
    299                "command_name": "pivot_mode",
    300                "asset_cfg": SceneEntityCfg(
    301                    "robot", body_names=REAR_SUPPORT_WHEEL_NAMES, preserve_order=True
    302                ),
    303                "std": 0.12,
    304            },
    305        )
    306        # GROUND/LAND: body-center drift; REAR_UP/BALANCE: frozen HL-HR midpoint.
    307        anchor = RewTerm(
    308            func=mdp.anchor_midpoint_reward,
    309            weight=1.0,
    310            params={
    311                "command_name": "pivot_mode",
    312                "asset_cfg": SceneEntityCfg(
    313                    "robot", body_names=REAR_SUPPORT_WHEEL_NAMES, preserve_order=True
    314                ),
    315            },
    316        )
    317        # CoM/capture geometry about HL-HR only in rear-supported modes.
    318        balance_alignment = RewTerm(
    319            func=mdp.balance_midpoint_reward,
    320            weight=1.0,
    321            params={
    322                "command_name": "pivot_mode",
    323                "asset_cfg": SceneEntityCfg(
    324                    "robot", body_names=REAR_SUPPORT_WHEEL_NAMES, preserve_order=True
    325                ),
    326            },
    327        )
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/balance_env_cfg.py`** — SceneEntityCfg(instantiates), configclass(decorates), RewardsCfg(class), TerminationsCfg(class), VQRTwoWheelBalanceEnvCfg(class), TwoWheelBalanceSceneCfg(instantiates), +5 more
    
    ```python
    233        )
    234
    235
    236    @configclass
    237    class RewardsCfg:
    238        """Seven task-specific rewards; no locomotion reward set is inherited."""
    239
    240        support_contact = RewTerm(
    241            func=mdp.pivot_support_contact,
    242            weight=1.0,
    243            params={
    244                "sensor_cfg": SceneEntityCfg(
    245                    "contact_forces", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
    246                ),
    247                "threshold": 1.0,
    248            },
    249        )
    250        lifted_diagonal = RewTerm(
    251            func=mdp.pivot_lifted_wheels,
    252            weight=1.0,
    253            params={
    254                "sensor_cfg": SceneEntityCfg(
    255                    "contact_forces", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
    256                ),
    257                "asset_cfg": SceneEntityCfg("robot", body_names=LIFTED_WHEEL_NAMES, preserve_order=True),
    258                "wheel_radius": WHEEL_RADIUS,
    259                "minimum_clearance": 0.05,
    260                "threshold": 1.0,
    261            },
    262        )
    263        balance = RewTerm(
    264            func=mdp.pivot_balance,
    265            weight=2.0,
    266            params={
    267                "nominal_roll": NOMINAL_ROLL,
    268                "nominal_pitch": NOMINAL_PITCH,
    269                "std": 0.25,
    270            },
    271        )
    272        angular_stability = RewTerm(func=mdp.pivot_angular_stability, weight=-0.10)
    273        planar_drift = RewTerm(func=mdp.pivot_planar_drift, weight=-0.50)
    274        action_rate = RewTerm(func=mdp.pivot_action_rate_l2, weight=-0.01)
    275        leg_effort = RewTerm(
    276            func=mdp.pivot_leg_effort_l2,
    277            weight=-1.0e-5,
    278            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    279        )
    280
    281
    282    @configclass
    283    class TerminationsCfg:
    284        """Failures that are clearly outside the recoverable balance region."""
    285
    286        time_out = DoneTerm(func=mdp.time_out, time_out=True)
    287        base_contact = DoneTerm(
    288            func=mdp.pivot_base_contact,
    289            params={
    290                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[BASE_LINK_NAME]),
    291                "threshold": 1.0,
    292            },
    293        )
    294        inverted = DoneTerm(func=mdp.pivot_inverted)
    295        tilt_limit = DoneTerm(
    296            func=mdp.pivot_tilt_limit,
    297            params={
    298                "nominal_roll": NOMINAL_ROLL,
    299                "nominal_pitch": NOMINAL_PITCH,
    300                "roll_limit": 1.05,
    301                "pitch_limit": 1.05,
    302            },
    303        )
    304        base_height = DoneTerm(func=mdp.pivot_base_height_below, params={"minimum_height": 0.22})
    305        drift = DoneTerm(func=mdp.pivot_drifted_away, params={"maximum_distance": 1.0})
    306
    307
    308    @configclass
    309    class VQRTwoWheelBalanceEnvCfg(ManagerBasedRLEnvCfg):
    310        """M1 balance task configuration."""
    311
    312        decimation = 4
    313        episode_length_s = 10.0
    314        sim = sim_utils.SimulationCfg(dt=0.005, render_interval=decimation, device="cuda:0")
    315        scene = TwoWheelBalanceSceneCfg(num_envs=16, env_spacing=2.5)
    316        observations = ObservationsCfg()
    317        actions = ActionsCfg()
    318        commands = None
    319        rewards = RewardsCfg()
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/mdp/to_transition.py`** — reference(method), pose_reference(calls), scheduled_contact_reward(function), unload_fraction(calls), transition_yaw_reward(function), smoothstep(calls), +1 more
    
    ```python
    66            return self._command
    67
    68        @property
    69        def reference(self):
    70            return pose_reference(self.standing, self.target, self.command[:, 1])
    71
    72        def contacts(self):
    73            return (self.sensor.data.net_forces_w[:, self.contact_ids].norm(dim=-1) > self.cfg.contact_threshold).float()
    
    ... (gap) ...
    
    199        return term.weights[term.episode_level] * torch.exp(-k_pose * error)
    200
    201
    202    def scheduled_contact_reward(env):
    203        term = env.command_manager.get_term("pivot")
    204
    205        contact = term.contacts()
    206        phase = term.command[:, 1]
    207        unload = unload_fraction(phase)
    208
    209        # FL, FR, HL, HR
    210        four_wheel = contact.mean(dim=-1)
    211
    212        # Desired final support: FL + HR
    213        diagonal = 0.5 * (
    214            contact[:, 0]
    215            + contact[:, 3]
    216            - contact[:, 1]
    217            - contact[:, 2]
    218        )
    219
    220        return (1.0 - unload) * four_wheel + unload * diagonal
    221
    222    def transition_yaw_reward(env, k_yaw=11.11):
    223        term = env.command_manager.get_term("pivot")
    224        error = term.robot.data.root_ang_vel_b[:, 2] - term.command[:, 0]
    225        return smoothstep((term.command[:, 1] - 0.8) / 0.2) * torch.exp(-k_yaw * error.square())
    226
    227
    228    def reset_transition_standing(
    
    ... (gap) ...
    
    244        reset_four_wheel_standing(env, env_ids, **kwargs)
    245
    246
    247    def transition_push(env, env_ids, max_velocity=0.15):
    248        term = env.command_manager.get_term("pivot")
    249        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    250        ids = ids[(term.episode_level[ids] >= 3) & (term.command[ids, 1] >= 1)]
    251        velocity = term.robot.data.root_vel_w[ids].clone()
    252        velocity[:, :2] += torch.empty(len(ids), 2, device=env.device).uniform_(-max_velocity, max_velocity)
    253        if len(ids):
    254            term.robot.write_root_velocity_to_sim(velocity, env_ids=ids)
    255
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/velocity_env_cfg.py`** — instantiates(instantiates), configclass(decorates), SceneEntityCfg(instantiates), TerminationsCfg(class), LocomotionVelocityRoughEnvCfg(class), TerminationsCfg(instantiates), +3 more
    
    ```python
    789        # ) # negetive
    790
    791
    792    @configclass
    793    class TerminationsCfg:
    794        """Termination terms for the MDP."""
    795
    796        # MDP terminations
    797        time_out = DoneTerm(func=mdp.time_out, time_out=True)
    798        # command_resample
    799        terrain_out_of_bounds = DoneTerm(
    800            func=mdp.terrain_out_of_bounds,
    801            params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
    802            time_out=True,
    803        )
    804
    805        # Contact sensor
    806        illegal_contact = DoneTerm(
    807            func=mdp.illegal_contact,
    808            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0},
    809        )
    810
    811        bad_orientation_2 = DoneTerm(func=mdp.bad_orientation_2)
    812
    813
    814
    
    ... (gap) ...
    
    900                        setattr(self.rewards, attr, None)
    901
    902
    903    def create_obsgroup_class(class_name, terms, enable_corruption=False, concatenate_terms=True):
    904        """
    905        Dynamically create and register a ObsGroup class based on the given configuration terms.
    906
    907        :param class_name: Name of the configuration class.
    908        :param terms: Configuration terms, a dictionary where keys are term names and values are term content.
    909        :param enable_corruption: Whether to enable corruption for the observation group. Defaults to False.
    910        :param concatenate_terms: Whether to concatenate the observation terms in the group. Defaults to True.
    911        :return: The dynamically created class.
    912        """
    913        # Dynamically determine the module name
    914        module_name = inspect.getmodule(inspect.currentframe()).__name__
    915
    916        # Define the post-init function
    917        def post_init_wrapper(self):
    918            setattr(self, "enable_corruption", enable_corruption)
    ```
    
    
    > Some file sections were trimmed for size. For a specific symbol you still need, run another `codegraph_explore` (or `codegraph_node`) with its exact name — line-numbered source, cheaper and more complete than Read.
    ✓ • 295ms

## Activity

    $ codegraph explore "yaw_env_fsm_cfg RewardsCfg TRANSITION com_support transition_support_load fsm_gated_tracking stability termination reward weights"
    **Exploration: yaw_env_fsm_cfg RewardsCfg TRANSITION com_support transition_support_load fsm_gated_tracking stability termination reward weights**
    
    Found 19 symbols across 5 files.
    
    **Blast radius — what depends on these (update/verify before editing)**
    
    - `RewardsCfg` (source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/balance_env_cfg.py:237) — 8 callers in `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/deeprobotics_m20/rough_env_cfg.py`, `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/rotate_env_cfg.py`, `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr_wheel/rough_env_cfg.py`, `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr_wheel/rough_env_cfg_imu.py` +4 more; no tests found within 3 caller hops
    - `RewardsCfg` (source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/velocity_env_cfg.py:379) — 1 caller in `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/velocity_env_cfg.py`; no tests found within 3 caller hops
    - `RewardsCfg` (source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py:379) — 1 caller in `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`; no tests found within 3 caller hops
    - `RewardsCfg` (source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_yaw_env_cfg.py:454) — 1 caller in `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_yaw_env_cfg.py`; no tests found within 3 caller hops
    - `VQRWheelRewardsCfg` (source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:88) — 1 caller in `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py`; no tests found within 3 caller hops
    
    **Source Code**
    
    > The code below is the **verbatim, current on-disk source** of these files — re-read from disk on this call and line-numbered, byte-for-byte identical to what the Read tool returns. It is NOT a summary, outline, or stale cache. Treat each block as a Read you have already performed: do not Read a file shown here.
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py`** — calls(calls), fsm_gated_tracking(function)
    
    ```python
    866        env._yaw_fsm_positive_budget_neg += positive * gates["diag_neg"].to(positive.dtype)
    867
    868
    869    def fsm_gated_tracking(
    870        env: ManagerBasedRLEnv,
    871        command_name: str,
    872        fsm_command_name: str,
    873        support_sensor_cfg: SceneEntityCfg,
    874        support_sensor_cfg_mirror: SceneEntityCfg,
    875        lifted_asset_cfg: SceneEntityCfg,
    876        lifted_asset_cfg_mirror: SceneEntityCfg,
    877        wheel_radius: float,
    878        target_clearance: float,
    879        std: float,
    880        contact_threshold: float = 1.0,
    881        clearance_gate_floor: float = 0.0,
    882        clearance_gate_floor_decay_s: float = 0.0,
    883        edge_command_fraction: float = 0.80,
    884        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    885        lift_progress_threshold: float = 0.80,
    886        support_threshold: float = 0.85,
    887    ) -> torch.Tensor:
    888        """Track signed yaw commands during TRANSITION and YAW states.
    889
    890        Tracking and lift-scaled support contact are available during transition;
    891        curriculum tracking accumulators update only in ``YAW_POS``/``YAW_NEG``.
    892        The signed command preserves the direction constraint for either branch.
    893        """
    894        if std <= 0.0:
    895            raise ValueError("std must be positive.")
    896        if target_clearance <= 0.0:
    897            raise ValueError("target_clearance must be positive.")
    898        if not 0.0 <= clearance_gate_floor <= 1.0:
    899            raise ValueError("clearance_gate_floor must be in [0, 1].")
    900        if clearance_gate_floor_decay_s < 0.0:
    901            raise ValueError("clearance_gate_floor_decay_s must be non-negative.")
    902        if not 0.0 < edge_command_fraction <= 1.0:
    903            raise ValueError("edge_command_fraction must be in (0, 1].")
    904
    905        gates = fsm_gates(env, fsm_command_name)
    906        _fsm_step_telemetry(env, gates)
    907        support_pos_contact = _yaw_wheel_contacts(env, support_sensor_cfg, contact_threshold)
    908        support_neg_contact = _yaw_wheel_contacts(env, support_sensor_cfg_mirror, contact_threshold)
    909        support_pos = support_pos_contact.to(dtype=torch.float32).prod(dim=1)
    910        support_neg = support_neg_contact.to(dtype=torch.float32).prod(dim=1)
    911        support_gate = torch.where(gates["diag_pos"], support_pos, support_neg)
    912        lost_support = gates["b_yaw"] & (support_gate == 0.0)
    913        if not hasattr(env, "_yaw_fsm_support_loss_run_steps"):
    914            env._yaw_fsm_support_loss_run_steps = torch.zeros(
    915                env.num_envs, device=support_gate.device, dtype=torch.long
    916            )
    917        env._yaw_fsm_support_loss_run_steps = torch.where(
    918            lost_support,
    919            env._yaw_fsm_support_loss_run_steps + 1,
    920            torch.zeros_like(env._yaw_fsm_support_loss_run_steps),
    921        )
    922        for suffix, direction in (("pos", gates["diag_pos"]), ("neg", gates["diag_neg"])):
    923            name = f"_yaw_fsm_{suffix}_support_loss_max_steps"
    924            if not hasattr(env, name):
    925                setattr(env, name, torch.zeros_like(env._yaw_fsm_support_loss_run_steps))
    926            previous_max = getattr(env, name)
    927            previous_max.copy_(torch.where(
    928                direction & lost_support,
    929                torch.maximum(previous_max, env._yaw_fsm_support_loss_run_steps),
    930                previous_max,
    931            ))
    932        support_contacts = torch.where(
    933            gates["diag_pos"].unsqueeze(1), support_pos_contact, support_neg_contact
    934        )
    935        support_shape = _yaw_support_shape(support_contacts)
    936        # The opposite support diagonal is exactly the active swing diagonal.
    937        swing_contact = select_swing_wheel_contact(
    938            support_pos_contact,
    939            support_neg_contact,
    940            gates["support_diagonal"],
    941        )
    942
    943        lift_pos = _yaw_lift_progress(env, lifted_asset_cfg, wheel_radius, target_clearance)
    944        lift_neg = _yaw_lift_progress(env, lifted_asset_cfg_mirror, wheel_radius, target_clearance)
    945        selected_lift = torch.where(gates["diag_pos"].unsqueeze(1), lift_pos, lift_neg)
    946        lift_progress = selected_lift.mean(dim=1)
    947
    948        asset: RigidObject = env.scene[asset_cfg.name]
    949        _fsm_transition_telemetry(
    950            env, gates, support_contacts, selected_lift, asset,
    951            env.command_manager.get_term(fsm_command_name),
    952        )
    953        yaw_command = env.command_manager.get_command(command_name)[:, 0]
    954        yaw_error = torch.abs(yaw_command - asset.data.root_ang_vel_b[:, 2])
    955        yaw_tracking = torch.exp(-yaw_error.square() / std**2)
    956
    957        # Metrics are intentionally YAW-only; otherwise transition/FOUR steps
    958        # dilute the curriculum score and make the scale ladder stall.
    959        yaw_mask = gates["b_yaw"].to(dtype=yaw_command.dtype)
    960        # Reuse the legacy yaw-curriculum accumulator names.  This keeps the
    961        # existing checkpoint/export path valid while changing only the sample
    962        # mask: FSM tracking statistics are normalized by YAW-state steps.
    963        _fsm_masked_accumulate(
    964            env,
    965            support_gate,
    966            yaw_mask,
    967            "_yaw_support_score_sum",
    968            "_yaw_support_score_samples",
    969        )
    970        _fsm_masked_accumulate(
    971            env,
    972            support_gate,
    973            yaw_mask,
    974            "_yaw_gate_open_sum",
    975            "_yaw_gate_open_samples",
    976        )
    977        _fsm_masked_accumulate_pair(
    978            env,
    979            torch.abs(yaw_command),
    980            yaw_error,
    981            yaw_mask,
    982            "_yaw_command_abs_sum",
    983            "_yaw_rate_abs_error_sum",
    984            "_yaw_tracking_metric_samples",
    985        )
    986
    987        # The legacy accumulators above deliberately remain direction-agnostic:
    988        # the original yaw curriculum and checkpoint export consume them.  The
    989        # FSM curriculum additionally needs independent evidence for each
    990        # diagonal; a strong POS branch must never promote a weak NEG branch.
    991        yaw_pos_mask = gates["b_yaw"] & gates["diag_pos"]
    992        yaw_neg_mask = gates["b_yaw"] & gates["diag_neg"]
    993        for suffix, contacts, direction_mask, wheel_names in (
    994            ("pos", support_pos_contact, yaw_pos_mask, ("FL", "HR")),
    995            ("neg", support_neg_contact, yaw_neg_mask, ("FR", "HL")),
    996        ):
    997            for index, wheel_name in enumerate(wheel_names):
    998                _fsm_masked_accumulate(
    999                    env,
    1000                    contacts[:, index].to(dtype=yaw_command.dtype),
    1001                    direction_mask.to(dtype=yaw_command.dtype),
    1002                    f"_yaw_fsm_{suffix}_support_{wheel_name}_sum",
    1003                    f"_yaw_fsm_{suffix}_support_{wheel_name}_samples",
    1004                )
    1005        for suffix, direction_mask in (("pos", yaw_pos_mask), ("neg", yaw_neg_mask)):
    1006            mask = direction_mask.to(dtype=yaw_command.dtype)
    1007            _fsm_masked_accumulate(
    1008                env,
    1009                lift_progress,
    1010                mask,
    1011                f"_yaw_fsm_{suffix}_lift_sum",
    1012                f"_yaw_fsm_{suffix}_yaw_samples",
    1013            )
    1014            _fsm_masked_accumulate(
    1015                env,
    1016                support_gate,
    1017                mask,
    1018                f"_yaw_fsm_{suffix}_support_sum",
    1019                f"_yaw_fsm_{suffix}_support_samples",
    1020            )
    1021            _fsm_masked_accumulate_pair(
    1022                env,
    1023                torch.abs(yaw_command),
    1024                yaw_error,
    1025                mask,
    1026                f"_yaw_fsm_{suffix}_command_abs_sum",
    1027                f"_yaw_fsm_{suffix}_yaw_abs_error_sum",
    1028                f"_yaw_fsm_{suffix}_tracking_samples",
    1029            )
    1030            _fsm_masked_accumulate(
    1031                env,
    1032                swing_contact.to(dtype=yaw_command.dtype),
    1033                mask,
    1034                f"_yaw_fsm_{suffix}_swing_contact_sum",
    1035                f"_yaw_fsm_{suffix}_swing_contact_samples",
    1036            )
    1037
    1038        _fsm_attempt_telemetry(
    1039            env, gates, lift_progress, support_gate, lift_progress_threshold, support_threshold
    1040        )
    1041        env._yaw_support_score_current = support_gate
    1042        env._yaw_gate_open_current = support_gate
    1043        env._yaw_command_abs_current = torch.abs(yaw_command)
    1044        env._yaw_rate_abs_error_current = yaw_error
    1045
    1046        command_term = env.command_manager.get_term(command_name)
    1047        yaw_limits = torch.as_tensor(
    1048            command_term.cfg.yaw_rate_range,
    1049            device=yaw_command.device,
    1050            dtype=yaw_command.dtype,
    1051        )
    1052        yaw_limit = yaw_limits.abs().amax()
    1053        edge_mask = (torch.abs(yaw_command) >= edge_command_fraction * yaw_limit) & gates["b_yaw"]
    1054        _fsm_masked_accumulate_pair(
    1055            env,
    1056            torch.abs(yaw_command),
    1057            yaw_error,
    1058            edge_mask.to(dtype=yaw_command.dtype),
    1059            "_yaw_edge_command_abs_sum",
    1060            "_yaw_edge_rate_abs_error_sum",
    1061            "_yaw_edge_tracking_samples",
    1062        )
    1063
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py`** — SceneEntityCfg(instantiates), configclass(decorates), VQRWheelRewardsCfg(class), VQRWheelFSMRewardsCfg(class)
    
    ```python
    85
    86
    87    @configclass
    88    class VQRWheelRewardsCfg:
    89        """The explicit reward set for diagonal-support rotate-in-place training."""
    90
    91        com_support = RewTerm(
    92            func=mdp.yaw_com_support,
    93            weight=3.0,
    94            params={
    95                "asset_cfg": SceneEntityCfg(
    96                    "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
    97                ),
    98                "std": 0.08,
    99            },
    100        )
    101        base_height = RewTerm(
    102            func=mdp.yaw_base_height_tracking,
    103            weight=2.0,
    104            params={
    105                "asset_cfg": SceneEntityCfg("robot"),
    106                "target_height": TARGET_BASE_HEIGHT,
    107                "error_scale": 0.10,
    108            },
    109        )
    110        low_base_height = RewTerm(
    111            func=mdp.yaw_low_base_height_l1,
    112            weight=-4.0,
    113            params={
    114                "asset_cfg": SceneEntityCfg("robot"),
    115                "minimum_height": MIN_BASE_HEIGHT,
    116                "error_scale": 0.10,
    117            },
    118        )
    119        downward_low_base_velocity = RewTerm(
    120            func=mdp.yaw_downward_low_base_velocity_l2,
    121            weight=-8.0,
    122            params={
    123                "asset_cfg": SceneEntityCfg("robot"),
    124                "minimum_height": MIN_BASE_HEIGHT,
    125                "height_margin": 0.10,
    126            },
    127        )
    128        support_span_band = RewTerm(
    129            func=mdp.yaw_support_span_band_l2,
    130            weight=-1.0,
    131            params={
    132                "asset_cfg": SceneEntityCfg(
    133                    "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
    134                ),
    135                "minimum_span": SUPPORT_SPAN_MIN,
    136                "maximum_span": SUPPORT_SPAN_MAX,
    137                "std": 0.05,
    138            },
    139        )
    140        lift_clearance = RewTerm(
    141            func=mdp.yaw_lift_clearance,
    142            weight=3.0,
    143            params={
    144                "asset_cfg": SceneEntityCfg(
    145                    "robot", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
    146                ),
    147                "wheel_radius": WHEEL_RADIUS,
    148                "target_clearance": LIFT_CLEARANCE_LEVELS[0],
    149            },
    150        )
    151        com_inside_segment = RewTerm(
    152            func=mdp.yaw_com_inside_support_segment,
    153            weight=2.0,
    154            params={
    155                "asset_cfg": SceneEntityCfg(
    156                    "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
    157                ),
    158                "std": 0.05,
    159            },
    160        )
    161        balance = RewTerm(
    162            func=mdp.yaw_balance,
    163            weight=2.0,
    164            params={
    165                "nominal_roll": 0.0,
    166                "nominal_pitch": 0.0,
    167                "std": 0.25,
    168                "fsm_command_name": "yaw_rate_cmd",
    169            },
    170        )
    171        gated_yaw_tracking = RewTerm(
    172            func=mdp.yaw_gated_tracking,
    173            weight=8.0,
    174            params={
    175                "command_name": "yaw_rate_cmd",
    176                "support_sensor_cfg": SceneEntityCfg(
    177                    "contact_forces",
    178                    body_names=SUPPORT_WHEEL_NAMES,
    179                    preserve_order=True,
    180                ),
    181                "lifted_asset_cfg": SceneEntityCfg(
    182                    "robot",
    183                    body_names=LIFTED_WHEEL_NAMES,
    184                    preserve_order=True,
    185                ),
    186                "wheel_radius": WHEEL_RADIUS,
    187                "target_clearance": LIFT_CLEARANCE_LEVELS[0],
    188                "std": 0.30,
    189                "contact_threshold": 1.0,
    190                "clearance_gate_floor": 0.25,
    191                "edge_command_fraction": 0.80,
    192            },
    193        )
    194        lateral_slip = RewTerm(
    195            func=mdp.yaw_lateral_wheel_slip,
    196            weight=-2.0,
    197            params={
    198                "sensor_cfg": SceneEntityCfg(
    199                    "contact_forces", body_names=WHEEL_NAMES, preserve_order=True
    200                ),
    201                "asset_cfg": SceneEntityCfg(
    202                    "robot", body_names=WHEEL_NAMES, preserve_order=True
    203                ),
    204                "threshold": 1.0,
    205            },
    206        )
    207        rolling_slip = RewTerm(
    208            func=mdp.yaw_rolling_wheel_slip,
    209            weight=-0.5,
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/balance_env_cfg.py`** — SceneEntityCfg(instantiates), configclass(decorates), instantiates(instantiates), RewardsCfg(class), VQRTwoWheelBalanceEnvCfg(class), TwoWheelBalanceSceneCfg(instantiates), +3 more
    
    ```python
    233        )
    234
    235
    236    @configclass
    237    class RewardsCfg:
    238        """Seven task-specific rewards; no locomotion reward set is inherited."""
    239
    240        support_contact = RewTerm(
    241            func=mdp.pivot_support_contact,
    242            weight=1.0,
    243            params={
    244                "sensor_cfg": SceneEntityCfg(
    245                    "contact_forces", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
    246                ),
    247                "threshold": 1.0,
    248            },
    249        )
    250        lifted_diagonal = RewTerm(
    251            func=mdp.pivot_lifted_wheels,
    252            weight=1.0,
    253            params={
    254                "sensor_cfg": SceneEntityCfg(
    255                    "contact_forces", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
    256                ),
    257                "asset_cfg": SceneEntityCfg("robot", body_names=LIFTED_WHEEL_NAMES, preserve_order=True),
    258                "wheel_radius": WHEEL_RADIUS,
    259                "minimum_clearance": 0.05,
    260                "threshold": 1.0,
    261            },
    262        )
    263        balance = RewTerm(
    264            func=mdp.pivot_balance,
    265            weight=2.0,
    266            params={
    267                "nominal_roll": NOMINAL_ROLL,
    268                "nominal_pitch": NOMINAL_PITCH,
    269                "std": 0.25,
    270            },
    271        )
    272        angular_stability = RewTerm(func=mdp.pivot_angular_stability, weight=-0.10)
    273        planar_drift = RewTerm(func=mdp.pivot_planar_drift, weight=-0.50)
    274        action_rate = RewTerm(func=mdp.pivot_action_rate_l2, weight=-0.01)
    275        leg_effort = RewTerm(
    276            func=mdp.pivot_leg_effort_l2,
    277            weight=-1.0e-5,
    278            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    279        )
    280
    281
    282    @configclass
    
    ... (gap) ...
    
    306
    307
    308    @configclass
    309    class VQRTwoWheelBalanceEnvCfg(ManagerBasedRLEnvCfg):
    310        """M1 balance task configuration."""
    311
    312        decimation = 4
    313        episode_length_s = 10.0
    314        sim = sim_utils.SimulationCfg(dt=0.005, render_interval=decimation, device="cuda:0")
    315        scene = TwoWheelBalanceSceneCfg(num_envs=16, env_spacing=2.5)
    316        observations = ObservationsCfg()
    317        actions = ActionsCfg()
    318        commands = None
    319        rewards = RewardsCfg()
    320        terminations = TerminationsCfg()
    321        events = EventCfg()
    322        curriculum = None
    323
    324        def __post_init__(self):
    325            self.sim.physics_material = self.scene.terrain.physics_material
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/velocity_env_cfg.py`** — SceneEntityCfg(instantiates), configclass(decorates), RewardsCfg(class), create_joint_deviation_l1_rewterm(method)
    
    ```python
    376
    377
    378    @configclass
    379    class RewardsCfg:
    380        """Reward terms for the MDP."""
    381
    382        # General
    383        is_terminated = RewTerm(func=mdp.is_terminated, weight=0.0)
    384
    385        # Root penalties
    386        lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    387        ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
    388        flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    389        base_height_l2 = RewTerm(
    390            func=mdp.base_height_l2,
    391            weight=0.0,
    392            params={
    393                "asset_cfg": SceneEntityCfg("robot", body_names=""),
    394                "sensor_cfg": SceneEntityCfg("height_scanner_base"),
    395                "target_height": 0.0,
    396            },
    397        )
    398        body_lin_acc_l2 = RewTerm(
    399            func=mdp.body_lin_acc_l2,
    400            weight=0.0,
    401            params={"asset_cfg": SceneEntityCfg("robot", body_names="")},
    402        )
    403
    404        # Joint penalties
    405        joint_torques_l2 = RewTerm(
    406            func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    407        )
    408        joint_vel_l2 = RewTerm(
    409            func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    410        )
    411        joint_acc_l2 = RewTerm(
    412            func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    413        )
    414
    415        joint_deviation_l1 = RewTerm(
    416                func=mdp.joint_deviation_l1,
    417                weight=0.0,
    418                params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    419            )
    420
    421        def create_joint_deviation_l1_rewterm(self, attr_name, weight, joint_names_pattern):
    422            rew_term = RewTerm(
    423                func=mdp.joint_deviation_l1,
    424                weight=weight,
    425                params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names_pattern)},
    426            )
    427            setattr(self, attr_name, rew_term)
    428
    429        joint_pos_limits = RewTerm(
    430            func=mdp.joint_pos_limits, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    431        )
    432        joint_vel_limits = RewTerm(
    433            func=mdp.joint_vel_limits,
    434            weight=0.0,
    435            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "soft_ratio": 1.0},
    436        )
    437        joint_power = RewTerm(
    438            func=mdp.joint_power,
    439            weight=0.0,
    440            params={
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`** — SceneEntityCfg(instantiates), configclass(decorates), RewardsCfg(class), create_joint_deviation_l1_rewterm(method)
    
    ```python
    376
    377
    378    @configclass
    379    class RewardsCfg:
    380        """Reward terms for the MDP."""
    381
    382        # General
    383        is_terminated = RewTerm(func=mdp.is_terminated, weight=0.0)
    384
    385        # Root penalties
    386        lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    387        ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
    388        flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    389        base_height_l2 = RewTerm(
    390            func=mdp.base_height_l2,
    391            weight=0.0,
    392            params={
    393                "asset_cfg": SceneEntityCfg("robot", body_names=""),
    394                "sensor_cfg": SceneEntityCfg("height_scanner_base"),
    395                "target_height": 0.0,
    396            },
    397        )
    398        body_lin_acc_l2 = RewTerm(
    399            func=mdp.body_lin_acc_l2,
    400            weight=0.0,
    401            params={"asset_cfg": SceneEntityCfg("robot", body_names="")},
    402        )
    403
    404        # Joint penalties
    405        joint_torques_l2 = RewTerm(
    406            func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    407        )
    408        joint_vel_l2 = RewTerm(
    409            func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    410        )
    411        joint_acc_l2 = RewTerm(
    412            func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    413        )
    414
    415        joint_deviation_l1 = RewTerm(
    416                func=mdp.joint_deviation_l1,
    417                weight=0.0,
    418                params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    419            )
    420
    421        def create_joint_deviation_l1_rewterm(self, attr_name, weight, joint_names_pattern):
    422            rew_term = RewTerm(
    423                func=mdp.joint_deviation_l1,
    424                weight=weight,
    425                params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names_pattern)},
    426            )
    427            setattr(self, attr_name, rew_term)
    428
    429        joint_pos_limits = RewTerm(
    430            func=mdp.joint_pos_limits, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    431        )
    432        joint_vel_limits = RewTerm(
    433            func=mdp.joint_vel_limits,
    434            weight=0.0,
    435            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "soft_ratio": 1.0},
    436        )
    437        joint_power = RewTerm(
    438            func=mdp.joint_power,
    439            weight=0.0,
    440            params={
    441                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    442            },
    443        )
    ```
    
    **Not shown above — explore these names for their source**
    
    - source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_yaw_env_cfg.py: RewardsCfg:454, create_obsgroup_class:978, create_joint_deviation_l1_rewterm:496, LocomotionVelocityRoughEnvCfg:914
    - source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/pivot_env_cfg.py: PivotRewardsCfg:287, PivotEnvCfg:493
    - ... and 16 more files
    
    > Some file sections were trimmed for size. For a specific symbol you still need, run another `codegraph_explore` (or `codegraph_node`) with its exact name — line-numbered source, cheaper and more complete than Read.
    ✓ • 295ms

## Activity

    $ codegraph explore "VQRWheelFSMRewardsCfg in yaw_env_fsm_cfg.py"
    **Exploration: VQRWheelFSMRewardsCfg in yaw_env_fsm_cfg.py**
    
    Found 68 symbols across 5 files. 1 file pinned from the query.
    
    **Blast radius — what depends on these (update/verify before editing)**
    
    - `RewardsCfg` (source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/balance_env_cfg.py:237) — 8 callers in `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/deeprobotics_m20/rough_env_cfg.py`, `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/rotate_env_cfg.py`, `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr_wheel/rough_env_cfg.py`, `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr_wheel/rough_env_cfg_imu.py` +4 more; no tests found within 3 caller hops
    - `RewardsCfg` (source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/velocity_env_cfg.py:379) — 1 caller in `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/velocity_env_cfg.py`; no tests found within 3 caller hops
    - `RewardsCfg` (source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py:379) — 1 caller in `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`; no tests found within 3 caller hops
    - `RewardsCfg` (source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_yaw_env_cfg.py:454) — 1 caller in `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_yaw_env_cfg.py`; no tests found within 3 caller hops
    - `VQRWheelFSMRewardsCfg` (source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:284) — 1 caller in `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py`; no tests found within 3 caller hops
    
    **Source Code**
    
    > The code below is the **verbatim, current on-disk source** of these files — re-read from disk on this call and line-numbered, byte-for-byte identical to what the Read tool returns. It is NOT a summary, outline, or stale cache. Treat each block as a Read you have already performed: do not Read a file shown here.
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py`** — SceneEntityCfg(instantiates), configclass(decorates), extends(extends), __post_init__(method), __post_init__(calls), VQRWheelActionsCfg(class), +31 more
    
    ```python
    281
    282
    283    @configclass
    284    class VQRWheelFSMRewardsCfg:
    285        """The 26-term reward contract for ``Flat-VQR-Wheel-Yaw-FSM``.
    286
    287        Safety/regularization penalties retain their baseline weights. Positive
    288        pose terms become nonpositive deficits in TRANSITION; YAW and RETURN
    289        retain their rewards. POS/NEG select the command's ``support_diagonal``.
    290        """
    291
    292        # ------------------------------ Group 1: always-on baseline ------------------------------
    293        balance = RewTerm(
    294            func=mdp.yaw_balance,
    295            weight=2.0,
    296            params={
    297                "nominal_roll": 0.0,
    298                "nominal_pitch": 0.0,
    299                "std": 0.25,
    300                "fsm_command_name": "yaw_rate_cmd",
    301            },
    302        )
    303        torque = RewTerm(
    304            func=mdp.yaw_joint_torque_l2,
    305            weight=-1.0e-4,
    306            params={
    307                "command_name": "yaw_rate_cmd",
    308                "yaw_reference": YAW_REF,
    309                "minimum_scale": 0.30,
    310                "asset_cfg": SceneEntityCfg(
    311                    "robot", joint_names=LEG_JOINT_NAMES + WHEEL_NAMES, preserve_order=True
    312                ),
    313            },
    314        )
    315        action_rate = RewTerm(func=mdp.yaw_action_rate_l2, weight=-0.02)
    316        joint_velocity = RewTerm(
    317            func=mdp.yaw_joint_velocity_l2,
    318            weight=-0.001,
    319            params={
    320                "asset_cfg": SceneEntityCfg(
    321                    "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
    322                )
    323            },
    324        )
    325        joint_limits = RewTerm(
    326            func=mdp.joint_pos_limits,
    327            weight=-0.5,
    328            params={
    329                "asset_cfg": SceneEntityCfg(
    330                    "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
    331                )
    332            },
    333        )
    334        lateral_slip = RewTerm(
    335            func=mdp.yaw_lateral_wheel_slip,
    336            weight=-2.0,
    337            params={
    338                "sensor_cfg": SceneEntityCfg(
    339                    "contact_forces", body_names=WHEEL_NAMES, preserve_order=True
    340                ),
    341                "asset_cfg": SceneEntityCfg(
    342                    "robot", body_names=WHEEL_NAMES, preserve_order=True
    343                ),
    344                "threshold": 1.0,
    345            },
    346        )
    347        undesired_contact = RewTerm(
    348            func=mdp.undesired_contacts,
    349            weight=-2.0,
    350            params={
    351                "sensor_cfg": SceneEntityCfg(
    352                    "contact_forces",
    353                    body_names=["^(?!(TORSO|.*_WHEEL)$).*"],
    354                    preserve_order=True,
    355                ),
    356                "threshold": 1.0,
    357            },
    358        )
    359        planar_velocity = RewTerm(func=mdp.yaw_planar_velocity_l2, weight=-1.0)
    360        low_base_height = RewTerm(
    361            func=mdp.yaw_low_base_height_l1,
    362            weight=-4.0,
    363            params={
    364                "asset_cfg": SceneEntityCfg("robot"),
    365                "minimum_height": MIN_BASE_HEIGHT,
    366                "error_scale": 0.10,
    367            },
    368        )
    369        transition_low_base_height = RewTerm(
    370            func=mdp.yaw_transition_low_base_height,
    371            weight=-4.0,
    372            params={
    373                "asset_cfg": SceneEntityCfg("robot"),
    374                "warning_height": TRANSITION_HEIGHT_WARNING,
    375                "minimum_height": MIN_BASE_HEIGHT,
    376                "fsm_command_name": "yaw_rate_cmd",
    377            },
    378        )
    379        downward_low_base_velocity = RewTerm(
    380            func=mdp.yaw_downward_low_base_velocity_l2,
    381            weight=-8.0,
    382            params={
    383                "asset_cfg": SceneEntityCfg("robot"),
    384                "minimum_height": MIN_BASE_HEIGHT,
    385                "height_margin": 0.10,
    386                "warning_height": TRANSITION_HEIGHT_WARNING,
    387                "fsm_command_name": "yaw_rate_cmd",
    388            },
    389        )
    390
    391        # ------------------------------ Group 2: FSM-gated geometry ------------------------------
    392        com_support = RewTerm(
    393            func=mdp.yaw_com_support,
    394            weight=3.0,
    395            params={
    396                "asset_cfg": SceneEntityCfg(
    397                    "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
    398                ),
    399                "asset_cfg_mirror": SceneEntityCfg(
    400                    "robot", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
    401                ),
    402                "std": 0.08,
    403                "fsm_command_name": "yaw_rate_cmd",
    404            },
    405        )
    406        com_inside_segment = RewTerm(
    407            func=mdp.yaw_com_inside_support_segment,
    408            weight=2.0,
    409            params={
    410                "asset_cfg": SceneEntityCfg(
    411                    "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
    412                ),
    413                "asset_cfg_mirror": SceneEntityCfg(
    414                    "robot", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
    415                ),
    416                "std": 0.05,
    417                "fsm_command_name": "yaw_rate_cmd",
    418            },
    419        )
    420        support_span_band = RewTerm(
    421            func=mdp.yaw_support_span_band_l2,
    422            weight=-1.0,
    423            params={
    424                "asset_cfg": SceneEntityCfg(
    425                    "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
    426                ),
    427                "asset_cfg_mirror": SceneEntityCfg(
    428                    "robot", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
    429                ),
    430                "minimum_span": SUPPORT_SPAN_MIN,
    431                "maximum_span": SUPPORT_SPAN_MAX,
    432                "std": 0.05,
    433                "fsm_command_name": "yaw_rate_cmd",
    434            },
    435        )
    436        transition_support_load = RewTerm(
    437            func=mdp.yaw_transition_support_load,
    438            weight=3.0,
    439            params={
    440                "sensor_cfg": SceneEntityCfg(
    441                    "contact_forces", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
    442                ),
    443                "sensor_cfg_mirror": SceneEntityCfg(
    444                    "contact_forces", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
    445                ),
    446                "fsm_command_name": "yaw_rate_cmd",
    447                "target_force_n": SUPPORT_LOAD_TARGET_N,
    448            },
    449        )
    450        lift_clearance = RewTerm(
    451            func=mdp.yaw_lift_clearance,
    452            weight=3.0,
    453            params={
    454                "asset_cfg": SceneEntityCfg(
    455                    "robot", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
    456                ),
    457                "asset_cfg_mirror": SceneEntityCfg(
    458                    "robot", body_names=LIFTED_WHEEL_NAMES_MIRROR, preserve_order=True
    459                ),
    460                "support_sensor_cfg": SceneEntityCfg(
    461                    "contact_forces", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
    462                ),
    463                "support_sensor_cfg_mirror": SceneEntityCfg(
    464                    "contact_forces", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
    465                ),
    466                "support_force_target_n": SUPPORT_LOAD_TARGET_N,
    467                "transition_ungated_fraction": TRANSITION_LIFT_UNGATED_FRACTION,
    468                "wheel_radius": WHEEL_RADIUS,
    469                "target_clearance": LIFT_CLEARANCE_LEVELS[0],
    470                "fsm_command_name": "yaw_rate_cmd",
    471            },
    472        )
    473        base_height = RewTerm(
    474            func=mdp.yaw_base_height_tracking,
    475            weight=0.49,
    476            params={
    477                "asset_cfg": SceneEntityCfg("robot"),
    478                "target_height": TARGET_BASE_HEIGHT,
    479                "error_scale": 0.10,
    480                "fsm_command_name": "yaw_rate_cmd",
    481            },
    482        )
    483        lifted_wheel_spin = RewTerm(
    484            func=mdp.yaw_lifted_wheel_spin_l2,
    485            weight=-0.02,
    486            params={
    487                "asset_cfg": SceneEntityCfg(
    488                    "robot", joint_names=LIFTED_WHEEL_NAMES, preserve_order=True
    489                ),
    490                "asset_cfg_mirror": SceneEntityCfg(
    491                    "robot", joint_names=LIFTED_WHEEL_NAMES_MIRROR, preserve_order=True
    492                ),
    493                "command_name": "yaw_rate_cmd",
    494                "yaw_reference": YAW_REF,
    495                "fsm_command_name": "yaw_rate_cmd",
    496            },
    497        )
    498        rolling_slip = RewTerm(
    499            func=mdp.yaw_rolling_wheel_slip,
    500            weight=-0.5,
    501            params={
    502                "sensor_cfg": SceneEntityCfg(
    503                    "contact_forces", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
    504                ),
    505                "sensor_cfg_mirror": SceneEntityCfg(
    506                    "contact_forces", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
    507                ),
    508                "body_asset_cfg": SceneEntityCfg(
    509                    "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
    510                ),
    511                "body_asset_cfg_mirror": SceneEntityCfg(
    512                    "robot", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
    513                ),
    514                "joint_asset_cfg": SceneEntityCfg(
    515                    "robot", joint_names=SUPPORT_WHEEL_NAMES, preserve_order=True
    516                ),
    517                "joint_asset_cfg_mirror": SceneEntityCfg(
    518                    "robot", joint_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
    519                ),
    520                "wheel_radius": WHEEL_RADIUS,
    521                "command_name": "yaw_rate_cmd",
    522                "yaw_reference": YAW_REF,
    523                "threshold": 1.0,
    524                "fsm_command_name": "yaw_rate_cmd",
    525            },
    526        )
    527        four_stand_stability = RewTerm(
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py`** — reset(calls), update(method), YawFSMVectorized(class), YawFSMCommand(class), YawRateCommand(extends), __init__(method), +20 more
    
    ```python
    529            return self.fsm_state
    530
    531
    532    class YawFSMCommand(YawRateCommand):
    533        """Yaw-rate command coupled to the per-environment yaw FSM.
    534
    535        ``VQRYawFSM`` is deliberately kept as the single source of truth for the
    
    ... (gap) ...
    
    545        cfg: "YawFSMCommandCfg"
    546
    547        def __init__(self, cfg: "YawFSMCommandCfg", env):
    548            super().__init__(cfg, env)
    549
    550            self._fsm = YawFSMVectorized(
    551                self.num_envs,
    552                device=self.device,
    553                yaw_enter=cfg.yaw_enter,
    
    ... (gap) ...
    
    597            self._fsm_scene_entities_resolved = False
    598
    599        @property
    600        def fsm_state(self) -> torch.Tensor:
    601            """Current FSM state for every environment as ``VQRFsmState`` values."""
    602            return self._fsm_state
    603
    604        def set_fsm_inputs(
    605            self,
    606            positive_pose_ready: torch.Tensor,
    607            negative_pose_ready: torch.Tensor,
    608            four_stand_ready: torch.Tensor,
    609            unsafe: torch.Tensor,
    610        ) -> None:
    611            """Set the batched readiness predicates consumed on the next update."""
    612            values = (
    613                (positive_pose_ready, self.positive_pose_ready),
    614                (negative_pose_ready, self.negative_pose_ready),
    615                (four_stand_ready, self.four_stand_ready),
    616                (unsafe, self.unsafe),
    617            )
    618            for value, target in values:
    619                if value.shape != target.shape:
    620                    raise ValueError(f"FSM predicate must have shape {tuple(target.shape)}, got {tuple(value.shape)}")
    621                target.copy_(value.to(device=self.device, dtype=torch.bool))
    622
    623        def _reset_fsm(self, env_ids: Sequence[int] | slice) -> None:
    624            self._fsm.reset(env_ids)
    625            self.yaw_entry_pos[env_ids] = 0.0
    626
    627            self.positive_pose_ready[env_ids] = False
    628            self.negative_pose_ready[env_ids] = False
    629            self.four_stand_ready[env_ids] = False
    630            self.unsafe[env_ids] = False
    631
    632        def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
    633            """Reset command metrics and FSM state for the selected environments."""
    634            extras = super().reset(env_ids)
    635            self._reset_fsm(slice(None) if env_ids is None else env_ids)
    636            return extras
    637
    638        def _step_fsm(self) -> None:
    639            self._update_fsm_predicates()
    640            self._fsm.update(
    641                yaw_cmd=self._command[:, 0],
    642                positive_pose_ready=self.positive_pose_ready,
    643                negative_pose_ready=self.negative_pose_ready,
    
    ... (gap) ...
    
    675
    676            from .observations import yaw_fsm_predicates
    677
    678            predicates = yaw_fsm_predicates(
    679                self._env,
    680                robot_name=self.cfg.asset_name,
    681                support_sensor_cfg=configs[0],
    
    ... (gap) ...
    
    692                unsafe_angle_limit=self.cfg.unsafe_angle_limit,
    693                minimum_base_height=self.cfg.minimum_base_height,
    694            )
    695            self.set_fsm_inputs(*predicates)
    696
    697        def _update_command(self):
    698            """Keep the base command behavior, then advance the FSM once."""
    699            super()._update_command()
    700            self._step_fsm()
    701
    702
    703    @configclass
    704    class YawFSMCommandCfg(YawRateCommandCfg):
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/balance_env_cfg.py`** — SceneEntityCfg(instantiates), configclass(decorates), RewardsCfg(class)
    
    ```python
    233        )
    234
    235
    236    @configclass
    237    class RewardsCfg:
    238        """Seven task-specific rewards; no locomotion reward set is inherited."""
    239
    240        support_contact = RewTerm(
    241            func=mdp.pivot_support_contact,
    242            weight=1.0,
    243            params={
    244                "sensor_cfg": SceneEntityCfg(
    245                    "contact_forces", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
    246                ),
    247                "threshold": 1.0,
    248            },
    249        )
    250        lifted_diagonal = RewTerm(
    251            func=mdp.pivot_lifted_wheels,
    252            weight=1.0,
    253            params={
    254                "sensor_cfg": SceneEntityCfg(
    255                    "contact_forces", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
    256                ),
    257                "asset_cfg": SceneEntityCfg("robot", body_names=LIFTED_WHEEL_NAMES, preserve_order=True),
    258                "wheel_radius": WHEEL_RADIUS,
    259                "minimum_clearance": 0.05,
    260                "threshold": 1.0,
    261            },
    262        )
    263        balance = RewTerm(
    264            func=mdp.pivot_balance,
    265            weight=2.0,
    266            params={
    267                "nominal_roll": NOMINAL_ROLL,
    268                "nominal_pitch": NOMINAL_PITCH,
    269                "std": 0.25,
    270            },
    271        )
    272        angular_stability = RewTerm(func=mdp.pivot_angular_stability, weight=-0.10)
    273        planar_drift = RewTerm(func=mdp.pivot_planar_drift, weight=-0.50)
    274        action_rate = RewTerm(func=mdp.pivot_action_rate_l2, weight=-0.01)
    275        leg_effort = RewTerm(
    276            func=mdp.pivot_leg_effort_l2,
    277            weight=-1.0e-5,
    278            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    279        )
    280
    281
    282    @configclass
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`** — SceneEntityCfg(instantiates), configclass(decorates), RewardsCfg(class), create_joint_deviation_l1_rewterm(method)
    
    ```python
    376
    377
    378    @configclass
    379    class RewardsCfg:
    380        """Reward terms for the MDP."""
    381
    382        # General
    383        is_terminated = RewTerm(func=mdp.is_terminated, weight=0.0)
    384
    385        # Root penalties
    386        lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    387        ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
    388        flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    389        base_height_l2 = RewTerm(
    390            func=mdp.base_height_l2,
    391            weight=0.0,
    392            params={
    393                "asset_cfg": SceneEntityCfg("robot", body_names=""),
    394                "sensor_cfg": SceneEntityCfg("height_scanner_base"),
    395                "target_height": 0.0,
    396            },
    397        )
    398        body_lin_acc_l2 = RewTerm(
    399            func=mdp.body_lin_acc_l2,
    400            weight=0.0,
    401            params={"asset_cfg": SceneEntityCfg("robot", body_names="")},
    402        )
    403
    404        # Joint penalties
    405        joint_torques_l2 = RewTerm(
    406            func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    407        )
    408        joint_vel_l2 = RewTerm(
    409            func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    410        )
    411        joint_acc_l2 = RewTerm(
    412            func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    413        )
    414
    415        joint_deviation_l1 = RewTerm(
    416                func=mdp.joint_deviation_l1,
    417                weight=0.0,
    418                params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    419            )
    420
    421        def create_joint_deviation_l1_rewterm(self, attr_name, weight, joint_names_pattern):
    422            rew_term = RewTerm(
    423                func=mdp.joint_deviation_l1,
    424                weight=weight,
    425                params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names_pattern)},
    426            )
    427            setattr(self, attr_name, rew_term)
    428
    429        joint_pos_limits = RewTerm(
    430            func=mdp.joint_pos_limits, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    431        )
    432        joint_vel_limits = RewTerm(
    433            func=mdp.joint_vel_limits,
    434            weight=0.0,
    435            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "soft_ratio": 1.0},
    436        )
    437        joint_power = RewTerm(
    438            func=mdp.joint_power,
    439            weight=0.0,
    440            params={
    441                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    442            },
    443        )
    444
    445        stand_still_without_cmd = RewTerm(
    446            func=mdp.stand_still_without_cmd,
    447            weight=0.0,
    448            params={
    449                "command_name": "base_velocity",
    450                "command_threshold": 0.1,
    451                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    452            },
    453        )
    454
    455        joint_pos_penalty = RewTerm(
    456            func=mdp.joint_pos_penalty,
    457            weight=0.0,
    458            params={
    459                "command_name": "base_velocity",
    460                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    461                "stand_still_scale": 5.0,
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_yaw_env_cfg.py`** — SceneEntityCfg(instantiates), configclass(decorates), RewardsCfg(class), create_joint_deviation_l1_rewterm(method)
    
    ```python
    451
    452
    453    @configclass
    454    class RewardsCfg:
    455        """Reward terms for the MDP."""
    456
    457        # General
    458        is_terminated = RewTerm(func=mdp.is_terminated, weight=0.0)
    459
    460        # Root penalties
    461        lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    462        ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
    463        flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    464        base_height_l2 = RewTerm(
    465            func=mdp.base_height_l2,
    466            weight=0.0,
    467            params={
    468                "asset_cfg": SceneEntityCfg("robot", body_names=""),
    469                "sensor_cfg": SceneEntityCfg("height_scanner_base"),
    470                "target_height": 0.0,
    471            },
    472        )
    473        body_lin_acc_l2 = RewTerm(
    474            func=mdp.body_lin_acc_l2,
    475            weight=0.0,
    476            params={"asset_cfg": SceneEntityCfg("robot", body_names="")},
    477        )
    478
    479        # Joint penalties
    480        joint_torques_l2 = RewTerm(
    481            func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    482        )
    483        joint_vel_l2 = RewTerm(
    484            func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    485        )
    486        joint_acc_l2 = RewTerm(
    487            func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    488        )
    489
    490        joint_deviation_l1 = RewTerm(
    491                func=mdp.joint_deviation_l1,
    492                weight=0.0,
    493                params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    494            )
    495
    496        def create_joint_deviation_l1_rewterm(self, attr_name, weight, joint_names_pattern):
    497            rew_term = RewTerm(
    498                func=mdp.joint_deviation_l1,
    499                weight=weight,
    500                params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names_pattern)},
    501            )
    502            setattr(self, attr_name, rew_term)
    503
    504        joint_pos_limits = RewTerm(
    505            func=mdp.joint_pos_limits, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    506        )
    507        joint_vel_limits = RewTerm(
    508            func=mdp.joint_vel_limits,
    509            weight=0.0,
    510            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "soft_ratio": 1.0},
    511        )
    512        joint_power = RewTerm(
    513            func=mdp.joint_power,
    514            weight=0.0,
    515            params={
    ```
    
    **Not shown above — explore these names for their source**
    
    - source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/velocity_env_cfg.py: RewardsCfg:379, create_obsgroup_class:903, create_joint_deviation_l1_rewterm:421, LocomotionVelocityRoughEnvCfg:839
    - source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm_gates.py: fsm_gates:51, _step_key:21, _required_tensor:37, _CACHE_ATTR:18, _CACHE_STEP_ATTR:17
    - source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py: yaw_transition_support_load:194, yaw_com_support:247, yaw_base_height_tracking:271, yaw_transition_low_base_height:322, yaw_downward_low_base_velocity_l2:338, yaw_support_span_band_l2:364, +15 more
    - pose_optimization/include/pose_optimization/wheel_contact_geometry.hpp: WheelCollisionGeometry:29
    - ... and 23 more files
    
    > Some file sections were trimmed for size. For a specific symbol you still need, run another `codegraph_explore` (or `codegraph_node`) with its exact name — line-numbered source, cheaper and more complete than Read.
    ✓ • 215ms

## Activity

    $ codegraph explore "yaw_env_fsm_cfg.py: VQRWheelFSMTerminationsCfg VQRWheelFSMRewardsCfg four_stand_stability transition_stability fsm_gated_tracking terminal failure"
    **Dynamic boundaries (the static path ends at runtime dispatch)**
    
    - `test_transition_stability_does_not_track_standing_height` (tests/test_yaw_transition_safety.py:52) — computed member call: `values.append(rewards["four_stand_stability"](`
      candidates for key `four_stand_stability`: `four_stand_stability` (source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:1087) ← you named this
    - `record` (tests/test_yaw_fsm_systemic.py:196) — computed member call: `namespace["_fsm_attempt_telemetry"](env, gates, torch.full((env.num_envs,), quality),`
      candidates for key `_fsm_attempt_telemetry`: `_fsm_attempt_telemetry` (source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:715)
    
    > These sites choose their call target at runtime (registry / bus / reflection) — the site shown IS where the flow continues. To follow it, run codegraph_explore or codegraph_node on a candidate; source for the sites above is included below.
    
    > Full source for these symbols is below — the call flow among them, followed by their bodies.
    **Exploration: yaw_env_fsm_cfg.py: VQRWheelFSMTerminationsCfg VQRWheelFSMRewardsCfg four_stand_stability transition_stability fsm_gated_tracking terminal failure**
    
    Found 42 symbols across 3 files.
    
    **Blast radius — what depends on these (update/verify before editing)**
    
    - `VQRWheelFSMRewardsCfg` (source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:284) — 1 caller in `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py`; no tests found within 3 caller hops
    - `FourWheelRewardsCfg` (source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/four_wheel_rotate_env_cfg.py:71) — 1 caller in `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/four_wheel_rotate_env_cfg.py`; tested via callers: `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/tests/test_four_wheel_rotate.py`
    - `FourWheelTerminationsCfg` (source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/four_wheel_rotate_env_cfg.py:101) — 1 caller in `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/four_wheel_rotate_env_cfg.py`; tested via callers: `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/tests/test_four_wheel_rotate.py`
    
    **Source Code**
    
    > The code below is the **verbatim, current on-disk source** of these files — re-read from disk on this call and line-numbered, byte-for-byte identical to what the Read tool returns. It is NOT a summary, outline, or stale cache. Treat each block as a Read you have already performed: do not Read a file shown here.
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py`** — _fsm_masked_accumulate(calls), fsm_gates(calls), _yaw_lift_progress(calls), _yaw_wheel_contacts(calls), _fsm_record_positive_budget(calls), _fsm_masked_accumulate_pair(calls), +18 more
    
    ```python
    816        if torso_cfg is None:
    817            return
    818        contact_threshold = command.cfg.contact_threshold
    819        torso_contact = _yaw_wheel_contacts(env, torso_cfg, contact_threshold)[:, 0]
    820        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    821        attitude_ready = (roll.abs() < command.cfg.pose_angle_limit) & (
    822            pitch.abs() < command.cfg.pose_angle_limit
    
    ... (gap) ...
    
    850        leaves the diagnostic unset and never changes the reward value.
    851        """
    852        try:
    853            weight = float(env.reward_manager.get_term_cfg(term_name).weight)
    854        except (AttributeError, KeyError, TypeError):
    855            return
    856        positive = torch.relu(value * weight)
    
    ... (gap) ...
    
    866        env._yaw_fsm_positive_budget_neg += positive * gates["diag_neg"].to(positive.dtype)
    867
    868
    869    def fsm_gated_tracking(
    870        env: ManagerBasedRLEnv,
    871        command_name: str,
    872        fsm_command_name: str,
    873        support_sensor_cfg: SceneEntityCfg,
    874        support_sensor_cfg_mirror: SceneEntityCfg,
    875        lifted_asset_cfg: SceneEntityCfg,
    876        lifted_asset_cfg_mirror: SceneEntityCfg,
    877        wheel_radius: float,
    878        target_clearance: float,
    879        std: float,
    880        contact_threshold: float = 1.0,
    881        clearance_gate_floor: float = 0.0,
    882        clearance_gate_floor_decay_s: float = 0.0,
    883        edge_command_fraction: float = 0.80,
    884        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    885        lift_progress_threshold: float = 0.80,
    886        support_threshold: float = 0.85,
    887    ) -> torch.Tensor:
    888        """Track signed yaw commands during TRANSITION and YAW states.
    889
    890        Tracking and lift-scaled support contact are available during transition;
    891        curriculum tracking accumulators update only in ``YAW_POS``/``YAW_NEG``.
    892        The signed command preserves the direction constraint for either branch.
    893        """
    894        if std <= 0.0:
    895            raise ValueError("std must be positive.")
    896        if target_clearance <= 0.0:
    897            raise ValueError("target_clearance must be positive.")
    898        if not 0.0 <= clearance_gate_floor <= 1.0:
    899            raise ValueError("clearance_gate_floor must be in [0, 1].")
    900        if clearance_gate_floor_decay_s < 0.0:
    901            raise ValueError("clearance_gate_floor_decay_s must be non-negative.")
    902        if not 0.0 < edge_command_fraction <= 1.0:
    903            raise ValueError("edge_command_fraction must be in (0, 1].")
    904
    905        gates = fsm_gates(env, fsm_command_name)
    906        _fsm_step_telemetry(env, gates)
    907        support_pos_contact = _yaw_wheel_contacts(env, support_sensor_cfg, contact_threshold)
    908        support_neg_contact = _yaw_wheel_contacts(env, support_sensor_cfg_mirror, contact_threshold)
    909        support_pos = support_pos_contact.to(dtype=torch.float32).prod(dim=1)
    910        support_neg = support_neg_contact.to(dtype=torch.float32).prod(dim=1)
    911        support_gate = torch.where(gates["diag_pos"], support_pos, support_neg)
    912        lost_support = gates["b_yaw"] & (support_gate == 0.0)
    913        if not hasattr(env, "_yaw_fsm_support_loss_run_steps"):
    914            env._yaw_fsm_support_loss_run_steps = torch.zeros(
    915                env.num_envs, device=support_gate.device, dtype=torch.long
    916            )
    917        env._yaw_fsm_support_loss_run_steps = torch.where(
    918            lost_support,
    919            env._yaw_fsm_support_loss_run_steps + 1,
    920            torch.zeros_like(env._yaw_fsm_support_loss_run_steps),
    921        )
    922        for suffix, direction in (("pos", gates["diag_pos"]), ("neg", gates["diag_neg"])):
    923            name = f"_yaw_fsm_{suffix}_support_loss_max_steps"
    924            if not hasattr(env, name):
    925                setattr(env, name, torch.zeros_like(env._yaw_fsm_support_loss_run_steps))
    926            previous_max = getattr(env, name)
    927            previous_max.copy_(torch.where(
    928                direction & lost_support,
    929                torch.maximum(previous_max, env._yaw_fsm_support_loss_run_steps),
    930                previous_max,
    931            ))
    932        support_contacts = torch.where(
    933            gates["diag_pos"].unsqueeze(1), support_pos_contact, support_neg_contact
    934        )
    935        support_shape = _yaw_support_shape(support_contacts)
    936        # The opposite support diagonal is exactly the active swing diagonal.
    937        swing_contact = select_swing_wheel_contact(
    938            support_pos_contact,
    939            support_neg_contact,
    940            gates["support_diagonal"],
    941        )
    942
    943        lift_pos = _yaw_lift_progress(env, lifted_asset_cfg, wheel_radius, target_clearance)
    944        lift_neg = _yaw_lift_progress(env, lifted_asset_cfg_mirror, wheel_radius, target_clearance)
    945        selected_lift = torch.where(gates["diag_pos"].unsqueeze(1), lift_pos, lift_neg)
    946        lift_progress = selected_lift.mean(dim=1)
    947
    948        asset: RigidObject = env.scene[asset_cfg.name]
    949        _fsm_transition_telemetry(
    950            env, gates, support_contacts, selected_lift, asset,
    951            env.command_manager.get_term(fsm_command_name),
    952        )
    953        yaw_command = env.command_manager.get_command(command_name)[:, 0]
    954        yaw_error = torch.abs(yaw_command - asset.data.root_ang_vel_b[:, 2])
    955        yaw_tracking = torch.exp(-yaw_error.square() / std**2)
    956
    957        # Metrics are intentionally YAW-only; otherwise transition/FOUR steps
    958        # dilute the curriculum score and make the scale ladder stall.
    959        yaw_mask = gates["b_yaw"].to(dtype=yaw_command.dtype)
    960        # Reuse the legacy yaw-curriculum accumulator names.  This keeps the
    961        # existing checkpoint/export path valid while changing only the sample
    962        # mask: FSM tracking statistics are normalized by YAW-state steps.
    963        _fsm_masked_accumulate(
    964            env,
    965            support_gate,
    966            yaw_mask,
    967            "_yaw_support_score_sum",
    968            "_yaw_support_score_samples",
    969        )
    970        _fsm_masked_accumulate(
    971            env,
    972            support_gate,
    973            yaw_mask,
    974            "_yaw_gate_open_sum",
    975            "_yaw_gate_open_samples",
    976        )
    977        _fsm_masked_accumulate_pair(
    978            env,
    979            torch.abs(yaw_command),
    980            yaw_error,
    981            yaw_mask,
    982            "_yaw_command_abs_sum",
    983            "_yaw_rate_abs_error_sum",
    984            "_yaw_tracking_metric_samples",
    985        )
    986
    987        # The legacy accumulators above deliberately remain direction-agnostic:
    988        # the original yaw curriculum and checkpoint export consume them.  The
    989        # FSM curriculum additionally needs independent evidence for each
    990        # diagonal; a strong POS branch must never promote a weak NEG branch.
    991        yaw_pos_mask = gates["b_yaw"] & gates["diag_pos"]
    992        yaw_neg_mask = gates["b_yaw"] & gates["diag_neg"]
    993        for suffix, contacts, direction_mask, wheel_names in (
    994            ("pos", support_pos_contact, yaw_pos_mask, ("FL", "HR")),
    995            ("neg", support_neg_contact, yaw_neg_mask, ("FR", "HL")),
    996        ):
    997            for index, wheel_name in enumerate(wheel_names):
    998                _fsm_masked_accumulate(
    999                    env,
    1000                    contacts[:, index].to(dtype=yaw_command.dtype),
    1001                    direction_mask.to(dtype=yaw_command.dtype),
    1002                    f"_yaw_fsm_{suffix}_support_{wheel_name}_sum",
    1003                    f"_yaw_fsm_{suffix}_support_{wheel_name}_samples",
    1004                )
    1005        for suffix, direction_mask in (("pos", yaw_pos_mask), ("neg", yaw_neg_mask)):
    1006            mask = direction_mask.to(dtype=yaw_command.dtype)
    1007            _fsm_masked_accumulate(
    1008                env,
    1009                lift_progress,
    1010                mask,
    1011                f"_yaw_fsm_{suffix}_lift_sum",
    1012                f"_yaw_fsm_{suffix}_yaw_samples",
    1013            )
    1014            _fsm_masked_accumulate(
    1015                env,
    1016                support_gate,
    1017                mask,
    1018                f"_yaw_fsm_{suffix}_support_sum",
    1019                f"_yaw_fsm_{suffix}_support_samples",
    1020            )
    1021            _fsm_masked_accumulate_pair(
    1022                env,
    1023                torch.abs(yaw_command),
    1024                yaw_error,
    1025                mask,
    1026                f"_yaw_fsm_{suffix}_command_abs_sum",
    1027                f"_yaw_fsm_{suffix}_yaw_abs_error_sum",
    1028                f"_yaw_fsm_{suffix}_tracking_samples",
    1029            )
    1030            _fsm_masked_accumulate(
    1031                env,
    1032                swing_contact.to(dtype=yaw_command.dtype),
    1033                mask,
    1034                f"_yaw_fsm_{suffix}_swing_contact_sum",
    1035                f"_yaw_fsm_{suffix}_swing_contact_samples",
    1036            )
    1037
    1038        _fsm_attempt_telemetry(
    1039            env, gates, lift_progress, support_gate, lift_progress_threshold, support_threshold
    1040        )
    1041        env._yaw_support_score_current = support_gate
    1042        env._yaw_gate_open_current = support_gate
    1043        env._yaw_command_abs_current = torch.abs(yaw_command)
    1044        env._yaw_rate_abs_error_current = yaw_error
    1045
    1046        command_term = env.command_manager.get_term(command_name)
    1047        yaw_limits = torch.as_tensor(
    1048            command_term.cfg.yaw_rate_range,
    1049            device=yaw_command.device,
    1050            dtype=yaw_command.dtype,
    1051        )
    1052        yaw_limit = yaw_limits.abs().amax()
    1053        edge_mask = (torch.abs(yaw_command) >= edge_command_fraction * yaw_limit) & gates["b_yaw"]
    1054        _fsm_masked_accumulate_pair(
    1055            env,
    1056            torch.abs(yaw_command),
    1057            yaw_error,
    1058            edge_mask.to(dtype=yaw_command.dtype),
    1059            "_yaw_edge_command_abs_sum",
    1060            "_yaw_edge_rate_abs_error_sum",
    1061            "_yaw_edge_tracking_samples",
    1062        )
    1063
    1064        # The phase-B floor provides a short exploration bridge at transition
    1065        # entry, then decays to zero so the policy cannot keep collecting reward
    1066        # without lifting.  A zero decay keeps the legacy/static-floor behavior.
    1067        if clearance_gate_floor_decay_s > 0.0:
    1068            floor = clearance_gate_floor * torch.clamp(
    1069                1.0 - gates["state_time"] / clearance_gate_floor_decay_s,
    1070                min=0.0,
    1071                max=1.0,
    1072            )
    1073        else:
    1074            floor = clearance_gate_floor
    1075        clearance_weight = floor + (1.0 - floor) * lift_progress
    1076        state_gate = gates["f_trans"] + gates["f_yaw"]
    1077        tracking = support_gate * clearance_weight * yaw_tracking * state_gate
    1078        # Contact alone gives no bonus while the swing pair is still on the ground.
    1079        support_bonus = 0.25 * support_shape * lift_progress * state_gate
    1080        # Center TRANSITION at its maximum: holding pose/tracking is never
    1081        # positive income. YAW retains the original tracking and support bonus.
    1082        reward = tracking + support_bonus - 1.25 * gates["f_trans"]
    1083        _fsm_record_positive_budget(env, gates, "fsm_gated_tracking", reward)
    1084        return reward
    1085
    1086
    1087    def four_stand_stability(
    1088        env: ManagerBasedRLEnv,
    1089        fsm_command_name: str,
    1090        target_height: float,
    1091        height_std: float = 0.08,
    1092        attitude_std: float = 0.25,
    1093        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    1094    ) -> torch.Tensor:
    1095        """Reward a stable four-wheel pose with a soft FSM phase gate.
    1096
    1097        SAFE_RECOVERY is intentionally absent from the gate: recovery gets only
    1098        its entry cost and baseline penalties, never a positive stability reward.
    1099        """
    1100        if height_std <= 0.0 or attitude_std <= 0.0:
    1101            raise ValueError("height_std and attitude_std must be positive.")
    1102
    1103        asset: Articulation = env.scene[asset_cfg.name]
    1104        base_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    1105        roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    1106        height_score = torch.exp(-(base_height - target_height).square() / height_std**2)
    1107        attitude_score = torch.exp(-(roll.square() + pitch.square()) / attitude_std**2)
    1108        gates = fsm_gates(env, fsm_command_name)
    1109        # Transition retains a decaying attitude deficit, without tracking the
    1110        # standing height. Height tracking remains active in FOUR_STAND/RETURN.
    1111        standing_gate = gates["f_four"] + gates["f_return"] * gates["tau"]
    1112        transition_gate = gates["f_trans"] * (1.0 - gates["tau"])
    1113        reward = (standing_gate * height_score + transition_gate) * attitude_score - transition_gate
    1114        _fsm_record_positive_budget(env, gates, "four_stand_stability", reward)
    1115        return reward
    1116
    1117
    1118    class ReturnToFourLanding(ManagerTermBase):
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py`** — SceneEntityCfg(instantiates), configclass(decorates), VQRWheelYawTerminationsCfg(class), TerminationsCfg(extends), VQRWheelFSMTerminationsCfg(class), VQRWheelYawTerminationsCfg(extends)
    
    ```python
    701        )
    702
    703
    704    @configclass
    705    class VQRWheelYawTerminationsCfg(TerminationsCfg):
    706        """Yaw-task failures; wheel contact is intentionally not terminal."""
    707
    708        torso_contact = DoneTerm(
    709            func=mdp.TorsoContactWithGrace,
    710            params={
    711                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["TORSO"]),
    712                "threshold": 1.0,
    713                "grace_period_s": 0.15,
    714            },
    715        )
    716
    717
    718    @configclass
    719    class VQRWheelFSMTerminationsCfg(VQRWheelYawTerminationsCfg):
    720        """Phase-aware unsafe handling with separate height and tilt failure logs."""
    721
    722        torso_contact = DoneTerm(
    723            func=mdp.FSMUnsafeWithGrace,
    724            params={
    725                "robot_name": "robot",
    726                "torso_sensor_cfg": SceneEntityCfg(
    727                    "contact_forces", body_names=["TORSO"], preserve_order=True
    728                ),
    729                "threshold": 1.0,
    730                "grace_period_s": 0.15,
    731                "minimum_base_height": MIN_BASE_HEIGHT,
    732                "unsafe_angle_limit": 0.80,
    733            },
    734        )
    735
    736        base_height_failure = DoneTerm(
    737            func=mdp.FSMUnsafeWithGrace,
    738            params={
    739                "robot_name": "robot",
    740                "torso_sensor_cfg": SceneEntityCfg(
    741                    "contact_forces", body_names=["TORSO"], preserve_order=True
    742                ),
    743                "threshold": 1.0,
    744                "grace_period_s": 0.15,
    745                "minimum_base_height": MIN_BASE_HEIGHT,
    746                "unsafe_angle_limit": 0.80,
    747                "failure_kind": "height",
    748            },
    749        )
    750        tilt_failure = DoneTerm(
    751            func=mdp.FSMUnsafeWithGrace,
    752            params={
    753                "robot_name": "robot",
    754                "torso_sensor_cfg": SceneEntityCfg(
    755                    "contact_forces", body_names=["TORSO"], preserve_order=True
    756                ),
    757                "threshold": 1.0,
    758                "grace_period_s": 0.15,
    759                "minimum_base_height": MIN_BASE_HEIGHT,
    760                "unsafe_angle_limit": 0.80,
    761                "failure_kind": "tilt",
    762            },
    763        )
    764
    765        fsm_transition_timeout = DoneTerm(
    766            func=mdp.fsm_transition_timeout,
    767            params={"command_name": "yaw_rate_cmd", "timeout_s": 3.0},
    768        )
    769        fsm_return_timeout = DoneTerm(
    770            func=mdp.fsm_return_timeout,
    771            params={"command_name": "yaw_rate_cmd", "timeout_s": 2.5},
    772        )
    773
    774        swing_contact_timeout = DoneTerm(
    775            func=mdp.SwingContactTimeout,
    776            params={
    777                "sensor_cfg": SceneEntityCfg(
    778                    "contact_forces", body_names=WHEEL_NAMES, preserve_order=True
    779                ),
    780                "command_name": "yaw_rate_cmd",
    781                "threshold": 1.0,
    782                "timeout_s": 0.20,
    783            },
    784        )
    785
    786
    787    @configclass
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/four_wheel_rotate_env_cfg.py`** — FourWheelRewardsCfg(class), FourWheelTerminationsCfg(class), FourWheelEventCfg(class), VQRFourWheelRotateEnvCfg(class)
    
    ```python
    1    # Copyright (c) 2026 Deep Robotics
    2    # SPDX-License-Identifier: BSD-3-Clause
    3
    4    """Minimal four-wheel in-place yaw-rate tracking task for VQRWheel."""
    5
    6    import math
    7
    8    from isaaclab.managers import EventTermCfg as EventTerm
    9    from isaaclab.managers import RewardTermCfg as RewTerm
    10    from isaaclab.managers import SceneEntityCfg
    11    from isaaclab.managers import TerminationTermCfg as DoneTerm
    12    from isaaclab.utils import configclass
    13
    14    import rl_training.tasks.manager_based.locomotion.pivot.mdp as mdp
    15
    16    from .balance_env_cfg import ActionsCfg, EventCfg
    17    from .robot_cfg import (
    18        ARTICULATION_JOINT_NAMES,
    19        DEFAULT_JOINT_POS,
    20        LEG_JOINT_NAMES,
    21        VQR_CFG,
    22        WHEEL_BODY_NAMES,
    23    )
    24    from .rotate_env_cfg import (
    25        YAW_COMMAND_NAME,
    26        RotateObservationsCfg,
    27        VQRTwoWheelRotateEnvCfg,
    28        YawRateCommandsCfg,
    29    )
    30
    31
    32    STANDING_JOINT_POSITIONS = [DEFAULT_JOINT_POS[name] for name in ARTICULATION_JOINT_NAMES]
    33    STANDING_LEG_POSITION_MAP = {name: DEFAULT_JOINT_POS[name] for name in LEG_JOINT_NAMES}
    34
    35
    36    @configclass
    37    class FourWheelActionsCfg(ActionsCfg):
    38        """Keep the existing 12-position + 4-velocity actions, centered on standing."""
    39
    40        leg_positions = ActionsCfg().leg_positions.replace(offset=STANDING_LEG_POSITION_MAP)
    41
    42
    43    @configclass
    44    class FourWheelEventCfg(EventCfg):
    45        """Reset directly into the existing four-wheel standing pose."""
    46
    47        reset_two_wheel = None
    48        reset_four_wheel = EventTerm(
    49            func=mdp.reset_four_wheel_standing,
    50            mode="reset",
    51            params={
    52                "asset_cfg": SceneEntityCfg(
    53                    "robot", joint_names=ARTICULATION_JOINT_NAMES, preserve_order=True
    54                ),
    55                "nominal_joint_positions": STANDING_JOINT_POSITIONS,
    56                "leg_joint_count": len(LEG_JOINT_NAMES),
    57                "root_height": VQR_CFG.init_state.pos[2],
    58                "joint_position_noise": (0.0, 0.0),
    59                "leg_velocity_noise": (0.0, 0.0),
    60                "wheel_velocity_noise": (0.0, 0.0),
    61                "roll_noise": (0.0, 0.0),
    62                "pitch_noise": (0.0, 0.0),
    63                "yaw_range": (0.0, 0.0),
    64                "angular_velocity_noise": (0.0, 0.0),
    65                "root_xy_noise": (0.0, 0.0),
    66            },
    67        )
    68
    69
    70    @configclass
    71    class FourWheelRewardsCfg:
    72        """Only the four reward terms needed for four-wheel rotation."""
    73
    74        track_ang_vel_z_exp = RewTerm(
    75            func=mdp.track_ang_vel_z_exp,
    76            weight=3.0,
    77            params={"command_name": YAW_COMMAND_NAME, "std": math.sqrt(0.5)},
    78        )
    79        track_lin_vel_xy_exp = RewTerm(
    80            func=mdp.track_lin_vel_xy_exp,
    81            weight=1.0,
    82            params={"command_name": YAW_COMMAND_NAME, "std": math.sqrt(0.5)},
    83        )
    84        flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    85        lateral_wheel_slip = RewTerm(
    86            func=mdp.lateral_wheel_slip,
    87            weight=-1.0,
    88            params={
    89                "sensor_cfg": SceneEntityCfg(
    90                    "contact_forces", body_names=WHEEL_BODY_NAMES, preserve_order=True
    91                ),
    92                "asset_cfg": SceneEntityCfg(
    93                    "robot", body_names=WHEEL_BODY_NAMES, preserve_order=True
    94                ),
    95                "threshold": 2.0,
    96            },
    97        )
    98
    99
    100    @configclass
    101    class FourWheelTerminationsCfg:
    102        """End an episode as soon as four-wheel support is lost."""
    103
    104        time_out = DoneTerm(func=mdp.time_out, time_out=True)
    105        lost_wheel_contact = DoneTerm(
    106            func=mdp.lost_wheel_contact,
    107            params={
    108                "sensor_cfg": SceneEntityCfg(
    109                    "contact_forces", body_names=WHEEL_BODY_NAMES, preserve_order=True
    110                ),
    111                "threshold": 2.0,
    112                "grace_period_s": 0.1,
    113            },
    114        )
    115
    116
    117    @configclass
    118    class VQRFourWheelRotateEnvCfg(VQRTwoWheelRotateEnvCfg):
    119        """Rotate in place while retaining contact on all four wheels."""
    120
    121        observations: RotateObservationsCfg = RotateObservationsCfg()
    122        actions: FourWheelActionsCfg = FourWheelActionsCfg()
    123        commands: YawRateCommandsCfg = YawRateCommandsCfg()
    124        rewards: FourWheelRewardsCfg = FourWheelRewardsCfg()
    125        terminations: FourWheelTerminationsCfg = FourWheelTerminationsCfg()
    126        events: FourWheelEventCfg = FourWheelEventCfg()
    127        curriculum = None
    ```
    
    
    > Some file sections were trimmed for size. For a specific symbol you still need, run another `codegraph_explore` (or `codegraph_node`) with its exact name — line-numbered source, cheaper and more complete than Read.
    ✓ • 264ms

## Assistant

Cấu hình hiện tại đúng là có nhiều reward âm trong TRANSITION, trong khi unsafe được phép kết thúc sau 0,15 s grace. Mình đang định lượng các nhánh reward theo từng trạng thái và kiểm tra xem đã có cost riêng cho unsafe termination chưa; điểm này quyết định cách sửa.

## Activity

    $ codegraph explore "yaw_transition_support_load yaw_com_support yaw_lift_clearance yaw_base_height_tracking yaw_balance in velocity/mdp/rewards.py"
    **Exploration: yaw_transition_support_load yaw_com_support yaw_lift_clearance yaw_base_height_tracking yaw_balance in velocity/mdp/rewards.py**
    
    Found 99 symbols across 1 file. 1 file pinned from the query.
    
    **Source Code**
    
    > The code below is the **verbatim, current on-disk source** of these files — re-read from disk on this call and line-numbered, byte-for-byte identical to what the Read tool returns. It is NOT a summary, outline, or stale cache. Treat each block as a Read you have already performed: do not Read a file shown here.
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py`** — fsm_gates(calls), _fsm_record_positive_budget(calls), _yaw_support_geometry(calls), get_gait_level_tensor(calls), _yaw_support_load_quality(calls), _yaw_lift_progress(calls), +32 more
    
    ```python
    103        """Penalize joint torques (curriculum-scaled by gait_level)."""
    104        asset: Articulation = env.scene[asset_cfg.name]
    105        reward = torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
    106        return reward * get_gait_level_tensor(env)
    107
    108
    109    def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    110        """Penalize action rate (curriculum-scaled by gait_level)."""
    111        reward = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    112        return reward * get_gait_level_tensor(env)
    113
    114    def action_smooth_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    115        """Penalize second-order action changes using an env-local action history."""
    
    ... (gap) ...
    
    125        reward = torch.sum(diff, dim=1)
    126
    127        setattr(env, cache_name, env.action_manager.prev_action.clone())
    128        return reward * get_gait_level_tensor(env)
    129
    130    def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    131        """Penalize contact force violations (curriculum-scaled by gait_level)."""
    132        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    133        net_contact_forces = contact_sensor.data.net_forces_w_history
    134        violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    135        reward = torch.sum(violation.clip(min=0.0), dim=1)
    136        return reward * get_gait_level_tensor(env)
    137
    138
    139    def track_ang_vel_z_exp(
    
    ... (gap) ...
    
    173        return torch.linalg.vector_norm(forces, dim=-1) > threshold
    174
    175
    176    def _yaw_support_load_quality(
    177        env: ManagerBasedRLEnv,
    178        sensor_cfg: SceneEntityCfg,
    179        target_force_n: float,
    180    ) -> torch.Tensor:
    181        """Score both wheels' upward normal load on the flat ground in [0, 1]."""
    182        if target_force_n <= 0.0:
    183            raise ValueError("target_force_n must be positive.")
    184        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    185        forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    186        if forces.ndim != 3 or forces.shape[1:] != (2, 3):
    187            raise ValueError("Support load requires exactly two wheel force vectors per environment.")
    188        per_wheel = torch.clamp(forces[..., 2] / target_force_n, min=0.0, max=1.0)
    189        # The weaker wheel dominates; one loaded wheel still provides a small
    190        # exploration signal toward recovering the other contact.
    191        return 0.8 * per_wheel.amin(dim=1) + 0.2 * per_wheel.mean(dim=1)
    192
    193
    194    def yaw_transition_support_load(
    195        env: ManagerBasedRLEnv,
    196        sensor_cfg: SceneEntityCfg,
    197        sensor_cfg_mirror: SceneEntityCfg,
    198        fsm_command_name: str,
    199        target_force_n: float,
    200    ) -> torch.Tensor:
    201        """Penalize missing support load in TRANSITION; a held pose earns no income."""
    202        gates = fsm_gates(env, fsm_command_name)
    203        pos = _yaw_support_load_quality(env, sensor_cfg, target_force_n)
    204        neg = _yaw_support_load_quality(env, sensor_cfg_mirror, target_force_n)
    205        reward = (torch.where(gates["diag_pos"], pos, neg) - 1.0) * gates["f_trans"]
    206        _fsm_record_positive_budget(env, gates, "transition_support_load", reward)
    207        return reward
    208
    209
    210    def _yaw_support_shape(contacts: torch.Tensor) -> torch.Tensor:
    
    ... (gap) ...
    
    222        return (asset.data.body_com_pos_w[..., :2] * masses.unsqueeze(-1)).sum(dim=1) / total_mass
    223
    224
    225    def _yaw_support_geometry(
    226        env: ManagerBasedRLEnv,
    227        asset_cfg: SceneEntityCfg,
    228    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    229        """Return CoM distance to the support line, projection coordinate, and segment length."""
    230        asset: Articulation = env.scene[asset_cfg.name]
    231        support_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    232        if support_xy.shape[1] != 2:
    233            raise ValueError("Yaw support rewards require exactly two ordered support wheel bodies.")
    234
    235        start = support_xy[:, 0]
    236        segment = support_xy[:, 1] - start
    237        length_sq = torch.sum(segment.square(), dim=-1).clamp_min(1.0e-8)
    238        length = torch.sqrt(length_sq)
    239        com_offset = _yaw_whole_body_com_xy(asset) - start
    240        projection = torch.sum(com_offset * segment, dim=-1) / length_sq
    241        perpendicular_distance = torch.abs(
    242            com_offset[:, 0] * segment[:, 1] - com_offset[:, 1] * segment[:, 0]
    243        ) / length
    244        return perpendicular_distance, projection, length
    245
    246
    247    def yaw_com_support(
    248        env: ManagerBasedRLEnv,
    249        asset_cfg: SceneEntityCfg,
    250        std: float,
    251        asset_cfg_mirror: SceneEntityCfg | None = None,
    252        fsm_command_name: str | None = None,
    253    ) -> torch.Tensor:
    254        """Reward CoM proximity with a non-flat reciprocal-L1 kernel."""
    255        if std <= 0.0:
    256            raise ValueError("std must be positive.")
    257        distance, _, _ = _yaw_support_geometry(env, asset_cfg)
    258        score = 1.0 / (1.0 + distance / std)
    259        if fsm_command_name is None:
    260            return score
    261        if asset_cfg_mirror is None:
    262            raise ValueError("asset_cfg_mirror is required when fsm_command_name is set.")
    263        mirror_distance, _, _ = _yaw_support_geometry(env, asset_cfg_mirror)
    264        mirror_score = 1.0 / (1.0 + mirror_distance / std)
    265        gates = fsm_gates(env, fsm_command_name)
    266        reward = torch.where(gates["diag_pos"], score, mirror_score) * gates["f_geom"] - gates["f_trans"]
    267        _fsm_record_positive_budget(env, gates, "com_support", reward)
    268        return reward
    269
    270
    271    def yaw_base_height_tracking(
    272        env: ManagerBasedRLEnv,
    273        target_height: float,
    274        error_scale: float,
    275        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    276        fsm_command_name: str | None = None,
    277    ) -> torch.Tensor:
    278        """Track TORSO height with non-saturating, normalized Huber shaping.
    279
    280        The raw score is ``1 - huber(abs(height - target) / error_scale)``.
    281        It is maximal at the finite target, quadratic nearby, and linear rather
    282        than exponentially flat when the torso is far from the target.
    283        """
    284        if error_scale <= 0.0:
    285            raise ValueError("error_scale must be positive.")
    286
    287        asset: Articulation = env.scene[asset_cfg.name]
    288        base_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    289        if not hasattr(env, "_yaw_base_height_min"):
    290            env._yaw_base_height_min = torch.full_like(base_height, torch.inf)
    291        env._yaw_base_height_min = torch.minimum(env._yaw_base_height_min, base_height)
    292
    293        normalized_error = torch.abs(base_height - target_height) / error_scale
    294        huber = torch.where(
    295            normalized_error <= 1.0,
    296            0.5 * normalized_error.square(),
    297            normalized_error - 0.5,
    298        )
    299        score = 1.0 - huber
    300        if fsm_command_name is None:
    301            return score
    302        gates = fsm_gates(env, fsm_command_name)
    303        reward = score * gates["f_yaw"]
    304        _fsm_record_positive_budget(env, gates, "base_height", reward)
    305        return reward
    306
    307
    308    def yaw_low_base_height_l1(
    
    ... (gap) ...
    
    319        return torch.relu(minimum_height - base_height) / error_scale
    320
    321
    322    def yaw_transition_low_base_height(
    323        env: ManagerBasedRLEnv,
    324        warning_height: float,
    325        minimum_height: float,
    326        fsm_command_name: str,
    327        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    328    ) -> torch.Tensor:
    329        """Soft collapse warning in TRANSITION, before the hard height boundary."""
    330        if warning_height <= minimum_height:
    331            raise ValueError("warning_height must exceed minimum_height.")
    332        asset: Articulation = env.scene[asset_cfg.name]
    333        base_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    334        proximity = ((warning_height - base_height) / (warning_height - minimum_height)).clamp(0.0, 1.0)
    335        return proximity.square() * fsm_gates(env, fsm_command_name)["f_trans"]
    336
    337
    338    def yaw_downward_low_base_velocity_l2(
    
    ... (gap) ...
    
    356            if warning_height is None or warning_height <= minimum_height:
    357                raise ValueError("warning_height must exceed minimum_height for FSM gating.")
    358            warning_gate = ((warning_height - base_height) / (warning_height - minimum_height)).clamp(0.0, 1.0)
    359            low_gate = torch.maximum(low_gate, warning_gate * fsm_gates(env, fsm_command_name)["f_trans"])
    360        downward_speed = torch.relu(-asset.data.root_lin_vel_w[:, 2])
    361        return low_gate * downward_speed.square()
    362
    
    ... (gap) ...
    
    376        if std <= 0.0:
    377            raise ValueError("std must be positive.")
    378
    379        _, _, span = _yaw_support_geometry(env, asset_cfg)
    380        outside_distance = torch.relu(minimum_span - span) + torch.relu(span - maximum_span)
    381        score = (outside_distance / std).square()
    382        if fsm_command_name is None:
    
    ... (gap) ...
    
    386        _, _, mirror_span = _yaw_support_geometry(env, asset_cfg_mirror)
    387        mirror_outside = torch.relu(minimum_span - mirror_span) + torch.relu(mirror_span - maximum_span)
    388        mirror_score = (mirror_outside / std).square()
    389        gates = fsm_gates(env, fsm_command_name)
    390        return torch.where(gates["diag_pos"], score, mirror_score) * gates["f_geom"]
    391
    392
    
    ... (gap) ...
    
    408        return _yaw_wheel_contacts(env, sensor_cfg, threshold).float().prod(dim=1)
    409
    410
    411    def _yaw_lift_progress(
    412        env: ManagerBasedRLEnv,
    413        asset_cfg: SceneEntityCfg,
    414        wheel_radius: float,
    415        target_clearance: float,
    416    ) -> torch.Tensor:
    417        """Return independent normalized clearance progress for the selected wheels."""
    418        if target_clearance <= 0.0:
    419            raise ValueError("target_clearance must be positive.")
    420
    421        asset: Articulation = env.scene[asset_cfg.name]
    422        wheel_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    423        ground_height = env.scene.env_origins[:, 2].unsqueeze(-1)
    424        clearance = wheel_height - ground_height - wheel_radius
    425        return torch.clamp(clearance / target_clearance, min=0.0, max=1.0)
    426
    427
    428    def yaw_lift_clearance(
    429        env: ManagerBasedRLEnv,
    430        asset_cfg: SceneEntityCfg,
    431        wheel_radius: float,
    432        target_clearance: float,
    433        asset_cfg_mirror: SceneEntityCfg | None = None,
    434        fsm_command_name: str | None = None,
    435        support_sensor_cfg: SceneEntityCfg | None = None,
    436        support_sensor_cfg_mirror: SceneEntityCfg | None = None,
    437        support_force_target_n: float = 80.0,
    438        transition_ungated_fraction: float = 0.25,
    439    ) -> torch.Tensor:
    440        """Shape lifted-wheel clearance with partial credit and a four-wheel penalty.
    441
    442        Each selected wheel contributes independently. The baseline score is in
    443        ``[-1, 1]``: both wheels on the ground score ``-1``, lifting either wheel
    444        improves the score, and both wheels must reach the target to score ``1``.
    445        In FSM TRANSITION, support load shapes the score before subtracting its
    446        maximum, leaving a nonpositive deficit. Raw metrics and YAW are unchanged.
    447        """
    448        progress = _yaw_lift_progress(env, asset_cfg, wheel_radius, target_clearance)
    449
    450        score = 2.0 * progress.mean(dim=1) - 1.0
    451        if fsm_command_name is None:
    452            # Keep a separate curriculum metric: the weaker wheel's progress,
    453            # averaged over the episode. This must not be reconstructed from the
    454            # signed mean reward because one fully lifted wheel can hide the other.
    455            min_progress = progress.amin(dim=1)
    456            if not hasattr(env, "_yaw_lift_min_progress_sum"):
    457                env._yaw_lift_min_progress_sum = torch.zeros_like(min_progress)
    458                env._yaw_lift_min_progress_samples = torch.zeros_like(min_progress, dtype=torch.long)
    459            env._yaw_lift_min_progress_sum += min_progress
    460            env._yaw_lift_min_progress_samples += 1
    461            return score
    462        if asset_cfg_mirror is None:
    463            raise ValueError("asset_cfg_mirror is required when fsm_command_name is set.")
    464        if support_sensor_cfg is None or support_sensor_cfg_mirror is None:
    465            raise ValueError("Both support sensor configs are required when fsm_command_name is set.")
    466        if not 0.0 < transition_ungated_fraction < 1.0:
    467            raise ValueError("transition_ungated_fraction must be in (0, 1).")
    468        mirror_progress = _yaw_lift_progress(
    469            env,
    470            asset_cfg_mirror,
    471            wheel_radius,
    472            target_clearance,
    473        )
    474        mirror_score = 2.0 * mirror_progress.mean(dim=1) - 1.0
    475        gates = fsm_gates(env, fsm_command_name)
    476        selected_progress = torch.where(gates["diag_pos"].unsqueeze(1), progress, mirror_progress)
    477        min_progress = selected_progress.amin(dim=1)
    478        if not hasattr(env, "_yaw_lift_min_progress_sum"):
    479            env._yaw_lift_min_progress_sum = torch.zeros_like(min_progress)
    480            env._yaw_lift_min_progress_samples = torch.zeros_like(min_progress, dtype=torch.long)
    481        env._yaw_lift_min_progress_sum += min_progress
    482        env._yaw_lift_min_progress_samples += 1
    483        selected_score = torch.where(gates["diag_pos"], score, mirror_score)
    484        support_pos = _yaw_support_load_quality(env, support_sensor_cfg, support_force_target_n)
    485        support_neg = _yaw_support_load_quality(env, support_sensor_cfg_mirror, support_force_target_n)
    486        support_quality = torch.where(gates["diag_pos"], support_pos, support_neg)
    487        lift_gate = transition_ungated_fraction + (1.0 - transition_ungated_fraction) * support_quality
    488        # Preserve the no-lift gradient before centering the transition score.
    489        # Only positive lift credit is reduced when support is weak.
    490        transition_score = torch.clamp(selected_score, max=0.0) + torch.relu(selected_score) * lift_gate
    491        reward = (transition_score - 1.0) * gates["f_trans"] + selected_score * gates["f_yaw"]
    492        _fsm_record_positive_budget(env, gates, "lift_clearance", reward)
    493        return reward
    494
    495
    496    def yaw_com_inside_support_segment(
    
    ... (gap) ...
    
    525        return reward
    526
    527
    528    def yaw_balance(
    529        env: ManagerBasedRLEnv,
    530        std: float,
    531        nominal_roll: float = 0.0,
    532        nominal_pitch: float = 0.0,
    533        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    534        fsm_command_name: str | None = None,
    535    ) -> torch.Tensor:
    536        """Reward roll and pitch near the configured diagonal-support equilibrium."""
    537        asset: Articulation = env.scene[asset_cfg.name]
    538        roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    539        roll_error = math_utils.wrap_to_pi(roll - nominal_roll)
    540        pitch_error = math_utils.wrap_to_pi(pitch - nominal_pitch)
    541        score = torch.exp(-(roll_error.square() + pitch_error.square()) / std**2)
    542        # ``None`` preserves the baseline task's exact calculation.  The FSM
    543        # task suppresses this otherwise-positive term in SAFE_RECOVERY.
    544        if fsm_command_name is None:
    545            return score
    546        gates = fsm_gates(env, fsm_command_name)
    547        reward = score * (1.0 - gates["f_safe"]) - gates["f_trans"]
    548        _fsm_record_positive_budget(env, gates, "balance", reward)
    549        return reward
    550
    551
    552    # def yaw_gated_tracking(
    ```
    
    **Not shown above — explore these names for their source**
    
    - source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm_gates.py: fsm_gates:51, _step_key:21, _required_tensor:37, _CACHE_ATTR:18, _CACHE_STEP_ATTR:17
    - tests/test_yaw_curriculum.py: get_term_cfg:159
    - tests/test_yaw_fsm_systemic.py: _reward_scene:19
    
    > Some file sections were trimmed for size. For a specific symbol you still need, run another `codegraph_explore` (or `codegraph_node`) with its exact name — line-numbered source, cheaper and more complete than Read.
    ✓ • 243ms

## Activity

    $ sed -n '495,705p' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py
                "fsm_command_name": "yaw_rate_cmd",
            },
        )
        rolling_slip = RewTerm(
            func=mdp.yaw_rolling_wheel_slip,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
                ),
                "sensor_cfg_mirror": SceneEntityCfg(
                    "contact_forces", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
                ),
                "body_asset_cfg": SceneEntityCfg(
                    "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
                ),
                "body_asset_cfg_mirror": SceneEntityCfg(
                    "robot", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
                ),
                "joint_asset_cfg": SceneEntityCfg(
                    "robot", joint_names=SUPPORT_WHEEL_NAMES, preserve_order=True
                ),
                "joint_asset_cfg_mirror": SceneEntityCfg(
                    "robot", joint_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
                ),
                "wheel_radius": WHEEL_RADIUS,
                "command_name": "yaw_rate_cmd",
                "yaw_reference": YAW_REF,
                "threshold": 1.0,
                "fsm_command_name": "yaw_rate_cmd",
            },
        )
        four_stand_stability = RewTerm(
            func=mdp.four_stand_stability,
            weight=3.0,
            params={
                "fsm_command_name": "yaw_rate_cmd",
                "target_height": TARGET_BASE_HEIGHT,
                "height_std": 0.08,
                "attitude_std": 0.25,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        return_to_four_landing = RewTerm(
            func=mdp.ReturnToFourLanding,
            weight=20.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
                ),
                "asset_cfg_mirror": SceneEntityCfg(
                    "robot", body_names=LIFTED_WHEEL_NAMES_MIRROR, preserve_order=True
                ),
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
                ),
                "sensor_cfg_mirror": SceneEntityCfg(
                    "contact_forces", body_names=LIFTED_WHEEL_NAMES_MIRROR, preserve_order=True
                ),
                "wheel_radius": WHEEL_RADIUS,
                "fsm_command_name": "yaw_rate_cmd",
                "contact_threshold": 1.0,
            },
        )
        four_stand_ready_bonus = RewTerm(
            func=mdp.four_stand_ready_bonus,
            weight=20.0,
            params={"fsm_command_name": "yaw_rate_cmd"},
        )
    
        # ------------------------------ Group 3: phase/curriculum terms ------------------------------
        fsm_gated_tracking = RewTerm(
            func=mdp.fsm_gated_tracking,
            weight=8.0,
            params={
                "command_name": "yaw_rate_cmd",
                "fsm_command_name": "yaw_rate_cmd",
                "support_sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
                ),
                "support_sensor_cfg_mirror": SceneEntityCfg(
                    "contact_forces", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
                ),
                "lifted_asset_cfg": SceneEntityCfg(
                    "robot", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
                ),
                "lifted_asset_cfg_mirror": SceneEntityCfg(
                    "robot", body_names=LIFTED_WHEEL_NAMES_MIRROR, preserve_order=True
                ),
                "wheel_radius": WHEEL_RADIUS,
                "target_clearance": LIFT_CLEARANCE_LEVELS[0],
                "std": 0.30,
                "contact_threshold": 1.0,
                "clearance_gate_floor": 0.0,
                "edge_command_fraction": 0.80,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        transition_progress = RewTerm(
            func=mdp.TransitionProgress,
            weight=2.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
                ),
                "asset_cfg_mirror": SceneEntityCfg(
                    "robot", body_names=LIFTED_WHEEL_NAMES_MIRROR, preserve_order=True
                ),
                "wheel_radius": WHEEL_RADIUS,
                "target_clearance": LIFT_CLEARANCE_LEVELS[0],
                "fsm_command_name": "yaw_rate_cmd",
                "gamma": 0.99,
                "support_sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
                ),
                "support_sensor_cfg_mirror": SceneEntityCfg(
                    "contact_forces", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
                ),
                "support_force_target_n": SUPPORT_LOAD_TARGET_N,
                "transition_ungated_fraction": TRANSITION_LIFT_UNGATED_FRACTION,
            },
        )
        spin_center_drift = RewTerm(
            func=mdp.spin_center_drift,
            weight=0.0,  # Phase A; curriculum raises this to -0.5/-2.0 in B/C.
            params={
                "fsm_command_name": "yaw_rate_cmd",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        safe_recovery_entry = RewTerm(
            func=mdp.safe_recovery_entry,
            weight=0.0,  # Termination semantics in phases A/B; -2.0 in phase C.
            params={"fsm_command_name": "yaw_rate_cmd"},
        )
    
    
    @configclass
    class VQRWheelYawCurriculumCfg:
        """Lift first, then jointly progress yaw command and online DR."""
    
        task_levels = CurrTerm(
            func=mdp.yaw_task_levels,
            params={
                "command_name": "yaw_rate_cmd",
                "clearance_levels": LIFT_CLEARANCE_LEVELS,
                "yaw_rate_levels": YAW_RATE_LEVELS,
                "dr_scale_levels": ONLINE_DR_SCALE_LEVELS,
                "tracking_ratio_thresholds": YAW_TRACKING_RATIO_THRESHOLDS,
                "edge_tracking_ratio_thresholds": YAW_EDGE_TRACKING_RATIO_THRESHOLDS,
                "lift_reward_name": "lift_clearance",
                "balance_reward_name": "balance",
                "yaw_reward_name": "gated_yaw_tracking",
                "torso_contact_termination_name": "torso_contact",
                "minimum_base_height": MIN_BASE_HEIGHT,
                "support_threshold": 0.85,
                "lift_progress_threshold": 0.80,
                "balance_threshold": 0.75,
                "yaw_threshold": 0.65,
                "min_evaluated_episodes": 2048,
                "required_success_rate": 0.85,
                "required_consecutive_windows": 3,
                "min_clearance_stage_steps": 1000,
                "min_yaw_stage_steps": 6000,
            },
        )
    
    
    @configclass
    class VQRWheelFSMCurriculumCfg(VQRWheelYawCurriculumCfg):
        """Three-phase FSM curriculum with independent POS/NEG certification."""
    
        task_levels = CurrTerm(
            func=mdp.yaw_fsm_task_levels,
            params={
                "command_name": "yaw_rate_cmd",
                "clearance_levels": LIFT_CLEARANCE_LEVELS,
                "yaw_rate_levels": YAW_RATE_LEVELS,
                "dr_scale_levels": ONLINE_DR_SCALE_LEVELS,
                "tracking_ratio_thresholds": YAW_TRACKING_RATIO_THRESHOLDS,
                "edge_tracking_ratio_thresholds": YAW_EDGE_TRACKING_RATIO_THRESHOLDS,
                "lift_reward_name": "lift_clearance",
                "balance_reward_name": "balance",
                "yaw_reward_name": "fsm_gated_tracking",
                "transition_reward_name": "transition_progress",
                "spin_center_drift_reward_name": "spin_center_drift",
                "safe_recovery_reward_name": "safe_recovery_entry",
                "torso_contact_termination_name": "torso_contact",
                "minimum_base_height": MIN_BASE_HEIGHT,
                "support_threshold": 0.85,
                "lift_progress_threshold": 0.80,
                "balance_threshold": 0.75,
                "yaw_threshold": 0.65,
                "min_evaluated_episodes": 2048,
                "required_success_rate": 0.85,
                "required_consecutive_windows": 3,
                "min_clearance_stage_steps": 1000,
                "min_yaw_stage_steps": 6000,
                # §7: both directions need their own evidence; one good diagonal
                # must not pull the other across a curriculum boundary.
                "min_directional_episodes": 1024,
                "transition_success_threshold": 0.85,
                "drift_thresholds": FSM_DRIFT_THRESHOLDS,
                # 150 PPO iterations at the standard 24-step rollout.
                "reward_ramp_steps": 3600,
            },
        )
    
    
    @configclass
    class VQRWheelYawTerminationsCfg(TerminationsCfg):
    ✓ • 0ms

## Activity

    $ rg -n 'failure|termination|fail|early|deficit|transition|budget' tests/test_yaw_fsm_systemic.py tests/test_yaw_transition_safety.py | head -180
    tests/test_yaw_transition_safety.py:1:"""CPU checks for the transition height warning and support-gated lift credit."""
    tests/test_yaw_transition_safety.py:12:TERMINATIONS_PATH = REWARDS_PATH.with_name("terminations.py")
    tests/test_yaw_transition_safety.py:16:def _transition_scene():
    tests/test_yaw_transition_safety.py:18:    names = {"yaw_transition_low_base_height", "yaw_downward_low_base_velocity_l2"}
    tests/test_yaw_transition_safety.py:28:def test_transition_height_warning_is_monotone_and_state_gated():
    tests/test_yaw_transition_safety.py:29:    rewards, state, env, command = _transition_scene()
    tests/test_yaw_transition_safety.py:35:        samples.append(rewards["yaw_transition_low_base_height"](env, **args))
    tests/test_yaw_transition_safety.py:42:    assert torch.equal(rewards["yaw_transition_low_base_height"](env, **args), torch.zeros(2))
    tests/test_yaw_transition_safety.py:45:def test_transition_stability_does_not_track_standing_height():
    tests/test_yaw_transition_safety.py:46:    rewards, _, env, _ = _transition_scene()
    tests/test_yaw_transition_safety.py:60:def test_downward_speed_is_penalized_before_hard_height_failure():
    tests/test_yaw_transition_safety.py:61:    rewards, state, env, _ = _transition_scene()
    tests/test_yaw_transition_safety.py:81:    rewards, _, env, _ = _transition_scene()
    tests/test_yaw_transition_safety.py:95:    clearance = rewards["yaw_lift_clearance"](env, **args, transition_ungated_fraction=.25)
    tests/test_yaw_transition_safety.py:99:    credit = progress_term(env, **args, transition_ungated_fraction=.25)
    tests/test_yaw_transition_safety.py:107:    full_credit = progress_term(env, **args, transition_ungated_fraction=.25)
    tests/test_yaw_transition_safety.py:111:def test_height_and_tilt_failure_are_separate_termination_masks():
    tests/test_yaw_transition_safety.py:130:    assert torch.equal(term(env, **args, failure_kind="height"), masks[1])
    tests/test_yaw_transition_safety.py:131:    assert torch.equal(term(env, **args, failure_kind="tilt"), masks[2])
    tests/test_yaw_fsm_systemic.py:66:        3 * rewards["yaw_transition_support_load"](env, target_force_n=80, **support, **fsm),
    tests/test_yaw_fsm_systemic.py:79:def test_stalled_transition_has_no_positive_pose_income_and_handoff_does_not_drop(clearance):
    tests/test_yaw_fsm_systemic.py:86:        transition = _pose_rewards(rewards, env)
    tests/test_yaw_fsm_systemic.py:87:        assert (transition <= 0).all()
    tests/test_yaw_fsm_systemic.py:88:        assert torch.allclose(transition[:, 0], transition[:, 1])
    tests/test_yaw_fsm_systemic.py:93:    assert (yaw.sum(0) >= transition.sum(0)).all()
    tests/test_yaw_fsm_systemic.py:157:    _load_nodes(FSM_PATH.with_name("terminations.py"), {"fsm_transition_timeout"}, namespace)
    tests/test_yaw_fsm_systemic.py:165:        if namespace["fsm_transition_timeout"](env).all():
    tests/test_yaw_fsm_systemic.py:168:        pytest.fail("Chatter escaped the cumulative transition deadline")
    tests/test_yaw_fsm_systemic.py:170:    assert torch.all(fsm.transition_time >= 3.)
    tests/test_yaw_fsm_systemic.py:171:    assert torch.allclose(fsm.transition_time[0], fsm.transition_time[1])
    tests/test_yaw_fsm_systemic.py:175:    budget = fsm.transition_time.clone()
    tests/test_yaw_fsm_systemic.py:178:        assert not namespace["fsm_transition_timeout"](env).any()
    tests/test_yaw_fsm_systemic.py:179:    assert torch.equal(fsm.transition_time, budget)
    tests/test_yaw_fsm_systemic.py:183:    assert not fsm.transition_time.any()
    tests/test_yaw_fsm_systemic.py:190:    def record(states, terminal=False, failed=False, quality=1.):
    tests/test_yaw_fsm_systemic.py:193:        env.reset_terminated = torch.full((env.num_envs,), failed)
    tests/test_yaw_fsm_systemic.py:202:def test_curriculum_counts_failed_attempts_even_with_no_yaw_or_a_prior_success(monkeypatch, phase):
    tests/test_yaw_fsm_systemic.py:218:    record([1, 3, 0], terminal=True, failed=True)
    tests/test_yaw_fsm_systemic.py:231:    assert not env._yaw_fsm_pos_transition_attempted.any()
    tests/test_yaw_fsm_systemic.py:246:    record([2, 4, 0], terminal=True)  # Healthy episode truncation is not a failure.
    tests/test_yaw_fsm_systemic.py:265:    record([1, 3, 0], terminal=True, failed=True)
    tests/test_yaw_fsm_systemic.py:266:    assert env._yaw_fsm_pos_transition_attempted.tolist() == [1, 0, 0]
    tests/test_yaw_fsm_systemic.py:267:    assert env._yaw_fsm_neg_transition_attempted.tolist() == [0, 1, 0]
    tests/test_yaw_fsm_systemic.py:268:    assert not env._yaw_fsm_pos_transition_succeeded.any()
    tests/test_yaw_fsm_systemic.py:272:def test_safe_recovery_fails_attempt_without_requiring_episode_termination(monkeypatch):
    tests/test_yaw_fsm_systemic.py:278:    assert env._yaw_fsm_pos_transition_attempted.sum() == 1
    tests/test_yaw_fsm_systemic.py:279:    assert env._yaw_fsm_neg_transition_attempted.sum() == 1
    tests/test_yaw_fsm_systemic.py:280:    assert not env._yaw_fsm_pos_transition_succeeded.any()
    ✓ • 0ms

## Activity

    $ rg -n 'unsafe|transition_duration|0.184|fail early|reward farming|termination' docs -g '*.md' | head -120
    docs/MILESTONES.md:10:| **M7 – Balance MDP**                  | Local          | Implement reward + termination tối thiểu                     | Có support/lift/balance/drift/action-rate/torque terms; không reward thừa      |
    docs/document_yaw_fsm/fsm.md:37:MỌI STATE ── unsafe ──> SAFE_RECOVERY
    docs/document_yaw_fsm/fsm.md:42:`unsafe` luôn bỏ qua dwell. Recovery cần `safe` và `four_stand_ready` liên tục
    docs/document_yaw_fsm/fsm.md:43:`0.50 s`. Ở curriculum phase A/B, unsafe terminate sau reset grace `0.15 s`;
    docs/document_yaw_fsm/fsm.md:45:liên tục quá `0.20 s` trong YAW vẫn là termination độc lập.
    docs/response.md:22:        │                            four_stand_ready / unsafe   (N,)
    docs/response.md:28:                ──►  terminations (transition timeout, tùy chọn)
    docs/response.md:42:- `unsafe` = torso contact grace ∧/∨ base_height < MIN_BASE_HEIGHT ∧/∨ |roll,pitch| lớn
    docs/response.md:62:- **Critic**: 83D + one-hot 7 + 4 ready/unsafe flags + `state_time` (1) → 95D.
    docs/response.md:81:| `mdp/terminations.py`                        | + `fsm_transition_timeout`                                                                        |
    docs/document_yaw_fsm/AGENTS.md:63:SAFE_RECOVERY ← unsafe, từ MỌI state
    docs/document_yaw_fsm/AGENTS.md:79:unsafe              = torso_contact ∨ base_height < MIN ∨ |roll,pitch| > θ_max
    docs/document_yaw_fsm/AGENTS.md:256:- Critic input mới: 83 + one-hot 7 + 4 ready/unsafe flag + `state_time` = **95**.
    docs/document_yaw_fsm/AGENTS.md:297:| `mdp/terminations.py` | + `fsm_transition_timeout` |
    docs/document_yaw_fsm/AGENTS.md:369:2. **Policy "gian lận" không nhấc bánh** (giữ 4 bánh vẫn quay được phần nào, an toàn hơn) → `lifted_wheel_spin` đủ nặng + termination khi swing chạm đất kéo dài + đưa `m_sup` vào obs.
    docs/document_yaw_fsm/fsm_review_2026-09-24.md:14:- Theo Isaac Lab đang dùng, một `env.step()` tính **termination → reward → reset các env hoàn tất → command/FSM → observation**. Vì thế reward và các watchdog đọc trạng thái FSM ở cuối step trước, còn observation trả về dùng trạng thái vừa cập nhật. Termination unsafe đọc sensor hiện tại riêng. Ở phase C, khi unsafe mới xuất hiện, reward của step đó vẫn có thể đọc state cũ; từ step kế tiếp SAFE gate mới có hiệu lực. Đây là độ trễ theo thứ tự framework, cần dùng cùng một mốc state khi đối chiếu log. Nguồn framework: `/home/robotics/datld5/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:204-238`.
    docs/document_yaw_fsm/fsm_review_2026-09-24.md:18:| State | Giá trị | Đường ra khi không unsafe |
    docs/document_yaw_fsm/fsm_review_2026-09-24.md:26:| `SAFE_RECOVERY` | 6 | Hết unsafe và `four_stand_ready` liên tục 0,50 s → `FOUR_STAND` |
    docs/document_yaw_fsm/fsm_review_2026-09-24.md:32:## 4. Reward, observation, termination và curriculum
    docs/document_yaw_fsm/fsm_review_2026-09-24.md:36:| Observation | Actor thêm one-hot 7 state (tổng dự kiến 62 chiều). Critic thêm one-hot, bốn ready/unsafe flag, `state_time` và hai hình học support được chọn theo diagonal (tổng dự kiến 95 chiều). Noise của nhóm kế thừa không đổi. [Cấu hình](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py#L906) |
    docs/document_yaw_fsm/fsm_review_2026-09-24.md:38:| Termination | Unsafe sau grace tính từ reset 0,15 s ở phase A/B; phase C đưa vào SAFE thay vì terminate. TRANSITION liên tục ≥3,0 s terminate. Một bánh swing contact liên tục ≥0,20 s **khi ở YAW** terminate. [Mã](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py#L62) |
    docs/document_yaw_fsm/fsm_review_2026-09-24.md:47:Nếu bánh swing chạm nền, clearance của nó thường thấp hơn ngưỡng pose ready. FSM rời `YAW_*` về `TRANSITION_*` sau 0,10 s mất pose liên tục; timer của `SwingContactTimeout` chỉ chạy ở YAW và bị xóa ngay khi rời YAW. Ngưỡng termination là 0,20 s. Vì vậy một chuỗi contact trên nền phẳng có thể liên tục gây mất pose nhưng **không bao giờ đủ 0,20 s trong YAW** để kích hoạt watchdog. Đây là suy luận từ [pose predicate](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/observations.py#L230), [pose-loss grace](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py#L393), [timer](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py#L159); contact do va vào vật cao có thể khác. Nên kiểm tra watchdog bằng trajectory contact thật và quyết định tính thời gian qua cả `TRANSITION_*` hoặc đổi ngưỡng/điều kiện cho phù hợp.
    docs/document_yaw_fsm/fsm_review_2026-09-24.md:51:Hai FSM lưu tham số nhưng không dùng nó trong predicate thoát. Một lệnh đi qua ngưỡng exit ngay sau khi vào YAW đưa state vào `RETURN_TO_4` ở lần cập nhật kế tiếp. [Mã](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py#L84), [test xác nhận thoát ngay](../../tests/test_yaw_fsm.py#L147). [Tài liệu FSM hiện có](fsm.md) lại nói chỉ thoát do command sau dwell 0,20 s. Cần chốt một semantics rồi đồng bộ mã, test và tài liệu; unsafe vẫn phải có ưu tiên tức thì.
    docs/document_yaw_fsm/fsm_review_2026-09-24.md:59:Reward ghi hai boolean `transition_attempted` và `transition_succeeded` cho **mỗi episode và mỗi hướng**. Curriculum đánh giá `success=transitioned` khi `phase==1`; một attempt abort rồi một attempt sau thành công trong cùng episode vẫn được tính một lần thành công. Vì thế metric không chứng minh mọi lần chuyển trạng thái đạt YAW trong 3 s. Watchdog 3 s chỉ bắt chuỗi TRANSITION liên tục: [ghi cờ](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py#L847), [đánh giá](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py#L844), [watchdog](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py#L173). Nếu tiêu chí nghiệm thu là tỉ lệ thành công **mỗi attempt**, cần đếm attempt và thời gian tương ứng.
    docs/document_yaw_fsm/fsm_review_2026-09-24.md:63:`RETURN_TO_4` chỉ thoát khi cả bốn bánh contact và roll/pitch đủ nhỏ; watchdog 3 s chỉ áp dụng cho `TRANSITION_*`, còn watchdog swing chỉ cho `YAW_*`. Nếu điều kiện bốn bánh không xảy ra, env có thể ở RETURN đến episode timeout 20 s: [cạnh RETURN](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py#L445), [watchdog](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py#L173). Nên log thời gian RETURN và tỉ lệ episode timeout trong RETURN trước khi quyết định thêm timeout hoặc recovery path.
    docs/document_yaw_fsm/training_status_2026-09-24_11-23-35.md:23:| Watchdog | TRANSITION quá 3.0 s thì terminate; một swing wheel contact liên tục ≥ 0.20 s trong YAW thì terminate; unsafe (torso contact, base thấp hoặc nghiêng quá `0.80 rad`) terminate sau grace `0.15 s` ở phase A/B |
    docs/document_yaw_fsm/training_status_2026-09-24_11-23-35.md:25:FSM có 7 trạng thái: `0 FOUR_STAND`, `1 TRANSITION_POS`, `2 YAW_POS`, `3 TRANSITION_NEG`, `4 YAW_NEG`, `5 RETURN_TO_4`, `6 SAFE_RECOVERY`. Phase A/B dùng termination cho unsafe; phase C dùng SAFE_RECOVERY với dwell tối thiểu `0.5 s`. Run hiện ở phase A.
    docs/document_yaw_fsm/training_status_2026-09-24_11-23-35.md:73:| `Episode_Termination/torso_contact` | 4.00% | Đây là tên term unsafe FSM, gồm torso contact **hoặc** base thấp/nghiêng quá ngưỡng; không được hiểu là chỉ torso va đất |
    docs/document_yaw_fsm/training_status_2026-09-24_11-23-35.md:120:| **B: transition** | Mỗi hướng cần ≥1024 episode đã thử transition/cửa sổ; tỉ lệ vào YAW ≥0.85 ở cả hai hướng trong 3 cửa sổ, giữ mức ≥6000 bước. | Clearance `0.20 m`; tracking floor `0.25` ở đầu transition rồi về 0 trong 3 s; drift weight ramp tới `−0.5`; unsafe vẫn terminate. |
    ✓ • 0ms

## Activity

    $ codegraph explore "TransitionProgress safe_recovery_entry FSMUnsafeWithGrace in velocity/mdp"
    **Exploration: TransitionProgress safe_recovery_entry FSMUnsafeWithGrace in velocity/mdp**
    
    Found 60 symbols across 5 files.
    
    **Blast radius — what depends on these (update/verify before editing)**
    
    - `transition_levels` (source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/mdp/to_transition.py:170) — 1 caller; tests: `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/tests/smoke_transition.py`
    
    **Source Code**
    
    > The code below is the **verbatim, current on-disk source** of these files — re-read from disk on this call and line-numbered, byte-for-byte identical to what the Read tool returns. It is NOT a summary, outline, or stale cache. Treat each block as a Read you have already performed: do not Read a file shown here.
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py`** — calls(calls), fsm_gates(calls), _yaw_lift_progress(calls), _fsm_record_positive_budget(calls), ManagerTermBase(extends), __call__(method), +12 more
    
    ```python
    850        leaves the diagnostic unset and never changes the reward value.
    851        """
    852        try:
    853            weight = float(env.reward_manager.get_term_cfg(term_name).weight)
    854        except (AttributeError, KeyError, TypeError):
    855            return
    856        positive = torch.relu(value * weight)
    
    ... (gap) ...
    
    902        if not 0.0 < edge_command_fraction <= 1.0:
    903            raise ValueError("edge_command_fraction must be in (0, 1].")
    904
    905        gates = fsm_gates(env, fsm_command_name)
    906        _fsm_step_telemetry(env, gates)
    907        support_pos_contact = _yaw_wheel_contacts(env, support_sensor_cfg, contact_threshold)
    908        support_neg_contact = _yaw_wheel_contacts(env, support_sensor_cfg_mirror, contact_threshold)
    909        support_pos = support_pos_contact.to(dtype=torch.float32).prod(dim=1)
    
    ... (gap) ...
    
    1084        return reward
    1085
    1086
    1087    def four_stand_stability(
    1088        env: ManagerBasedRLEnv,
    1089        fsm_command_name: str,
    1090        target_height: float,
    1091        height_std: float = 0.08,
    1092        attitude_std: float = 0.25,
    1093        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    1094    ) -> torch.Tensor:
    1095        """Reward a stable four-wheel pose with a soft FSM phase gate.
    1096
    1097        SAFE_RECOVERY is intentionally absent from the gate: recovery gets only
    1098        its entry cost and baseline penalties, never a positive stability reward.
    1099        """
    1100        if height_std <= 0.0 or attitude_std <= 0.0:
    1101            raise ValueError("height_std and attitude_std must be positive.")
    1102
    1103        asset: Articulation = env.scene[asset_cfg.name]
    1104        base_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    1105        roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    1106        height_score = torch.exp(-(base_height - target_height).square() / height_std**2)
    1107        attitude_score = torch.exp(-(roll.square() + pitch.square()) / attitude_std**2)
    1108        gates = fsm_gates(env, fsm_command_name)
    1109        # Transition retains a decaying attitude deficit, without tracking the
    1110        # standing height. Height tracking remains active in FOUR_STAND/RETURN.
    1111        standing_gate = gates["f_four"] + gates["f_return"] * gates["tau"]
    1112        transition_gate = gates["f_trans"] * (1.0 - gates["tau"])
    1113        reward = (standing_gate * height_score + transition_gate) * attitude_score - transition_gate
    1114        _fsm_record_positive_budget(env, gates, "four_stand_stability", reward)
    1115        return reward
    1116
    1117
    1118    class ReturnToFourLanding(ManagerTermBase):
    
    ... (gap) ...
    
    1130            self.contact_seen[index] = False
    1131            self.was_return[index] = False
    1132
    1133        def __call__(
    1134            self,
    1135            env: ManagerBasedRLEnv,
    1136            asset_cfg: SceneEntityCfg,
    1137            asset_cfg_mirror: SceneEntityCfg,
    1138            sensor_cfg: SceneEntityCfg,
    1139            sensor_cfg_mirror: SceneEntityCfg,
    1140            wheel_radius: float,
    1141            fsm_command_name: str,
    1142            contact_threshold: float = 1.0,
    1143        ) -> torch.Tensor:
    1144            gates = fsm_gates(env, fsm_command_name)
    1145            target_clearance = env.command_manager.get_term(fsm_command_name).cfg.target_clearance
    1146            progress_pos = _yaw_lift_progress(env, asset_cfg, wheel_radius, target_clearance)
    1147            progress_neg = _yaw_lift_progress(env, asset_cfg_mirror, wheel_radius, target_clearance)
    1148            contact_pos = _yaw_wheel_contacts(env, sensor_cfg, contact_threshold)
    1149            contact_neg = _yaw_wheel_contacts(env, sensor_cfg_mirror, contact_threshold)
    1150            progress = torch.where(gates["diag_pos"].unsqueeze(1), progress_pos, progress_neg)
    1151            contact = torch.where(gates["diag_pos"].unsqueeze(1), contact_pos, contact_neg)
    1152            in_return = gates["b_return"]
    1153            continuing = in_return & self.was_return & ~gates["just_switched"]
    1154            lowering = (self.prev_progress - progress).mean(dim=1) * continuing
    1155            # Contacts present at return entry were never lost and earn no bonus.
    1156            first_contacts = (contact & ~self.contact_seen & continuing.unsqueeze(1)).float().mean(dim=1)
    1157            reward = lowering + 2.0 * first_contacts
    1158            self.prev_progress.copy_(torch.where(in_return.unsqueeze(1), progress, torch.zeros_like(progress)))
    1159            self.contact_seen.copy_(torch.where(
    1160                in_return.unsqueeze(1), self.contact_seen | contact, torch.zeros_like(contact)
    1161            ))
    1162            self.was_return.copy_(in_return)
    1163            _fsm_record_positive_budget(env, gates, "return_to_four_landing", reward)
    1164            return reward
    1165
    1166
    1167    def four_stand_ready_bonus(env: ManagerBasedRLEnv, fsm_command_name: str) -> torch.Tensor:
    1168        """Pay once when RETURN_TO_4 exits through four-wheel readiness."""
    1169        gates = fsm_gates(env, fsm_command_name)
    1170        reward = gates["b_return_complete"].to(dtype=torch.float32)
    1171        _fsm_record_positive_budget(env, gates, "four_stand_ready_bonus", reward)
    1172        return reward
    1173
    1174
    1175    class TransitionProgress(ManagerTermBase):
    1176        """Bounded new lift credit in TRANSITION; potential-based shaping in RETURN."""
    1177
    1178        def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
    1179            super().__init__(cfg, env)
    1180            self.prev_phi = torch.zeros(env.num_envs, device=env.device)
    1181            self.best_transition_progress = torch.zeros(env.num_envs, device=env.device)
    1182
    1183        def reset(self, env_ids=None) -> None:
    1184            if env_ids is None:
    1185                self.prev_phi.zero_()
    1186                self.best_transition_progress.zero_()
    1187            else:
    1188                self.prev_phi[env_ids] = 0.0
    1189                self.best_transition_progress[env_ids] = 0.0
    1190
    1191        def __call__(
    1192            self,
    1193            env: ManagerBasedRLEnv,
    1194            asset_cfg: SceneEntityCfg,
    1195            asset_cfg_mirror: SceneEntityCfg,
    1196            wheel_radius: float,
    1197            target_clearance: float,
    1198            fsm_command_name: str,
    1199            gamma: float = 0.99,
    1200            support_sensor_cfg: SceneEntityCfg | None = None,
    1201            support_sensor_cfg_mirror: SceneEntityCfg | None = None,
    1202            support_force_target_n: float = 80.0,
    1203            transition_ungated_fraction: float = 0.25,
    1204        ) -> torch.Tensor:
    1205            if not 0.0 < gamma <= 1.0:
    1206                raise ValueError("gamma must be in (0, 1].")
    1207            if support_sensor_cfg is None or support_sensor_cfg_mirror is None:
    1208                raise ValueError("Both support sensor configs are required for transition progress.")
    1209            if not 0.0 < transition_ungated_fraction < 1.0:
    1210                raise ValueError("transition_ungated_fraction must be in (0, 1).")
    1211
    1212            gates = fsm_gates(env, fsm_command_name)
    1213            progress_pos = _yaw_lift_progress(
    1214                env, asset_cfg, wheel_radius, target_clearance
    1215            ).mean(dim=1)
    1216            progress_neg = _yaw_lift_progress(
    1217                env, asset_cfg_mirror, wheel_radius, target_clearance
    1218            ).mean(dim=1)
    1219            progress = torch.where(gates["diag_pos"], progress_pos, progress_neg)
    1220            phi = torch.where(gates["b_return"], 1.0 - progress, progress)
    1221
    1222            # Always advance the potential, including FOUR/YAW/SAFE states.  This
    1223            # prevents a spurious pulse when the next gated phase begins.
    1224            reward = gamma * phi - self.prev_phi
    1225            self.prev_phi.copy_(phi)
    1226            reward = torch.where(gates["just_switched"], torch.zeros_like(reward), reward)
    1227            # Credit a new lift maximum only once across TRANSITION/YAW chatter.
    1228            # YAW/RETURN keep their previous reward behavior.
    1229            active = gates["b_trans"] | gates["b_yaw"]
    1230            self.best_transition_progress.masked_fill_(~active, 0.0)
    1231            improvement = torch.relu(progress - self.best_transition_progress)
    1232            transition_reward = improvement - (1.0 - gamma) * progress
    1233            support_pos = _yaw_support_load_quality(env, support_sensor_cfg, support_force_target_n)
    1234            support_neg = _yaw_support_load_quality(env, support_sensor_cfg_mirror, support_force_target_n)
    1235            support_quality = torch.where(gates["diag_pos"], support_pos, support_neg)
    1236            lift_gate = transition_ungated_fraction + (1.0 - transition_ungated_fraction) * support_quality
    1237            transition_reward = torch.relu(transition_reward) * lift_gate + torch.clamp(transition_reward, max=0.0)
    1238            self.best_transition_progress.copy_(torch.where(
    1239                active, torch.maximum(self.best_transition_progress, progress),
    1240                self.best_transition_progress,
    1241            ))
    1242            reward = transition_reward * gates["f_trans"] + reward * gates["f_return"]
    1243            _fsm_record_positive_budget(env, gates, "transition_progress", reward)
    1244            return reward
    1245
    1246
    1247    def spin_center_drift(
    
    ... (gap) ...
    
    1282        return drift * gates["f_yaw"]
    1283
    1284
    1285    def safe_recovery_entry(
    1286        env: ManagerBasedRLEnv,
    1287        fsm_command_name: str,
    1288    ) -> torch.Tensor:
    1289        """Return one event pulse on entry to SAFE_RECOVERY."""
    1290        gates = fsm_gates(env, fsm_command_name)
    1291        return (gates["b_safe"] & gates["just_switched"]).to(dtype=torch.float32)
    1292
    1293
    1294    def _yaw_command_penalty_scale(
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py`** — ManagerTermBase(extends), TorsoContactWithGrace(class), FSMUnsafeWithGrace(class), __init__(method), __init__(calls), reset(method), +3 more
    
    ```python
    59            return terminated
    60
    61
    62    class FSMUnsafeWithGrace(ManagerTermBase):
    63        """Phase-aware unsafe termination using the live FSM safety predicate.
    64
    65        Phases A/B terminate after the short reset grace.  Phase C intentionally
    66        never terminates for this predicate: the command FSM receives the same
    67        unsafe signal and enters ``SAFE_RECOVERY`` instead.  This class never
    68        reads ``command.unsafe`` because that buffer can be one simulation step
    69        old when the termination manager runs.
    70        """
    71
    72        def __init__(self, cfg, env: ManagerBasedRLEnv):
    73            super().__init__(cfg, env)
    74            self._elapsed_s = torch.zeros(env.num_envs, device=env.device)
    75
    76        def reset(self, env_ids: Sequence[int] | None = None) -> None:
    77            if env_ids is None:
    78                self._elapsed_s.zero_()
    79            else:
    80                self._elapsed_s[env_ids] = 0.0
    81
    82        def __call__(
    83            self,
    84            env: ManagerBasedRLEnv,
    85            robot_name: str,
    86            torso_sensor_cfg: SceneEntityCfg,
    87            threshold: float,
    88            grace_period_s: float,
    89            minimum_base_height: float,
    90            unsafe_angle_limit: float,
    91            failure_kind: str = "all",
    92        ) -> torch.Tensor:
    93            if grace_period_s < 0.0:
    94                raise ValueError("grace_period_s must be non-negative.")
    95            torso, height, tilt = yaw_fsm_unsafe_components(
    96                env,
    97                robot_name=robot_name,
    98                torso_sensor_cfg=torso_sensor_cfg,
    99                minimum_base_height=minimum_base_height,
    100                unsafe_angle_limit=unsafe_angle_limit,
    101                contact_threshold=threshold,
    102            )
    103            failures = {"all": torso | height | tilt, "height": height, "tilt": tilt}
    104            if failure_kind not in failures:
    105                raise ValueError("failure_kind must be 'all', 'height', or 'tilt'.")
    106            unsafe = failures[failure_kind]
    107            phase_value = getattr(env, "_yaw_fsm_task_curriculum_phase", 0)
    108            phase = int(phase_value.item()) if torch.is_tensor(phase_value) else int(phase_value)
    109            grace_finished = self._elapsed_s + 1.0e-6 >= grace_period_s
    110            self._elapsed_s += env.step_dt
    111            if phase >= 2:
    112                return torch.zeros_like(unsafe, dtype=torch.bool)
    113            return grace_finished & unsafe
    114
    115
    116    class SwingContactTimeout(ManagerTermBase):
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/mdp/to_transition.py`** — transition_levels(function), scheduled_contact_reward(function), unload_fraction(calls), transition_yaw_reward(function), smoothstep(calls), transition_push(function)
    
    ```python
    167        freeze_level: bool = False
    168
    169
    170    def transition_levels(env, env_ids, command_name="pivot", min_episodes=256, success_rate=0.8):
    171        term = env.command_manager.get_term(command_name)
    172        ids = torch.arange(env.num_envs, device=env.device) if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=env.device)
    173        ids = ids[(term.steps[ids] > 0) & (term.episode_level[ids] == term.level)]
    174        term.window_count += len(ids)
    175        term.window_success += int(term.successful(ids).sum().item())
    176        # Old behavior (kept here as a reference): every successful window could
    177        # advance the global level automatically.
    178        # if term.window_count >= min_episodes:
    179        #     term.last_success_rate = term.window_success / term.window_count
    180        #     if term.last_success_rate >= success_rate:
    181        #         term.level = min(term.level + 1, 4)
    182        #     term.window_count = term.window_success = 0
    183
    184        # New behavior: an explicit freeze keeps the configured level fixed while
    185        # still reporting the observed window success rate for diagnostics. The
    186        # fallback keeps older externally-created command configs runnable.
    187        if term.window_count >= min_episodes:
    
    ... (gap) ...
    
    199        return term.weights[term.episode_level] * torch.exp(-k_pose * error)
    200
    201
    202    def scheduled_contact_reward(env):
    203        term = env.command_manager.get_term("pivot")
    204
    205        contact = term.contacts()
    206        phase = term.command[:, 1]
    207        unload = unload_fraction(phase)
    208
    209        # FL, FR, HL, HR
    210        four_wheel = contact.mean(dim=-1)
    211
    212        # Desired final support: FL + HR
    213        diagonal = 0.5 * (
    214            contact[:, 0]
    215            + contact[:, 3]
    216            - contact[:, 1]
    217            - contact[:, 2]
    218        )
    219
    220        return (1.0 - unload) * four_wheel + unload * diagonal
    221
    222    def transition_yaw_reward(env, k_yaw=11.11):
    223        term = env.command_manager.get_term("pivot")
    224        error = term.robot.data.root_ang_vel_b[:, 2] - term.command[:, 0]
    225        return smoothstep((term.command[:, 1] - 0.8) / 0.2) * torch.exp(-k_yaw * error.square())
    226
    227
    228    def reset_transition_standing(
    
    ... (gap) ...
    
    244        reset_four_wheel_standing(env, env_ids, **kwargs)
    245
    246
    247    def transition_push(env, env_ids, max_velocity=0.15):
    248        term = env.command_manager.get_term("pivot")
    249        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    250        ids = ids[(term.episode_level[ids] >= 3) & (term.command[ids, 1] >= 1)]
    251        velocity = term.robot.data.root_vel_w[ids].clone()
    252        velocity[:, :2] += torch.empty(len(ids), 2, device=env.device).uniform_(-max_velocity, max_velocity)
    253        if len(ids):
    254            term.robot.write_root_velocity_to_sim(velocity, env_ids=ids)
    255
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py`** — reset(calls), update(method), YawFSMVectorized(class), YawFSMCommand(class), YawRateCommand(extends), __init__(method), +20 more
    
    ```python
    529            return self.fsm_state
    530
    531
    532    class YawFSMCommand(YawRateCommand):
    533        """Yaw-rate command coupled to the per-environment yaw FSM.
    534
    535        ``VQRYawFSM`` is deliberately kept as the single source of truth for the
    
    ... (gap) ...
    
    545        cfg: "YawFSMCommandCfg"
    546
    547        def __init__(self, cfg: "YawFSMCommandCfg", env):
    548            super().__init__(cfg, env)
    549
    550            self._fsm = YawFSMVectorized(
    551                self.num_envs,
    552                device=self.device,
    553                yaw_enter=cfg.yaw_enter,
    
    ... (gap) ...
    
    597            self._fsm_scene_entities_resolved = False
    598
    599        @property
    600        def fsm_state(self) -> torch.Tensor:
    601            """Current FSM state for every environment as ``VQRFsmState`` values."""
    602            return self._fsm_state
    603
    604        def set_fsm_inputs(
    605            self,
    
    ... (gap) ...
    
    620                    raise ValueError(f"FSM predicate must have shape {tuple(target.shape)}, got {tuple(value.shape)}")
    621                target.copy_(value.to(device=self.device, dtype=torch.bool))
    622
    623        def _reset_fsm(self, env_ids: Sequence[int] | slice) -> None:
    624            self._fsm.reset(env_ids)
    625            self.yaw_entry_pos[env_ids] = 0.0
    626
    627            self.positive_pose_ready[env_ids] = False
    628            self.negative_pose_ready[env_ids] = False
    629            self.four_stand_ready[env_ids] = False
    630            self.unsafe[env_ids] = False
    631
    632        def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
    633            """Reset command metrics and FSM state for the selected environments."""
    634            extras = super().reset(env_ids)
    635            self._reset_fsm(slice(None) if env_ids is None else env_ids)
    636            return extras
    637
    638        def _step_fsm(self) -> None:
    639            self._update_fsm_predicates()
    640            self._fsm.update(
    641                yaw_cmd=self._command[:, 0],
    642                positive_pose_ready=self.positive_pose_ready,
    643                negative_pose_ready=self.negative_pose_ready,
    
    ... (gap) ...
    
    675
    676            from .observations import yaw_fsm_predicates
    677
    678            predicates = yaw_fsm_predicates(
    679                self._env,
    680                robot_name=self.cfg.asset_name,
    681                support_sensor_cfg=configs[0],
    
    ... (gap) ...
    
    692                unsafe_angle_limit=self.cfg.unsafe_angle_limit,
    693                minimum_base_height=self.cfg.minimum_base_height,
    694            )
    695            self.set_fsm_inputs(*predicates)
    696
    697        def _update_command(self):
    698            """Keep the base command behavior, then advance the FSM once."""
    699            super()._update_command()
    700            self._step_fsm()
    701
    702
    703    @configclass
    704    class YawFSMCommandCfg(YawRateCommandCfg):
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm_gates.py`** — _required_tensor(calls), _CACHE_STEP_ATTR(variable), _CACHE_ATTR(variable), _step_key(function), _required_tensor(function), fsm_gates(function), +1 more
    
    ```python
    48        return value
    49
    50
    51    def fsm_gates(env, command_name: str) -> dict[str, torch.Tensor]:
    52        """Return all shared hard, soft, and diagonal FSM gates.
    53
    54        The returned dictionary is cached per ``(env.common_step_counter,
    55        command_name)``.  All state-derived masks are boolean; ``f_*`` entries
    56        are float masks suitable for reward multiplication.
    57
    58        Keys:
    59            ``b_four``, ``b_trans``, ``b_yaw``, ``b_return``, ``b_safe``:
    60                Hard state masks.
    61            ``f_four``, ``f_trans``, ``f_yaw``, ``f_return``, ``f_safe``:
    62                Float versions of the hard masks.
    63            ``f_geom``:
    64                Float mask for diagonal geometry terms during transition and yaw.
    65            ``f_stability``:
    66                Soft four-stand stability gate.  It fades out over the first
    67                second of a transition and fades in over the first second of a
    68                return, while remaining active in FOUR_STAND. SAFE_RECOVERY is
    69                deliberately excluded from every positive FSM reward gate.
    70            ``diag_pos``, ``diag_neg``:
    71                Support-diagonal selectors from ``support_diagonal``.
    72            ``b_return_complete``:
    73                One-step pulse when RETURN_TO_4 reaches four-wheel readiness.
    74            ``fsm_state``, ``support_diagonal``, ``state_time``, ``tau``,
    75            ``just_switched``:
    76                The live command buffers used by stateful reward terms.
    77        """
    78        step = _step_key(env)
    79        cache_step = getattr(env, _CACHE_STEP_ATTR, None)
    80        cache = getattr(env, _CACHE_ATTR, None)
    81
    82        if cache_step != step or cache is None:
    83            cache_step = step
    84            cache = {}
    85            setattr(env, _CACHE_STEP_ATTR, cache_step)
    86            setattr(env, _CACHE_ATTR, cache)
    87
    88        cached = cache.get(command_name)
    89        if cached is not None:
    90            return cached
    91
    92        command = env.command_manager.get_term(command_name)
    93        device = torch.device(getattr(env, "device", "cpu"))
    94
    95        state = _required_tensor(command, "fsm_state", device=device).to(dtype=torch.long)
    96        support_diagonal = _required_tensor(command, "support_diagonal", device=device).to(dtype=torch.long)
    97        state_time = _required_tensor(command, "state_time", device=device).to(dtype=torch.float32)
    98        just_switched = _required_tensor(command, "just_switched", device=device).to(dtype=torch.bool)
    99        return_complete = torch.as_tensor(
    100            getattr(command, "just_returned_to_four", torch.zeros_like(just_switched)),
    101            device=device,
    102            dtype=torch.bool,
    103        )
    104
    105        if state.shape != support_diagonal.shape or state.shape != state_time.shape:
    106            raise ValueError(
    ```
    
    
    > Some file sections were trimmed for size. For a specific symbol you still need, run another `codegraph_explore` (or `codegraph_node`) with its exact name — line-numbered source, cheaper and more complete than Read.
    ✓ • 207ms

## Activity

    $ sed -n '780,880p' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py
                "command_name": "yaw_rate_cmd",
                "threshold": 1.0,
                "timeout_s": 0.20,
            },
        )
    
    
    @configclass
    class VQRWheelRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
        actions: VQRWheelActionsCfg = VQRWheelActionsCfg()
        rewards: VQRWheelRewardsCfg = VQRWheelRewardsCfg()
        terminations: VQRWheelYawTerminationsCfg = VQRWheelYawTerminationsCfg()
        curriculum: VQRWheelYawCurriculumCfg = VQRWheelYawCurriculumCfg()
    
        base_link_name = "TORSO"
        foot_link_name = ".*_WHEEL"
    
        # fmt: off
        leg_joint_names = [
            "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
            "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
            "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
            "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
        ]
        wheel_joint_names = [
            "FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL",
        ]
    
        hipx_joint_names = [
            "FL_HipX_joint", "FR_HipX_joint", "HL_HipX_joint", "HR_HipX_joint",
        ]
    
        hipy_joint_names = [
            "FL_HipY_joint", "FR_HipY_joint", "HL_HipY_joint", "HR_HipY_joint",
        ]
    
        knee_joint_names = [
            "FL_Knee_joint", "FR_Knee_joint", "HL_Knee_joint", "HR_Knee_joint",
        ]
        joint_names = leg_joint_names + wheel_joint_names
        # fmt: on
    
        def __post_init__(self):
            # post init of parent
            super().__post_init__()
    
            # ------------------------------Sence------------------------------
            self.scene.robot = VQRWHEEL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
            self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
    
            # ------------------------------Observations------------------------------
            self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
            self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=self.wheel_joint_names
            )
            self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
            self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=self.wheel_joint_names
            )
            self.observations.policy.base_ang_vel.scale = 0.25
            self.observations.policy.joint_pos.scale = 1.0
            self.observations.policy.joint_vel.scale = 0.05
            self.observations.policy.height_scan = None
            self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
            self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
    
            # increase observation noise to reduce reliance on precise instantaneous feedback
            # (helps close the sim-to-real gap and reduce high-frequency action jitter)
            self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.4, n_max=0.4)
            self.observations.policy.projected_gravity.noise = Unoise(n_min=-0.1, n_max=0.1)
            self.observations.policy.joint_vel.noise = Unoise(n_min=-3.0, n_max=3.0)
            # joint_pos: per-step noise + a per-episode, per-joint constant bias (~encoder/mechanical
            # calibration offset observed on real hardware, up to ~0.1 rad)
            self.observations.policy.joint_pos.noise = NoiseModelWithAdditiveBiasCfg(
                noise_cfg=Unoise(n_min=-0.02, n_max=0.02),
                bias_noise_cfg=Unoise(n_min=-0.1, n_max=0.1, operation="abs"),
                sample_bias_per_component=True,
            )
    
            # ------------------------------Actions------------------------------
            # Wider per-joint residual ranges improve pose discovery.
            self.actions.joint_pos.scale = {
                ".*_HipX_joint": 0.30,
                ".*_HipY_joint": 0.60,
                ".*_Knee_joint": 0.50,
            }
            self.actions.joint_vel.scale = 5.0
            self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
            self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
            self.actions.joint_pos.joint_names = self.leg_joint_names
            self.actions.joint_vel.joint_names = self.wheel_joint_names
    
            # ------------------------------Events------------------------------
            self.events.randomize_reset_base.params = {
                "pose_range": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
    ✓ • 0ms

## Activity

    $ rg --files | rg 'audit|status|run|fsm' | head -100
    docs/document_yaw_fsm/AGENTS.md
    docs/document_yaw_fsm/training_status_2026-09-24_11-23-35.md
    docs/document_yaw_fsm/fsm_review_2026-09-24.md
    docs/document_yaw_fsm/observation.md
    docs/document_yaw_fsm/fsm.md
    tests/test_yaw_fsm_observations.py
    tests/test_yaw_fsm_training_contract.py
    tests/test_yaw_fsm.py
    tests/test_yaw_fsm_systemic.py
    scripts/tools/compare_runs.py
    scripts/tools/run_yaw_contact_trace.sh
    pose_optimization/run_rear_axle.sh
    pose_optimization/src/model_audit.cpp
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm_gates.py
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm_mirror.py
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py
    ✓ • 0ms

## Activity

    $ rg -n 'unsafe|termination|transition_duration|0\.184' docs/document_yaw_fsm/training_status_2026-09-24_11-23-35.md | tail -80
    23:| Watchdog | TRANSITION quá 3.0 s thì terminate; một swing wheel contact liên tục ≥ 0.20 s trong YAW thì terminate; unsafe (torso contact, base thấp hoặc nghiêng quá `0.80 rad`) terminate sau grace `0.15 s` ở phase A/B |
    25:FSM có 7 trạng thái: `0 FOUR_STAND`, `1 TRANSITION_POS`, `2 YAW_POS`, `3 TRANSITION_NEG`, `4 YAW_NEG`, `5 RETURN_TO_4`, `6 SAFE_RECOVERY`. Phase A/B dùng termination cho unsafe; phase C dùng SAFE_RECOVERY với dwell tối thiểu `0.5 s`. Run hiện ở phase A.
    73:| `Episode_Termination/torso_contact` | 4.00% | Đây là tên term unsafe FSM, gồm torso contact **hoặc** base thấp/nghiêng quá ngưỡng; không được hiểu là chỉ torso va đất |
    120:| **B: transition** | Mỗi hướng cần ≥1024 episode đã thử transition/cửa sổ; tỉ lệ vào YAW ≥0.85 ở cả hai hướng trong 3 cửa sổ, giữ mức ≥6000 bước. | Clearance `0.20 m`; tracking floor `0.25` ở đầu transition rồi về 0 trong 3 s; drift weight ramp tới `−0.5`; unsafe vẫn terminate. |
    ✓ • 0ms

## Activity

    $ sed -n '1,120p' tests/test_yaw_fsm_systemic.py
    """CPU regressions for reward stalling, watchdog chatter, and survivor bias."""
    
    import ast
    from types import SimpleNamespace
    
    import pytest
    import torch
    
    from test_yaw_curriculum import _fsm_checkpoint_env, _fsm_curriculum_params, _load_curriculums_module
    from test_yaw_fsm import FSM_PATH, REWARDS_PATH, _load_fsm_module, _load_fsm_tracking_reward
    
    
    def _load_nodes(path, names, namespace):
        nodes = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    
    
    def _reward_scene():
        rewards, state = _load_fsm_tracking_reward()
        math = SimpleNamespace(
            euler_xyz_from_quat=lambda q: (q[:, 0], q[:, 1], q[:, 2]),
            wrap_to_pi=lambda x: x,
        )
        rewards.update(math_utils=math, Articulation=object)
        _load_nodes(REWARDS_PATH, {"TransitionProgress", "yaw_balance", "four_stand_stability",
                                  "yaw_base_height_tracking"}, rewards)
        command = SimpleNamespace(
            fsm_state=torch.tensor([state.TRANSITION_POS, state.TRANSITION_NEG]),
            support_diagonal=torch.tensor([1, -1]), state_time=torch.zeros(2),
            just_switched=torch.ones(2, dtype=torch.bool),
            cfg=SimpleNamespace(yaw_rate_range=(-0.25, 0.25)),
        )
        forces = torch.zeros(2, 4, 3)
        forces[0, [0, 3], 2] = 80
        forces[1, [1, 2], 2] = 80
    
        class Scene(dict):
            sensors = {"contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))}
            env_origins = torch.zeros(2, 3)
    
        env = SimpleNamespace(
            num_envs=2, device="cpu", common_step_counter=0,
            scene=Scene(robot=SimpleNamespace(data=SimpleNamespace(
                root_pos_w=torch.tensor([[0., 0., .49], [0., 0., .49]]),
                root_quat_w=torch.zeros(2, 4), root_ang_vel_b=torch.zeros(2, 3),
            ))),
            lift_progress={"pos": torch.zeros(2, 2), "neg": torch.zeros(2, 2)},
            command_manager=SimpleNamespace(get_term=lambda _: command,
                                            get_command=lambda _: torch.tensor([[.2], [-.2]])),
        )
        return rewards, state, env, command
    
    
    def _pose_rewards(rewards, env):
        fsm = {"fsm_command_name": "yaw_rate_cmd"}
        lift = dict(asset_cfg=SimpleNamespace(name="pos"), asset_cfg_mirror=SimpleNamespace(name="neg"),
                    wheel_radius=.091, target_clearance=.05, **fsm)
        support = dict(sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=[0, 3]),
                       sensor_cfg_mirror=SimpleNamespace(name="contact_forces", body_ids=[1, 2]))
        geom = dict(asset_cfg=SimpleNamespace(name="robot"), asset_cfg_mirror=SimpleNamespace(name="robot"))
        return torch.stack([
            2 * rewards["yaw_balance"](env, std=.25, **fsm),
            3 * rewards["yaw_com_support"](env, std=.08, **geom, **fsm),
            2 * rewards["yaw_com_inside_support_segment"](env, std=.05, **geom, **fsm),
            3 * rewards["yaw_transition_support_load"](env, target_force_n=80, **support, **fsm),
            3 * rewards["yaw_lift_clearance"](env, **lift,
                support_sensor_cfg=support["sensor_cfg"], support_sensor_cfg_mirror=support["sensor_cfg_mirror"]),
            8 * rewards["fsm_gated_tracking"](env, command_name="yaw_rate_cmd", **fsm,
                support_sensor_cfg=support["sensor_cfg"], support_sensor_cfg_mirror=support["sensor_cfg_mirror"],
                lifted_asset_cfg=lift["asset_cfg"], lifted_asset_cfg_mirror=lift["asset_cfg_mirror"],
                wheel_radius=.091, target_clearance=.05, std=.3),
            3 * rewards["four_stand_stability"](env, target_height=.49, **fsm),
            .49 * rewards["yaw_base_height_tracking"](env, target_height=.49, error_scale=.1, **fsm),
        ])
    
    
    @pytest.mark.parametrize("clearance", [0., .5, .79, 1.])
    def test_stalled_transition_has_no_positive_pose_income_and_handoff_does_not_drop(clearance):
        rewards, state, env, command = _reward_scene()
        for progress in env.lift_progress.values():
            progress.fill_(clearance)
        for elapsed in (0., .5, 1., 2.9):
            env.common_step_counter += 1
            command.state_time.fill_(elapsed)
            transition = _pose_rewards(rewards, env)
            assert (transition <= 0).all()
            assert torch.allclose(transition[:, 0], transition[:, 1])
        # Identical physical pose, changed FSM gate: no acquisition reward cliff.
        command.fsm_state[:] = torch.tensor([state.YAW_POS, state.YAW_NEG])
        env.common_step_counter += 1
        yaw = _pose_rewards(rewards, env)
        assert (yaw.sum(0) >= transition.sum(0)).all()
        assert torch.allclose(yaw[:, 0], yaw[:, 1])
    
    
    def test_lift_progress_cannot_be_refarmed_after_lowering_or_yaw_chatter():
        rewards, state, env, command = _reward_scene()
        progress_reward = rewards["TransitionProgress"](None, env)
        args = dict(asset_cfg=SimpleNamespace(name="pos"), asset_cfg_mirror=SimpleNamespace(name="neg"),
                    wheel_radius=.091, target_clearance=.05, fsm_command_name="yaw_rate_cmd",
                    support_sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=[0, 3]),
                    support_sensor_cfg_mirror=SimpleNamespace(name="contact_forces", body_ids=[1, 2]))
    
        def step(clearance, states):
            env.common_step_counter += 1
            command.fsm_state[:] = torch.tensor(states)
            for progress in env.lift_progress.values():
                progress.fill_(clearance)
            return progress_reward(env, **args)
    
        trans = [state.TRANSITION_POS, state.TRANSITION_NEG]
        yaw = [state.YAW_POS, state.YAW_NEG]
        assert (step(.5, trans) > 0).all()
        for _ in range(20):
            assert (step(.5, trans) <= 0).all()
        assert (step(.2, trans) <= 0).all()
        assert (step(.5, trans) <= 0).all()
        assert torch.equal(step(.8, yaw), torch.zeros(2))
        assert (step(.8, trans) <= 0).all()
    ✓ • 0ms

## Activity

    $ rg -n 'safe_recovery_entry|transition_progress|fsm_gated_tracking|com_support|transition_support_load|four_stand_stability|lift_clearance' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py | head -100
    687:    safe_recovery_reward_name: str = "safe_recovery_entry",
    ✓ • 0ms

## Activity

    $ sed -n '650,750p' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py
            "support_threshold": log_scalar(support_threshold),
            "lift_progress_threshold": log_scalar(lift_progress_threshold),
            "balance_threshold": log_scalar(balance_threshold),
            "yaw_threshold": log_scalar(yaw_threshold),
            "tracking_ratio_threshold": log_scalar(tracking_ratio_threshold),
            "edge_tracking_ratio_threshold": log_scalar(edge_tracking_ratio_threshold),
            "minimum_base_height": log_scalar(minimum_base_height),
            "required_success_rate": log_scalar(required_success_rate),
            "required_consecutive_windows": log_scalar(required_consecutive_windows),
        }
    
    
    def yaw_fsm_task_levels(
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str,
        clearance_levels: Sequence[float],
        yaw_rate_levels: Sequence[float],
        dr_scale_levels: Sequence[float],
        tracking_ratio_thresholds: Sequence[float],
        edge_tracking_ratio_thresholds: Sequence[float],
        lift_reward_name: str,
        balance_reward_name: str,
        yaw_reward_name: str,
        torso_contact_termination_name: str,
        minimum_base_height: float,
        support_threshold: float,
        lift_progress_threshold: float,
        balance_threshold: float,
        yaw_threshold: float,
        min_evaluated_episodes: int,
        required_success_rate: float,
        required_consecutive_windows: int,
        min_clearance_stage_steps: int,
        min_yaw_stage_steps: int,
        transition_reward_name: str,
        spin_center_drift_reward_name: str = "spin_center_drift",
        safe_recovery_reward_name: str = "safe_recovery_entry",
        min_directional_episodes: int | None = None,
        transition_success_threshold: float | None = None,
        drift_thresholds: Sequence[float] | None = None,
        reward_ramp_steps: int = 3600,
    ) -> dict[str, torch.Tensor]:
        """Three-phase FSM curriculum with independent POS/NEG gates.
    
        Phase A certifies lift per attempted maneuver, phase B certifies that
        each diagonal reaches ``YAW_*`` without a terminal attempt failure, and
        phase C climbs the yaw/DR ladder. A phase can advance
        only when *both* diagonals pass the same evaluation window.  In
        particular, this intentionally never averages POS and NEG scores.
    
        The FSM rewards own the directional per-episode buffers.  They coexist
        with the ``_yaw_*`` buffers used by :func:`yaw_task_levels`, preserving the
        original task and its checkpoint-export contract. The legacy ``episodes``
        counters/parameter names count individual maneuver attempts in this FSM.
        """
        del balance_reward_name, torso_contact_termination_name, minimum_base_height, balance_threshold
        del yaw_threshold, edge_tracking_ratio_thresholds
        validate_yaw_curriculum_levels(
            clearance_levels,
            yaw_rate_levels,
            dr_scale_levels,
            tracking_ratio_thresholds,
            # FSM phase C uses signed YAW-state samples instead of legacy edge samples.
            tracking_ratio_thresholds,
        )
        if min_evaluated_episodes <= 0:
            raise ValueError("min_evaluated_episodes must be positive.")
        if required_consecutive_windows <= 0:
            raise ValueError("required_consecutive_windows must be positive.")
        if min_clearance_stage_steps < 0 or min_yaw_stage_steps < 0:
            raise ValueError("Minimum stage durations must be non-negative.")
        if reward_ramp_steps < 0:
            raise ValueError("reward_ramp_steps must be non-negative.")
        if not 0.0 <= required_success_rate <= 1.0:
            raise ValueError("required_success_rate must be in [0, 1].")
    
        min_directional_episodes = (
            max(1, min_evaluated_episodes // 2)
            if min_directional_episodes is None
            else int(min_directional_episodes)
        )
        if min_directional_episodes <= 0:
            raise ValueError("min_directional_episodes must be positive.")
        transition_success_threshold = (
            required_success_rate
            if transition_success_threshold is None
            else float(transition_success_threshold)
        )
        if not 0.0 <= transition_success_threshold <= 1.0:
            raise ValueError("transition_success_threshold must be in [0, 1].")
        if drift_thresholds is None:
            drift_thresholds = (0.08,) * len(yaw_rate_levels)
        if len(drift_thresholds) != len(yaw_rate_levels) or any(value <= 0.0 for value in drift_thresholds):
            raise ValueError("drift_thresholds must be positive and match yaw_rate_levels.")
    
        current_step = int(env.common_step_counter)
        defaults = {
            "_yaw_fsm_task_curriculum_phase": 0,  # 0=lift, 1=transition, 2=yaw+DR
            "_yaw_task_curriculum_stage": 0,  # Retained for existing curriculum-state export.
            "_yaw_task_curriculum_yaw_stage": 0,
    ✓ • 0ms

## Activity

    $ rg -n 'set_term_cfg|reward_manager|safe_recovery_reward_name|transition_reward_name|weight' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py | tail -60
    80:        episode_sums = env.reward_manager._episode_sums[reward_term_name]
    81:        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    85:        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
    195:    transition_reward_name: str | None = None,
    294:            reward_weight = env.reward_manager.get_term_cfg(reward_name).weight
    295:            if reward_weight <= 0.0:
    296:                raise ValueError(f"Curriculum reward '{reward_name}' must have a positive weight.")
    297:            weighted_sum = env.reward_manager._episode_sums[reward_name][completed_env_ids]
    298:            return weighted_sum / (episode_duration * reward_weight)
    546:    if transition_reward_name is not None:
    547:        reward_names.append(transition_reward_name)
    549:        reward_cfg = env.reward_manager.get_term_cfg(reward_name)
    551:        env.reward_manager.set_term_cfg(reward_name, reward_cfg)
    559:    env.event_manager.set_term_cfg("randomize_apply_external_force_torque", force_cfg)
    565:    env.event_manager.set_term_cfg("randomize_actuator_gains", gain_cfg)
    573:    env.event_manager.set_term_cfg("randomize_push_robot", push_cfg)
    591:    env.event_manager.set_term_cfg("randomize_reset_base", reset_cfg)
    685:    transition_reward_name: str,
    687:    safe_recovery_reward_name: str = "safe_recovery_entry",
    1185:    for reward_name in (lift_reward_name, yaw_reward_name, transition_reward_name):
    1186:        reward_cfg = env.reward_manager.get_term_cfg(reward_name)
    1188:        env.reward_manager.set_term_cfg(reward_name, reward_cfg)
    1193:    phase_weights = ((8.0, 2.0, 0.0, 0.0), (8.0, 2.0, -0.5, 0.0), (8.0, 2.0, -2.0, -2.0))[phase]
    1196:        transition_reward_name,
    1198:        safe_recovery_reward_name,
    1204:    for reward_name, target_weight in zip(phase_terms, phase_weights):
    1205:        reward_cfg = env.reward_manager.get_term_cfg(reward_name)
    1208:            setattr(env, start_name, float(reward_cfg.weight))
    1210:            setattr(env, start_name, float(reward_cfg.weight))
    1211:        reward_cfg.weight = float(getattr(env, start_name) + ramp * (target_weight - getattr(env, start_name)))
    1217:        env.reward_manager.set_term_cfg(reward_name, reward_cfg)
    1223:    env.event_manager.set_term_cfg("randomize_apply_external_force_torque", force_cfg)
    1228:    env.event_manager.set_term_cfg("randomize_actuator_gains", gain_cfg)
    1232:    env.event_manager.set_term_cfg("randomize_push_robot", push_cfg)
    1240:    env.event_manager.set_term_cfg("randomize_reset_base", reset_cfg)
    ✓ • 0ms

## Activity

    $ rg -n 'decimation|dt=|episode_length_s|class VQRWheelFSMEnvCfg|class VQRWheelYawFSMEnvCfg' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_yaw_env_cfg.py | tail -70
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_yaw_env_cfg.py:918:    decimation = 4
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_yaw_env_cfg.py:919:    episode_length_s = 20.0
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_yaw_env_cfg.py:922:        dt=1 / 200,
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_yaw_env_cfg.py:923:        render_interval=decimation,
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_yaw_env_cfg.py:944:        self.decimation = 4
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_yaw_env_cfg.py:945:        self.episode_length_s = 20.0
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_yaw_env_cfg.py:948:        self.sim.render_interval = self.decimation
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_yaw_env_cfg.py:956:            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
    ✓ • 0ms

## Activity

    $ rg -n 'termination|fail|episode|transition' docs/document_yaw_fsm/fsm_review_2026-09-24.md | head -80
    7:FSM 7 trạng thái, đối xứng POS/NEG, đường sign flip qua `FOUR_STAND`, reset `support_diagonal`, và gate reward theo trạng thái đã được nối vào task. Mã hiện có một tham số không có tác dụng (`yaw_min_dwell`) và ba giới hạn cần lưu ý khi đọc kết quả training: watchdog swing contact có thể bị vô hiệu hóa bởi thời gian thoát YAW ngắn hơn; điểm bắt đầu đo drift được đặt lại mỗi lần tái nhập YAW; và phase B đánh giá thành công theo episode thay vì từng lần chuyển trạng thái. `RETURN_TO_4` không có watchdog riêng.
    12:- Physics chạy ở 200 Hz (`dt=0,005 s`), decimation 4: command, reward và FSM ở 50 Hz (`step_dt=0,020 s`); episode 20 s. Command yaw là tensor `(N, 1)`, resample sau 4–6 s. Biên yaw ban đầu do curriculum đặt thành `[-0,25; +0,25] rad/s`, dù cấu hình command khai báo `[-1; +1]` trước khi curriculum cập nhật: [môi trường](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_yaw_env_cfg.py#L914), [command](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/commands.py#L191), [curriculum](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py#L1144).
    14:- Theo Isaac Lab đang dùng, một `env.step()` tính **termination → reward → reset các env hoàn tất → command/FSM → observation**. Vì thế reward và các watchdog đọc trạng thái FSM ở cuối step trước, còn observation trả về dùng trạng thái vừa cập nhật. Termination unsafe đọc sensor hiện tại riêng. Ở phase C, khi unsafe mới xuất hiện, reward của step đó vẫn có thể đọc state cũ; từ step kế tiếp SAFE gate mới có hiệu lực. Đây là độ trễ theo thứ tự framework, cần dùng cùng một mốc state khi đối chiếu log. Nguồn framework: `/home/robotics/datld5/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:204-238`.
    32:## 4. Reward, observation, termination và curriculum
    37:| Reward | 22 term: 10 nền; 8 term hình học/ổn định có gate; 4 term tracking, tiến độ transition, drift và entry SAFE. `fsm_gated_tracking` chỉ cho tracking yaw khi **cả hai** bánh support contact; POS/NEG chọn chung một term qua `support_diagonal`. Hình học CoM và lift vẫn có tín hiệu khi mất contact. [Cấu hình](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py#L281), [tracking](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py#L680), [gate](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm_gates.py#L49) |
    38:| Termination | Unsafe sau grace tính từ reset 0,15 s ở phase A/B; phase C đưa vào SAFE thay vì terminate. TRANSITION liên tục ≥3,0 s terminate. Một bánh swing contact liên tục ≥0,20 s **khi ở YAW** terminate. [Mã](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py#L62) |
    39:| Curriculum | A: clearance `0,05→0,10→0,15→0,20 m`, đủ lift **và** support mỗi hướng; B: tỉ lệ episode từng hướng có TRANSITION rồi đến YAW ≥0,85; C: tăng yaw/online DR theo 6 mức, dùng tracking ratio và drift từng hướng. Mỗi hướng cần ≥1024 episode/cửa sổ và 3 cửa sổ pass; weight drift thay đổi bằng ramp 3600 step. [Mã](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py#L662) |
    41:Reward dương bị tắt trong `SAFE_RECOVERY`; penalty nền vẫn tính. `transition_progress` dùng potential có `gamma=0,99`, cập nhật potential cả ngoài state được gate và zero reward ở bước vừa đổi state. `spin_center_drift` chỉ có weight ở phase B/C. Khi log, `drift_10s=0` mà `drift_10s_samples=0` nghĩa là **chưa đo được**, không phải drift bằng 0.
    47:Nếu bánh swing chạm nền, clearance của nó thường thấp hơn ngưỡng pose ready. FSM rời `YAW_*` về `TRANSITION_*` sau 0,10 s mất pose liên tục; timer của `SwingContactTimeout` chỉ chạy ở YAW và bị xóa ngay khi rời YAW. Ngưỡng termination là 0,20 s. Vì vậy một chuỗi contact trên nền phẳng có thể liên tục gây mất pose nhưng **không bao giờ đủ 0,20 s trong YAW** để kích hoạt watchdog. Đây là suy luận từ [pose predicate](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/observations.py#L230), [pose-loss grace](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py#L393), [timer](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py#L159); contact do va vào vật cao có thể khác. Nên kiểm tra watchdog bằng trajectory contact thật và quyết định tính thời gian qua cả `TRANSITION_*` hoặc đổi ngưỡng/điều kiện cho phù hợp.
    57:### F4 — Trung bình: phase B đếm thành công theo episode, không theo từng attempt
    59:Reward ghi hai boolean `transition_attempted` và `transition_succeeded` cho **mỗi episode và mỗi hướng**. Curriculum đánh giá `success=transitioned` khi `phase==1`; một attempt abort rồi một attempt sau thành công trong cùng episode vẫn được tính một lần thành công. Vì thế metric không chứng minh mọi lần chuyển trạng thái đạt YAW trong 3 s. Watchdog 3 s chỉ bắt chuỗi TRANSITION liên tục: [ghi cờ](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py#L847), [đánh giá](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py#L844), [watchdog](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py#L173). Nếu tiêu chí nghiệm thu là tỉ lệ thành công **mỗi attempt**, cần đếm attempt và thời gian tương ứng.
    63:`RETURN_TO_4` chỉ thoát khi cả bốn bánh contact và roll/pitch đủ nhỏ; watchdog 3 s chỉ áp dụng cho `TRANSITION_*`, còn watchdog swing chỉ cho `YAW_*`. Nếu điều kiện bốn bánh không xảy ra, env có thể ở RETURN đến episode timeout 20 s: [cạnh RETURN](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py#L445), [watchdog](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py#L173). Nên log thời gian RETURN và tỉ lệ episode timeout trong RETURN trước khi quyết định thêm timeout hoặc recovery path.
    67:Ở iteration 629 của [run cũ](training_status_2026-09-24_11-23-35.md), support score POS/NEG là 0,383/0,421 và `fail_support_rate=100%` cả hai hướng, trong khi lift progress khoảng 0,92. `com_support` và `com_inside_segment` vẫn thưởng hình học khi support contact mất; `fsm_gated_tracking` mới yêu cầu cả hai support contact cho phần tracking và cho partial bonus. Vì vậy reward tổng tăng chưa chứng minh robot giữ đúng hai bánh support. Đây là **quan sát training cũ cộng với rủi ro thiết kế reward**, không phải lỗi transition graph mới được chứng minh. Nên ưu tiên xem `mean_support_score`, contact từng bánh, `support_loss_max_dwell_s` và episode pass phase A ở run với mã hiện tại.
    ✓ • 0ms

## Activity

    $ find . -maxdepth 2 -type d \( -name 'logs' -o -name 'runs' -o -name 'outputs' \) -print
    ./logs
    ./.git/logs
    ./outputs
    ✓ • 0ms

## Activity

    $ rg -n 'transition_duration|unsafe.*termination|0\.184' logs outputs -g '*.md' -g '*.json' -g '*.csv' -g '*.txt' | head -80
    ✓ • 0ms

## Activity

    $ rg --files logs outputs | head -60
    ✓ • 0ms

## Assistant

Mình đã xác nhận cơ chế gây lệch incentive trong mã: các term TRANSITION bị trừ mỗi bước 0,02 s; `transition_progress` chỉ thưởng khi tạo được mức lift mới; cấu hình hiện không có cost cho unsafe termination ở phase A/B. Vì vậy một attempt chưa đạt support có thể giảm return khi kéo dài, còn terminate sớm chặn các khoản trừ tiếp theo. Mình sẽ chạy kiểm tra CPU để xác nhận đường reward hiện tại và tách kết luận mã nguồn khỏi số liệu của run trước.

## Activity

    $ ls scripts source
    scripts:
    reinforcement_learning
    tools
    
    source:
    rl_training
    ✓ • 0ms

## Activity

    $ which python; python --version
    /bin/bash: line 1: python: command not found
    ✗ (127) • 0ms

## Activity

    $ python -m pytest -q tests/test_yaw_fsm_systemic.py tests/test_yaw_transition_safety.py
    /bin/bash: line 1: python: command not found
    ✗ (127) • 0ms

## Activity

    $ command -v python3; python3 --version; python3 -c 'import torch, pytest; print(torch.__version__, pytest.__version__)'
    /usr/bin/python3
    Python 3.10.12
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
    ModuleNotFoundError: No module named 'torch'
    ✗ (1) • 0ms

## Assistant

**Đúng: mã hiện tại vẫn tạo incentive “fail sớm” trong TRANSITION.** Khi mất cả hai bánh support và chưa nhấc bánh, ba term riêng đã tạo khoảng **−19 reward mỗi giây**: `fsm_gated_tracking` −10, `transition_support_load` −3 và `lift_clearance` −6. Các deficit về CoM, balance và stability có thể trừ thêm. [Reward functions](/home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:194) · [Weights](/home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:284)

Ở phase A/B, unsafe có thể terminate sau grace 0,15 s, nhưng cấu hình không có penalty terminal tương ứng; `safe_recovery_entry` có weight 0. Kết thúc ở khoảng 0,18 s vì thế tránh được phần deficit của những giây acquisition tiếp theo. Đây là **lỗi incentive được xác nhận từ mã**. Số `transition_duration≈0,184 s`, gần zero YAW và unsafe termination cao phù hợp với cơ chế này, nhưng riêng chúng chưa chứng minh policy chủ động gây ngã. [Termination](/home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py:62) · [Reward config](/home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:641)

Hướng sửa nên là giảm deficit **tích lũy theo thời gian** trong acquisition và đặt cost rõ ràng cho attempt thất bại, rồi so sánh return của các trajectory cùng trạng thái đầu: fail ở 0,18 s, tiếp tục acquisition, và vào YAW. Chỉ thêm một terminal penalty cố định mà không kiểm tra phép so sánh này có thể vẫn để lại incentive fail sớm.

Mình chưa sửa mã. Kiểm tra CPU chưa chạy được vì session không có `python` đang kích hoạt; `/usr/bin/python3` cũng thiếu `torch`.

## User

hãy đưa ra phương án sửa

## Assistant

Mình đề xuất sửa theo **hai phần đi cùng nhau**:

1. **Đổi reward acquisition trong TRANSITION thành thưởng tiến bộ có giới hạn.** Với support load, CoM và lift, chỉ trả thưởng khi chất lượng vượt mức tốt nhất đã đạt của cùng hướng trong episode. Khởi tạo mức tốt nhất từ tư thế lúc vào TRANSITION; không reset khi FSM chuyển qua lại giữa TRANSITION và YAW. Như vậy đứng yên không tích lũy deficit, còn hạ rồi nhấc bánh lại không farm reward. Giữ reward duy trì tư thế và tracking hiện tại trong YAW.

2. **Thêm cost một lần cho attempt thất bại.** Áp dụng khi unsafe hoặc transition timeout kết thúc attempt ở phase A/B; phase C áp dụng khi vào SAFE_RECOVERY. Chỉ tính một lần dù các mask torso, height và tilt cùng bật. Vì RewardManager nhân reward với `step_dt=0,02 s`, cost sự kiện phải được chuẩn hóa theo `1/step_dt` để giá trị không vô tình nhỏ đi 50 lần.

Mình **không khuyên chỉ tăng terminal penalty** trong khi giữ nguyên toàn bộ deficit: cost cố định vẫn có thể thấp hơn penalty tương lai, còn cost quá lớn dễ làm training nhiễu. Các penalty an toàn vật lý vẫn giữ, nhưng cần đo tổng penalty mỗi bước để chọn cost thất bại lớn hơn lợi ích của việc cắt ngắn attempt.

Điều kiện nghiệm thu là một test so sánh return có discount từ **cùng trạng thái ban đầu**, cho cả POS và NEG: terminate ở ~0,18 s phải kém hơn tiếp tục acquisition thêm 1–2 s, ngay cả khi chưa vào YAW; vào YAW phải tốt nhất. Sau đó mới dùng run ngắn để kiểm tra transition duration, unsafe rate, support từng bánh và tỷ lệ vào YAW.

## User

oke sửa đi

## Assistant

Mình sẽ sửa reward TRANSITION theo đúng tiêu chí vừa chốt: chỉ trả credit khi support, CoM hoặc lift vượt mức tốt nhất của episode, và tính cost một lần khi attempt thất bại. Trước khi đổi mã, mình đang kiểm tra thứ tự reward/termination và các test hiện có để tránh trả credit lặp khi FSM chuyển qua lại giữa TRANSITION và YAW.

## Activity

    $ codegraph explore "TransitionProgress test_lift_progress_cannot_be_refarmed_after_lowering_or_yaw_chatter yaw_com_inside_support_segment VQRWheelFSMRewardsCfg reward tests"
    **Dynamic boundaries (the static path ends at runtime dispatch)**
    
    - `test_lift_progress_cannot_be_refarmed_after_lowering_or_yaw_chatter` (tests/test_yaw_fsm_systemic.py:99) — computed member call: `progress_reward = rewards["TransitionProgress"](None, env)`
      candidates for key `TransitionProgress`: `TransitionProgress` (source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:1175)
    
    > These sites choose their call target at runtime (registry / bus / reflection) — the site shown IS where the flow continues. To follow it, run codegraph_explore or codegraph_node on a candidate; source for the sites above is included below.
    
    > Full source for these symbols is below — the call flow among them, followed by their bodies.
    **Exploration: TransitionProgress test_lift_progress_cannot_be_refarmed_after_lowering_or_yaw_chatter yaw_com_inside_support_segment VQRWheelFSMRewardsCfg reward tests**
    
    Found 33 symbols across 2 files.
    
    **Source Code**
    
    > The code below is the **verbatim, current on-disk source** of these files — re-read from disk on this call and line-numbered, byte-for-byte identical to what the Read tool returns. It is NOT a summary, outline, or stale cache. Treat each block as a Read you have already performed: do not Read a file shown here.
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py`** — calls(calls), fsm_gates(calls), _yaw_lift_progress(calls), _fsm_record_positive_budget(calls), _yaw_support_load_quality(calls), _yaw_support_geometry(calls), +17 more
    
    ```python
    408        return _yaw_wheel_contacts(env, sensor_cfg, threshold).float().prod(dim=1)
    409
    410
    411    def _yaw_lift_progress(
    412        env: ManagerBasedRLEnv,
    413        asset_cfg: SceneEntityCfg,
    414        wheel_radius: float,
    415        target_clearance: float,
    416    ) -> torch.Tensor:
    417        """Return independent normalized clearance progress for the selected wheels."""
    418        if target_clearance <= 0.0:
    419            raise ValueError("target_clearance must be positive.")
    420
    421        asset: Articulation = env.scene[asset_cfg.name]
    422        wheel_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    423        ground_height = env.scene.env_origins[:, 2].unsqueeze(-1)
    424        clearance = wheel_height - ground_height - wheel_radius
    425        return torch.clamp(clearance / target_clearance, min=0.0, max=1.0)
    426
    427
    428    def yaw_lift_clearance(
    429        env: ManagerBasedRLEnv,
    430        asset_cfg: SceneEntityCfg,
    431        wheel_radius: float,
    432        target_clearance: float,
    433        asset_cfg_mirror: SceneEntityCfg | None = None,
    434        fsm_command_name: str | None = None,
    435        support_sensor_cfg: SceneEntityCfg | None = None,
    436        support_sensor_cfg_mirror: SceneEntityCfg | None = None,
    437        support_force_target_n: float = 80.0,
    438        transition_ungated_fraction: float = 0.25,
    439    ) -> torch.Tensor:
    440        """Shape lifted-wheel clearance with partial credit and a four-wheel penalty.
    441
    442        Each selected wheel contributes independently. The baseline score is in
    443        ``[-1, 1]``: both wheels on the ground score ``-1``, lifting either wheel
    444        improves the score, and both wheels must reach the target to score ``1``.
    445        In FSM TRANSITION, support load shapes the score before subtracting its
    446        maximum, leaving a nonpositive deficit. Raw metrics and YAW are unchanged.
    447        """
    448        progress = _yaw_lift_progress(env, asset_cfg, wheel_radius, target_clearance)
    449
    450        score = 2.0 * progress.mean(dim=1) - 1.0
    451        if fsm_command_name is None:
    452            # Keep a separate curriculum metric: the weaker wheel's progress,
    453            # averaged over the episode. This must not be reconstructed from the
    454            # signed mean reward because one fully lifted wheel can hide the other.
    455            min_progress = progress.amin(dim=1)
    456            if not hasattr(env, "_yaw_lift_min_progress_sum"):
    457                env._yaw_lift_min_progress_sum = torch.zeros_like(min_progress)
    458                env._yaw_lift_min_progress_samples = torch.zeros_like(min_progress, dtype=torch.long)
    459            env._yaw_lift_min_progress_sum += min_progress
    460            env._yaw_lift_min_progress_samples += 1
    461            return score
    462        if asset_cfg_mirror is None:
    463            raise ValueError("asset_cfg_mirror is required when fsm_command_name is set.")
    464        if support_sensor_cfg is None or support_sensor_cfg_mirror is None:
    465            raise ValueError("Both support sensor configs are required when fsm_command_name is set.")
    466        if not 0.0 < transition_ungated_fraction < 1.0:
    467            raise ValueError("transition_ungated_fraction must be in (0, 1).")
    468        mirror_progress = _yaw_lift_progress(
    469            env,
    470            asset_cfg_mirror,
    471            wheel_radius,
    472            target_clearance,
    473        )
    474        mirror_score = 2.0 * mirror_progress.mean(dim=1) - 1.0
    475        gates = fsm_gates(env, fsm_command_name)
    476        selected_progress = torch.where(gates["diag_pos"].unsqueeze(1), progress, mirror_progress)
    477        min_progress = selected_progress.amin(dim=1)
    478        if not hasattr(env, "_yaw_lift_min_progress_sum"):
    479            env._yaw_lift_min_progress_sum = torch.zeros_like(min_progress)
    480            env._yaw_lift_min_progress_samples = torch.zeros_like(min_progress, dtype=torch.long)
    481        env._yaw_lift_min_progress_sum += min_progress
    482        env._yaw_lift_min_progress_samples += 1
    483        selected_score = torch.where(gates["diag_pos"], score, mirror_score)
    484        support_pos = _yaw_support_load_quality(env, support_sensor_cfg, support_force_target_n)
    485        support_neg = _yaw_support_load_quality(env, support_sensor_cfg_mirror, support_force_target_n)
    486        support_quality = torch.where(gates["diag_pos"], support_pos, support_neg)
    487        lift_gate = transition_ungated_fraction + (1.0 - transition_ungated_fraction) * support_quality
    488        # Preserve the no-lift gradient before centering the transition score.
    489        # Only positive lift credit is reduced when support is weak.
    490        transition_score = torch.clamp(selected_score, max=0.0) + torch.relu(selected_score) * lift_gate
    491        reward = (transition_score - 1.0) * gates["f_trans"] + selected_score * gates["f_yaw"]
    492        _fsm_record_positive_budget(env, gates, "lift_clearance", reward)
    493        return reward
    494
    495
    496    def yaw_com_inside_support_segment(
    497        env: ManagerBasedRLEnv,
    498        asset_cfg: SceneEntityCfg,
    499        std: float,
    500        asset_cfg_mirror: SceneEntityCfg | None = None,
    501        fsm_command_name: str | None = None,
    502    ) -> torch.Tensor:
    503        """Reward CoM projection inside the finite FL-HR support segment.
    504
    505        The score is one anywhere inside the segment and decays smoothly with the
    506        physical distance past either endpoint.
    507        """
    508        _, projection, segment_length = _yaw_support_geometry(env, asset_cfg)
    509        outside_distance = (
    510            torch.relu(-projection) + torch.relu(projection - 1.0)
    511        ) * segment_length
    512        score = torch.exp(-outside_distance.square() / std**2)
    513        if fsm_command_name is None:
    514            return score
    515        if asset_cfg_mirror is None:
    516            raise ValueError("asset_cfg_mirror is required when fsm_command_name is set.")
    517        _, mirror_projection, mirror_segment_length = _yaw_support_geometry(env, asset_cfg_mirror)
    518        mirror_outside_distance = (
    519            torch.relu(-mirror_projection) + torch.relu(mirror_projection - 1.0)
    520        ) * mirror_segment_length
    521        mirror_score = torch.exp(-mirror_outside_distance.square() / std**2)
    522        gates = fsm_gates(env, fsm_command_name)
    523        reward = torch.where(gates["diag_pos"], score, mirror_score) * gates["f_geom"] - gates["f_trans"]
    524        _fsm_record_positive_budget(env, gates, "com_inside_segment", reward)
    525        return reward
    526
    527
    528    def yaw_balance(
    529        env: ManagerBasedRLEnv,
    530        std: float,
    531        nominal_roll: float = 0.0,
    532        nominal_pitch: float = 0.0,
    533        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    534        fsm_command_name: str | None = None,
    535    ) -> torch.Tensor:
    536        """Reward roll and pitch near the configured diagonal-support equilibrium."""
    537        asset: Articulation = env.scene[asset_cfg.name]
    538        roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    539        roll_error = math_utils.wrap_to_pi(roll - nominal_roll)
    540        pitch_error = math_utils.wrap_to_pi(pitch - nominal_pitch)
    541        score = torch.exp(-(roll_error.square() + pitch_error.square()) / std**2)
    542        # ``None`` preserves the baseline task's exact calculation.  The FSM
    543        # task suppresses this otherwise-positive term in SAFE_RECOVERY.
    544        if fsm_command_name is None:
    545            return score
    546        gates = fsm_gates(env, fsm_command_name)
    547        reward = score * (1.0 - gates["f_safe"]) - gates["f_trans"]
    548        _fsm_record_positive_budget(env, gates, "balance", reward)
    549        return reward
    550
    551
    552    # def yaw_gated_tracking(
    
    ... (gap) ...
    
    850        leaves the diagnostic unset and never changes the reward value.
    851        """
    852        try:
    853            weight = float(env.reward_manager.get_term_cfg(term_name).weight)
    854        except (AttributeError, KeyError, TypeError):
    855            return
    856        positive = torch.relu(value * weight)
    
    ... (gap) ...
    
    902        if not 0.0 < edge_command_fraction <= 1.0:
    903            raise ValueError("edge_command_fraction must be in (0, 1].")
    904
    905        gates = fsm_gates(env, fsm_command_name)
    906        _fsm_step_telemetry(env, gates)
    907        support_pos_contact = _yaw_wheel_contacts(env, support_sensor_cfg, contact_threshold)
    908        support_neg_contact = _yaw_wheel_contacts(env, support_sensor_cfg_mirror, contact_threshold)
    909        support_pos = support_pos_contact.to(dtype=torch.float32).prod(dim=1)
    910        support_neg = support_neg_contact.to(dtype=torch.float32).prod(dim=1)
    911        support_gate = torch.where(gates["diag_pos"], support_pos, support_neg)
    
    ... (gap) ...
    
    932        support_contacts = torch.where(
    933            gates["diag_pos"].unsqueeze(1), support_pos_contact, support_neg_contact
    934        )
    935        support_shape = _yaw_support_shape(support_contacts)
    936        # The opposite support diagonal is exactly the active swing diagonal.
    937        swing_contact = select_swing_wheel_contact(
    938            support_pos_contact,
    939            support_neg_contact,
    940            gates["support_diagonal"],
    941        )
    942
    943        lift_pos = _yaw_lift_progress(env, lifted_asset_cfg, wheel_radius, target_clearance)
    944        lift_neg = _yaw_lift_progress(env, lifted_asset_cfg_mirror, wheel_radius, target_clearance)
    945        selected_lift = torch.where(gates["diag_pos"].unsqueeze(1), lift_pos, lift_neg)
    946        lift_progress = selected_lift.mean(dim=1)
    947
    948        asset: RigidObject = env.scene[asset_cfg.name]
    949        _fsm_transition_telemetry(
    950            env, gates, support_contacts, selected_lift, asset,
    951            env.command_manager.get_term(fsm_command_name),
    952        )
    
    ... (gap) ...
    
    960        # Reuse the legacy yaw-curriculum accumulator names.  This keeps the
    961        # existing checkpoint/export path valid while changing only the sample
    962        # mask: FSM tracking statistics are normalized by YAW-state steps.
    963        _fsm_masked_accumulate(
    964            env,
    965            support_gate,
    966            yaw_mask,
    967            "_yaw_support_score_sum",
    968            "_yaw_support_score_samples",
    969        )
    970        _fsm_masked_accumulate(
    971            env,
    972            support_gate,
    973            yaw_mask,
    
    ... (gap) ...
    
    1035                f"_yaw_fsm_{suffix}_swing_contact_samples",
    1036            )
    1037
    1038        _fsm_attempt_telemetry(
    1039            env, gates, lift_progress, support_gate, lift_progress_threshold, support_threshold
    1040        )
    1041        env._yaw_support_score_current = support_gate
    
    ... (gap) ...
    
    1172        return reward
    1173
    1174
    1175    class TransitionProgress(ManagerTermBase):
    1176        """Bounded new lift credit in TRANSITION; potential-based shaping in RETURN."""
    1177
    1178        def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
    1179            super().__init__(cfg, env)
    1180            self.prev_phi = torch.zeros(env.num_envs, device=env.device)
    1181            self.best_transition_progress = torch.zeros(env.num_envs, device=env.device)
    1182
    1183        def reset(self, env_ids=None) -> None:
    1184            if env_ids is None:
    1185                self.prev_phi.zero_()
    1186                self.best_transition_progress.zero_()
    1187            else:
    1188                self.prev_phi[env_ids] = 0.0
    1189                self.best_transition_progress[env_ids] = 0.0
    1190
    1191        def __call__(
    1192            self,
    1193            env: ManagerBasedRLEnv,
    1194            asset_cfg: SceneEntityCfg,
    1195            asset_cfg_mirror: SceneEntityCfg,
    1196            wheel_radius: float,
    1197            target_clearance: float,
    1198            fsm_command_name: str,
    1199            gamma: float = 0.99,
    1200            support_sensor_cfg: SceneEntityCfg | None = None,
    1201            support_sensor_cfg_mirror: SceneEntityCfg | None = None,
    1202            support_force_target_n: float = 80.0,
    1203            transition_ungated_fraction: float = 0.25,
    1204        ) -> torch.Tensor:
    1205            if not 0.0 < gamma <= 1.0:
    1206                raise ValueError("gamma must be in (0, 1].")
    1207            if support_sensor_cfg is None or support_sensor_cfg_mirror is None:
    1208                raise ValueError("Both support sensor configs are required for transition progress.")
    1209            if not 0.0 < transition_ungated_fraction < 1.0:
    1210                raise ValueError("transition_ungated_fraction must be in (0, 1).")
    1211
    1212            gates = fsm_gates(env, fsm_command_name)
    1213            progress_pos = _yaw_lift_progress(
    1214                env, asset_cfg, wheel_radius, target_clearance
    1215            ).mean(dim=1)
    1216            progress_neg = _yaw_lift_progress(
    1217                env, asset_cfg_mirror, wheel_radius, target_clearance
    1218            ).mean(dim=1)
    1219            progress = torch.where(gates["diag_pos"], progress_pos, progress_neg)
    1220            phi = torch.where(gates["b_return"], 1.0 - progress, progress)
    1221
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/mdp/to_transition.py`** — smoothstep(calls), reference(method), pose_reference(calls), _update_command(method), scheduled_contact_reward(function), unload_fraction(calls), +1 more
    
    ```python
    66            return self._command
    67
    68        @property
    69        def reference(self):
    70            return pose_reference(self.standing, self.target, self.command[:, 1])
    71
    72        def contacts(self):
    73            return (self.sensor.data.net_forces_w[:, self.contact_ids].norm(dim=-1) > self.cfg.contact_threshold).float()
    
    ... (gap) ...
    
    82            sign = torch.where(torch.rand_like(random) < 0.5, -1.0, 1.0)
    83            self.sampled_yaw[env_ids] = sign * magnitude * (0.25 + 0.75 * random)
    84
    85        def _update_command(self):
    86            self.elapsed += self._env.step_dt
    87            phase = ((self.elapsed - self.cfg.standing_time_s) / self.cfg.transition_duration_s).clamp(0, 1)
    88            self._command[:, 1] = phase
    89            self._command[:, 0] = self.sampled_yaw * smoothstep((phase - 0.8) / 0.2)
    90
    91        def _update_metrics(self):
    92            contact = self.contacts()
    
    ... (gap) ...
    
    199        return term.weights[term.episode_level] * torch.exp(-k_pose * error)
    200
    201
    202    def scheduled_contact_reward(env):
    203        term = env.command_manager.get_term("pivot")
    204
    205        contact = term.contacts()
    206        phase = term.command[:, 1]
    207        unload = unload_fraction(phase)
    208
    209        # FL, FR, HL, HR
    210        four_wheel = contact.mean(dim=-1)
    211
    212        # Desired final support: FL + HR
    213        diagonal = 0.5 * (
    214            contact[:, 0]
    215            + contact[:, 3]
    216            - contact[:, 1]
    217            - contact[:, 2]
    218        )
    219
    220        return (1.0 - unload) * four_wheel + unload * diagonal
    221
    222    def transition_yaw_reward(env, k_yaw=11.11):
    223        term = env.command_manager.get_term("pivot")
    224        error = term.robot.data.root_ang_vel_b[:, 2] - term.command[:, 0]
    225        return smoothstep((term.command[:, 1] - 0.8) / 0.2) * torch.exp(-k_yaw * error.square())
    226
    227
    228    def reset_transition_standing(
    ```
    
    **Not shown above — explore these names for their source**
    
    - tests/test_yaw_fsm_systemic.py: test_lift_progress_cannot_be_refarmed_after_lowering_or_yaw_chatter:97, step:105, _reward_scene:19, Scene:38, _load_nodes:13, test_stalled_transition_has_no_positive_pose_income_and_handoff_does_not_drop:79, +4 more
    - source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/mdp/to_reference.py: smoothstep:45, pose_reference:50, unload_fraction:54, contact_targets:58
    - source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm_gates.py: fsm_gates:51, _step_key:21, _required_tensor:37, _CACHE_ATTR:18
    - tests/test_yaw_transition_safety.py: ManagerTermBase:116, _transition_scene:16, test_lift_credit_uses_selected_support_diagonal_and_keeps_exploration:80, __init__:117, test_height_and_tilt_failure_are_separate_termination_masks:111
    - source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py: TorsoContactWithGrace:23, FSMUnsafeWithGrace:62, SwingContactTimeout:116
    - tests/test_yaw_fsm.py: _load_fsm_tracking_reward:39, FakeManagerTermBase:75, FakeSceneEntityCfg:79, _load_fsm_module:30, test_fsm_tracking_support_bonus_requires_lift_and_is_independent_of_yaw_error:530, test_fsm_pose_rewards_keep_dense_yaw_signal_without_support_contact:614, +5 more
    - source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/mdp/rewards.py: GaitReward:444, SustainedWheelTorquePenalty:1633
    - source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/tests/validate_four_wheel_stability.py: cfg:34, cfg:67, cfg:110
    - source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/tests/smoke_four_wheel_rotate.py: cfg:31
    
    > Some file sections were trimmed for size. For a specific symbol you still need, run another `codegraph_explore` (or `codegraph_node`) with its exact name — line-numbered source, cheaper and more complete than Read.
    ✓ • 273ms

## Activity

    $ sed -n '1,65p' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py; sed -n '1180,1220p' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py; sed -n '1,105p' tests/test_yaw_fsm.py
    # Copyright (c) 2025 Deep Robotics
    # SPDX-License-Identifier: BSD 3-Clause
    
    # Copyright (c) 2024-2025 Ziqi Fan
    # SPDX-License-Identifier: Apache-2.0
    
    from __future__ import annotations
    
    import math
    import torch
    from typing import TYPE_CHECKING
    
    import isaaclab.utils.math as math_utils
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.managers import ManagerTermBase
    from isaaclab.managers import RewardTermCfg as RewTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import ContactSensor, RayCaster
    from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, yaw_quat
    
    from .fsm_gates import fsm_gates
    from .fsm import select_swing_wheel_contact
    
    if TYPE_CHECKING:
        from isaaclab.envs import ManagerBasedRLEnv
    
    
    # Global curriculum scalar in [0, 1], updated from terrain-level mean.
    gait_level: float = 0.0
    
    #Reward mới cho task,làm lại
    #Phần phạt task chính:
    def custom_yaw_vel_tracking_exp(env: ManagerBasedRLEnv, std:float, command_name: str,
                                asset_cfg: SceneEntityCfg("robot")) -> torch.Tensor:
        robot: RigidObject = env.scene[asset_cfg.name]
        # Để lấy vận tốc góc yaw thực tế trong Isaaclab
        #
        actual_yaw_vel = robot.data.root_ang_vel_b[:,2]
        yaw_cmd = env.command_manager.get_command("command_name")
        return torch.exp(-torch.sum(torch.square(actual_yaw_vel - yaw_cmd))/0.25)
    
    #Phần phạt lệnh điều khiển:
    def motor_effort_penalty_l2(env: ManagerBasedRLEnv, std:float, command_name: str,
                                asset_cfg: SceneEntityCfg("robot")) -> torch.Tensor:
        action_effort = env.command_manager.get_command("command_name")
    
        raise NotImplementedError("motor_effort_penalty_l2 is not implemented")
    
    #Phần phạt dáng đứng:
    
    def update_gait_level_from_terrain_mean(terrain_level_mean: float | torch.Tensor) -> float:
        """Update global gait_level from mean terrain level.
    
        Mapping rule:
        - mean <= 0.0 -> 0.0
        - 0.0 < mean < 3.0 -> 使用 exp 函数映射
        - mean == 3.0 -> 1.0
        - mean >= 3.0 -> 1.0
        """
        global gait_level
    
        mean_tensor = torch.as_tensor(terrain_level_mean, dtype=torch.float32)
        if mean_tensor.numel() == 0:
            mean_val = 0.0
        else:
    
        command_term = env.command_manager.get_term(command_name)
        command_term.cfg.yaw_rate_range = (-yaw_limit, yaw_limit)
        if hasattr(command_term.cfg, "target_clearance"):
            command_term.cfg.target_clearance = target_clearance
        for reward_name in (lift_reward_name, yaw_reward_name, transition_reward_name):
            reward_cfg = env.reward_manager.get_term_cfg(reward_name)
            reward_cfg.params["target_clearance"] = target_clearance
            env.reward_manager.set_term_cfg(reward_name, reward_cfg)
    
        # Changes to phase rewards are linearly ramped, avoiding a discontinuous
        # value-target change when B/C opens.  `common_step_counter` advances once
        # per vectorized environment step, independent of how many envs reset.
        phase_weights = ((8.0, 2.0, 0.0, 0.0), (8.0, 2.0, -0.5, 0.0), (8.0, 2.0, -2.0, -2.0))[phase]
        phase_terms = (
            yaw_reward_name,
            transition_reward_name,
            spin_center_drift_reward_name,
            safe_recovery_reward_name,
        )
        ramp = 1.0 if reward_ramp_steps == 0 else min(
            1.0,
            max(0.0, (current_step - env._yaw_fsm_task_curriculum_reward_ramp_start_step) / reward_ramp_steps),
        )
        for reward_name, target_weight in zip(phase_terms, phase_weights):
            reward_cfg = env.reward_manager.get_term_cfg(reward_name)
            start_name = f"_yaw_fsm_task_curriculum_ramp_start_{reward_name}"
            if not hasattr(env, start_name):
                setattr(env, start_name, float(reward_cfg.weight))
            if current_step == env._yaw_fsm_task_curriculum_reward_ramp_start_step:
                setattr(env, start_name, float(reward_cfg.weight))
            reward_cfg.weight = float(getattr(env, start_name) + ramp * (target_weight - getattr(env, start_name)))
            if reward_name == yaw_reward_name:
                reward_cfg.params["lift_progress_threshold"] = lift_progress_threshold
                reward_cfg.params["support_threshold"] = support_threshold
                reward_cfg.params["clearance_gate_floor"] = 0.25 if phase == 1 else 0.0
                reward_cfg.params["clearance_gate_floor_decay_s"] = 3.0 if phase == 1 else 0.0
            env.reward_manager.set_term_cfg(reward_name, reward_cfg)
    
        force_limit = 10.0 * dr_scale
        force_cfg = env.event_manager.get_term_cfg("randomize_apply_external_force_torque")
    """CPU-only ownership tests for the yaw-FSM position anchor."""
    
    from __future__ import annotations
    
    import ast
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace
    
    import torch
    
    
    REPO_ROOT = Path(__file__).parents[1]
    FSM_PATH = (
        REPO_ROOT
        / "source"
        / "rl_training"
        / "rl_training"
        / "tasks"
        / "manager_based"
        / "locomotion"
        / "velocity"
        / "mdp"
        / "fsm.py"
    )
    FSM_GATES_PATH = FSM_PATH.with_name("fsm_gates.py")
    REWARDS_PATH = FSM_PATH.with_name("rewards.py")
    
    
    def _load_fsm_module():
        """Load the FSM module through its Isaac-Lab-independent fallback."""
        spec = importlib.util.spec_from_file_location("yaw_fsm_under_test", FSM_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    
    def _load_fsm_tracking_reward():
        """Load the real FSM pose rewards and gates without importing Isaac Sim."""
        fsm_module = _load_fsm_module()
        gate_tree = ast.parse(FSM_GATES_PATH.read_text(encoding="utf-8"))
        gate_names = {"_CACHE_STEP_ATTR", "_CACHE_ATTR", "_step_key", "_required_tensor", "fsm_gates"}
        gate_nodes = []
        for node in gate_tree.body:
            if isinstance(node, ast.Assign) and node.targets[0].id in gate_names:
                gate_nodes.append(node)
            elif isinstance(node, ast.FunctionDef) and node.name in gate_names:
                gate_nodes.append(node)
        namespace = {
            "torch": torch,
            "VQRFsmState": fsm_module.VQRFsmState,
            "select_swing_wheel_contact": fsm_module.select_swing_wheel_contact,
        }
        exec(compile(ast.Module(body=gate_nodes, type_ignores=[]), FSM_GATES_PATH, "exec"), namespace)
    
        reward_names = {
            "_yaw_wheel_contacts",
            "_yaw_support_load_quality",
            "_yaw_support_shape",
            "yaw_transition_support_load",
            "yaw_com_support",
            "yaw_com_inside_support_segment",
            "yaw_lift_clearance",
            "fsm_gated_tracking",
            "four_stand_ready_bonus",
        }
        reward_tree = ast.parse(REWARDS_PATH.read_text(encoding="utf-8"))
        reward_nodes = [
            node
            for node in reward_tree.body
            if (isinstance(node, ast.FunctionDef) and node.name in reward_names)
            or (isinstance(node, ast.ClassDef) and node.name == "ReturnToFourLanding")
        ]
        class FakeManagerTermBase:
            def __init__(self, cfg, env):
                pass
    
        class FakeSceneEntityCfg(SimpleNamespace):
            def __init__(self, name):
                super().__init__(name=name)
    
        namespace.update(
            {
                "ManagerBasedRLEnv": object,
                "SceneEntityCfg": FakeSceneEntityCfg,
                "ContactSensor": object,
                "RigidObject": object,
                "ManagerTermBase": FakeManagerTermBase,
                "RewTerm": object,
                "_fsm_step_telemetry": lambda *args: None,
                "_fsm_transition_telemetry": lambda *args: None,
                "_fsm_masked_accumulate": lambda *args: None,
                "_fsm_masked_accumulate_pair": lambda *args: None,
                "_fsm_attempt_telemetry": lambda *args: None,
                "_fsm_record_positive_budget": lambda *args: None,
            }
        )
        exec(compile(ast.Module(body=reward_nodes, type_ignores=[]), REWARDS_PATH, "exec"), namespace)
        namespace["_yaw_lift_progress"] = lambda env, cfg, *args: env.lift_progress[cfg.name]
        namespace["_yaw_support_geometry"] = lambda env, cfg: (
            torch.zeros(env.num_envs),
            torch.full((env.num_envs,), 0.5),
            torch.ones(env.num_envs),
        )
    ✓ • 0ms

## Activity

    file changes: Completed · 1 changes
    Update { move_path: None }: /home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py
    @@ -200,9 +200,5 @@
     ) -> torch.Tensor:
    -    """Penalize missing support load in TRANSITION; a held pose earns no income."""
    +    """Legacy transition term; support-load improvement is paid by TransitionProgress."""
         gates = fsm_gates(env, fsm_command_name)
    -    pos = _yaw_support_load_quality(env, sensor_cfg, target_force_n)
    -    neg = _yaw_support_load_quality(env, sensor_cfg_mirror, target_force_n)
    -    reward = (torch.where(gates["diag_pos"], pos, neg) - 1.0) * gates["f_trans"]
    -    _fsm_record_positive_budget(env, gates, "transition_support_load", reward)
    -    return reward
    +    return torch.zeros_like(gates["f_trans"])
     
    @@ -265,3 +261,3 @@
         gates = fsm_gates(env, fsm_command_name)
    -    reward = torch.where(gates["diag_pos"], score, mirror_score) * gates["f_geom"] - gates["f_trans"]
    +    reward = torch.where(gates["diag_pos"], score, mirror_score) * gates["f_yaw"]
         _fsm_record_positive_budget(env, gates, "com_support", reward)
    @@ -444,4 +440,4 @@
         improves the score, and both wheels must reach the target to score ``1``.
    -    In FSM TRANSITION, support load shapes the score before subtracting its
    -    maximum, leaving a nonpositive deficit. Raw metrics and YAW are unchanged.
    +    TRANSITION acquisition credit is paid once by TransitionProgress. Raw
    +    metrics and YAW keep their original signed score.
         """
    @@ -483,10 +479,3 @@
         selected_score = torch.where(gates["diag_pos"], score, mirror_score)
    -    support_pos = _yaw_support_load_quality(env, support_sensor_cfg, support_force_target_n)
    -    support_neg = _yaw_support_load_quality(env, support_sensor_cfg_mirror, support_force_target_n)
    -    support_quality = torch.where(gates["diag_pos"], support_pos, support_neg)
    -    lift_gate = transition_ungated_fraction + (1.0 - transition_ungated_fraction) * support_quality
    -    # Preserve the no-lift gradient before centering the transition score.
    -    # Only positive lift credit is reduced when support is weak.
    -    transition_score = torch.clamp(selected_score, max=0.0) + torch.relu(selected_score) * lift_gate
    -    reward = (transition_score - 1.0) * gates["f_trans"] + selected_score * gates["f_yaw"]
    +    reward = selected_score * gates["f_yaw"]
         _fsm_record_positive_budget(env, gates, "lift_clearance", reward)
    @@ -522,3 +511,3 @@
         gates = fsm_gates(env, fsm_command_name)
    -    reward = torch.where(gates["diag_pos"], score, mirror_score) * gates["f_geom"] - gates["f_trans"]
    +    reward = torch.where(gates["diag_pos"], score, mirror_score) * gates["f_yaw"]
         _fsm_record_positive_budget(env, gates, "com_inside_segment", reward)
    @@ -546,3 +535,3 @@
         gates = fsm_gates(env, fsm_command_name)
    -    reward = score * (1.0 - gates["f_safe"]) - gates["f_trans"]
    +    reward = score * (1.0 - gates["f_safe"] - gates["f_trans"])
         _fsm_record_positive_budget(env, gates, "balance", reward)
    @@ -1075,3 +1064,3 @@
         clearance_weight = floor + (1.0 - floor) * lift_progress
    -    state_gate = gates["f_trans"] + gates["f_yaw"]
    +    state_gate = gates["f_yaw"]
         tracking = support_gate * clearance_weight * yaw_tracking * state_gate
    @@ -1079,5 +1068,3 @@
         support_bonus = 0.25 * support_shape * lift_progress * state_gate
    -    # Center TRANSITION at its maximum: holding pose/tracking is never
    -    # positive income. YAW retains the original tracking and support bonus.
    -    reward = tracking + support_bonus - 1.25 * gates["f_trans"]
    +    reward = tracking + support_bonus
         _fsm_record_positive_budget(env, gates, "fsm_gated_tracking", reward)
    @@ -1108,7 +1095,6 @@
         gates = fsm_gates(env, fsm_command_name)
    -    # Transition retains a decaying attitude deficit, without tracking the
    -    # standing height. Height tracking remains active in FOUR_STAND/RETURN.
    +    # Height tracking remains active in FOUR_STAND/RETURN. Transition
    +    # acquisition is paid only for new progress by TransitionProgress.
         standing_gate = gates["f_four"] + gates["f_return"] * gates["tau"]
    -    transition_gate = gates["f_trans"] * (1.0 - gates["tau"])
    -    reward = (standing_gate * height_score + transition_gate) * attitude_score - transition_gate
    +    reward = standing_gate * height_score * attitude_score
         _fsm_record_positive_budget(env, gates, "four_stand_stability", reward)

## Activity

    $ git diff --stat; sed -n '1170,1250p' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py
     .../locomotion/velocity/mdp/rewards.py             | 40 +++++++---------------
     1 file changed, 13 insertions(+), 27 deletions(-)
            if env_ids is None:
                self.prev_phi.zero_()
                self.best_transition_progress.zero_()
            else:
                self.prev_phi[env_ids] = 0.0
                self.best_transition_progress[env_ids] = 0.0
    
        def __call__(
            self,
            env: ManagerBasedRLEnv,
            asset_cfg: SceneEntityCfg,
            asset_cfg_mirror: SceneEntityCfg,
            wheel_radius: float,
            target_clearance: float,
            fsm_command_name: str,
            gamma: float = 0.99,
            support_sensor_cfg: SceneEntityCfg | None = None,
            support_sensor_cfg_mirror: SceneEntityCfg | None = None,
            support_force_target_n: float = 80.0,
            transition_ungated_fraction: float = 0.25,
        ) -> torch.Tensor:
            if not 0.0 < gamma <= 1.0:
                raise ValueError("gamma must be in (0, 1].")
            if support_sensor_cfg is None or support_sensor_cfg_mirror is None:
                raise ValueError("Both support sensor configs are required for transition progress.")
            if not 0.0 < transition_ungated_fraction < 1.0:
                raise ValueError("transition_ungated_fraction must be in (0, 1).")
    
            gates = fsm_gates(env, fsm_command_name)
            progress_pos = _yaw_lift_progress(
                env, asset_cfg, wheel_radius, target_clearance
            ).mean(dim=1)
            progress_neg = _yaw_lift_progress(
                env, asset_cfg_mirror, wheel_radius, target_clearance
            ).mean(dim=1)
            progress = torch.where(gates["diag_pos"], progress_pos, progress_neg)
            phi = torch.where(gates["b_return"], 1.0 - progress, progress)
    
            # Always advance the potential, including FOUR/YAW/SAFE states.  This
            # prevents a spurious pulse when the next gated phase begins.
            reward = gamma * phi - self.prev_phi
            self.prev_phi.copy_(phi)
            reward = torch.where(gates["just_switched"], torch.zeros_like(reward), reward)
            # Credit a new lift maximum only once across TRANSITION/YAW chatter.
            # YAW/RETURN keep their previous reward behavior.
            active = gates["b_trans"] | gates["b_yaw"]
            self.best_transition_progress.masked_fill_(~active, 0.0)
            improvement = torch.relu(progress - self.best_transition_progress)
            transition_reward = improvement - (1.0 - gamma) * progress
            support_pos = _yaw_support_load_quality(env, support_sensor_cfg, support_force_target_n)
            support_neg = _yaw_support_load_quality(env, support_sensor_cfg_mirror, support_force_target_n)
            support_quality = torch.where(gates["diag_pos"], support_pos, support_neg)
            lift_gate = transition_ungated_fraction + (1.0 - transition_ungated_fraction) * support_quality
            transition_reward = torch.relu(transition_reward) * lift_gate + torch.clamp(transition_reward, max=0.0)
            self.best_transition_progress.copy_(torch.where(
                active, torch.maximum(self.best_transition_progress, progress),
                self.best_transition_progress,
            ))
            reward = transition_reward * gates["f_trans"] + reward * gates["f_return"]
            _fsm_record_positive_budget(env, gates, "transition_progress", reward)
            return reward
    
    
    def spin_center_drift(
        env: ManagerBasedRLEnv,
        fsm_command_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Penalize accumulated XY drift from the yaw-entry position."""
        asset: Articulation = env.scene[asset_cfg.name]
        current_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
        gates = fsm_gates(env, fsm_command_name)
        command = env.command_manager.get_term(fsm_command_name)
        entry_xy = command.yaw_entry_pos
        drift = torch.linalg.vector_norm(current_xy - entry_xy, dim=1)
        for suffix, direction_mask in (
            ("pos", gates["b_yaw"] & gates["diag_pos"]),
            ("neg", gates["b_yaw"] & gates["diag_neg"]),
        ):
            _fsm_masked_accumulate(
                env,
    ✓ • 0ms

## Activity

    file changes: Completed · 1 changes
    Update { move_path: None }: /home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py
    @@ -1161,3 +1161,3 @@
     class TransitionProgress(ManagerTermBase):
    -    """Bounded new lift credit in TRANSITION; potential-based shaping in RETURN."""
    +    """Pay bounded acquisition improvements once per episode and diagonal."""
     
    @@ -1166,3 +1166,5 @@
             self.prev_phi = torch.zeros(env.num_envs, device=env.device)
    -        self.best_transition_progress = torch.zeros(env.num_envs, device=env.device)
    +        # Columns are lift, support load, CoM line proximity, and segment fit.
    +        self.best_quality = torch.zeros(env.num_envs, 2, 4, device=env.device)
    +        self.seen_diagonal = torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.bool)
     
    @@ -1171,6 +1173,8 @@
                 self.prev_phi.zero_()
    -            self.best_transition_progress.zero_()
    +            self.best_quality.zero_()
    +            self.seen_diagonal.zero_()
             else:
                 self.prev_phi[env_ids] = 0.0
    -            self.best_transition_progress[env_ids] = 0.0
    +            self.best_quality[env_ids] = 0.0
    +            self.seen_diagonal[env_ids] = False
     
    @@ -1189,2 +1193,10 @@
             transition_ungated_fraction: float = 0.25,
    +        com_asset_cfg: SceneEntityCfg | None = None,
    +        com_asset_cfg_mirror: SceneEntityCfg | None = None,
    +        com_std: float = 0.08,
    +        segment_std: float = 0.05,
    +        lift_credit: float = 6.0,
    +        support_credit: float = 3.0,
    +        com_credit: float = 3.0,
    +        segment_credit: float = 2.0,
         ) -> torch.Tensor:
    @@ -1196,2 +1208,6 @@
                 raise ValueError("transition_ungated_fraction must be in (0, 1).")
    +        if com_std <= 0.0 or segment_std <= 0.0:
    +            raise ValueError("CoM and segment standard deviations must be positive.")
    +        if min(lift_credit, support_credit, com_credit, segment_credit) < 0.0:
    +            raise ValueError("Acquisition credits must be non-negative.")
     
    @@ -1212,8 +1228,5 @@
             reward = torch.where(gates["just_switched"], torch.zeros_like(reward), reward)
    -        # Credit a new lift maximum only once across TRANSITION/YAW chatter.
    -        # YAW/RETURN keep their previous reward behavior.
    +        # Seed from the first observed transition pose. Update best values in
    +        # YAW too, so leaving and re-entering TRANSITION cannot replay credit.
             active = gates["b_trans"] | gates["b_yaw"]
    -        self.best_transition_progress.masked_fill_(~active, 0.0)
    -        improvement = torch.relu(progress - self.best_transition_progress)
    -        transition_reward = improvement - (1.0 - gamma) * progress
             support_pos = _yaw_support_load_quality(env, support_sensor_cfg, support_force_target_n)
    @@ -1222,8 +1235,31 @@
             lift_gate = transition_ungated_fraction + (1.0 - transition_ungated_fraction) * support_quality
    -        transition_reward = torch.relu(transition_reward) * lift_gate + torch.clamp(transition_reward, max=0.0)
    -        self.best_transition_progress.copy_(torch.where(
    -            active, torch.maximum(self.best_transition_progress, progress),
    -            self.best_transition_progress,
    -        ))
    -        reward = transition_reward * gates["f_trans"] + reward * gates["f_return"]
    +        if com_asset_cfg is None or com_asset_cfg_mirror is None:
    +            raise ValueError("Both support body configs are required for acquisition credit.")
    +        pos_distance, pos_projection, pos_length = _yaw_support_geometry(env, com_asset_cfg)
    +        neg_distance, neg_projection, neg_length = _yaw_support_geometry(env, com_asset_cfg_mirror)
    +        distance = torch.where(gates["diag_pos"], pos_distance, neg_distance)
    +        projection = torch.where(gates["diag_pos"], pos_projection, neg_projection)
    +        length = torch.where(gates["diag_pos"], pos_length, neg_length)
    +        outside = (torch.relu(-projection) + torch.relu(projection - 1.0)) * length
    +        quality = torch.stack((
    +            progress * lift_gate,
    +            support_quality,
    +            1.0 / (1.0 + distance / com_std),
    +            torch.exp(-outside.square() / segment_std**2),
    +        ), dim=1)
    +        rows = torch.arange(env.num_envs, device=progress.device)
    +        diagonal = (~gates["diag_pos"]).long()
    +        previous = self.best_quality[rows, diagonal]
    +        first = active & ~self.seen_diagonal[rows, diagonal]
    +        improvement = torch.relu(quality - previous)
    +        improvement = torch.where(first.unsqueeze(1), torch.zeros_like(improvement), improvement)
    +        credits = quality.new_tensor((lift_credit, support_credit, com_credit, segment_credit))
    +        transition_reward = (improvement * credits).sum(dim=1)
    +        self.best_quality[rows, diagonal] = torch.where(
    +            active.unsqueeze(1), torch.maximum(previous, quality), previous
    +        )
    +        self.seen_diagonal[rows, diagonal] |= active
    +        # RewardManager multiplies every term by step_dt. Event credit and
    +        # failure cost are episode-scale quantities, independent of that dt.
    +        reward = transition_reward * gates["f_trans"] / env.step_dt + reward * gates["f_return"]
             _fsm_record_positive_budget(env, gates, "transition_progress", reward)
    @@ -1232,2 +1268,7 @@
     
    +def fsm_failure_cost(env: ManagerBasedRLEnv) -> torch.Tensor:
    +    """One cost for each failed episode, even if several unsafe masks fire."""
    +    return env.termination_manager.terminated.to(dtype=torch.float32) / env.step_dt
    +
    +
     def spin_center_drift(
    @@ -1276,3 +1317,3 @@
         gates = fsm_gates(env, fsm_command_name)
    -    return (gates["b_safe"] & gates["just_switched"]).to(dtype=torch.float32)
    +    return (gates["b_safe"] & gates["just_switched"]).to(dtype=torch.float32) / env.step_dt
     

## Activity

    file changes: Completed · 1 changes
    Update { move_path: None }: /home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py
    @@ -284,7 +284,6 @@
     class VQRWheelFSMRewardsCfg:
    -    """The 26-term reward contract for ``Flat-VQR-Wheel-Yaw-FSM``.
    +    """The reward contract for ``Flat-VQR-Wheel-Yaw-FSM``.
     
    -    Safety/regularization penalties retain their baseline weights. Positive
    -    pose terms become nonpositive deficits in TRANSITION; YAW and RETURN
    -    retain their rewards. POS/NEG select the command's ``support_diagonal``.
    +    Acquisition improvement is paid once per diagonal and episode. YAW and
    +    RETURN retain their pose rewards; failure has one episode-scale cost.
         """
    @@ -437,3 +436,3 @@
             func=mdp.yaw_transition_support_load,
    -        weight=3.0,
    +        weight=0.0,  # Replaced by the bounded support credit in transition_progress.
             params={
    @@ -594,3 +593,3 @@
             func=mdp.TransitionProgress,
    -        weight=2.0,
    +        weight=1.0,
             params={
    @@ -614,4 +613,14 @@
                 "transition_ungated_fraction": TRANSITION_LIFT_UNGATED_FRACTION,
    +            "com_asset_cfg": SceneEntityCfg(
    +                "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
    +            ),
    +            "com_asset_cfg_mirror": SceneEntityCfg(
    +                "robot", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
    +            ),
             },
         )
    +    fsm_failure = RewTerm(
    +        func=mdp.fsm_failure_cost,
    +        weight=-60.0,
    +    )
         spin_center_drift = RewTerm(
    @@ -626,3 +635,3 @@
             func=mdp.safe_recovery_entry,
    -        weight=0.0,  # Termination semantics in phases A/B; -2.0 in phase C.
    +        weight=0.0,  # Terminal cost in phases A/B; entry cost in phase C.
             params={"fsm_command_name": "yaw_rate_cmd"},

## Activity

    file changes: Completed · 1 changes
    Update { move_path: None }: /home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py
    @@ -1192,3 +1192,3 @@
         # per vectorized environment step, independent of how many envs reset.
    -    phase_weights = ((8.0, 2.0, 0.0, 0.0), (8.0, 2.0, -0.5, 0.0), (8.0, 2.0, -2.0, -2.0))[phase]
    +    phase_weights = ((8.0, 1.0, 0.0, 0.0), (8.0, 1.0, -0.5, 0.0), (8.0, 1.0, -2.0, -60.0))[phase]
         phase_terms = (

## Activity

    $ sed -n '840,870p' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py; rg -n '_yaw_fsm_positive_budget|positive_budget|transition_progress' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py tests/test_yaw_fsm_training_contract.py tests/test_yaw_fsm_systemic.py | head -100
        """
        try:
            weight = float(env.reward_manager.get_term_cfg(term_name).weight)
        except (AttributeError, KeyError, TypeError):
            return
        positive = torch.relu(value * weight)
        state = gates["fsm_state"]
        if not hasattr(env, "_yaw_fsm_positive_budget"):
            env._yaw_fsm_positive_budget = torch.zeros(
                env.num_envs, 7, dtype=positive.dtype, device=positive.device
            )
            env._yaw_fsm_positive_budget_pos = torch.zeros_like(positive)
            env._yaw_fsm_positive_budget_neg = torch.zeros_like(positive)
        env._yaw_fsm_positive_budget.scatter_add_(1, state.unsqueeze(1), positive.unsqueeze(1))
        env._yaw_fsm_positive_budget_pos += positive * gates["diag_pos"].to(positive.dtype)
        env._yaw_fsm_positive_budget_neg += positive * gates["diag_neg"].to(positive.dtype)
    
    
    def fsm_gated_tracking(
        env: ManagerBasedRLEnv,
        command_name: str,
        fsm_command_name: str,
        support_sensor_cfg: SceneEntityCfg,
        support_sensor_cfg_mirror: SceneEntityCfg,
        lifted_asset_cfg: SceneEntityCfg,
        lifted_asset_cfg_mirror: SceneEntityCfg,
        wheel_radius: float,
        target_clearance: float,
        std: float,
        contact_threshold: float = 1.0,
        clearance_gate_floor: float = 0.0,
    tests/test_yaw_fsm_training_contract.py:149:        "transition_progress",
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py:800:        "_yaw_fsm_telemetry_positive_budget": torch.zeros(7, dtype=torch.float32, device=env.device),
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py:801:        "_yaw_fsm_telemetry_positive_budget_pos": 0.0,
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py:802:        "_yaw_fsm_telemetry_positive_budget_neg": 0.0,
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py:975:        if hasattr(env, "_yaw_fsm_positive_budget"):
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py:976:            env._yaw_fsm_telemetry_positive_budget += env._yaw_fsm_positive_budget[
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py:979:            env._yaw_fsm_telemetry_positive_budget_pos += float(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py:980:                env._yaw_fsm_positive_budget_pos[completed_env_ids].sum().item()
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py:982:            env._yaw_fsm_telemetry_positive_budget_neg += float(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py:983:                env._yaw_fsm_positive_budget_neg[completed_env_ids].sum().item()
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py:1037:    if hasattr(env, "_yaw_fsm_positive_budget"):
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py:1038:        env._yaw_fsm_positive_budget[selected_env_ids] = 0.0
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py:1039:        env._yaw_fsm_positive_budget_pos[selected_env_ids] = 0.0
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py:1040:        env._yaw_fsm_positive_budget_neg[selected_env_ids] = 0.0
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py:1252:    budgets = env._yaw_fsm_telemetry_positive_budget / telemetry_steps
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py:1255:        env._yaw_fsm_telemetry_positive_budget_pos / telemetry_steps
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py:1258:        env._yaw_fsm_telemetry_positive_budget_neg / telemetry_steps
    ✓ • 0ms

## Activity

    $ sed -n '510,665p' tests/test_yaw_fsm.py; sed -n '1,150p' tests/test_yaw_transition_safety.py; sed -n '130,180p' tests/test_yaw_fsm_training_contract.py
        diagonal = torch.tensor([1, 1, -1, 0])
    
        selected = fsm_module.select_swing_wheel_contact(
            positive_support_contact,
            negative_support_contact,
            diagonal,
        )
    
        # POS swings the NEG pair, NEG swings the POS pair, and FOUR has no
        # selected swing pair. A single contacted wheel is sufficient.
        assert torch.equal(selected, torch.tensor([True, False, True, False]))
    
        mirrored = fsm_module.select_swing_wheel_contact(
            negative_support_contact,
            positive_support_contact,
            -diagonal,
        )
        assert torch.equal(mirrored, selected)
    
    
    def test_fsm_tracking_support_bonus_requires_lift_and_is_independent_of_yaw_error():
        rewards, state = _load_fsm_tracking_reward()
        tracking_reward = rewards["fsm_gated_tracking"]
        states = torch.tensor(
            [
                state.TRANSITION_POS,
                state.TRANSITION_POS,
                state.YAW_POS,
                state.YAW_NEG,
                state.FOUR_STAND,
                state.RETURN_TO_4,
                state.SAFE_RECOVERY,
                state.TRANSITION_NEG,
                state.YAW_POS,
                state.YAW_POS,
                state.YAW_NEG,
                state.YAW_NEG,
            ]
        )
        diagonal = torch.tensor([1, 1, 1, -1, 0, 1, 0, -1, 1, 1, -1, -1])
        forces = torch.zeros(12, 4, 3)
        forces[:, :, 2] = 2.0
        forces[2, 3, 2] = 0.0  # POS loses HR: partial bonus, no tracking.
        forces[9, [0, 3], 2] = 0.0  # POS loses both support wheels.
        forces[10, 2, 2] = 0.0  # NEG loses HL: mirror of case 2.
        forces[11, [1, 2], 2] = 0.0  # NEG loses both support wheels.
        lift = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        yaw_velocity = torch.tensor([0.0, 10.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
        command = SimpleNamespace(
            fsm_state=states,
            support_diagonal=diagonal,
            state_time=torch.zeros(12),
            just_switched=torch.zeros(12, dtype=torch.bool),
            cfg=SimpleNamespace(yaw_rate_range=(-1.0, 1.0)),
        )
        sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))
    
        class Scene(dict):
            sensors = {"contact_forces": sensor}
    
        root_ang_vel_b = torch.stack([torch.zeros(12), torch.zeros(12), yaw_velocity], dim=1)
        env = SimpleNamespace(
            num_envs=12,
            device="cpu",
            common_step_counter=0,
            scene=Scene(robot=SimpleNamespace(data=SimpleNamespace(root_ang_vel_b=root_ang_vel_b))),
            lift_progress={
                "lift_pos": lift[:, None].expand(-1, 2),
                "lift_neg": lift[:, None].expand(-1, 2),
            },
            command_manager=SimpleNamespace(
                get_term=lambda _: command,
                get_command=lambda _: torch.zeros(12, 1),
            ),
        )
        positive_cfg = SimpleNamespace(name="contact_forces", body_ids=torch.tensor([0, 3]))
        negative_cfg = SimpleNamespace(name="contact_forces", body_ids=torch.tensor([1, 2]))
    
        observed = tracking_reward(
            env,
            command_name="yaw_rate_cmd",
            fsm_command_name="yaw_rate_cmd",
            support_sensor_cfg=positive_cfg,
            support_sensor_cfg_mirror=negative_cfg,
            lifted_asset_cfg=SimpleNamespace(name="lift_pos"),
            lifted_asset_cfg_mirror=SimpleNamespace(name="lift_neg"),
            wheel_radius=0.091,
            target_clearance=0.05,
            std=0.30,
        )
        expected = torch.tensor([
            -1.25, -1.0, 0.0125, 0.25, 0.0, 0.0,
            0.0, -1.25, 1.25, 0.0, 0.0125, 0.0,
        ])
        assert torch.allclose(observed, expected, atol=1e-6)
        # With lift=1 and tracking either blocked or negligible, the +2 maximum
        # support bonus maps 0/1/2 support contacts to 0/0.05/1 respectively.
        assert torch.allclose(observed[[9, 2, 3]] * 4.0, torch.tensor([0.0, 0.05, 1.0]))
        assert torch.allclose(observed[[11, 10, 3]] * 4.0, torch.tensor([0.0, 0.05, 1.0]))
        assert env._yaw_fsm_pos_support_loss_max_steps[[2, 9]].tolist() == [1, 1]
        assert env._yaw_fsm_neg_support_loss_max_steps[[10, 11]].tolist() == [1, 1]
    
    
    def test_fsm_pose_rewards_keep_dense_yaw_signal_without_support_contact():
        rewards, state = _load_fsm_tracking_reward()
        n = 8
        command = SimpleNamespace(
            fsm_state=torch.tensor([
                state.YAW_POS, state.YAW_POS, state.YAW_POS,
                state.YAW_NEG, state.YAW_NEG, state.YAW_NEG,
                state.TRANSITION_POS, state.RETURN_TO_4,
            ]),
            support_diagonal=torch.tensor([1, 1, 1, -1, -1, -1, 1, 1]),
            state_time=torch.zeros(n),
            just_switched=torch.zeros(n, dtype=torch.bool),
        )
        lift_pos = torch.ones(n, 2)
        lift_neg = torch.ones(n, 2)
        lift_pos[2] = 0.0
        lift_neg[5] = 0.0
        forces = torch.zeros(n, 4, 3)
        forces[..., 2] = 80.0
        sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))
        env = SimpleNamespace(
            num_envs=n,
            device="cpu",
            common_step_counter=0,
            scene=SimpleNamespace(sensors={"contact_forces": sensor}),
            command_manager=SimpleNamespace(get_term=lambda _: command),
            lift_progress={"lift_pos": lift_pos, "lift_neg": lift_neg},
        )
        fsm_args = {"fsm_command_name": "yaw_rate_cmd"}
        geom_args = {
            "asset_cfg": SimpleNamespace(name="robot"),
            "asset_cfg_mirror": SimpleNamespace(name="robot"),
        }
        com = rewards["yaw_com_support"](env, std=0.08, **geom_args, **fsm_args)
        inside = rewards["yaw_com_inside_support_segment"](
            env, std=0.05, **geom_args, **fsm_args
        )
        expected_geom = torch.ones(n)
        expected_geom[-2:] = 0.0  # Perfect TRANSITION has no income; RETURN has no geometry.
        assert torch.allclose(com, expected_geom)
        assert torch.allclose(inside, expected_geom)
    
        lift = rewards["yaw_lift_clearance"](
            env,
            asset_cfg=SimpleNamespace(name="lift_pos"),
            asset_cfg_mirror=SimpleNamespace(name="lift_neg"),
            wheel_radius=0.091,
            target_clearance=0.05,
            support_sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=[0, 3]),
            support_sensor_cfg_mirror=SimpleNamespace(name="contact_forces", body_ids=[1, 2]),
            **fsm_args,
        )
    """CPU checks for the transition height warning and support-gated lift credit."""
    
    import ast
    from collections.abc import Sequence
    from types import SimpleNamespace
    
    import torch
    
    from test_yaw_fsm import REWARDS_PATH
    from test_yaw_fsm_systemic import _reward_scene
    
    TERMINATIONS_PATH = REWARDS_PATH.with_name("terminations.py")
    OBSERVATIONS_PATH = REWARDS_PATH.with_name("observations.py")
    
    
    def _transition_scene():
        rewards, state, env, command = _reward_scene()
        names = {"yaw_transition_low_base_height", "yaw_downward_low_base_velocity_l2"}
        nodes = [
            node for node in ast.parse(REWARDS_PATH.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(REWARDS_PATH), "exec"), rewards)
        env.scene["robot"].data.root_lin_vel_w = torch.zeros(2, 3)
        return rewards, state, env, command
    
    
    def test_transition_height_warning_is_monotone_and_state_gated():
        rewards, state, env, command = _transition_scene()
        cfg = SimpleNamespace(name="robot")
        args = dict(warning_height=.43, minimum_height=.35, fsm_command_name="yaw_rate_cmd", asset_cfg=cfg)
        samples = []
        for height in (.44, .43, .41, .39, .36, .351):
            env.scene["robot"].data.root_pos_w[:, 2] = height
            samples.append(rewards["yaw_transition_low_base_height"](env, **args))
        assert torch.equal(samples[0], torch.zeros(2))
        assert torch.equal(samples[1], torch.zeros(2))
        assert all((later > earlier).all() for earlier, later in zip(samples[1:-1], samples[2:]))
    
        command.fsm_state[:] = torch.tensor([state.YAW_POS, state.FOUR_STAND])
        env.common_step_counter += 1
        assert torch.equal(rewards["yaw_transition_low_base_height"](env, **args), torch.zeros(2))
    
    
    def test_transition_stability_does_not_track_standing_height():
        rewards, _, env, _ = _transition_scene()
        robot = env.scene["robot"].data
        robot.root_quat_w[:, 0] = .1  # The test double interprets this as roll.
        values = []
        for height in (.44, .49, .55):
            robot.root_pos_w[:, 2] = height
            values.append(rewards["four_stand_stability"](
                env, target_height=.49, fsm_command_name="yaw_rate_cmd"
            ))
        assert torch.allclose(values[0], values[1])
        assert torch.allclose(values[1], values[2])
        assert (values[0] < 0).all()
    
    
    def test_downward_speed_is_penalized_before_hard_height_failure():
        rewards, state, env, _ = _transition_scene()
        cfg = SimpleNamespace(name="robot")
        env.scene["robot"].data.root_lin_vel_w[:, 2] = -.5
        args = dict(minimum_height=.35, height_margin=.10, warning_height=.43,
                    fsm_command_name="yaw_rate_cmd", asset_cfg=cfg)
        env.scene["robot"].data.root_pos_w[:, 2] = .43
        assert torch.equal(rewards["yaw_downward_low_base_velocity_l2"](env, **args), torch.zeros(2))
        env.scene["robot"].data.root_pos_w[:, 2] = .39
        middle = rewards["yaw_downward_low_base_velocity_l2"](env, **args)
        env.scene["robot"].data.root_pos_w[:, 2] = .351
        near_boundary = rewards["yaw_downward_low_base_velocity_l2"](env, **args)
        assert torch.allclose(middle, torch.full((2,), .125), atol=1e-6)
        assert (near_boundary > middle).all()
        command = env.command_manager.get_term("yaw_rate_cmd")
        command.fsm_state[:] = torch.tensor([state.YAW_POS, state.FOUR_STAND])
        env.common_step_counter += 1
        assert torch.equal(rewards["yaw_downward_low_base_velocity_l2"](env, **args), torch.zeros(2))
    
    
    def test_lift_credit_uses_selected_support_diagonal_and_keeps_exploration():
        rewards, _, env, _ = _transition_scene()
        forces = env.scene.sensors["contact_forces"].data.net_forces_w
        # POS is FL/HR and NEG is FR/HL. Swap which pair is loaded in each env.
        forces.zero_()
        forces[0, [1, 2], 2] = 80.0
        forces[1, [0, 3], 2] = 80.0
        for progress in env.lift_progress.values():
            progress.fill_(1.0)
        support = dict(
            support_sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=[0, 3]),
            support_sensor_cfg_mirror=SimpleNamespace(name="contact_forces", body_ids=[1, 2]),
        )
        args = dict(asset_cfg=SimpleNamespace(name="pos"), asset_cfg_mirror=SimpleNamespace(name="neg"),
                    wheel_radius=.091, target_clearance=.05, fsm_command_name="yaw_rate_cmd", **support)
        clearance = rewards["yaw_lift_clearance"](env, **args, transition_ungated_fraction=.25)
        assert torch.allclose(clearance, torch.full((2,), -.75))
    
        progress_term = rewards["TransitionProgress"](None, env)
        credit = progress_term(env, **args, transition_ungated_fraction=.25)
        assert torch.allclose(credit, torch.full((2,), .2475))
        progress_term.reset()
        forces.zero_()
        forces[0, [0, 3], 2] = 80.0
        forces[1, [1, 2], 2] = 80.0
        env.common_step_counter += 1
        assert torch.equal(rewards["yaw_lift_clearance"](env, **args), torch.zeros(2))
        full_credit = progress_term(env, **args, transition_ungated_fraction=.25)
        assert torch.allclose(full_credit, torch.full((2,), .99))
    
    
    def test_height_and_tilt_failure_are_separate_termination_masks():
        node = next(
            node for node in ast.parse(TERMINATIONS_PATH.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == "FSMUnsafeWithGrace"
        )
        class ManagerTermBase:
            def __init__(self, cfg, env):
                pass
    
        masks = (torch.tensor([False, False]), torch.tensor([True, False]),
                 torch.tensor([False, True]))
        namespace = dict(torch=torch, ManagerTermBase=ManagerTermBase,
                         ManagerBasedRLEnv=object, SceneEntityCfg=object, Sequence=Sequence,
                         yaw_fsm_unsafe_components=lambda *args, **kwargs: masks)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(TERMINATIONS_PATH), "exec"), namespace)
        env = SimpleNamespace(num_envs=2, device="cpu", step_dt=.05)
        args = dict(robot_name="robot", torso_sensor_cfg=object(), threshold=1.0,
                    grace_period_s=0.0, minimum_base_height=.35, unsafe_angle_limit=.8)
        term = namespace["FSMUnsafeWithGrace"](None, env)
        assert torch.equal(term(env, **args, failure_kind="height"), masks[1])
        assert torch.equal(term(env, **args, failure_kind="tilt"), masks[2])
        assert torch.equal(term(env, **args), torch.tensor([True, True]))
    
    
    def test_unsafe_components_keep_height_and_tilt_thresholds_distinct():
        names = {"yaw_fsm_unsafe", "yaw_fsm_unsafe_components"}
        nodes = [
            node for node in ast.parse(OBSERVATIONS_PATH.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        namespace = dict(
            torch=torch, ManagerBasedEnv=object, SceneEntityCfg=object, Articulation=object,
            wheel_contact=lambda *args, **kwargs: torch.zeros(3, 1),
            euler_xyz_from_quat=lambda q: (q[:, 0], q[:, 1], q[:, 2]),
        )
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(OBSERVATIONS_PATH), "exec"), namespace)
        class Scene(dict):
            env_origins = torch.zeros(3, 3)
    
        env = SimpleNamespace(scene=Scene(robot=SimpleNamespace(data=SimpleNamespace(
        exec(compile(ast.Module(body=[verifier], type_ignores=[]), TRAIN_SCRIPT, "exec"), namespace)
    
        command = YawFSMCommand()
        for name in (
            "fsm_state",
            "support_diagonal",
            "state_time",
            "transition_time",
            "just_switched",
            "just_returned_to_four",
            "positive_pose_ready",
            "negative_pose_ready",
            "four_stand_ready",
            "unsafe",
            "yaw_entry_pos",
        ):
            setattr(command, name, object())
        required = {
            "fsm_gated_tracking",
            "transition_progress",
            "transition_support_load",
            "transition_low_base_height",
            "return_to_four_landing",
            "four_stand_ready_bonus",
            "spin_center_drift",
            "safe_recovery_entry",
        }
        active_terms = sorted(required | {f"term_{index}" for index in range(18)})
        reward_manager = SimpleNamespace(
            active_terms=active_terms,
            get_term_cfg=lambda _: SimpleNamespace(weight=8.0),
        )
        task_env = SimpleNamespace(
            reward_manager=reward_manager,
            command_manager=SimpleNamespace(get_term=lambda _: command),
        )
        env = SimpleNamespace(unwrapped=task_env)
    
        namespace["_verify_yaw_fsm_contract"](env, SimpleNamespace())
    
        reward_manager.active_terms = active_terms[:-1]
        with pytest.raises(RuntimeError, match="expected 26 terms"):
            namespace["_verify_yaw_fsm_contract"](env, SimpleNamespace())
    ✓ • 0ms

## Activity

    $ rg -n 'expected 26|transition_support_load|len\(reward_manager.active_terms|fsm_failure' scripts source/rl_training/rl_training/tasks/manager_based/locomotion/velocity tests/test_yaw_fsm_training_contract.py | head -100
    tests/test_yaw_fsm_training_contract.py:66:    assert "transition_support_load" in fsm
    tests/test_yaw_fsm_training_contract.py:69:    assert ast.literal_eval(_keyword(fsm["transition_support_load"], "weight")) == 3.0
    tests/test_yaw_fsm_training_contract.py:150:        "transition_support_load",
    tests/test_yaw_fsm_training_contract.py:171:    with pytest.raises(RuntimeError, match="expected 26 terms"):
    scripts/reinforcement_learning/rsl_rl/play.py:382:        reward_diagnostic_sum = torch.zeros(len(reward_manager.active_terms), device=env.unwrapped.device)
    scripts/reinforcement_learning/rsl_rl/train.py:345:        "transition_support_load",
    scripts/reinforcement_learning/rsl_rl/train.py:381:            "Unexpected Flat-VQR-Wheel-Yaw-FSM reward config: expected 26 terms, "
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:194:def yaw_transition_support_load(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:1269:def fsm_failure_cost(env: ManagerBasedRLEnv) -> torch.Tensor:
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:435:    transition_support_load = RewTerm(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:436:        func=mdp.yaw_transition_support_load,
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:622:    fsm_failure = RewTerm(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:623:        func=mdp.fsm_failure_cost,
    ✓ • 0ms

## Activity

    $ sed -n '330,390p' scripts/reinforcement_learning/rsl_rl/train.py; sed -n '35,90p' tests/test_yaw_fsm_training_contract.py; sed -n '660,700p' tests/test_yaw_fsm.py
            raise RuntimeError("Stale yaw reward config detected: support_contact must not be an active reward term.")
        if len(term_names) != 18 or yaw_weight != 8.0:
            raise RuntimeError(
                "Unexpected Flat-VQR-Wheel-Yaw reward config: expected 18 terms and gated_yaw_tracking weight 8.0."
            )
    
    
    def _verify_yaw_fsm_contract(env, env_cfg) -> None:
        """Enforce the FSM task contract before constructing the PPO runner."""
        task_env = env.unwrapped
        reward_manager = task_env.reward_manager
        term_names = list(reward_manager.active_terms)
        required_reward_terms = {
            "fsm_gated_tracking",
            "transition_progress",
            "transition_support_load",
            "transition_low_base_height",
            "return_to_four_landing",
            "four_stand_ready_bonus",
            "spin_center_drift",
            "safe_recovery_entry",
        }
        missing_reward_terms = sorted(required_reward_terms.difference(term_names))
        tracking_weight = (
            float(reward_manager.get_term_cfg("fsm_gated_tracking").weight)
            if "fsm_gated_tracking" in term_names
            else float("nan")
        )
        command = task_env.command_manager.get_term("yaw_rate_cmd")
        required_command_buffers = (
            "fsm_state",
            "support_diagonal",
            "state_time",
            "transition_time",
            "just_switched",
            "just_returned_to_four",
            "positive_pose_ready",
            "negative_pose_ready",
            "four_stand_ready",
            "unsafe",
            "yaw_entry_pos",
        )
        missing_command_buffers = [
            name for name in required_command_buffers if not hasattr(command, name)
        ]
    
        print(f"[INFO] Flat-VQR-Wheel-Yaw-FSM env config source: {inspect.getfile(type(env_cfg))}")
        print(f"[INFO] Flat-VQR-Wheel-Yaw-FSM reward terms ({len(term_names)}): {term_names}")
        print(f"[INFO] Flat-VQR-Wheel-Yaw-FSM fsm_gated_tracking weight: {tracking_weight}")
        if len(term_names) != 26 or missing_reward_terms or tracking_weight != 8.0:
            raise RuntimeError(
                "Unexpected Flat-VQR-Wheel-Yaw-FSM reward config: expected 26 terms, "
                "fsm_gated_tracking weight 8.0, and all required FSM reward terms; "
                f"missing={missing_reward_terms}."
            )
        if not isinstance(command, YawFSMCommand) or missing_command_buffers:
            raise RuntimeError(
                "Unexpected Flat-VQR-Wheel-Yaw-FSM command contract: expected YawFSMCommand "
                f"with all public FSM buffers; missing={missing_command_buffers}."
            )
    
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name
        )
    
    
    def _reward_assignments(path: Path, class_name: str) -> dict[str, ast.Call]:
        node = _class_node(path, class_name)
        return {
            item.targets[0].id: item.value
            for item in node.body
            if isinstance(item, ast.Assign)
            and isinstance(item.targets[0], ast.Name)
            and isinstance(item.value, ast.Call)
            and isinstance(item.value.func, ast.Name)
            and item.value.func.id == "RewTerm"
        }
    
    
    def _keyword(call: ast.Call, name: str) -> ast.expr:
        return next(keyword.value for keyword in call.keywords if keyword.arg == name)
    
    
    def test_fsm_balance_is_gated_and_reward_counts_are_isolated():
        baseline = _reward_assignments(BASELINE_CONFIG, "VQRWheelRewardsCfg")
        fsm = _reward_assignments(FSM_CONFIG, "VQRWheelFSMRewardsCfg")
    
        assert len(baseline) == 18
        assert len(fsm) == 26
        assert "transition_low_base_height" in fsm
        assert "support_contact" not in fsm
        assert "transition_support_load" in fsm
        assert "return_to_four_landing" in fsm
        assert "four_stand_ready_bonus" in fsm
        assert ast.literal_eval(_keyword(fsm["transition_support_load"], "weight")) == 3.0
    
        fsm_balance_params = ast.literal_eval(_keyword(fsm["balance"], "params"))
        baseline_balance_params = ast.literal_eval(_keyword(baseline["balance"], "params"))
        assert fsm_balance_params["fsm_command_name"] == "yaw_rate_cmd"
        assert "fsm_command_name" not in baseline_balance_params
    
    
    def test_fsm_registration_uses_its_own_runner_and_experiment_directory():
        runner = _class_node(RUNNER_CONFIG, "VQRWheelYawFlatFSMPPORunnerCfg")
        assert [base.id for base in runner.bases if isinstance(base, ast.Name)] == [
            "VQRWheelYawFlatPPORunnerCfg"
        ]
        post_init = next(
            item for item in runner.body if isinstance(item, ast.FunctionDef) and item.name == "__post_init__"
        )
        experiment_assignment = next(
            item
            for item in post_init.body
            if isinstance(item, ast.Assign)
            and isinstance(item.targets[0], ast.Attribute)
            and item.targets[0].attr == "experiment_name"
            wheel_radius=0.091,
            target_clearance=0.05,
            support_sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=[0, 3]),
            support_sensor_cfg_mirror=SimpleNamespace(name="contact_forces", body_ids=[1, 2]),
            **fsm_args,
        )
        expected_lift = torch.tensor([1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 0.0])
        assert torch.allclose(lift, expected_lift)
    
        # The legacy branch does not inspect FSM contact and retains exact values.
        legacy_com = rewards["yaw_com_support"](env, geom_args["asset_cfg"], std=0.08)
        legacy_inside = rewards["yaw_com_inside_support_segment"](
            env, geom_args["asset_cfg"], std=0.05
        )
        legacy_lift = rewards["yaw_lift_clearance"](
            env, SimpleNamespace(name="lift_pos"), wheel_radius=0.091, target_clearance=0.05
        )
        assert torch.equal(legacy_com, torch.ones(n))
        assert torch.equal(legacy_inside, torch.ones(n))
        assert torch.equal(legacy_lift, 2.0 * lift_pos.mean(dim=1) - 1.0)
    
    
    def test_transition_support_load_uses_normal_force_and_mirrors_diagonals():
        rewards, state = _load_fsm_tracking_reward()
        states = torch.tensor([
            state.TRANSITION_POS, state.TRANSITION_NEG,
            state.YAW_POS, state.RETURN_TO_4, state.TRANSITION_POS,
        ])
        command = SimpleNamespace(
            fsm_state=states,
            support_diagonal=torch.tensor([1, -1, 1, -1, 1]),
            state_time=torch.zeros(5),
            just_switched=torch.zeros(5, dtype=torch.bool),
        )
        forces = torch.zeros(5, 4, 3)
        forces[0, 0, 2], forces[0, 3, 2] = 80.0, 40.0  # POS: FL + HR
        forces[1, 1, 2], forces[1, 2, 2] = 80.0, 40.0  # NEG: FR + HL
        forces[2, [0, 3], 2] = 80.0
        forces[3, [1, 2], 2] = 80.0
        forces[4, 0, 2] = 80.0
        forces[4, 3, 0] = 500.0  # Large tangential force is not normal support.
    ✓ • 0ms

## Activity

    file changes: Completed · 1 changes
    Update { move_path: None }: /home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py
    @@ -434,16 +434,2 @@
         )
    -    transition_support_load = RewTerm(
    -        func=mdp.yaw_transition_support_load,
    -        weight=0.0,  # Replaced by the bounded support credit in transition_progress.
    -        params={
    -            "sensor_cfg": SceneEntityCfg(
    -                "contact_forces", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
    -            ),
    -            "sensor_cfg_mirror": SceneEntityCfg(
    -                "contact_forces", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
    -            ),
    -            "fsm_command_name": "yaw_rate_cmd",
    -            "target_force_n": SUPPORT_LOAD_TARGET_N,
    -        },
    -    )
         lift_clearance = RewTerm(

## Activity

    file changes: Completed · 1 changes
    Update { move_path: None }: /home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py
    @@ -193,14 +193,2 @@
     
    -def yaw_transition_support_load(
    -    env: ManagerBasedRLEnv,
    -    sensor_cfg: SceneEntityCfg,
    -    sensor_cfg_mirror: SceneEntityCfg,
    -    fsm_command_name: str,
    -    target_force_n: float,
    -) -> torch.Tensor:
    -    """Legacy transition term; support-load improvement is paid by TransitionProgress."""
    -    gates = fsm_gates(env, fsm_command_name)
    -    return torch.zeros_like(gates["f_trans"])
    -
    -
     def _yaw_support_shape(contacts: torch.Tensor) -> torch.Tensor:

## Activity

    file changes: Completed · 1 changes
    Update { move_path: None }: /home/robotics/tuanpm48/vqr/rl_training/scripts/reinforcement_learning/rsl_rl/train.py
    @@ -344,3 +344,3 @@
             "transition_progress",
    -        "transition_support_load",
    +        "fsm_failure",
             "transition_low_base_height",

## Activity

    $ rg -n 'termination_manager\.terminated|termination_manager\.time_outs|termination_manager\.dones' source tests scripts | head -80
    source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/mdp/to_transition.py:124:        return good & ~self._env.termination_manager.terminated[ids]
    source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/mdp/to_transition.py:138:        extra["fall_rate"] = self._env.termination_manager.terminated[ids].float().mean().item()
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:1259:    return env.termination_manager.terminated.to(dtype=torch.float32) / env.step_dt
    ✓ • 0ms

## Activity

    $ sed -n '680,745p' tests/test_yaw_fsm.py; sed -n '118,150p' tests/test_yaw_fsm_systemic.py
    
    
    def test_transition_support_load_uses_normal_force_and_mirrors_diagonals():
        rewards, state = _load_fsm_tracking_reward()
        states = torch.tensor([
            state.TRANSITION_POS, state.TRANSITION_NEG,
            state.YAW_POS, state.RETURN_TO_4, state.TRANSITION_POS,
        ])
        command = SimpleNamespace(
            fsm_state=states,
            support_diagonal=torch.tensor([1, -1, 1, -1, 1]),
            state_time=torch.zeros(5),
            just_switched=torch.zeros(5, dtype=torch.bool),
        )
        forces = torch.zeros(5, 4, 3)
        forces[0, 0, 2], forces[0, 3, 2] = 80.0, 40.0  # POS: FL + HR
        forces[1, 1, 2], forces[1, 2, 2] = 80.0, 40.0  # NEG: FR + HL
        forces[2, [0, 3], 2] = 80.0
        forces[3, [1, 2], 2] = 80.0
        forces[4, 0, 2] = 80.0
        forces[4, 3, 0] = 500.0  # Large tangential force is not normal support.
        sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))
        env = SimpleNamespace(
            num_envs=5, device="cpu", common_step_counter=0,
            scene=SimpleNamespace(sensors={"contact_forces": sensor}),
            command_manager=SimpleNamespace(get_term=lambda _: command),
        )
        reward = rewards["yaw_transition_support_load"](
            env,
            sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=[0, 3]),
            sensor_cfg_mirror=SimpleNamespace(name="contact_forces", body_ids=[1, 2]),
            fsm_command_name="yaw_rate_cmd",
            target_force_n=80.0,
        )
        assert torch.allclose(reward, torch.tensor([-0.45, -0.45, 0.0, 0.0, -0.90]))
    
    
    def test_transition_lift_deficit_improves_with_support_without_positive_income():
        rewards, state = _load_fsm_tracking_reward()
        command = SimpleNamespace(
            fsm_state=torch.tensor([
                state.TRANSITION_POS, state.TRANSITION_NEG,
                state.TRANSITION_POS, state.TRANSITION_NEG,
                state.TRANSITION_POS, state.YAW_NEG, state.TRANSITION_POS,
            ]),
            support_diagonal=torch.tensor([1, -1, 1, -1, 1, -1, 1]),
            state_time=torch.zeros(7),
            just_switched=torch.zeros(7, dtype=torch.bool),
        )
        forces = torch.zeros(7, 4, 3)
        forces[2, [0, 3], 2] = 80.0
        forces[3, [1, 2], 2] = 80.0
        forces[4, 0, 2], forces[4, 3, 2] = 80.0, 40.0
        sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))
        lift_pos = torch.ones(7, 2)
        lift_neg = torch.ones(7, 2)
        lift_pos[6] = 0.0
        env = SimpleNamespace(
            num_envs=7, device="cpu", common_step_counter=0,
            scene=SimpleNamespace(sensors={"contact_forces": sensor}),
            command_manager=SimpleNamespace(get_term=lambda _: command),
            lift_progress={"pos": lift_pos, "neg": lift_neg},
        )
        reward = rewards["yaw_lift_clearance"](
            env,
            asset_cfg=SimpleNamespace(name="pos"),
        assert (step(.5, trans) <= 0).all()
        assert torch.equal(step(.8, yaw), torch.zeros(2))
        assert (step(.8, trans) <= 0).all()
        assert (step(1., trans) > 0).all()
        progress_reward.reset(torch.tensor([0]))
        result = step(.5, trans)
        assert result[0] > 0 and result[1] <= 0
    
    
    def test_return_progress_and_stability_keep_their_original_values():
        rewards, state, env, command = _reward_scene()
        progress_reward = rewards["TransitionProgress"](None, env)
        args = dict(asset_cfg=SimpleNamespace(name="pos"), asset_cfg_mirror=SimpleNamespace(name="neg"),
                    wheel_radius=.091, target_clearance=.05, fsm_command_name="yaw_rate_cmd",
                    support_sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=[0, 3]),
                    support_sensor_cfg_mirror=SimpleNamespace(name="contact_forces", body_ids=[1, 2]))
        command.fsm_state.fill_(state.RETURN_TO_4)
        for progress in env.lift_progress.values():
            progress.fill_(1.)
        assert torch.equal(progress_reward(env, **args), torch.zeros(2))
        command.just_switched.fill_(False)
        command.state_time.fill_(.5)
        env.common_step_counter += 1
        for progress in env.lift_progress.values():
            progress.fill_(.5)
        assert torch.allclose(progress_reward(env, **args), torch.full((2,), .495))
        stability = rewards["four_stand_stability"](env, target_height=.49, fsm_command_name="yaw_rate_cmd")
        assert torch.equal(stability, torch.full((2,), .5))
        command.fsm_state.fill_(state.FOUR_STAND)
        env.common_step_counter += 1
        assert torch.equal(progress_reward(env, **args), torch.zeros(2))
        assert torch.equal(rewards["four_stand_stability"](
            env, target_height=.49, fsm_command_name="yaw_rate_cmd"), torch.ones(2))
    ✓ • 0ms

## Activity

    $ sed -n '745,795p' tests/test_yaw_fsm.py
            asset_cfg=SimpleNamespace(name="pos"),
            asset_cfg_mirror=SimpleNamespace(name="neg"),
            support_sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=[0, 3]),
            support_sensor_cfg_mirror=SimpleNamespace(name="contact_forces", body_ids=[1, 2]),
            wheel_radius=0.091, target_clearance=0.05,
            fsm_command_name="yaw_rate_cmd",
            support_force_target_n=80.0,
            transition_ungated_fraction=0.35,
        )
        assert torch.allclose(reward, torch.tensor([-0.65, -0.65, 0.0, 0.0, -0.2925, 1.0, -2.0]))
        assert torch.equal(env._yaw_lift_min_progress_sum, torch.tensor([1., 1., 1., 1., 1., 1., 0.]))
    
    
    def test_return_landing_and_completion_bonus_are_mirrored_one_time_events():
        rewards, state = _load_fsm_tracking_reward()
        command = SimpleNamespace(
            fsm_state=torch.tensor([state.RETURN_TO_4, state.RETURN_TO_4]),
            support_diagonal=torch.tensor([1, -1]),
            state_time=torch.zeros(2),
            just_switched=torch.ones(2, dtype=torch.bool),
            just_returned_to_four=torch.zeros(2, dtype=torch.bool),
            cfg=SimpleNamespace(target_clearance=0.05),
        )
        forces = torch.zeros(2, 4, 3)
        sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))
        env = SimpleNamespace(
            num_envs=2,
            device="cpu",
            common_step_counter=0,
            scene=SimpleNamespace(sensors={"contact_forces": sensor}),
            command_manager=SimpleNamespace(get_term=lambda _: command),
            lift_progress={"pos": torch.ones(2, 2), "neg": torch.ones(2, 2)},
        )
        term = rewards["ReturnToFourLanding"](None, env)
        args = {
            "asset_cfg": SimpleNamespace(name="pos"),
            "asset_cfg_mirror": SimpleNamespace(name="neg"),
            "sensor_cfg": SimpleNamespace(name="contact_forces", body_ids=[1, 2]),
            "sensor_cfg_mirror": SimpleNamespace(name="contact_forces", body_ids=[0, 3]),
            "wheel_radius": 0.091,
            "fsm_command_name": "yaw_rate_cmd",
        }
        assert torch.equal(term(env, **args), torch.zeros(2))
    
        command.just_switched.fill_(False)
        env.common_step_counter += 1
        env.lift_progress["pos"][:] = 0.5
        env.lift_progress["neg"][:] = 0.5
        assert torch.allclose(term(env, **args), torch.full((2,), 0.5))
    
        env.common_step_counter += 1
    ✓ • 0ms

## Assistant

Phần reward chính đã được đổi: các term tư thế không còn trừ đều mỗi bước TRANSITION; `transition_progress` theo dõi mức tốt nhất riêng cho POS/NEG và chỉ trả credit khi vượt mốc đó. Mình đang cập nhật cost thất bại và test contract vì task này kiểm tra chính xác danh sách 26 reward term trước khi train.

## Activity

    file changes: Completed · 1 changes
    Update { move_path: None }: /home/robotics/tuanpm48/vqr/rl_training/tests/test_yaw_fsm_systemic.py
    @@ -42,3 +42,3 @@
         env = SimpleNamespace(
    -        num_envs=2, device="cpu", common_step_counter=0,
    +        num_envs=2, device="cpu", step_dt=.02, common_step_counter=0,
             scene=Scene(robot=SimpleNamespace(data=SimpleNamespace(
    @@ -65,3 +65,2 @@
             2 * rewards["yaw_com_inside_support_segment"](env, std=.05, **geom, **fsm),
    -        3 * rewards["yaw_transition_support_load"](env, target_force_n=80, **support, **fsm),
             3 * rewards["yaw_lift_clearance"](env, **lift,
    @@ -78,3 +77,3 @@
     @pytest.mark.parametrize("clearance", [0., .5, .79, 1.])
    -def test_stalled_transition_has_no_positive_pose_income_and_handoff_does_not_drop(clearance):
    +def test_stalled_transition_has_no_pose_income_and_handoff_does_not_drop(clearance):
         rewards, state, env, command = _reward_scene()
    @@ -86,3 +85,3 @@
             transition = _pose_rewards(rewards, env)
    -        assert (transition <= 0).all()
    +        assert torch.equal(transition, torch.zeros_like(transition))
             assert torch.allclose(transition[:, 0], transition[:, 1])
    @@ -102,3 +101,5 @@
                     support_sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=[0, 3]),
    -                support_sensor_cfg_mirror=SimpleNamespace(name="contact_forces", body_ids=[1, 2]))
    +                support_sensor_cfg_mirror=SimpleNamespace(name="contact_forces", body_ids=[1, 2]),
    +                com_asset_cfg=SimpleNamespace(name="robot"),
    +                com_asset_cfg_mirror=SimpleNamespace(name="robot"))
     
    @@ -113,2 +114,3 @@
         yaw = [state.YAW_POS, state.YAW_NEG]
    +    assert torch.equal(step(0., trans), torch.zeros(2))
         assert (step(.5, trans) > 0).all()
    @@ -117,5 +119,5 @@
         assert (step(.2, trans) <= 0).all()
    -    assert (step(.5, trans) <= 0).all()
    +    assert torch.equal(step(.5, trans), torch.zeros(2))
         assert torch.equal(step(.8, yaw), torch.zeros(2))
    -    assert (step(.8, trans) <= 0).all()
    +    assert torch.equal(step(.8, trans), torch.zeros(2))
         assert (step(1., trans) > 0).all()
    @@ -123,3 +125,3 @@
         result = step(.5, trans)
    -    assert result[0] > 0 and result[1] <= 0
    +    assert result[0] == 0 and result[1] == 0
     
    @@ -132,3 +134,5 @@
                     support_sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=[0, 3]),
    -                support_sensor_cfg_mirror=SimpleNamespace(name="contact_forces", body_ids=[1, 2]))
    +                support_sensor_cfg_mirror=SimpleNamespace(name="contact_forces", body_ids=[1, 2]),
    +                com_asset_cfg=SimpleNamespace(name="robot"),
    +                com_asset_cfg_mirror=SimpleNamespace(name="robot"))
         command.fsm_state.fill_(state.RETURN_TO_4)

## Activity

    file changes: Completed · 1 changes
    Update { move_path: None }: /home/robotics/tuanpm48/vqr/rl_training/tests/test_yaw_fsm_systemic.py
    @@ -126,2 +126,4 @@
         assert result[0] == 0 and result[1] == 0
    +    result = step(.75, trans)
    +    assert result[0] > 0 and result[1] == 0
     

## Activity

    file changes: Completed · 1 changes
    Update { move_path: None }: /home/robotics/tuanpm48/vqr/rl_training/tests/test_yaw_transition_safety.py
    @@ -56,3 +56,3 @@
         assert torch.allclose(values[1], values[2])
    -    assert (values[0] < 0).all()
    +    assert torch.equal(values[0], torch.zeros(2))
     
    @@ -87,3 +87,3 @@
         for progress in env.lift_progress.values():
    -        progress.fill_(1.0)
    +        progress.zero_()
         support = dict(
    @@ -93,10 +93,13 @@
         args = dict(asset_cfg=SimpleNamespace(name="pos"), asset_cfg_mirror=SimpleNamespace(name="neg"),
    -                wheel_radius=.091, target_clearance=.05, fsm_command_name="yaw_rate_cmd", **support)
    -    clearance = rewards["yaw_lift_clearance"](env, **args, transition_ungated_fraction=.25)
    -    assert torch.allclose(clearance, torch.full((2,), -.75))
    +                wheel_radius=.091, target_clearance=.05, fsm_command_name="yaw_rate_cmd",
    +                com_asset_cfg=SimpleNamespace(name="robot"),
    +                com_asset_cfg_mirror=SimpleNamespace(name="robot"), **support)
     
         progress_term = rewards["TransitionProgress"](None, env)
    +    assert torch.equal(progress_term(env, **args, transition_ungated_fraction=.25), torch.zeros(2))
    +    for progress in env.lift_progress.values():
    +        progress.fill_(1.0)
    +    env.common_step_counter += 1
         credit = progress_term(env, **args, transition_ungated_fraction=.25)
    -    assert torch.allclose(credit, torch.full((2,), .2475))
    -    progress_term.reset()
    +    assert torch.allclose(credit, torch.full((2,), 75.0))
         forces.zero_()
    @@ -105,5 +108,6 @@
         env.common_step_counter += 1
    -    assert torch.equal(rewards["yaw_lift_clearance"](env, **args), torch.zeros(2))
    +    clearance_args = {key: value for key, value in args.items() if not key.startswith("com_asset")}
    +    assert torch.equal(rewards["yaw_lift_clearance"](env, **clearance_args), torch.zeros(2))
         full_credit = progress_term(env, **args, transition_ungated_fraction=.25)
    -    assert torch.allclose(full_credit, torch.full((2,), .99))
    +    assert torch.allclose(full_credit, torch.full((2,), 375.0))
     

## Activity

    file changes: Completed · 1 changes
    Update { move_path: None }: /home/robotics/tuanpm48/vqr/rl_training/tests/test_yaw_fsm.py
    @@ -60,3 +60,2 @@
             "_yaw_support_shape",
    -        "yaw_transition_support_load",
             "yaw_com_support",
    @@ -601,4 +600,4 @@
         expected = torch.tensor([
    -        -1.25, -1.0, 0.0125, 0.25, 0.0, 0.0,
    -        0.0, -1.25, 1.25, 0.0, 0.0125, 0.0,
    +        0.0, 0.0, 0.0125, 0.25, 0.0, 0.0,
    +        0.0, 0.0, 1.25, 0.0, 0.0125, 0.0,
         ])
    @@ -681,3 +680,3 @@
     
    -def test_transition_support_load_uses_normal_force_and_mirrors_diagonals():
    +def test_acquisition_support_quality_uses_normal_force_and_mirrors_diagonals():
         rewards, state = _load_fsm_tracking_reward()
    @@ -706,13 +705,13 @@
         )
    -    reward = rewards["yaw_transition_support_load"](
    -        env,
    -        sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=[0, 3]),
    -        sensor_cfg_mirror=SimpleNamespace(name="contact_forces", body_ids=[1, 2]),
    -        fsm_command_name="yaw_rate_cmd",
    -        target_force_n=80.0,
    +    pos = rewards["_yaw_support_load_quality"](
    +        env, SimpleNamespace(name="contact_forces", body_ids=[0, 3]), 80.0
    +    )
    +    neg = rewards["_yaw_support_load_quality"](
    +        env, SimpleNamespace(name="contact_forces", body_ids=[1, 2]), 80.0
         )
    -    assert torch.allclose(reward, torch.tensor([-0.45, -0.45, 0.0, 0.0, -0.90]))
    +    selected = torch.where(command.support_diagonal > 0, pos, neg)
    +    assert torch.allclose(selected, torch.tensor([.55, .55, 1.0, 1.0, .10]))
     
     
    -def test_transition_lift_deficit_improves_with_support_without_positive_income():
    +def test_transition_lift_has_no_per_step_deficit():
         rewards, state = _load_fsm_tracking_reward()
    @@ -753,3 +752,3 @@
         )
    -    assert torch.allclose(reward, torch.tensor([-0.65, -0.65, 0.0, 0.0, -0.2925, 1.0, -2.0]))
    +    assert torch.allclose(reward, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
         assert torch.equal(env._yaw_lift_min_progress_sum, torch.tensor([1., 1., 1., 1., 1., 1., 0.]))

## Activity

    file changes: Completed · 1 changes
    Update { move_path: None }: /home/robotics/tuanpm48/vqr/rl_training/tests/test_yaw_fsm_training_contract.py
    @@ -65,6 +65,8 @@
         assert "support_contact" not in fsm
    -    assert "transition_support_load" in fsm
    +    assert "transition_support_load" not in fsm
    +    assert "fsm_failure" in fsm
         assert "return_to_four_landing" in fsm
         assert "four_stand_ready_bonus" in fsm
    -    assert ast.literal_eval(_keyword(fsm["transition_support_load"], "weight")) == 3.0
    +    assert ast.literal_eval(_keyword(fsm["fsm_failure"], "weight")) == -60.0
    +    assert ast.literal_eval(_keyword(fsm["transition_progress"], "weight")) == 1.0
     
    @@ -149,3 +151,3 @@
             "transition_progress",
    -        "transition_support_load",
    +        "fsm_failure",
             "transition_low_base_height",

## Activity

    $ rg -n 'transition_support_load|best_transition_progress|safe_recovery_entry|transition_progress.*2\.0|deficit|\.2475|\.99\)' tests source/rl_training/rl_training/tasks/manager_based/locomotion/velocity scripts/reinforcement_learning/rsl_rl/train.py | head -150
    scripts/reinforcement_learning/rsl_rl/train.py:138:    "_yaw_fsm_task_curriculum_ramp_start_safe_recovery_entry",
    scripts/reinforcement_learning/rsl_rl/train.py:283:        term_params.get("safe_recovery_reward_name", "safe_recovery_entry"),
    scripts/reinforcement_learning/rsl_rl/train.py:350:        "safe_recovery_entry",
    tests/test_yaw_fsm_training_contract.py:66:    assert "transition_support_load" not in fsm
    tests/test_yaw_fsm_training_contract.py:157:        "safe_recovery_entry",
    tests/test_yaw_curriculum.py:290:            "transition_progress": reward_cfg(2.0),
    tests/test_yaw_curriculum.py:292:            "safe_recovery_entry": reward_cfg(0.0),
    tests/test_yaw_curriculum.py:522:        "safe_recovery_reward_name": "safe_recovery_entry",
    tests/test_yaw_curriculum.py:547:            "safe_recovery_entry": SimpleNamespace(weight=0.0, params={}),
    tests/test_yaw_curriculum.py:579:        "_yaw_fsm_task_curriculum_ramp_start_transition_progress": 2.0,
    tests/test_yaw_curriculum.py:581:        "_yaw_fsm_task_curriculum_ramp_start_safe_recovery_entry": 0.0,
    tests/test_yaw_curriculum.py:614:    assert resumed_env.reward_manager.get_term_cfg("transition_progress").weight == pytest.approx(2.0)
    tests/test_yaw_curriculum.py:616:    assert resumed_env.reward_manager.get_term_cfg("safe_recovery_entry").weight == pytest.approx(
    tests/test_yaw_fsm.py:716:def test_transition_lift_has_no_per_step_deficit():
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py:687:    safe_recovery_reward_name: str = "safe_recovery_entry",
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:341:        # height cost is the potential change, with no time-at-height deficit.
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:1330:def safe_recovery_entry(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:620:    safe_recovery_entry = RewTerm(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:621:        func=mdp.safe_recovery_entry,
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:676:            "safe_recovery_reward_name": "safe_recovery_entry",
    ✓ • 0ms

## Activity

    $ sed -n '320,350p' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py; sed -n '270,305p' tests/test_yaw_curriculum.py; sed -n '530,625p' tests/test_yaw_curriculum.py
                self.was_transition[env_ids] = False
    
        def __call__(
            self,
            env: ManagerBasedRLEnv,
            safe_height: float,
            minimum_height: float,
            fsm_command_name: str,
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ) -> torch.Tensor:
            if safe_height <= minimum_height:
                raise ValueError("safe_height must exceed minimum_height.")
            asset: Articulation = env.scene[asset_cfg.name]
            base_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
            margin = ((base_height - minimum_height) / (safe_height - minimum_height)).clamp(0.0, 1.0)
            # Concave progress gives an earlier signal and a steeper loss near the
            # unsafe boundary, while staying bounded in [0, 1].
            potential = 1.5 * margin - 0.5 * margin.square()
            transition = fsm_gates(env, fsm_command_name)["b_trans"]
            continuing = transition & self.was_transition
            # RewardManager multiplies by step_dt; cancel it so the episode-scale
            # height cost is the potential change, with no time-at-height deficit.
            reward = torch.where(
                continuing, (potential - self.prev_potential) / env.step_dt,
                torch.zeros_like(potential),
            )
            self.prev_potential.copy_(potential)
            self.was_transition.copy_(transition)
            return reward
    
    
        curriculum = _load_curriculums_module(monkeypatch)
    
        class ConfigManager:
            def __init__(self, configs):
                self.configs = configs
    
            def get_term_cfg(self, name):
                return self.configs[name]
    
            def set_term_cfg(self, name, cfg):
                self.configs[name] = cfg
    
        def reward_cfg(weight=1.0):
            return SimpleNamespace(weight=weight, params={"target_clearance": 0.05})
    
        command = SimpleNamespace(cfg=SimpleNamespace(yaw_rate_range=(-0.25, 0.25), target_clearance=0.05))
        rewards = ConfigManager(
            {
                "lift_clearance": reward_cfg(3.0),
                "fsm_gated_tracking": reward_cfg(8.0),
                "transition_progress": reward_cfg(2.0),
                "spin_center_drift": reward_cfg(0.0),
                "safe_recovery_entry": reward_cfg(0.0),
            }
        )
        events = ConfigManager(
            {
                "randomize_apply_external_force_torque": SimpleNamespace(
                    params={"force_range": (0.0, 0.0), "torque_range": (0.0, 0.0)}
                ),
                "randomize_actuator_gains": SimpleNamespace(
                    params={
                        "stiffness_distribution_params": (1.0, 1.0),
                        "damping_distribution_params": (1.0, 1.0),
                    }
                ),
    def _fsm_checkpoint_env(curriculum, common_step_counter: int):
        command = _ResettableCommand()
        rewards = _ConfigManager(
            {
                "lift_clearance": SimpleNamespace(weight=3.0, params={"target_clearance": 0.05}),
                "fsm_gated_tracking": SimpleNamespace(
                    weight=8.0,
                    params={
                        "target_clearance": 0.05,
                        "clearance_gate_floor": 0.0,
                        "clearance_gate_floor_decay_s": 0.0,
                    },
                ),
                "transition_progress": SimpleNamespace(
                    weight=2.0, params={"target_clearance": 0.05}
                ),
                "spin_center_drift": SimpleNamespace(weight=0.0, params={}),
                "safe_recovery_entry": SimpleNamespace(weight=0.0, params={}),
            }
        )
        params = _fsm_curriculum_params()
        term_cfg = SimpleNamespace(func=curriculum.yaw_fsm_task_levels, params=params)
        return SimpleNamespace(
            common_step_counter=common_step_counter,
            num_envs=3,
            device="cpu",
            step_dt=0.02,
            episode_length_buf=torch.zeros(3, dtype=torch.long),
            command_manager=SimpleNamespace(get_term=lambda _: command),
            reward_manager=rewards,
            event_manager=_online_dr_manager(),
            cfg=SimpleNamespace(curriculum=SimpleNamespace(task_levels=term_cfg)),
        )
    
    
    def test_fsm_checkpoint_restores_phase_c_timing_and_reapplies_config(
        monkeypatch: pytest.MonkeyPatch,
    ):
        curriculum = _load_curriculums_module(monkeypatch)
        helpers = _load_train_curriculum_helpers()
        saved_env = _fsm_checkpoint_env(curriculum, common_step_counter=10_000)
        saved_values = {
            "_yaw_fsm_task_curriculum_phase": 2,
            "_yaw_task_curriculum_stage": 3,
            "_yaw_task_curriculum_yaw_stage": 2,
            "_yaw_task_curriculum_consecutive_passes": 2,
            "_yaw_fsm_task_curriculum_stage_start_step": 9_000,
            "_yaw_fsm_task_curriculum_reward_ramp_start_step": 9_400,
            "_yaw_fsm_task_curriculum_ramp_start_fsm_gated_tracking": 8.0,
            "_yaw_fsm_task_curriculum_ramp_start_transition_progress": 2.0,
            "_yaw_fsm_task_curriculum_ramp_start_spin_center_drift": -0.5,
            "_yaw_fsm_task_curriculum_ramp_start_safe_recovery_entry": 0.0,
        }
        for name, value in saved_values.items():
            setattr(saved_env, name, value)
    
        payload = helpers._export_yaw_fsm_curriculum_state(saved_env)
        assert payload["stage_elapsed_steps"] == 1_000
        assert payload["reward_ramp_elapsed_steps"] == 600
        assert not any(isinstance(value, torch.Tensor) for value in payload["values"].values())
    
        resumed_env = _fsm_checkpoint_env(curriculum, common_step_counter=200)
        restored = helpers._restore_yaw_fsm_curriculum_state(
            resumed_env, {"yaw_fsm_curriculum": payload}
        )
    
        assert restored is True
        assert resumed_env._yaw_fsm_task_curriculum_phase == 2
        assert resumed_env._yaw_task_curriculum_stage == 3
        assert resumed_env._yaw_task_curriculum_yaw_stage == 2
        assert resumed_env._yaw_task_curriculum_consecutive_passes == 2
        assert resumed_env._yaw_fsm_task_curriculum_stage_start_step == -800
        assert resumed_env._yaw_fsm_task_curriculum_reward_ramp_start_step == -400
    
        command = resumed_env.command_manager.get_term("yaw_rate_cmd")
        assert command.cfg.yaw_rate_range == pytest.approx((-0.55, 0.55))
        assert command.cfg.target_clearance == pytest.approx(0.20)
        assert torch.equal(command.reset_env_ids, torch.arange(3))
        for reward_name in ("lift_clearance", "fsm_gated_tracking", "transition_progress"):
            assert resumed_env.reward_manager.get_term_cfg(reward_name).params[
                "target_clearance"
            ] == pytest.approx(0.20)
    
        assert resumed_env.reward_manager.get_term_cfg("fsm_gated_tracking").weight == pytest.approx(8.0)
        assert resumed_env.reward_manager.get_term_cfg("transition_progress").weight == pytest.approx(2.0)
        assert resumed_env.reward_manager.get_term_cfg("spin_center_drift").weight == pytest.approx(-0.75)
        assert resumed_env.reward_manager.get_term_cfg("safe_recovery_entry").weight == pytest.approx(
            -1.0 / 3.0
        )
        tracking_params = resumed_env.reward_manager.get_term_cfg("fsm_gated_tracking").params
        assert tracking_params["clearance_gate_floor"] == 0.0
        assert tracking_params["clearance_gate_floor_decay_s"] == 0.0
    
        force_cfg = resumed_env.event_manager.get_term_cfg("randomize_apply_external_force_torque")
        assert force_cfg.params["force_range"] == pytest.approx((-5.5, 5.5))
        gain_cfg = resumed_env.event_manager.get_term_cfg("randomize_actuator_gains")
    ✓ • 0ms

## Activity

    $ rg -n 'TransitionHeightMargin|transition_height_margin|yaw_transition_low_base_height' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py tests | head -100
    ✓ • 0ms

## Activity

    $ sed -n '260,355p' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py; git diff --stat
        fsm_command_name: str | None = None,
    ) -> torch.Tensor:
        """Track TORSO height with non-saturating, normalized Huber shaping.
    
        The raw score is ``1 - huber(abs(height - target) / error_scale)``.
        It is maximal at the finite target, quadratic nearby, and linear rather
        than exponentially flat when the torso is far from the target.
        """
        if error_scale <= 0.0:
            raise ValueError("error_scale must be positive.")
    
        asset: Articulation = env.scene[asset_cfg.name]
        base_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        if not hasattr(env, "_yaw_base_height_min"):
            env._yaw_base_height_min = torch.full_like(base_height, torch.inf)
        env._yaw_base_height_min = torch.minimum(env._yaw_base_height_min, base_height)
    
        normalized_error = torch.abs(base_height - target_height) / error_scale
        huber = torch.where(
            normalized_error <= 1.0,
            0.5 * normalized_error.square(),
            normalized_error - 0.5,
        )
        score = 1.0 - huber
        if fsm_command_name is None:
            return score
        gates = fsm_gates(env, fsm_command_name)
        reward = score * gates["f_yaw"]
        _fsm_record_positive_budget(env, gates, "base_height", reward)
        return reward
    
    
    def yaw_low_base_height_l1(
        env: ManagerBasedRLEnv,
        minimum_height: float,
        error_scale: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Return normalized L1 violation below the minimum safe base height."""
        if error_scale <= 0.0:
            raise ValueError("error_scale must be positive.")
        asset: Articulation = env.scene[asset_cfg.name]
        base_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        return torch.relu(minimum_height - base_height) / error_scale
    
    
    class TransitionHeightProgress(ManagerTermBase):
        """Pay changes in a bounded safety margin only during TRANSITION."""
    
        def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
            super().__init__(cfg, env)
            self.prev_potential = torch.zeros(env.num_envs, device=env.device)
            self.was_transition = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    
        def reset(self, env_ids=None) -> None:
            if env_ids is None:
                self.prev_potential.zero_()
                self.was_transition.zero_()
            else:
                self.prev_potential[env_ids] = 0.0
                self.was_transition[env_ids] = False
    
        def __call__(
            self,
            env: ManagerBasedRLEnv,
            safe_height: float,
            minimum_height: float,
            fsm_command_name: str,
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ) -> torch.Tensor:
            if safe_height <= minimum_height:
                raise ValueError("safe_height must exceed minimum_height.")
            asset: Articulation = env.scene[asset_cfg.name]
            base_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
            margin = ((base_height - minimum_height) / (safe_height - minimum_height)).clamp(0.0, 1.0)
            # Concave progress gives an earlier signal and a steeper loss near the
            # unsafe boundary, while staying bounded in [0, 1].
            potential = 1.5 * margin - 0.5 * margin.square()
            transition = fsm_gates(env, fsm_command_name)["b_trans"]
            continuing = transition & self.was_transition
            # RewardManager multiplies by step_dt; cancel it so the episode-scale
            # height cost is the potential change, with no time-at-height deficit.
            reward = torch.where(
                continuing, (potential - self.prev_potential) / env.step_dt,
                torch.zeros_like(potential),
            )
            self.prev_potential.copy_(potential)
            self.was_transition.copy_(transition)
            return reward
    
    
    def yaw_downward_low_base_velocity_l2(
        env: ManagerBasedRLEnv,
        minimum_height: float,
        height_margin: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
     scripts/reinforcement_learning/rsl_rl/train.py     |   2 +-
     .../config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py    |  45 +++--
     .../locomotion/velocity/mdp/curriculums.py         |   2 +-
     .../locomotion/velocity/mdp/rewards.py             | 181 +++++++++++++--------
     tests/test_yaw_fsm.py                              |  25 ++-
     tests/test_yaw_fsm_systemic.py                     |  24 ++-
     tests/test_yaw_fsm_training_contract.py            |   8 +-
     tests/test_yaw_transition_safety.py                |  90 ++++++----
     8 files changed, 223 insertions(+), 154 deletions(-)
    ✓ • 0ms

## Activity

    $ git diff -- source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py | head -230; git diff -- tests/test_yaw_transition_safety.py | head -200
    diff --git a/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py b/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py
    index eb42a62..2fc46ca 100755
    --- a/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py
    +++ b/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py
    @@ -191,22 +191,6 @@ def _yaw_support_load_quality(
         return 0.8 * per_wheel.amin(dim=1) + 0.2 * per_wheel.mean(dim=1)
    
    
    -def yaw_transition_support_load(
    -    env: ManagerBasedRLEnv,
    -    sensor_cfg: SceneEntityCfg,
    -    sensor_cfg_mirror: SceneEntityCfg,
    -    fsm_command_name: str,
    -    target_force_n: float,
    -) -> torch.Tensor:
    -    """Penalize missing support load in TRANSITION; a held pose earns no income."""
    -    gates = fsm_gates(env, fsm_command_name)
    -    pos = _yaw_support_load_quality(env, sensor_cfg, target_force_n)
    -    neg = _yaw_support_load_quality(env, sensor_cfg_mirror, target_force_n)
    -    reward = (torch.where(gates["diag_pos"], pos, neg) - 1.0) * gates["f_trans"]
    -    _fsm_record_positive_budget(env, gates, "transition_support_load", reward)
    -    return reward
    -
    -
     def _yaw_support_shape(contacts: torch.Tensor) -> torch.Tensor:
         """Give nearly all support credit only when both selected wheels contact."""
         c1, c2 = contacts.to(dtype=torch.float32).unbind(dim=1)
    @@ -263,7 +247,7 @@ def yaw_com_support(
         mirror_distance, _, _ = _yaw_support_geometry(env, asset_cfg_mirror)
         mirror_score = 1.0 / (1.0 + mirror_distance / std)
         gates = fsm_gates(env, fsm_command_name)
    -    reward = torch.where(gates["diag_pos"], score, mirror_score) * gates["f_geom"] - gates["f_trans"]
    +    reward = torch.where(gates["diag_pos"], score, mirror_score) * gates["f_yaw"]
         _fsm_record_positive_budget(env, gates, "com_support", reward)
         return reward
    
    @@ -319,20 +303,49 @@ def yaw_low_base_height_l1(
         return torch.relu(minimum_height - base_height) / error_scale
    
    
    -def yaw_transition_low_base_height(
    -    env: ManagerBasedRLEnv,
    -    warning_height: float,
    -    minimum_height: float,
    -    fsm_command_name: str,
    -    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    -) -> torch.Tensor:
    -    """Soft collapse warning in TRANSITION, before the hard height boundary."""
    -    if warning_height <= minimum_height:
    -        raise ValueError("warning_height must exceed minimum_height.")
    -    asset: Articulation = env.scene[asset_cfg.name]
    -    base_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    -    proximity = ((warning_height - base_height) / (warning_height - minimum_height)).clamp(0.0, 1.0)
    -    return proximity.square() * fsm_gates(env, fsm_command_name)["f_trans"]
    +class TransitionHeightProgress(ManagerTermBase):
    +    """Pay changes in a bounded safety margin only during TRANSITION."""
    +
    +    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
    +        super().__init__(cfg, env)
    +        self.prev_potential = torch.zeros(env.num_envs, device=env.device)
    +        self.was_transition = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    +
    +    def reset(self, env_ids=None) -> None:
    +        if env_ids is None:
    +            self.prev_potential.zero_()
    +            self.was_transition.zero_()
    +        else:
    +            self.prev_potential[env_ids] = 0.0
    +            self.was_transition[env_ids] = False
    +
    +    def __call__(
    +        self,
    +        env: ManagerBasedRLEnv,
    +        safe_height: float,
    +        minimum_height: float,
    +        fsm_command_name: str,
    +        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    +    ) -> torch.Tensor:
    +        if safe_height <= minimum_height:
    +            raise ValueError("safe_height must exceed minimum_height.")
    +        asset: Articulation = env.scene[asset_cfg.name]
    +        base_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    +        margin = ((base_height - minimum_height) / (safe_height - minimum_height)).clamp(0.0, 1.0)
    +        # Concave progress gives an earlier signal and a steeper loss near the
    +        # unsafe boundary, while staying bounded in [0, 1].
    +        potential = 1.5 * margin - 0.5 * margin.square()
    +        transition = fsm_gates(env, fsm_command_name)["b_trans"]
    +        continuing = transition & self.was_transition
    +        # RewardManager multiplies by step_dt; cancel it so the episode-scale
    +        # height cost is the potential change, with no time-at-height deficit.
    +        reward = torch.where(
    +            continuing, (potential - self.prev_potential) / env.step_dt,
    +            torch.zeros_like(potential),
    +        )
    +        self.prev_potential.copy_(potential)
    +        self.was_transition.copy_(transition)
    +        return reward
    
    
     def yaw_downward_low_base_velocity_l2(
    @@ -355,7 +368,8 @@ def yaw_downward_low_base_velocity_l2(
         if fsm_command_name is not None:
             if warning_height is None or warning_height <= minimum_height:
                 raise ValueError("warning_height must exceed minimum_height for FSM gating.")
    -        warning_gate = ((warning_height - base_height) / (warning_height - minimum_height)).clamp(0.0, 1.0)
    +        proximity = ((warning_height - base_height) / (warning_height - minimum_height)).clamp(0.0, 1.0)
    +        warning_gate = proximity.square() * (3.0 - 2.0 * proximity)
             low_gate = torch.maximum(low_gate, warning_gate * fsm_gates(env, fsm_command_name)["f_trans"])
         downward_speed = torch.relu(-asset.data.root_lin_vel_w[:, 2])
         return low_gate * downward_speed.square()
    @@ -442,8 +456,8 @@ def yaw_lift_clearance(
         Each selected wheel contributes independently. The baseline score is in
         ``[-1, 1]``: both wheels on the ground score ``-1``, lifting either wheel
         improves the score, and both wheels must reach the target to score ``1``.
    -    In FSM TRANSITION, support load shapes the score before subtracting its
    -    maximum, leaving a nonpositive deficit. Raw metrics and YAW are unchanged.
    +    TRANSITION acquisition credit is paid once by TransitionProgress. Raw
    +    metrics and YAW keep their original signed score.
         """
         progress = _yaw_lift_progress(env, asset_cfg, wheel_radius, target_clearance)
    
    @@ -481,14 +495,7 @@ def yaw_lift_clearance(
         env._yaw_lift_min_progress_sum += min_progress
         env._yaw_lift_min_progress_samples += 1
         selected_score = torch.where(gates["diag_pos"], score, mirror_score)
    -    support_pos = _yaw_support_load_quality(env, support_sensor_cfg, support_force_target_n)
    -    support_neg = _yaw_support_load_quality(env, support_sensor_cfg_mirror, support_force_target_n)
    -    support_quality = torch.where(gates["diag_pos"], support_pos, support_neg)
    -    lift_gate = transition_ungated_fraction + (1.0 - transition_ungated_fraction) * support_quality
    -    # Preserve the no-lift gradient before centering the transition score.
    -    # Only positive lift credit is reduced when support is weak.
    -    transition_score = torch.clamp(selected_score, max=0.0) + torch.relu(selected_score) * lift_gate
    -    reward = (transition_score - 1.0) * gates["f_trans"] + selected_score * gates["f_yaw"]
    +    reward = selected_score * gates["f_yaw"]
         _fsm_record_positive_budget(env, gates, "lift_clearance", reward)
         return reward
    
    @@ -520,7 +527,7 @@ def yaw_com_inside_support_segment(
         ) * mirror_segment_length
         mirror_score = torch.exp(-mirror_outside_distance.square() / std**2)
         gates = fsm_gates(env, fsm_command_name)
    -    reward = torch.where(gates["diag_pos"], score, mirror_score) * gates["f_geom"] - gates["f_trans"]
    +    reward = torch.where(gates["diag_pos"], score, mirror_score) * gates["f_yaw"]
         _fsm_record_positive_budget(env, gates, "com_inside_segment", reward)
         return reward
    
    @@ -544,7 +551,7 @@ def yaw_balance(
         if fsm_command_name is None:
             return score
         gates = fsm_gates(env, fsm_command_name)
    -    reward = score * (1.0 - gates["f_safe"]) - gates["f_trans"]
    +    reward = score * (1.0 - gates["f_safe"] - gates["f_trans"])
         _fsm_record_positive_budget(env, gates, "balance", reward)
         return reward
    
    @@ -1073,13 +1080,11 @@ def fsm_gated_tracking(
         else:
             floor = clearance_gate_floor
         clearance_weight = floor + (1.0 - floor) * lift_progress
    -    state_gate = gates["f_trans"] + gates["f_yaw"]
    +    state_gate = gates["f_yaw"]
         tracking = support_gate * clearance_weight * yaw_tracking * state_gate
         # Contact alone gives no bonus while the swing pair is still on the ground.
         support_bonus = 0.25 * support_shape * lift_progress * state_gate
    -    # Center TRANSITION at its maximum: holding pose/tracking is never
    -    # positive income. YAW retains the original tracking and support bonus.
    -    reward = tracking + support_bonus - 1.25 * gates["f_trans"]
    +    reward = tracking + support_bonus
         _fsm_record_positive_budget(env, gates, "fsm_gated_tracking", reward)
         return reward
    
    @@ -1106,11 +1111,10 @@ def four_stand_stability(
         height_score = torch.exp(-(base_height - target_height).square() / height_std**2)
         attitude_score = torch.exp(-(roll.square() + pitch.square()) / attitude_std**2)
         gates = fsm_gates(env, fsm_command_name)
    -    # Transition retains a decaying attitude deficit, without tracking the
    -    # standing height. Height tracking remains active in FOUR_STAND/RETURN.
    +    # Height tracking remains active in FOUR_STAND/RETURN. Transition
    +    # acquisition is paid only for new progress by TransitionProgress.
         standing_gate = gates["f_four"] + gates["f_return"] * gates["tau"]
    -    transition_gate = gates["f_trans"] * (1.0 - gates["tau"])
    -    reward = (standing_gate * height_score + transition_gate) * attitude_score - transition_gate
    +    reward = standing_gate * height_score * attitude_score
         _fsm_record_positive_budget(env, gates, "four_stand_stability", reward)
         return reward
    
    @@ -1173,20 +1177,24 @@ def four_stand_ready_bonus(env: ManagerBasedRLEnv, fsm_command_name: str) -> tor
    
    
     class TransitionProgress(ManagerTermBase):
    -    """Bounded new lift credit in TRANSITION; potential-based shaping in RETURN."""
    +    """Pay bounded acquisition improvements once per episode and diagonal."""
    
         def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
             super().__init__(cfg, env)
             self.prev_phi = torch.zeros(env.num_envs, device=env.device)
    -        self.best_transition_progress = torch.zeros(env.num_envs, device=env.device)
    +        # Columns are lift, support load, CoM line proximity, and segment fit.
    +        self.best_quality = torch.zeros(env.num_envs, 2, 4, device=env.device)
    +        self.seen_diagonal = torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.bool)
    
         def reset(self, env_ids=None) -> None:
             if env_ids is None:
                 self.prev_phi.zero_()
    -            self.best_transition_progress.zero_()
    +            self.best_quality.zero_()
    +            self.seen_diagonal.zero_()
             else:
                 self.prev_phi[env_ids] = 0.0
    -            self.best_transition_progress[env_ids] = 0.0
    +            self.best_quality[env_ids] = 0.0
    +            self.seen_diagonal[env_ids] = False
    
         def __call__(
             self,
    @@ -1201,6 +1209,14 @@ class TransitionProgress(ManagerTermBase):
             support_sensor_cfg_mirror: SceneEntityCfg | None = None,
             support_force_target_n: float = 80.0,
             transition_ungated_fraction: float = 0.25,
    +        com_asset_cfg: SceneEntityCfg | None = None,
    +        com_asset_cfg_mirror: SceneEntityCfg | None = None,
    +        com_std: float = 0.08,
    +        segment_std: float = 0.05,
    +        lift_credit: float = 6.0,
    +        support_credit: float = 3.0,
    +        com_credit: float = 3.0,
    +        segment_credit: float = 2.0,
         ) -> torch.Tensor:
             if not 0.0 < gamma <= 1.0:
                 raise ValueError("gamma must be in (0, 1].")
    diff --git a/tests/test_yaw_transition_safety.py b/tests/test_yaw_transition_safety.py
    index 6de4e8f..c1747b7 100644
    --- a/tests/test_yaw_transition_safety.py
    +++ b/tests/test_yaw_transition_safety.py
    @@ -1,4 +1,4 @@
    -"""CPU checks for the transition height warning and support-gated lift credit."""
    +"""CPU checks for transition height progress and support-gated lift credit."""
    
     import ast
     from collections.abc import Sequence
    @@ -15,31 +15,50 @@ OBSERVATIONS_PATH = REWARDS_PATH.with_name("observations.py")
    
     def _transition_scene():
         rewards, state, env, command = _reward_scene()
    -    names = {"yaw_transition_low_base_height", "yaw_downward_low_base_velocity_l2"}
    +    names = {"TransitionHeightProgress", "yaw_downward_low_base_velocity_l2"}
         nodes = [
             node for node in ast.parse(REWARDS_PATH.read_text()).body
    -        if isinstance(node, ast.FunctionDef) and node.name in names
    +        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
         ]
         exec(compile(ast.Module(body=nodes, type_ignores=[]), str(REWARDS_PATH), "exec"), rewards)
         env.scene["robot"].data.root_lin_vel_w = torch.zeros(2, 3)
    +    env.step_dt = .02
         return rewards, state, env, command
    
    
    -def test_transition_height_warning_is_monotone_and_state_gated():
    +def test_transition_height_progress_is_bounded_monotone_and_reversible():
         rewards, state, env, command = _transition_scene()
         cfg = SimpleNamespace(name="robot")
    -    args = dict(warning_height=.43, minimum_height=.35, fsm_command_name="yaw_rate_cmd", asset_cfg=cfg)
    -    samples = []
    -    for height in (.44, .43, .41, .39, .36, .351):
    -        env.scene["robot"].data.root_pos_w[:, 2] = height
    -        samples.append(rewards["yaw_transition_low_base_height"](env, **args))
    -    assert torch.equal(samples[0], torch.zeros(2))
    -    assert torch.equal(samples[1], torch.zeros(2))
    -    assert all((later > earlier).all() for earlier, later in zip(samples[1:-1], samples[2:]))
    -
    -    command.fsm_state[:] = torch.tensor([state.YAW_POS, state.FOUR_STAND])
    -    env.common_step_counter += 1
    -    assert torch.equal(rewards["yaw_transition_low_base_height"](env, **args), torch.zeros(2))
    +    args = dict(safe_height=.50, minimum_height=.35, fsm_command_name="yaw_rate_cmd", asset_cfg=cfg)
    +    term = rewards["TransitionHeightProgress"](None, env)
    +    descent_costs = []
    +    for height in (.49, .45, .43, .40, .37, .35):
    +        term.reset()
    +        robot_height = env.scene["robot"].data.root_pos_w[:, 2]
    +        robot_height.fill_(height + .01)
    +        assert torch.equal(term(env, **args), torch.zeros(2))  # first transition sample
    +        robot_height.fill_(height)
    +        descent = term(env, **args) * env.step_dt * 4.0
    +        assert (descent < 0).all()
    +        descent_costs.append(-descent[0].item())
    +        assert torch.equal(term(env, **args), torch.zeros(2))  # holding has no income
    +        robot_height.fill_(height + .01)
    +        recovery = term(env, **args) * env.step_dt * 4.0
    +        assert (recovery > 0).all()
    +        assert torch.allclose(recovery, -descent, atol=1e-6)
    +        assert torch.equal(term(env, **args), torch.zeros(2))
    +    assert all(later > earlier for earlier, later in zip(descent_costs, descent_costs[1:]))
    +    assert descent_costs[-1] < 4.0  # total possible loss is bounded by the weight
    +
    +    # State changes seed the next transition height without paying for the handoff.
    +    for inactive_state in (state.FOUR_STAND, state.YAW_POS, state.RETURN_TO_4, state.SAFE_RECOVERY):
    +        command.fsm_state.fill_(inactive_state)
    +        env.common_step_counter += 1
    +        env.scene["robot"].data.root_pos_w[:, 2] = .40
    +        assert torch.equal(term(env, **args), torch.zeros(2))
    +        command.fsm_state.fill_(state.TRANSITION_POS)
    +        env.common_step_counter += 1
    +        assert torch.equal(term(env, **args), torch.zeros(2))
    
    
     def test_transition_stability_does_not_track_standing_height():
    @@ -54,23 +73,22 @@ def test_transition_stability_does_not_track_standing_height():
             ))
         assert torch.allclose(values[0], values[1])
         assert torch.allclose(values[1], values[2])
    -    assert (values[0] < 0).all()
    +    assert torch.equal(values[0], torch.zeros(2))
    
    
     def test_downward_speed_is_penalized_before_hard_height_failure():
         rewards, state, env, _ = _transition_scene()
         cfg = SimpleNamespace(name="robot")
         env.scene["robot"].data.root_lin_vel_w[:, 2] = -.5
    -    args = dict(minimum_height=.35, height_margin=.10, warning_height=.43,
    +    args = dict(minimum_height=.35, height_margin=.10, warning_height=.50,
                     fsm_command_name="yaw_rate_cmd", asset_cfg=cfg)
    -    env.scene["robot"].data.root_pos_w[:, 2] = .43
    -    assert torch.equal(rewards["yaw_downward_low_base_velocity_l2"](env, **args), torch.zeros(2))
    -    env.scene["robot"].data.root_pos_w[:, 2] = .39
    -    middle = rewards["yaw_downward_low_base_velocity_l2"](env, **args)
    -    env.scene["robot"].data.root_pos_w[:, 2] = .351
    -    near_boundary = rewards["yaw_downward_low_base_velocity_l2"](env, **args)
    -    assert torch.allclose(middle, torch.full((2,), .125), atol=1e-6)
    -    assert (near_boundary > middle).all()
    +    samples = []
    +    for height in (.49, .45, .43, .40, .37, .35):
    +        env.scene["robot"].data.root_pos_w[:, 2] = height
    +        samples.append(rewards["yaw_downward_low_base_velocity_l2"](env, **args))
    +    assert (samples[0] > 0).all()
    +    assert all((later > earlier).all() for earlier, later in zip(samples, samples[1:]))
    +    assert torch.allclose(samples[-1], torch.full((2,), .25))
         command = env.command_manager.get_term("yaw_rate_cmd")
         command.fsm_state[:] = torch.tensor([state.YAW_POS, state.FOUR_STAND])
         env.common_step_counter += 1
    @@ -85,27 +103,31 @@ def test_lift_credit_uses_selected_support_diagonal_and_keeps_exploration():
         forces[0, [1, 2], 2] = 80.0
         forces[1, [0, 3], 2] = 80.0
         for progress in env.lift_progress.values():
    -        progress.fill_(1.0)
    +        progress.zero_()
         support = dict(
             support_sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=[0, 3]),
             support_sensor_cfg_mirror=SimpleNamespace(name="contact_forces", body_ids=[1, 2]),
         )
         args = dict(asset_cfg=SimpleNamespace(name="pos"), asset_cfg_mirror=SimpleNamespace(name="neg"),
    -                wheel_radius=.091, target_clearance=.05, fsm_command_name="yaw_rate_cmd", **support)
    -    clearance = rewards["yaw_lift_clearance"](env, **args, transition_ungated_fraction=.25)
    -    assert torch.allclose(clearance, torch.full((2,), -.75))
    +                wheel_radius=.091, target_clearance=.05, fsm_command_name="yaw_rate_cmd",
    +                com_asset_cfg=SimpleNamespace(name="robot"),
    +                com_asset_cfg_mirror=SimpleNamespace(name="robot"), **support)
    
         progress_term = rewards["TransitionProgress"](None, env)
    +    assert torch.equal(progress_term(env, **args, transition_ungated_fraction=.25), torch.zeros(2))
    +    for progress in env.lift_progress.values():
    +        progress.fill_(1.0)
    +    env.common_step_counter += 1
         credit = progress_term(env, **args, transition_ungated_fraction=.25)
    -    assert torch.allclose(credit, torch.full((2,), .2475))
    -    progress_term.reset()
    +    assert torch.allclose(credit, torch.full((2,), 75.0))
         forces.zero_()
         forces[0, [0, 3], 2] = 80.0
         forces[1, [1, 2], 2] = 80.0
         env.common_step_counter += 1
    -    assert torch.equal(rewards["yaw_lift_clearance"](env, **args), torch.zeros(2))
    +    clearance_args = {key: value for key, value in args.items() if not key.startswith("com_asset")}
    +    assert torch.equal(rewards["yaw_lift_clearance"](env, **clearance_args), torch.zeros(2))
         full_credit = progress_term(env, **args, transition_ungated_fraction=.25)
    -    assert torch.allclose(full_credit, torch.full((2,), .99))
    +    assert torch.allclose(full_credit, torch.full((2,), 375.0))
    
    
     def test_height_and_tilt_failure_are_separate_termination_masks():
    ✓ • 0ms

## Activity

    $ git diff -- source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py | head -180; git diff -- tests/test_yaw_fsm_training_contract.py | head -120
    diff --git a/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py b/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py
    index c272804..b0a1cf0 100644
    --- a/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py
    +++ b/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py
    @@ -44,7 +44,7 @@ LEG_JOINT_NAMES = [
     WHEEL_RADIUS = 0.091
     TARGET_BASE_HEIGHT = 0.49
     MIN_BASE_HEIGHT = 0.35
    -TRANSITION_HEIGHT_WARNING = 0.43
    +TRANSITION_HEIGHT_SAFE = 0.50
     SUPPORT_SPAN_MIN = 0.50
     SUPPORT_SPAN_MAX = 0.70
     TARGET_LIFT_CLEARANCE = 0.20
    @@ -282,11 +282,10 @@ class VQRWheelRewardsCfg:
    
     @configclass
     class VQRWheelFSMRewardsCfg:
    -    """The 26-term reward contract for ``Flat-VQR-Wheel-Yaw-FSM``.
    +    """The reward contract for ``Flat-VQR-Wheel-Yaw-FSM``.
    
    -    Safety/regularization penalties retain their baseline weights. Positive
    -    pose terms become nonpositive deficits in TRANSITION; YAW and RETURN
    -    retain their rewards. POS/NEG select the command's ``support_diagonal``.
    +    Acquisition improvement is paid once per diagonal and episode. YAW and
    +    RETURN retain their pose rewards; failure has one episode-scale cost.
         """
    
         # ------------------------------ Group 1: always-on baseline ------------------------------
    @@ -367,11 +366,11 @@ class VQRWheelFSMRewardsCfg:
             },
         )
         transition_low_base_height = RewTerm(
    -        func=mdp.yaw_transition_low_base_height,
    -        weight=-4.0,
    +        func=mdp.TransitionHeightProgress,
    +        weight=4.0,
             params={
                 "asset_cfg": SceneEntityCfg("robot"),
    -            "warning_height": TRANSITION_HEIGHT_WARNING,
    +            "safe_height": TRANSITION_HEIGHT_SAFE,
                 "minimum_height": MIN_BASE_HEIGHT,
                 "fsm_command_name": "yaw_rate_cmd",
             },
    @@ -383,7 +382,7 @@ class VQRWheelFSMRewardsCfg:
                 "asset_cfg": SceneEntityCfg("robot"),
                 "minimum_height": MIN_BASE_HEIGHT,
                 "height_margin": 0.10,
    -            "warning_height": TRANSITION_HEIGHT_WARNING,
    +            "warning_height": TRANSITION_HEIGHT_SAFE,
                 "fsm_command_name": "yaw_rate_cmd",
             },
         )
    @@ -433,20 +432,6 @@ class VQRWheelFSMRewardsCfg:
                 "fsm_command_name": "yaw_rate_cmd",
             },
         )
    -    transition_support_load = RewTerm(
    -        func=mdp.yaw_transition_support_load,
    -        weight=3.0,
    -        params={
    -            "sensor_cfg": SceneEntityCfg(
    -                "contact_forces", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
    -            ),
    -            "sensor_cfg_mirror": SceneEntityCfg(
    -                "contact_forces", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
    -            ),
    -            "fsm_command_name": "yaw_rate_cmd",
    -            "target_force_n": SUPPORT_LOAD_TARGET_N,
    -        },
    -    )
         lift_clearance = RewTerm(
             func=mdp.yaw_lift_clearance,
             weight=3.0,
    @@ -592,7 +577,7 @@ class VQRWheelFSMRewardsCfg:
         )
         transition_progress = RewTerm(
             func=mdp.TransitionProgress,
    -        weight=2.0,
    +        weight=1.0,
             params={
                 "asset_cfg": SceneEntityCfg(
                     "robot", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
    @@ -612,8 +597,18 @@ class VQRWheelFSMRewardsCfg:
                 ),
                 "support_force_target_n": SUPPORT_LOAD_TARGET_N,
                 "transition_ungated_fraction": TRANSITION_LIFT_UNGATED_FRACTION,
    +            "com_asset_cfg": SceneEntityCfg(
    +                "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
    +            ),
    +            "com_asset_cfg_mirror": SceneEntityCfg(
    +                "robot", body_names=SUPPORT_WHEEL_NAMES_MIRROR, preserve_order=True
    +            ),
             },
         )
    +    fsm_failure = RewTerm(
    +        func=mdp.fsm_failure_cost,
    +        weight=-60.0,
    +    )
         spin_center_drift = RewTerm(
             func=mdp.spin_center_drift,
             weight=0.0,  # Phase A; curriculum raises this to -0.5/-2.0 in B/C.
    @@ -624,7 +619,7 @@ class VQRWheelFSMRewardsCfg:
         )
         safe_recovery_entry = RewTerm(
             func=mdp.safe_recovery_entry,
    -        weight=0.0,  # Termination semantics in phases A/B; -2.0 in phase C.
    +        weight=0.0,  # Terminal cost in phases A/B; entry cost in phase C.
             params={"fsm_command_name": "yaw_rate_cmd"},
         )
    
    diff --git a/tests/test_yaw_fsm_training_contract.py b/tests/test_yaw_fsm_training_contract.py
    index 189ce91..b7cfb7a 100644
    --- a/tests/test_yaw_fsm_training_contract.py
    +++ b/tests/test_yaw_fsm_training_contract.py
    @@ -63,10 +63,12 @@ def test_fsm_balance_is_gated_and_reward_counts_are_isolated():
         assert len(fsm) == 26
         assert "transition_low_base_height" in fsm
         assert "support_contact" not in fsm
    -    assert "transition_support_load" in fsm
    +    assert "transition_support_load" not in fsm
    +    assert "fsm_failure" in fsm
         assert "return_to_four_landing" in fsm
         assert "four_stand_ready_bonus" in fsm
    -    assert ast.literal_eval(_keyword(fsm["transition_support_load"], "weight")) == 3.0
    +    assert ast.literal_eval(_keyword(fsm["fsm_failure"], "weight")) == -60.0
    +    assert ast.literal_eval(_keyword(fsm["transition_progress"], "weight")) == 1.0
    
         fsm_balance_params = ast.literal_eval(_keyword(fsm["balance"], "params"))
         baseline_balance_params = ast.literal_eval(_keyword(baseline["balance"], "params"))
    @@ -147,7 +149,7 @@ def test_fsm_startup_contract_checks_reward_and_command_runtime_objects():
         required = {
             "fsm_gated_tracking",
             "transition_progress",
    -        "transition_support_load",
    +        "fsm_failure",
             "transition_low_base_height",
             "return_to_four_landing",
             "four_stand_ready_bonus",
    ✓ • 0ms

## Activity

    $ rg -n 'transition_progress|safe_recovery_entry|phase_weights|tracking.*weight|spin_center_drift.*weight' tests/test_yaw_curriculum.py
    290:            "transition_progress": reward_cfg(2.0),
    292:            "safe_recovery_entry": reward_cfg(0.0),
    385:        transition_reward_name="transition_progress",
    520:        "transition_reward_name": "transition_progress",
    522:        "safe_recovery_reward_name": "safe_recovery_entry",
    543:            "transition_progress": SimpleNamespace(
    546:            "spin_center_drift": SimpleNamespace(weight=0.0, params={}),
    547:            "safe_recovery_entry": SimpleNamespace(weight=0.0, params={}),
    579:        "_yaw_fsm_task_curriculum_ramp_start_transition_progress": 2.0,
    581:        "_yaw_fsm_task_curriculum_ramp_start_safe_recovery_entry": 0.0,
    608:    for reward_name in ("lift_clearance", "fsm_gated_tracking", "transition_progress"):
    613:    assert resumed_env.reward_manager.get_term_cfg("fsm_gated_tracking").weight == pytest.approx(8.0)
    614:    assert resumed_env.reward_manager.get_term_cfg("transition_progress").weight == pytest.approx(2.0)
    615:    assert resumed_env.reward_manager.get_term_cfg("spin_center_drift").weight == pytest.approx(-0.75)
    616:    assert resumed_env.reward_manager.get_term_cfg("safe_recovery_entry").weight == pytest.approx(
    ✓ • 0ms

## Activity

    file changes: Completed · 1 changes
    Update { move_path: None }: /home/robotics/tuanpm48/vqr/rl_training/tests/test_yaw_curriculum.py
    @@ -613,7 +613,5 @@
         assert resumed_env.reward_manager.get_term_cfg("fsm_gated_tracking").weight == pytest.approx(8.0)
    -    assert resumed_env.reward_manager.get_term_cfg("transition_progress").weight == pytest.approx(2.0)
    +    assert resumed_env.reward_manager.get_term_cfg("transition_progress").weight == pytest.approx(2.0 - 1.0 / 6.0)
         assert resumed_env.reward_manager.get_term_cfg("spin_center_drift").weight == pytest.approx(-0.75)
    -    assert resumed_env.reward_manager.get_term_cfg("safe_recovery_entry").weight == pytest.approx(
    -        -1.0 / 3.0
    -    )
    +    assert resumed_env.reward_manager.get_term_cfg("safe_recovery_entry").weight == pytest.approx(-10.0)
         tracking_params = resumed_env.reward_manager.get_term_cfg("fsm_gated_tracking").params

## Activity

    file changes: Completed · 1 changes
    Update { move_path: None }: /home/robotics/tuanpm48/vqr/rl_training/tests/test_yaw_fsm_systemic.py
    @@ -158,2 +158,20 @@
     
    +def test_failure_cost_is_one_episode_scale_event_and_waiting_beats_early_failure():
    +    rewards, _, env, _ = _reward_scene()
    +    _load_nodes(REWARDS_PATH, {"fsm_failure_cost"}, rewards)
    +    env.termination_manager = SimpleNamespace(terminated=torch.tensor([True, False]))
    +    pulse = rewards["fsm_failure_cost"](env)
    +    assert torch.equal(pulse * env.step_dt, torch.tensor([1.0, 0.0]))
    +
    +    # Even at the former -19/s from tracking, lift and support load, the
    +    # episode-scale cost makes trying for another second better than failing
    +    # immediately. The current transition pose terms no longer pay this cost.
    +    gamma = .99
    +    steps = int(1.0 / env.step_dt)
    +    early_failure = -60.0
    +    continued_attempt = sum(-19.0 * env.step_dt * gamma**step for step in range(steps))
    +    continued_attempt -= 60.0 * gamma**steps
    +    assert continued_attempt > early_failure
    +
    +
     def test_watchdog_survives_chatter_pauses_in_yaw_and_resets_only_at_maneuver_end():

## Activity

    $ git diff --check
    ✓ • 0ms

## Activity

    $ rg -n 'yaw_transition_support_load|TRANSITION.*deficit|transition.*deficit|transition_support_load' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity scripts/reinforcement_learning/rsl_rl/train.py tests/test_yaw_* | head -100
    tests/test_yaw_fsm.py:716:def test_transition_lift_has_no_per_step_deficit():
    tests/test_yaw_fsm_training_contract.py:66:    assert "transition_support_load" not in fsm
    ✓ • 0ms

## Activity

    $ python3 -m compileall -q scripts/reinforcement_learning/rsl_rl/train.py source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py tests/test_yaw_fsm.py tests/test_yaw_fsm_systemic.py tests/test_yaw_transition_safety.py tests/test_yaw_curriculum.py tests/test_yaw_fsm_training_contract.py
    ✓ • 0ms

## Activity

    $ git status --short; git diff --numstat
     M scripts/reinforcement_learning/rsl_rl/train.py
     M source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py
     M source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py
     M source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py
     M tests/test_yaw_curriculum.py
     M tests/test_yaw_fsm.py
     M tests/test_yaw_fsm_systemic.py
     M tests/test_yaw_fsm_training_contract.py
     M tests/test_yaw_transition_safety.py
    1    1    scripts/reinforcement_learning/rsl_rl/train.py
    20    25    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py
    1    1    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py
    113    68    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py
    2    4    tests/test_yaw_curriculum.py
    12    13    tests/test_yaw_fsm.py
    33    9    tests/test_yaw_fsm_systemic.py
    5    3    tests/test_yaw_fsm_training_contract.py
    67    33    tests/test_yaw_transition_safety.py
    ✓ • 0ms

## Assistant

Mã và các test đã qua kiểm tra cú pháp và `git diff --check`. Session này vẫn chưa có Python chứa `torch`, nên mình đang rà lại các test theo logic thực thi và tìm runtime đã cài sẵn để chạy bộ CPU test, thay vì coi kiểm tra cú pháp là test pass.

## Activity

    $ command -v conda || true; command -v uv || true; command -v pytest || true; command -v python3.11 || true
    /home/robotics/miniconda3/condabin/conda
    /home/robotics/.local/bin/uv
    /usr/bin/python3.11
    ✓ • 0ms

## Activity

    $ rg -n 'def test_.*transition|safe_recovery_entry|fsm_failure_cost|transition_progress' tests/test_yaw_fsm.py tests/test_yaw_fsm_systemic.py tests/test_yaw_transition_safety.py | tail -100
    tests/test_yaw_transition_safety.py:29:def test_transition_height_progress_is_bounded_monotone_and_reversible():
    tests/test_yaw_transition_safety.py:70:def test_transition_stability_does_not_track_standing_height():
    tests/test_yaw_fsm.py:108:def test_transition_telemetry_uses_active_support_lift_and_pose():
    tests/test_yaw_fsm.py:716:def test_transition_lift_has_no_per_step_deficit():
    tests/test_yaw_fsm_systemic.py:78:def test_stalled_transition_has_no_pose_income_and_handoff_does_not_drop(clearance):
    tests/test_yaw_fsm_systemic.py:161:    _load_nodes(REWARDS_PATH, {"fsm_failure_cost"}, rewards)
    tests/test_yaw_fsm_systemic.py:163:    pulse = rewards["fsm_failure_cost"](env)
    ✓ • 0ms

## Activity

    $ ls -a /home/robotics | head -80
    .
    ..
    .99-realsense-libusb.rules
    Algorithm.toml
    All_for_new
    All_for_new.zip
    .anaconda
    .android
    .antigravity
    .antigravity-ide-server
    .antigravity-server
    .anydesk
    anylabeling_data
    .anylabelingrc
    .bash_history
    .bash_logout
    .bashrc
    benbp
    bin
    binhnt27
    .cache
    .cert
    .claude
    .claude.json
    .claude.json.tmp.1820392.0f50c00329ff
    cloudxr.env
    .codegraph
    .codex
    codex-session-01a0d7bb-5ba2-7212-bc9d-6aa36dbc0ef4.md
    collect_usd.py
    .conda
    .condarc
    .config
    .copilot
    cosmos3
    datld5
    Desktop
    .docker
    docker
    Documents
    .dotnet
    Downloads
    dung_ws
    .fontconfig
    .gazebo
    .gemini
    get-docker.sh
    .gitconfig
    .git-credentials
    .gnupg
    .gphoto
    .humming
    .ignition
    .ipython
    .jupyter
    .keras
    .labelmerc
    .lesshst
    Lite3_rl_deploy
    Lite3_ROS
    .local
    logs
    MinhL1
    miniconda3
    models
    .mozilla
    Music
    .netrc
    .npm
    .nv
    .nvidia-omniverse
    .nvidia-settings-rc
    .nvm
    .nx
    .parallel
    Pictures
    .pki
    .profile
    Public
    .python_history
    ✓ • 0ms

## Activity

    $ conda env list
    
    # conda environments:
    #
    # * -> active
    # + -> frozen
                             /home/robotics/MinhL1/0910_BEV_Simulation/isaacsim_env
    base                     /home/robotics/miniconda3
    ahawam                   /home/robotics/miniconda3/envs/ahawam
    augment_seg              /home/robotics/miniconda3/envs/augment_seg
    env_isaaclab             /home/robotics/miniconda3/envs/env_isaaclab
    env_isaaclab_3.0         /home/robotics/miniconda3/envs/env_isaaclab_3.0
    env_isaaclab_datld       /home/robotics/miniconda3/envs/env_isaaclab_datld
    env_isaaclab_datld_vqr     /home/robotics/miniconda3/envs/env_isaaclab_datld_vqr
    env_isaaclab_tuanpm48_vqr     /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr
    env_isaaclab_tungna      /home/robotics/miniconda3/envs/env_isaaclab_tungna
    isaacgym                 /home/robotics/miniconda3/envs/isaacgym
    labelme                  /home/robotics/miniconda3/envs/labelme
    lite3_isaaclab           /home/robotics/miniconda3/envs/lite3_isaaclab
    mask2former              /home/robotics/miniconda3/envs/mask2former
    minhl1_seg               /home/robotics/miniconda3/envs/minhl1_seg
    nnlandmark               /home/robotics/miniconda3/envs/nnlandmark
    salsanext                /home/robotics/miniconda3/envs/salsanext
    train_landmark3d         /home/robotics/miniconda3/envs/train_landmark3d
    vqr_poseopt              /home/robotics/miniconda3/envs/vqr_poseopt
    yoloworld_seg            /home/robotics/miniconda3/envs/yoloworld_seg
    
    ✓ • 393ms

## Activity

    $ /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python --version; /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python -c 'import torch, pytest; print(torch.__version__, pytest.__version__)'
    2.7.0+cu128 9.1.1
    ✓ • 1.54s

## Activity

    $ /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python -m pytest -q tests/test_yaw_fsm.py tests/test_yaw_fsm_systemic.py tests/test_yaw_transition_safety.py tests/test_yaw_fsm_training_contract.py tests/test_yaw_curriculum.py
    Traceback (most recent call last):
      File "<frozen runpy>", line 198, in _run_module_as_main
      File "<frozen runpy>", line 88, in _run_code
      File "/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/pytest/__main__.py", line 9, in <module>
        raise SystemExit(_console_main())
                         ^^^^^^^^^^^^^^^
      File "/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/_pytest/config/__init__.py", line 253, in _console_main
        code = _main(prog=_get_prog_name(sys.argv))
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/_pytest/config/__init__.py", line 223, in _main
        config = _prepareconfig(new_args, plugins, prog=prog)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/_pytest/config/__init__.py", line 410, in _prepareconfig
        config: Config = pluginmanager.hook.pytest_cmdline_parse(
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/pluggy/_hooks.py", line 512, in __call__
        return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/pluggy/_manager.py", line 120, in _hookexec
        return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/pluggy/_callers.py", line 167, in _multicall
        raise exception
      File "/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/pluggy/_callers.py", line 139, in _multicall
        teardown.throw(exception)
      File "/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/_pytest/helpconfig.py", line 124, in pytest_cmdline_parse
        config = yield
                 ^^^^^
      File "/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/pluggy/_callers.py", line 121, in _multicall
        res = hook_impl.function(*args)
              ^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/_pytest/config/__init__.py", line 1232, in pytest_cmdline_parse
        self.parse(args)
      File "/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/_pytest/config/__init__.py", line 1583, in parse
        self.pluginmanager.load_setuptools_entrypoints("pytest11")
      File "/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/pluggy/_manager.py", line 416, in load_setuptools_entrypoints
        plugin = ep.load()
                 ^^^^^^^^^
      File "/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/importlib/metadata/__init__.py", line 202, in load
        module = import_module(match.group('module'))
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/importlib/__init__.py", line 126, in import_module
        return _bootstrap._gcd_import(name[level:], package, level)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
      File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
      File "<frozen importlib._bootstrap>", line 1126, in _find_and_load_unlocked
      File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
      File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
      File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
      File "<frozen importlib._bootstrap>", line 1126, in _find_and_load_unlocked
      File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
      File "<frozen importlib._bootstrap>", line 1204, in _gcd_import
      File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
      File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
      File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
      File "<frozen importlib._bootstrap_external>", line 940, in exec_module
      File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
      File "/opt/ros/humble/lib/python3.10/site-packages/launch_testing/__init__.py", line 15, in <module>
        from . import tools
      File "/opt/ros/humble/lib/python3.10/site-packages/launch_testing/tools/__init__.py", line 18, in <module>
        from .process import launch_process
      File "/opt/ros/humble/lib/python3.10/site-packages/launch_testing/tools/process.py", line 17, in <module>
        import launch
      File "/opt/ros/humble/lib/python3.10/site-packages/launch/__init__.py", line 17, in <module>
        from . import actions
      File "/opt/ros/humble/lib/python3.10/site-packages/launch/actions/__init__.py", line 17, in <module>
        from .declare_launch_argument import DeclareLaunchArgument
      File "/opt/ros/humble/lib/python3.10/site-packages/launch/actions/declare_launch_argument.py", line 25, in <module>
        from ..frontend import Entity
      File "/opt/ros/humble/lib/python3.10/site-packages/launch/frontend/__init__.py", line 20, in <module>
        from .parser import InvalidFrontendLaunchFileError, Parser
      File "/opt/ros/humble/lib/python3.10/site-packages/launch/frontend/parser.py", line 38, in <module>
        from .parse_substitution import parse_if_substitutions
      File "/opt/ros/humble/lib/python3.10/site-packages/launch/frontend/parse_substitution.py", line 23, in <module>
        from lark import Lark
    ModuleNotFoundError: No module named 'lark'
    ✗ (1) • 137ms

## Activity

    $ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python -m pytest -q tests/test_yaw_fsm.py tests/test_yaw_fsm_systemic.py tests/test_yaw_transition_safety.py tests/test_yaw_fsm_training_contract.py tests/test_yaw_curriculum.py
    ........................................................                 [100%]
    56 passed in 3.24s
    ✓ • 3.70s

## Activity

    $ rg --files tests | rg '^tests/test_yaw.*\.py$'
    tests/test_yaw_transition_safety.py
    tests/test_yaw_fsm_observations.py
    tests/test_yaw_curriculum.py
    tests/test_yaw_fsm_training_contract.py
    tests/test_yaw_fsm_systemic.py
    tests/test_yaw_fsm.py
    ✓ • 0ms

## Activity

    file changes: Completed · 1 changes
    Update { move_path: None }: /home/robotics/tuanpm48/vqr/rl_training/tests/test_yaw_fsm_systemic.py
    @@ -176,2 +176,17 @@
     
    +def test_safe_recovery_entry_cost_is_one_pulse():
    +    rewards, state, env, command = _reward_scene()
    +    _load_nodes(REWARDS_PATH, {"safe_recovery_entry"}, rewards)
    +    command.fsm_state.fill_(state.SAFE_RECOVERY)
    +    command.just_switched.fill_(True)
    +    env.common_step_counter += 1
    +    assert torch.equal(
    +        rewards["safe_recovery_entry"](env, "yaw_rate_cmd") * env.step_dt,
    +        torch.ones(2),
    +    )
    +    command.just_switched.fill_(False)
    +    env.common_step_counter += 1
    +    assert torch.equal(rewards["safe_recovery_entry"](env, "yaw_rate_cmd"), torch.zeros(2))
    +
    +
     def test_watchdog_survives_chatter_pauses_in_yaw_and_resets_only_at_maneuver_end():

## Activity

    $ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python -m pytest -q tests/test_yaw_fsm.py tests/test_yaw_fsm_systemic.py tests/test_yaw_transition_safety.py tests/test_yaw_fsm_training_contract.py tests/test_yaw_curriculum.py tests/test_yaw_fsm_observations.py
    .............................................................            [100%]
    61 passed in 3.38s
    ✓ • 3.77s

## Activity

    $ git diff --check; git status --short
     M scripts/reinforcement_learning/rsl_rl/train.py
     M source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py
     M source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py
     M source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py
     M tests/test_yaw_curriculum.py
     M tests/test_yaw_fsm.py
     M tests/test_yaw_fsm_systemic.py
     M tests/test_yaw_fsm_training_contract.py
     M tests/test_yaw_transition_safety.py
    ✓ • 0ms

## Activity

    $ rg -n 'TransitionProgress\(|func=mdp\.TransitionProgress|transition_progress.*weight|fsm_failure|TransitionHeightProgress' source tests scripts | head -130
    tests/test_yaw_transition_safety.py:18:    names = {"TransitionHeightProgress", "yaw_downward_low_base_velocity_l2"}
    tests/test_yaw_transition_safety.py:33:    term = rewards["TransitionHeightProgress"](None, env)
    tests/test_yaw_curriculum.py:614:    assert resumed_env.reward_manager.get_term_cfg("transition_progress").weight == pytest.approx(2.0 - 1.0 / 6.0)
    tests/test_yaw_fsm_training_contract.py:67:    assert "fsm_failure" in fsm
    tests/test_yaw_fsm_training_contract.py:70:    assert ast.literal_eval(_keyword(fsm["fsm_failure"], "weight")) == -60.0
    tests/test_yaw_fsm_training_contract.py:71:    assert ast.literal_eval(_keyword(fsm["transition_progress"], "weight")) == 1.0
    tests/test_yaw_fsm_training_contract.py:152:        "fsm_failure",
    tests/test_yaw_fsm_systemic.py:161:    _load_nodes(REWARDS_PATH, {"fsm_failure_cost"}, rewards)
    tests/test_yaw_fsm_systemic.py:163:    pulse = rewards["fsm_failure_cost"](env)
    scripts/reinforcement_learning/rsl_rl/train.py:345:        "fsm_failure",
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:306:class TransitionHeightProgress(ManagerTermBase):
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:1179:class TransitionProgress(ManagerTermBase):
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:1287:def fsm_failure_cost(env: ManagerBasedRLEnv) -> torch.Tensor:
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:369:        func=mdp.TransitionHeightProgress,
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:579:        func=mdp.TransitionProgress,
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:608:    fsm_failure = RewTerm(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:609:        func=mdp.fsm_failure_cost,
    ✓ • 0ms

## Activity

    $ sed -n '1200,1270p' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py
            self,
            env: ManagerBasedRLEnv,
            asset_cfg: SceneEntityCfg,
            asset_cfg_mirror: SceneEntityCfg,
            wheel_radius: float,
            target_clearance: float,
            fsm_command_name: str,
            gamma: float = 0.99,
            support_sensor_cfg: SceneEntityCfg | None = None,
            support_sensor_cfg_mirror: SceneEntityCfg | None = None,
            support_force_target_n: float = 80.0,
            transition_ungated_fraction: float = 0.25,
            com_asset_cfg: SceneEntityCfg | None = None,
            com_asset_cfg_mirror: SceneEntityCfg | None = None,
            com_std: float = 0.08,
            segment_std: float = 0.05,
            lift_credit: float = 6.0,
            support_credit: float = 3.0,
            com_credit: float = 3.0,
            segment_credit: float = 2.0,
        ) -> torch.Tensor:
            if not 0.0 < gamma <= 1.0:
                raise ValueError("gamma must be in (0, 1].")
            if support_sensor_cfg is None or support_sensor_cfg_mirror is None:
                raise ValueError("Both support sensor configs are required for transition progress.")
            if not 0.0 < transition_ungated_fraction < 1.0:
                raise ValueError("transition_ungated_fraction must be in (0, 1).")
            if com_std <= 0.0 or segment_std <= 0.0:
                raise ValueError("CoM and segment standard deviations must be positive.")
            if min(lift_credit, support_credit, com_credit, segment_credit) < 0.0:
                raise ValueError("Acquisition credits must be non-negative.")
    
            gates = fsm_gates(env, fsm_command_name)
            progress_pos = _yaw_lift_progress(
                env, asset_cfg, wheel_radius, target_clearance
            ).mean(dim=1)
            progress_neg = _yaw_lift_progress(
                env, asset_cfg_mirror, wheel_radius, target_clearance
            ).mean(dim=1)
            progress = torch.where(gates["diag_pos"], progress_pos, progress_neg)
            phi = torch.where(gates["b_return"], 1.0 - progress, progress)
    
            # Always advance the potential, including FOUR/YAW/SAFE states.  This
            # prevents a spurious pulse when the next gated phase begins.
            reward = gamma * phi - self.prev_phi
            self.prev_phi.copy_(phi)
            reward = torch.where(gates["just_switched"], torch.zeros_like(reward), reward)
            # Seed from the first observed transition pose. Update best values in
            # YAW too, so leaving and re-entering TRANSITION cannot replay credit.
            active = gates["b_trans"] | gates["b_yaw"]
            support_pos = _yaw_support_load_quality(env, support_sensor_cfg, support_force_target_n)
            support_neg = _yaw_support_load_quality(env, support_sensor_cfg_mirror, support_force_target_n)
            support_quality = torch.where(gates["diag_pos"], support_pos, support_neg)
            lift_gate = transition_ungated_fraction + (1.0 - transition_ungated_fraction) * support_quality
            if com_asset_cfg is None or com_asset_cfg_mirror is None:
                raise ValueError("Both support body configs are required for acquisition credit.")
            pos_distance, pos_projection, pos_length = _yaw_support_geometry(env, com_asset_cfg)
            neg_distance, neg_projection, neg_length = _yaw_support_geometry(env, com_asset_cfg_mirror)
            distance = torch.where(gates["diag_pos"], pos_distance, neg_distance)
            projection = torch.where(gates["diag_pos"], pos_projection, neg_projection)
            length = torch.where(gates["diag_pos"], pos_length, neg_length)
            outside = (torch.relu(-projection) + torch.relu(projection - 1.0)) * length
            quality = torch.stack((
                progress * lift_gate,
                support_quality,
                1.0 / (1.0 + distance / com_std),
                torch.exp(-outside.square() / segment_std**2),
            ), dim=1)
            rows = torch.arange(env.num_envs, device=progress.device)
            diagonal = (~gates["diag_pos"]).long()
            previous = self.best_quality[rows, diagonal]
    ✓ • 0ms

## Activity

    file changes: Completed · 1 changes
    Update { move_path: None }: /home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py
    @@ -312,2 +312,3 @@
             self.was_transition = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    +        self.loss_budget = torch.zeros(env.num_envs, device=env.device)
     
    @@ -317,2 +318,3 @@
                 self.was_transition.zero_()
    +            self.loss_budget.zero_()
             else:
    @@ -320,2 +322,3 @@
                 self.was_transition[env_ids] = False
    +            self.loss_budget[env_ids] = 0.0
     
    @@ -341,6 +344,9 @@
             # height cost is the potential change, with no time-at-height deficit.
    -        reward = torch.where(
    -            continuing, (potential - self.prev_potential) / env.step_dt,
    -            torch.zeros_like(potential),
    -        )
    +        delta = torch.where(continuing, potential - self.prev_potential, torch.zeros_like(potential))
    +        loss = torch.relu(-delta)
    +        recovery = torch.minimum(torch.relu(delta), self.loss_budget + loss)
    +        self.loss_budget += loss - recovery
    +        # Recovery only refunds height lost during TRANSITION in this episode.
    +        # Falling in YAW then re-entering TRANSITION cannot create income.
    +        reward = (recovery - loss) / env.step_dt
             self.prev_potential.copy_(potential)

## Activity

    file changes: Completed · 1 changes
    Update { move_path: None }: /home/robotics/tuanpm48/vqr/rl_training/tests/test_yaw_transition_safety.py
    @@ -68,2 +68,15 @@
     
    +    term.reset()
    +    env.scene["robot"].data.root_pos_w[:, 2] = .49
    +    assert torch.equal(term(env, **args), torch.zeros(2))
    +    command.fsm_state.fill_(state.YAW_POS)
    +    env.scene["robot"].data.root_pos_w[:, 2] = .40
    +    env.common_step_counter += 1
    +    assert torch.equal(term(env, **args), torch.zeros(2))
    +    command.fsm_state.fill_(state.TRANSITION_POS)
    +    env.common_step_counter += 1
    +    assert torch.equal(term(env, **args), torch.zeros(2))
    +    env.scene["robot"].data.root_pos_w[:, 2] = .49
    +    assert torch.equal(term(env, **args), torch.zeros(2))
    +
     

## Activity

    $ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python -m pytest -q tests/test_yaw_fsm.py tests/test_yaw_fsm_systemic.py tests/test_yaw_transition_safety.py tests/test_yaw_fsm_training_contract.py tests/test_yaw_curriculum.py tests/test_yaw_fsm_observations.py
    .............................................................            [100%]
    61 passed in 3.38s
    ✓ • 3.79s

## Activity

    $ sed -n '820,920p' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py; sed -n '920,980p' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py
                f"_yaw_fsm_telemetry_{suffix}_support_loss_max_steps_sum": 0,
                f"_yaw_fsm_telemetry_{suffix}_support_loss_episodes": 0,
            })
            for wheel_name in (("FL", "HR") if suffix == "pos" else ("FR", "HL")):
                telemetry_defaults[f"_yaw_fsm_telemetry_{suffix}_support_{wheel_name}_sum"] = 0.0
                telemetry_defaults[f"_yaw_fsm_telemetry_{suffix}_support_{wheel_name}_samples"] = 0
        for name, default in telemetry_defaults.items():
            if not hasattr(env, name):
                setattr(env, name, default)
    
        phase = max(0, min(int(env._yaw_fsm_task_curriculum_phase), 2))
        env._yaw_fsm_task_curriculum_phase = phase
        env._yaw_task_curriculum_stage = max(
            0, min(int(env._yaw_task_curriculum_stage), len(clearance_levels) - 1)
        )
        env._yaw_task_curriculum_yaw_stage = max(
            0, min(int(env._yaw_task_curriculum_yaw_stage), len(yaw_rate_levels) - 1)
        )
    
        if isinstance(env_ids, slice):
            selected_env_ids = torch.arange(env.num_envs, device=env.device)
        else:
            selected_env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        completed_env_ids = selected_env_ids[env.episode_length_buf[selected_env_ids] > 0]
    
        def episode_buffer(name: str, dtype: torch.dtype) -> torch.Tensor:
            value = getattr(env, name, None)
            if value is None:
                return torch.zeros(len(completed_env_ids), dtype=dtype, device=env.device)
            return torch.as_tensor(value, device=env.device, dtype=dtype)[completed_env_ids]
    
        if len(completed_env_ids):
            for suffix in ("pos", "neg"):
                yaw_samples = episode_buffer(f"_yaw_fsm_{suffix}_yaw_samples", torch.long)
                attempted = episode_buffer(f"_yaw_fsm_{suffix}_transition_attempted", torch.long)
                transitioned = episode_buffer(f"_yaw_fsm_{suffix}_transition_succeeded", torch.long)
                lift_sum = episode_buffer(f"_yaw_fsm_{suffix}_lift_sum", torch.float32)
                support_sum = episode_buffer(f"_yaw_fsm_{suffix}_support_sum", torch.float32)
    
                # Every maneuver stays in the denominator, including attempts
                # with no YAW samples and terminal reacquisition failures.
                evaluated = attempted
                success = transitioned if phase == 1 else episode_buffer(
                    f"_yaw_fsm_{suffix}_pose_succeeded", torch.long
                )
                env_name = f"_yaw_fsm_task_curriculum_{suffix}"
                setattr(env, f"{env_name}_episodes", getattr(env, f"{env_name}_episodes") + int(evaluated.sum().item()))
                setattr(
                    env,
                    f"{env_name}_successes",
                    getattr(env, f"{env_name}_successes") + int(success.sum().item()),
                )
                setattr(env, f"{env_name}_lift_sum", getattr(env, f"{env_name}_lift_sum") + float(lift_sum.sum().item()))
                setattr(env, f"{env_name}_lift_samples", getattr(env, f"{env_name}_lift_samples") + int(yaw_samples.sum().item()))
                setattr(
                    env,
                    f"{env_name}_support_sum",
                    getattr(env, f"{env_name}_support_sum") + float(support_sum.sum().item()),
                )
                setattr(
                    env,
                    f"{env_name}_support_samples",
                    getattr(env, f"{env_name}_support_samples") + int(yaw_samples.sum().item()),
                )
                if phase == 0:
                    setattr(
                        env,
                        f"{env_name}_lift_failures",
                        getattr(env, f"{env_name}_lift_failures")
                        + int(episode_buffer(f"_yaw_fsm_{suffix}_lift_failed", torch.long).sum().item()),
                    )
                    setattr(
                        env,
                        f"{env_name}_support_failures",
                        getattr(env, f"{env_name}_support_failures")
                        + int(episode_buffer(f"_yaw_fsm_{suffix}_support_failed", torch.long).sum().item()),
                    )
                setattr(
                    env,
                    f"{env_name}_command_sum",
                    getattr(env, f"{env_name}_command_sum")
                    + float(episode_buffer(f"_yaw_fsm_{suffix}_command_abs_sum", torch.float32).sum().item()),
                )
                setattr(
                    env,
                    f"{env_name}_error_sum",
                    getattr(env, f"{env_name}_error_sum")
                    + float(episode_buffer(f"_yaw_fsm_{suffix}_yaw_abs_error_sum", torch.float32).sum().item()),
                )
                setattr(
                    env,
                    f"{env_name}_tracking_samples",
                    getattr(env, f"{env_name}_tracking_samples") + int(yaw_samples.sum().item()),
                )
                setattr(
                    env,
                    f"{env_name}_drift_sum",
                    getattr(env, f"{env_name}_drift_sum")
                    + float(episode_buffer(f"_yaw_fsm_{suffix}_drift_sum", torch.float32).sum().item()),
                )
                setattr(
                setattr(
                    env,
                    f"{env_name}_drift_samples",
                    getattr(env, f"{env_name}_drift_samples")
                    + int(episode_buffer(f"_yaw_fsm_{suffix}_drift_samples", torch.long).sum().item()),
                )
    
            if phase == 0:
                for suffix in ("pos", "neg"):
                    prefix = f"_yaw_fsm_task_curriculum_{suffix}"
                    episodes = max(getattr(env, f"{prefix}_episodes"), 1)
                    setattr(
                        env,
                        f"_yaw_fsm_task_curriculum_last_{suffix}_mean_lift_progress",
                        getattr(env, f"{prefix}_lift_sum")
                        / max(getattr(env, f"{prefix}_lift_samples"), 1),
                    )
                    setattr(
                        env,
                        f"_yaw_fsm_task_curriculum_last_{suffix}_mean_support_score",
                        getattr(env, f"{prefix}_support_sum")
                        / max(getattr(env, f"{prefix}_support_samples"), 1),
                    )
                    setattr(
                        env,
                        f"_yaw_fsm_task_curriculum_last_{suffix}_fail_lift_rate",
                        getattr(env, f"{prefix}_lift_failures") / episodes,
                    )
                    setattr(
                        env,
                        f"_yaw_fsm_task_curriculum_last_{suffix}_fail_support_rate",
                        getattr(env, f"{prefix}_support_failures") / episodes,
                    )
    
            if hasattr(env, "_yaw_fsm_state_steps"):
                state_steps = env._yaw_fsm_state_steps[completed_env_ids]
                env._yaw_fsm_telemetry_state_totals += state_steps.sum(dim=0)
                env._yaw_fsm_telemetry_steps += int(state_steps.sum().item())
                env._yaw_fsm_telemetry_switches += int(
                    env._yaw_fsm_switches[completed_env_ids].sum().item()
                )
            if hasattr(env, "_yaw_fsm_transition_duration_steps"):
                env._yaw_fsm_telemetry_transition_steps += int(
                    env._yaw_fsm_transition_duration_steps[completed_env_ids].sum().item()
                )
                env._yaw_fsm_telemetry_transition_attempts += int(
                    env._yaw_fsm_transition_attempts[completed_env_ids].sum().item()
                )
            for field in transition_fields:
                episode_name = f"_yaw_fsm_transition_{field}_sum"
                if hasattr(env, episode_name):
                    lifetime_name = f"_yaw_fsm_telemetry_transition_{field}_sum"
                    setattr(env, lifetime_name, getattr(env, lifetime_name) + float(
                        getattr(env, episode_name)[completed_env_ids].sum().item()
                    ))
            if hasattr(env, "_yaw_fsm_positive_budget"):
                env._yaw_fsm_telemetry_positive_budget += env._yaw_fsm_positive_budget[
                    completed_env_ids
                ].sum(dim=0)
                env._yaw_fsm_telemetry_positive_budget_pos += float(
                    env._yaw_fsm_positive_budget_pos[completed_env_ids].sum().item()
    ✓ • 0ms

## Assistant

Đã chạy được bộ CPU test bằng môi trường Isaac Lab đã có sẵn: **61 test pass**. Mình cũng chặn một đường farm phụ ở reward độ cao: hồi phục độ cao trong TRANSITION chỉ được hoàn lại phần đã mất trước đó trong TRANSITION của cùng episode, nên rơi ở YAW rồi quay lại không tạo credit miễn phí. Mình đang kiểm tra diff cuối cùng và sẽ báo rõ phần nào chưa được xác nhận bằng simulator.

## Activity

    $ rg -n '^class TransitionHeightProgress|^class TransitionProgress|^def fsm_failure_cost|^def safe_recovery_entry|^class VQRWheelFSMRewardsCfg|^    fsm_failure =|^    transition_progress =|^    phase_weights =|^def _verify_yaw_fsm_contract' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py scripts/reinforcement_learning/rsl_rl/train.py
    scripts/reinforcement_learning/rsl_rl/train.py:337:def _verify_yaw_fsm_contract(env, env_cfg) -> None:
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:284:class VQRWheelFSMRewardsCfg:
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:578:    transition_progress = RewTerm(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:608:    fsm_failure = RewTerm(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py:1193:    phase_weights = ((8.0, 1.0, 0.0, 0.0), (8.0, 1.0, -0.5, 0.0), (8.0, 1.0, -2.0, -60.0))[phase]
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:306:class TransitionHeightProgress(ManagerTermBase):
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:1185:class TransitionProgress(ManagerTermBase):
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:1293:def fsm_failure_cost(env: ManagerBasedRLEnv) -> torch.Tensor:
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:1336:def safe_recovery_entry(
    ✓ • 0ms

## Activity

    $ git diff --check; git diff --stat
     scripts/reinforcement_learning/rsl_rl/train.py     |   2 +-
     .../config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py    |  45 +++--
     .../locomotion/velocity/mdp/curriculums.py         |   2 +-
     .../locomotion/velocity/mdp/rewards.py             | 187 +++++++++++++--------
     tests/test_yaw_curriculum.py                       |   6 +-
     tests/test_yaw_fsm.py                              |  25 ++-
     tests/test_yaw_fsm_systemic.py                     |  57 ++++++-
     tests/test_yaw_fsm_training_contract.py            |   8 +-
     tests/test_yaw_transition_safety.py                | 111 ++++++++----
     9 files changed, 287 insertions(+), 156 deletions(-)
    ✓ • 0ms

## Assistant

Đã sửa incentive “fail sớm” trong TRANSITION.

- Reward acquisition chỉ trả khi **lift, support load và CoM vượt mức tốt nhất** của từng hướng trong episode; đứng yên hoặc chuyển qua lại giữa TRANSITION và YAW không nhận thêm credit. Các pose term không còn tích lũy deficit mỗi bước TRANSITION. [rewards.py](/home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:1185)
- Unsafe termination và transition timeout chịu **cost một lần −60** ở phase A/B; vào SAFE_RECOVERY chịu cost sự kiện ở phase C. Cost được chuẩn hóa theo `step_dt`. Reward độ cao chỉ hoàn lại phần đã mất trong TRANSITION, chặn đường farm khi rơi ở YAW rồi quay lại. [reward config](/home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:578)

**Xác minh:** 61 CPU test pass; `git diff --check` sạch. Chưa chạy simulator hoặc PPO, nên tác động lên transition duration và tỷ lệ vào YAW cần được xác nhận ở run tiếp theo.

## User

Validate all recent `Flat-VQR-Wheel-Yaw-FSM` fixes together in one pass. Do not modify code unless validation reveals a clear defect.

Check these issues:

1. Encoder bias reset
   - confirm bias no longer accumulates across episode resets
   - verify bounded distribution after many resets

2. TRANSITION early-failure incentive
   - compare discounted full returns from the same initial condition for:
     a) successful safe transition,
     b) continued safe acquisition for 1–2 s,
     c) failure/unsafe termination at ~0.18 s
   - require: success > continued safe acquisition > early failure
   - test both POS and NEG

3. Acquisition reward topology
   - stationary TRANSITION must not farm reward
   - TRANSITION↔YAW chatter must not reset/farm progress
   - support, CoM, and lift only receive bounded new-maximum credit
   - lift/CoM must not earn meaningful credit while support collapses
   - down→recover cycles must not produce positive net reward

4. Height gradient
   - no entry/reset reward spike
   - stationary height ≈ 0
   - descending gives monotonic negative signal
   - recovery gives bounded positive credit
   - closer to 0.35 m gives stronger descent penalty
   - verify actual weighted magnitude after RewardManager `dt=0.02`

5. Stable FOUR entry
   - verify whether FOUR→TRANSITION currently requires continuous `four_stand_ready` dwell
   - if not, flag it as unresolved; do not silently fix it

6. Watchdogs/accounting
   - transition_time survives TRANSITION↔YAW chatter
   - failed attempts stay in curriculum denominator
   - prior brief YAW success is revoked by terminal reacquisition failure
   - POS/NEG accounting remains symmetric

7. Unsafe telemetry
   - verify torso-contact, low-height, and tilt causes are distinguishable
   - report which cause actually dominates in a short rollout

Run all focused CPU tests and, if Isaac Sim is available, run a short rollout/training validation.

Report:
- PASS / PARTIAL / FAIL for each item
- exact file/line evidence
- numerical return comparisons
- any remaining blocker before the next long training run

Do not tune weights or redesign rewards during validation.

## Assistant

I’ll validate the current FSM code as a single system, including numerical return comparisons for POS and NEG. I’ll run the focused CPU suite first, then check whether the installed Isaac Sim environment can support a short rollout. I’ll leave the code unchanged unless a test exposes a concrete defect.

## Activity

    $ codegraph explore "Flat-VQR-Wheel-Yaw-FSM encoder bias reset transition early failure reward watchdog accounting unsafe telemetry short rollout"
    **Dynamic boundaries (the static path ends at runtime dispatch)**
    
    - `test_failure_cost_is_one_episode_scale_event_and_waiting_beats_early_failure` (tests/test_yaw_fsm_systemic.py:163) — computed member call: `pulse = rewards["fsm_failure_cost"](env)`
      candidates for key `fsm_failure_cost`: `fsm_failure_cost` (source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:1293) ← you named this
    - `test_watchdog_survives_chatter_pauses_in_yaw_and_resets_only_at_maneuver_end` (tests/test_yaw_fsm_systemic.py:204) — computed member call: `if namespace["fsm_transition_timeout"](env).all():` (+1 more such site in this body)
      candidates for key `fsm_transition_timeout`: `fsm_transition_timeout` (source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py:177)
    
    > These sites choose their call target at runtime (registry / bus / reflection) — the site shown IS where the flow continues. To follow it, run codegraph_explore or codegraph_node on a candidate; source for the sites above is included below.
    
    **Interface dispatch (a named method has many implementations)**
    
    - `reset` → runtime dispatch to **9** types implementing `ManagerTermBase` — the static path ends here, the target is chosen at runtime. e.g. `TransitionHeightProgress.reset` (source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:315), `ReturnToFourLanding.reset` (source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:1137), `TransitionProgress.reset` (source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:1195), `SustainedWheelTorquePenalty.reset` (source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/mdp/rewards.py:1649) +5 more
    
    > The method above is dispatched at runtime to one of the listed implementations (a registry / plugin / strategy interface) — there is no single static caller→callee edge; the implementations ARE the continuations. To follow one, run codegraph_explore on a listed target.
    
    > Full source for these symbols is below — the call flow among them, followed by their bodies.
    **Exploration: Flat-VQR-Wheel-Yaw-FSM encoder bias reset transition early failure reward watchdog accounting unsafe telemetry short rollout**
    
    Found 27 symbols across 5 files.
    
    **Blast radius — what depends on these (update/verify before editing)**
    
    - `VQRWheelFSMRewardsCfg` (source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:284) — 1 caller in `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py`; no tests found within 3 caller hops
    - `RewardsCfg` (source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/balance_env_cfg.py:237) — 8 callers in `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/deeprobotics_m20/rough_env_cfg.py`, `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/rotate_env_cfg.py`, `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr_wheel/rough_env_cfg.py`, `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr_wheel/rough_env_cfg_imu.py` +4 more; no tests found within 3 caller hops
    - `TransitionCurriculumCfg` (source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/four_to_two_rotate_env_cfg.py:201) — 1 caller in `source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/four_to_two_rotate_env_cfg.py`; no tests found within 3 caller hops
    
    **Source Code**
    
    > The code below is the **verbatim, current on-disk source** of these files — re-read from disk on this call and line-numbered, byte-for-byte identical to what the Read tool returns. It is NOT a summary, outline, or stale cache. Treat each block as a Read you have already performed: do not Read a file shown here.
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py`** — VQRYawFSM(class), __init__(method), update(method)
    
    ```python
    75        )
    76
    77
    78    class VQRYawFSM:
    79        def __init__(
    80            self,
    81            yaw_enter=0.10,
    82            yaw_exit=0.05,
    83            dt=0.02,
    84            yaw_min_dwell=0.20,
    85            yaw_pose_loss_grace=0.10,
    86            recovery_dwell=0.5,
    87            yaw_pose_ready_dwell=0.10,
    88        ):
    89            self.yaw_enter = yaw_enter
    90            self.yaw_exit = yaw_exit
    91            self.dt = dt
    92            self.yaw_min_dwell = yaw_min_dwell
    93            if yaw_pose_ready_dwell < 0.0:
    94                raise ValueError("yaw_pose_ready_dwell must be non-negative.")
    95            self.yaw_pose_ready_dwell = yaw_pose_ready_dwell
    96            if yaw_pose_loss_grace < 0.0:
    97                raise ValueError("yaw_pose_loss_grace must be non-negative.")
    98            self.yaw_pose_loss_grace = yaw_pose_loss_grace
    99            self.recovery_dwell = recovery_dwell
    100            self.state = VQRFsmState.FOUR_STAND
    101            self.state_time = 0.0
    102            self.transition_time = 0.0
    103            self.just_returned_to_four = False
    104            self._recovery_safe_time = 0.0
    105            self._yaw_pose_invalid_time = 0.0
    106            self._yaw_pose_ready_time = 0.0
    107
    108        def update(
    109            self,
    110            yaw_cmd: float,
    111            positive_pose_ready: bool,
    112            negative_pose_ready: bool,
    113            four_stand_ready: bool,
    114            unsafe: bool,
    115        ):
    116            previous = self.state
    117            self.just_returned_to_four = False
    118            if unsafe:
    119                self.state = VQRFsmState.SAFE_RECOVERY
    120                self._recovery_safe_time = 0.0
    121            elif self.state == VQRFsmState.SAFE_RECOVERY:
    122                if four_stand_ready:
    123                    self._recovery_safe_time += self.dt
    124                else:
    125                    self._recovery_safe_time = 0.0
    126
    127                if self._recovery_safe_time >= self.recovery_dwell:
    128                    self.state = VQRFsmState.FOUR_STAND
    129                    self._recovery_safe_time = 0.0
    130
    131            elif self.state == VQRFsmState.FOUR_STAND:
    132                if yaw_cmd > self.yaw_enter:
    133                    self.state = VQRFsmState.TRANSITION_POS
    134
    135                elif yaw_cmd < -self.yaw_enter:
    136                    self.state = VQRFsmState.TRANSITION_NEG
    137
    138            elif self.state == VQRFsmState.TRANSITION_POS:
    139                if yaw_cmd < self.yaw_exit:
    140                    self.state = VQRFsmState.RETURN_TO_4
    141                else:
    142                    self._yaw_pose_ready_time = (
    143                        self._yaw_pose_ready_time + self.dt if positive_pose_ready else 0.0
    144                    )
    145                    if positive_pose_ready and self._yaw_pose_ready_time >= self.yaw_pose_ready_dwell - 1.0e-6:
    146                        self.state = VQRFsmState.YAW_POS
    147
    148            elif self.state == VQRFsmState.YAW_POS:
    149                if yaw_cmd < self.yaw_exit:
    150                    self.state = VQRFsmState.RETURN_TO_4
    151                elif positive_pose_ready:
    152                    self._yaw_pose_invalid_time = 0.0
    153                else:
    154                    self._yaw_pose_invalid_time += self.dt
    155                    if self._yaw_pose_invalid_time >= self.yaw_pose_loss_grace - 1.0e-6:
    156                        self.state = VQRFsmState.TRANSITION_POS
    157
    158            elif self.state == VQRFsmState.TRANSITION_NEG:
    159                if yaw_cmd > -self.yaw_exit:
    160                    self.state = VQRFsmState.RETURN_TO_4
    161                else:
    162                    self._yaw_pose_ready_time = (
    163                        self._yaw_pose_ready_time + self.dt if negative_pose_ready else 0.0
    164                    )
    165                    if negative_pose_ready and self._yaw_pose_ready_time >= self.yaw_pose_ready_dwell - 1.0e-6:
    166                        self.state = VQRFsmState.YAW_NEG
    167
    168            elif self.state == VQRFsmState.YAW_NEG:
    169                if yaw_cmd > -self.yaw_exit:
    170                    self.state = VQRFsmState.RETURN_TO_4
    171                elif negative_pose_ready:
    172                    self._yaw_pose_invalid_time = 0.0
    173                else:
    174                    self._yaw_pose_invalid_time += self.dt
    175                    if self._yaw_pose_invalid_time >= self.yaw_pose_loss_grace - 1.0e-6:
    176                        self.state = VQRFsmState.TRANSITION_NEG
    177
    178            elif self.state == VQRFsmState.RETURN_TO_4:
    179                if four_stand_ready:
    180                    self.state = VQRFsmState.FOUR_STAND
    181                    self.just_returned_to_four = True
    182
    183            if self.state not in (VQRFsmState.YAW_POS, VQRFsmState.YAW_NEG):
    184                self._yaw_pose_invalid_time = 0.0
    185            if self.state not in (VQRFsmState.TRANSITION_POS, VQRFsmState.TRANSITION_NEG):
    186                self._yaw_pose_ready_time = 0.0
    187            active_states = (VQRFsmState.TRANSITION_POS, VQRFsmState.TRANSITION_NEG,
    188                             VQRFsmState.YAW_POS, VQRFsmState.YAW_NEG)
    189            if self.state not in active_states:
    190                self.transition_time = 0.0
    191            elif previous in (VQRFsmState.TRANSITION_POS, VQRFsmState.TRANSITION_NEG):
    192                self.transition_time += self.dt
    193            self.state_time = 0.0 if self.state != previous else self.state_time + self.dt
    194            return self.state
    195
    196
    197    class YawFSMVectorized:
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/four_to_two_rotate_env_cfg.py`** — instantiates(instantiates), configclass(decorates), TransitionCurriculumCfg(class), VQRFourToTwoWheelRotateEnvCfg(class), extends(extends), TransitionCurriculumCfg(instantiates)
    
    ```python
    197        )
    198
    199
    200    @configclass
    201    class TransitionCurriculumCfg:
    202        transition_levels = CurrTerm(func=transition.transition_levels)
    203
    204
    205    @configclass
    206    class VQRFourToTwoWheelRotateEnvCfg(VQRTwoWheelRotateEnvCfg):
    207        """M3: learn the transition from four-wheel standing before M2 rotation."""
    208
    209        scene: FourWheelStartSceneCfg = FourWheelStartSceneCfg(num_envs=16, env_spacing=2.5)
    210        events: FourWheelStartEventCfg = FourWheelStartEventCfg()
    211        commands: FourToTwoCommandsCfg = FourToTwoCommandsCfg()
    212        observations: FourToTwoObservationsCfg = FourToTwoObservationsCfg()
    213        actions: FourToTwoActionsCfg = FourToTwoActionsCfg()
    214        rewards: FourToTwoRewardsCfg = FourToTwoRewardsCfg()
    215        terminations: FourToTwoTerminationsCfg = FourToTwoTerminationsCfg()
    216        curriculum: TransitionCurriculumCfg = TransitionCurriculumCfg()
    217        episode_length_s = 12.0
    218
    219        def __post_init__(self):
    220            super().__post_init__()
    221
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/agents/rsl_rl_ppo_cfg.py`** — VQRWheelYawFlatFSMPPORunnerCfg(class), VQRWheelYawFlatPPORunnerCfg(class), VQRWheelFlatPPORunnerCfg(class), VQRWheelRoughPPORunnerCfg(class), __post_init__(method)
    
    ```python
    1    # Copyright (c) 2025 Deep Robotics
    2    # SPDX-License-Identifier: BSD 3-Clause
    3
    4    # Copyright (c) 2024-2025 Ziqi Fan
    5    # SPDX-License-Identifier: Apache-2.0
    6
    7    from isaaclab.utils import configclass
    8    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
    9
    10
    11    @configclass
    12    class VQRWheelRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    13        num_steps_per_env = 24
    14        max_iterations = 20000
    15        save_interval = 500
    16        experiment_name = "vqr_wheel_rough"
    17        empirical_normalization = False
    18        clip_actions = 100
    19        policy = RslRlPpoActorCriticCfg(
    20            init_noise_std=1.0,
    21            noise_std_type="log",
    22            actor_hidden_dims=[512, 256, 128],
    23            critic_hidden_dims=[512, 256, 128],
    24            activation="elu",
    25        )
    26        algorithm = RslRlPpoAlgorithmCfg(
    27            value_loss_coef=1.0,
    28            use_clipped_value_loss=True,
    29            clip_param=0.2,
    30            entropy_coef=0.01,
    31            num_learning_epochs=5,
    32            num_mini_batches=4,
    33            learning_rate=1.0e-3,
    34            schedule="adaptive",
    35            gamma=0.99,
    36            lam=0.95,
    37            desired_kl=0.01,
    38            max_grad_norm=1.0,
    39        )
    40
    41
    42    @configclass
    43    class VQRWheelIMURoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    44        num_steps_per_env = 24
    45        max_iterations = 20000
    46        save_interval = 500
    47        experiment_name = "vqr_wheel_imu_rough"
    48        empirical_normalization = False
    49        clip_actions = 100
    50        policy = RslRlPpoActorCriticCfg(
    51            init_noise_std=1.0,
    52            noise_std_type="log",
    53            actor_hidden_dims=[512, 256, 128],
    54            critic_hidden_dims=[512, 256, 128],
    55            activation="elu",
    56        )
    57        algorithm = RslRlPpoAlgorithmCfg(
    58            value_loss_coef=1.0,
    59            use_clipped_value_loss=True,
    60            clip_param=0.2,
    61            entropy_coef=0.01,
    62            num_learning_epochs=5,
    63            num_mini_batches=4,
    64            learning_rate=1.0e-3,
    65            schedule="adaptive",
    66            gamma=0.99,
    67            lam=0.95,
    68            desired_kl=0.01,
    69            max_grad_norm=1.0,
    70        )
    71
    72    @configclass
    73    class VQRWheelFlatPPORunnerCfg(VQRWheelRoughPPORunnerCfg):
    74        def __post_init__(self):
    75            super().__post_init__()
    76
    77            self.max_iterations = 5000
    78            self.experiment_name = "vqr_wheel_flat"
    79
    80
    81    @configclass
    82    class VQRWheelYawFlatPPORunnerCfg(VQRWheelFlatPPORunnerCfg):
    83        """Runner settings for the flat yaw-rate task."""
    84
    85        def __post_init__(self):
    86            super().__post_init__()
    87            self.experiment_name = "vqr_wheel_yaw_flat"
    88            # The yaw/DR curriculum has six yaw stages and needs a longer default
    89            # horizon than the generic flat locomotion task.
    90            self.max_iterations = 30000
    91
    92
    93    @configclass
    94    class VQRWheelYawFlatFSMPPORunnerCfg(VQRWheelYawFlatPPORunnerCfg):
    95        """Runner settings with an isolated experiment directory for the FSM task."""
    96
    97        def __post_init__(self):
    98            super().__post_init__()
    99            self.experiment_name = "vqr_wheel_yaw_flat_fsm"
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr_wheel/rough_env_cfg_imu.py`** — imports(imports), configclass(decorates), SceneEntityCfg(instantiates), SceneEntityCfg(imports), configclass(imports), ActionsCfg(imports), +11 more
    
    ```python
    4    # Copyright (c) 2024-2025 Ziqi Fan
    5    # SPDX-License-Identifier: Apache-2.0
    6
    7    from isaaclab.managers import RewardTermCfg as RewTerm
    8    from isaaclab.managers import SceneEntityCfg
    9    from isaaclab.sensors import ImuCfg
    10    from isaaclab.terrains import MeshPlaneTerrainCfg
    11    from isaaclab.utils import configclass
    12    from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
    13    from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg
    14
    15    import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
    16    from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    17        ActionsCfg,
    18        LocomotionVelocityRoughEnvCfg,
    19        MySceneCfg,
    20        RewardsCfg,
    21    )
    22
    23    ##
    24    # Pre-defined configs
    25    ##
    26    from rl_training.assets.deeprobotics import VQRWHEEL_CFG  # isort: skip
    27
    28
    29    @configclass
    
    ... (gap) ...
    
    41        )
    42
    43
    44    @configclass
    45    class VQRWheelSceneCfg(MySceneCfg):
    46        """Scene specifications for the MDP."""
    47
    48        # real IMU mounting pose relative to TORSO: x=237mm, y=0, z=44mm, yaw=+90deg
    49        imu = ImuCfg(
    50            prim_path="{ENV_REGEX_NS}/Robot/base",
    51            offset=ImuCfg.OffsetCfg(
    52                pos=(0.237, 0.0, 0.044),
    53                rot=(0.7071068, 0.0, 0.0, 0.7071068),
    54            ),
    55        )
    56
    57
    58    @configclass
    59    class VQRWheelRewardsCfg(RewardsCfg):
    60        """Reward terms for the MDP."""
    61
    62        joint_vel_wheel_l2 = RewTerm(
    63            func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    64        )
    65
    66        joint_acc_wheel_l2 = RewTerm(
    67            func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    68        )
    69
    70        joint_torques_wheel_l2 = RewTerm(
    71            func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    72        )
    73
    74
    75    @configclass
    76    class VQRWheelRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    77        scene: VQRWheelSceneCfg = VQRWheelSceneCfg(num_envs=4096, env_spacing=2.5)
    78        actions: VQRWheelActionsCfg = VQRWheelActionsCfg()
    79        rewards: VQRWheelRewardsCfg = VQRWheelRewardsCfg()
    80
    81        base_link_name = "TORSO"
    82        foot_link_name = ".*_WHEEL"
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/balance_env_cfg.py`** — SceneEntityCfg(instantiates), configclass(decorates), instantiates(instantiates), TwoWheelBalanceSceneCfg(class), ActionsCfg(class), ObservationsCfg(class), +10 more
    
    ```python
    58    WHEEL_RADIUS = 0.091
    59
    60
    61    @configclass
    62    class TwoWheelBalanceSceneCfg(InteractiveSceneCfg):
    63        """Flat plane, the audited VQR articulation, and one contact sensor."""
    64
    
    ... (gap) ...
    
    88        )
    89
    90
    91    @configclass
    92    class ActionsCfg:
    93        """Twelve leg position residuals followed by four wheel velocities."""
    94
    95        leg_positions = mdp.JointPositionActionCfg(
    96            asset_name="robot",
    97            joint_names=LEG_JOINT_NAMES,
    98            scale=0.18,
    99            offset=NOMINAL_LEG_POSITION_MAP,
    100            use_default_offset=False,
    101            preserve_order=True,
    102        )
    103        wheel_velocities = mdp.JointVelocityActionCfg(
    104            asset_name="robot",
    105            joint_names=WHEEL_JOINT_NAMES,
    106            scale=4.0,
    107            use_default_offset=True,
    108            preserve_order=True,
    109        )
    110
    111
    112    @configclass
    113    class ObservationsCfg:
    114        """Deployable policy observations and a separate privileged critic group."""
    115
    116        @configclass
    117        class PolicyCfg(ObsGroup):
    118            base_ang_vel = ObsTerm(
    119                func=mdp.base_ang_vel,
    
    ... (gap) ...
    
    128            yaw_rate_command = None
    129            leg_joint_pos = ObsTerm(
    130                func=mdp.joint_pos_rel,
    131                params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    132                noise=Unoise(n_min=-0.01, n_max=0.01),
    133            )
    134            leg_joint_vel = ObsTerm(
    135                func=mdp.joint_vel_rel,
    136                params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    137                noise=Unoise(n_min=-0.10, n_max=0.10),
    138                scale=0.05,
    139            )
    140            wheel_joint_vel = ObsTerm(
    141                func=mdp.joint_vel_rel,
    142                params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES, preserve_order=True)},
    143                noise=Unoise(n_min=-0.10, n_max=0.10),
    144                scale=0.05,
    145            )
    
    ... (gap) ...
    
    233        )
    234
    235
    236    @configclass
    237    class RewardsCfg:
    238        """Seven task-specific rewards; no locomotion reward set is inherited."""
    239
    240        support_contact = RewTerm(
    241            func=mdp.pivot_support_contact,
    242            weight=1.0,
    243            params={
    244                "sensor_cfg": SceneEntityCfg(
    245                    "contact_forces", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
    246                ),
    247                "threshold": 1.0,
    248            },
    249        )
    250        lifted_diagonal = RewTerm(
    251            func=mdp.pivot_lifted_wheels,
    252            weight=1.0,
    253            params={
    254                "sensor_cfg": SceneEntityCfg(
    255                    "contact_forces", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
    256                ),
    257                "asset_cfg": SceneEntityCfg("robot", body_names=LIFTED_WHEEL_NAMES, preserve_order=True),
    258                "wheel_radius": WHEEL_RADIUS,
    259                "minimum_clearance": 0.05,
    260                "threshold": 1.0,
    261            },
    262        )
    263        balance = RewTerm(
    264            func=mdp.pivot_balance,
    265            weight=2.0,
    266            params={
    267                "nominal_roll": NOMINAL_ROLL,
    268                "nominal_pitch": NOMINAL_PITCH,
    269                "std": 0.25,
    270            },
    271        )
    272        angular_stability = RewTerm(func=mdp.pivot_angular_stability, weight=-0.10)
    273        planar_drift = RewTerm(func=mdp.pivot_planar_drift, weight=-0.50)
    274        action_rate = RewTerm(func=mdp.pivot_action_rate_l2, weight=-0.01)
    275        leg_effort = RewTerm(
    276            func=mdp.pivot_leg_effort_l2,
    277            weight=-1.0e-5,
    278            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    279        )
    280
    281
    282    @configclass
    
    ... (gap) ...
    
    305        drift = DoneTerm(func=mdp.pivot_drifted_away, params={"maximum_distance": 1.0})
    306
    307
    308    @configclass
    309    class VQRTwoWheelBalanceEnvCfg(ManagerBasedRLEnvCfg):
    310        """M1 balance task configuration."""
    311
    312        decimation = 4
    313        episode_length_s = 10.0
    314        sim = sim_utils.SimulationCfg(dt=0.005, render_interval=decimation, device="cuda:0")
    315        scene = TwoWheelBalanceSceneCfg(num_envs=16, env_spacing=2.5)
    316        observations = ObservationsCfg()
    317        actions = ActionsCfg()
    318        commands = None
    319        rewards = RewardsCfg()
    320        terminations = TerminationsCfg()
    321        events = EventCfg()
    322        curriculum = None
    323
    324        def __post_init__(self):
    325            self.sim.physics_material = self.scene.terrain.physics_material
    326            self.scene.contact_forces.update_period = self.sim.dt
    327            self.sim.physx.gpu_max_rigid_patch_count = 2**19
    328
    ```
    
    **Not shown above — explore these names for their source**
    
    - source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py: VQRWheelFSMRewardsCfg:284, VQRWheelFlatEnvFSMCfg:1056
    - source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/velocity_env_cfg.py: LocomotionVelocityRoughEnvCfg:839, MySceneCfg:48, create_obsgroup_class:903, disable_zero_weight_rewards:894
    - source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/quadruped/vqr/flat_env_cfg.py: VQRFlatEnvCfg:13, __post_init__:14
    - source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/quadruped/vqr/flat_env_cfg.py: VQRFlatEnvCfg:13, __post_init__:14
    - source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/deeprobotics_m20/rough_env_cfg.py: DeeproboticsM20ActionsCfg:25, DeeproboticsM20RewardsCfg:38, DeeproboticsM20RoughEnvCfg:55, rough_env_cfg.py:1
    - source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/rotate_env_cfg.py: RotateRewardsCfg:62, rotate_env_cfg.py:1
    - source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr_wheel/rough_env_cfg.py: VQRWheelRewardsCfg:43, rough_env_cfg.py:1
    - source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/rough_env_cfg.py: DeeproboticsM20RewardsCfg:38, rough_env_cfg.py:1
    - source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/rough_env_cfg.py: VQRWheelRewardsCfg:43, rough_env_cfg.py:1
    - source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/rough_env_cfg_imu.py: VQRWheelRewardsCfg:59, rough_env_cfg_imu.py:1
    - ... and 10 more files
    
    > Some file sections were trimmed for size. For a specific symbol you still need, run another `codegraph_explore` (or `codegraph_node`) with its exact name — line-numbered source, cheaper and more complete than Read.
    ✓ • 311ms

## Activity

    $ codegraph explore "NoiseModelWithAdditiveBias reset encoder bias yaw_env_fsm_cfg.py"
    **Exploration: NoiseModelWithAdditiveBias reset encoder bias yaw_env_fsm_cfg.py**
    
    Found 84 symbols across 5 files. 1 file pinned from the query.
    
    **Blast radius — what depends on these (update/verify before editing)**
    
    - `reset` (source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:1195) — 1 caller; tests: `tests/test_yaw_fsm_systemic.py`
    
    **Source Code**
    
    > The code below is the **verbatim, current on-disk source** of these files — re-read from disk on this call and line-numbered, byte-for-byte identical to what the Read tool returns. It is NOT a summary, outline, or stale cache. Treat each block as a Read you have already performed: do not Read a file shown here.
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py`** — instantiates(instantiates), decorates(decorates), extends(extends), __post_init__(method), __post_init__(calls), VQRWheelActionsCfg(class), +28 more
    
    ```python
    69    # curriculum scales reset/interval (online) disturbances together with yaw.
    70
    71
    72    @configclass
    73    class VQRWheelActionsCfg(ActionsCfg):
    74        """Action specifications for the MDP."""
    75
    76        joint_pos = mdp.JointPositionActionCfg(
    
    ... (gap) ...
    
    84        )
    85
    86
    87    @configclass
    88    class VQRWheelRewardsCfg:
    89        """The explicit reward set for diagonal-support rotate-in-place training."""
    90
    91        com_support = RewTerm(
    92            func=mdp.yaw_com_support,
    93            weight=3.0,
    94            params={
    95                "asset_cfg": SceneEntityCfg(
    96                    "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
    97                ),
    98                "std": 0.08,
    
    ... (gap) ...
    
    102            func=mdp.yaw_base_height_tracking,
    103            weight=2.0,
    104            params={
    105                "asset_cfg": SceneEntityCfg("robot"),
    106                "target_height": TARGET_BASE_HEIGHT,
    107                "error_scale": 0.10,
    108            },
    
    ... (gap) ...
    
    111            func=mdp.yaw_low_base_height_l1,
    112            weight=-4.0,
    113            params={
    114                "asset_cfg": SceneEntityCfg("robot"),
    115                "minimum_height": MIN_BASE_HEIGHT,
    116                "error_scale": 0.10,
    117            },
    
    ... (gap) ...
    
    120            func=mdp.yaw_downward_low_base_velocity_l2,
    121            weight=-8.0,
    122            params={
    123                "asset_cfg": SceneEntityCfg("robot"),
    124                "minimum_height": MIN_BASE_HEIGHT,
    125                "height_margin": 0.10,
    126            },
    
    ... (gap) ...
    
    129            func=mdp.yaw_support_span_band_l2,
    130            weight=-1.0,
    131            params={
    132                "asset_cfg": SceneEntityCfg(
    133                    "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
    134                ),
    135                "minimum_span": SUPPORT_SPAN_MIN,
    
    ... (gap) ...
    
    141            func=mdp.yaw_lift_clearance,
    142            weight=3.0,
    143            params={
    144                "asset_cfg": SceneEntityCfg(
    145                    "robot", body_names=LIFTED_WHEEL_NAMES, preserve_order=True
    146                ),
    147                "wheel_radius": WHEEL_RADIUS,
    
    ... (gap) ...
    
    152            func=mdp.yaw_com_inside_support_segment,
    153            weight=2.0,
    154            params={
    155                "asset_cfg": SceneEntityCfg(
    156                    "robot", body_names=SUPPORT_WHEEL_NAMES, preserve_order=True
    157                ),
    158                "std": 0.05,
    
    ... (gap) ...
    
    173            weight=8.0,
    174            params={
    175                "command_name": "yaw_rate_cmd",
    176                "support_sensor_cfg": SceneEntityCfg(
    177                    "contact_forces",
    178                    body_names=SUPPORT_WHEEL_NAMES,
    179                    preserve_order=True,
    180                ),
    181                "lifted_asset_cfg": SceneEntityCfg(
    182                    "robot",
    183                    body_names=LIFTED_WHEEL_NAMES,
    184                    preserve_order=True,
    
    ... (gap) ...
    
    195            func=mdp.yaw_lateral_wheel_slip,
    196            weight=-2.0,
    197            params={
    198                "sensor_cfg": SceneEntityCfg(
    199                    "contact_forces", body_names=WHEEL_NAMES, preserve_order=True
    200                ),
    201                "asset_cfg": SceneEntityCfg(
    202                    "robot", body_names=WHEEL_NAMES, preserve_order=True
    203                ),
    204                "threshold": 1.0,
    
    ... (gap) ...
    
    208            func=mdp.yaw_rolling_wheel_slip,
    209            weight=-0.5,
    210            params={
    211                "sensor_cfg": SceneEntityCfg(
    212                    "contact_forces", body_names=WHEEL_NAMES, preserve_order=True
    213                ),
    214                "body_asset_cfg": SceneEntityCfg(
    215                    "robot", body_names=WHEEL_NAMES, preserve_order=True
    216                ),
    217                "joint_asset_cfg": SceneEntityCfg(
    218                    "robot", joint_names=WHEEL_NAMES, preserve_order=True
    219                ),
    220                "wheel_radius": WHEEL_RADIUS,
    
    ... (gap) ...
    
    227            func=mdp.undesired_contacts,
    228            weight=-2.0,
    229            params={
    230                "sensor_cfg": SceneEntityCfg(
    231                    "contact_forces",
    232                    body_names=["^(?!(TORSO|.*_WHEEL)$).*"],
    233                    preserve_order=True,
    
    ... (gap) ...
    
    239            func=mdp.joint_pos_limits,
    240            weight=-0.5,
    241            params={
    242                "asset_cfg": SceneEntityCfg(
    243                    "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
    244                )
    245            },
    
    ... (gap) ...
    
    249            func=mdp.yaw_joint_velocity_l2,
    250            weight=-0.001,
    251            params={
    252                "asset_cfg": SceneEntityCfg(
    253                    "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
    254                )
    255            },
    
    ... (gap) ...
    
    261                "command_name": "yaw_rate_cmd",
    262                "yaw_reference": YAW_REF,
    263                "minimum_scale": 0.30,
    264                "asset_cfg": SceneEntityCfg(
    265                    "robot", joint_names=LEG_JOINT_NAMES + WHEEL_NAMES, preserve_order=True
    266                )
    267            },
    
    ... (gap) ...
    
    272            params={
    273                "command_name": "yaw_rate_cmd",
    274                "yaw_reference": YAW_REF,
    275                "asset_cfg": SceneEntityCfg(
    276                    "robot", joint_names=LIFTED_WHEEL_NAMES, preserve_order=True
    277                )
    278            },
    279        )
    280        planar_velocity = RewTerm(func=mdp.yaw_planar_velocity_l2, weight=-1.0)
    281
    282
    283    @configclass
    284    class VQRWheelFSMRewardsCfg:
    285        """The reward contract for ``Flat-VQR-Wheel-Yaw-FSM``.
    286
    
    ... (gap) ...
    
    306                "command_name": "yaw_rate_cmd",
    307                "yaw_reference": YAW_REF,
    308                "minimum_scale": 0.30,
    309                "asset_cfg": SceneEntityCfg(
    310                    "robot", joint_names=LEG_JOINT_NAMES + WHEEL_NAMES, preserve_order=True
    311                ),
    312            },
    
    ... (gap) ...
    
    316            func=mdp.yaw_joint_velocity_l2,
    317            weight=-0.001,
    318            params={
    319                "asset_cfg": SceneEntityCfg(
    320                    "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
    321                )
    322            },
    
    ... (gap) ...
    
    325            func=mdp.joint_pos_limits,
    326            weight=-0.5,
    327            params={
    328                "asset_cfg": SceneEntityCfg(
    329                    "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True
    330                )
    331            },
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py`** — ManagerTermBase(extends), __init__(method), __init__(calls), reset(method), __call__(method), TorsoContactWithGrace(class), +3 more
    
    ```python
    20        from isaaclab.envs import ManagerBasedRLEnv
    21
    22
    23    class TorsoContactWithGrace(ManagerTermBase):
    24        """Terminate selected torso contact after a reset-relative grace period.
    25
    26        The elapsed-time buffer is owned by this term and reset through the
    27        termination manager. It deliberately does not use ``episode_length_buf``,
    28        which may be randomized at startup by an RL runner.
    29        """
    30
    31        def __init__(self, cfg, env: ManagerBasedRLEnv):
    32            super().__init__(cfg, env)
    33            self._elapsed_s = torch.zeros(env.num_envs, device=env.device)
    34
    35        def reset(self, env_ids: Sequence[int] | None = None) -> None:
    36            if env_ids is None:
    37                self._elapsed_s.zero_()
    38            else:
    39                self._elapsed_s[env_ids] = 0.0
    40
    41        def __call__(
    42            self,
    43            env: ManagerBasedRLEnv,
    44            sensor_cfg: SceneEntityCfg,
    45            threshold: float,
    46            grace_period_s: float,
    47        ) -> torch.Tensor:
    48            if threshold < 0.0:
    49                raise ValueError("threshold must be non-negative.")
    50            if grace_period_s < 0.0:
    51                raise ValueError("grace_period_s must be non-negative.")
    52
    53            contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    54            force_history = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    55            torso_contact = torch.linalg.vector_norm(force_history, dim=-1).amax(dim=1).amax(dim=1) > threshold
    56            grace_finished = self._elapsed_s + 1.0e-6 >= grace_period_s
    57            terminated = grace_finished & torso_contact
    58            self._elapsed_s += env.step_dt
    59            return terminated
    60
    61
    62    class FSMUnsafeWithGrace(ManagerTermBase):
    63        """Phase-aware unsafe termination using the live FSM safety predicate.
    64
    65        Phases A/B terminate after the short reset grace.  Phase C intentionally
    
    ... (gap) ...
    
    69        old when the termination manager runs.
    70        """
    71
    72        def __init__(self, cfg, env: ManagerBasedRLEnv):
    73            super().__init__(cfg, env)
    74            self._elapsed_s = torch.zeros(env.num_envs, device=env.device)
    75
    76        def reset(self, env_ids: Sequence[int] | None = None) -> None:
    77            if env_ids is None:
    78                self._elapsed_s.zero_()
    79            else:
    80                self._elapsed_s[env_ids] = 0.0
    81
    82        def __call__(
    83            self,
    
    ... (gap) ...
    
    92        ) -> torch.Tensor:
    93            if grace_period_s < 0.0:
    94                raise ValueError("grace_period_s must be non-negative.")
    95            torso, height, tilt = yaw_fsm_unsafe_components(
    96                env,
    97                robot_name=robot_name,
    98                torso_sensor_cfg=torso_sensor_cfg,
    
    ... (gap) ...
    
    113            return grace_finished & unsafe
    114
    115
    116    class SwingContactTimeout(ManagerTermBase):
    117        """Terminate persistent swing-wheel contact during either YAW state.
    118
    119        Contact timers are independent for the two swing wheels.  They are reset
    120        as soon as the FSM leaves YAW or that wheel loses contact, making the
    121        timeout a continuous-contact condition rather than an episode counter.
    122        """
    123
    124        def __init__(self, cfg, env: ManagerBasedRLEnv):
    125            super().__init__(cfg, env)
    126            self._contact_time = torch.zeros(env.num_envs, 2, device=env.device)
    127
    128        def reset(self, env_ids: Sequence[int] | None = None) -> None:
    129            if env_ids is None:
    130                self._contact_time.zero_()
    131            else:
    132                self._contact_time[env_ids] = 0.0
    133
    134        def __call__(
    135            self,
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py`** — ManagerTermBase(extends), __init__(method), __init__(calls), reset(method), __call__(method), fsm_gates(calls), +5 more
    
    ```python
    303        return torch.relu(minimum_height - base_height) / error_scale
    304
    305
    306    class TransitionHeightProgress(ManagerTermBase):
    307        """Pay changes in a bounded safety margin only during TRANSITION."""
    308
    309        def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
    310            super().__init__(cfg, env)
    311            self.prev_potential = torch.zeros(env.num_envs, device=env.device)
    312            self.was_transition = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    313            self.loss_budget = torch.zeros(env.num_envs, device=env.device)
    314
    315        def reset(self, env_ids=None) -> None:
    316            if env_ids is None:
    317                self.prev_potential.zero_()
    318                self.was_transition.zero_()
    319                self.loss_budget.zero_()
    320            else:
    321                self.prev_potential[env_ids] = 0.0
    322                self.was_transition[env_ids] = False
    323                self.loss_budget[env_ids] = 0.0
    324
    325        def __call__(
    326            self,
    
    ... (gap) ...
    
    1125        return reward
    1126
    1127
    1128    class ReturnToFourLanding(ManagerTermBase):
    1129        """Reward lowering progress and each regained contact once per return."""
    1130
    1131        def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
    1132            super().__init__(cfg, env)
    1133            self.prev_progress = torch.zeros(env.num_envs, 2, device=env.device)
    1134            self.contact_seen = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
    1135            self.was_return = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    1136
    1137        def reset(self, env_ids=None) -> None:
    1138            index = slice(None) if env_ids is None else env_ids
    1139            self.prev_progress[index] = 0.0
    1140            self.contact_seen[index] = False
    1141            self.was_return[index] = False
    1142
    1143        def __call__(
    1144            self,
    1145            env: ManagerBasedRLEnv,
    1146            asset_cfg: SceneEntityCfg,
    1147            asset_cfg_mirror: SceneEntityCfg,
    1148            sensor_cfg: SceneEntityCfg,
    1149            sensor_cfg_mirror: SceneEntityCfg,
    1150            wheel_radius: float,
    1151            fsm_command_name: str,
    1152            contact_threshold: float = 1.0,
    1153        ) -> torch.Tensor:
    1154            gates = fsm_gates(env, fsm_command_name)
    1155            target_clearance = env.command_manager.get_term(fsm_command_name).cfg.target_clearance
    1156            progress_pos = _yaw_lift_progress(env, asset_cfg, wheel_radius, target_clearance)
    1157            progress_neg = _yaw_lift_progress(env, asset_cfg_mirror, wheel_radius, target_clearance)
    1158            contact_pos = _yaw_wheel_contacts(env, sensor_cfg, contact_threshold)
    1159            contact_neg = _yaw_wheel_contacts(env, sensor_cfg_mirror, contact_threshold)
    1160            progress = torch.where(gates["diag_pos"].unsqueeze(1), progress_pos, progress_neg)
    1161            contact = torch.where(gates["diag_pos"].unsqueeze(1), contact_pos, contact_neg)
    1162            in_return = gates["b_return"]
    1163            continuing = in_return & self.was_return & ~gates["just_switched"]
    1164            lowering = (self.prev_progress - progress).mean(dim=1) * continuing
    1165            # Contacts present at return entry were never lost and earn no bonus.
    1166            first_contacts = (contact & ~self.contact_seen & continuing.unsqueeze(1)).float().mean(dim=1)
    1167            reward = lowering + 2.0 * first_contacts
    1168            self.prev_progress.copy_(torch.where(in_return.unsqueeze(1), progress, torch.zeros_like(progress)))
    1169            self.contact_seen.copy_(torch.where(
    1170                in_return.unsqueeze(1), self.contact_seen | contact, torch.zeros_like(contact)
    1171            ))
    1172            self.was_return.copy_(in_return)
    1173            _fsm_record_positive_budget(env, gates, "return_to_four_landing", reward)
    1174            return reward
    1175
    1176
    1177    def four_stand_ready_bonus(env: ManagerBasedRLEnv, fsm_command_name: str) -> torch.Tensor:
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/state/hard_state_buffer.py`** — reset(method), HardStateBuffer(class), __init__(method), capacity(method), filled(method), update(method), +1 more
    
    ```python
    1    # Copyright (c) 2026 Deep Robotics
    2    # SPDX-License-Identifier: BSD 3-Clause
    3
    4    """Ring buffer of recent hard states used for the 15% reset bucket.
    5
    6    Section 9: 15% of resets replay a state drawn from the last T - 0.3 s of
    7    training so the policy repeatedly faces the near-tipover states it actually
    8    visited instead of only freshly sampled ones.
    9    """
    10
    11    from __future__ import annotations
    12
    13    import torch
    14
    15    from ..config.wheeled.vqr.physical_params import VQR_PHYSICS
    16
    17
    18    class HardStateBuffer:
    19        """Fixed-horizon ring buffer over arbitrary per-env tensor snapshots."""
    20
    21        def __init__(
    22            self,
    23            num_envs: int,
    24            device: torch.device | str = "cpu",
    25            window: float | None = None,
    26            dt: float | None = None,
    27        ):
    28            dt = VQR_PHYSICS.control_dt if dt is None else dt
    29            horizon = VQR_PHYSICS.hard_state_window if window is None else window
    30            self._length = max(1, int(round(horizon / dt)))
    31            self._index = -1  # nothing stored yet
    32            self._filled = 0
    33            self._states: dict[str, torch.Tensor] = {}
    34            self._num_envs = num_envs
    35            self._device = device
    36
    37        @property
    38        def capacity(self) -> int:
    39            return self._length
    40
    41        @property
    42        def filled(self) -> int:
    43            return min(self._filled, self._length)
    44
    45        def update(self, **snapshots: torch.Tensor) -> None:
    46            """Store one per-env snapshot keyed by name (all ``(N, ...)``)."""
    47            if not snapshots:
    48                return
    49            if not self._states:
    50                self._states = {
    51                    name: torch.zeros(self._length, *tensor.shape, device=tensor.device, dtype=tensor.dtype)
    52                    for name, tensor in snapshots.items()
    53                }
    54            for name, tensor in snapshots.items():
    55                slot = (self._filled % self._length) if self._index < 0 else (self._index + 1) % self._length
    56                self._states[name][slot] = tensor
    57            self._index = (self._index + 1) % self._length
    58            self._filled += 1
    59
    60        def sample(self, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    61            """Uniformly sample one buffered frame; caller slices env ids."""
    62            if self.filled == 0:
    63                raise RuntimeError("HardStateBuffer.sample called before any update()")
    64            slot = int(torch.randint(0, self.filled, (1,), generator=generator).item())
    65            return {name: states[slot] for name, states in self._states.items()}
    66
    67        def reset(self) -> None:
    68            self._states = {}
    69            self._index = -1
    70            self._filled = 0
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py`** — _as_env_vector(calls), __init__(method), state(method), YawFSMVectorized(class), _as_env_vector(method), reset(method), +5 more
    
    ```python
    285            )
    286
    287        @property
    288        def state(self) -> torch.Tensor:
    289            """Alias for ``fsm_state`` matching the scalar FSM interface."""
    290            return self.fsm_state
    291
    292        @state.setter
    293        def state(self, value: torch.Tensor) -> None:
    294            value = self._as_env_vector(value, name="state", dtype=torch.long)
    295            self.fsm_state.copy_(value)
    296
    297        def _as_env_vector(
    298            self,
    
    ... (gap) ...
    
    314                )
    315            return tensor
    316
    317        def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    318            """Reset selected environments to ``FOUR_STAND``."""
    319            if env_ids is None:
    320                index = slice(None)
    321            elif isinstance(env_ids, slice):
    322                index = env_ids
    323            else:
    324                index = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
    325                if index.ndim == 0:
    326                    index = index.unsqueeze(0)
    327
    328            self.fsm_state[index] = int(VQRFsmState.FOUR_STAND)
    329            self.support_diagonal[index] = 0
    330            self.state_time[index] = 0.0
    331            self.transition_time[index] = 0.0
    332            self.just_switched[index] = False
    333            self.just_returned_to_four[index] = False
    334            self._recovery_safe_time[index] = 0.0
    335            self._yaw_pose_invalid_time[index] = 0.0
    336            self._yaw_pose_ready_time[index] = 0.0
    337
    338        def update(
    339            self,
    
    ... (gap) ...
    
    350            enter ``FOUR_STAND`` here; its yaw command is considered on the next
    351            update, preventing a direct ``RETURN_TO_4 -> TRANSITION_*`` edge.
    352            """
    353            yaw = self._as_env_vector(yaw_cmd, name="yaw_cmd", dtype=torch.float64)
    354            positive_ready = self._as_env_vector(
    355                positive_pose_ready, name="positive_pose_ready", dtype=torch.bool
    356            )
    357            negative_ready = self._as_env_vector(
    358                negative_pose_ready, name="negative_pose_ready", dtype=torch.bool
    359            )
    360            four_ready = self._as_env_vector(
    361                four_stand_ready, name="four_stand_ready", dtype=torch.bool
    362            )
    363            unsafe_mask = self._as_env_vector(unsafe, name="unsafe", dtype=torch.bool)
    364
    365            previous = self.fsm_state
    366            next_state = previous.clone()
    
    ... (gap) ...
    
    529            return self.fsm_state
    530
    531
    532    class YawFSMCommand(YawRateCommand):
    533        """Yaw-rate command coupled to the per-environment yaw FSM.
    534
    535        ``VQRYawFSM`` is deliberately kept as the single source of truth for the
    
    ... (gap) ...
    
    545        cfg: "YawFSMCommandCfg"
    546
    547        def __init__(self, cfg: "YawFSMCommandCfg", env):
    548            super().__init__(cfg, env)
    549
    550            self._fsm = YawFSMVectorized(
    551                self.num_envs,
    552                device=self.device,
    553                yaw_enter=cfg.yaw_enter,
    ```
    
    
    > Some file sections were trimmed for size. For a specific symbol you still need, run another `codegraph_explore` (or `codegraph_node`) with its exact name — line-numbered source, cheaper and more complete than Read.
    ✓ • 213ms

## Activity

    $ codegraph explore "yaw_fsm_unsafe_components FSMUnsafeWithGrace transition_time _fsm_attempt_telemetry failed attempts curriculum POS NEG"
    **Exploration: yaw_fsm_unsafe_components FSMUnsafeWithGrace transition_time _fsm_attempt_telemetry failed attempts curriculum POS NEG**
    
    Found 52 symbols across 5 files.
    
    **Blast radius — what depends on these (update/verify before editing)**
    
    - `_fsm_attempt_telemetry` (source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:728) — 1 caller in `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py`; no tests found within 3 caller hops
    - `_fsm_transition_telemetry` (source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:817) — 1 caller in `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py`; no tests found within 3 caller hops
    - `yaw_fsm_unsafe_components` (source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/observations.py:276) — 3 callers in `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/observations.py`, `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py`; no tests found within 3 caller hops
    
    **Source Code**
    
    > The code below is the **verbatim, current on-disk source** of these files — re-read from disk on this call and line-numbered, byte-for-byte identical to what the Read tool returns. It is NOT a summary, outline, or stale cache. Treat each block as a Read you have already performed: do not Read a file shown here.
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py`** — _fsm_masked_accumulate(calls), _yaw_lift_progress(calls), _yaw_wheel_contacts(calls), fsm_gates(calls), _fsm_record_positive_budget(calls), _fsm_masked_accumulate_pair(calls), +22 more
    
    ```python
    611
    612        # Hard safety/task-topology gate:
    613        # 1 only when both FL and HR support wheels are in contact.
    614        support_gate = yaw_support_contact(
    615            env,
    616            support_sensor_cfg,
    617            contact_threshold,
    
    ... (gap) ...
    
    633        # Smooth [0, 1] partial-credit progress toward lifting FR and HL. Keep this
    634        # separate from yaw_lift_clearance, whose signed score penalizes four-wheel
    635        # stance and therefore is not suitable as a multiplicative gate.
    636        clearance_gate = _yaw_lift_progress(
    637            env,
    638            lifted_asset_cfg,
    639            wheel_radius,
    
    ... (gap) ...
    
    669        env._yaw_edge_tracking_samples += edge_mask.long()
    670
    671        # Yaw tracking score in [0, 1].
    672        yaw_tracking = track_yaw_rate_exp(
    673            env,
    674            std,
    675            command_name,
    
    ... (gap) ...
    
    725        getattr(env, samples_name).add_(mask.to(dtype=torch.long))
    726
    727
    728    def _fsm_attempt_telemetry(
    729        env, gates, lift_progress, support_gate,
    730        lift_progress_threshold: float = 0.80, support_threshold: float = 0.85,
    731    ) -> None:
    732        """Count maneuvers, not YAW survivors or repeated reacquisition visits.
    733
    734        Rewards run after termination computation and before command advancement.
    735        Finalize a live attempt on its terminal reward step, or when its FSM enters
    736        RETURN/SAFE. A timeout/fall cannot retain a provisional YAW success.
    737        """
    738        if not hasattr(env, "_yaw_fsm_attempt_diagonal"):
    739            env._yaw_fsm_attempt_diagonal = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    740            env._yaw_fsm_attempt_yaw_samples = torch.zeros_like(env._yaw_fsm_attempt_diagonal)
    741            env._yaw_fsm_attempt_lift_sum = torch.zeros_like(lift_progress)
    742            env._yaw_fsm_attempt_support_sum = torch.zeros_like(support_gate)
    743            for suffix in ("pos", "neg"):
    744                for field in ("transition_attempted", "transition_succeeded", "pose_succeeded",
    745                              "lift_failed", "support_failed"):
    746                    setattr(env, f"_yaw_fsm_{suffix}_{field}", torch.zeros_like(env._yaw_fsm_attempt_diagonal))
    747        active = gates["b_trans"] | gates["b_yaw"]
    748        diagonal = env._yaw_fsm_attempt_diagonal
    749        entering = active & (diagonal == 0)
    750        diagonal.copy_(torch.where(entering, gates["support_diagonal"], diagonal))
    751        for suffix, sign in (("pos", 1), ("neg", -1)):
    752            getattr(env, f"_yaw_fsm_{suffix}_transition_attempted").add_((entering & (diagonal == sign)).long())
    753
    754        yaw = gates["b_yaw"]
    755        env._yaw_fsm_attempt_yaw_samples += yaw.long()
    756        env._yaw_fsm_attempt_lift_sum += lift_progress * yaw
    757        env._yaw_fsm_attempt_support_sum += support_gate * yaw
    758        finishing = (diagonal != 0) & (~active | env.reset_buf)
    759        samples = env._yaw_fsm_attempt_yaw_samples
    760        lift_ok = (samples > 0) & (
    761            env._yaw_fsm_attempt_lift_sum / samples.clamp_min(1) >= lift_progress_threshold
    762        )
    763        support_ok = (samples > 0) & (
    764            env._yaw_fsm_attempt_support_sum / samples.clamp_min(1) >= support_threshold
    765        )
    766        reached = (samples > 0) & ~env.reset_terminated & ~gates["b_safe"]
    767        for suffix, sign in (("pos", 1), ("neg", -1)):
    768            ended = finishing & (diagonal == sign)
    769            for field, outcome in (
    770                ("transition_succeeded", reached),
    771                ("pose_succeeded", reached & lift_ok & support_ok),
    772                ("lift_failed", ~lift_ok), ("support_failed", ~support_ok),
    773            ):
    774                getattr(env, f"_yaw_fsm_{suffix}_{field}").add_((ended & outcome).long())
    775        diagonal.masked_fill_(finishing, 0)
    776        env._yaw_fsm_attempt_yaw_samples.masked_fill_(finishing, 0)
    777        env._yaw_fsm_attempt_lift_sum.masked_fill_(finishing, 0.0)
    778        env._yaw_fsm_attempt_support_sum.masked_fill_(finishing, 0.0)
    779
    780
    781    def _fsm_step_telemetry(env: ManagerBasedRLEnv, gates: dict[str, torch.Tensor]) -> None:
    
    ... (gap) ...
    
    808        env._yaw_fsm_transition_active.copy_(in_transition)
    809
    810
    811    _TRANSITION_TELEMETRY_FIELDS = (
    812        "support_ready", "lift_wheel_1_progress", "lift_wheel_2_progress",
    813        "clearance_ready", "attitude_ready", "pose_ready", "torso_contact",
    814    )
    815
    816
    817    def _fsm_transition_telemetry(
    818        env: ManagerBasedRLEnv,
    819        gates: dict[str, torch.Tensor],
    820        support_contacts: torch.Tensor,
    821        lift_progress: torch.Tensor,
    822        robot: RigidObject,
    823        command,
    824    ) -> None:
    825        """Sample the active diagonal's transition predicates once per reward step."""
    826        # CPU reward doubles can omit predicate wiring; production command configs
    827        # provide all six sensor/body selections together.
    828        torso_cfg = getattr(command.cfg, "torso_sensor_cfg", None)
    829        if torso_cfg is None:
    830            return
    831        contact_threshold = command.cfg.contact_threshold
    832        torso_contact = _yaw_wheel_contacts(env, torso_cfg, contact_threshold)[:, 0]
    833        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    834        attitude_ready = (roll.abs() < command.cfg.pose_angle_limit) & (
    835            pitch.abs() < command.cfg.pose_angle_limit
    836        )
    837        support_ready = support_contacts.all(dim=1)
    838        clearance_ready = (lift_progress >= command.cfg.clearance_fraction).all(dim=1)
    839        values = (
    840            support_ready, lift_progress[:, 0], lift_progress[:, 1],
    841            clearance_ready, attitude_ready,
    842            support_ready & clearance_ready & attitude_ready, torso_contact,
    843        )
    844        mask = gates["b_trans"]
    845        for field, value in zip(_TRANSITION_TELEMETRY_FIELDS, values):
    846            name = f"_yaw_fsm_transition_{field}_sum"
    847            if not hasattr(env, name):
    848                setattr(env, name, torch.zeros(env.num_envs, device=mask.device))
    849            getattr(env, name).add_(value.to(dtype=torch.float32) * mask)
    850
    851
    852    def _fsm_record_positive_budget(
    
    ... (gap) ...
    
    863        leaves the diagnostic unset and never changes the reward value.
    864        """
    865        try:
    866            weight = float(env.reward_manager.get_term_cfg(term_name).weight)
    867        except (AttributeError, KeyError, TypeError):
    868            return
    869        positive = torch.relu(value * weight)
    
    ... (gap) ...
    
    915        if not 0.0 < edge_command_fraction <= 1.0:
    916            raise ValueError("edge_command_fraction must be in (0, 1].")
    917
    918        gates = fsm_gates(env, fsm_command_name)
    919        _fsm_step_telemetry(env, gates)
    920        support_pos_contact = _yaw_wheel_contacts(env, support_sensor_cfg, contact_threshold)
    921        support_neg_contact = _yaw_wheel_contacts(env, support_sensor_cfg_mirror, contact_threshold)
    922        support_pos = support_pos_contact.to(dtype=torch.float32).prod(dim=1)
    923        support_neg = support_neg_contact.to(dtype=torch.float32).prod(dim=1)
    924        support_gate = torch.where(gates["diag_pos"], support_pos, support_neg)
    
    ... (gap) ...
    
    945        support_contacts = torch.where(
    946            gates["diag_pos"].unsqueeze(1), support_pos_contact, support_neg_contact
    947        )
    948        support_shape = _yaw_support_shape(support_contacts)
    949        # The opposite support diagonal is exactly the active swing diagonal.
    950        swing_contact = select_swing_wheel_contact(
    951            support_pos_contact,
    952            support_neg_contact,
    953            gates["support_diagonal"],
    954        )
    955
    956        lift_pos = _yaw_lift_progress(env, lifted_asset_cfg, wheel_radius, target_clearance)
    957        lift_neg = _yaw_lift_progress(env, lifted_asset_cfg_mirror, wheel_radius, target_clearance)
    958        selected_lift = torch.where(gates["diag_pos"].unsqueeze(1), lift_pos, lift_neg)
    959        lift_progress = selected_lift.mean(dim=1)
    960
    961        asset: RigidObject = env.scene[asset_cfg.name]
    962        _fsm_transition_telemetry(
    963            env, gates, support_contacts, selected_lift, asset,
    964            env.command_manager.get_term(fsm_command_name),
    965        )
    
    ... (gap) ...
    
    973        # Reuse the legacy yaw-curriculum accumulator names.  This keeps the
    974        # existing checkpoint/export path valid while changing only the sample
    975        # mask: FSM tracking statistics are normalized by YAW-state steps.
    976        _fsm_masked_accumulate(
    977            env,
    978            support_gate,
    979            yaw_mask,
    980            "_yaw_support_score_sum",
    981            "_yaw_support_score_samples",
    982        )
    983        _fsm_masked_accumulate(
    984            env,
    985            support_gate,
    986            yaw_mask,
    987            "_yaw_gate_open_sum",
    988            "_yaw_gate_open_samples",
    989        )
    990        _fsm_masked_accumulate_pair(
    991            env,
    992            torch.abs(yaw_command),
    993            yaw_error,
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/observations.py`** — wheel_contact(calls), wheel_clearance(calls), wheel_contact(function), wheel_clearance(function), yaw_fsm_predicates(function), yaw_fsm_unsafe(calls), +3 more
    
    ```python
    118        return phase_tensor
    119
    120
    121    def wheel_contact(
    122        env: ManagerBasedRLEnv,
    123        sensor_cfg: SceneEntityCfg,
    124        threshold: float = 1.0,
    125    ) -> torch.Tensor:
    126        """Return one binary contact flag per selected wheel body."""
    127
    128        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    129        force = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    130
    131        return (torch.linalg.vector_norm(force, dim=-1) > threshold).float()
    132
    133
    134    def wheel_clearance(
    135        env: ManagerBasedRLEnv,
    136        asset_cfg: SceneEntityCfg,
    137        wheel_radius: float,
    138    ) -> torch.Tensor:
    139        """Return wheel-bottom clearance relative to each environment origin."""
    140
    141        robot: Articulation = env.scene[asset_cfg.name]
    142        wheel_height = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    143        ground_height = env.scene.env_origins[:, 2].unsqueeze(-1)
    144
    145        return wheel_height - ground_height - wheel_radius
    146
    147
    148    def wheel_normal_force(
    
    ... (gap) ...
    
    193        if pose_angle_limit < 0.0 or unsafe_angle_limit < 0.0:
    194            raise ValueError("Pose angle limits must be non-negative.")
    195
    196        positive_support_contact = wheel_contact(
    197            env, support_sensor_cfg, threshold=contact_threshold
    198        ).bool()
    199        negative_support_contact = wheel_contact(
    200            env, support_sensor_cfg_mirror, threshold=contact_threshold
    201        ).bool()
    202        four_wheel_contact = wheel_contact(
    203            env, all_wheel_sensor_cfg, threshold=contact_threshold
    204        ).bool()
    205        torso_contact = wheel_contact(
    206            env, torso_sensor_cfg, threshold=contact_threshold
    207        ).bool()
    208
    209        lifted_clearance = wheel_clearance(
    210            env, lifted_asset_cfg, wheel_radius=wheel_radius
    211        )
    212        lifted_clearance_mirror = wheel_clearance(
    213            env, lifted_asset_cfg_mirror, wheel_radius=wheel_radius
    214        )
    215
    
    ... (gap) ...
    
    239        )
    240        four_stand_ready = four_wheel_contact.all(dim=-1) & pose_is_safe
    241
    242        unsafe = yaw_fsm_unsafe(
    243            env,
    244            robot_name=robot_name,
    245            torso_sensor_cfg=torso_sensor_cfg,
    
    ... (gap) ...
    
    251        return positive_pose_ready, negative_pose_ready, four_stand_ready, unsafe
    252
    253
    254    def yaw_fsm_unsafe(
    255        env: ManagerBasedEnv,
    256        robot_name: str,
    257        torso_sensor_cfg: SceneEntityCfg,
    258        minimum_base_height: float,
    259        unsafe_angle_limit: float,
    260        contact_threshold: float = 1.0,
    261    ) -> torch.Tensor:
    262        """Return the single current-state unsafe predicate used by FSM and done.
    263
    264        Keeping this separate from :func:`yaw_fsm_predicates` is deliberate: a
    265        termination term must inspect live sensor/pose data, not the command
    266        buffer from the preceding command update.  Both paths nevertheless use
    267        exactly the same thresholds and contact interpretation.
    268        """
    269        torso, height, tilt = yaw_fsm_unsafe_components(
    270            env, robot_name, torso_sensor_cfg, minimum_base_height,
    271            unsafe_angle_limit, contact_threshold,
    272        )
    273        return torso | height | tilt
    274
    275
    276    def yaw_fsm_unsafe_components(
    277        env: ManagerBasedEnv,
    278        robot_name: str,
    279        torso_sensor_cfg: SceneEntityCfg,
    280        minimum_base_height: float,
    281        unsafe_angle_limit: float,
    282        contact_threshold: float = 1.0,
    283    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    284        """Return separate torso-contact, low-height, and tilt failure masks."""
    285        if minimum_base_height < 0.0:
    286            raise ValueError("minimum_base_height must be non-negative.")
    287        if unsafe_angle_limit < 0.0:
    288            raise ValueError("unsafe_angle_limit must be non-negative.")
    289        if contact_threshold < 0.0:
    290            raise ValueError("contact_threshold must be non-negative.")
    291
    292        torso_contact = wheel_contact(
    293            env, torso_sensor_cfg, threshold=contact_threshold
    294        ).bool()
    295        if torso_contact.ndim != 2 or torso_contact.shape[-1] != 1:
    296            raise ValueError("Yaw FSM torso contact selection must contain exactly one body.")
    297        robot: Articulation = env.scene[robot_name]
    298        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    299        base_height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    300        return (
    301            torso_contact[:, 0],
    302            base_height < minimum_base_height,
    303            (torch.abs(roll) > unsafe_angle_limit) | (torch.abs(pitch) > unsafe_angle_limit),
    304        )
    305
    306
    307    def base_height(
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py`** — imports(imports), ManagerTermBase(extends), ManagerTermBase(imports), yaw_fsm_unsafe_components(imports), TorsoContactWithGrace(class), FSMUnsafeWithGrace(class), +7 more
    
    ```python
    5
    6    from __future__ import annotations
    7
    8    from collections.abc import Sequence
    9    from typing import TYPE_CHECKING
    10
    11    import torch
    12
    13    from isaaclab.managers import ManagerTermBase, SceneEntityCfg
    14    from isaaclab.sensors import ContactSensor
    15
    16    from .fsm import VQRFsmState
    17    from .observations import yaw_fsm_unsafe_components
    18
    19    if TYPE_CHECKING:
    20        from isaaclab.envs import ManagerBasedRLEnv
    21
    22
    23    class TorsoContactWithGrace(ManagerTermBase):
    24        """Terminate selected torso contact after a reset-relative grace period.
    25
    26        The elapsed-time buffer is owned by this term and reset through the
    
    ... (gap) ...
    
    59            return terminated
    60
    61
    62    class FSMUnsafeWithGrace(ManagerTermBase):
    63        """Phase-aware unsafe termination using the live FSM safety predicate.
    64
    65        Phases A/B terminate after the short reset grace.  Phase C intentionally
    
    ... (gap) ...
    
    69        old when the termination manager runs.
    70        """
    71
    72        def __init__(self, cfg, env: ManagerBasedRLEnv):
    73            super().__init__(cfg, env)
    74            self._elapsed_s = torch.zeros(env.num_envs, device=env.device)
    75
    76        def reset(self, env_ids: Sequence[int] | None = None) -> None:
    77            if env_ids is None:
    78                self._elapsed_s.zero_()
    79            else:
    80                self._elapsed_s[env_ids] = 0.0
    81
    82        def __call__(
    83            self,
    
    ... (gap) ...
    
    113            return grace_finished & unsafe
    114
    115
    116    class SwingContactTimeout(ManagerTermBase):
    117        """Terminate persistent swing-wheel contact during either YAW state.
    118
    119        Contact timers are independent for the two swing wheels.  They are reset
    
    ... (gap) ...
    
    174            return (self._contact_time >= timeout_s).any(dim=1)
    175
    176
    177    def fsm_transition_timeout(
    178        env: ManagerBasedRLEnv,
    179        command_name: str = "yaw_rate_cmd",
    180        timeout_s: float = 3.0,
    181    ) -> torch.Tensor:
    182        """Terminate after the cumulative TRANSITION budget for one maneuver.
    183
    184        ``transition_time`` is owned by the command term and is reset with the FSM, so
    185        this term remains independent of randomized episode lengths and command
    186        resampling.  YAW pauses the budget; a later TRANSITION resumes it.
    187        """
    188        if timeout_s < 0.0:
    189            raise ValueError("timeout_s must be non-negative.")
    190
    191        command_term = env.command_manager.get_term(command_name)
    192        missing = [
    193            name
    194            for name in ("fsm_state", "transition_time")
    195            if not hasattr(command_term, name)
    196        ]
    197        if missing:
    198            raise TypeError(
    199                f"Command '{command_name}' does not expose FSM buffers {missing}; "
    200                "use YawFSMCommand for the FSM task."
    201            )
    202
    203        device = getattr(env, "device", None)
    204        state = torch.as_tensor(getattr(command_term, "fsm_state"), device=device)
    205        transition_time = torch.as_tensor(getattr(command_term, "transition_time"), device=device)
    206        expected_shape = (env.num_envs,)
    207        if state.shape != expected_shape or transition_time.shape != expected_shape:
    208            raise ValueError(
    209                "FSM state and transition_time must each have shape "
    210                f"{expected_shape}; got {tuple(state.shape)} and {tuple(transition_time.shape)}"
    211            )
    212
    213        in_transition = (state == int(VQRFsmState.TRANSITION_POS)) | (
    214            state == int(VQRFsmState.TRANSITION_NEG)
    215        )
    216        return in_transition & (transition_time >= timeout_s)
    217
    218
    219    def fsm_return_timeout(
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py`** — configclass(decorates), VQRWheelYawCurriculumCfg(class), VQRWheelFSMCurriculumCfg(class), VQRWheelYawCurriculumCfg(extends)
    
    ```python
    624        )
    625
    626
    627    @configclass
    628    class VQRWheelYawCurriculumCfg:
    629        """Lift first, then jointly progress yaw command and online DR."""
    630
    631        task_levels = CurrTerm(
    632            func=mdp.yaw_task_levels,
    633            params={
    634                "command_name": "yaw_rate_cmd",
    635                "clearance_levels": LIFT_CLEARANCE_LEVELS,
    636                "yaw_rate_levels": YAW_RATE_LEVELS,
    637                "dr_scale_levels": ONLINE_DR_SCALE_LEVELS,
    638                "tracking_ratio_thresholds": YAW_TRACKING_RATIO_THRESHOLDS,
    639                "edge_tracking_ratio_thresholds": YAW_EDGE_TRACKING_RATIO_THRESHOLDS,
    640                "lift_reward_name": "lift_clearance",
    641                "balance_reward_name": "balance",
    642                "yaw_reward_name": "gated_yaw_tracking",
    643                "torso_contact_termination_name": "torso_contact",
    644                "minimum_base_height": MIN_BASE_HEIGHT,
    645                "support_threshold": 0.85,
    646                "lift_progress_threshold": 0.80,
    647                "balance_threshold": 0.75,
    648                "yaw_threshold": 0.65,
    649                "min_evaluated_episodes": 2048,
    650                "required_success_rate": 0.85,
    651                "required_consecutive_windows": 3,
    652                "min_clearance_stage_steps": 1000,
    653                "min_yaw_stage_steps": 6000,
    654            },
    655        )
    656
    657
    658    @configclass
    659    class VQRWheelFSMCurriculumCfg(VQRWheelYawCurriculumCfg):
    660        """Three-phase FSM curriculum with independent POS/NEG certification."""
    661
    662        task_levels = CurrTerm(
    ```
    
    **`source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/mdp/to_transition.py`** — reference(method), pose_reference(calls), scheduled_contact_reward(function), unload_fraction(calls), transition_yaw_reward(function), smoothstep(calls)
    
    ```python
    66            return self._command
    67
    68        @property
    69        def reference(self):
    70            return pose_reference(self.standing, self.target, self.command[:, 1])
    71
    72        def contacts(self):
    73            return (self.sensor.data.net_forces_w[:, self.contact_ids].norm(dim=-1) > self.cfg.contact_threshold).float()
    
    ... (gap) ...
    
    199        return term.weights[term.episode_level] * torch.exp(-k_pose * error)
    200
    201
    202    def scheduled_contact_reward(env):
    203        term = env.command_manager.get_term("pivot")
    204
    205        contact = term.contacts()
    206        phase = term.command[:, 1]
    207        unload = unload_fraction(phase)
    208
    209        # FL, FR, HL, HR
    210        four_wheel = contact.mean(dim=-1)
    211
    212        # Desired final support: FL + HR
    213        diagonal = 0.5 * (
    214            contact[:, 0]
    215            + contact[:, 3]
    216            - contact[:, 1]
    217            - contact[:, 2]
    218        )
    219
    220        return (1.0 - unload) * four_wheel + unload * diagonal
    221
    222    def transition_yaw_reward(env, k_yaw=11.11):
    223        term = env.command_manager.get_term("pivot")
    224        error = term.robot.data.root_ang_vel_b[:, 2] - term.command[:, 0]
    225        return smoothstep((term.command[:, 1] - 0.8) / 0.2) * torch.exp(-k_yaw * error.square())
    226
    227
    228    def reset_transition_standing(
    ```
    
    
    > Some file sections were trimmed for size. For a specific symbol you still need, run another `codegraph_explore` (or `codegraph_node`) with its exact name — line-numbered source, cheaper and more complete than Read.
    ✓ • 247ms

## Activity

    $ git status --short; rg -n 'NoiseModelWithAdditiveBias|bias_noise|bias.*reset|reset.*bias' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity tests | head -130
     M scripts/reinforcement_learning/rsl_rl/train.py
     M source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py
     M source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py
     M source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py
     M tests/test_yaw_curriculum.py
     M tests/test_yaw_fsm.py
     M tests/test_yaw_fsm_systemic.py
     M tests/test_yaw_fsm_training_contract.py
     M tests/test_yaw_transition_safety.py
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_cfg.py:13:from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_cfg.py:377:        self.observations.policy.joint_pos.noise = NoiseModelWithAdditiveBiasCfg(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_cfg.py:379:            bias_noise_cfg=Unoise(n_min=-0.1, n_max=0.1),
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/rough_env_cfg_imu.py:13:from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/rough_env_cfg_imu.py:152:        self.observations.policy.joint_pos.noise = NoiseModelWithAdditiveBiasCfg(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/rough_env_cfg_imu.py:154:            bias_noise_cfg=Unoise(n_min=-0.1, n_max=0.1),
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/rough_env_cfg.py:12:from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/rough_env_cfg.py:126:        self.observations.policy.joint_pos.noise = NoiseModelWithAdditiveBiasCfg(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/rough_env_cfg.py:128:            bias_noise_cfg=Unoise(n_min=-0.1, n_max=0.1),
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:14:from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:849:        self.observations.policy.joint_pos.noise = NoiseModelWithAdditiveBiasCfg(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:851:            bias_noise_cfg=Unoise(n_min=-0.1, n_max=0.1, operation="abs"),
    ✓ • 0ms

## Activity

    $ rg --files scripts tests | rg 'rollout|smoke|yaw|compare_runs|play.py|train.py' | head -100
    tests/test_yaw_curriculum.py
    tests/test_yaw_transition_safety.py
    tests/test_yaw_fsm_systemic.py
    tests/test_yaw_fsm.py
    tests/test_yaw_fsm_training_contract.py
    tests/test_yaw_fsm_observations.py
    scripts/tools/compare_runs.py
    scripts/tools/analyze_yaw_contact_trace.py
    scripts/tools/run_yaw_contact_trace.sh
    scripts/reinforcement_learning/rsl_rl/play.py
    scripts/reinforcement_learning/rsl_rl/train.py
    ✓ • 0ms

## Activity

    $ rg -n 'Flat-VQR-Wheel-Yaw-FSM|vqr_wheel_yaw_flat_fsm|unsafe.*cause|torso_contact|base_height_failure|tilt_failure' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel scripts/reinforcement_learning/rsl_rl | head -150
    scripts/reinforcement_learning/rsl_rl/train.py:376:    print(f"[INFO] Flat-VQR-Wheel-Yaw-FSM env config source: {inspect.getfile(type(env_cfg))}")
    scripts/reinforcement_learning/rsl_rl/train.py:377:    print(f"[INFO] Flat-VQR-Wheel-Yaw-FSM reward terms ({len(term_names)}): {term_names}")
    scripts/reinforcement_learning/rsl_rl/train.py:378:    print(f"[INFO] Flat-VQR-Wheel-Yaw-FSM fsm_gated_tracking weight: {tracking_weight}")
    scripts/reinforcement_learning/rsl_rl/train.py:381:            "Unexpected Flat-VQR-Wheel-Yaw-FSM reward config: expected 26 terms, "
    scripts/reinforcement_learning/rsl_rl/train.py:387:            "Unexpected Flat-VQR-Wheel-Yaw-FSM command contract: expected YawFSMCommand "
    scripts/reinforcement_learning/rsl_rl/train.py:474:    yaw_fsm_task_env = env.unwrapped if task_name == "Flat-VQR-Wheel-Yaw-FSM" else None
    scripts/reinforcement_learning/rsl_rl/train.py:477:    elif task_name == "Flat-VQR-Wheel-Yaw-FSM":
    scripts/reinforcement_learning/rsl_rl/train.py:532:                "[INFO] Restored Flat-VQR-Wheel-Yaw-FSM curriculum: "
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_cfg.py:281:            "torso_contact_termination_name": "torso_contact",
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_cfg.py:300:    torso_contact = DoneTerm(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:285:    """The reward contract for ``Flat-VQR-Wheel-Yaw-FSM``.
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:643:            "torso_contact_termination_name": "torso_contact",
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:677:            "torso_contact_termination_name": "torso_contact",
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:703:    torso_contact = DoneTerm(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:717:    torso_contact = DoneTerm(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:731:    base_height_failure = DoneTerm(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:745:    tilt_failure = DoneTerm(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/__init__.py:59:    id="Flat-VQR-Wheel-Yaw-FSM",
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/agents/rsl_rl_ppo_cfg.py:99:        self.experiment_name = "vqr_wheel_yaw_flat_fsm"
    ✓ • 0ms

## Activity

    $ sed -n '835,860p' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py; sed -n '1,115p' scripts/tools/run_yaw_contact_trace.sh; sed -n '1,95p' scripts/tools/analyze_yaw_contact_trace.py
            self.observations.policy.base_ang_vel.scale = 0.25
            self.observations.policy.joint_pos.scale = 1.0
            self.observations.policy.joint_vel.scale = 0.05
            self.observations.policy.height_scan = None
            self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
            self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
    
            # increase observation noise to reduce reliance on precise instantaneous feedback
            # (helps close the sim-to-real gap and reduce high-frequency action jitter)
            self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.4, n_max=0.4)
            self.observations.policy.projected_gravity.noise = Unoise(n_min=-0.1, n_max=0.1)
            self.observations.policy.joint_vel.noise = Unoise(n_min=-3.0, n_max=3.0)
            # joint_pos: per-step noise + a per-episode, per-joint constant bias (~encoder/mechanical
            # calibration offset observed on real hardware, up to ~0.1 rad)
            self.observations.policy.joint_pos.noise = NoiseModelWithAdditiveBiasCfg(
                noise_cfg=Unoise(n_min=-0.02, n_max=0.02),
                bias_noise_cfg=Unoise(n_min=-0.1, n_max=0.1, operation="abs"),
                sample_bias_per_component=True,
            )
    
            # ------------------------------Actions------------------------------
            # Wider per-joint residual ranges improve pose discovery.
            self.actions.joint_pos.scale = {
                ".*_HipX_joint": 0.30,
                ".*_HipY_joint": 0.60,
                ".*_Knee_joint": 0.50,
    #!/usr/bin/env bash
    set -euo pipefail
    
    repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    isaaclab_launcher="${1:-}"
    checkpoint="${2:-$repo_dir/logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_11-23-35/model_500.pt}"
    trace_csv="${3:-$repo_dir/logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_11-23-35/contact_trace_model_500.csv}"
    
    
    
    python3 "$repo_dir/scripts/reinforcement_learning/rsl_rl/play.py" \
        --task Flat-VQR-Wheel-Yaw-FSM \
        --checkpoint "$checkpoint" \
        --num_envs 1 \
        --max_steps 2000 \
        --headless \
        --yaw_contact_trace "$trace_csv" \
        --yaw_contact_limit 0.25
    
    python3 "$repo_dir/scripts/tools/analyze_yaw_contact_trace.py" "$trace_csv"
    """Summarize yaw FSM wheel-contact traces written by rsl_rl/play.py."""
    
    from __future__ import annotations
    
    import argparse
    import csv
    from collections import Counter
    
    
    SUPPORT_WHEELS = {2: ("FL", "HR"), 4: ("FR", "HL")}
    
    
    def analyze(path: str, threshold: float, airborne_clearance: float, near_ground: float) -> None:
        counts: dict[str, Counter[str]] = {"POS": Counter(), "NEG": Counter()}
        direction_envs: dict[str, set[int]] = {"POS": set(), "NEG": set()}
        state_counts: Counter[int] = Counter()
        command_counts: Counter[str] = Counter()
        with open(path, newline="") as stream:
            for row in csv.DictReader(stream):
                state = int(row["fsm_state"])
                state_counts[state] += 1
                command = float(row["yaw_command"])
                if command < -0.10:
                    command_counts["negative_enter"] += 1
                    if state == 5:
                        command_counts["negative_enter_in_return"] += 1
                elif command > 0.10:
                    command_counts["positive_enter"] += 1
                if state not in SUPPORT_WHEELS:
                    continue
                direction = "POS" if state == 2 else "NEG"
                direction_envs[direction].add(int(row["env_id"]))
                result = counts[direction]
                result["yaw_steps"] += 1
                wheels = SUPPORT_WHEELS[state]
                norm_flags = []
                fz_flags = []
                for wheel in wheels:
                    clearance = float(row[f"{wheel}_clearance_m"])
                    fz = float(row[f"{wheel}_fz_n"])
                    norm = float(row[f"{wheel}_force_norm_n"])
                    norm_flag = norm > threshold
                    fz_flag = fz > threshold
                    norm_flags.append(norm_flag)
                    fz_flags.append(fz_flag)
                    result["wheel_samples"] += 1
                    result["norm_contact"] += norm_flag
                    result["fz_contact"] += fz_flag
                    result["norm_without_fz"] += norm_flag and not fz_flag
                    result["airborne_zero_force"] += clearance > airborne_clearance and norm < 0.2 * threshold
                    result["near_ground_threshold_band"] += (
                        abs(clearance) <= near_ground and 0.5 * threshold <= fz <= 1.5 * threshold
                    )
                    history_norms = [
                        float(value)
                        for key, value in row.items()
                        if key.startswith(f"{wheel}_history_") and key.endswith("_norm_n") and value != ""
                    ]
                    result["history_current_miss"] += (
                        not norm_flag and any(value > threshold for value in history_norms)
                    )
                norm_pair = all(norm_flags)
                fz_pair = all(fz_flags)
                result["norm_pair"] += norm_pair
                result["fz_pair"] += fz_pair
                result["norm_pair_without_fz_pair"] += norm_pair and not fz_pair
                metric = row.get("metric_support_gate", "")
                if metric != "":
                    result["metric_samples"] += 1
                    result["metric_mismatch"] += (float(metric) > 0.5) != norm_pair
    
        print(f"FSM state samples: {dict(sorted(state_counts.items()))}")
        print(f"Commands beyond entry threshold: + {command_counts['positive_enter']}, - {command_counts['negative_enter']} ({command_counts['negative_enter_in_return']} while RETURN_TO_4)")
        for direction, result in counts.items():
            steps = result["yaw_steps"]
            wheels = result["wheel_samples"]
            print(f"{direction}: {steps} YAW steps from {len(direction_envs[direction])} environments")
            if not steps:
                continue
            print(f"  both support wheels: ||F||>{threshold:g} N {result['norm_pair'] / steps:.1%}; Fz>{threshold:g} N {result['fz_pair'] / steps:.1%}")
            print(f"  norm pair passes but Fz pair fails: {result['norm_pair_without_fz_pair']} steps")
            print(f"  individual wheel samples: norm without Fz {result['norm_without_fz']} / {wheels}")
            print(f"  clearance>{airborne_clearance:g} m and ||F||<{0.2 * threshold:g} N: {result['airborne_zero_force']} / {wheels}")
            print(f"  |clearance|<={near_ground:g} m and Fz near threshold: {result['near_ground_threshold_band']} / {wheels}")
            print(f"  current ||F|| fails while a history sample passes: {result['history_current_miss']} / {wheels}")
            print(f"  recorded metric differs from recalculated ||F|| pair: {result['metric_mismatch']} / {result['metric_samples']}")
    
    
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("trace_csv")
        parser.add_argument("--threshold", type=float, default=1.0, help="Contact-force threshold in N.")
        parser.add_argument("--airborne-clearance", type=float, default=0.02, help="Airborne wheel clearance in m.")
        parser.add_argument("--near-ground", type=float, default=0.01, help="Near-ground clearance band in m.")
        args = parser.parse_args()
    ✓ • 0ms

## Activity

    $ /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python -c 'import inspect; from isaaclab.utils.noise import NoiseModelWithAdditiveBias; print(inspect.getfile(NoiseModelWithAdditiveBias))'
    /home/robotics/datld5/IsaacLab/source/isaaclab/isaaclab/utils/noise/noise_model.py
    ✓ • 2.87s

## Activity

    $ rg -n 'class NoiseModelWithAdditiveBias|def reset|bias_noise_cfg|sample_bias_per_component' /home/robotics/datld5/IsaacLab/source/isaaclab/isaaclab/utils/noise/noise_model.py; sed -n '85,180p' /home/robotics/datld5/IsaacLab/source/isaaclab/isaaclab/utils/noise/noise_model.py
    122:    def reset(self, env_ids: Sequence[int] | None = None):
    146:class NoiseModelWithAdditiveBias(NoiseModel):
    156:        self._bias_noise_cfg = noise_model_cfg.bias_noise_cfg
    159:        self._sample_bias_per_component = noise_model_cfg.sample_bias_per_component
    161:    def reset(self, env_ids: Sequence[int] | None = None):
    174:        self._bias[env_ids] = self._bias_noise_cfg.func(self._bias[env_ids], self._bias_noise_cfg)
    185:        # if sample_bias_per_component, on first apply, expand bias to match last dim of data
    186:        if self._sample_bias_per_component and self._num_components is None:
        # fix tensor device for mean on first call and update config parameters
        if isinstance(cfg.mean, torch.Tensor):
            cfg.mean = cfg.mean.to(data.device)
        # fix tensor device for std on first call and update config parameters
        if isinstance(cfg.std, torch.Tensor):
            cfg.std = cfg.std.to(data.device)
    
        if cfg.operation == "add":
            return data + cfg.mean + cfg.std * torch.randn_like(data)
        elif cfg.operation == "scale":
            return data * (cfg.mean + cfg.std * torch.randn_like(data))
        elif cfg.operation == "abs":
            return cfg.mean + cfg.std * torch.randn_like(data)
        else:
            raise ValueError(f"Unknown operation in noise: {cfg.operation}")
    
    
    ##
    # Noise models as classes
    ##
    
    
    class NoiseModel:
        """Base class for noise models."""
    
        def __init__(self, noise_model_cfg: noise_cfg.NoiseModelCfg, num_envs: int, device: str):
            """Initialize the noise model.
    
            Args:
                noise_model_cfg: The noise configuration to use.
                num_envs: The number of environments.
                device: The device to use for the noise model.
            """
            self._noise_model_cfg = noise_model_cfg
            self._num_envs = num_envs
            self._device = device
    
        def reset(self, env_ids: Sequence[int] | None = None):
            """Reset the noise model.
    
            This method can be implemented by derived classes to reset the noise model.
            This is useful when implementing temporal noise models such as random walk.
    
            Args:
                env_ids: The environment ids to reset the noise model for. Defaults to None,
                    in which case all environments are considered.
            """
            pass
    
        def __call__(self, data: torch.Tensor) -> torch.Tensor:
            """Apply the noise to the data.
    
            Args:
                data: The data to apply the noise to. Shape is (num_envs, ...).
    
            Returns:
                The data with the noise applied. Shape is the same as the input data.
            """
            return self._noise_model_cfg.noise_cfg.func(data, self._noise_model_cfg.noise_cfg)
    
    
    class NoiseModelWithAdditiveBias(NoiseModel):
        """Noise model with an additive bias.
    
        The bias term is sampled from a the specified distribution on reset.
        """
    
        def __init__(self, noise_model_cfg: noise_cfg.NoiseModelWithAdditiveBiasCfg, num_envs: int, device: str):
            # initialize parent class
            super().__init__(noise_model_cfg, num_envs, device)
            # store the bias noise configuration
            self._bias_noise_cfg = noise_model_cfg.bias_noise_cfg
            self._bias = torch.zeros((num_envs, 1), device=self._device)
            self._num_components: int | None = None
            self._sample_bias_per_component = noise_model_cfg.sample_bias_per_component
    
        def reset(self, env_ids: Sequence[int] | None = None):
            """Reset the noise model.
    
            This method resets the bias term for the specified environments.
    
            Args:
                env_ids: The environment ids to reset the noise model for. Defaults to None,
                    in which case all environments are considered.
            """
            # resolve the environment ids
            if env_ids is None:
                env_ids = slice(None)
            # reset the bias term
            self._bias[env_ids] = self._bias_noise_cfg.func(self._bias[env_ids], self._bias_noise_cfg)
    
        def __call__(self, data: torch.Tensor) -> torch.Tensor:
            """Apply bias noise to the data.
    
            Args:
                data: The data to apply the noise to. Shape is (num_envs, ...).
    ✓ • 0ms

## Activity

    $ sed -n '1,90p' /home/robotics/datld5/IsaacLab/source/isaaclab/isaaclab/utils/noise/noise_model.py; sed -n '175,205p' /home/robotics/datld5/IsaacLab/source/isaaclab/isaaclab/utils/noise/noise_model.py; rg -n 'NoiseModelWithAdditiveBiasCfg' /home/robotics/datld5/IsaacLab/source/isaaclab/isaaclab/utils/noise/noise_cfg.py
    # Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
    # All rights reserved.
    #
    # SPDX-License-Identifier: BSD-3-Clause
    
    from __future__ import annotations
    
    from collections.abc import Sequence
    from typing import TYPE_CHECKING
    
    import torch
    
    if TYPE_CHECKING:
        from . import noise_cfg
    
    ##
    # Noise as functions.
    ##
    
    
    def constant_noise(data: torch.Tensor, cfg: noise_cfg.ConstantNoiseCfg) -> torch.Tensor:
        """Applies a constant noise bias to a given data set.
    
        Args:
            data: The unmodified data set to apply noise to.
            cfg: The configuration parameters for constant noise.
    
        Returns:
            The data modified by the noise parameters provided.
        """
    
        # fix tensor device for bias on first call and update config parameters
        if isinstance(cfg.bias, torch.Tensor):
            cfg.bias = cfg.bias.to(device=data.device)
    
        if cfg.operation == "add":
            return data + cfg.bias
        elif cfg.operation == "scale":
            return data * cfg.bias
        elif cfg.operation == "abs":
            return torch.zeros_like(data) + cfg.bias
        else:
            raise ValueError(f"Unknown operation in noise: {cfg.operation}")
    
    
    def uniform_noise(data: torch.Tensor, cfg: noise_cfg.UniformNoiseCfg) -> torch.Tensor:
        """Applies a uniform noise to a given data set.
    
        Args:
            data: The unmodified data set to apply noise to.
            cfg: The configuration parameters for uniform noise.
    
        Returns:
            The data modified by the noise parameters provided.
        """
    
        # fix tensor device for n_max on first call and update config parameters
        if isinstance(cfg.n_max, torch.Tensor):
            cfg.n_max = cfg.n_max.to(data.device)
        # fix tensor device for n_min on first call and update config parameters
        if isinstance(cfg.n_min, torch.Tensor):
            cfg.n_min = cfg.n_min.to(data.device)
    
        if cfg.operation == "add":
            return data + torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min
        elif cfg.operation == "scale":
            return data * (torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min)
        elif cfg.operation == "abs":
            return torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min
        else:
            raise ValueError(f"Unknown operation in noise: {cfg.operation}")
    
    
    def gaussian_noise(data: torch.Tensor, cfg: noise_cfg.GaussianNoiseCfg) -> torch.Tensor:
        """Applies a gaussian noise to a given data set.
    
        Args:
            data: The unmodified data set to apply noise to.
            cfg: The configuration parameters for gaussian noise.
    
        Returns:
            The data modified by the noise parameters provided.
        """
    
        # fix tensor device for mean on first call and update config parameters
        if isinstance(cfg.mean, torch.Tensor):
            cfg.mean = cfg.mean.to(data.device)
        # fix tensor device for std on first call and update config parameters
        if isinstance(cfg.std, torch.Tensor):
            cfg.std = cfg.std.to(data.device)
    
        def __call__(self, data: torch.Tensor) -> torch.Tensor:
            """Apply bias noise to the data.
    
            Args:
                data: The data to apply the noise to. Shape is (num_envs, ...).
    
            Returns:
                The data with the noise applied. Shape is the same as the input data.
            """
            # if sample_bias_per_component, on first apply, expand bias to match last dim of data
            if self._sample_bias_per_component and self._num_components is None:
                *_, self._num_components = data.shape
                # expand bias from (num_envs,1) to (num_envs, num_components)
                self._bias = self._bias.repeat(1, self._num_components)
                # now re-sample that expanded bias in-place
                self.reset()
            return super().__call__(data) + self._bias
    97:class NoiseModelWithAdditiveBiasCfg(NoiseModelCfg):
    ✓ • 0ms

## Activity

    $ /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python - <<'PY'
        import torch
        from isaaclab.utils.noise import AdditiveUniformNoiseCfg, NoiseModelWithAdditiveBiasCfg
        from isaaclab.utils.noise.noise_model import NoiseModelWithAdditiveBias
        cfg = NoiseModelWithAdditiveBiasCfg(
            noise_cfg=AdditiveUniformNoiseCfg(n_min=-.02, n_max=.02),
            bias_noise_cfg=AdditiveUniformNoiseCfg(n_min=-.1, n_max=.1, operation='abs'),
            sample_bias_per_component=True,
        )
        torch.manual_seed(17)
        model = NoiseModelWithAdditiveBias(cfg, num_envs=256, device='cpu')
        model(torch.zeros(256, 16))
        old = model._bias.clone()
        model.reset(torch.arange(128))
        assert torch.equal(model._bias[128:], old[128:])
        assert not torch.equal(model._bias[:128], old[:128])
        minimum, maximum = 1., -1.
        means = []
        for _ in range(2000):
            model.reset()
            bias = model._bias
            minimum = min(minimum, bias.min().item())
            maximum = max(maximum, bias.max().item())
            means.append(bias.mean().item())
        assert minimum >= -.1 and maximum <= .1
        print(f'bias shape={tuple(model._bias.shape)} resets=2000 min={minimum:.6f} max={maximum:.6f} mean_of_reset_means={sum(means)/len(means):.6f} final_std={model._bias.std().item():.6f}')
        PY
    Traceback (most recent call last):
      File "<stdin>", line 24, in <module>
    AssertionError
    ✗ (1) • 2.48s

## Activity

    $ /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python - <<'PY'
        import torch
        from isaaclab.utils.noise import AdditiveUniformNoiseCfg, NoiseModelWithAdditiveBiasCfg
        from isaaclab.utils.noise.noise_model import NoiseModelWithAdditiveBias
        cfg=NoiseModelWithAdditiveBiasCfg(noise_cfg=AdditiveUniformNoiseCfg(n_min=-.02,n_max=.02),bias_noise_cfg=AdditiveUniformNoiseCfg(n_min=-.1,n_max=.1,operation='abs'),sample_bias_per_component=True)
        torch.manual_seed(17)
        model=NoiseModelWithAdditiveBias(cfg,256,'cpu')
        model(torch.zeros(256,16))
        old=model._bias.clone(); model.reset(torch.arange(128))
        print('subset unchanged',torch.equal(model._bias[128:],old[128:]),'subset changed',not torch.equal(model._bias[:128],old[:128]))
        minimum,maximum=1.,-1.; means=[]
        for _ in range(2000):
         model.reset(); bias=model._bias; minimum=min(minimum,bias.min().item()); maximum=max(maximum,bias.max().item()); means.append(bias.mean().item())
        print('shape',tuple(model._bias.shape),'min',minimum,'max',maximum,'mean',sum(means)/len(means),'std',model._bias.std().item())
        PY
    subset unchanged True subset changed True
    shape (256, 16) min -0.10000000149011612 max 0.09999998658895493 mean 1.945366415566241e-05 std 0.0570974126458168
    ✓ • 2.40s

## Assistant

The encoder bias check passes in the installed Isaac Lab noise model: after 2,000 resets across 256 environments and 16 joints, the sampled bias stayed within ±0.1 rad and did not carry into the next reset. I’m now checking the return ordering against the actual reward weights and FSM timing; the simulator result will be reported separately from controlled CPU counterfactuals.

## Activity

    $ ls -la logs; ls -la outputs | head -35; rg --files /home/robotics/tuanpm48/vqr/rl_training -g '*.pt' -g '*.pth' | head -40
    total 12
    drwxr-xr-x  3 robotics robotics 4096 Sep 14 15:31 .
    drwxr-xr-x 18 robotics robotics 4096 Sep 28 10:22 ..
    drwxr-xr-x  9 robotics robotics 4096 Sep 23 16:52 rsl_rl
    total 84
    drwxr-xr-x 21 robotics robotics 4096 Sep 25 09:42 .
    drwxr-xr-x 18 robotics robotics 4096 Sep 28 10:22 ..
    drwxrwxr-x  3 robotics robotics 4096 Sep 14 15:31 2026-09-04
    drwxrwxr-x  3 robotics robotics 4096 Sep 14 15:31 2026-09-06
    drwxrwxr-x  4 robotics robotics 4096 Sep 14 15:31 2026-09-07
    drwxrwxr-x  3 robotics robotics 4096 Sep 14 15:31 2026-09-08
    drwxrwxr-x  3 robotics robotics 4096 Sep 14 15:31 2026-09-10
    drwxrwxr-x  5 robotics robotics 4096 Sep 14 15:31 2026-09-11
    drwxrwxr-x  6 robotics robotics 4096 Sep 14 15:31 2026-09-13
    drwxrwxr-x  3 robotics robotics 4096 Sep 14 15:31 2026-09-14
    drwxrwxr-x 17 robotics robotics 4096 Sep 15 16:11 2026-09-15
    drwxrwxr-x 23 robotics robotics 4096 Sep 16 16:10 2026-09-16
    drwxrwxr-x 23 robotics robotics 4096 Sep 17 17:37 2026-09-17
    drwxrwxr-x  4 robotics robotics 4096 Sep 18 16:00 2026-09-18
    drwxrwxr-x  9 robotics robotics 4096 Sep 19 18:49 2026-09-19
    drwxrwxr-x 18 robotics robotics 4096 Sep 20 23:49 2026-09-20
    drwxrwxr-x 21 robotics robotics 4096 Sep 21 15:32 2026-09-21
    drwxrwxr-x 14 robotics robotics 4096 Sep 22 10:05 2026-09-22
    drwxrwxr-x 14 robotics robotics 4096 Sep 23 18:09 2026-09-23
    drwxrwxr-x 12 robotics robotics 4096 Sep 24 11:23 2026-09-24
    drwxrwxr-x 10 robotics robotics 4096 Sep 25 16:47 2026-09-25
    ✓ • 0ms

## Activity

    $ rg -n 'max_steps|yaw_contact_trace|checkpoint|num_envs|--task|headless|args_cli' scripts/reinforcement_learning/rsl_rl/play.py | head -100; rg -n 'max_iterations|num_envs|--task|headless|args_cli' scripts/reinforcement_learning/rsl_rl/train.py | head -100
    12:"""Script to play a checkpoint if an RL agent from RSL-RL."""
    33:parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    34:parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    41:parser.add_argument("--max_steps", type=int, default=None, help="Stop playback after this many environment steps.")
    43:    "--yaw_contact_trace",
    52:    help="Environment index to record with --yaw_contact_trace; use -1 for all environments.",
    69:    help="Sample actions from the checkpoint policy during contact tracing, as in PPO training.",
    88:args_cli, hydra_args = parser.parse_known_args()
    90:if args_cli.video:
    91:    args_cli.enable_cameras = True
    97:app_launcher = AppLauncher(args_cli)
    150:from isaaclab_tasks.utils import get_checkpoint_path
    156:@hydra_task_config(args_cli.task, args_cli.agent)
    159:    task_name = args_cli.task.split(":")[-1]
    161:    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    162:    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else (1 if args_cli.yaw_contact_trace else 50)
    163:    if args_cli.reward_diagnostics_interval <= 0:
    165:    if args_cli.max_steps is not None and args_cli.max_steps <= 0:
    166:        raise ValueError("--max_steps must be positive.")
    167:    if args_cli.yaw_contact_trace and not -1 <= args_cli.yaw_contact_env_id < env_cfg.scene.num_envs:
    169:    if args_cli.yaw_contact_trace and args_cli.yaw_contact_limit <= 0.0:
    171:    if args_cli.yaw_contact_trace and args_cli.yaw_contact_ready_dwell < 0.0:
    173:    if args_cli.yaw_contact_stochastic and not args_cli.yaw_contact_trace:
    174:        raise ValueError("--yaw_contact_stochastic requires --yaw_contact_trace.")
    182:    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    194:    if not args_cli.yaw_contact_trace:
    196:    if env_cfg.events is not None and not args_cli.yaw_contact_trace:
    205:    if args_cli.yaw_contact_trace:
    207:            -args_cli.yaw_contact_limit,
    208:            args_cli.yaw_contact_limit,
    210:        # The checkpoint used for this audit predates ready-pose dwell.  Keep
    213:            env_cfg.commands.yaw_rate_cmd.yaw_pose_ready_dwell = args_cli.yaw_contact_ready_dwell
    235:    if args_cli.keyboard:
    236:        env_cfg.scene.num_envs = 1
    253:    if args_cli.checkpoint:
    254:        resume_path = retrieve_file_path(args_cli.checkpoint)
    256:        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    261:    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    268:    if args_cli.video:
    272:            "video_length": args_cli.video_length,
    282:    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    291:    if args_cli.yaw_contact_stochastic:
    336:    if args_cli.yaw_contact_trace:
    339:            range(raw.num_envs)
    340:            if args_cli.yaw_contact_env_id == -1
    341:            else (args_cli.yaw_contact_env_id,)
    344:            raise ValueError("--yaw_contact_trace requires a yaw_rate_cmd task.")
    357:        trace_path = os.path.abspath(args_cli.yaw_contact_trace)
    371:            f"ready_dwell={args_cli.yaw_contact_ready_dwell} s, "
    372:            f"actions={'sampled' if args_cli.yaw_contact_stochastic else 'deterministic'}"
    380:    if args_cli.reward_diagnostics:
    447:        if args_cli.reward_diagnostics:
    457:            if reward_diagnostic_samples == args_cli.reward_diagnostics_interval:
    490:        if args_cli.video:
    492:            if timestep == args_cli.video_length:
    494:        if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
    497:        if args_cli.keyboard:
    502:        if args_cli.real_time and sleep_time > 0:
    31:parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    32:parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    37:parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
    51:args_cli, hydra_args = parser.parse_known_args()
    54:if args_cli.video:
    55:    args_cli.enable_cameras = True
    61:app_launcher = AppLauncher(args_cli)
    191:    all_env_ids = torch.arange(task_env.num_envs, device=task_env.device)
    296:    all_env_ids = torch.arange(task_env.num_envs, device=task_env.device)
    430:@hydra_task_config(args_cli.task, args_cli.agent)
    433:    task_name = args_cli.task.split(":")[-1]
    435:    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    436:    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    437:    agent_cfg.max_iterations = (
    438:        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    447:    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    450:    if args_cli.distributed:
    472:    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    489:    if args_cli.video:
    492:            "step_trigger": lambda step: step % args_cli.video_interval == 0,
    493:            "video_length": args_cli.video_length,
    544:    critic_warmup_iterations = args_cli.critic_warmup_iterations
    554:        final_update_label = start_iteration + agent_cfg.max_iterations - 1
    556:            "[INFO] --max_iterations is relative on resume: "
    557:            f"{agent_cfg.max_iterations} updates from checkpoint iteration {start_iteration}; "
    567:    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    ✓ • 0ms

## Activity

    $ nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
    NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.
    
    ✗ (9) • 0ms

## Activity

    $ find logs/rsl_rl -maxdepth 4 -type f \( -name 'model_*.pt' -o -name '*.csv' -o -name 'events.out.tfevents*' \) | head -60
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_11000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_6500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_5000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_17500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_8000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_19000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_3000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_13000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_19999.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_1000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_2500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_7000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_8500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_19500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_14000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_1500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_10500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_17000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_14500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_2000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/events.out.tfevents.1789128286.robotics-Precision-7920-Tower.169845.0
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_4000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_15500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_3500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_18500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_9000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_10000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_13500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_4500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_0.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_18000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_6000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_15000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_9500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_12500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_11500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_16500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_16000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_7500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_5500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-11_19-04-15/model_12000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-13_15-57-03/model_11000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-13_15-57-03/model_6500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-13_15-57-03/model_5000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-13_15-57-03/model_8000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-13_15-57-03/model_3000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-13_15-57-03/model_1000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-13_15-57-03/model_2500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-13_15-57-03/model_7000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-13_15-57-03/model_8500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-13_15-57-03/model_1500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-13_15-57-03/events.out.tfevents.1789289845.robotics-Precision-7920-Tower.769427.0
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-13_15-57-03/model_10500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-13_15-57-03/model_2000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-13_15-57-03/model_4000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-13_15-57-03/model_3500.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-13_15-57-03/model_9000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-13_15-57-03/model_10000.pt
    logs/rsl_rl/vqr_wheel_imu_rough/2026-09-13_15-57-03/model_4500.pt
    find: ‘standard output’: Broken pipe
    find: write error
    ✓ • 0ms

## Activity

    $ find logs/rsl_rl/vqr_wheel_yaw_flat_fsm -maxdepth 2 -type f \( -name 'model_*.pt' -o -name 'contact_trace*.csv' -o -name 'events.out.tfevents*' \) | tail -50
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-23_17-05-23_fsm_fresh/model_0.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_08-14-04/events.out.tfevents.1790212462.robotics-Precision-7920-Tower.2160759.0
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_08-14-04/model_0.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_08-14-04/model_499.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_08-44-25/events.out.tfevents.1790214277.robotics-Precision-7920-Tower.2203024.0
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_08-44-25/model_999.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_08-44-25/model_0.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_08-44-25/model_500.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_09-31-33/model_1000.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_09-31-33/model_1500.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_09-31-33/events.out.tfevents.1790217104.robotics-Precision-7920-Tower.2223296.0
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-23_17-08-07_fsm_fresh/model_299.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-23_17-08-07_fsm_fresh/model_0.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-23_17-08-07_fsm_fresh/events.out.tfevents.1790158104.robotics-Precision-7920-Tower.1625943.0
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-25_14-41-52/events.out.tfevents.1790322128.robotics-Precision-7920-Tower.3018767.0
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-25_14-41-52/model_0.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-23_18-09-47_fsm_fresh/model_1000.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-23_18-09-47_fsm_fresh/model_2500.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-23_18-09-47_fsm_fresh/model_1500.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-23_18-09-47_fsm_fresh/model_2999.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-23_18-09-47_fsm_fresh/model_2000.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-23_18-09-47_fsm_fresh/model_0.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-23_18-09-47_fsm_fresh/model_500.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-23_18-09-47_fsm_fresh/events.out.tfevents.1790161801.robotics-Precision-7920-Tower.1744710.0
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_10-43-31/events.out.tfevents.1790221416.robotics-Precision-7920-Tower.2277863.0
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_10-43-31/model_4.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_10-43-31/model_0.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_11-23-35/contact_trace_model_500.csv
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_11-23-35/model_0.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_11-23-35/model_500.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_11-23-35/contact_trace_batch.csv
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_11-23-35/events.out.tfevents.1790223828.robotics-Precision-7920-Tower.2297911.0
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_11-19-58/events.out.tfevents.1790223604.robotics-Precision-7920-Tower.2295183.0
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_11-19-58/model_4.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_11-19-58/model_0.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-23_17-03-33_fsm_smoke/events.out.tfevents.1790157817.robotics-Precision-7920-Tower.1622906.0
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-23_17-03-33_fsm_smoke/model_0.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-23_18-06-51_fsm_fresh/events.out.tfevents.1790161632.robotics-Precision-7920-Tower.1743302.0
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-23_18-06-51_fsm_fresh/model_0.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-25_16-47-03/model_0.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-25_16-47-03/events.out.tfevents.1790329637.robotics-Precision-7920-Tower.3100170.0
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_10-10-36/model_1000.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_10-10-36/events.out.tfevents.1790219447.robotics-Precision-7920-Tower.2258900.0
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_10-46-04/events.out.tfevents.1790221576.robotics-Precision-7920-Tower.2279849.0
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_10-46-04/model_0.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-23_16-51-58_fsm_smoke/model_4.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-23_16-51-58_fsm_smoke/model_0.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-23_16-51-58_fsm_smoke/events.out.tfevents.1790157122.robotics-Precision-7920-Tower.1614784.0
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-25_11-54-28/model_0.pt
    logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-25_11-54-28/events.out.tfevents.1790312094.robotics-Precision-7920-Tower.2949390.0
    ✓ • 0ms

## Activity

    $ nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
    NVIDIA RTX A6000, 48525 MiB
    NVIDIA RTX A6000, 48328 MiB
    ✓ • 0ms

## Activity

    $ sed -n '25,90p' scripts/reinforcement_learning/rsl_rl/play.py; sed -n '330,460p' scripts/reinforcement_learning/rsl_rl/play.py; sed -n '460,505p' scripts/reinforcement_learning/rsl_rl/play.py
    
    # add argparse arguments
    parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
    parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
    parser.add_argument("--max_steps", type=int, default=None, help="Stop playback after this many environment steps.")
    parser.add_argument(
        "--yaw_contact_trace",
        type=str,
        default=None,
        help="Write per-step yaw FSM wheel clearance and contact forces to this CSV path.",
    )
    parser.add_argument(
        "--yaw_contact_env_id",
        type=int,
        default=0,
        help="Environment index to record with --yaw_contact_trace; use -1 for all environments.",
    )
    parser.add_argument(
        "--yaw_contact_limit",
        type=float,
        default=0.25,
        help="Yaw command limit in rad/s for contact tracing (default: training stage 0).",
    )
    parser.add_argument(
        "--yaw_contact_ready_dwell",
        type=float,
        default=0.0,
        help="Continuous ready-pose dwell in seconds during tracing (default: 0, matching the 2026-09-24 run).",
    )
    parser.add_argument(
        "--yaw_contact_stochastic",
        action="store_true",
        help="Sample actions from the checkpoint policy during contact tracing, as in PPO training.",
    )
    parser.add_argument(
        "--reward_diagnostics",
        action="store_true",
        default=False,
        help="Print the measured weighted reward decomposition during playback.",
    )
    parser.add_argument(
        "--reward_diagnostics_interval",
        type=int,
        default=250,
        help="Playback steps per reward decomposition report.",
    )
    # append RSL-RL cli arguments
    cli_args.add_rsl_rl_args(parser)
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    # parse the arguments
    args_cli, hydra_args = parser.parse_known_args()
    # always enable cameras to record video
    if args_cli.video:
        contact_trace_file = None
        contact_trace_writer = None
        contact_trace_ids = None
        contact_trace_env_ids = ()
        contact_trace_history_length = 0
        contact_trace_skipped_resets = 0
        if args_cli.yaw_contact_trace:
            raw = env.unwrapped
            contact_trace_env_ids = (
                range(raw.num_envs)
                if args_cli.yaw_contact_env_id == -1
                else (args_cli.yaw_contact_env_id,)
            )
            if "yaw_rate_cmd" not in raw.command_manager.active_terms:
                raise ValueError("--yaw_contact_trace requires a yaw_rate_cmd task.")
            wheel_names = ("FL_WHEEL", "FR_WHEEL", "HL_WHEEL", "HR_WHEEL")
            sensor = raw.scene.sensors["contact_forces"]
            robot = raw.scene["robot"]
            force_ids, force_names = sensor.find_bodies(list(wheel_names), preserve_order=True)
            body_ids, body_names = robot.find_bodies(list(wheel_names), preserve_order=True)
            if tuple(force_names) != wheel_names or tuple(body_names) != wheel_names:
                raise RuntimeError(f"Wheel body resolution differs from {wheel_names}: {force_names}, {body_names}")
            contact_trace_ids = (force_ids, body_ids)
            if getattr(sensor.data, "net_forces_w_history", None) is not None:
                contact_trace_history_length = int(sensor.data.net_forces_w_history.shape[1])
            contact_trace_radius = float(raw.command_manager.get_term("yaw_rate_cmd").cfg.wheel_radius)
            contact_trace_threshold = float(raw.command_manager.get_term("yaw_rate_cmd").cfg.contact_threshold)
            trace_path = os.path.abspath(args_cli.yaw_contact_trace)
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            contact_trace_file = open(trace_path, "w", newline="", buffering=1)
            contact_trace_writer = csv.writer(contact_trace_file)
            header = ["step", "time_s", "env_id", "fsm_state", "support_diagonal", "yaw_command", "yaw_rate_b_z", "metric_support_gate"]
            for name in wheel_names:
                prefix = name.removesuffix("_WHEEL")
                header += [f"{prefix}_clearance_m", f"{prefix}_fx_n", f"{prefix}_fy_n", f"{prefix}_fz_n", f"{prefix}_force_norm_n", f"{prefix}_norm_gt_threshold", f"{prefix}_fz_gt_threshold"]
                for history_index in range(contact_trace_history_length):
                    header += [f"{prefix}_history_{history_index}_fz_n", f"{prefix}_history_{history_index}_norm_n"]
            contact_trace_writer.writerow(header)
            print(
                f"[INFO] Yaw contact trace: {trace_path}; {len(contact_trace_env_ids)} environments, "
                f"threshold={contact_trace_threshold} N, radius={contact_trace_radius} m, "
                f"ready_dwell={args_cli.yaw_contact_ready_dwell} s, "
                f"actions={'sampled' if args_cli.yaw_contact_stochastic else 'deterministic'}"
            )
    
        reward_diagnostic_sum = None
        reward_diagnostic_samples = 0
        gate_open_sum = 0.0
        mean_abs_yaw_command_sum = 0.0
        yaw_rate_error_sum = 0.0
        if args_cli.reward_diagnostics:
            reward_manager = env.unwrapped.reward_manager
            reward_diagnostic_sum = torch.zeros(len(reward_manager.active_terms), device=env.unwrapped.device)
            print("[INFO] Reward diagnostics enabled; values are weighted reward rates averaged across environments.")
            if "yaw_rate_cmd" in env.unwrapped.command_manager.active_terms:
                yaw_command_term = env.unwrapped.command_manager.get_term("yaw_rate_cmd")
                initial_abs_command = env.unwrapped.command_manager.get_command("yaw_rate_cmd")[:, 0].abs().mean()
                print(f"[INFO] yaw_rate_cmd range: {yaw_command_term.cfg.yaw_rate_range}")
                print(f"[INFO] initial mean |yaw_rate_cmd|: {initial_abs_command.item():.6f}")
    
        timestep = 0
        # simulate environment
        while simulation_app.is_running():
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
    
                # env stepping
                obs, _, dones, _ = env.step(actions)
            if contact_trace_writer is not None:
                raw = env.unwrapped
                sensor = raw.scene.sensors["contact_forces"]
                robot = raw.scene["robot"]
                command = raw.command_manager.get_term("yaw_rate_cmd")
                force_ids, body_ids = contact_trace_ids
                metric_gate = getattr(raw, "_yaw_support_score_current", None)
                yaw_commands = raw.command_manager.get_command("yaw_rate_cmd")
                # Isaac Lab auto-resets completed environments inside step(); their
                # post-step sensor values no longer belong to the reward sample.
                for env_id in contact_trace_env_ids:
                    if bool(dones[env_id]):
                        contact_trace_skipped_resets += 1
                        continue
                    forces = sensor.data.net_forces_w[env_id, force_ids, :].detach().cpu()
                    history = (
                        [
                            sensor.data.net_forces_w_history[env_id, :, int(force_id), :].detach().cpu()
                            for force_id in force_ids
                        ]
                        if contact_trace_history_length
                        else None
                    )
                    heights = robot.data.body_pos_w[env_id, body_ids, 2].detach().cpu()
                    origin_z = float(raw.scene.env_origins[env_id, 2].item())
                    row = [
                        timestep,
                        (timestep + 1) * dt,
                        env_id,
                        int(command.fsm_state[env_id].item()),
                        int(command.support_diagonal[env_id].item()),
                        float(yaw_commands[env_id, 0].item()),
                        float(robot.data.root_ang_vel_b[env_id, 2].item()),
                        float(metric_gate[env_id].item()) if metric_gate is not None else "",
                    ]
                    for wheel_index, (height, force) in enumerate(zip(heights, forces)):
                        fx, fy, fz = (float(value) for value in force)
                        norm = float(torch.linalg.vector_norm(force).item())
                        row += [float(height) - origin_z - contact_trace_radius, fx, fy, fz, norm, int(norm > contact_trace_threshold), int(fz > contact_trace_threshold)]
                        if history is not None:
                            for history_force in history[wheel_index]:
                                row += [
                                    float(history_force[2].item()),
                                    float(torch.linalg.vector_norm(history_force).item()),
                                ]
                    contact_trace_writer.writerow(row)
            if args_cli.reward_diagnostics:
                reward_manager = env.unwrapped.reward_manager
                reward_diagnostic_sum += reward_manager._step_reward.mean(dim=0)
                reward_diagnostic_samples += 1
                unwrapped_env = env.unwrapped
                if hasattr(unwrapped_env, "_yaw_gate_open_current"):
                    gate_open_sum += float(unwrapped_env._yaw_gate_open_current.mean().item())
                if hasattr(unwrapped_env, "_yaw_command_abs_current"):
                    mean_abs_yaw_command_sum += float(unwrapped_env._yaw_command_abs_current.mean().item())
                    yaw_rate_error_sum += float(unwrapped_env._yaw_rate_abs_error_current.mean().item())
                if reward_diagnostic_samples == args_cli.reward_diagnostics_interval:
                    mean_terms = reward_diagnostic_sum / reward_diagnostic_samples
                    positive_total = mean_terms.clamp_min(0.0).sum()
                    yaw_index = (
                    yaw_index = (
                        reward_manager.active_terms.index("gated_yaw_tracking")
                        if "gated_yaw_tracking" in reward_manager.active_terms
                        else None
                    )
                    print("\n[REWARD DIAGNOSTICS] weighted rate by term")
                    for name, value in zip(reward_manager.active_terms, mean_terms.tolist()):
                        print(f"  {name:32s} {value: .6f}")
                    print(f"  {'positive_total':32s} {positive_total.item(): .6f}")
                    if yaw_index is not None and positive_total.item() > 0.0:
                        yaw_share = mean_terms[yaw_index].clamp_min(0.0) / positive_total
                        print(f"  {'gated_yaw_positive_share':32s} {yaw_share.item(): .3%}")
                    gate_open_rate = gate_open_sum / reward_diagnostic_samples
                    mean_abs_yaw_command = mean_abs_yaw_command_sum / reward_diagnostic_samples
                    mean_yaw_rate_error = yaw_rate_error_sum / reward_diagnostic_samples
                    tracking_ratio = (
                        1.0 - mean_yaw_rate_error / mean_abs_yaw_command
                        if mean_abs_yaw_command > 1.0e-6
                        else 0.0
                    )
                    print(f"  {'gate_open_rate':32s} {gate_open_rate: .6f}")
                    print(f"  {'mean_abs_yaw_cmd':32s} {mean_abs_yaw_command: .6f}")
                    print(f"  {'mean_error_yaw_rate':32s} {mean_yaw_rate_error: .6f}")
                    print(f"  {'tracking_ratio':32s} {tracking_ratio: .6f}")
                    reward_diagnostic_sum.zero_()
                    reward_diagnostic_samples = 0
                    gate_open_sum = 0.0
                    mean_abs_yaw_command_sum = 0.0
                    yaw_rate_error_sum = 0.0
            timestep += 1
            if args_cli.video:
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break
            if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
                break
    
            if args_cli.keyboard:
                camera_follow(env)
    
            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    
        if contact_trace_file is not None:
    ✓ • 0ms

## Activity

    $ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python -m pytest -q tests/test_yaw_*.py
    .............................................................            [100%]
    61 passed in 3.34s
    ✓ • 3.73s

## Activity

    $ /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python scripts/reinforcement_learning/rsl_rl/train.py --task Flat-VQR-Wheel-Yaw-FSM --num_envs 64 --max_iterations 4 --headless
    /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/gymnasium/envs/registration.py:636: UserWarning: WARN: Overriding environment Flat-Deeprobotics-Lite3-v0 already in registry.
      logger.warn(f"Overriding environment {new_spec.id} already in registry.")
    /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/gymnasium/envs/registration.py:636: UserWarning: WARN: Overriding environment Rough-Deeprobotics-Lite3-v0 already in registry.
      logger.warn(f"Overriding environment {new_spec.id} already in registry.")
    /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/gymnasium/envs/registration.py:636: UserWarning: WARN: Overriding environment Flat-VQR-v0 already in registry.
      logger.warn(f"Overriding environment {new_spec.id} already in registry.")
    /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/gymnasium/envs/registration.py:636: UserWarning: WARN: Overriding environment Rough-VQR-v0 already in registry.
      logger.warn(f"Overriding environment {new_spec.id} already in registry.")
    /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/gymnasium/envs/registration.py:636: UserWarning: WARN: Overriding environment Flat-Deeprobotics-M20-v0 already in registry.
      logger.warn(f"Overriding environment {new_spec.id} already in registry.")
    /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/gymnasium/envs/registration.py:636: UserWarning: WARN: Overriding environment Rough-Deeprobotics-M20-v0 already in registry.
      logger.warn(f"Overriding environment {new_spec.id} already in registry.")
    /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/gymnasium/envs/registration.py:636: UserWarning: WARN: Overriding environment Flat-VQR-Wheel already in registry.
      logger.warn(f"Overriding environment {new_spec.id} already in registry.")
    /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/gymnasium/envs/registration.py:636: UserWarning: WARN: Overriding environment Rough-VQR-Wheel already in registry.
      logger.warn(f"Overriding environment {new_spec.id} already in registry.")
    /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/gymnasium/envs/registration.py:636: UserWarning: WARN: Overriding environment Rough-VQR-IMU-Wheel already in registry.
      logger.warn(f"Overriding environment {new_spec.id} already in registry.")
    [INFO][AppLauncher]: Using device: cuda:0
    [INFO][AppLauncher]: Loading experience file: /home/robotics/datld5/IsaacLab/apps/isaaclab.python.headless.kit
    [INFO]: Parsing configuration from: rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.vqr_wheel.yaw_env_fsm_cfg:VQRWheelFlatEnvFSMCfg
    [INFO]: Parsing configuration from: rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.vqr_wheel.agents.rsl_rl_ppo_cfg:VQRWheelYawFlatFSMPPORunnerCfg
    [INFO] Logging experiment in directory: /home/robotics/tuanpm48/vqr/rl_training/logs/rsl_rl/vqr_wheel_yaw_flat_fsm
    Exact experiment name requested from command line: 2026-09-28_10-26-03
    
    ======================================================================================
    [INFO][IsaacLab]: Logging to file: /tmp/isaaclab/logs/isaaclab_2026-09-28_10-26-03.log
    ======================================================================================
    
    10:26:03 [simulation_context.py] WARNING: The `enable_external_forces_every_iteration` parameter in the PhysxCfg is set to False. If you are experiencing noisy velocities, consider enabling this flag. You may need to slightly increase the number of velocity iterations (setting it to 1 or 2 rather than 0), together with this flag, to improve the accuracy of velocity updates.
    [INFO]: Base environment:
        Environment device    : cuda:0
        Environment seed      : 42
        Physics step-size     : 0.005
        Rendering step-size   : 0.02
        Environment step-size : 0.02
    10:26:03 [terrain_importer.py] WARNING: Visual material specified for ground plane but no diffuse color found. Using default color: (0.0, 0.0, 0.0)
    [INFO]: Time taken for scene creation : 2.403917 seconds
    [INFO]: Scene manager:  <class InteractiveScene>
        Number of environments: 64
        Environment spacing   : 2.5
        Source prim name      : /World/envs/env_0
        Global prim paths     : ['/World/ground']
        Replicate physics     : True
    [INFO]: Starting the simulation. This may take a few seconds. Please wait...
    [INFO]: Time taken for simulation start : 0.953950 seconds
    [INFO] Command Manager:  <CommandManager> contains 1 active terms.
    +--------------------------------------+
    |         Active Command Terms         |
    +-------+--------------+---------------+
    | Index | Name         |      Type     |
    +-------+--------------+---------------+
    |   0   | yaw_rate_cmd | YawFSMCommand |
    +-------+--------------+---------------+
    
    [INFO] Event Manager:  <EventManager> contains 3 active terms.
    +----------------------------------------+
    | Active Event Terms in Mode: 'startup'  |
    +-------+--------------------------------+
    | Index | Name                           |
    +-------+--------------------------------+
    |   0   | randomize_rigid_body_material  |
    |   1   | randomize_rigid_body_mass      |
    |   2   | randomize_rigid_body_mass_base |
    |   3   | randomize_rigid_body_inertia   |
    |   4   | randomize_com_positions        |
    +-------+--------------------------------+
    +-----------------------------------------------+
    |      Active Event Terms in Mode: 'reset'      |
    +-------+---------------------------------------+
    | Index | Name                                  |
    +-------+---------------------------------------+
    |   0   | randomize_apply_external_force_torque |
    |   1   | randomize_reset_joints                |
    |   2   | randomize_actuator_gains              |
    |   3   | randomize_reset_base                  |
    +-------+---------------------------------------+
    +--------------------------------------------------------+
    |         Active Event Terms in Mode: 'interval'         |
    +-------+----------------------+-------------------------+
    | Index | Name                 | Interval time range (s) |
    +-------+----------------------+-------------------------+
    |   0   | randomize_push_robot |       (10.0, 15.0)      |
    +-------+----------------------+-------------------------+
    
    [INFO] Recorder Manager:  <RecorderManager> contains 0 active terms.
    +---------------------+
    | Active Recorder Terms |
    +-----------+---------+
    |   Index   | Name    |
    +-----------+---------+
    +-----------+---------+
    
    [INFO] Action Manager:  <ActionManager> contains 2 active terms.
    +------------------------------------+
    |  Active Action Terms (shape: 16)   |
    +--------+-------------+-------------+
    | Index  | Name        |   Dimension |
    +--------+-------------+-------------+
    |   0    | joint_pos   |          12 |
    |   1    | joint_vel   |           4 |
    +--------+-------------+-------------+
    
    [INFO] Observation Manager: <ObservationManager> contains 2 groups.
    +---------------------------------------------------------+
    | Active Observation Terms in Group: 'policy' (shape: (62,)) |
    +-----------+---------------------------------+-----------+
    |   Index   | Name                            |   Shape   |
    +-----------+---------------------------------+-----------+
    |     0     | base_ang_vel                    |    (3,)   |
    |     1     | projected_gravity               |    (3,)   |
    |     2     | yaw_rate_cmd                    |    (1,)   |
    |     3     | joint_pos                       |   (16,)   |
    |     4     | joint_vel                       |   (16,)   |
    |     5     | actions                         |   (16,)   |
    |     6     | fsm_state                       |    (7,)   |
    +-----------+---------------------------------+-----------+
    +--------------------------------------------------------------+
    |  Active Observation Terms in Group: 'critic' (shape: (95,))  |
    +---------+------------------------------------------+---------+
    |  Index  | Name                                     |  Shape  |
    +---------+------------------------------------------+---------+
    |    0    | base_lin_vel                             |   (3,)  |
    |    1    | base_height                              |   (1,)  |
    |    2    | base_ang_vel                             |   (3,)  |
    |    3    | projected_gravity                        |   (3,)  |
    |    4    | yaw_rate_cmd                             |   (1,)  |
    |    5    | joint_pos                                |  (16,)  |
    |    6    | joint_vel                                |  (16,)  |
    |    7    | actions                                  |  (16,)  |
    |    8    | wheel_normal_force                       |   (4,)  |
    |    9    | wheel_contact                            |   (4,)  |
    |    10   | wheel_clearance                          |   (4,)  |
    |    11   | support_wheel_alignment                  |   (2,)  |
    |    12   | com_support_coordinate                   |   (2,)  |
    |    13   | rolling_lateral_contact_velocity         |   (8,)  |
    |    14   | fsm_state                                |   (7,)  |
    |    15   | fsm_ready_flags                          |   (4,)  |
    |    16   | fsm_state_time                           |   (1,)  |
    +---------+------------------------------------------+---------+
    
    [INFO] Termination Manager:  <TerminationManager> contains 8 active terms.
    +-------------------------------------------+
    |          Active Termination Terms         |
    +-------+------------------------+----------+
    | Index | Name                   | Time Out |
    +-------+------------------------+----------+
    |   0   | time_out               |   True   |
    |   1   | terrain_out_of_bounds  |   True   |
    |   2   | torso_contact          |  False   |
    |   3   | base_height_failure    |  False   |
    |   4   | tilt_failure           |  False   |
    |   5   | fsm_transition_timeout |  False   |
    |   6   | fsm_return_timeout     |  False   |
    |   7   | swing_contact_timeout  |  False   |
    +-------+------------------------+----------+
    
    [INFO] Reward Manager:  <RewardManager> contains 26 active terms.
    +----------------------------------------------+
    |             Active Reward Terms              |
    +-------+----------------------------+---------+
    | Index | Name                       |  Weight |
    +-------+----------------------------+---------+
    |   0   | balance                    |     2.0 |
    |   1   | torque                     | -0.0001 |
    |   2   | action_rate                |   -0.02 |
    |   3   | joint_velocity             |  -0.001 |
    |   4   | joint_limits               |    -0.5 |
    |   5   | lateral_slip               |    -2.0 |
    |   6   | undesired_contact          |    -2.0 |
    |   7   | planar_velocity            |    -1.0 |
    |   8   | low_base_height            |    -4.0 |
    |   9   | transition_low_base_height |     4.0 |
    |   10  | downward_low_base_velocity |    -8.0 |
    |   11  | com_support                |     3.0 |
    |   12  | com_inside_segment         |     2.0 |
    |   13  | support_span_band          |    -1.0 |
    |   14  | lift_clearance             |     3.0 |
    |   15  | base_height                |    0.49 |
    |   16  | lifted_wheel_spin          |   -0.02 |
    |   17  | rolling_slip               |    -0.5 |
    |   18  | four_stand_stability       |     3.0 |
    |   19  | return_to_four_landing     |    20.0 |
    |   20  | four_stand_ready_bonus     |    20.0 |
    |   21  | fsm_gated_tracking         |     8.0 |
    |   22  | transition_progress        |     1.0 |
    |   23  | fsm_failure                |   -60.0 |
    |   24  | spin_center_drift          |     0.0 |
    |   25  | safe_recovery_entry        |     0.0 |
    +-------+----------------------------+---------+
    
    [INFO] Curriculum Manager:  <CurriculumManager> contains 1 active terms.
    +--------------------------+
    | Active Curriculum Terms  |
    +---------+----------------+
    |  Index  | Name           |
    +---------+----------------+
    |    0    | task_levels    |
    +---------+----------------+
    
    [INFO]: Completed setting up the environment...
    [INFO] Flat-VQR-Wheel-Yaw-FSM env config source: /home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py
    [INFO] Flat-VQR-Wheel-Yaw-FSM reward terms (26): ['balance', 'torque', 'action_rate', 'joint_velocity', 'joint_limits', 'lateral_slip', 'undesired_contact', 'planar_velocity', 'low_base_height', 'transition_low_base_height', 'downward_low_base_velocity', 'com_support', 'com_inside_segment', 'support_span_band', 'lift_clearance', 'base_height', 'lifted_wheel_spin', 'rolling_slip', 'four_stand_stability', 'return_to_four_landing', 'four_stand_ready_bonus', 'fsm_gated_tracking', 'transition_progress', 'fsm_failure', 'spin_center_drift', 'safe_recovery_entry']/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/rsl_rl/utils/utils.py:245: UserWarning: The observation configuration dictionary 'obs_groups' must contain the 'policy' key. As an observation group with the name 'policy' was found, this is assumed to be the observation set. Consider adding the 'policy' key to the 'obs_groups' dictionary for clarity. This behavior will be removed in a future version.
      warnings.warn(
    /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/rsl_rl/utils/utils.py:283: UserWarning: The observation configuration dictionary 'obs_groups' must contain the 'critic' key. As an observation group with the name 'critic' was found, this is assumed to be the observation set. Consider adding the 'critic' key to the 'obs_groups' dictionary for clarity. This behavior will be removed in a future version.
      warnings.warn(
    
    [INFO] Flat-VQR-Wheel-Yaw-FSM fsm_gated_tracking weight: 8.0
    --------------------------------------------------------------------------------
    Resolved observation sets:
         policy :  ['policy']
         critic :  ['critic']
    --------------------------------------------------------------------------------
    Actor MLP: MLP(
      (0): Linear(in_features=62, out_features=512, bias=True)
      (1): ELU(alpha=1.0)
      (2): Linear(in_features=512, out_features=256, bias=True)
      (3): ELU(alpha=1.0)
      (4): Linear(in_features=256, out_features=128, bias=True)
      (5): ELU(alpha=1.0)
      (6): Linear(in_features=128, out_features=16, bias=True)
    )
    Critic MLP: MLP(
      (0): Linear(in_features=95, out_features=512, bias=True)
      (1): ELU(alpha=1.0)
      (2): Linear(in_features=512, out_features=256, bias=True)
      (3): ELU(alpha=1.0)
      (4): Linear(in_features=256, out_features=128, bias=True)
      (5): ELU(alpha=1.0)
      (6): Linear(in_features=128, out_features=1, bias=True)
    )
    ################################################################################
                             Learning iteration 0/4
    
                           Computation: 566 steps/s (collection: 2.501s, learning 0.212s)
                 Mean action noise std: 1.00
              Mean value_function loss: 158.7664
                   Mean surrogate loss: -0.0204
                     Mean entropy loss: 22.6945
                           Mean reward: -49.66
                   Mean episode length: 18.00
                Episode_Reward/balance: 0.0084
                 Episode_Reward/torque: -0.0133
            Episode_Reward/action_rate: -0.0070
         Episode_Reward/joint_velocity: -0.0069
           Episode_Reward/joint_limits: -0.0000
           Episode_Reward/lateral_slip: -0.0034
      Episode_Reward/undesired_contact: 0.0000
        Episode_Reward/planar_velocity: -0.0014
        Episode_Reward/low_base_height: -0.0001
    Episode_Reward/transition_low_base_height: -0.0441
    Episode_Reward/downward_low_base_velocity: -0.0079
            Episode_Reward/com_support: 0.0000
     Episode_Reward/com_inside_segment: 0.0000
      Episode_Reward/support_span_band: -0.0109
         Episode_Reward/lift_clearance: 0.0000
            Episode_Reward/base_height: 0.0000
      Episode_Reward/lifted_wheel_spin: 0.0000
           Episode_Reward/rolling_slip: 0.0000
    Episode_Reward/four_stand_stability: 0.0083
    Episode_Reward/return_to_four_landing: 0.0000
    Episode_Reward/four_stand_ready_bonus: 0.0000
     Episode_Reward/fsm_gated_tracking: 0.0000
    Episode_Reward/transition_progress: 0.0494
            Episode_Reward/fsm_failure: -1.3750
      Episode_Reward/spin_center_drift: 0.0000
    Episode_Reward/safe_recovery_entry: 0.0000
          Curriculum/task_levels/phase: 0.0000
     Curriculum/task_levels/lift_stage: 0.0000
      Curriculum/task_levels/yaw_stage: 0.0000
      Curriculum/task_levels/yaw_limit: 0.2500
    Curriculum/task_levels/target_clearance: 0.0500
    Curriculum/task_levels/online_dr_scale: 0.3000
    Curriculum/task_levels/stage_steps: 10.5833
    Curriculum/task_levels/consecutive_pass_windows: 0.0000
    Curriculum/task_levels/pos/episodes: 0.0000
    Curriculum/task_levels/neg/episodes: 1.7500
      Curriculum/task_levels/pos/score: 0.0000
      Curriculum/task_levels/neg/score: 0.0000
    Curriculum/task_levels/pos/mean_lift_progress: 0.0000
    Curriculum/task_levels/neg/mean_lift_progress: 0.0000
    Curriculum/task_levels/pos/mean_support_score: 0.0000
    Curriculum/task_levels/neg/mean_support_score: 0.0000
    Curriculum/task_levels/pos/fail_lift_rate: 0.0000
    Curriculum/task_levels/neg/fail_lift_rate: 0.8333
    Curriculum/task_levels/pos/fail_support_rate: 0.0000
    Curriculum/task_levels/neg/fail_support_rate: 0.8333
    Curriculum/task_levels/pos/tracking_ratio: 0.0000
    Curriculum/task_levels/neg/tracking_ratio: 0.0000
      Curriculum/task_levels/pos/drift: 0.0000
      Curriculum/task_levels/neg/drift: 0.0000
    Curriculum/task_levels/window_passed: 0.0000
    Curriculum/task_levels/phase_advanced: 0.0000
    Curriculum/task_levels/yaw_limit_advanced: 0.0000
    Curriculum/task_levels/required_directional_episodes: 1024.0000
    Curriculum/task_levels/state_fraction/0: 0.3722
    Curriculum/task_levels/state_fraction/1: 0.0000
    Curriculum/task_levels/state_fraction/2: 0.0000
    Curriculum/task_levels/state_fraction/3: 0.4611
    Curriculum/task_levels/state_fraction/4: 0.0000
    Curriculum/task_levels/state_fraction/5: 0.0000
    Curriculum/task_levels/state_fraction/6: 0.0000
    Curriculum/task_levels/switch_rate: 0.0857
    Curriculum/task_levels/budget/state/0: 1.4256
    Curriculum/task_levels/budget/state/1: 0.0000
    Curriculum/task_levels/budget/state/2: 0.0000
    Curriculum/task_levels/budget/state/3: 3.8734
    Curriculum/task_levels/budget/state/4: 0.0000
    Curriculum/task_levels/budget/state/5: 0.0000
    Curriculum/task_levels/budget/state/6: 0.0000
     Curriculum/task_levels/budget/pos: 0.0000
     Curriculum/task_levels/budget/neg: 3.8734
    Curriculum/task_levels/transition/support_ready_rate: 0.0129
    Curriculum/task_levels/transition/lift_wheel_1_progress: 0.5421
    Curriculum/task_levels/transition/lift_wheel_2_progress: 0.5671
    Curriculum/task_levels/transition/clearance_ready_rate: 0.1604
    Curriculum/task_levels/transition/attitude_ready_rate: 0.8276
    Curriculum/task_levels/transition/pose_ready_rate: 0.0000
    Curriculum/task_levels/transition/torso_contact_rate: 0.0000
    Curriculum/task_levels/transition/duration_mean: 0.1275
    Curriculum/task_levels/pos/transition_success: 0.0000
    Curriculum/task_levels/pos/yaw_mae: 0.0000
    Curriculum/task_levels/pos/drift_10s: 0.0000
    Curriculum/task_levels/pos/drift_10s_samples: 0.0000
    Curriculum/task_levels/pos/swing_contact_rate: 0.0000
    Curriculum/task_levels/pos/support_loss_max_dwell_s: 0.0000
    Curriculum/task_levels/pos/support_FL_rate: 0.0000
    Curriculum/task_levels/pos/support_HR_rate: 0.0000
    Curriculum/task_levels/neg/transition_success: 0.0000
    Curriculum/task_levels/neg/yaw_mae: 0.0000
    Curriculum/task_levels/neg/drift_10s: 0.0000
    Curriculum/task_levels/neg/drift_10s_samples: 0.0000
    Curriculum/task_levels/neg/swing_contact_rate: 0.0000
    Curriculum/task_levels/neg/support_loss_max_dwell_s: 0.0000
    Curriculum/task_levels/neg/support_FR_rate: 0.0000
    Curriculum/task_levels/neg/support_HL_rate: 0.0000
    Metrics/yaw_rate_cmd/error_yaw_rate: 1.2329
          Episode_Termination/time_out: 0.0208
    Episode_Termination/terrain_out_of_bounds: 0.0000
     Episode_Termination/torso_contact: 0.0247
    Episode_Termination/base_height_failure: 0.0247
      Episode_Termination/tilt_failure: 0.0000
    Episode_Termination/fsm_transition_timeout: 0.0000
    Episode_Termination/fsm_return_timeout: 0.0000
    Episode_Termination/swing_contact_timeout: 0.0000
    --------------------------------------------------------------------------------
                       Total timesteps: 1536
                        Iteration time: 2.71s
                          Time elapsed: 00:00:02
                                   ETA: 00:00:10
    
    Could not find git repository in /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/rsl_rl/__init__.py. Skipping.
    Storing git diff for 'rl_training' in: /home/robotics/tuanpm48/vqr/rl_training/logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-28_10-26-03/git/rl_training.diff
    ################################################################################
                             Learning iteration 1/4
    
                           Computation: 768 steps/s (collection: 1.894s, learning 0.105s)
                 Mean action noise std: 1.00
              Mean value_function loss: 312.7972
                   Mean surrogate loss: -0.0222
                     Mean entropy loss: 22.7031
                           Mean reward: -58.48
                   Mean episode length: 27.97
                Episode_Reward/balance: 0.0176
                 Episode_Reward/torque: -0.0393
            Episode_Reward/action_rate: -0.0207
         Episode_Reward/joint_velocity: -0.0237
           Episode_Reward/joint_limits: -0.0002
           Episode_Reward/lateral_slip: -0.0098
      Episode_Reward/undesired_contact: -0.0020
        Episode_Reward/planar_velocity: -0.0084
        Episode_Reward/low_base_height: -0.0001
    Episode_Reward/transition_low_base_height: -0.0853
    Episode_Reward/downward_low_base_velocity: -0.0109
            Episode_Reward/com_support: 0.0000
     Episode_Reward/com_inside_segment: 0.0000
      Episode_Reward/support_span_band: -0.0742
         Episode_Reward/lift_clearance: 0.0000
            Episode_Reward/base_height: 0.0000
      Episode_Reward/lifted_wheel_spin: 0.0000
           Episode_Reward/rolling_slip: 0.0000
    Episode_Reward/four_stand_stability: 0.0183
    Episode_Reward/return_to_four_landing: 0.0000
    Episode_Reward/four_stand_ready_bonus: 0.0000
     Episode_Reward/fsm_gated_tracking: 0.0000
    Episode_Reward/transition_progress: 0.0896
            Episode_Reward/fsm_failure: -3.0000
      Episode_Reward/spin_center_drift: 0.0000
    Episode_Reward/safe_recovery_entry: 0.0000
          Curriculum/task_levels/phase: 0.0000
     Curriculum/task_levels/lift_stage: 0.0000
      Curriculum/task_levels/yaw_stage: 0.0000
      Curriculum/task_levels/yaw_limit: 0.2500
    Curriculum/task_levels/target_clearance: 0.0500
    Curriculum/task_levels/online_dr_scale: 0.3000
    Curriculum/task_levels/stage_steps: 36.0833
    Curriculum/task_levels/consecutive_pass_windows: 0.0000
    Curriculum/task_levels/pos/episodes: 3.4583
    Curriculum/task_levels/neg/episodes: 6.8750
      Curriculum/task_levels/pos/score: 0.0000
      Curriculum/task_levels/neg/score: 0.0000
    Curriculum/task_levels/pos/mean_lift_progress: 0.0000
    Curriculum/task_levels/neg/mean_lift_progress: 0.0000
    Curriculum/task_levels/pos/mean_support_score: 0.0000
    Curriculum/task_levels/neg/mean_support_score: 0.0000
    Curriculum/task_levels/pos/fail_lift_rate: 0.9167
    Curriculum/task_levels/neg/fail_lift_rate: 1.0000
    Curriculum/task_levels/pos/fail_support_rate: 0.9167
    Curriculum/task_levels/neg/fail_support_rate: 1.0000
    Curriculum/task_levels/pos/tracking_ratio: 0.0000
    Curriculum/task_levels/neg/tracking_ratio: 0.0000
      Curriculum/task_levels/pos/drift: 0.0000
      Curriculum/task_levels/neg/drift: 0.0000
    Curriculum/task_levels/window_passed: 0.0000
    Curriculum/task_levels/phase_advanced: 0.0000
    Curriculum/task_levels/yaw_limit_advanced: 0.0000
    Curriculum/task_levels/required_directional_episodes: 1024.0000
    Curriculum/task_levels/state_fraction/0: 0.5391
    Curriculum/task_levels/state_fraction/1: 0.1585
    Curriculum/task_levels/state_fraction/2: 0.0000
    Curriculum/task_levels/state_fraction/3: 0.3025
    Curriculum/task_levels/state_fraction/4: 0.0000
    Curriculum/task_levels/state_fraction/5: 0.0000
    Curriculum/task_levels/state_fraction/6: 0.0000
    Curriculum/task_levels/switch_rate: 0.0225
    Curriculum/task_levels/budget/state/0: 1.2908
    Curriculum/task_levels/budget/state/1: 0.8791
    Curriculum/task_levels/budget/state/2: 0.0000
    Curriculum/task_levels/budget/state/3: 2.6574
    Curriculum/task_levels/budget/state/4: 0.0000
    Curriculum/task_levels/budget/state/5: 0.0000
    Curriculum/task_levels/budget/state/6: 0.0000
     Curriculum/task_levels/budget/pos: 0.8791
     Curriculum/task_levels/budget/neg: 2.6574
    Curriculum/task_levels/transition/support_ready_rate: 0.0378
    Curriculum/task_levels/transition/lift_wheel_1_progress: 0.5514
    Curriculum/task_levels/transition/lift_wheel_2_progress: 0.4942
    Curriculum/task_levels/transition/clearance_ready_rate: 0.1147
    Curriculum/task_levels/transition/attitude_ready_rate: 0.8648
    Curriculum/task_levels/transition/pose_ready_rate: 0.0000
    Curriculum/task_levels/transition/torso_contact_rate: 0.0000
    Curriculum/task_levels/transition/duration_mean: 0.4157
    Curriculum/task_levels/pos/transition_success: 0.0000
    Curriculum/task_levels/pos/yaw_mae: 0.0000
    Curriculum/task_levels/pos/drift_10s: 0.0000
    Curriculum/task_levels/pos/drift_10s_samples: 0.0000
    Curriculum/task_levels/pos/swing_contact_rate: 0.0000
    Curriculum/task_levels/pos/support_loss_max_dwell_s: 0.0000
    Curriculum/task_levels/pos/support_FL_rate: 0.0000
    Curriculum/task_levels/pos/support_HR_rate: 0.0000
    Curriculum/task_levels/neg/transition_success: 0.0000
    Curriculum/task_levels/neg/yaw_mae: 0.0000
    Curriculum/task_levels/neg/drift_10s: 0.0000
    Curriculum/task_levels/neg/drift_10s_samples: 0.0000
    Curriculum/task_levels/neg/swing_contact_rate: 0.0000
    Curriculum/task_levels/neg/support_loss_max_dwell_s: 0.0000
    Curriculum/task_levels/neg/support_FR_rate: 0.0000
    Curriculum/task_levels/neg/support_HL_rate: 0.0000
    Metrics/yaw_rate_cmd/error_yaw_rate: 1.3974
          Episode_Termination/time_out: 0.0247
    Episode_Termination/terrain_out_of_bounds: 0.0000
     Episode_Termination/torso_contact: 0.2552
    Episode_Termination/base_height_failure: 0.2487
      Episode_Termination/tilt_failure: 0.0065
    Episode_Termination/fsm_transition_timeout: 0.0000
    Episode_Termination/fsm_return_timeout: 0.0000
    Episode_Termination/swing_contact_timeout: 0.0000
    --------------------------------------------------------------------------------
                       Total timesteps: 3072
                        Iteration time: 2.00s
                          Time elapsed: 00:00:04
                                   ETA: 00:00:07
    
    ################################################################################
                             Learning iteration 2/4
    
                           Computation: 968 steps/s (collection: 1.481s, learning 0.105s)
                 Mean action noise std: 1.00
              Mean value_function loss: 266.2777
                   Mean surrogate loss: -0.0203
                     Mean entropy loss: 22.7260
                           Mean reward: -59.92
                   Mean episode length: 34.63
                Episode_Reward/balance: 0.0193
                 Episode_Reward/torque: -0.0568
            Episode_Reward/action_rate: -0.0294
         Episode_Reward/joint_velocity: -0.0317
           Episode_Reward/joint_limits: -0.0003
           Episode_Reward/lateral_slip: -0.0199
      Episode_Reward/undesired_contact: -0.0030
        Episode_Reward/planar_velocity: -0.0109
        Episode_Reward/low_base_height: -0.0001
    Episode_Reward/transition_low_base_height: -0.0937
    Episode_Reward/downward_low_base_velocity: -0.0126
            Episode_Reward/com_support: 0.0000
     Episode_Reward/com_inside_segment: 0.0000
      Episode_Reward/support_span_band: -0.0193
         Episode_Reward/lift_clearance: 0.0000
            Episode_Reward/base_height: 0.0000
      Episode_Reward/lifted_wheel_spin: 0.0000
           Episode_Reward/rolling_slip: 0.0000
    Episode_Reward/four_stand_stability: 0.0210
    Episode_Reward/return_to_four_landing: 0.0000
    Episode_Reward/four_stand_ready_bonus: 0.0000
     Episode_Reward/fsm_gated_tracking: 0.0000
    Episode_Reward/transition_progress: 0.1313
            Episode_Reward/fsm_failure: -3.0000
      Episode_Reward/spin_center_drift: 0.0000
    Episode_Reward/safe_recovery_entry: 0.0000
          Curriculum/task_levels/phase: 0.0000
     Curriculum/task_levels/lift_stage: 0.0000
      Curriculum/task_levels/yaw_stage: 0.0000
      Curriculum/task_levels/yaw_limit: 0.2500
    Curriculum/task_levels/target_clearance: 0.0500
    Curriculum/task_levels/online_dr_scale: 0.3000
    Curriculum/task_levels/stage_steps: 60.0000
    Curriculum/task_levels/consecutive_pass_windows: 0.0000
    Curriculum/task_levels/pos/episodes: 8.9583
    Curriculum/task_levels/neg/episodes: 14.7083
      Curriculum/task_levels/pos/score: 0.0000
      Curriculum/task_levels/neg/score: 0.0000
    Curriculum/task_levels/pos/mean_lift_progress: 0.0000
    Curriculum/task_levels/neg/mean_lift_progress: 0.0000
    Curriculum/task_levels/pos/mean_support_score: 0.0000
    Curriculum/task_levels/neg/mean_support_score: 0.0000
    Curriculum/task_levels/pos/fail_lift_rate: 1.0000
    Curriculum/task_levels/neg/fail_lift_rate: 1.0000
    Curriculum/task_levels/pos/fail_support_rate: 1.0000
    Curriculum/task_levels/neg/fail_support_rate: 1.0000
    Curriculum/task_levels/pos/tracking_ratio: 0.0000
    Curriculum/task_levels/neg/tracking_ratio: 0.0000
      Curriculum/task_levels/pos/drift: 0.0000
      Curriculum/task_levels/neg/drift: 0.0000
    Curriculum/task_levels/window_passed: 0.0000
    Curriculum/task_levels/phase_advanced: 0.0000
    Curriculum/task_levels/yaw_limit_advanced: 0.0000
    Curriculum/task_levels/required_directional_episodes: 1024.0000
    Curriculum/task_levels/state_fraction/0: 0.4487
    Curriculum/task_levels/state_fraction/1: 0.2181
    Curriculum/task_levels/state_fraction/2: 0.0000
    Curriculum/task_levels/state_fraction/3: 0.3332
    Curriculum/task_levels/state_fraction/4: 0.0000
    Curriculum/task_levels/state_fraction/5: 0.0000
    Curriculum/task_levels/state_fraction/6: 0.0000
    Curriculum/task_levels/switch_rate: 0.0176
    Curriculum/task_levels/budget/state/0: 0.9807
    Curriculum/task_levels/budget/state/1: 0.8941
    Curriculum/task_levels/budget/state/2: 0.0000
    Curriculum/task_levels/budget/state/3: 2.4273
    Curriculum/task_levels/budget/state/4: 0.0000
    Curriculum/task_levels/budget/state/5: 0.0000
    Curriculum/task_levels/budget/state/6: 0.0000
     Curriculum/task_levels/budget/pos: 0.8941
     Curriculum/task_levels/budget/neg: 2.4273
    Curriculum/task_levels/transition/support_ready_rate: 0.0438
    Curriculum/task_levels/transition/lift_wheel_1_progress: 0.5444
    Curriculum/task_levels/transition/lift_wheel_2_progress: 0.4754
    Curriculum/task_levels/transition/clearance_ready_rate: 0.1326
    Curriculum/task_levels/transition/attitude_ready_rate: 0.7703
    Curriculum/task_levels/transition/pose_ready_rate: 0.0000
    Curriculum/task_levels/transition/torso_contact_rate: 0.0000
    Curriculum/task_levels/transition/duration_mean: 0.6260
    Curriculum/task_levels/pos/transition_success: 0.0000
    Curriculum/task_levels/pos/yaw_mae: 0.0000
    Curriculum/task_levels/pos/drift_10s: 0.0000
    Curriculum/task_levels/pos/drift_10s_samples: 0.0000
    Curriculum/task_levels/pos/swing_contact_rate: 0.0000
    Curriculum/task_levels/pos/support_loss_max_dwell_s: 0.0000
    Curriculum/task_levels/pos/support_FL_rate: 0.0000
    Curriculum/task_levels/pos/support_HR_rate: 0.0000
    Curriculum/task_levels/neg/transition_success: 0.0000
    Curriculum/task_levels/neg/yaw_mae: 0.0000
    Curriculum/task_levels/neg/drift_10s: 0.0000
    Curriculum/task_levels/neg/drift_10s_samples: 0.0000
    Curriculum/task_levels/neg/swing_contact_rate: 0.0000
    Curriculum/task_levels/neg/support_loss_max_dwell_s: 0.0000
    Curriculum/task_levels/neg/support_FR_rate: 0.0000
    Curriculum/task_levels/neg/support_HL_rate: 0.0000
    Metrics/yaw_rate_cmd/error_yaw_rate: 0.9563
          Episode_Termination/time_out: 0.0156
    Episode_Termination/terrain_out_of_bounds: 0.0000
     Episode_Termination/torso_contact: 0.4928
    Episode_Termination/base_height_failure: 0.4225
      Episode_Termination/tilt_failure: 0.0703
    Episode_Termination/fsm_transition_timeout: 0.0000
    Episode_Termination/fsm_return_timeout: 0.0000
    Episode_Termination/swing_contact_timeout: 0.0000
    --------------------------------------------------------------------------------
                       Total timesteps: 4608
                        Iteration time: 1.59s
                          Time elapsed: 00:00:06
                                   ETA: 00:00:04
    
    ################################################################################
                             Learning iteration 3/4
    
                           Computation: 1054 steps/s (collection: 1.353s, learning 0.104s)
                 Mean action noise std: 1.00
              Mean value_function loss: 159.4159
                   Mean surrogate loss: -0.0236
                     Mean entropy loss: 22.7506
                           Mean reward: -60.78
                   Mean episode length: 40.38
                Episode_Reward/balance: 0.0434
                 Episode_Reward/torque: -0.0843
            Episode_Reward/action_rate: -0.0435
         Episode_Reward/joint_velocity: -0.0457
           Episode_Reward/joint_limits: -0.0003
           Episode_Reward/lateral_slip: -0.0213
      Episode_Reward/undesired_contact: -0.0017
        Episode_Reward/planar_velocity: -0.0155
        Episode_Reward/low_base_height: -0.0002
    Episode_Reward/transition_low_base_height: -0.0947
    Episode_Reward/downward_low_base_velocity: -0.0134
            Episode_Reward/com_support: 0.0000
     Episode_Reward/com_inside_segment: 0.0000
      Episode_Reward/support_span_band: -0.0657
         Episode_Reward/lift_clearance: 0.0000
            Episode_Reward/base_height: 0.0000
      Episode_Reward/lifted_wheel_spin: 0.0000
           Episode_Reward/rolling_slip: 0.0000
    Episode_Reward/four_stand_stability: 0.0451
    Episode_Reward/return_to_four_landing: 0.0000
    Episode_Reward/four_stand_ready_bonus: 0.0000
     Episode_Reward/fsm_gated_tracking: 0.0000
    Episode_Reward/transition_progress: 0.1440
            Episode_Reward/fsm_failure: -3.0000
      Episode_Reward/spin_center_drift: 0.0000
    Episode_Reward/safe_recovery_entry: 0.0000
          Curriculum/task_levels/phase: 0.0000
     Curriculum/task_levels/lift_stage: 0.0000
      Curriculum/task_levels/yaw_stage: 0.0000
      Curriculum/task_levels/yaw_limit: 0.2500
    Curriculum/task_levels/target_clearance: 0.0500
    Curriculum/task_levels/online_dr_scale: 0.3000
    Curriculum/task_levels/stage_steps: 83.2500
    Curriculum/task_levels/consecutive_pass_windows: 0.0000
    Curriculum/task_levels/pos/episodes: 14.5833
    Curriculum/task_levels/neg/episodes: 19.8333
      Curriculum/task_levels/pos/score: 0.0000
      Curriculum/task_levels/neg/score: 0.0000
    Curriculum/task_levels/pos/mean_lift_progress: 0.0000
    Curriculum/task_levels/neg/mean_lift_progress: 0.0000
    Curriculum/task_levels/pos/mean_support_score: 0.0000
    Curriculum/task_levels/neg/mean_support_score: 0.0000
    Curriculum/task_levels/pos/fail_lift_rate: 1.0000
    Curriculum/task_levels/neg/fail_lift_rate: 1.0000
    Curriculum/task_levels/pos/fail_support_rate: 1.0000
    Curriculum/task_levels/neg/fail_support_rate: 1.0000
    Curriculum/task_levels/pos/tracking_ratio: 0.0000
    Curriculum/task_levels/neg/tracking_ratio: 0.0000
      Curriculum/task_levels/pos/drift: 0.0000
      Curriculum/task_levels/neg/drift: 0.0000
    Curriculum/task_levels/window_passed: 0.0000
    Curriculum/task_levels/phase_advanced: 0.0000
    Curriculum/task_levels/yaw_limit_advanced: 0.0000
    Curriculum/task_levels/required_directional_episodes: 1024.0000
    Curriculum/task_levels/state_fraction/0: 0.4167
    Curriculum/task_levels/state_fraction/1: 0.2503
    Curriculum/task_levels/state_fraction/2: 0.0000
    Curriculum/task_levels/state_fraction/3: 0.3330
    Curriculum/task_levels/state_fraction/4: 0.0000
    Curriculum/task_levels/state_fraction/5: 0.0000
    Curriculum/task_levels/state_fraction/6: 0.0000
    Curriculum/task_levels/switch_rate: 0.0150
    Curriculum/task_levels/budget/state/0: 0.8976
    Curriculum/task_levels/budget/state/1: 1.1282
    Curriculum/task_levels/budget/state/2: 0.0000
    Curriculum/task_levels/budget/state/3: 2.1109
    Curriculum/task_levels/budget/state/4: 0.0000
    Curriculum/task_levels/budget/state/5: 0.0000
    Curriculum/task_levels/budget/state/6: 0.0000
     Curriculum/task_levels/budget/pos: 1.1282
     Curriculum/task_levels/budget/neg: 2.1109
    Curriculum/task_levels/transition/support_ready_rate: 0.0561
    Curriculum/task_levels/transition/lift_wheel_1_progress: 0.5460
    Curriculum/task_levels/transition/lift_wheel_2_progress: 0.4706
    Curriculum/task_levels/transition/clearance_ready_rate: 0.1230
    Curriculum/task_levels/transition/attitude_ready_rate: 0.7771
    Curriculum/task_levels/transition/pose_ready_rate: 0.0005
    Curriculum/task_levels/transition/torso_contact_rate: 0.0000
    Curriculum/task_levels/transition/duration_mean: 0.7790
    Curriculum/task_levels/pos/transition_success: 0.0000
    Curriculum/task_levels/pos/yaw_mae: 0.0000
    Curriculum/task_levels/pos/drift_10s: 0.0000
    Curriculum/task_levels/pos/drift_10s_samples: 0.0000
    Curriculum/task_levels/pos/swing_contact_rate: 0.0000
    Curriculum/task_levels/pos/support_loss_max_dwell_s: 0.0000
    Curriculum/task_levels/pos/support_FL_rate: 0.0000
    Curriculum/task_levels/pos/support_HR_rate: 0.0000
    Curriculum/task_levels/neg/transition_success: 0.0000
    Curriculum/task_levels/neg/yaw_mae: 0.0000
    Curriculum/task_levels/neg/drift_10s: 0.0000
    Curriculum/task_levels/neg/drift_10s_samples: 0.0000
    Curriculum/task_levels/neg/swing_contact_rate: 0.0000
    Curriculum/task_levels/neg/support_loss_max_dwell_s: 0.0000
    Curriculum/task_levels/neg/support_FR_rate: 0.0000
    Curriculum/task_levels/neg/support_HL_rate: 0.0000
    Metrics/yaw_rate_cmd/error_yaw_rate: 0.9848
          Episode_Termination/time_out: 0.0033
    Episode_Termination/terrain_out_of_bounds: 0.0000
     Episode_Termination/torso_contact: 0.6335
    Episode_Termination/base_height_failure: 0.5345
      Episode_Termination/tilt_failure: 0.0990
    Episode_Termination/fsm_transition_timeout: 0.0000
    Episode_Termination/fsm_return_timeout: 0.0000
    Episode_Termination/swing_contact_timeout: 0.0000
    --------------------------------------------------------------------------------
                       Total timesteps: 6144
                        Iteration time: 1.46s
                          Time elapsed: 00:00:07
                                   ETA: 00:00:01
    
    Loading user config located at: '/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/isaacsim/kit/data/Kit/Isaac-Sim/5.1/user.config.json'
    [Info] [carb] Logging to file: /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/kit_20260928_102552.log
    2026-09-28T03:25:52Z [251ms] [Warning] [omni.usd_config.extension] Enable omni.materialx.libs extension to use MaterialX
    2026-09-28T03:25:53Z [925ms] [Warning] [omni.platforminfo.plugin] failed to open the default display.  Can't verify X Server version.
    2026-09-28T03:25:53Z [1,256ms] [Warning] [carb] Acquiring non optional plugin interface which is not listed as dependency: [omni::physx::IPhysxBenchmarks v1.0] (plugin: <default plugin>), by client: omni.physics.physx.plugin. Add it to CARB_PLUGIN_IMPL_DEPS() macro of a client.
    2026-09-28T03:25:53Z [1,273ms] [Warning] [omni.isaac.dynamic_control] omni.isaac.dynamic_control is deprecated as of Isaac Sim 4.5. No action is needed from end-users.
    2026-09-28T03:25:58Z [6,134ms] [Warning] [carb.cudainterop.plugin] CUDA_VISIBLE_DEVICES environment variable is set.
    2026-09-28T03:25:58Z [6,134ms] [Warning] [carb.cudainterop.plugin] Note CUDA device enumeration and Omniverse device enumeration are different.
    2026-09-28T03:25:58Z [6,134ms] [Warning] [carb.cudainterop.plugin] Setting CUDA_VISIBLE_DEVICES can lead to undesired behavior or crashes.
    2026-09-28T03:25:58Z [6,570ms] [Warning] [gpu.foundation.plugin] Skipping NVIDIA GPU due CUDA being in bad state: NVIDIA RTX A6000
    2026-09-28T03:25:58Z [6,570ms] [Warning] [gpu.foundation.plugin] Please restart your system if CUDA is known to work in your system.
    2026-09-28T03:25:58Z [6,570ms] [Warning] [gpu.foundation.plugin] Skipping NVIDIA GPU due CUDA being in bad state: NVIDIA RTX A6000
    2026-09-28T03:25:58Z [6,570ms] [Warning] [gpu.foundation.plugin] Please restart your system if CUDA is known to work in your system.
    
    |---------------------------------------------------------------------------------------------|
    | Driver Version: 580.178.04    | Graphics API: Vulkan
    |=============================================================================================|
    | GPU | Name                             | Active | LDA | GPU Memory | Vendor-ID | LUID       |
    |     |                                  |        |     |            | Device-ID | UUID       |
    |     |                                  |        |     |            | Bus-ID    |            |
    |---------------------------------------------------------------------------------------------|
    | 0   | NVIDIA RTX A6000                 | Yes: 0 |     | 49386   MB | 10de      | 0          |
    |     |                                  |        |     |            | 2230      | 28bcdf9d.. |
    |     |                                  |        |     |            | 17        |            |
    |---------------------------------------------------------------------------------------------|
    | 1   | NVIDIA RTX A6000                 |        |     | 49386   MB | 10de      | 0          |
    |     |                                  |        |     |            | 2230      | cfb2dc46.. |
    |     |                                  |        |     |            | 73        |            |
    |=============================================================================================|
    | OS: 22.04.5 LTS (Jammy Jellyfish) ubuntu, Version: 22.04.5, Kernel: 6.8.0-106-generic
    | Processor: Intel(R) Xeon(R) Gold 6258R CPU @ 2.70GHz
    | Cores: 56 | Logical Cores: 112
    |---------------------------------------------------------------------------------------------|
    | Total Memory (MB): 128496 | Free Memory: 122185
    | Total Page/Swap (MB): 0 | Free Page/Swap: 0
    |---------------------------------------------------------------------------------------------|
    2026-09-28T03:25:59Z [6,881ms] [Warning] [gpu.foundation.plugin] CPU performance profile is set to powersave. This profile sets the CPU to the lowest frequency reducing performance.
    ✓ • 25.17s

## Activity

    $ /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python - <<'PY'
        import sys
        from types import SimpleNamespace
        import torch
        sys.path.insert(0, 'tests')
        from test_yaw_fsm_systemic import _reward_scene, _pose_rewards, _load_nodes
        from test_yaw_fsm import REWARDS_PATH
        
        DT=.02; GAMMA=.99; STEPS=100
        
        def evaluate(kind):
            rewards,state,env,command=_reward_scene()
            _load_nodes(REWARDS_PATH, {'fsm_failure_cost'}, rewards)
            progress=rewards['TransitionProgress'](None,env)
            args=dict(asset_cfg=SimpleNamespace(name='pos'),asset_cfg_mirror=SimpleNamespace(name='neg'),
                wheel_radius=.091,target_clearance=.05,fsm_command_name='yaw_rate_cmd',
                support_sensor_cfg=SimpleNamespace(name='contact_forces',body_ids=[0,3]),
                support_sensor_cfg_mirror=SimpleNamespace(name='contact_forces',body_ids=[1,2]),
                com_asset_cfg=SimpleNamespace(name='robot'),com_asset_cfg_mirror=SimpleNamespace(name='robot'))
            env.termination_manager=SimpleNamespace(terminated=torch.zeros(2,dtype=torch.bool))
            returns=torch.zeros(2); totals=torch.zeros(2)
            for step in range(STEPS):
                if kind=='failure' and step>8: break
                env.common_step_counter += 1
                command.just_switched.fill_(False)
                command.fsm_state[:]=torch.tensor([state.TRANSITION_POS,state.TRANSITION_NEG])
                lift = 0.0
                if kind=='success' and step>=10: lift=1.0
                if kind=='continue' and step>=50: lift=.4
                if kind=='success' and step>=15:
                    command.fsm_state[:]=torch.tensor([state.YAW_POS,state.YAW_NEG])
                for value in env.lift_progress.values(): value.fill_(lift)
                reward_rate=_pose_rewards(rewards,env).sum(dim=0)+progress(env,**args)
                if kind=='failure' and step==8:
                    env.termination_manager.terminated.fill_(True)
                    reward_rate -= 60.0*rewards['fsm_failure_cost'](env)
                step_reward=reward_rate*DT
                totals += step_reward
                returns += GAMMA**step*step_reward
            return totals.tolist(),returns.tolist()
        for kind in ('success','continue','failure'):
            total,discounted=evaluate(kind)
            print(f'{kind}: undiscounted POS={total[0]:.6f} NEG={total[1]:.6f}; discounted POS={discounted[0]:.6f} NEG={discounted[1]:.6f}')
        PY
    success: undiscounted POS=35.953037 NEG=35.953037; discounted POS=22.835224 NEG=22.835224
    continue: undiscounted POS=2.400000 NEG=2.400000; discounted POS=1.452015 NEG=1.452015
    failure: undiscounted POS=-60.000000 NEG=-60.000000; discounted POS=-55.364681 NEG=-55.364681
    ✓ • 2.23s

## Activity

    $ /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python - <<'PY'
        import sys
        from types import SimpleNamespace
        import torch
        sys.path.insert(0,'tests')
        from test_yaw_fsm_systemic import _reward_scene
        rewards,state,env,command=_reward_scene()
        term=rewards['TransitionProgress'](None,env)
        args=dict(asset_cfg=SimpleNamespace(name='pos'),asset_cfg_mirror=SimpleNamespace(name='neg'),wheel_radius=.091,target_clearance=.05,fsm_command_name='yaw_rate_cmd',support_sensor_cfg=SimpleNamespace(name='contact_forces',body_ids=[0,3]),support_sensor_cfg_mirror=SimpleNamespace(name='contact_forces',body_ids=[1,2]),com_asset_cfg=SimpleNamespace(name='robot'),com_asset_cfg_mirror=SimpleNamespace(name='robot'))
        geometry={'distance':.32,'projection':-1.0}
        def geom(_env,_cfg):
         return torch.full((2,),geometry['distance']),torch.full((2,),geometry['projection']),torch.ones(2)
        rewards['_yaw_support_geometry']=geom
        term(env,**args)
        env.scene.sensors['contact_forces'].data.net_forces_w.zero_()
        for values in env.lift_progress.values(): values.fill_(1.0)
        geometry.update(distance=0.,projection=.5)
        env.common_step_counter+=1
        raw=term(env,**args)
        weighted=raw*env.step_dt
        print('support=0 on both diagonals; new lift=1, CoM distance .32->0, projection -1->.5')
        print('raw rate',raw.tolist(),'episode-scale credit',weighted.tolist())
        PY
    support=0 on both diagonals; new lift=1, CoM distance .32->0, projection -1->.5
    raw rate [295.0, 295.0] episode-scale credit [5.900000095367432, 5.900000095367432]
    ✓ • 1.66s

## Assistant

The controlled return ordering is correct for both diagonals, but the topology check found a concrete blocker: with **zero support load**, a lift plus CoM improvement still receives **+5.9 episode reward**. That is enough to violate the requested support constraint, so I’m recording it as a validation failure. I’ll finish the height, watchdog, telemetry, and simulator checks without changing reward design during this pass.

## Activity

    $ /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python - <<'PY'
        import sys
        import torch
        from types import SimpleNamespace
        sys.path.insert(0,'tests')
        from test_yaw_transition_safety import _transition_scene
        rewards,state,env,command=_transition_scene()
        term=rewards['TransitionHeightProgress'](None,env)
        args=dict(safe_height=.50,minimum_height=.35,fsm_command_name='yaw_rate_cmd',asset_cfg=SimpleNamespace(name='robot'))
        height=env.scene['robot'].data.root_pos_w[:,2]
        for start in (.49,.45,.40,.36):
         term.reset(); height.fill_(start); entry=term(env,**args)*4*env.step_dt
         height.fill_(start-.01); descent=term(env,**args)*4*env.step_dt
         stationary=term(env,**args)*4*env.step_dt
         height.fill_(start); recovery=term(env,**args)*4*env.step_dt
         print(f'{start:.2f}->{start-.01:.2f}: entry={entry[0].item():.6f} descent={descent[0].item():.6f} stationary={stationary[0].item():.6f} recovery={recovery[0].item():.6f} net={(descent+recovery)[0].item():.6f}')
        PY
    0.49->0.48: entry=0.000000 descent=-0.160000 stationary=0.000000 recovery=0.160000 net=0.000000
    0.45->0.44: entry=0.000000 descent=-0.231111 stationary=0.000000 recovery=0.231111 net=0.000000
    0.40->0.39: entry=0.000000 descent=-0.320001 stationary=0.000000 recovery=0.320001 net=0.000000
    0.36->0.35: entry=0.000000 descent=-0.391112 stationary=0.000000 recovery=0.391112 net=0.000000
    ✓ • 1.69s

## Activity

    $ sed -n '195,325p' tests/test_yaw_fsm_systemic.py; sed -n '225,285p' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py; sed -n '700,780p' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py
        namespace = {"torch": torch, "ManagerBasedRLEnv": object, "VQRFsmState": module.VQRFsmState}
        _load_nodes(FSM_PATH.with_name("terminations.py"), {"fsm_transition_timeout"}, namespace)
        env = SimpleNamespace(num_envs=2, device="cpu", command_manager=SimpleNamespace(get_term=lambda _: fsm))
        false = torch.zeros(2, dtype=torch.bool)
        true = ~false
        command = torch.tensor([.2, -.2])
        for _ in range(400):
            ready = (fsm.fsm_state != 2) & (fsm.fsm_state != 4)
            fsm.update(command, ready, ready, false, false)
            if namespace["fsm_transition_timeout"](env).all():
                break
        else:
            pytest.fail("Chatter escaped the cumulative transition deadline")
        assert torch.all(fsm.state_time < .11)
        assert torch.all(fsm.transition_time >= 3.)
        assert torch.allclose(fsm.transition_time[0], fsm.transition_time[1])
        fsm.reset()
        for _ in range(6):
            fsm.update(command, true, true, false, false)
        budget = fsm.transition_time.clone()
        for _ in range(500):
            fsm.update(command, true, true, false, false)
            assert not namespace["fsm_transition_timeout"](env).any()
        assert torch.equal(fsm.transition_time, budget)
        # Both command exits and unsafe priority still end the maneuver.
        fsm.update(-command, true, true, false, torch.tensor([False, True]))
        assert fsm.fsm_state.tolist() == [5, 6]
        assert not fsm.transition_time.any()
    
    
    def _attempt_recorder(env):
        namespace = {"torch": torch}
        _load_nodes(REWARDS_PATH, {"_fsm_attempt_telemetry"}, namespace)
    
        def record(states, terminal=False, failed=False, quality=1.):
            states = torch.tensor(states)
            env.reset_buf = torch.full((env.num_envs,), terminal)
            env.reset_terminated = torch.full((env.num_envs,), failed)
            gates = dict(b_trans=(states == 1) | (states == 3), b_yaw=(states == 2) | (states == 4),
                         b_safe=states == 6, support_diagonal=torch.tensor([1, -1, 1]))
            namespace["_fsm_attempt_telemetry"](env, gates, torch.full((env.num_envs,), quality),
                                               torch.full((env.num_envs,), quality))
        return record
    
    
    @pytest.mark.parametrize("phase", [0, 1, 2])
    def test_curriculum_counts_failed_attempts_even_with_no_yaw_or_a_prior_success(monkeypatch, phase):
        curriculum = _load_curriculums_module(monkeypatch)
        env = _fsm_checkpoint_env(curriculum, 10000)
        params = _fsm_curriculum_params()
        params.update(min_directional_episodes=2, required_consecutive_windows=1,
                      min_clearance_stage_steps=0, min_yaw_stage_steps=0)
        env.cfg.curriculum.task_levels.params = params
        env._yaw_fsm_task_curriculum_phase = phase
        env.episode_length_buf.fill_(100)
        record = _attempt_recorder(env)
        # First maneuver succeeds, then another times out without ever entering YAW.
        record([1, 3, 0])
        record([2, 4, 0])
        record([5, 5, 0])
        record([0, 0, 0])
        record([1, 3, 0])
        record([1, 3, 0], terminal=True, failed=True)
        # Favorable YAW-only tracking/drift must not bypass the attempt denominator.
        for suffix, index in (("pos", 0), ("neg", 1)):
            for field, value in (("yaw_samples", 1), ("command_abs_sum", .2),
                                 ("yaw_abs_error_sum", 0.), ("drift_sum", 0.), ("drift_samples", 1)):
                buffer = torch.zeros(3)
                buffer[index] = value
                setattr(env, f"_yaw_fsm_{suffix}_{field}", buffer)
        result = curriculum.yaw_fsm_task_levels(env, torch.arange(3), **params)
        assert result["pos/score"] == .5 and result["neg/score"] == .5
        assert result["window_passed"] == 0
        assert result["phase"] == phase and result["lift_stage"] == 0 and result["yaw_stage"] == 0
        assert not env._yaw_fsm_attempt_diagonal.any()
        assert not env._yaw_fsm_pos_transition_attempted.any()
    
    
    @pytest.mark.parametrize("phase", [0, 1, 2])
    def test_curriculum_successful_attempts_still_promote(monkeypatch, phase):
        curriculum = _load_curriculums_module(monkeypatch)
        env = _fsm_checkpoint_env(curriculum, 10000)
        params = _fsm_curriculum_params()
        params.update(min_directional_episodes=1, required_consecutive_windows=1,
                      min_clearance_stage_steps=0, min_yaw_stage_steps=0)
        env.cfg.curriculum.task_levels.params = params
        env._yaw_fsm_task_curriculum_phase = phase
        env.episode_length_buf.fill_(100)
        record = _attempt_recorder(env)
        record([1, 3, 0])
        record([2, 4, 0], terminal=True)  # Healthy episode truncation is not a failure.
        for suffix, index in (("pos", 0), ("neg", 1)):
            for field, value in (("yaw_samples", 1), ("command_abs_sum", .2), ("yaw_abs_error_sum", 0.),
                                 ("drift_sum", 0.), ("drift_samples", 1)):
                buffer = torch.zeros(3)
                buffer[index] = value
                setattr(env, f"_yaw_fsm_{suffix}_{field}", buffer)
        result = curriculum.yaw_fsm_task_levels(env, torch.arange(3), **params)
        assert result["window_passed"] == 1
        assert result["pos/score"] == 1 and result["neg/score"] == 1
        assert result[{0: "lift_stage", 1: "phase", 2: "yaw_stage"}[phase]] == (2 if phase == 1 else 1)
    
    
    def test_reacquisition_timeout_revokes_provisional_success_and_counts_chatter_once(monkeypatch):
        env = _fsm_checkpoint_env(_load_curriculums_module(monkeypatch), 0)
        record = _attempt_recorder(env)
        for _ in range(10):
            record([1, 3, 0])
            record([2, 4, 0])
        record([1, 3, 0], terminal=True, failed=True)
        assert env._yaw_fsm_pos_transition_attempted.tolist() == [1, 0, 0]
        assert env._yaw_fsm_neg_transition_attempted.tolist() == [0, 1, 0]
        assert not env._yaw_fsm_pos_transition_succeeded.any()
        assert not env._yaw_fsm_neg_pose_succeeded.any()
    
    
    def test_safe_recovery_fails_attempt_without_requiring_episode_termination(monkeypatch):
        env = _fsm_checkpoint_env(_load_curriculums_module(monkeypatch), 0)
        record = _attempt_recorder(env)
        record([1, 3, 0])
        record([2, 4, 0])
        record([6, 6, 0])  # Phase C safety handles this without reset_terminated.
        assert env._yaw_fsm_pos_transition_attempted.sum() == 1
        assert env._yaw_fsm_neg_transition_attempted.sum() == 1
        assert not env._yaw_fsm_pos_transition_succeeded.any()
        assert not env._yaw_fsm_neg_pose_succeeded.any()
            yaw_min_dwell: float = 0.20,
            yaw_pose_loss_grace: float = 0.10,
            recovery_dwell: float = 0.5,
            yaw_pose_ready_dwell: float = 0.10,
        ):
            self.num_envs = int(num_envs)
            if self.num_envs < 1:
                raise ValueError(f"num_envs must be positive, got {num_envs}")
    
            self.device = torch.device(device)
            # Keep the hysteresis comparisons at Python-scalar precision.  This
            # matches the scalar oracle when a float32 command is materialized as
            # a Python value, including right on a threshold.
            self.yaw_enter = torch.as_tensor(yaw_enter, dtype=torch.float64, device=self.device)
            self.yaw_exit = torch.as_tensor(yaw_exit, dtype=torch.float64, device=self.device)
            self.dt = torch.as_tensor(dt, dtype=torch.float32, device=self.device)
            self.yaw_min_dwell = torch.as_tensor(
                yaw_min_dwell, dtype=torch.float32, device=self.device
            )
            if yaw_pose_ready_dwell < 0.0:
                raise ValueError("yaw_pose_ready_dwell must be non-negative.")
            self.yaw_pose_ready_dwell = torch.as_tensor(
                yaw_pose_ready_dwell, dtype=torch.float32, device=self.device
            )
            if yaw_pose_loss_grace < 0.0:
                raise ValueError("yaw_pose_loss_grace must be non-negative.")
            self.yaw_pose_loss_grace = torch.as_tensor(
                yaw_pose_loss_grace, dtype=torch.float32, device=self.device
            )
            self.recovery_dwell = torch.as_tensor(
                recovery_dwell, dtype=torch.float32, device=self.device
            )
    
            self.fsm_state = torch.full(
                (self.num_envs,),
                int(VQRFsmState.FOUR_STAND),
                dtype=torch.long,
                device=self.device,
            )
            self.support_diagonal = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.state_time = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            self.transition_time = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            self.just_switched = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.just_returned_to_four = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._recovery_safe_time = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            self._yaw_pose_invalid_time = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            self._yaw_pose_ready_time = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
        if not hasattr(env, sum_name):
            setattr(env, sum_name, torch.zeros_like(value))
            setattr(env, samples_name, torch.zeros_like(value, dtype=torch.long))
        getattr(env, sum_name).add_(value * mask)
        getattr(env, samples_name).add_(mask.to(dtype=torch.long))
    
    
    def _fsm_masked_accumulate_pair(
        env: ManagerBasedRLEnv,
        first: torch.Tensor,
        second: torch.Tensor,
        mask: torch.Tensor,
        first_sum_name: str,
        second_sum_name: str,
        samples_name: str,
    ) -> None:
        """Accumulate two metrics with one shared, masked sample count."""
        if not hasattr(env, first_sum_name):
            setattr(env, first_sum_name, torch.zeros_like(first))
        if not hasattr(env, second_sum_name):
            setattr(env, second_sum_name, torch.zeros_like(second))
        if not hasattr(env, samples_name):
            setattr(env, samples_name, torch.zeros_like(mask, dtype=torch.long))
        getattr(env, first_sum_name).add_(first * mask)
        getattr(env, second_sum_name).add_(second * mask)
        getattr(env, samples_name).add_(mask.to(dtype=torch.long))
    
    
    def _fsm_attempt_telemetry(
        env, gates, lift_progress, support_gate,
        lift_progress_threshold: float = 0.80, support_threshold: float = 0.85,
    ) -> None:
        """Count maneuvers, not YAW survivors or repeated reacquisition visits.
    
        Rewards run after termination computation and before command advancement.
        Finalize a live attempt on its terminal reward step, or when its FSM enters
        RETURN/SAFE. A timeout/fall cannot retain a provisional YAW success.
        """
        if not hasattr(env, "_yaw_fsm_attempt_diagonal"):
            env._yaw_fsm_attempt_diagonal = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
            env._yaw_fsm_attempt_yaw_samples = torch.zeros_like(env._yaw_fsm_attempt_diagonal)
            env._yaw_fsm_attempt_lift_sum = torch.zeros_like(lift_progress)
            env._yaw_fsm_attempt_support_sum = torch.zeros_like(support_gate)
            for suffix in ("pos", "neg"):
                for field in ("transition_attempted", "transition_succeeded", "pose_succeeded",
                              "lift_failed", "support_failed"):
                    setattr(env, f"_yaw_fsm_{suffix}_{field}", torch.zeros_like(env._yaw_fsm_attempt_diagonal))
        active = gates["b_trans"] | gates["b_yaw"]
        diagonal = env._yaw_fsm_attempt_diagonal
        entering = active & (diagonal == 0)
        diagonal.copy_(torch.where(entering, gates["support_diagonal"], diagonal))
        for suffix, sign in (("pos", 1), ("neg", -1)):
            getattr(env, f"_yaw_fsm_{suffix}_transition_attempted").add_((entering & (diagonal == sign)).long())
    
        yaw = gates["b_yaw"]
        env._yaw_fsm_attempt_yaw_samples += yaw.long()
        env._yaw_fsm_attempt_lift_sum += lift_progress * yaw
        env._yaw_fsm_attempt_support_sum += support_gate * yaw
        finishing = (diagonal != 0) & (~active | env.reset_buf)
        samples = env._yaw_fsm_attempt_yaw_samples
        lift_ok = (samples > 0) & (
            env._yaw_fsm_attempt_lift_sum / samples.clamp_min(1) >= lift_progress_threshold
        )
        support_ok = (samples > 0) & (
            env._yaw_fsm_attempt_support_sum / samples.clamp_min(1) >= support_threshold
        )
        reached = (samples > 0) & ~env.reset_terminated & ~gates["b_safe"]
        for suffix, sign in (("pos", 1), ("neg", -1)):
            ended = finishing & (diagonal == sign)
            for field, outcome in (
                ("transition_succeeded", reached),
                ("pose_succeeded", reached & lift_ok & support_ok),
                ("lift_failed", ~lift_ok), ("support_failed", ~support_ok),
            ):
                getattr(env, f"_yaw_fsm_{suffix}_{field}").add_((ended & outcome).long())
        diagonal.masked_fill_(finishing, 0)
        env._yaw_fsm_attempt_yaw_samples.masked_fill_(finishing, 0)
        env._yaw_fsm_attempt_lift_sum.masked_fill_(finishing, 0.0)
        env._yaw_fsm_attempt_support_sum.masked_fill_(finishing, 0.0)
    
    
    ✓ • 0ms

## Activity

    $ sed -n '325,415p' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py; sed -n '1055,1100p' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py
                if index.ndim == 0:
                    index = index.unsqueeze(0)
    
            self.fsm_state[index] = int(VQRFsmState.FOUR_STAND)
            self.support_diagonal[index] = 0
            self.state_time[index] = 0.0
            self.transition_time[index] = 0.0
            self.just_switched[index] = False
            self.just_returned_to_four[index] = False
            self._recovery_safe_time[index] = 0.0
            self._yaw_pose_invalid_time[index] = 0.0
            self._yaw_pose_ready_time[index] = 0.0
    
        def update(
            self,
            yaw_cmd: torch.Tensor,
            positive_pose_ready: torch.Tensor,
            negative_pose_ready: torch.Tensor,
            four_stand_ready: torch.Tensor,
            unsafe: torch.Tensor,
        ) -> torch.Tensor:
            """Advance every FSM by one update and return the live state buffer.
    
            All predicates are evaluated from the state at the start of this
            update.  In particular, an environment in ``RETURN_TO_4`` can only
            enter ``FOUR_STAND`` here; its yaw command is considered on the next
            update, preventing a direct ``RETURN_TO_4 -> TRANSITION_*`` edge.
            """
            yaw = self._as_env_vector(yaw_cmd, name="yaw_cmd", dtype=torch.float64)
            positive_ready = self._as_env_vector(
                positive_pose_ready, name="positive_pose_ready", dtype=torch.bool
            )
            negative_ready = self._as_env_vector(
                negative_pose_ready, name="negative_pose_ready", dtype=torch.bool
            )
            four_ready = self._as_env_vector(
                four_stand_ready, name="four_stand_ready", dtype=torch.bool
            )
            unsafe_mask = self._as_env_vector(unsafe, name="unsafe", dtype=torch.bool)
    
            previous = self.fsm_state
            next_state = previous.clone()
    
            # Safety has priority over all normal transitions.
            next_state = torch.where(
                unsafe_mask,
                torch.full_like(previous, int(VQRFsmState.SAFE_RECOVERY)),
                next_state,
            )
            active = ~unsafe_mask
    
            four_stand = previous == int(VQRFsmState.FOUR_STAND)
            transition_pos = previous == int(VQRFsmState.TRANSITION_POS)
            yaw_pos = previous == int(VQRFsmState.YAW_POS)
            transition_neg = previous == int(VQRFsmState.TRANSITION_NEG)
            yaw_neg = previous == int(VQRFsmState.YAW_NEG)
            return_to_four = previous == int(VQRFsmState.RETURN_TO_4)
            safe_recovery = previous == int(VQRFsmState.SAFE_RECOVERY)
    
            enter_pos = active & four_stand & (yaw > self.yaw_enter)
            enter_neg = active & four_stand & (yaw < -self.yaw_enter)
            next_state = torch.where(
                enter_pos,
                torch.full_like(previous, int(VQRFsmState.TRANSITION_POS)),
                next_state,
            )
            next_state = torch.where(
                enter_neg,
                torch.full_like(previous, int(VQRFsmState.TRANSITION_NEG)),
                next_state,
            )
    
            # A command exit takes priority over a temporary loss of pose readiness.
            abort_pos = active & (
                (transition_pos & (yaw < self.yaw_exit))
                | (yaw_pos & (yaw < self.yaw_exit))
            )
            abort_neg = active & (
                (transition_neg & (yaw > -self.yaw_exit))
                | (yaw_neg & (yaw > -self.yaw_exit))
            )
            next_state = torch.where(
                abort_pos | abort_neg,
                torch.full_like(previous, int(VQRFsmState.RETURN_TO_4)),
                next_state,
            )
    
            invalid_yaw_pose = active & ~(abort_pos | abort_neg) & (
                (yaw_pos & ~positive_ready) | (yaw_neg & ~negative_ready)
            )
            invalid_time = torch.where(
    @configclass
    class VQRWheelFlatEnvFSMCfg(VQRWheelFlatEnvCfg):
        """Flat-VQR-Wheel-Yaw with the FSM, 22-term reward set, and watchdog."""
    
        commands: VQRWheelFSMCommandsCfg = VQRWheelFSMCommandsCfg()
        observations: VQRWheelFSMObservationsCfg = VQRWheelFSMObservationsCfg()
        rewards: VQRWheelFSMRewardsCfg = VQRWheelFSMRewardsCfg()
        terminations: VQRWheelFSMTerminationsCfg = VQRWheelFSMTerminationsCfg()
        curriculum: VQRWheelFSMCurriculumCfg = VQRWheelFSMCurriculumCfg()
    ✓ • 0ms

## Activity

    $ ls -la logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-28_10-26-03 | head -50; find logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-28_10-26-03 -maxdepth 1 -type f -printf '%f\n'
    total 9756
    drwxrwxr-x  4 robotics robotics    4096 Sep 28 10:26 .
    drwxrwxr-x 20 robotics robotics    4096 Sep 28 10:26 ..
    -rw-rw-r--  1 robotics robotics   34148 Sep 28 10:26 events.out.tfevents.1790565968.robotics-Precision-7920-Tower.22093.0
    drwxrwxr-x  2 robotics robotics    4096 Sep 28 10:26 git
    -rw-rw-r--  1 robotics robotics 4965783 Sep 28 10:26 model_0.pt
    -rw-rw-r--  1 robotics robotics 4965783 Sep 28 10:26 model_3.pt
    drwxrwxr-x  2 robotics robotics    4096 Sep 28 10:26 params
    events.out.tfevents.1790565968.robotics-Precision-7920-Tower.22093.0
    model_3.pt
    model_0.pt
    ✓ • 0ms

## Activity

    $ /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python - <<'PY'
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        path='logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-28_10-26-03/events.out.tfevents.1790565968.robotics-Precision-7920-Tower.22093.0'
        a=EventAccumulator(path); a.Reload()
        for key in sorted(a.Tags()['scalars']):
         if any(s in key for s in ('Termination','transition/duration','transition/support_ready','transition/pose_ready','state_fraction/2','state_fraction/4','Reward/fsm_failure','Reward/transition_progress','reward/transition_progress','Reward/transition_low_base_height')):
          print(key,[(v.step,round(v.value,5)) for v in a.Scalars(key)])
        PY
    Curriculum/task_levels/state_fraction/2 [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]
    Curriculum/task_levels/state_fraction/4 [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]
    Curriculum/task_levels/transition/duration_mean [(0, 0.1275), (1, 0.41572), (2, 0.62601), (3, 0.779)]
    Curriculum/task_levels/transition/pose_ready_rate [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.00047)]
    Curriculum/task_levels/transition/support_ready_rate [(0, 0.01294), (1, 0.03785), (2, 0.04382), (3, 0.05615)]
    Episode_Reward/fsm_failure [(0, -1.375), (1, -3.0), (2, -3.0), (3, -3.0)]
    Episode_Reward/transition_low_base_height [(0, -0.04405), (1, -0.0853), (2, -0.09369), (3, -0.09469)]
    Episode_Reward/transition_progress [(0, 0.04944), (1, 0.08961), (2, 0.13131), (3, 0.14398)]
    Episode_Termination/base_height_failure [(0, 0.02474), (1, 0.2487), (2, 0.42253), (3, 0.53451)]
    Episode_Termination/fsm_return_timeout [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]
    Episode_Termination/fsm_transition_timeout [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]
    Episode_Termination/swing_contact_timeout [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]
    Episode_Termination/terrain_out_of_bounds [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]
    Episode_Termination/tilt_failure [(0, 0.0), (1, 0.00651), (2, 0.07031), (3, 0.09896)]
    Episode_Termination/time_out [(0, 0.02083), (1, 0.02474), (2, 0.01562), (3, 0.00326)]
    Episode_Termination/torso_contact [(0, 0.02474), (1, 0.25521), (2, 0.49284), (3, 0.63346)]
    ✓ • 248ms

## Activity

    $ /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python - <<'PY'
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        path='logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-28_10-26-03/events.out.tfevents.1790565968.robotics-Precision-7920-Tower.22093.0'
        a=EventAccumulator(path);a.Reload()
        for key in sorted(a.Tags()['scalars']):
         if any(s in key for s in ('Episode/length','Episode/rew','Episode_Reward','Curriculum/task_levels/pos/','Curriculum/task_levels/neg/','Curriculum/task_levels/phase','Curriculum/task_levels/transition/torso_contact')):
          values=a.Scalars(key)
          print(key, round(values[-1].value,5))
        PY
    Curriculum/task_levels/neg/drift 0.0
    Curriculum/task_levels/neg/drift_10s 0.0
    Curriculum/task_levels/neg/drift_10s_samples 0.0
    Curriculum/task_levels/neg/episodes 19.83333
    Curriculum/task_levels/neg/fail_lift_rate 1.0
    Curriculum/task_levels/neg/fail_support_rate 1.0
    Curriculum/task_levels/neg/mean_lift_progress 0.0
    Curriculum/task_levels/neg/mean_support_score 0.0
    Curriculum/task_levels/neg/score 0.0
    Curriculum/task_levels/neg/support_FR_rate 0.0
    Curriculum/task_levels/neg/support_HL_rate 0.0
    Curriculum/task_levels/neg/support_loss_max_dwell_s 0.0
    Curriculum/task_levels/neg/swing_contact_rate 0.0
    Curriculum/task_levels/neg/tracking_ratio 0.0
    Curriculum/task_levels/neg/transition_success 0.0
    Curriculum/task_levels/neg/yaw_mae 0.0
    Curriculum/task_levels/phase 0.0
    Curriculum/task_levels/phase_advanced 0.0
    Curriculum/task_levels/pos/drift 0.0
    Curriculum/task_levels/pos/drift_10s 0.0
    Curriculum/task_levels/pos/drift_10s_samples 0.0
    Curriculum/task_levels/pos/episodes 14.58333
    Curriculum/task_levels/pos/fail_lift_rate 1.0
    Curriculum/task_levels/pos/fail_support_rate 1.0
    Curriculum/task_levels/pos/mean_lift_progress 0.0
    Curriculum/task_levels/pos/mean_support_score 0.0
    Curriculum/task_levels/pos/score 0.0
    Curriculum/task_levels/pos/support_FL_rate 0.0
    Curriculum/task_levels/pos/support_HR_rate 0.0
    Curriculum/task_levels/pos/support_loss_max_dwell_s 0.0
    Curriculum/task_levels/pos/swing_contact_rate 0.0
    Curriculum/task_levels/pos/tracking_ratio 0.0
    Curriculum/task_levels/pos/transition_success 0.0
    Curriculum/task_levels/pos/yaw_mae 0.0
    Curriculum/task_levels/transition/torso_contact_rate 0.0
    Episode_Reward/action_rate -0.04345
    Episode_Reward/balance 0.04339
    Episode_Reward/base_height 0.0
    Episode_Reward/com_inside_segment 0.0
    Episode_Reward/com_support 0.0
    Episode_Reward/downward_low_base_velocity -0.01342
    Episode_Reward/four_stand_ready_bonus 0.0
    Episode_Reward/four_stand_stability 0.04512
    Episode_Reward/fsm_failure -3.0
    Episode_Reward/fsm_gated_tracking 0.0
    Episode_Reward/joint_limits -0.00025
    Episode_Reward/joint_velocity -0.04573
    Episode_Reward/lateral_slip -0.02127
    Episode_Reward/lift_clearance 0.0
    Episode_Reward/lifted_wheel_spin 0.0
    Episode_Reward/low_base_height -0.00019
    Episode_Reward/planar_velocity -0.01553
    Episode_Reward/return_to_four_landing 0.0
    Episode_Reward/rolling_slip 0.0
    Episode_Reward/safe_recovery_entry 0.0
    Episode_Reward/spin_center_drift 0.0
    Episode_Reward/support_span_band -0.06572
    Episode_Reward/torque -0.08429
    Episode_Reward/transition_low_base_height -0.09469
    Episode_Reward/transition_progress 0.14398
    Episode_Reward/undesired_contact -0.00171
    ✓ • 259ms

## Activity

    $ rg -n 'noise_model.*reset|noise_model.*_bias|NoiseModelWithAdditiveBias' /home/robotics/datld5/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py | head -90; sed -n '115,175p' /home/robotics/datld5/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py
            self._obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] | None = None
    
        def __str__(self) -> str:
            """Returns: A string representation for the observation manager."""
            msg = f"<ObservationManager> contains {len(self._group_obs_term_names)} groups.\n"
    
            # add info for each group
            for group_name, group_dim in self._group_obs_dim.items():
                # create table for term information
                table = PrettyTable()
                table.title = f"Active Observation Terms in Group: '{group_name}'"
                if self._group_obs_concatenate[group_name]:
                    table.title += f" (shape: {group_dim})"
                table.field_names = ["Index", "Name", "Shape"]
                # set alignment of table columns
                table.align["Name"] = "l"
                # add info for each term
                obs_terms = zip(
                    self._group_obs_term_names[group_name],
                    self._group_obs_term_dim[group_name],
                )
                for index, (name, dims) in enumerate(obs_terms):
                    # resolve inputs to simplify prints
                    tab_dims = tuple(dims)
                    # add row
                    table.add_row([index, name, tab_dims])
                # convert table to string
                msg += table.get_string()
                msg += "\n"
    
            return msg
    
        def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
            """Returns the active terms as iterable sequence of tuples.
    
            The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.
    
            Args:
                env_idx: The specific environment to pull the active terms from.
    
            Returns:
                The active terms.
            """
            terms = []
    
            if self._obs_buffer is None:
                self.compute()
            obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] = self._obs_buffer
    
            for group_name, _ in self._group_obs_dim.items():
                if not self.group_obs_concatenate[group_name]:
                    for name, term in obs_buffer[group_name].items():
                        terms.append((group_name + "-" + name, term[env_idx].cpu().tolist()))
                    continue
    
                idx = 0
                concat_dim = self._group_obs_concatenate_dim[group_name]
                # handle cases where concat dim is positive, account for the batch dimension
                if concat_dim > 0:
                    concat_dim -= 1
                # add info for each term
    ✓ • 0ms

## Activity

    $ rg -n 'def reset\(|_group_obs_class_term_cfgs|noise_model' /home/robotics/datld5/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py | head -100
    301:    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
    303:        for group_name, group_cfg in self._group_obs_class_term_cfgs.items():
    471:        self._group_obs_class_term_cfgs: dict[str, list[ObservationTermCfg]] = dict()
    508:            self._group_obs_class_term_cfgs[group_name] = list()
    625:                    noise_model_cls = term_cfg.noise.class_type
    626:                    if not issubclass(noise_model_cls, noise.NoiseModel):
    629:                            f" is not a subclass of 'NoiseModel'. Received: '{type(noise_model_cls)}'."
    632:                    term_cfg.noise.func = noise_model_cls(
    652:                    self._group_obs_class_term_cfgs[group_name].append(term_cfg)
    ✓ • 0ms

## Activity

    $ sed -n '295,323p' /home/robotics/datld5/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py; sed -n '640,660p' /home/robotics/datld5/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py; rg -n 'observation_manager.reset' /home/robotics/datld5/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py /home/robotics/datld5/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py
            return formatted_data
    
        """
        Operations.
        """
    
        def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
            # call all terms that are classes
            for group_name, group_cfg in self._group_obs_class_term_cfgs.items():
                for term_cfg in group_cfg:
                    term_cfg.func.reset(env_ids=env_ids)
                # reset terms with history
                for term_name in self._group_obs_term_names[group_name]:
                    if term_name in self._group_obs_term_history_buffer[group_name]:
                        self._group_obs_term_history_buffer[group_name][term_name].reset(batch_ids=env_ids)
            # call all modifiers that are classes
            for mod in self._group_obs_class_instances:
                mod.reset(env_ids=env_ids)
    
            # nothing to log here
            return {}
    
        def compute(self, update_history: bool = False) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
            """Compute the observations per group for all groups.
    
            The method computes the observations for all the groups handled by the observation manager.
            Please check the :meth:`compute_group` on the processing of observations per group.
    
            Args:
                            max_len=term_cfg.history_length, batch_size=self._env.num_envs, device=self._env.device
                        )
                        old_dims = list(obs_dims)
                        old_dims.insert(1, term_cfg.history_length)
                        obs_dims = tuple(old_dims)
                        if term_cfg.flatten_history_dim:
                            obs_dims = (obs_dims[0], np.prod(obs_dims[1:]))
    
                    self._group_obs_term_dim[group_name].append(obs_dims[1:])
    
                    # add term in a separate list if term is a class
                    if isinstance(term_cfg.func, ManagerTermBase):
                        self._group_obs_class_term_cfgs[group_name].append(term_cfg)
                        # call reset (in-case above call to get obs dims changed the state)
                        term_cfg.func.reset()
                # add history buffers for each group
                self._group_obs_term_history_buffer[group_name] = group_entry_history_buffer
    /home/robotics/datld5/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:369:        info = self.observation_manager.reset(env_ids)
    /home/robotics/datld5/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py:575:        info = self.observation_manager.reset(env_ids)
    ✓ • 0ms

## Activity

    $ sed -n '580,645p' /home/robotics/datld5/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py; rg -n 'term_cfg.noise|group_cfg.*noise|noise.*reset' /home/robotics/datld5/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py
                        for mod_cfg in term_cfg.modifiers:
                            # check if class modifier and initialize with observation size when adding
                            if isinstance(mod_cfg, modifiers.ModifierCfg):
                                # to list of modifiers
                                if inspect.isclass(mod_cfg.func):
                                    if not issubclass(mod_cfg.func, modifiers.ModifierBase):
                                        raise TypeError(
                                            f"Modifier function '{mod_cfg.func}' for observation term '{term_name}'"
                                            f" is not a subclass of 'ModifierBase'. Received: '{type(mod_cfg.func)}'."
                                        )
                                    mod_cfg.func = mod_cfg.func(cfg=mod_cfg, data_dim=obs_dims, device=self._env.device)
    
                                    # add to list of class modifiers
                                    self._group_obs_class_instances.append(mod_cfg.func)
                            else:
                                raise TypeError(
                                    f"Modifier configuration '{mod_cfg}' of observation term '{term_name}' is not of"
                                    f" required type ModifierCfg, Received: '{type(mod_cfg)}'"
                                )
    
                            # check if function is callable
                            if not callable(mod_cfg.func):
                                raise AttributeError(
                                    f"Modifier '{mod_cfg}' of observation term '{term_name}' is not callable."
                                    f" Received: {mod_cfg.func}"
                                )
    
                            # check if term's arguments are matched by params
                            term_params = list(mod_cfg.params.keys())
                            args = inspect.signature(mod_cfg.func).parameters
                            args_with_defaults = [arg for arg in args if args[arg].default is not inspect.Parameter.empty]
                            args_without_defaults = [arg for arg in args if args[arg].default is inspect.Parameter.empty]
                            args = args_without_defaults + args_with_defaults
                            # ignore first two arguments for env and env_ids
                            # Think: Check for cases when kwargs are set inside the function?
                            if len(args) > 1:
                                if set(args[1:]) != set(term_params + args_with_defaults):
                                    raise ValueError(
                                        f"Modifier '{mod_cfg}' of observation term '{term_name}' expects"
                                        f" mandatory parameters: {args_without_defaults[1:]}"
                                        f" and optional parameters: {args_with_defaults}, but received: {term_params}."
                                    )
    
                    # prepare noise model classes
                    if term_cfg.noise is not None and isinstance(term_cfg.noise, noise.NoiseModelCfg):
                        noise_model_cls = term_cfg.noise.class_type
                        if not issubclass(noise_model_cls, noise.NoiseModel):
                            raise TypeError(
                                f"Class type for observation term '{term_name}' NoiseModelCfg"
                                f" is not a subclass of 'NoiseModel'. Received: '{type(noise_model_cls)}'."
                            )
                        # initialize func to be the noise model class instance
                        term_cfg.noise.func = noise_model_cls(
                            term_cfg.noise, num_envs=self._env.num_envs, device=self._env.device
                        )
                        self._group_obs_class_instances.append(term_cfg.noise.func)
    
                    # create history buffers and calculate history term dimensions
                    if term_cfg.history_length > 0:
                        group_entry_history_buffer[term_name] = CircularBuffer(
                            max_len=term_cfg.history_length, batch_size=self._env.num_envs, device=self._env.device
                        )
                        old_dims = list(obs_dims)
                        old_dims.insert(1, term_cfg.history_length)
                        obs_dims = tuple(old_dims)
                        if term_cfg.flatten_history_dim:
    400:            if isinstance(term_cfg.noise, noise.NoiseCfg):
    401:                obs = term_cfg.noise.func(obs, term_cfg.noise)
    402:            elif isinstance(term_cfg.noise, noise.NoiseModelCfg) and term_cfg.noise.func is not None:
    403:                obs = term_cfg.noise.func(obs)
    548:                    term_cfg.noise = None
    624:                if term_cfg.noise is not None and isinstance(term_cfg.noise, noise.NoiseModelCfg):
    625:                    noise_model_cls = term_cfg.noise.class_type
    632:                    term_cfg.noise.func = noise_model_cls(
    633:                        term_cfg.noise, num_envs=self._env.num_envs, device=self._env.device
    635:                    self._group_obs_class_instances.append(term_cfg.noise.func)
    ✓ • 0ms

## Activity

    $ git status --short; git diff --check
     M scripts/reinforcement_learning/rsl_rl/train.py
     M source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py
     M source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py
     M source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py
     M tests/test_yaw_curriculum.py
     M tests/test_yaw_fsm.py
     M tests/test_yaw_fsm_systemic.py
     M tests/test_yaw_fsm_training_contract.py
     M tests/test_yaw_transition_safety.py
    ✓ • 0ms

## Activity

    $ rg -n '^class TransitionHeightProgress|^class TransitionProgress|^def fsm_failure_cost|^def safe_recovery_entry|^def _fsm_attempt_telemetry|^def yaw_fsm_unsafe_components|^class FSMUnsafeWithGrace|^def fsm_transition_timeout|^    transition_low_base_height =|^    transition_progress =|^    fsm_failure =|^    torso_contact =|^    base_height_failure =|^    tilt_failure =|^        enter_pos =|^        enter_neg =' source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/observations.py source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:368:    transition_low_base_height = RewTerm(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:578:    transition_progress = RewTerm(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:608:    fsm_failure = RewTerm(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:703:    torso_contact = DoneTerm(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:717:    torso_contact = DoneTerm(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:731:    base_height_failure = DoneTerm(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:745:    tilt_failure = DoneTerm(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py:384:        enter_pos = active & four_stand & (yaw > self.yaw_enter)
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py:385:        enter_neg = active & four_stand & (yaw < -self.yaw_enter)
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/observations.py:205:    torso_contact = wheel_contact(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/observations.py:276:def yaw_fsm_unsafe_components(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/observations.py:292:    torso_contact = wheel_contact(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:306:class TransitionHeightProgress(ManagerTermBase):
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:728:def _fsm_attempt_telemetry(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:832:    torso_contact = _yaw_wheel_contacts(env, torso_cfg, contact_threshold)[:, 0]
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:1185:class TransitionProgress(ManagerTermBase):
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:1293:def fsm_failure_cost(env: ManagerBasedRLEnv) -> torch.Tensor:
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:1336:def safe_recovery_entry(
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py:62:class FSMUnsafeWithGrace(ManagerTermBase):
    source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py:177:def fsm_transition_timeout(
    ✓ • 0ms

## Activity

    $ rg -n '^def test_watchdog|^def test_curriculum_counts|^def test_reacquisition|^def test_transition_height|^def test_lift_credit|^def test_stalled|^def test_lift_progress' tests/test_yaw_fsm_systemic.py tests/test_yaw_transition_safety.py
    tests/test_yaw_transition_safety.py:29:def test_transition_height_progress_is_bounded_monotone_and_reversible():
    tests/test_yaw_transition_safety.py:123:def test_lift_credit_uses_selected_support_diagonal_and_keeps_exploration():
    tests/test_yaw_fsm_systemic.py:78:def test_stalled_transition_has_no_pose_income_and_handoff_does_not_drop(clearance):
    tests/test_yaw_fsm_systemic.py:96:def test_lift_progress_cannot_be_refarmed_after_lowering_or_yaw_chatter():
    tests/test_yaw_fsm_systemic.py:192:def test_watchdog_survives_chatter_pauses_in_yaw_and_resets_only_at_maneuver_end():
    tests/test_yaw_fsm_systemic.py:241:def test_curriculum_counts_failed_attempts_even_with_no_yaw_or_a_prior_success(monkeypatch, phase):
    tests/test_yaw_fsm_systemic.py:298:def test_reacquisition_timeout_revokes_provisional_success_and_counts_chatter_once(monkeypatch):
    ✓ • 0ms

## Assistant

## Validation result

**Do not start the next long training run yet.** The focused CPU suite passed (**61 tests**), and a short headless Isaac Sim run completed (**64 environments, 4 iterations, 6,144 steps**). I made **no code changes** during this validation.

| Item | Status | Evidence |
|---|---|---|
| 1. Encoder bias reset | **PASS** | The [FSM noise config](/home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:849) samples bias with `operation="abs"`. Isaac Lab [resamples it on reset](/home/robotics/datld5/IsaacLab/source/isaaclab/isaaclab/utils/noise/noise_model.py:161). Across 2,000 resets of 256 environments × 16 joints, bias stayed within ±0.100000002 rad; mean was 0.000019 rad. Resetting half the environments left the other half unchanged. |
| 2. Early-failure incentive | **PARTIAL** | Controlled returns satisfy the required ordering for POS and NEG; numbers below. The short simulator run produced **zero YAW steps**, so it could not verify a successful physical trajectory. The one-time failure cost is in [rewards.py](/home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:1293). |
| 3. Acquisition topology | **FAIL** | Stationary reward and chatter tests pass, and [progress uses per-diagonal maxima](/home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:1185). But with **zero support load**, full lift and improved CoM geometry still paid **+5.9 episode reward**: lift +1.5, CoM line +2.4, segment +2.0. That violates the requested support constraint. |
| 4. Height gradient | **PASS** | The [height term](/home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:306) gave zero on entry and while stationary. After weight 4 and `dt=0.02`, a 1 cm descent cost −0.160 at 0.49 m, −0.231 at 0.45 m, −0.320 at 0.40 m, and −0.391 at 0.36 m. Recovery refunded at most the prior loss; each down→recover cycle netted 0 before discount. |
| 5. Stable FOUR entry | **FAIL — unresolved** | [FOUR→TRANSITION](/home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py:384) checks the yaw threshold but does **not** check `four_stand_ready` or a continuous dwell. |
| 6. Watchdogs and accounting | **PASS (CPU)** | [Transition time](/home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py:187) survives TRANSITION↔YAW chatter; the [watchdog](/home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py:177) eventually fires. [Attempt accounting](/home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py:728) retains failures in the denominator and revokes brief YAW success after terminal reacquisition failure. POS/NEG tests pass symmetrically. |
| 7. Unsafe telemetry | **PARTIAL** | The [unsafe predicate](/home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/observations.py:276) separates torso, height and tilt internally. The configured [`torso_contact` termination](/home/robotics/tuanpm48/vqr/rl_training/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py:717) logs their **union**, so torso-only failures are not directly distinguishable in the rollout log. |

### Discounted return comparison

The CPU counterfactuals started from the same safe state, used the configured reward functions and weights, `dt=0.02`, and PPO `γ=0.99`. Baseline penalties were zero in these controlled states.

| Trajectory | POS | NEG |
|---|---:|---:|
| Safe lift at 0.20 s, YAW from 0.30 s | **+22.835** | **+22.835** |
| Safe acquisition for 2 s, partial lift at 1 s | **+1.452** | **+1.452** |
| Unsafe termination at 0.18 s | **−55.365** | **−55.365** |

Thus **success > continued acquisition > early failure** in the reward model. These are controlled returns, not matched physical simulator rollouts.

In the [short simulator run](/home/robotics/tuanpm48/vqr/rl_training/logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-28_10-26-03/events.out.tfevents.1790565968.robotics-Precision-7920-Tower.22093.0), the final logged low-height failure metric was **53.45%**, tilt **9.90%**, and their union under the misleading `torso_contact` label **63.35%**. Low height dominates the separately measured causes. YAW occupancy was zero; transition support-ready rate was **5.62%** and pose-ready rate **0.047%**.

**Blockers before long training:** remove the +5.9 credit available after support collapse, add the requested FOUR readiness dwell, and obtain a successful physical transition to validate the return ordering in simulation. Unsafe logging also needs a torso-only cause if all three causes must be compared directly.
