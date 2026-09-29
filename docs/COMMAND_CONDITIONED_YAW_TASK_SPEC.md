# Technical Specification — Command-Conditioned `Flat-VQR-Wheel-Yaw`

## 1. Mục tiêu

Refactor task `Flat-VQR-Wheel-Yaw` trong repo `vqr_stunt_rl` thành một **command-conditioned policy** điều khiển trực tiếp bởi scalar `yaw_rate_cmd`, không sử dụng FSM trong đường điều khiển.

Policy phải học ba hành vi theo command:

```text
yaw_cmd > +0.1
    -> POSE_POS
    -> support: FL + HR
    -> lift:    FR + HL
    -> quay yaw dương

|yaw_cmd| <= 0.1
    -> FOUR_STAND
    -> cả 4 bánh contact
    -> trở về default/nominal 4-leg pose an toàn
    -> yaw rate -> 0
    -> planar velocity -> 0

yaw_cmd < -0.1
    -> POSE_NEG
    -> support: FR + HL
    -> lift:    FL + HR
    -> quay yaw âm
```

Task phải cho phép command thay đổi **ngay trong cùng episode**, ví dụ:

```text
+0.6 -> 0.0 -> -0.5 -> 0.0 -> +0.7
```

Policy phải tự học các transition:

```text
FOUR -> POS
POS  -> FOUR
FOUR -> NEG
NEG  -> FOUR
```

Không thêm FSM state, transition timer, pose-ready state machine, mode ID hoặc one-hot mode vào policy observation.

---

# 2. Phạm vi

## 2.1 Task chính

Task cần sửa:

```text
Flat-VQR-Wheel-Yaw
```

Registration hiện tại:

```text
source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/
config/wheeled/vqr_wheel/__init__.py
```

Task trỏ tới:

```text
config/wheeled/vqr_wheel/yaw_env_cfg.py:VQRWheelFlatEnvCfg
```

## 2.2 Không dùng

Không sử dụng logic từ:

```text
Flat-VQR-Wheel-Yaw-FSM
Flat-VQR-Wheel-Yaw-FSM-Staged
```

Có thể tái sử dụng utility/helper toán học từ FSM code nếu độc lập và sạch, nhưng task mới không được phụ thuộc vào:

```text
fsm_state
fsm_gates
support_diagonal state buffer
state_time
transition_time
pose_ready
four_stand_ready
SAFE_RECOVERY
RETURN_TO_4
TRANSITION_POS
TRANSITION_NEG
YAW_POS
YAW_NEG
```

---

# 3. Kiến trúc mong muốn

```text
           external joystick / training sampler
                        |
                        v
                 yaw_rate_cmd
                   scalar (N,1)
                        |
                 CommandManager
                /       |       \
               /        |        \
              v         v         v
           Actor      Critic     Rewards
              \         |         /
               \        |        /
                    actions
                       |
                       v
                     Robot
```

Actor và critic đều phải thấy cùng một `yaw_rate_cmd` mà reward đọc.

Không inject joystick trực tiếp vào observation nếu `CommandManager` vẫn giữ command khác.

**Single source of truth cho command phải là `CommandManager`.**

---

# 4. Hiện trạng cần giữ nguyên

Trong:

```text
source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/
velocity_yaw_env_cfg.py
```

Actor hiện đã có:

```python
yaw_rate_cmd = ObsTerm(
    func=mdp.generated_commands,
    params={"command_name": "yaw_rate_cmd"},
)
```

Critic cũng đã có cùng command observation.

## Yêu cầu

- Giữ nguyên scalar `yaw_rate_cmd` trong actor observation.
- Giữ nguyên scalar `yaw_rate_cmd` trong critic observation.
- Không thêm mode ID.
- Không thêm sign bit.
- Không thêm FSM state.
- Không thêm transition flag.

Network architecture hiện tại của PPO không cần thay đổi chỉ để hỗ trợ command conditioning.

---

# 5. Command contract

File:

```text
source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/commands.py
```

Hiện tại `YawRateCommand` có command shape:

```text
(num_envs, 1)
```

và:

```text
command[:, 0] = yaw_rate_cmd
```

Giữ nguyên contract này.

## 5.1 Config mới

Mở rộng `YawRateCommandCfg` tối thiểu:

```python
@configclass
class YawRateCommandCfg(CommandTermCfg):
    class_type: type = YawRateCommand
    asset_name: str = "robot"
    yaw_rate_range: tuple[float, float] = (-1.0, 1.0)

    deadband: float = 0.1
    neutral_probability: float = 0.30
```

Có thể thêm external-control API nếu cần cho joystick runtime:

```python
external_control: bool = False
```

hoặc state runtime tương đương trong `YawRateCommand`.

## 5.2 Training sampling

Không sample đơn thuần:

```python
uniform(-yaw_limit, +yaw_limit)
```

vì policy phải được train có chủ đích với command neutral.

Distribution mặc định:

```text
30%: yaw_cmd = 0
35%: yaw_cmd in [+deadband, +yaw_limit]
35%: yaw_cmd in [-yaw_limit, -deadband]
```

Không sample active command bên trong deadband.

Ví dụ:

```python
neutral_probability = 0.30
positive_probability = 0.35
negative_probability = 0.35
```

`resampling_time_range=(4.0, 6.0)` có thể giữ nguyên ban đầu.

## 5.3 Runtime external command

Cần có API sạch để runtime ghi joystick command vào command term.

Ví dụ contract mong muốn:

```python
command_term.set_external_command(yaw_cmd)
```

hoặc tương đương.

Yêu cầu:

1. Khi external control bật, `_resample_command()` không được overwrite command joystick.
2. External command phải được clamp vào `yaw_rate_range`.
3. Deadband phải được áp trước khi ghi command:

```python
if abs(raw_cmd) <= deadband:
    yaw_cmd = 0.0
else:
    yaw_cmd = scaled_cmd
```

4. Reward, actor và critic đều đọc cùng command qua:

```python
env.command_manager.get_command("yaw_rate_cmd")
```

Không cho phép tình huống:

```text
actor sees joystick command
reward sees sampled training command
```

---

# 6. Command masks dùng chung

File:

```text
mdp/rewards.py
```

Thêm helper dùng chung:

```python
def _yaw_command_masks(
    env: ManagerBasedRLEnv,
    command_name: str,
    deadband: float,
):
    cmd = env.command_manager.get_command(command_name)[:, 0]

    positive = cmd > deadband
    negative = cmd < -deadband
    neutral = ~(positive | negative)

    return cmd, positive, negative, neutral
```

Yêu cầu:

- Mọi reward command-conditioned phải dùng helper này hoặc logic tương đương duy nhất.
- Không copy-paste ba điều kiện ở nhiều nơi nếu có thể tránh.
- `neutral` phải bao gồm toàn bộ khoảng đóng:

```text
[-deadband, +deadband]
```

---

# 7. Diagonal mapping

File:

```text
config/wheeled/vqr_wheel/yaw_env_cfg.py
```

Hiện tại task hard-code một diagonal:

```python
SUPPORT_WHEEL_NAMES = ["FL_WHEEL", "HR_WHEEL"]
LIFTED_WHEEL_NAMES = ["FR_WHEEL", "HL_WHEEL"]
```

Thay bằng khai báo explicit cho cả hai chiều:

```python
POS_SUPPORT_WHEELS = ["FL_WHEEL", "HR_WHEEL"]
POS_LIFTED_WHEELS  = ["FR_WHEEL", "HL_WHEEL"]

NEG_SUPPORT_WHEELS = ["FR_WHEEL", "HL_WHEEL"]
NEG_LIFTED_WHEELS  = ["FL_WHEEL", "HR_WHEEL"]

ALL_WHEELS = [
    "FL_WHEEL",
    "FR_WHEEL",
    "HL_WHEEL",
    "HR_WHEEL",
]

YAW_DEADBAND = 0.1
```

Mapping là authoritative:

```text
POS:
    support = FL + HR
    lift    = FR + HL

NEG:
    support = FR + HL
    lift    = FL + HR

NEUTRAL:
    support = all 4 wheels
    lift    = none
```

---

# 8. Reward architecture

Mục tiêu là giữ reward structure dễ debug.

Chia reward thành hai nhóm:

## 8.1 Always-on safety/stability rewards

Các term này không được tắt chỉ vì command đổi mode:

```text
base_height
low_base_height
downward_low_base_velocity
balance
lateral_slip
rolling_slip
undesired_contact
joint_limits
action_rate
joint_velocity
torque
planar_velocity
```

Có thể tinh chỉnh weight sau, nhưng semantics phải luôn active.

Không để task lặp lại lỗi cũ: mất stability signal trong transition.

## 8.2 Command-conditioned task rewards

Các term phụ thuộc mode phải đọc trực tiếp `yaw_cmd`:

```text
com_support
support_span_band
lift_clearance
com_inside_segment
gated_yaw_tracking
lifted_wheel_spin
four_stand_pose
four_wheel_contact
```

---

# 9. Sửa `com_support`

Hiện tại baseline dùng fixed FL-HR geometry.

Task mới phải chọn geometry theo command.

Pseudo-contract:

```python
cmd, pos, neg, neutral = _yaw_command_masks(...)

pos_score = score_COM_to_FL_HR_support_line()
neg_score = score_COM_to_FR_HL_support_line()

reward = torch.where(
    pos,
    pos_score,
    torch.where(
        neg,
        neg_score,
        torch.zeros_like(pos_score),
    ),
)
```

Trong neutral:

```text
com_support = 0
```

Không ép CoM vào một diagonal khi robot phải đứng 4 bánh.

---

# 10. Sửa `support_span_band`

POS:

```text
measure FL-HR support span
```

NEG:

```text
measure FR-HL support span
```

NEUTRAL:

```text
reward/penalty của two-wheel support span phải bằng 0
```

Không penalize robot FOUR_STAND dựa trên diagonal span.

---

# 11. Sửa `lift_clearance`

POS:

```text
lift target = FR + HL
```

NEG:

```text
lift target = FL + HR
```

NEUTRAL:

```text
không có lifted pair
lift reward phải bằng 0
```

Pseudo-logic:

```python
pos_progress = _yaw_lift_progress(... POS_LIFTED_WHEELS ...)
neg_progress = _yaw_lift_progress(... NEG_LIFTED_WHEELS ...)

selected_progress = torch.where(
    pos.unsqueeze(1),
    pos_progress,
    neg_progress,
)

active = pos | neg
score = 2.0 * selected_progress.mean(dim=1) - 1.0
reward = score * active.float()
```

Không được reward lift khi `|cmd| <= deadband`.

---

# 12. Sửa `com_inside_segment`

POS:

```text
segment = FL -> HR
```

NEG:

```text
segment = FR -> HL
```

NEUTRAL:

```text
reward = 0
```

---

# 13. Sửa `gated_yaw_tracking`

Đây là term quan trọng nhất.

## 13.1 Active mode

Khi `cmd > deadband` hoặc `cmd < -deadband`:

```text
reward = selected_support_gate
       * clearance_weight
       * yaw_tracking(cmd)
```

Support gate phải chọn đúng diagonal theo sign command.

Yaw tracking:

```python
yaw_error = root_ang_vel_b[:, 2] - yaw_cmd
yaw_tracking = exp(-(yaw_error**2) / std**2)
```

## 13.2 Neutral/deadband mode

Khi:

```text
|cmd| <= deadband
```

reward phải khuyến khích:

```text
4-wheel contact
+ yaw_rate -> 0
```

Ví dụ:

```python
neutral_tracking = four_wheel_contact_score * torch.exp(
    -(root_ang_vel_b[:, 2] ** 2) / std**2
)
```

Final:

```python
reward = torch.where(
    neutral,
    neutral_tracking,
    active_tracking,
)
```

Quan trọng:

- baseline cũ không được tiếp tục yêu cầu FL+HR support trong neutral.
- neutral phải có semantics FOUR_STAND rõ ràng.

---

# 14. Reward mới: `yaw_deadband_four_stand_pose`

Thêm reward rõ ràng để policy học quay về nominal/default four-leg pose.

Mục tiêu:

```text
|yaw_cmd| <= deadband
    -> joint_pos -> default_joint_pos
```

Suggested implementation:

```python
def yaw_deadband_four_stand_pose(
    env: ManagerBasedRLEnv,
    command_name: str,
    deadband: float,
    asset_cfg: SceneEntityCfg,
    std: float = 0.25,
) -> torch.Tensor:
    _, _, _, neutral = _yaw_command_masks(
        env, command_name, deadband
    )

    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    q0 = robot.data.default_joint_pos[:, asset_cfg.joint_ids]

    error = torch.mean((q - q0).square(), dim=1)
    score = torch.exp(-error / std**2)

    return neutral.float() * score
```

Yêu cầu:

- chỉ áp dụng leg joints, không cần wheel angle.
- reward chỉ active trong neutral.
- default pose phải là safe four-leg pose của asset hiện tại.

---

# 15. Reward mới: `yaw_deadband_four_wheel_contact`

Mục tiêu:

```text
|yaw_cmd| <= deadband
    -> tất cả 4 bánh contact
```

Không nên chỉ dùng hard binary `all_four` vì policy cần dense gradient khi đang hạ chân.

Suggested shaping:

```python
contacts = wheel_contacts.float()  # shape (N,4)
partial = contacts.mean(dim=1)
all_four = contacts.prod(dim=1)

score = 0.2 * partial + 0.8 * all_four
reward = neutral.float() * score
```

Ý nghĩa:

```text
0/4 -> 0
1/4 -> small
2/4 -> small
3/4 -> moderate
4/4 -> full reward
```

Yêu cầu cuối cùng vẫn là 4/4.

---

# 16. Neutral yaw/velocity behavior

Khi neutral, policy phải được khuyến khích:

```text
yaw_rate -> 0
planar velocity -> 0
base stable/upright
4 wheel contact
default four-leg pose
```

Do đó:

- `planar_velocity` giữ always-on.
- `balance` giữ always-on.
- `base_height` giữ always-on.
- `gated_yaw_tracking` neutral branch track `yaw_rate=0`.
- `four_stand_pose` active trong neutral.
- `four_wheel_contact` active trong neutral.

---

# 17. `lifted_wheel_spin`

Hiện tại term dùng một lifted pair cố định.

Task mới phải:

```text
POS -> evaluate FR + HL
NEG -> evaluate FL + HR
NEUTRAL -> no lifted pair penalty OR use zero active mask
```

Không penalize arbitrary pair trong FOUR_STAND.

---

# 18. Curriculum

File:

```text
mdp/curriculums.py
```

Có thể giữ yaw ladder hiện tại:

```text
0.25
0.40
0.55
0.70
0.85
1.00
```

và clearance ladder:

```text
0.05
0.10
0.15
0.20
```

Nhưng curriculum metrics phải phân biệt active và neutral samples.

## 18.1 Không tính neutral vào active yaw metrics

Các accumulator dạng:

```text
_yaw_command_abs_sum
_yaw_rate_abs_error_sum
_yaw_tracking_metric_samples
_yaw_edge_*
_yaw_lift_min_progress_*
_yaw_support_score_*
```

nếu dùng để đánh giá two-wheel yaw skill thì chỉ được accumulate khi:

```python
active_mask = torch.abs(yaw_cmd) > deadband
```

Không để 30% neutral sample kéo `lift_progress` về 0 và làm curriculum stall.

## 18.2 Nên thêm neutral metrics

Khuyến nghị thêm telemetry riêng:

```text
neutral_four_contact_rate
neutral_pose_error
neutral_yaw_abs_rate
neutral_planar_speed
```

Các metric này không nhất thiết phải gate curriculum ngay ở version đầu, nhưng phải log để debug.

---

# 19. Reward count / training contract

Repo hiện có static test kiểm tra baseline `VQRWheelRewardsCfg` có 18 reward terms.

Khuyến nghị implementation đầu tiên:

**Giữ tổng reward count = 18 nếu có thể.**

Cách làm:

- thay implementation của các term cũ bằng command-conditioned semantics;
- tái sử dụng/đổi tên hai term ít quan trọng nếu cần để chứa:
  - `four_stand_pose`
  - `four_wheel_contact`

hoặc nếu cần tăng reward count thì phải cập nhật rõ:

```text
tests/test_yaw_fsm_training_contract.py
scripts/reinforcement_learning/rsl_rl/train.py
```

Không được để test hard-code cũ fail không rõ nguyên nhân.

Nếu đổi reward count, agent phải cập nhật test contract tương ứng và giải thích thay đổi.

---

# 20. Safety / termination

Giữ các safety termination hiện tại của `Flat-VQR-Wheel-Yaw` trừ khi có lý do cụ thể.

Không tạo FSM recovery.

Yêu cầu:

- torso contact termination vẫn hoạt động;
- minimum base height protection vẫn hoạt động;
- bad orientation/safety logic hiện dùng cho baseline phải được audit nhưng không thêm state machine;
- neutral không được miễn safety penalty.

Safety phải orthogonal với command mode.

---

# 21. Joystick integration

`play.py` hiện có đường `--keyboard` dành cho `base_velocity`; nó không phải interface đúng cho scalar `yaw_rate_cmd` của task này.

Cần tạo path riêng cho joystick/gamepad hoặc generic external yaw command.

Ví dụ:

```text
right stick X
    -> normalize [-1,1]
    -> deadband ±0.1
    -> scale max_yaw_rate
    -> YawRateCommand.set_external_command(...)
    -> CommandManager
    -> Actor/Critic/Reward
```

Không inject trực tiếp vào `env_cfg.observations.policy.yaw_rate_cmd`.

Command phải đi qua command term.

---

# 22. Training sequence requirement

Task phải được train với command changing within episode.

Ví dụ expected sequence:

```text
t=0s     +0.6   POS
t=5s      0.0   FOUR_STAND
t=10s    -0.5   NEG
t=15s     0.0   FOUR_STAND
```

Không reset env chỉ để đổi mode.

Mục tiêu quan trọng là policy tự học transition physical state giữa các command region.

---

# 23. Tests bắt buộc

Tạo CPU/unit tests cho command semantics.

## 23.1 Mask boundaries

Test chính xác:

```text
cmd = +0.1000 -> neutral
cmd = -0.1000 -> neutral
cmd = +0.1001 -> positive
cmd = -0.1001 -> negative
cmd = 0.0     -> neutral
```

## 23.2 Diagonal selection

Với `cmd > 0.1`:

```text
support = FL + HR
lift = FR + HL
```

Với `cmd < -0.1`:

```text
support = FR + HL
lift = FL + HR
```

## 23.3 Neutral behavior

Trong neutral:

```text
lift reward = 0
com_support two-wheel reward = 0
support-span two-wheel term = 0
four_stand_pose > 0 khi gần default pose
four_wheel_contact max khi 4/4 contact
yaw tracking max khi yaw_rate = 0
```

## 23.4 Wrong-pose rejection

Trong neutral:

- 2-wheel pose không được có reward cao hơn valid four-wheel pose.
- giữ lifted pair trên không không được là local optimum tốt hơn đặt đủ 4 bánh xuống.

## 23.5 Direction symmetry

Mirror state POS/NEG phải cho reward gần tương đương nếu physical state mirror tương ứng.

Test:

```text
POS mirrored to NEG
reward_pos ~= reward_neg
```

trong sai số số học hợp lý.

## 23.6 Command sequence test

Test ít nhất logical reward contract cho sequence:

```text
+0.5 -> 0 -> -0.5 -> 0 -> +0.5
```

Không cần FSM state để sequence này hợp lệ.

## 23.7 Observation test

Assert actor observation chứa đúng 1 scalar yaw command.

Assert critic observation chứa đúng cùng scalar command.

---

# 24. Smoke tests trong Isaac Lab

Sau unit tests, chạy smoke env với:

```text
num_envs = 1
```

Inject lần lượt command cố định:

```text
+0.5
0.0
-0.5
0.0
```

Log tối thiểu:

```text
yaw_cmd
root_yaw_rate
wheel contacts FL FR HL HR
wheel clearance FL FR HL HR
joint pose error to default
base height
roll/pitch
reward terms
```

Expected qualitative result sau training:

```text
+cmd -> POS diagonal
0    -> four-wheel safe pose
-cmd -> NEG diagonal
0    -> four-wheel safe pose
```

---

# 25. Telemetry cần thêm

Khuyến nghị log:

```text
mode_positive_fraction
mode_negative_fraction
mode_neutral_fraction

pos_support_score
neg_support_score

neutral_four_contact_rate
neutral_pose_error
neutral_abs_yaw_rate
neutral_planar_speed

positive_yaw_tracking_ratio
negative_yaw_tracking_ratio
```

Không cần mode state buffer; các fraction được suy ra trực tiếp từ command masks.

---

# 26. Reward trap audit

Agent phải audit các reward để tránh các exploit sau.

## 26.1 Neutral giữ hai bánh trên không

Không được tồn tại reward path mà:

```text
cmd=0
robot giữ two-wheel pose
```

lại có tổng reward cao hơn:

```text
cmd=0
robot về four-wheel pose
```

## 26.2 Một bánh contact

Four-wheel contact shaping có thể dense nhưng full reward phải yêu cầu 4/4.

## 26.3 Yaw reward khi sai support diagonal

Active yaw tracking không được trả full reward nếu support pair sai diagonal.

## 26.4 Lift mà mất support

Không được cho full positive lift reward nếu robot nhấc đúng swing wheels nhưng support pair collapse.

Nếu cần, lift shaping có thể được nhân một soft support-quality factor như baseline/FSM experiment trước đó.

## 26.5 Deadband chattering

Runtime command processing phải map toàn bộ `[-0.1,+0.1]` về exact zero để tránh policy đổi target liên tục do joystick noise.

Có thể thêm hysteresis ở deployment layer sau nếu hardware cần, nhưng training contract vẫn dùng deadband ±0.1.

---

# 27. Giữ code đơn giản

Ưu tiên helper thuần vectorized PyTorch.

Không tạo class/state machine mới nếu không bắt buộc.

Không tạo các buffer kiểu:

```text
current_mode
previous_mode
transition_mode
mode_timer
pose_ready_timer
```

trong training task version này.

Command hiện tại đã đủ để xác định mode tại mỗi step.

---

# 28. Backward compatibility

Task FSM cũ không được vô tình bị phá bởi refactor baseline helpers.

Nếu sửa helper đang được FSM dùng:

- giữ default params để FSM behavior cũ không đổi; hoặc
- tạo helper mới riêng cho command-conditioned baseline.

Ưu tiên tạo function mới có tên rõ ràng, ví dụ:

```text
yaw_command_com_support
yaw_command_lift_clearance
yaw_command_com_inside_support_segment
yaw_command_gated_tracking
yaw_deadband_four_stand_pose
yaw_deadband_four_wheel_contact
```

thay vì thay đổi semantics ngầm của function FSM đang phụ thuộc.

---

# 29. Training strategy

Train mới từ đầu.

Không resume trực tiếp checkpoint cũ của `Flat-VQR-Wheel-Yaw` vì checkpoint cũ học topology:

```text
support = FL + HR
lift = FR + HL
```

cho cả hai dấu command.

Task mới đổi semantics observation->behavior mapping nên checkpoint cũ có thể tạo bias mạnh và khó audit.

Recommended initial command curriculum:

```text
Stage 0:
    active yaw magnitude = 0.25
    neutral_probability = 0.30

Stage 1+:
    increase active yaw limit theo ladder hiện tại
```

Giữ neutral sampling ở mọi stage.

---

# 30. Acceptance criteria

Implementation được coi là hoàn thành khi thỏa tất cả điều kiện sau.

## Static/code

- `Flat-VQR-Wheel-Yaw` không phụ thuộc FSM state.
- Actor thấy scalar `yaw_rate_cmd`.
- Critic thấy scalar `yaw_rate_cmd`.
- Reward đọc cùng command từ `CommandManager`.
- POS/NEG diagonal mapping đúng.
- Neutral mapping về FOUR_STAND đúng.

## Unit tests

- command masks boundary pass.
- POS/NEG diagonal selection pass.
- neutral pose/contact reward pass.
- active lift/support reward neutral masking pass.
- POS/NEG mirror symmetry pass.
- test suite hiện có liên quan baseline/FSM không bị regression ngoài những contract được chủ động cập nhật.

## Runtime

Có thể inject external command:

```text
+0.5
0.0
-0.5
```

mà Actor/Critic/Reward đều đọc cùng giá trị.

## Training behavior

Sau training đủ lâu, playback phải cho thấy:

```text
cmd > +0.1
    -> robot chuyển sang positive diagonal và quay dương

|cmd| <= 0.1
    -> robot chủ động hạ lifted wheels
    -> trở về 4-wheel contact
    -> về nominal safe pose
    -> yaw rate giảm về gần 0

cmd < -0.1
    -> robot chuyển sang negative diagonal và quay âm
```

Quan trọng nhất:

```text
POS -> neutral -> FOUR_STAND
NEG -> neutral -> FOUR_STAND
```

phải xảy ra do policy, không do FSM hoặc scripted transition controller.

---

# 31. Deliverables từ agent

Agent phải trả về:

1. Danh sách file đã sửa.
2. Tóm tắt logic trước/sau.
3. Diff/implementation cho command sampler.
4. Diff/implementation cho command-conditioned rewards.
5. Deadband four-stand rewards.
6. External command API hoặc path rõ ràng để joystick runtime ghi `yaw_rate_cmd`.
7. Unit tests mới.
8. Kết quả test.
9. Nếu reward count/config contract đổi, liệt kê rõ test/config nào được cập nhật.
10. Không sửa FSM task trừ khi cần backward compatibility tối thiểu.

---

# 32. Ưu tiên implementation

Thực hiện theo thứ tự:

```text
1. commands.py
   - deadband config
   - neutral sampling
   - external command API

2. rewards.py
   - command mask helper
   - POS/NEG selection
   - neutral FOUR_STAND rewards

3. yaw_env_cfg.py
   - explicit POS/NEG/ALL wheel configs
   - wire reward terms

4. curriculums.py
   - active-only yaw/lift metrics
   - neutral diagnostics

5. tests
   - boundary
   - symmetry
   - neutral return behavior contract

6. play/deploy input path
   - joystick -> CommandManager -> policy
```

Không bắt đầu bằng sửa PPO architecture hoặc thêm FSM mới.

---

# 33. Summary contract for implementation agent

```text
INPUT:
    yaw_rate_cmd scalar

POLICY OBSERVATION:
    existing robot observations + yaw_rate_cmd

TARGET BY COMMAND:
    cmd > +0.1     -> POS two-wheel yaw
    cmd < -0.1     -> NEG two-wheel yaw
    |cmd| <= 0.1   -> safe FOUR_STAND

NO FSM.
NO MODE ONE-HOT.
NO SCRIPTED TRANSITION.

TRAINING:
    30% neutral
    35% positive
    35% negative
    resample within episode

NEUTRAL MUST EXPLICITLY REWARD:
    default four-leg pose
    4/4 wheel contact
    yaw_rate = 0
    low planar velocity
    stable base

ALL SAFETY REWARDS REMAIN ACTIVE IN ALL COMMAND REGIONS.
```
