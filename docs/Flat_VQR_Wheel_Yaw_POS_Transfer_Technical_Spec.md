# Technical Specification: `Flat-VQR-Wheel-Yaw-POS-Transfer`

## 1. Mục tiêu

Tạo một task mới tên:

```text
Flat-VQR-Wheel-Yaw-POS-Transfer
```

Task mới phải **không sửa, không overwrite và không thay đổi behavior** của:

```text
Flat-VQR-Wheel-Yaw
Flat-VQR-Wheel-Yaw-POS
```

Mục tiêu của task mới là:

1. Kế thừa **con đường học tư thế hai bánh + quay yaw** đã thành công ở `Flat-VQR-Wheel-Yaw`.
2. Chỉ chuyên biệt hóa command theo hướng POS.
3. Giữ semantics deadband bắt buộc:
   - `yaw_cmd <= 0.10 rad/s` → robot trở về và duy trì `FOUR_STAND`.
   - `yaw_cmd > 0.10 rad/s` → robot chuyển sang tư thế hai bánh POS và tracking yaw dương.
4. Không ép `HipX = 0`.
5. Không dùng `support_x_rms`, support-line alignment hoặc các pose-shaping mới trong phiên bản đầu tiên.
6. Dùng checkpoint thành công của `Flat-VQR-Wheel-Yaw` làm initialization để task mới đi theo cùng local optimum/learning path đã chứng minh được.

---

## 2. Bối cảnh kỹ thuật

`Flat-VQR-Wheel-Yaw` đã chứng minh robot có thể học một tư thế hai bánh khả thi và quay tốt. Vì vậy task mới không nên thiết kế lại pose từ đầu bằng các constraint joint-space hoặc geometry-space chưa được kiểm chứng.

`Flat-VQR-Wheel-Yaw-POS` hiện tại đã diverge khỏi learning landscape của task gốc do có các reward/curriculum khác. Task này phải được giữ nguyên để làm baseline đối chứng.

Thiết kế mới theo nguyên tắc:

```text
Successful base task
        |
        | inherit learning landscape
        v
POS-Transfer
        |
        +-- positive command only
        +-- deadband -> FOUR_STAND
```

---

## 3. Behavioral Contract

### 3.1 Mode definition

Định nghĩa hai mode từ command:

```python
DEADBAND = 0.10

neutral_mask = yaw_cmd <= DEADBAND
positive_mask = yaw_cmd > DEADBAND
```

Hai mask phải mutually exclusive và exhaustive đối với command distribution của task:

```python
assert not (neutral_mask & positive_mask).any()
assert (neutral_mask | positive_mask).all()
```

Task mới **không sample yaw âm**.

### 3.2 Neutral mode

Khi:

```text
yaw_cmd <= 0.10 rad/s
```

robot phải học behavior:

```text
FOUR_STAND
```

Mục tiêu neutral:

- bốn wheel/foot contact đúng;
- trở về pose đứng neutral/default hợp lý;
- yaw rate nhỏ;
- planar velocity nhỏ;
- không giữ incentive phải tiếp tục đứng hai bánh;
- không nhận positive two-wheel/yaw objective.

### 3.3 Positive mode

Khi:

```text
yaw_cmd > 0.10 rad/s
```

robot phải học behavior POS hai bánh bằng **learning landscape của `Flat-VQR-Wheel-Yaw`**.

Mục tiêu:

- dùng đúng support pair POS;
- lift đúng pair còn lại;
- giữ balance;
- tracking positive yaw command;
- giữ các contact/slip/regularization behavior đã giúp base task thành công.

Không thêm target:

```text
HipX -> 0
support_x_rms -> 0
support-line alignment -> 1
```

trong phiên bản đầu tiên.

---

## 4. Nguyên tắc reward

Reward chia thành ba nhóm.

### 4.1 Positive-specific rewards

Các reward định nghĩa kỹ năng hai bánh/yaw phải lấy trực tiếp từ `Flat-VQR-Wheel-Yaw`, gồm:

- cùng hàm reward;
- cùng parameter;
- cùng weight;
- cùng normalization/gating nội bộ nếu có.

Các reward này chỉ active khi:

```python
positive_mask == True
```

Không copy từ `Flat-VQR-Wheel-Yaw-POS` nếu implementation POS đã drift khỏi base task.

### 4.2 Neutral-specific rewards

Neutral rewards chỉ active khi:

```python
neutral_mask == True
```

Các objective tối thiểu:

- four-contact/four-stand;
- neutral/default pose;
- low absolute yaw rate;
- low planar velocity.

Ưu tiên reuse implementation đã được test trong repo nếu có.

Không để `lift_clearance`, two-wheel support bonus hoặc yaw-tracking positive tiếp tục thưởng khi neutral mode đang active.

### 4.3 Global rewards / penalties

Các term không mang semantics mode và cần thiết cho safety/regularization có thể tiếp tục active ở cả hai mode, ví dụ tùy theo base task:

- unsafe contact penalty;
- joint-limit penalty;
- action-rate penalty;
- torque/energy regularization;
- termination-related penalties;
- các regularizer global khác.

Không được mask toàn bộ reward graph chỉ vì positive/neutral objectives mutually exclusive.

Agent phải audit từng reward và phân loại rõ:

```text
POSITIVE_SPECIFIC
NEUTRAL_SPECIFIC
GLOBAL
```

trước khi code.

---

## 5. Command distribution

Task mới không có yaw âm.

Khởi đầu nên giữ distribution gần với learning setup của base task nhất có thể, nhưng thêm neutral samples để học return-to-four-stand.

Target ban đầu:

```text
positive samples: khoảng 70-80%
neutral samples:  khoảng 20-30%
negative samples: 0%
```

Không hard-code tỷ lệ mới nếu repo đã có sampler phù hợp; ưu tiên reuse mechanism hiện tại và chỉ thay ranges/probabilities cần thiết.

### Positive command

Positive magnitude bắt đầu từ stage dễ tương ứng với base task.

Không resume curriculum state từ checkpoint base.

### Neutral command

Neutral command phải nằm trong:

```text
0 <= yaw_cmd <= 0.10
```

Nếu implementation dùng exact zero cho neutral thì phải ghi rõ.

---

## 6. Curriculum

### 6.1 Reset curriculum state

Khi initialize từ checkpoint `Flat-VQR-Wheel-Yaw`:

```text
load policy weights
load critic weights
DO NOT load optimizer state
DO NOT restore old curriculum level
```

Task mới bắt đầu curriculum riêng từ stage dễ.

### 6.2 Positive-yaw progression

Positive-yaw curriculum phải bám sát progression của `Flat-VQR-Wheel-Yaw`.

Không tự ý thay:

- yaw threshold;
- support threshold;
- lift threshold;
- balance threshold;
- stage ordering;

trừ khi audit code base chứng minh cần thiết.

### 6.3 Neutral preservation

Curriculum không được promote chỉ vì positive yaw tốt trong khi neutral FOUR_STAND bị hỏng.

Phải log riêng:

```text
positive success
neutral success
```

và đảm bảo neutral behavior không collapse trong quá trình tăng yaw range.

Không invent threshold mới nếu repo đã có logic phù hợp; reuse threshold hiện có hoặc báo cáo rõ mọi threshold mới.

---

## 7. Transfer learning

Task mới phải hỗ trợ load checkpoint thành công của:

```text
Flat-VQR-Wheel-Yaw
```

### Yêu cầu compatibility

Actor và critic phải checkpoint-compatible.

Agent phải xác nhận:

```text
actor observation dimension
critic observation dimension
action dimension
action ordering
observation ordering
```

Nếu dimension/order khác, **không được silently load partial weights**.

Phải fail loudly hoặc có migration được giải thích/test rõ ràng.

### Load semantics

Ưu tiên:

```python
load_actor = True
load_critic = True
load_optimizer = False
reset_curriculum = True
```

Action noise/std có thể load cùng policy nếu nó là một phần learned policy state của runner; agent phải audit runner implementation và báo cáo behavior thực tế.

---

## 8. Cấu trúc code đề xuất

Không overwrite file POS hiện tại.

Ưu tiên inheritance thay vì copy toàn bộ config.

Ví dụ:

```text
config/wheeled/vqr_wheel/
    yaw_env_cfg.py
    yaw_env_pos_cfg.py
    yaw_env_pos_transfer_cfg.py   <-- NEW

agents/
    ... base runner cfg ...
    ... pos runner cfg ...
    ... pos-transfer runner cfg ... <-- NEW
```

Concept:

```python
class VQRWheelYawPosTransferEnvCfg(VQRWheelYawEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # only intentional specialization:
        # - positive-only command distribution
        # - neutral deadband behavior
        # - mode-gated reward objectives
        # - independent curriculum state
```

Tên/path thực tế phải theo structure hiện có trong repo.

---

## 9. Registration

Đăng ký task mới độc lập:

```text
Flat-VQR-Wheel-Yaw-POS-Transfer
```

Không thay registration hiện tại của:

```text
Flat-VQR-Wheel-Yaw
Flat-VQR-Wheel-Yaw-POS
```

Task mới phải có:

- EnvCfg riêng;
- RunnerCfg riêng nếu cần;
- log experiment namespace riêng;
- task ID riêng.

---

## 10. Diagnostics bắt buộc

Task mới phải log ít nhất:

### Mode / command

```text
mode_positive_fraction
mode_neutral_fraction
mean_abs_yaw_cmd
```

### Positive mode

```text
positive_yaw_score
positive_yaw_rate_error
positive_tracking_ratio
support_score
lift_progress
balance_score
```

### Neutral mode

```text
neutral_four_contact_rate
neutral_pose_error
neutral_abs_yaw_rate
neutral_planar_speed
```

### Safety

```text
torso_contact
terrain_out_of_bounds
time_out
```

Geometry diagnostics như:

```text
FL_x_from_CoM
HR_x_from_CoM
support_x_rms
support-line alignment
HipX
```

có thể log để phân tích, nhưng **không dùng làm reward/curriculum gate trong phiên bản đầu tiên**.

---

## 11. Test plan

### 11.1 Regression

Xác nhận task cũ không thay đổi:

```text
Flat-VQR-Wheel-Yaw
Flat-VQR-Wheel-Yaw-POS
```

Ít nhất:

- config snapshot/diff;
- registration;
- reward weights;
- command behavior.

### 11.2 Deadband boundary

Test tối thiểu:

```text
yaw_cmd = 0.00 -> neutral only
yaw_cmd = 0.05 -> neutral only
yaw_cmd = 0.10 -> neutral only
yaw_cmd = 0.100001 -> positive only
yaw_cmd = 0.25 -> positive only
```

### 11.3 Reward masking

Với cùng simulator state:

- neutral mode: positive-specific reward contribution phải bằng 0;
- positive mode: neutral-specific reward contribution phải bằng 0;
- global penalties vẫn hoạt động ở cả hai mode.

### 11.4 Base reward equivalence

Trong positive mode, với cùng state/action/command magnitude:

```text
POS-Transfer positive reward terms
==
Flat-VQR-Wheel-Yaw corresponding reward terms
```

trong tolerance số học hợp lý.

### 11.5 Checkpoint compatibility

Test:

```text
load successful Flat-VQR-Wheel-Yaw checkpoint
-> no shape mismatch
-> actor forward pass OK
-> critic forward pass OK
-> one environment rollout OK
```

### 11.6 PPO smoke test

Chạy ít nhất một PPO update:

```text
32 envs
headless
```

Xác nhận:

- policy weights được update;
- optimizer mới được tạo;
- curriculum bắt đầu từ stage mong muốn;
- metrics positive/neutral xuất hiện;
- checkpoint save thành công.

---

## 12. Training protocol

### Phase A — Transfer smoke test

```text
source checkpoint: successful Flat-VQR-Wheel-Yaw
optimizer: reset
curriculum: reset
envs: 32
updates: 1-10
```

Mục tiêu: verify mechanics, không đánh giá quality.

### Phase B — Short adaptation

```text
updates: ~500-1000
```

Đánh giá:

- neutral FOUR_STAND có học được không;
- positive two-wheel stance của base task có được giữ không;
- yaw score có collapse không;
- support/lift/balance có collapse không.

### Phase C — Long training

Chỉ bắt đầu khi Phase B cho thấy:

```text
positive behavior retained
AND
neutral behavior learned
AND
no safety regression
```

Sau đó mở curriculum positive yaw theo base task.

---

## 13. Acceptance criteria

Task mới được coi là implementation đúng khi:

1. Có task ID mới `Flat-VQR-Wheel-Yaw-POS-Transfer`.
2. Hai task cũ không bị thay đổi behavior/config ngoài các shared bugfix thực sự cần thiết và được báo cáo rõ.
3. Deadband contract đúng chính xác:
   ```text
   yaw_cmd <= 0.10 -> FOUR_STAND objective
   yaw_cmd > 0.10  -> POS two-wheel yaw objective
   ```
4. Positive reward landscape khớp base task.
5. Neutral và positive objectives không đồng thời thưởng các behavior mâu thuẫn.
6. Global safety/regularization vẫn hoạt động.
7. Checkpoint base load được mà không mismatch.
8. Optimizer và curriculum được reset.
9. PPO smoke test pass.
10. Agent cung cấp config diff rõ ràng:
    ```text
    Base task -> POS-Transfer
    ```
    và giải thích từng intentional difference.

---

## 14. Non-goals

Trong iteration đầu của task mới, **không**:

- ép `HipX = 0`;
- thêm `support_x_rms` reward;
- thêm support-line alignment reward;
- thay đổi base-height target chỉ dựa trên pose giả định;
- thay đổi support-span target chỉ dựa trên pose giả định;
- tune hàng loạt reward weights;
- overwrite `Flat-VQR-Wheel-Yaw-POS`;
- resume optimizer/curriculum cũ một cách vô thức.

Các thay đổi geometry-specific chỉ được cân nhắc sau khi POS-Transfer đã tái tạo được learning behavior của base task và có rollout/diagnostic chứng minh một thiếu sót cụ thể.

---

## 15. Deliverables

Agent phải trả về:

1. Danh sách file added/modified.
2. Task registration mới.
3. EnvCfg/RunnerCfg mới.
4. Bảng diff:
   ```text
   Flat-VQR-Wheel-Yaw vs Flat-VQR-Wheel-Yaw-POS-Transfer
   ```
5. Phân loại toàn bộ reward:
   ```text
   POSITIVE_SPECIFIC / NEUTRAL_SPECIFIC / GLOBAL
   ```
6. Kết quả unit tests.
7. Kết quả checkpoint-load smoke test.
8. Kết quả 1 PPO update.
9. Lệnh train chính thức từ checkpoint base.
10. Xác nhận rõ:
    ```text
    Flat-VQR-Wheel-Yaw-POS was not overwritten.
    ```
