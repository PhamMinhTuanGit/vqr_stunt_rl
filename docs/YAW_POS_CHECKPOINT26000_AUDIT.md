# Audit checkpoint 26000 và dense differential rolling

Đã audit **trước khi thay shaping reward**. Checkpoint được cố định là
`logs/rsl_rl/vqr_wheel_yaw_flat_pos/2026-10-02_09-05-01/model_26000.pt`,
SHA256 `69e03a43c7bf1312f8c1e70ede497095f619014164845ab46499494b007727f8`.
Đây là checkpoint mới nhất của run chính khi bắt đầu audit. Stage clearance
3, yaw 5: clearance 0.20 m, giới hạn 1 rad/s, DR 1.0.

Rollout deterministic seed 42, một env, chuỗi **0 → 1 → 0**, mỗi pha 4 s,
600 steps ở 50 Hz. Tắt observation corruption và interval disturbances;
startup/reset randomization vẫn dùng seed cố định. Không episode reset.
Mọi telemetry là snapshot tại reward evaluation, trước auto-reset.

## Geometry và convention

Target vẫn là `dot(yaw_command * (normal × (wheel_center - CoM)), tangent)`,
với `tangent = normalize(actual_axle × normal)`; measured là vận tốc **link
center** project vào cùng tangent. CoM là mass-weighted toàn robot.

- Tính lại độc lập từ vector world-frame trong CSV: sai số target tối đa
  FL `2.89e-8`, HR `1.52e-8 m/s`; measured tối đa `8.85e-8 m/s`.
- Sai số MAE so với finite difference vị trí center: FL 0.0078, HR 0.0051 m/s.
  Finite difference 50 Hz lấy trung bình chuyển động giữa bước, khác vận tốc
  physics tức thời, đặc biệt trong các transient đổi command.
- URDF dùng trục joint `(0, -1, 0)` và cylinder radius 0.091 m, tâm tại link
  origin, trục cylinder song song local Y. Runtime đối chứng xác nhận
  `(omega_wheel - omega_shank) · axle_+Y = -qdot`: sai số tối đa FL/HR dưới
  `1.15e-5 rad/s`.
- Tests đảo axle/joint convention, đổi heading/translation và wheel phase
  giữ nguyên reward; target có thể không bằng nhau khi lever arm khác nhau.

**Geometry/sign đúng; không cần đảo dấu hay hard-code tốc độ motor.**

## Hành vi deterministic

Các giá trị dưới đây là trung bình pha quay, bỏ 0.5 s đầu.
Ratio là trung bình `measured / target` từng step; neutral đánh dấu ratio
undefined bằng `ratio_valid=0`, không diễn giải số 0 như tỷ lệ đạt target.

| Đại lượng | FL | HR |
| --- | ---: | ---: |
| Target ground rolling | -0.24785 m/s | +0.18191 m/s |
| Measured ground rolling | -0.11013 m/s | +0.27123 m/s |
| Measured / target | 0.4503 | 1.4924 |
| Motor qdot | +1.3411 rad/s | -4.4825 rad/s |
| Motor velocity reference | +1.9424 rad/s | -4.7072 rad/s |
| `R abs(qdot) / abs(target)` | 0.4976 | 2.2391 |
| Contact fraction | 100% | 100% |
| Mean rolling residual | -0.00366 m/s | -0.02206 m/s |
| Mean motor mechanical power | 0.5669 W | 3.6972 W |

Ít nhất một motor dưới 50% activity trong **70.86%** mẫu quay đã settled.
Certificate vi sai chỉ đạt **29.14%**; heading rate trung bình **0.8662 rad/s**,
heading MAE **0.1666 rad/s**. CoM XY speed **0.12816 m/s**, thành phần theo
heading-X **+0.12302 m/s**. Robot vẫn dùng HR nhiều hơn FL và dịch chuyển
trong lúc quay, dù contact đủ tốt. Tốc độ motor là activity, không tự chứng
minh lực gây chuyển động; CSV ghi thêm torque và power để kiểm tra.

## Neutral speed và anchor drift

Số `~0.063 m/s`, `~8.3 cm` đến từ snapshot training iterations
26192–26291, trên 4096 envs với sampled actions, observation noise và DR,
không phải đánh giá policy mean. Neutral speed của logger gồm **toàn bộ**
pha neutral, kể cả transient landing; drift chỉ tính khi đã chốt anchor.

| Đối chứng cùng checkpoint, seed 42 | Neutral đầu: speed / drift | Neutral cuối: speed / drift |
| --- | ---: | ---: |
| Deterministic, toàn pha | 0.0416 m/s / 2.24 cm | 0.0174 m/s / 1.38 cm |
| Deterministic, bỏ 1 s đầu pha | 0.0124 m/s / 2.42 cm | 0.0043 m/s / 1.44 cm |
| Gaussian action noise, toàn pha | 0.0621 m/s / 3.42 cm | 0.0447 m/s / 0.74 cm |
| Gaussian action noise, bỏ 1 s đầu pha | 0.0299 m/s / 3.60 cm | 0.0346 m/s / 0.77 cm |
| Zero wheel actions trong neutral, bỏ 1 s đầu | 0.0081 m/s / 1.89 cm | 0.0077 m/s / 2.24 cm |

Speed trong bảng là norm XY của **torso CoM** ở world-frame; drift chỉ lấy
các mẫu đã anchored. CoM toàn robot cũng đã ghi riêng. Body-frame XY trong
training logger có thể khác world XY khi thân nghiêng/đang có vận tốc Z.
Đây là một seed, không phải kiểm định tổng thể. Đối chứng Gaussian thêm
checkpoint action std vào policy mean, không thêm observation noise hoặc
interval DR. Nó kiểm tra ảnh hưởng của exploration, không tái tạo toàn bộ
phân bố training.

Các nguyên nhân và giới hạn đã xác định:

1. **Exploration gây vận tốc dư:** checkpoint wheel action std lần lượt
   FL/FR/HL/HR `0.2325/0.3702/0.3729/0.3302`; action scale 5 tương đương
   reference noise `1.16/1.85/1.86/1.65 rad/s`. Đối chứng thêm action noise
   làm neutral cuối settled speed tăng từ 0.0043 lên 0.0346 m/s, vượt 0.03.
   Trong neutral đầu toàn pha, 0.0621 m/s gần số training đã quan sát.
2. **Policy vẫn ra lệnh bánh ở neutral:** neutral đầu settled mean reference
   FL/FR/HL/HR `-1.677/-1.170/+3.054/-0.275 rad/s`. Motor thực tế phần lớn
   bị hạn chế bởi contact và chuyển động toàn robot; torque vẫn tồn tại.
   Rolling residual khi settled dưới khoảng 0.001 m/s: chuyển động nhẹ lúc
   này chủ yếu là rolling/pose motion, không phải bánh trượt mạnh. Zero wheel
   actions cải thiện neutral đầu nhưng làm neutral cuối kém hơn, nên không
   thể quy toàn bộ drift cho wheel commands hoặc dùng zero actions như fix.
3. **Anchor chốt khi vẫn đang landing:** đủ bốn contact 0.2 s là điều kiện
   acquisition, không đòi hỏi speed thấp. Baseline chốt anchor ở 0.64 s và
   8.36 s với torso XY speed **0.1173** và **0.0926 m/s**. Quán tính/chuyển
   động sau chốt tạo drift ngay cả khi policy về sau đứng khá yên.
4. **Anchor không tự trôi:** trong từng pha neutral, anchor movement đúng
   **0 m**. Không reacquire sau contact mất ngắn; reset/active mới xóa anchor.
   Isaac Lab `root_pos_w` là link origin, `root_lin_vel_w` là torso CoM velocity.
   Chênh XY velocity hai điểm trung bình toàn rollout 0.0094 m/s, nhưng drift
   đối chiếu đúng `root_link_pos_w - anchor`; alias không tạo lỗi drift 8.3 cm.
5. **Policy thiếu phản hồi XY/anchor:** actor 55D chỉ có angular velocity,
   gravity, yaw command, joint position/velocity, last actions; không có
   linear velocity hoặc anchor error/history. Critic có linear velocity nhưng
   không cấp nó cho actor. Khả năng dừng/chống drift dựa trên proprioception;
   actor không thể trực tiếp quan sát phải đi bao xa để trở lại anchor.
6. **Drift training chưa được tái tạo đủ bởi ba rollout này:** action noise
   làm speed tăng nhưng không nhất quán làm drift tăng trong mọi pha. Full
   observation bias, DR, lịch command/landing và thời lượng episode dài hơn
   có thể đóng góp; chưa có ablation tách định lượng từng yếu tố đó. Tại drift
   8.3 cm, Gaussian position score chỉ còn ~0.064; tại 13 cm còn ~0.0012.
   Mốc đã vượt xa vẫn bị giữ, nên đứng yên ở vị trí mới không xóa lỗi cũ.

Không thay reward/observations/action policy neutral trong thay đổi này.
Ngưỡng speed 0.03 m/s, drift 0.05 m và pass fraction 90% giữ nguyên.

## Shaping mới

`differential_rolling_score` dùng `h(e)=sqrt(1+e²)-1`, với
`scale=max(0.05,0.25*abs(target))`:

```text
tracking_cost = max_FL_HR h((measured-target)/scale)
slip_cost = max_FL_HR h(rolling_residual/scale)
participation = min_FL_HR clamp(motor_fraction/0.5, 0, 1)
score = both_valid_and_contact / (1 + tracking_cost + slip_cost + 2*(1-participation))
```

Cost tăng gần tuyến tính khi sai số lớn; một reciprocal chung tránh tích
các kernel gần 0. Motor dừng vẫn có partial shaping để học giảm lỗi; perfect
tracking/slip với một motor dừng chỉ đạt tối đa 1/3. Score 1 đòi cả hai motor
đạt activity. Bánh có normalized tracking error lớn nhất quyết định tracking;
bánh còn lại không bù được lỗi của nó. Motor ít hoạt động nhất quyết định
participation. Contact/geometry hợp lệ là gate topology.

Ở error 0.30 m/s, scale 0.05, zero slip và đủ activity: score mới **0.1644**,
tracking slope **-0.533 m⁻¹**, so với Gaussian **2.32e-16**. Test kiểm tra slope
ở errors 0.3, 0.6, 1, 3 m/s, cả khi một motor inactive.

Rescore chính trajectory baseline: mean score settled **0.05586 → 0.33346**.
Đây là thay shaping trên cùng dữ liệu; policy không thay đổi và certificate
vẫn **29.14%**. Weight giữ **2.0**. Toàn bộ boolean certificate, scale residual,
activity threshold, settle time và curriculum thresholds giữ nguyên.

## Kiểm chứng và dữ liệu

- 296 CPU tests qua, gồm 128 trạng thái kiểm tra certificate cũ và mới giống
  hệt, sign/phase invariance, worse-wheel tracking, motor participation và
  slope ở lỗi lớn. Loại hai file FSM-Staged có 23 lỗi tồn tại từ HEAD đã
  được xác minh trong lần triển khai trước.
- Smoke 64 envs, 5 PPO updates resume policy/optimizer/curriculum từ 26000.
  Run `2026-10-02_14-41-13_pos_dense_differential_smoke`, checkpoint 26004;
  tensors hữu hạn, optimizer state còn nguyên cấu trúc, stage 3/5 giữ nguyên,
  elapsed 14565 → 14685. Đây là kiểm tra tích hợp, không chứng minh hội tụ.
- Sửa recorder: `RewardManager._step_reward` đã là weighted reward **rate**.
  CSV cũ chia `dt` lần nữa làm số reward lớn 50 lần. Baseline đã normalize;
  file gốc được giữ tên `telemetry_original_units.csv`. Không ảnh hưởng action,
  geometry, vận tốc hoặc dữ liệu TensorBoard training.

Dữ liệu tại `outputs/yaw_pos_checkpoint26000_audit/`:

- `baseline/20261002_133423_942271/telemetry.csv`
- `neutral_wheel_stop/20261002_141320_794590/telemetry.csv`
- `action_noise/20261002_142808_728582/telemetry.csv`
- `rollout_summary.json`, `rollout_audit.png`, `rescore.json`, `rescore.csv`,
  `reward_shaping.png`, `smoke_summary.json`, `dense_smoke.log`.

Recorder ghi cả bốn bánh: target/measured, ratio validity, motor speed/target,
torque/power, axial/relative angular velocity, contacts/force, raw world
positions/velocities/axles/tangents, torso link/CoM và whole-body CoM XY,
ground heading rate và anchor/error. Baseline trước khi thêm kênh quaternion
và relative axial omega; đối chứng có đủ các kênh này.

```bash
python scripts/reinforcement_learning/rsl_rl/record_yaw_pos_video.py \
  --checkpoint logs/rsl_rl/vqr_wheel_yaw_flat_pos/2026-10-02_09-05-01/model_26000.pt \
  --headless --device cuda:0 --trace-only --seed 42 \
  --initial-neutral-seconds 4 --positive-seconds 4 --neutral-seconds 4 --cycles 1

# Đối chứng: thêm --sample-actions hoặc --neutral-wheel-stop.
python scripts/reinforcement_learning/rsl_rl/analyze_yaw_pos_rollout.py TRACE_DIR [COUNTERFACTUAL_DIR ...]

# Resume đầy đủ từ checkpoint đã audit để dùng reward mới trong một run mới.
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Flat-VQR-Wheel-Yaw-POS --headless --device cuda:0 \
  --resume --load_run 2026-10-02_09-05-01 --checkpoint model_26000.pt \
  --run_name pos_dense_differential --seed 42
```
