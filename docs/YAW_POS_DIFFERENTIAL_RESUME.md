# Yaw-POS: quay vi sai, đứng yên và resume đến 3 rad/s

Chỉ áp dụng cho `Flat-VQR-Wheel-Yaw-POS`. Checkpoint nguồn là
`logs/rsl_rl/vqr_wheel_yaw_flat_pos/2026-10-01_18-28-44/model_25500.pt`.
Giữ actor/critic, optimizer, action std và iteration; observation vẫn là
55 chiều cho actor, 83 chiều cho critic, action 16 chiều.

## Reward và chứng nhận

Với mỗi bánh trụ FL/HR, tính trong ground-plane:

```text
r_i = wheel_center_xy - whole_body_com_xy
v_target_i = yaw_command * (ground_normal × r_i)
t_i = normalize(actual_wheel_axle × ground_normal)
target_i = dot(v_target_i, t_i)
measured_i = dot(wheel_link_center_velocity, t_i)
```

Axle lấy từ orientation thực tế của wheel body, local +Y. Đảo convention
axle làm cả hai projection đảo dấu, giữ nguyên sai số. Không dùng dấu joint
velocity làm target; độ lớn joint velocity chỉ đo mức hoạt động của motor.
Không yêu cầu hai bánh có cùng tốc độ khi lever arm khác nhau.

Sau audit checkpoint `model_26000.pt`, shaping vi sai dùng cost pseudo-Huber
`h(e) = sqrt(1 + e²) - 1` và một reciprocal chung:

```text
scale_i = max(0.05, 0.25 * abs(target_i))
tracking_cost = max_i h((measured_i - target_i) / scale_i)
slip_cost = max_i h(rolling_residual_i / scale_i)
participation = min_i clamp(R * abs(qdot_i) / (0.5 * abs(target_i)), 0, 1)
score = both_valid_and_contact / (1 + tracking_cost + slip_cost + 2 * (1 - participation))
```

Bánh có sai số normalized lớn nhất quyết định tracking; motor ít hoạt động
nhất quyết định participation. Motor dừng vẫn có partial credit cho tracking,
nhưng score tối đa chỉ 1/3 ngay cả khi được kéo lăn đúng target. Điều này giữ
độ dốc ở trạng thái thất bại; certification bên dưới không cho trạng thái đó
đạt chuẩn. Không tăng weight hoặc thay đổi các ngưỡng certification.
Chi tiết audit và đối chứng: [YAW_POS_CHECKPOINT26000_AUDIT.md](YAW_POS_CHECKPOINT26000_AUDIT.md).

| Reward mới | Trọng số | Cách chấm |
| --- | ---: | --- |
| `differential_rolling` | 2.0 | Điểm bánh kém hơn: tracking, rolling residual, hoạt động motor và contact |
| `neutral_velocity` | 3.0 | Vận tốc XY bằng 0, Gaussian scale 0.05 m/s, kể cả khi đang hạ chân |
| `neutral_position` | 2.0 | Giữ mốc XY, Gaussian scale 0.05 m |

Mốc đứng được ghi sau bốn bánh contact liên tục 0.2 s. Mốc không chạy theo
robot và không được ghi lại sau mất contact ngắn. Chuyển sang command quay
hoặc reset episode xóa mốc, kể cả playback đã tắt curriculum.

Chứng nhận quay bỏ 0.5 s đầu sau chuyển sang command dương. Trong ít nhất
90% mẫu, cả hai bánh phải contact, lăn đúng chiều theo target hình học, đạt
ít nhất 50% độ lớn target, và mỗi motor cũng đạt mức hoạt động tương ứng.
Rolling residual phải nằm trong scale `max(0.05 m/s, 0.25 * abs(target))`.
Target gần 0 hoặc hướng lăn suy biến không được chứng nhận.

Chứng nhận neutral đo từ lúc có mốc: ít nhất 90% mẫu phải contact đủ bốn bánh,
vận tốc XY ≤ 0.03 m/s và drift ≤ 0.05 m. Cửa sổ không có mẫu neutral đã neo
không được nâng mức. Episode chỉ có neutral vẫn đóng góp vào chứng nhận này.

## Curriculum và checkpoint

POS giữ các mức `0.25, 0.40, 0.55, 0.70, 0.85, 1.00`, nối thêm
`1.25, 1.50, 1.75, 2.00, 2.50, 3.00 rad/s`. DR của mức mới là 1.0;
tracking-ratio threshold 0.55 và edge threshold 0.45.

Giữ yêu cầu 2048 active episodes mỗi cửa sổ, 3 cửa sổ đạt liên tiếp và
6000 environment steps tại yaw stage. Cửa sổ phải đạt thêm hai chứng nhận
hành vi. Reward yaw vẫn dùng ground-heading rate.

Resume chấp nhận bảng cũ là tiền tố khớp hoàn toàn của bảng mới, giữ clearance
stage 3, yaw stage 5, tốc độ 1 rad/s, DR 1.0 và elapsed 2541 steps của
checkpoint nguồn. Các cửa sổ đạt của checkpoint chưa có chứng nhận hành vi
được đánh giá lại; stage và elapsed không reset. Checkpoint mới lưu cả các
bộ đếm cửa sổ chưa hoàn tất. Playback checkpoint cũ vẫn giới hạn ở 1 rad/s.

## Kiểm chứng ngày 2026-10-02

Smoke thực tế: 64 envs, seed 42, 5 PPO iterations, resume từ iteration 25500.
Run mới:
`logs/rsl_rl/vqr_wheel_yaw_flat_pos/2026-10-02_08-41-36_pos_differential_3rad_smoke`.
Checkpoint cuối `model_25504.pt` có model hữu hạn, optimizer state, stage 3/5,
elapsed 2661 steps và bảng 12 mức đến 3 rad/s. Log:
`outputs/yaw_pos_differential_3rad_smoke.log`.

288 test CPU đã qua (không tính hai file FSM-Staged có lỗi từ trước), kiểm tra geometry/sign invariance, wheel phase, pivot một bánh,
trượt, contact, anchor/reset, certification, bảng curriculum và resume.
Test với checkpoint thật xác nhận từng tensor actor/critic/std và toàn bộ
optimizer state được giữ chính xác trước update.

Test toàn repo có 23 lỗi FSM-Staged do thiếu symbol. Đã chạy hai file test này
trên bản HEAD riêng trong `/tmp` và xác nhận cùng 23 lỗi có sẵn từ trước.
Log kiểm tra các test còn lại: `outputs/yaw_pos_differential_cpu_tests.log`.

Hai rollout deterministic cùng seed và điều kiện, `0 → 1 → 0 rad/s`, mỗi pha
4 s, dùng clearance của checkpoint là 0.20 m. Mỗi rollout ghi 600 mẫu hữu hạn,
không reset episode. CSV và kết quả tổng hợp nằm trong
`outputs/yaw_pos_differential_validation/summary.json`.

| Chỉ số | Policy nguồn | Sau 5 iterations smoke |
| --- | ---: | ---: |
| Mean heading error, pha quay sau 0.5 s | 0.142 rad/s | 0.175 rad/s |
| Tỷ lệ mẫu quay vi sai đạt chuẩn | 46.6% | 9.7% |
| Mean vận tốc XY sau hạ chân cuối | 0.0717 m/s | 0.0482 m/s |
| Mean drift từ mốc sau hạ chân cuối | 0.1301 m | 0.0647 m |

Đây là kiểm tra tích hợp và telemetry, chưa phải kết quả hội tụ. Vận tốc trôi
giảm nhưng quay vi sai chưa đạt chứng nhận; cần tiếp tục training và đánh giá
ở từng stage. Policy chưa được chứng nhận ở 3 rad/s.

## Lệnh tiếp tục training

Chạy trong môi trường Isaac Lab hiện có. Lệnh dưới đây tiếp tục từ checkpoint
nguồn đã chọn, tạo run mới và giữ mức curriculum hiện tại. `max_iterations`
là số iterations bổ sung; mặc định runner POS là 30000.

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Flat-VQR-Wheel-Yaw-POS --headless --device cuda:0 \
  --resume --load_run 2026-10-01_18-28-44 --checkpoint model_25500.pt \
  --run_name pos_differential_3rad --seed 42
```

Để kiểm tra checkpoint tiếp theo bằng cùng chuỗi lệnh:

```bash
python scripts/reinforcement_learning/rsl_rl/record_yaw_pos_video.py \
  --checkpoint /absolute/path/to/model.pt --headless --device cuda:0 \
  --trace-only --initial-neutral-seconds 4 --positive-seconds 4 \
  --neutral-seconds 4 --cycles 1 --seed 42
```

Tốc độ quay mặc định lấy từ stage đã lưu. Khi checkpoint được chứng nhận đến
stage cuối, dùng thêm `--positive-yaw 3.0` để đánh giá riêng ở 3 rad/s.
