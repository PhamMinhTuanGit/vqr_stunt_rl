# Training POS với USD mới và yaw tối đa 3 rad/s

Cấu hình áp dụng cho `Flat-VQR-Wheel-Yaw-POS`. Train từ đầu; `play.py` giữ nguyên.

## USD và pose

- Asset: `deep_robotics_model/VQRWheel/VQRWheel_usd/VQRWheel.usd`, gồm các layer USD được tham chiếu.
- Khối lượng đọc từ asset: **36,58756 kg**; bán kính bánh khoảng **0,091 m**.
- Target chiều cao đứng: **0,49 m**. Reference chân được giải bằng FK từ joint frame, collider và giới hạn trong USD; reset, action offset, observation q-relative và reward đứng dùng cùng reference.
- Reference hiện tại: FL/FR `[0, -0.500385, 0.995400]`, HL/HR `[0, -0.497567, 0.995630]` rad, theo thứ tự HipX/HipY/Knee.
- Target chiều cao yaw: **0,455 m**; khoảng cách tối thiểu hai tâm bánh trụ theo hướng ngang thân: **0,20 m**. Đây là giảm chiều cao có chủ đích khi vào pose yaw. Phạt hạ thân quá mức bắt đầu ở **0,44 m** khi yaw và **0,46 m** khi đứng; safety floor vẫn **0,35 m**.
- Tâm collider bánh lệch khoảng 0,0303 m theo trục bánh so với gốc link. Reward/clearance/telemetry tính tại tâm collider, có xét bề rộng lốp và độ nghiêng trục khi tính clearance.

## Curriculum

Giữ nguyên thứ tự lift trước, yaw sau; không sửa sampler, thời gian command, điều kiện lên stage hoặc lịch DR cũ. Source POS đã có các mốc mở rộng đến 3 rad/s; cấu hình này sử dụng toàn bộ dãy đó:

```text
Lift: 0.05 → 0.10 → 0.15 → 0.20 m
Yaw:  0.25 → 0.40 → 0.55 → 0.70 → 0.85 → 1.00
      → 1.25 → 1.50 → 1.75 → 2.00 → 2.50 → 3.00 rad/s
```

DR ở các mốc trên 1 rad/s giữ scale 1.0. Command giữ 30% neutral chính xác bằng 0 và 70% uniform trong `(0.1, yaw_limit)`, resample 4–6 s.

Các gate giữ nguyên: support 0.85, lift 0.80, balance 0.75, yaw 0.65; 2048 episode/window, success 0.85, ba window liên tiếp; tối thiểu 1000 step/lift stage và 6000 step/yaw stage. Certification differential/neutral giữ ngưỡng 0.90. Điểm yaw dùng để lên stage vẫn dùng std 0.20; tolerance sàn cho certification rolling vẫn 0.05 m/s.

Default `max_iterations` của POS tăng từ 30000 lên **60000** để dành thời gian học các mốc cuối. Đây là số PPO iteration, không phải bảo đảm curriculum đã đạt 3 rad/s khi hết run.

## Action và reward

- Vẫn 16 action: 12 target position chân và 4 target velocity bánh, theo FL/FR/HL/HR.
- Target chân bị giới hạn bằng hard joint limits của USD; tốc độ target tối đa 2 rad/s, gia tốc target tối đa 10 rad/s², có giảm tốc trước giới hạn khớp.
- Target bánh tối đa 58.9 rad/s, thay đổi tối đa 20 rad/s².
- Bộ giới hạn chạy mỗi physics tick **5 ms**; policy vẫn **20 ms**. Khi reset, target chân bắt đầu từ joint position đang đo, vận tốc target chân và target velocity bánh về 0.
- Actor vẫn 55 observation, critic vẫn 83. Kênh actor `[39:55]` đổi thành **target đã áp dụng, chuẩn hóa theo action offset/scale**, thay cho raw action. Action-rate penalty cũng dùng thay đổi target đã áp dụng.
- Weight yaw tracking 6, differential rolling 4, base-height tracking 6, height deficit -2. Yaw tracking cho PPO dùng std `max(0.04, 0.20*abs(command))`; differential shaping dùng tolerance `max(0.015, 0.25*abs(target_speed))` m/s.
- Differential shaping kiểm tra cả vận tốc lăn tại tâm bánh và motor có dấu, có cộng chuyển động quay của shank. Với USD hiện tại, chiều joint dương của cả bốn bánh là **-Y của wheel body**; không suy dấu từ tên FL/HR.
- Phạt vận tốc/trôi CoM khi yaw sau thời gian ổn định 0.5 s, anchor không chạy theo robot. HipX có vùng tự do ±0.50 rad để pose yaw có thể đạt hình học mới.
- Gaussian std ban đầu: chân 0.5, bánh 0.3; entropy coef giữ 0.0025. Noise observation giảm để học điều khiển yaw chậm; đây là giá trị khởi đầu, cần hiệu chỉnh theo log sensor thực.
- Cấu hình actuator/PD và delay hiện tại vẫn được dùng. Giới hạn tốc độ target là giới hạn tín hiệu điều khiển; chuyển động khớp thực và torque còn phụ thuộc actuator và tiếp xúc.

## Chạy training

```bash
cd /home/robotics/tuanpm48/vqr/rl_training
conda activate env_isaaclab_tuanpm48_vqr
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Flat-VQR-Wheel-Yaw-POS \
  --num_envs 4096 --headless \
  --max_iterations 60000 --run_name new_usd_yaw3_v2
```

Không truyền `--resume`, `--pos_adaptation_checkpoint` hoặc checkpoint cũ. Run ghi `params/env.yaml`, `params/agent.yaml`, `params/yaw_pos_contract.json`; checkpoint chứa `infos.yaw_pos_contract` và `infos.yaw_curriculum`. Training từ chối resume nếu USD hoặc quy ước observation/action không khớp.

Theo dõi yaw stage/yaw limit thực tế cùng heading error, differential pass, neutral hold pass, motor error, CoM drift và base height; không chỉ nhìn tổng reward. Kiểm tra các tốc độ thấp 0.15–0.25 rad/s và các mốc cuối sau khi policy đã được học ở stage tương ứng.

## Lưu ý khi dùng play.py nguyên bản

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Flat-VQR-Wheel-Yaw-POS \
  --checkpoint /duong/dan/run_moi/model_N.pt \
  --num_envs 1 --keyboard
```

1. Dùng checkpoint **của run mới này** với đúng asset và cấu hình đã train. `model_40500.pt`/`yaw_pos_v5.onnx` cũ không tương thích với reference và lịch sử target mới, dù kích thước observation vẫn là 55. `play.py` không được bổ sung kiểm tra contract tự động.
2. Play POS nguyên bản đọc yaw limit từ stage đã lưu trong checkpoint. Mốc cấu hình cuối là 3 rad/s không có nghĩa mọi checkpoint đã học tới 3 rad/s; xem dòng `POS checkpoint yaw limit`.
3. Play tắt curriculum; clearance reward mặc định có thể còn 0.05 m. Nếu đánh giá reward/lift tại stage khác, truyền đúng clearance đã train cho cả ba term qua Hydra, ví dụ với stage lift cuối:

   ```text
   env.rewards.lift_clearance.params.target_clearance=0.20
   env.rewards.gated_yaw_tracking.params.target_clearance=0.20
   env.rewards.neutral_landing_progress.params.target_clearance=0.20
   ```

4. File ONNX tự export từ play không kèm đầy đủ thông tin bộ giới hạn ngoài network. Dùng exporter bên dưới để lưu contract trước khi chuẩn bị deploy.

Để ghi trace có command cố định và clearance đọc từ checkpoint, dùng script kiểm tra POS:

```bash
python scripts/reinforcement_learning/rsl_rl/record_yaw_pos_video.py \
  --checkpoint /duong/dan/run_moi/model_N.pt \
  --positive-yaw 0.20 --initial-neutral-seconds 2 \
  --positive-seconds 4 --neutral-seconds 4 --cycles 2 \
  --trace-only --headless
```

Script này có kiểm tra contract. Sau khi đạt stage cuối, đổi `--positive-yaw` lần lượt theo các tốc độ cần đánh giá, gồm 3.0.

## Export và deploy

```bash
python scripts/tools/export_yaw_pos_policy.py \
  --checkpoint /duong/dan/run_moi/model_N.pt \
  --output exported/yaw_pos_new_usd.onnx
```

Exporter xuất actor mean và ghi `yaw_pos_new_usd.contract.json`, đồng thời nhúng contract trong ONNX metadata. Contract chứa hash các layer USD, reference, joint order, observation layout, action scale, giới hạn target, physics dt và **yaw limit thực sự đã train**.

Trước khi chạy policy mới trên robot, luồng deploy cần áp dụng cùng reference, bộ giới hạn target ở 200 Hz và lịch sử target chuẩn hóa trong observation. Seed trạng thái limiter chân bằng encoder position lúc bàn giao điều khiển; giới hạn command theo stage đã train. `yaw_deploy` hiện tại chưa được chỉnh trong thay đổi này. Nếu động tác vào pose mạnh xảy ra ở đoạn PD/transition trước khi policy chạy, cần làm mượt đoạn đó trong deploy; thay đổi training chỉ tác động khi policy điều khiển.

Không dùng chênh lệch RPM thô làm tiêu chí duy nhất: hai bánh cần vận tốc lăn đối nghịch quanh tâm quay; giá trị motor còn phụ thuộc khoảng cách tới CoM, dấu trục và chuyển động shank.

## Kiểm tra đã thực hiện

- 243 kiểm thử CPU của POS, curriculum, transition safety và các thay đổi mới đều qua; gồm giới hạn target khi đảo lệnh/chạm hard limit, reset riêng từng env, history sau limiter, motor có dấu và rotation của parent, gate curriculum không đổi, kiểm tra USD mới và ONNX Runtime khớp actor mean RSL-RL.
- 45 kiểm thử hồi quy FSM thông thường đều qua.
- Chạy toàn bộ thư mục tests còn gặp lỗi ở bộ `test_yaw_fsm_staged.py`: code FSM hiện thiếu `StagedYawSchedule`/`StagedCycleTracker`. File FSM đó không được sửa trong thay đổi này.
- Chưa chạy PPO/rollout PhysX với USD mới hoặc thử trên robot: `nvidia-smi` báo không giao tiếp được NVIDIA driver trong môi trường kiểm tra. Kết quả CPU xác nhận các công thức và quy ước tín hiệu; hành vi policy cần được đánh giá sau training.
