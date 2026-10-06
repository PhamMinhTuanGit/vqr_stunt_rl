# Phase C: motor participation và ground-speed coverage

Experiment chỉ thêm hai dense rewards, override exploration FL/HR khi resume Phase C, reset dữ liệu window cũ và sửa biên certificate 0.90. Giữ task structure, 16 actions, observation policy/critic 55/83, phase design, neutral logic và final certification thresholds.

## Files changed

- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/yaw_pos_skill_rewards.py`: hai rewards và telemetry mean/min cho FL/HR.
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_pos_skill_cfg.py`: đăng ký hai term chỉ trong Skill, mỗi weight +1.0.
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/yaw_pos_skill_state.py`: gate differential dùng counters; aggregation minima của telemetry mới.
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/yaw_pos_skill_curriculums.py`: paired diagnostics dùng counters; thu và reset telemetry.
- `scripts/reinforcement_learning/rsl_rl/yaw_pos_skill_training.py`: override std/moments, reset pending/window và log success rate/coverage.
- `tests/test_yaw_pos_skill_motor_ground.py`: CPU tests mới.
- `tests/test_yaw_pos_skill_state.py`: fixture differential dùng raw pass/sample counts thay vì fractional pseudo-counts.
- Tài liệu report này. Các tài liệu/artifacts chưa tracked có từ trước được giữ nguyên.

## Reward formulas và weights

Với mỗi `i ∈ {FL, HR}`:

```text
motor_fraction_i = 0.091 * abs(joint_qdot_i) / max(abs(target_i), 1e-4)
motor_score_i = clamp(motor_fraction_i / 0.5, 0, 1)
motor_participation = min(motor_score_FL, motor_score_HR)         weight = +1.0

coverage_i = clamp(abs(measured_i) / max(0.5 * abs(target_i), 1e-4), 0, 1)
coverage_i = 0 nếu measured_i * target_i <= 0
ground_speed_coverage = min(coverage_FL, coverage_HR)              weight = +1.0
```

Target/measured lấy từ cùng `differential_rolling_kinematics` và motor qdot lấy từ cùng support joints của certificate. Chỉ trả điểm khi command `>0.1 rad/s`, geometry/heading hợp lệ, `abs(target_i)>1e-4`, support entities đúng thứ tự FL/HR và cả hai contact theo threshold hiện có. Motor reward loại mẫu qdot không finite. Ngoài mask trả 0. Dense shaping có điểm ngay trong active yaw, không cần chờ settle.

`rolling_tracking` giữ nguyên function, weight +2.0 và tracking magnitude. Không đổi certificate, sign gate, contact threshold, motor fraction >=0.5, ground speed >=0.5*target, residual tolerance `max(0.25*abs(target), 0.05)`, settle 0.5 s và reset settle timer khi đổi command. Hai rewards mới không cập nhật certificate counters/timers.

Theo lựa chọn đã xác nhận, lệnh experiment giữ `rolling_slip.normalized_excess=true` như run nguồn. Mặc định source của term này không bị sửa.

## Std override và checkpoint/window semantics

`runner.load(..., load_optimizer=True)` rồi `verify_skill_resume` kiểm tra actor/critic/log_std, toàn bộ optimizer và iteration khớp checkpoint trước khi override. Adaptive learning-rate scalar được phục hồi từ optimizer group như trước.

Chỉ khi checkpoint ở phase 2 (`C_differential`), đặt hai phần tử `log_std` của FL/HR theo joint names thành `log(0.09)` trước rollout đầu tiên. Không đổi initial std toàn policy; fresh training vẫn dùng logic cũ. Std vẫn learnable, không freeze/clamp trong PPO.

Chỉ zero `exp_avg`/`exp_avg_sq` và `max_exp_avg_sq` nếu có tại hai index FL/HR để bỏ quán tính Adam của exploration cũ. Giữ optimizer step, LR, các param groups và tất cả entries khác. Clear cached policy distribution để lần lấy mẫu tiếp theo dùng std mới. Actor/critic và log_std của 12 leg actions, FR/HL được giữ nguyên trước PPO update đầu tiên.

Restore curriculum giữ settings, phase, yaw stage, robustness stage và elapsed stage duration; không thay schema checkpoint. Reset `window`, `pending`, `consecutive_passes`, `last` trước `env.reset()`. Xóa `pending` trước reset ngăn promotion cũ bị commit. Simulator khởi động mới; episode accumulators, telemetry minima và settle/neutral timers được reset theo lifecycle hiện có. Chỉ samples của objective mới đi vào window mới.

## Boundary fix 0.90

Gate episode và window differential lấy raw passed/total counters, không so Python float32 ratio với 0.90:

```text
total_samples > 0 and passed_samples * 10 >= total_samples * 9
```

Implementation lấy numerator/denominator từ `Fraction(str(certificate))` để giữ hỗ trợ settings hiện có; threshold mặc định 0.90 cho phép so nguyên 10/9 như trên. Paired fixed/legacy diagnostics cũng so counters nguyên. Ratio vẫn dùng để log.

9/10 đạt chính xác; 8/10 trượt; 0 samples trượt. Không thêm epsilon, không đổi nghĩa threshold. Yêu cầu >=85% episode pass của window, joint episode success, ba consecutive windows, minimum duration và yaw/tracking thresholds giữ nguyên.

## Logging

- `Episode_Reward/motor_participation`, `Episode_Reward/ground_speed_coverage` qua reward manager.
- `Curriculum/skill/metric/differential_fail_{motor_speed,ground_speed,sign,residual}` giữ nguyên; mẫu số là active samples đã settle của completed episodes.
- `Curriculum/skill/gate_pass_rate/differential_episode` giữ nguyên; thêm `Curriculum/skill/success_rate`.
- `Policy/action_std/FL_WHEEL`, `Policy/action_std/HR_WHEEL` và mọi action std khác.
- `Live/skill/reward/{motor_participation,ground_speed_coverage}`: mean raw reward của bước điều khiển cuối mỗi PPO update; kiểm tra từng environment đều finite. Không phụ thuộc episode đã kết thúc.
- `Live/skill/metric/support_{fl,hr}_{motor_fraction,ground_speed_coverage}_{mean,min}`: thống kê các samples đủ điều kiện trong episode đang chạy; bổ sung cho window metrics khi smoke chưa có completed episode.
- `Curriculum/skill/metric/support_{fl,hr}_{motor_fraction,ground_speed_coverage}_{mean,min}`. Mean weighted theo đủ-điều-kiện samples; min là minimum thật trên các samples của completed episodes trong window, không phải trung bình các episode minima. Khi chưa có samples, logger ghi 0; các metric này không tham gia gate.
- `skill_prestart.json` ghi source checkpoint, iteration, phase/yaw limit, observation dims, std sau override, std FL/HR trước override, Adam moments đã clear, weights và residual mode.

## CPU tests

**132 passed, 3 warnings, 3.74 s** trên implementation cuối. Warnings từ RSL-RL về observation groups và empirical normalization, không phải test failures. Chạy CPU với CUDA bị ẩn và tắt pytest plugin autoload để tránh ROS plugin không tương thích trong environment này.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' \
  /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python -m pytest -q \
  tests/test_yaw_pos_skill_motor_ground.py tests/test_yaw_pos_skill_state.py \
  tests/test_yaw_pos_settle_resume.py tests/test_yaw_pos_differential.py \
  tests/test_yaw_pos_resume_exploration.py tests/test_yaw_pos_normalized_residual.py \
  tests/test_yaw_pos_task.py
```

Coverage: hai công thức, denominator biên, saturation, motor magnitude, worse wheel, wrong-sign/zero speed, neutral/deadband/negative commands, contact/geometry/heading invalid, non-finite measured, đúng thứ tự support, certificate/tracking không bị shaping thay đổi, 9/10 qua episode và window, paired diagnostics, mean/min aggregation/reset, std override theo tên với hai thứ tự action, moments selective, integrated resume hook có partial window/pending promotion và live logging trước completed episode, gồm rejection khi reward không finite.

## Resume smoke và lệnh đánh giá

Checkpoint mới nhất đã chốt ngay trước smoke cuối là `model_8550.pt` của run `2026-10-06_00-48-36_settle_normalized_residual`, phase 2, yaw stage 0, yaw limit 0.25, elapsed stage 166246 steps. Run nguồn tiếp tục lưu checkpoints; smoke dùng đường dẫn cố định dưới đây.

```bash
PHASE_C_CHECKPOINT='/home/robotics/tuanpm48/vqr/rl_training/logs/rsl_rl/vqr_wheel_yaw_flat_pos_skill/2026-10-06_00-48-36_settle_normalized_residual/model_8550.pt'

/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task Flat-VQR-Wheel-Yaw-POS-Skill --headless \
  --resume --checkpoint "$PHASE_C_CHECKPOINT" \
  --num_envs 4096 --seed 42 --max_iterations 10 \
  --run_name phase_c_motor_ground_smoke_live \
  env.rewards.rolling_slip.params.normalized_excess=true \
  agent.save_interval=5
```

Smoke đầu tiên từ model 8500 đã chạy đủ 10 updates, exit 0, giữ Phase C/yaw limit; FL/HR std cuối 0.089855/0.089932. Tuy nhiên 10 updates chỉ có 4.8 s simulation, chưa có episode kết thúc nên reward/window logs bằng 0. Bổ sung live logs và lặp lại 10 updates từ checkpoint 8550 để kiểm chứng reward thực tế, không kéo dài run đầu.

**Smoke cuối: PASS, 10 PPO updates, exit code 0.** Run: `2026-10-06_10-31-40_phase_c_motor_ground_smoke_live`; checkpoint cuối `model_8559.pt`.

| Kiểm tra | Kết quả |
|---|---|
| Phase/yaw stage/robustness stage | 2 / 0 / 0; Phase C suốt cả 10 updates |
| Yaw limit | 0.25 rad/s suốt cả 10 updates |
| FL/HR std trước override | 0.131283 / 0.124423 |
| FL/HR std ở prestart | 0.09 / 0.09 |
| FL/HR std sau 10 updates | 0.089994 / 0.089866 |
| Motor participation live, update cuối | 0.685351; range 0.619305–0.702505 |
| Ground-speed coverage live, update cuối | 0.690195; range 0.403939–0.702252 |
| Partial-window episodes nguồn / log đầu / cuối | 1 / 0 / 0; dữ liệu cũ bị loại |
| Elapsed stage trước / sau | 166246 / 166486 steps; cộng đúng 240 steps mới |
| Các log bắt buộc | Mỗi tag có đủ 10 điểm finite |
| Model và PPO losses cuối | Finite |

Live telemetry cuối trên samples của các episode đang chạy:

| Metric | Mean | Min |
|---|---:|---:|
| FL motor fraction | 2.874149 | 0.000009701 |
| HR motor fraction | 2.253225 | 0.000012035 |
| FL signed ground-speed coverage | 0.898344 | 0 |
| HR signed ground-speed coverage | 0.994177 | 0 |

Đối chiếu `params/env.yaml` của run nguồn và smoke xác nhận actions, observations, curriculum, commands, terminations và **tất cả rewards cũ** giống nhau; chỉ thêm hai rewards weight +1.0. Settings contract trong checkpoint cuối giống nguồn. Prestart verification xác nhận model/optimizer/iteration khớp trước override; 14 std còn lại khớp nguồn, với tolerance 1e-7 cho phép khác biệt phép exp CPU/GPU.

Smoke chưa có completed episode nên các reward/window metrics cấp episode vẫn bằng 0. Đây là kiểm tra resume, reward và logging; chưa chứng minh hai blocker giảm hoặc Phase C đủ điều kiện promotion. Motor fraction mean cũng không thay thế worse-wheel certificate; minima và ground-coverage minima cho thấy vẫn có samples yếu/sai dấu.

Bằng chứng: [log smoke cuối](../outputs/yaw_pos_skill_motor_ground_smoke_live.log), [JSON kiểm chứng và toàn bộ scalar summaries](../outputs/yaw_pos_skill_motor_ground_smoke_live_result.json), [prestart audit](../logs/rsl_rl/vqr_wheel_yaw_flat_pos_skill/2026-10-06_10-31-40_phase_c_motor_ground_smoke_live/skill_prestart.json). Log/JSON của smoke đầu được giữ tại `outputs/yaw_pos_skill_motor_ground_smoke{.log,_result.json}`.

Lệnh đánh giá **400 updates**, cung cấp để chạy khi được yêu cầu; chưa chạy training dài:

```bash
PHASE_C_CHECKPOINT='/home/robotics/tuanpm48/vqr/rl_training/logs/rsl_rl/vqr_wheel_yaw_flat_pos_skill/2026-10-06_00-48-36_settle_normalized_residual/model_8550.pt'

/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task Flat-VQR-Wheel-Yaw-POS-Skill --headless \
  --resume --checkpoint "$PHASE_C_CHECKPOINT" \
  --num_envs 4096 --seed 42 --max_iterations 400 \
  --run_name phase_c_motor_ground_eval400 \
  env.rewards.rolling_slip.params.normalized_excess=true \
  agent.save_interval=50
```

`--max_iterations` là số PPO updates bổ sung khi resume. Với source iteration 8550, 10 updates được gắn nhãn 8550–8559; 400 updates được gắn nhãn 8550–8949 theo convention hiện có của runner.
