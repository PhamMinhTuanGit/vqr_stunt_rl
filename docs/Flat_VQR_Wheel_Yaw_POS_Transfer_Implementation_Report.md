**Kết quả triển khai `Flat-VQR-Wheel-Yaw-POS-Transfer` — 2026-10-01**

Task mới đã được đăng ký, nạp checkpoint base strict, chạy rollout PhysX/GPU 32-env và PPO smoke thành công. Không chạy Phase B hoặc training dài. `Flat-VQR-Wheel-Yaw-POS` không bị overwrite; các thay đổi POS đã tồn tại trước phiên làm việc được giữ nguyên.

**Audit trước triển khai**

- Base: `yaw_env_cfg.VQRWheelFlatEnvCfg`, 18 reward, runner `VQRWheelYawFlatPPORunnerCfg`. Support POS: FL/HR; lift: FR/HL. Action và observation kế thừa `velocity_yaw_env_cfg`.
- POS hiện tại: `yaw_env_pos_cfg.VQRWheelFlatEnvPOSCfg`. POS đã đổi yaw tracking std sang 0.20, thay lift/support gating và thêm heading support shaping. Transfer không dùng các positive implementation này.
- Command base: scalar `YawRateCommand`, uniform đối xứng. Reuse `YawPosCommand` cho task mới: exact zero 30%, uniform từ trên deadband tới yaw limit 70%, resample 4–6 s; không yaw âm. External setter cũng clamp yaw âm về zero.
- Curriculum base: clearance 0.05/0.10/0.15/0.20 m, rồi yaw 0.25/0.40/0.55/0.70/0.85/1.00 rad/s và online DR 0.30/0.40/0.55/0.70/0.85/1.00. Giữ support 0.85, lift 0.80, balance 0.75, yaw score 0.65, success rate 0.85, 2048 episode/window, 3 consecutive windows và minimum stage duration 1000/6000 steps.
- RSL-RL cài tại môi trường này: **3.1.2**. `runner.load(..., load_optimizer=False)` vẫn restore iteration; loader transfer reset iteration về zero rõ ràng. Policy load strict; `log_std` thuộc policy state nên được nạp nguyên trạng. Không load optimizer hoặc checkpoint infos/curriculum của base.
- Gym: giữ nguyên registration của hai task cũ; thêm registration mới, EnvCfg/RunnerCfg và namespace riêng.

**File thêm**

1. `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_pos_transfer_cfg.py`: EnvCfg/reward/command/curriculum kế thừa base; bảng phân loại reward.
2. `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/agents/rsl_rl_pos_transfer_cfg.py`: RunnerCfg riêng, giữ PPO settings của base.
3. `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/yaw_pos_transfer_rewards.py`: wrapper mode cho các reward base, neutral yaw và diagnostics.
4. `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/yaw_pos_transfer_curriculums.py`: chứng nhận positive bằng positive samples, gate bảo toàn neutral.
5. `scripts/reinforcement_learning/rsl_rl/yaw_pos_transfer.py`: audit contract YAML/checkpoint và policy-only initialization.
6. `scripts/reinforcement_learning/rsl_rl/smoke_yaw_pos_transfer.py`: kiểm tra native simulator, rollout và một PPO update.
7. `tests/test_yaw_pos_transfer_task.py`: unit/contract tests, gồm loader RSL-RL thật trên CPU.
8. `docs/Flat_VQR_Wheel_Yaw_POS_Transfer_Implementation_Report.md`: báo cáo này.

**File sửa**

1. `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/__init__.py`: chỉ thêm registration.
2. `scripts/reinforcement_learning/rsl_rl/train.py`: thêm `--transfer_checkpoint` và nhánh riêng cho task transfer; checkpointing curriculum của task mới. Luồng base/POS hiện tại giữ nguyên.

Spec `docs/Flat_VQR_Wheel_Yaw_POS_Transfer_Technical_Spec.md` là file người dùng cung cấp, không sửa. `play.py`, `yaw_env_pos_cfg.py`, `yaw_pos_rewards.py`, `yaw_pos_curriculums.py`, `test_yaw_pos_task.py` và `test_yaw_pos_keyboard_play.py` đã dirty từ trước phiên làm việc; không sửa thêm các file đó. SHA-256 của 11 file được bảo vệ khớp snapshot trước triển khai. AST của registration base/POS khớp HEAD; 52 regression tests hiện có pass.

**Registration và config diff**

- Task ID chính xác: `Flat-VQR-Wheel-Yaw-POS-Transfer`.
- EnvCfg: `yaw_env_pos_transfer_cfg:VQRWheelYawPosTransferEnvCfg`.
- RunnerCfg: `agents.rsl_rl_pos_transfer_cfg:VQRWheelYawPosTransferPPORunnerCfg`.
- Namespace: `logs/rsl_rl/vqr_wheel_yaw_flat_pos_transfer/`.

| Thành phần | Flat-VQR-Wheel-Yaw | POS-Transfer |
|---|---|---|
| Command ban đầu | uniform [-0.25, +0.25] | 30% exact zero; 70% uniform (>0.10, +0.25] |
| Mode | luôn objective hai bánh | <=0.10 neutral FOUR_STAND; >0.10 POS hai bánh |
| Positive objectives | hàm/params/weight base | cùng hàm/params/weight, bọc positive mask |
| Neutral objectives | không có FOUR_STAND conditioning | landing + default pose + four-contact + low absolute yaw |
| Planar velocity | penalty -1, luôn active | giữ nguyên, cũng đáp ứng mục tiêu neutral đứng yên |
| Safety/regularization | luôn active | giữ nguyên 10 GLOBAL terms |
| Positive certification | toàn episode | chỉ positive samples; balance normalize theo positive duration |
| Promotion | positive thresholds/stage tables base | giữ tất cả; thêm gate neutral độc lập |
| PPO/network | base settings | giữ nguyên, kể cả entropy 0.01 |
| Checkpoint initialization | resume thường có optimizer/curriculum/iteration | nạp actor/critic/log_std; optimizer mới; curriculum/iteration zero |
| Actor / critic / action | 55 / 83 / 16 | 55 / 83 / 16, cùng ordering/preprocessing |
| Log namespace | vqr_wheel_yaw_flat | vqr_wheel_yaw_flat_pos_transfer |

Không thêm reward HipX, support_x_rms hoặc alignment. Critic vẫn giữ **observation** `support_wheel_alignment` có sẵn để tương thích checkpoint; không dùng nó làm reward hoặc curriculum gate.

**Phân loại toàn bộ reward**

| Reward | Weight | Classification |
|---|---:|---|
| com_support | 3 | POSITIVE_SPECIFIC |
| base_height | 2 | POSITIVE_SPECIFIC |
| support_span_band | -1 | POSITIVE_SPECIFIC |
| lift_clearance | 3 | POSITIVE_SPECIFIC |
| com_inside_segment | 2 | POSITIVE_SPECIFIC |
| balance | 2 | POSITIVE_SPECIFIC |
| gated_yaw_tracking | 8 | POSITIVE_SPECIFIC |
| lifted_wheel_spin | -0.02 | POSITIVE_SPECIFIC |
| neutral_landing_progress | 3 | NEUTRAL_SPECIFIC |
| four_stand_pose | 2 | NEUTRAL_SPECIFIC |
| four_wheel_contact | 3 | NEUTRAL_SPECIFIC |
| neutral_yaw_tracking | 8 | NEUTRAL_SPECIFIC |
| low_base_height | -4 | GLOBAL |
| downward_low_base_velocity | -8 | GLOBAL |
| lateral_slip | -2 | GLOBAL |
| rolling_slip | -0.5 | GLOBAL |
| undesired_contact | -2 | GLOBAL |
| joint_limits | -0.5 | GLOBAL |
| action_rate | -0.02 | GLOBAL |
| joint_velocity | -0.001 | GLOBAL |
| torque | -0.0001 | GLOBAL |
| planar_velocity | -1 | GLOBAL |

Positive yaw tracking giữ `std=0.30`, support contact gate và clearance gate floor 0.25 của base. Positive wrappers cũng mask cập nhật episode metrics của lift/yaw để neutral không cấp bằng chứng positive. Các neutral landing/pose/contact reuse implementation đã được test, với std pose 0.25 và contact threshold 1 N; neutral yaw dùng absolute yaw và std 0.30.

Neutral episode thành công khi four-contact rate >=0.85, mean default-pose score >=0.75, mean absolute yaw <=0.10 rad/s, mean planar speed <=0.10 m/s, minimum base height >=0.35 m và không torso termination. Contact/pose/success/evidence thresholds reuse base; **ngưỡng mới duy nhất** là neutral planar speed 0.10 m/s. Neutral evidence cần >=2048 completed episodes có neutral samples. Mỗi positive evaluation window phải pass cả neutral gate để tích lũy consecutive passes. Nếu neutral fail ở cửa sổ promotion, rollback stage, reward targets và online DR, rồi reset streak. Episode metrics chỉ reset tại env IDs hoàn tất.

**Checkpoint compatibility**

Nguồn được người dùng chỉ định:

```text
logs/rsl_rl/vqr_wheel_yaw_flat/2026-09-21_15-32-03_yaw_dr_curriculum/model_29998.pt
```

Checkpoint có actor input 55, critic input 83, action output 16; MLP [512,256,128], ELU, learned `log_std[16]`. Loader kiểm tra source `params/env.yaml`, thứ tự term/joint, preprocessing, default offsets, action scale/clip và toàn bộ state keys/shapes trước khi load. Thiếu metadata hoặc thay đổi ordering/shape thì fail loudly; không partial load. Source curriculum clearance stage 3, yaw stage 4 được bỏ qua.

Actor order/dims: `base_ang_vel(3), projected_gravity(3), yaw_rate_cmd(1), joint_pos(16), joint_vel(16), actions(16)`.

Critic order/dims: `base_lin_vel(3), base_height(1), base_ang_vel(3), projected_gravity(3), yaw_rate_cmd(1), joint_pos(16), joint_vel(16), actions(16), wheel_normal_force(4), wheel_contact(4), wheel_clearance(4), support_wheel_alignment(2), com_support_coordinate(2), rolling_lateral_contact_velocity(8)`.

Wheel positions vẫn là 4 zero slots trong joint_pos(16), đúng base. Thứ tự joint observation là 12 leg joints theo FL, FR, HL, HR (mỗi chân HipX/HipY/Knee), rồi FL/FR/HL/HR wheel.

Action order kiểm tra tại runtime: `FL_HipX_joint, FL_HipY_joint, FL_Knee_joint, FR_HipX_joint, FR_HipY_joint, FR_Knee_joint, HL_HipX_joint, HL_HipY_joint, HL_Knee_joint, HR_HipX_joint, HR_HipY_joint, HR_Knee_joint, FL_WHEEL, FR_WHEEL, HL_WHEEL, HR_WHEEL`.

Sau transfer: iteration 0, optimizer state rỗng, clearance stage 0/target 0.05 m, yaw stage 0/range [0,0.25], DR 0.30. Actor/critic state và log_std khớp checkpoint từng tensor trước PPO.

**Kết quả kiểm tra**

| Kiểm tra | Kết quả |
|---|---|
| CPU/unit + regression | **82 passed**, 3 cảnh báo RSL-RL có sẵn |
| Base/POS giữ nguyên | PASS: source hashes, registration AST, config/weight/command regression |
| yaw 0 / 0.05 / 0.10 | neutral only, positive contributions = 0 |
| yaw 0.100001 / 0.25 | positive only, neutral contributions = 0 |
| 8 positive rewards trong simulator | **bit-exact** so với base trên cùng state/command |
| 10 global terms | bit-exact so với base; finite ở hai mode; low-height/action-rate nonzero khi kích hoạt |
| Loader reject ordering/preprocessing/shape/keys/NaN mismatch | PASS |
| Loader RSL-RL CPU và GPU | PASS: strict full policy, fresh optimizer, iteration/curriculum reset |
| Actor / critic forward | PASS: (32,16) / (32,1), finite |
| Headless PhysX/GPU rollout | PASS: **32 envs x 240 steps**, finite obs/rewards |
| Harness PPO | PASS: **1 update, 24 steps x 32 envs**, actor/critic/log_std đổi, checkpoint lưu |
| Entrypoint train.py | PASS: **1 update riêng**, log namespace mới, model_0.pt lưu |
| Optimizer của CLI smoke | Adam step = 20 sau 5 epochs x 4 minibatches; không restore momentum/step base |
| Saved CLI curriculum | clearance_stage=0, yaw_stage=0, consecutive_passes=0 |
| Diagnostics | Đủ tất cả mode, positive, neutral, safety metrics dưới Transfer/ và Curriculum/task_levels/ |
| Whitespace diff | git diff --check PASS |

Tổng cộng hai run smoke độc lập, mỗi run đúng một PPO update; không adaptation/training dài. Neutral success trong smoke là 0: checkpoint base chưa học FOUR_STAND theo command. Smoke xác nhận mechanics, không chứng minh chất lượng neutral hoặc retention dài hạn; Phase B vẫn cần đánh giá theo spec.

**Artifacts**

- `outputs/yaw_pos_transfer_smoke/regression_before.json`: hash snapshot trước triển khai.
- `outputs/yaw_pos_transfer_smoke/results.json`: kết quả máy đọc được, order/dims, reward weights, checks và CLI checkpoint metadata.
- `outputs/yaw_pos_transfer_smoke/unit_tests.log`: 82 tests.
- `outputs/yaw_pos_transfer_smoke/simulator_smoke.log`: checkpoint-load/rollout/PPO harness.
- `outputs/yaw_pos_transfer_smoke/ppo_smoke.pt`: checkpoint sau một PPO update của harness.
- `outputs/yaw_pos_transfer_smoke/ppo/`: TensorBoard event và model_0.pt của harness.
- `outputs/yaw_pos_transfer_smoke/training_cli_smoke.log`: run qua entrypoint chính thức.
- `logs/rsl_rl/vqr_wheel_yaw_flat_pos_transfer/2026-10-01_00-36-00_pos_transfer_cli_smoke/`: model_0.pt, TensorBoard event, params/env.yaml, params/agent.yaml và git diff của CLI smoke.

**Lệnh tái lập đã chạy**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python -m pytest \
  tests/test_yaw_pos_transfer_task.py tests/test_yaw_curriculum.py tests/test_yaw_pos_task.py \
  tests/test_yaw_pos_keyboard_play.py tests/test_yaw_pos_resume_exploration.py tests/test_yaw_pos_video.py -q

/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python scripts/reinforcement_learning/rsl_rl/smoke_yaw_pos_transfer.py \
  --checkpoint logs/rsl_rl/vqr_wheel_yaw_flat/2026-09-21_15-32-03_yaw_dr_curriculum/model_29998.pt \
  --device cuda:0

/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Flat-VQR-Wheel-Yaw-POS-Transfer \
  --transfer_checkpoint logs/rsl_rl/vqr_wheel_yaw_flat/2026-09-21_15-32-03_yaw_dr_curriculum/model_29998.pt \
  --num_envs 32 --headless --device cuda:0 --max_iterations 1 --run_name pos_transfer_cli_smoke
```

**Lệnh training từ checkpoint base, chỉ cung cấp — chưa chạy**

Lệnh dưới bắt đầu Phase B, 1000 adaptation updates. Chạy tại repository root. Không thêm `--resume`: nó restore optimizer/curriculum của run cũ; task transfer yêu cầu `--transfer_checkpoint` cho initialization và từ chối dùng resume ở phiên bản này.

```bash
/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Flat-VQR-Wheel-Yaw-POS-Transfer \
  --transfer_checkpoint logs/rsl_rl/vqr_wheel_yaw_flat/2026-09-21_15-32-03_yaw_dr_curriculum/model_29998.pt \
  --num_envs 4096 --headless --device cuda:0 --max_iterations 1000 --run_name pos_transfer_adaptation
```

**Xác nhận: `Flat-VQR-Wheel-Yaw-POS was not overwritten.`**
