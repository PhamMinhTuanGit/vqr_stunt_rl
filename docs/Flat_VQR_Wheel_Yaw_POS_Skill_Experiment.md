# Flat-VQR-Wheel-Yaw-POS-Skill: curriculum theo kỹ năng

Ngày kiểm tra: 2026-10-05. Branch: `experiment/yaw-pos-skill-phases`. Task Gym mới là `Flat-VQR-Wheel-Yaw-POS-Skill`; runner và log nằm riêng trong `vqr_wheel_yaw_flat_pos_skill`. Task `Flat-VQR-Wheel-Yaw-POS` vẫn dùng config, runner và curriculum cũ. Experiment chỉ hỗ trợ PPO **train from scratch**; `resume`, checkpoint đầu vào, transfer và adaptation đều bị từ chối.

## Files changed

| File | Vai trò |
| --- | --- |
| `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/__init__.py` | Đăng ký task riêng. |
| `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/agents/rsl_rl_ppo_cfg.py` | Runner riêng, `resume=False`, không load checkpoint; giữ hyperparameters PPO của POS. |
| `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_pos_skill_cfg.py` | Kế thừa reward POS, thay curriculum, DR/noise và khai báo rõ contract critic. |
| `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/yaw_pos_skill_state.py` | State machine và quyết định promotion độc lập simulator. |
| `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/yaw_pos_skill_curriculums.py` | Thu telemetry episode, áp dụng độ khó, xuất gate và blocker. |
| `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/yaw_pos_skill_noise.py` | Noise policy theo robustness stage; wheel velocity giữ ±0,5 rad/s. |
| `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/yaw_pos_skill_dr.py` | DR vật lý có giới hạn 1024 material buckets/stage và delay actuator theo stage. |
| `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/yaw_pos_skill_rewards.py` | Wrapper chỉ thêm telemetry anchor; giá trị neutral reward bằng POS. |
| `scripts/reinforcement_learning/rsl_rl/yaw_pos_skill_training.py` | Khởi tạo std wheel, kiểm tra observation, log PPO, save/restore state. |
| `scripts/reinforcement_learning/rsl_rl/train.py` | Kích hoạt hooks riêng cho task mới, bắt đầu episode từ đầu. |
| `tests/test_yaw_pos_skill_state.py` | CPU tests cho phase, checkpoint, std, noise, DR, reward parity. |
| `docs/Flat_VQR_Wheel_Yaw_POS_Skill_Experiment.md` | Báo cáo này. |

Tên observation critic là `rolling_lateral_contact_velocity`, khớp với hàm `wheel_contact_kinematics.rolling_lateral_contact_velocity`; config task mới kiểm tra không xuất hiện alias `contact_velocity`. Actor vẫn 55 chiều, critic 83 chiều. Không thêm base linear velocity XY hoặc anchor error XY ở experiment này. Nếu A–C đã pass mà D vẫn không học được, đó là điều kiện để mở experiment observation kế tiếp.

## State machine và gate

Tất cả phase giữ clearance stage 0 = **0,05 m**. A–D giữ yaw stage 0 = **0,25 rad/s**. Reward weights và các công thức reward POS được kế thừa; chỉ thay neutral-position bằng wrapper trả về đúng cùng tensor reward để đo anchor coverage. Mỗi lần xét promotion dùng cửa sổ rời nhau có ít nhất **2048 eligible episodes**, **≥85%** episode cùng đạt mọi gate cấp episode đang bật, **≥85%** pass cho từng gate cấp episode, và **3 cửa sổ đạt liên tiếp**. Phase A cần ít nhất **1000 environment steps**, các phase/stage còn lại **6000 steps**. Reset toàn bộ environments tại biên chuyển phase/stage để không trộn episode hai mức khó. Tại Phase A mọi episode hoàn thành đều được xét, kể cả không có mẫu active-yaw; từ B trở đi chỉ episode có active-yaw sample được xét.

| Phase | Mục tiêu để advance | Episode gates bật | Window gates bật | Gate tắt |
| --- | --- | --- | --- | --- |
| A — Pose acquisition | Có pose đỡ và nâng bánh ổn định ở yaw 0,25 | support, lift, balance, minimum height, safe | Không | yaw, tracking, differential, neutral |
| B — Yaw acquisition | Pose vẫn đạt, yaw và tracking đạt | A + yaw | tracking và edge tracking | differential, neutral |
| C — Differential rolling | Yaw ổn định và hai bánh FL/HR rolling đúng chiều, tiếp xúc, ít slip | B + differential episode | B + differential pass rate | neutral |
| D — Neutral landing/hold | Differential vẫn đạt; neutral hạ đủ 4 bánh, dừng và giữ anchor | C | C + neutral hold, four-contact, anchor coverage | Không |
| E — Yaw-speed curriculum | Tăng yaw theo bảng; sau 3,00 mới bật robustness | Giữ toàn bộ D | Giữ toàn bộ D | Không |

`safe` đòi không có unsafe/torso termination. `joint_episode_success` là phép AND của đúng các episode gates **đang bật**, không phải tỷ lệ thành công duy nhất che mất nguyên nhân. Log ghi riêng enabled/disabled, threshold, pass rate và blocker của từng gate; đồng thời ghi blocker do thiếu episodes, số cửa sổ liên tiếp và thời gian stage. Differential/neutral final certification **≥90%** được giữ nguyên tại C/D/E; nếu một trong hai tụt, E không tăng yaw hoặc robustness.

Yaw limit của E: **0,25; 0,40; 0,55; 0,70; 0,85; 1,00; 1,25; 1,50; 1,75; 2,00; 2,50; 3,00 rad/s**. Tracking overall threshold tương ứng: **0,30; 0,35; 0,40; 0,45; 0,50; 0,55; sau đó 0,55**. Edge tracking threshold: **0,20; 0,25; 0,30; 0,35; 0,40; 0,45; sau đó 0,45**. Yaw score ≥**0,65**. Pose thresholds: support ≥**0,85**, lift progress ≥**0,80**, balance ≥**0,75**, minimum base height ≥**0,35 m**.

Differential certificate dùng logic POS hiện tại: FL và HR đều tiếp xúc với ngưỡng lực **1 N**, ground rolling đúng dấu target, độ lớn đo được và wheel-motor fraction đều ≥**0,5×** target, rolling residual ≤`max(0,25×|target|, 0,05 m/s)`, sau **0,5 s** active settle. Neutral hold vẫn dùng 4-contact **1 N**, planar speed ≤**0,03 m/s**, anchor drift ≤**0,05 m**, contact dwell **0,2 s**. Phase D còn yêu cầu four-contact và anchor coverage ≥**90%** trên neutral samples; các ngưỡng certificate tổng thể vẫn ≥**90%**.

## Std, noise và robustness

Khởi tạo PPO `log_std` theo **tên joint thực tế**: 12 leg actions có std **1,0** như POS; cả 4 wheel actions có std **0,15** (bao gồm FL/HR), sau đó PPO tiếp tục học parameter này. Policy joint-velocity noise của wheel là uniform **±0,5 rad/s** ở mọi phase. Noise cho các channel khác bằng 0 ở nominal, rồi tăng theo robustness scale đến angular velocity ±0,4, projected gravity ±0,1, joint position ±0,02 cộng bias ±0,1, leg velocity ±3,0. Actor/critic dimension không đổi.

Phase A–D và toàn bộ yaw ramp E chạy với robustness scale **0**: không có external force/torque, interval push, mass/inertia/CoM/gain/material randomization; actuator delay cố định **2 steps**. Reset XY/yaw position baseline vẫn được giữ để tập nhiều vị trí/hướng. Sau khi E đạt yaw **3,00 rad/s** với cả hai certificate final ổn định, robustness lần lượt là **0; 0,10; 0,25; 0,50; 0,75; 1,00**. Ở scale `s`, force/torque tối đa ±`10s`, interval push XY ±`0,15s` m/s (interval baseline 10–15 s), actuator gains `1±0,15s`, reset tilt ±`0,3s` rad, reset linear velocity ±`0,2s` m/s, reset angular roll/pitch ±`0,05s` rad/s và actuator delay từ 2 đến `2+floor(6s)` steps. Mass non-base và inertia diagonal `1±0,15s`, TORSO có thêm −`s` đến +`3s` kg, COM XY ±`0,03s` m và Z ±`0,02s` m. Friction từ 1,0 tiến đến [0,35;1,50], restitution từ 0 đến [0;0,7]; dùng tối đa 1024 material buckets cho mỗi robustness stage. Thuộc tính vật lý robust được lấy mẫu lại trên mỗi reset episode từ nominal gốc.

## Logging và checkpoint

TensorBoard `Curriculum/skill/*` ghi phase, clearance/yaw stage và limit, robustness stage/scale, `gate_enabled/*`, `gate_threshold/*`, `gate_pass_rate/*`, `blocker/*`, support/lift/balance/height/yaw scores, tracking overall/edge, differential và neutral pass rate, FL/HR signed ratio, wrong-sign %, neutral four-contact, planar speed, anchor drift và anchor coverage. `Policy/action_std/*` ghi std theo joint; `actual_delay_min_steps`/`actual_delay_max_steps` kiểm tra delay thật. `skill_prestart.json` ghi iteration, phase, std và observation dimensions trước rollout.

Checkpoint RSL-RL chứa thêm `infos["yaw_pos_skill_curriculum"]`: `version=1`, settings contract, phase/yaw/robustness stage, streak, `complete`, pending transition, partial window metrics/pass counts, last evaluation và elapsed steps trong stage. Hàm `restore_skill_state` kiểm tra version, settings, index và window trước khi áp dụng lại độ khó. Lệnh train mới cố ý không resume checkpoint; state này dành cho audit và khả năng restore có kiểm soát về sau.

## Kết quả xác minh

- CPU: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .../python -m pytest -q tests/test_yaw_pos_*.py tests/test_yaw_curriculum.py` → **264 passed**; gồm kiểm tra hành vi task POS cũ, từng gate A–E, save/load state, reward parity, std wheel và noise/DR. `git diff --check` và `py_compile` đều qua.
- GPU smoke: 256 environments, 10 PPO updates (iterations 0–9), seed 42; xác nhận `starting_iteration=0`, phase A, yaw limit 0,25, robustness 0, policy 55 / critic 83, leg std 1,0 và 4 wheel std `0,150000006`. TensorBoard có 10 điểm cho phase A, yaw/differential/neutral gate đều disabled, delay thật luôn 2 steps; FL wheel std tăng từ 0,15004 đến 0,15051 do PPO học. Checkpoints 0, 5, 9 có state mới; các tensor checkpoint hữu hạn. Artefacts nằm tại `logs/rsl_rl/vqr_wheel_yaw_flat_pos_skill/2026-10-05_15-38-24_smoke_skill_verified/`. Smoke ngắn chưa đạt điều kiện 2048 eligible episodes/cửa sổ nên không chứng minh promotion trong simulator; CPU tests kiểm tra transitions và blockers.

Lệnh smoke từ repo root:

```bash
/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Flat-VQR-Wheel-Yaw-POS-Skill --headless --device cuda:0 \
  --num_envs 256 --max_iterations 10 --seed 42 --run_name smoke_skill_verified \
  agent.save_interval=5
```

Lệnh **train from scratch** khi quyết định chạy dài (chưa thực thi):

```bash
/home/robotics/miniconda3/envs/env_isaaclab_tuanpm48_vqr/bin/python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Flat-VQR-Wheel-Yaw-POS-Skill --headless --device cuda:0 \
  --num_envs 4096 --max_iterations 30000 --seed 42 --run_name skill_phases_scratch
```
