> Cập nhật quyết định training: task FSM được train mới từ iteration 0. Không sửa `train.py` hoặc framework training, không resume checkpoint task gốc, không model surgery và không critic warm-up. `docs/document_yaw_fsm/AGENTS.md` là contract hiện hành nếu nội dung cũ bên dưới mâu thuẫn.

Đã nắm đủ ngữ cảnh. Trình bày phương án chi tiết dưới đây — bám vào task `VQRWheelFlatEnvFSMCfg` và `VQRYawFSM` 7 trạng thái.

# Phương án: task `Flat-VQR-Wheel-Yaw-FSM` kế thừa Flat-VQR-Wheel-Yaw + FSM

## 1. Hiện trạng & ràng buộc

- `VQRYawFSM` trong `mdp/fsm.py` là **logic Python scalar** (1 robot, hysteresis enter 0.10 / exit 0.05), chưa dùng được cho 4096 envs vectorized.
- FSM chia task thành 2 đường chéo: `YAW_POS` = cấu hình hiện tại (support FL+HR, nhấc FR+HL, yaw reference +1.0), `YAW_NEG` = mirror (support FR+HL, nhấc FL+HR).
- **Ràng buộc quan trọng nhất**: `Flat-VQR-Wheel-Yaw` là baseline đang có hard-check trong `train.py:206-216` (đúng 18 reward term, weight 8.0) và đang train — mọi thay đổi trong `mdp/` phải giữ task gốc **hoạt động y hệt** (tham số mới default `None` → đi nhánh cũ).

## 2. Kiến trúc đề xuất — FSM sống trong CommandTerm

```
yaw_rate_cmd (resample 4–6s)
        │
        ▼
YawFSMCommand.compute()  ← hook mỗi step (như YawRateCommand._update_command đã chạy mỗi step)
        │  đọc: contact Fz (sensor), wheel_clearance, roll/pitch, base_height
        │  tính predicate vectorized: positive_pose_ready / negative_pose_ready /
        │                            four_stand_ready / unsafe   (N,)
        │  update: fsm_state_buf (N,), support_diagonal_buf (N,), state_time_buf (N,)
        ▼
env.fsm_state  ──►  rewards (gate per-state, dual-diagonal)
                ──►  observations (one-hot policy + ready flags critic)
                ──►  curriculum (metrics chỉ đếm trong YAW_*)
                ──►  terminations (transition timeout, tùy chọn)
```

Chọn CommandTerm làm host vì: (a) `_update_command()` được gọi mỗi step, đúng thứ tự trước physics, reward đọc state 1-step-old (20ms) — chấp nhận được; (b) `command` property vẫn trả `(N,1)` yaw rate nên mọi obs/reward hiện tại dùng `generated_commands` không phải đổi; (c) tránh hack event `interval=(0,0)`.

Lưu ý cần thêm buffer `support_diagonal_buf` (±1) — vì `RETURN_TO_4` có thể đến từ YAW_POS **hoặc** YAW_NEG, cần nhớ đường chéo nào đang держ để biết cặp bánh nào phải hạ xuống.

## 3. Thành phần cụ thể

### 3.1 Vector hóa FSM (`mdp/fsm.py`)
Giữ `VQRYawFSM` scalar làm **oracle** cho test; thêm class torch `YawFSMVectorized` cùng bảng chuyển state, chạy (N,) một lần. Predicate threshold tái dùng các helper có sẵn trong `observations.py` (`wheel_contact`, `wheel_clearance`, `support_wheel_alignment`):
- `positive_pose_ready` = FL+HR contact (F>1.0) ∧ clearance(FR,HL) ≥ 0.8×target ∧ |roll,pitch| < ngưỡng
- `negative_pose_ready` = mirror
- `four_stand_ready` = 4 bánh contact ∧ |roll,pitch| < ngưỡng
- `unsafe` = torso contact grace ∧/∨ base_height < MIN_BASE_HEIGHT ∧/∨ |roll,pitch| lớn

### 3.2 Ma trận reward theo state (các dấu × = term hoạt động)

| Term (hệ số giữ nguyên)                                                                                                                                                           | FOUR_STAND  | TRANS_*    | YAW_POS        | YAW_NEG        | RETURN                    | SAFE |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ---------- | -------------- | -------------- | ------------------------- | ---- |
| `com_support`, `com_inside_segment`, `support_span_band`                                                                                                                          |             | ×          | × (FL+HR)      | × (FR+HL)      | × (theo diagonal buf)     |      |
| `lift_clearance`                                                                                                                                                                  |             | ×          | × (nhấc FR+HL) | × (nhấc FL+HR) |                           |      |
| `gated_yaw_tracking` (8.0)                                                                                                                                                        |             | floor 0.25 | ×              | ×              |                           |      |
| **mới** `four_stand_stability` (+3): 4 bánh tiếp đất, CoM tâm, height band ~0.38–0.46                                                                                             | ×           | fade-out   |                |                | fade-in                   | ×    |
| **mới** `transition_progress` (+2): clearance tiến dần target theo hướng active                                                                                                   |             | ×          |                |                | × (hướng ngược: hạ xuống) |      |
| `balance`                                                                                                                                                                         | ×           | ×          | ×              | ×              | ×                         | ×    |
| `base_height` (0.49)                                                                                                                                                              |             |            | ×              | ×              |                           |      |
| penalty cứng (`downward_low_base_velocity`, `low_base_height`, `planar_velocity`, `torque`, `action_rate`, `joint_velocity`, `joint_limits`, `lateral_slip`, `undesired_contact`) | × mọi state |            |                |                |                           |      |
| `lifted_wheel_spin`, `rolling_slip`                                                                                                                                               |             |            | ×              | ×              |                           |      |

Cách impl dual-diagonal không copy-paste: reward hàm tính **cả hai chéo** với 2 bộ `body_ids` có sẵn, rồi `torch.where` theo mask state — chỉ ~6 hàm trong `rewards.py` cần thêm param tùy chọn (`fsm_command_name=None` default). Riêng `gated_yaw_tracking` cần bản mới `fsm_gated_tracking` vì các accumulator `_yaw_support_score_*`/edge-metrics phải **chỉ cập nhật trong YAW_\*** (bản hiện tại luôn cộng, nếu chỉ zero-mask thì metrics sẽ bị pha loãng 60-80%).

### 3.3 Observations
- **Policy**: giữ nguyên 55D + **one-hot 7 state** (tự nhiên, robust cho MLP; command scalar đã có sẵn) → 62D.
- **Critic**: 83D + one-hot 7 + 4 ready/unsafe flags + `state_time` (1) → 95D.
- Không đổi noise model.

### 3.4 Curriculum — chèn 1 pha mới vào `yaw_task_levels`
- **Pha A (lift)**: như hiện tại nhưng phải certify **cả 2 đường chéo** (đủ episode mỗi bên).
- **Pha B (transition) — mới**: từ FOUR_STAND đạt `pose_ready` trong timeout (vd 3s) với success rate ≥0.85 trên 2048 episode, 3 cửa sổ liên tiếp, cả 2 hướng.
- **Pha C (yaw+DR)**: giữ nguyên scale ladder, metrics chỉ đếm trong YAW_*, giữ accumulator names để không làm hồi quy task gốc.
- Hysteresis 0.10/0.05 với limit ±0.25: ~60% lệnh vào chuyển tiếp, ~20% đứng yên — cân bằng mode tự nhiên; có thể thêm `stand_probability` trong sampler nếu lệch.

### 3.5 Terminations
Giữ `torso_contact` grace 0.15s. Thêm tùy chọn `fsm_transition_timeout` (DoneTerm đọc `state_time_buf`) — mặc định bật 3s vì transition kẹt làm Foodedžit dead-time; SAFE_RECOVERY semantics là 1 điểm quyết định (xem câu hỏi cuối).

### 3.6 Files chạm đến
| File                                         | Việc                                                                                              |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `mdp/fsm.py`                                 | + `YawFSMVectorized` (giữ scalar làm oracle test)                                                 |
| `mdp/commands.py`                            | + `YawFSMCommand` / `YawFSMCommandCfg` (kế thừa `YawRateCommand`)                                 |
| `mdp/observations.py`                        | + `fsm_state_onehot`, `fsm_ready_flags`                                                           |
| `mdp/rewards.py`                             | param tùy chọn trên ~6 hàm + `fsm_gated_tracking`, `four_stand_stability`, `transition_progress`  |
| `mdp/terminations.py`                        | + `fsm_transition_timeout`                                                                        |
| `mdp/curriculums.py`                         | `yaw_fsm_task_levels` (mở rộng từ `yaw_task_levels`)                                              |
| `config/.../yaw_env_fsm_cfg.py` (đang trống) | điền `VQRWheelFlatEnvFSMCfg(VQRWheelFlatEnvCfg)`                                                  |
| `agents/rsl_rl_ppo_cfg.py`                   | ưu tiên tái dùng runner hiện có; config riêng chỉ khi cần tách `experiment_name`                    |
| `scripts/.../train.py`                       | **không đổi**; không checkpoint injection, model surgery hoặc critic warm-up cho FSM                |
| `tests/test_yaw_fsm.py`                      | equivalence scalar↔vectorized, dual-diagonal gates, curriculum masking                            |

### 3.7 Quy trình nghiệm thu
1. Unit test FSM (không cần Isaac — chạy nhanh như `test_yaw_curriculum.py`).
2. Smoke: `train.py --task Flat-VQR-Wheel-Yaw-FSM --num_envs 64 --max_iterations 5`.
3. Sanity không hồi quy: chạy task gốc, xác nhận log vẫn in đúng 18 terms / weight 8.0.
4. Short run 4096 envs vài nghìn iteration, theo dõi TensorBoard `experiment_name = vqr_wheel_yaw_flat_fsm`: tỷ lệ time trong mỗi FSM state, transition success rate.

## 4. Rủi ro chính & giảm thiểu
- **Reward imbalance 2 mode** (policy "trốn" vào FOUR_STAND hoặc refuse hạ bánh) → ma trận gate ở trên + pha curriculum B ép certify cả 2 hướng.
- **Obs đổi chiều (55→62)**: train from-scratch với actor/critic khởi tạo mới; không load checkpoint cũ và không model surgery.
- **Threshold predicate cứng** (0.8×target, ngưỡng roll/pitch) sai → bắt đầu hời hợt, tighten dần theo curriculum hoặc tune thủ công sau short run.

Trước khi implement, có 4 quyết định thuộc về bạn:
