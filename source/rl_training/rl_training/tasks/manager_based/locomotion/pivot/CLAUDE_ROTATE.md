> 🤔 Đang suy luận sâu…

Tạo lại — lần này tôi in toàn bộ nội dung ra đây rồi lưu file, bạn có thể copy trực tiếp nếu link vẫn lỗi.

---

# AGENTS.md

Hướng dẫn cho AI agent làm việc trên repo này.

## 1. Bối cảnh

Robot: **VQRWheel** — 4 chân có bánh, 12 khớp chân + 4 bánh continuous.

Task: `locomotion/pivot/` — xoay tại chỗ với 4 chế độ, có recovery.

```
GROUND ──► REAR_UP ──► BALANCE ──► LAND ──► GROUND
                └──────── RECOVER ◄───────┘
```

| Chế độ | Nội dung |
|---|---|
| GROUND | Xoay 4 bánh (skid-steer) |
| REAR_UP | Chồm lên, nhấc 2 bánh trước |
| BALANCE | Thăng bằng 2 bánh sau + xoay quanh trục đứng |
| LAND | Hạ 4 bánh, xung lực thấp |
| RECOVER | Tự cứu / tự lật dậy |

Ngoài phạm vi: di chuyển tịnh tiến, địa hình gồ ghề, thao tác vật thể.

## 2. Hằng số vật lý

⚠️ **Mọi hằng số vật lý nằm trong `config/wheeled/vqr/physical_params.py`. Không hard-code số ở bất kỳ file nào khác.**

| Ký hiệu | Giá trị | Ghi chú |
|---|---|---|
| W / L / R | 0.4730 / 0.4968 / 0.091 m | track / wheelbase / bán kính bánh |
| m | 32.86 kg | |
| d | 0.2382 m | CoM trước trục bánh sau |
| h | 0.4424 m | CoM height khi balance |
| **ω₀** | **4.118 rad/s** | rigid-body, **không** dùng point-mass 4.709 |
| τ | 242.9 ms | hằng số thời gian lật |
| **θ*₀** | **32.58°** | pitch cân bằng danh định |
| I_C | diag(1.28498, 1.97865, 1.81878); I_xz = −0.02332 | kg·m² |
| I_spin | 1.685 kg·m² | pose balance |
| Bánh | **24 Nm peak / 3 Nm liên tục** / 58.90 rad/s | |
| Chân | 60 Nm / 14.66 rad/s | |
| HAA / HFE / KFE | ±45° / [−194.8°, 137.5°] / [46.5°, 158.5°] | |

Đại lượng dẫn xuất tính trong `__post_init__`, không viết tay.

`scripts/pivot/verify_physical_params.py` đối chiếu SSOT với URDF — chạy trong CI.

## 3. Công thức chuẩn

```
ξ       = ζ + ζ̇ / 4.118            # capture point; ζ = x_CoM − x_contact,sau
θ*(ω_z) = 32.58° + 0.0931°·ω_z²    # feed-forward momen spin
ξ_safe  = 0.50 · μ̂ · h
ξ_max   = 0.85 · μ̂ · h
ζ_thermal = 0.0906 m               # giới hạn duy trì vô thời hạn @ 3 Nm
F_n,peak < 645 N                   # = 2mg, ngân sách LAND
```

## 4. Bất biến — không được vi phạm

| # | Quy tắc |
|---|---|
| I1 | Biến cân bằng là **ξ**, không dùng pitch thô (θ* dịch theo `tuck*` và ω_z²) |
| I2 | Bánh điều khiển bằng **torque**. Không dùng velocity target |
| I3 | Chân là **PD residual scale 0.3** quanh `q_prior(mode, tuck*)` |
| I4 | Khi \|ξ\| > ξ_safe: **TẮT anchor, TẮT bám ω_z** |
| I5 | Phạt ngửa vượt θ* **×3**; setpoint lệch 3–5° về phía trước |
| I6 | Self-collision tính bằng **reward analytic** (capsule). PhysX `enabled_self_collisions=False` |
| I7 | Không đưa **yaw tuyệt đối** vào observation |
| I8 | ζ/ξ/EMA\|τ_w\|/clearance tính **một lần mỗi step** trong `PivotStateCache`; MDP term chỉ đọc |
| I9 | `supervisor_gate.py` **không import Isaac Lab** — train và deploy dùng chung |
| I10 | Trần ω_z: **±6.0** (BALANCE), **±3.0** (GROUND) |
| I11 | Collision bánh dùng **cylinder primitive**, không dùng mesh |

## 5. Kiến trúc

```
Supervisor (không học)  → gate ξ_safe/ξ_max, rate-limit, chuyển mode
π_main    (1 policy, 4 mode, 50 Hz) → 12 PD residual + 4 wheel torque
π_recover (tách riêng, 50 Hz)       → tự lật dậy
```

Một policy cho cả 4 mode vì chuyển tiếp là phần khó nhất. `π_recover` tách riêng vì phân bố state rời rạc.

## 6. Bản đồ file

```
source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/
├── __init__.py                          ✅ có sẵn
├── pivot_env.py                         ➕ PivotEnv: _pre_physics_step → cập nhật cache
│
├── mdp/
│   ├── __init__.py                      ✅ có sẵn — export thêm term mới
│   ├── to_reference.py                  ✅ có sẵn — GIỮ NGUYÊN, dùng cho q_prior
│   ├── observations.py                  ✅/➕ đọc cache; history stack 5 frame
│   ├── rewards.py                       ✅/➕ term thuần + mode_gated()
│   ├── terminations.py                  ✅/➕ fallen, tilt, drift, grace window
│   ├── events.py                        ✅/➕ reset distribution + DR
│   ├── curriculums.py                   ✅/➕ S0–S6
│   ├── commands.py                      ➕ mode one-hot + ω_z* + Δθ* + tuck*
│   ├── theta_star.py                    ➕ θ*(ω_z) = 32.58 + 0.0931·ω_z²
│   ├── supervisor_gate.py               ➕ pure fn, KHÔNG import Isaac Lab  ← I9
│   ├── actions.py                       ➕ PriorResidualJointPositionAction
│   └── leg_prior.py                     ➕ q_prior(mode, tuck*) — bọc to_reference
│
├── state/                               ➕ subpackage mới (không đụng file cũ)
│   ├── __init__.py
│   ├── cache.py                         PivotStateCache                  ← I8
│   ├── balance_provider.py              Protocol + SimBalanceState
│   ├── balance_estimator.py             EstimatedBalanceState (IMU+kin+contact)
│   ├── thermal_tracker.py               EMA|τ_w|, τ = 2 s
│   ├── clearance.py                     capsule distance                 ← I6
│   ├── capsule_geometry.py              segment-segment distance (vectorised)
│   └── hard_state_buffer.py             ring buffer T−0.3 s
│
└── config/wheeled/vqr/
    ├── __init__.py                      ✅ có sẵn — thêm gym.register
    ├── robot_cfg.py                     ✅ có sẵn SỬA :94  20 → 24 Nm
    ├── balance_env_cfg.py               ✅ có sẵn — GIỮ, dùng làm BALANCE đơn lẻ
    ├── physical_params.py               ➕ SSOT                          ← §2
    ├── capsule_model.py                 ➕ capsule ảo cho *_HIP
    ├── pivot_env_cfg.py                 ➕ cfg đầy đủ 4 mode
    ├── recover_env_cfg.py               ➕ π_recover (dùng lại robot_cfg + mdp)
    └── agents/
        └── rsl_rl_ppo_cfg.py            ✅ có sẵn — thêm entry cho pivot/recover                    runtime, supervisor_runtime, mu_estimator
```

## 7. MDP

**Command (8):** mode one-hot(4) · ω_z* · Δθ* (±8° quanh θ*(ω_z)) · tuck* ∈ [0,1]

**Observation:** ω_base(3), g_proj(3), q(12), q̇(12), ω_wheel(4), a_{t−1}(16), **ζ/ζ̇/ξ(3)**, EMA\|τ_w\|(4), contact flags(4), command(8), **history stack 5 frame**

**Privileged (critic):** contact force, μ, CoM offset, mass, delay, v_base

**Action (16):** 12 chân PD residual · 4 bánh torque

**Reward** (`mode_gated`):

| Term | GROUND | REAR_UP | BALANCE | LAND |
|---|---|---|---|---|
| Bám ω_z | 1.0 | 0.3 | 1.5 | 0.3 |
| exp(−ξ²/σ) | — | 1.5 | **2.5** | — |
| Anchor trung điểm trục sau | 1.0 | 0.5 | 1.0 | 0.5 |
| Bánh trước rời đất | — | 1.5 | 1.0 | — |
| Roll → 0 | 0.5 | 1.0 | 1.0 | 1.0 |
| Bám tuck* | — | — | 0.5 | — |
| max(0, EMA\|τ_w\|−3)² | 0.3 | 0.3 | 0.5 | 0.3 |
| Self-collision analytic | 1.0 | 1.0 | 1.0 | 1.0 |
| −max F_n | — | — | — | **2.0** |
| Pitch ngửa vượt θ* | — | ×3 | ×3 | — |
| action_rate / joint_accel | −0.015 / −2.5e-7 | ← | ← | ← |

## 8. Thang recovery

| Tầng | Kích hoạt | Hành vi |
|---|---|---|
| L0 | V(s) < V_bail hoặc EMA\|τ_w\| > 3 Nm | Hạ ω_z*, duỗi chân |
| L1 | ξ vượt ngưỡng trong REAR_UP | Về GROUND |
| L2 | ξ_safe < \|ξ\| < ξ_max | Bánh đưa điểm đỡ về dưới CoM + vung chân trước |
| L3 | ξ > ξ_max | Đổ **về trước** có kiểm soát, PD mềm |
| L4 | Đã ngã | π_recover |

## 9. Training

**Curriculum:**

| GĐ | Nội dung | Gate |
|---|---|---|
| S0 | Xoay 4 bánh | r_ω > 0.8 |
| S1 | Chồm → hạ, ω_z = 0 | success > 90% |
| S2 | Balance tĩnh + push | r_ξ > 0.7, 20 s |
| S3 | Ramp ω_z → ±6, bật feed-forward θ* | \|Δθ\| < 4° |
| S4 | Mở tuck* + bật `EstimatedBalanceState` | |
| S5 | Ép mất bám: μ → 0.3 ở 30% env | |
| S6 | Full DR + hard-state buffer | |

**Reset distribution:** 30% đứng 4 bánh · 25% giữa chừng chồm · **30% gần lật (ζ, ζ̇ ~ U[±ξ_max])** · 15% hard-state buffer

**Termination:** sớm = terminate ngay khi ngã → sau = grace window 0.5 s

**DR:** μ ∈ [0.3, 1.2] · CoM offset x ±4 cm · mass ±15% · delay 2–6 steps (10–30 ms) · IMU pitch ±1°, gyro 0.02 rad/s · wheel radius ±3% · push 0–120 N·s

**Sim:** physics 200 Hz · control 50 Hz (decimation 4) · 4096 env · PPO, GAE 0.95, clip 0.2

**ActuatorCfg:**
```
wheel: effort_limit=24, velocity_limit=58.9, stiffness=0,  damping=0.3
leg:   effort_limit=60,                     stiffness=60, damping=2.0
```

## 10. Nghiệm thu

| Chỉ số | Mục tiêu |
|---|---|
| Chồm lên thành công | > 95% |
| Balance @ ω_z = 0 / 4 / 6 | > 60 / 30 / 15 s |
| Recovery success vs xung đẩy | biểu đồ chính |
| Recovery success vs μ | không sụp dưới μ = 0.5 |
| F_n,peak khi LAND | < 645 N |
| Trôi sau 10 s BALANCE | < 20 cm |
| Slip ratio @ 2 bánh | < 0.05 |
| Sai số θ khi ramp ω_z 0→6 | < 2° |
| EMA\|τ_w\| trong 60 s BALANCE | < 3 Nm |
| Clearance tối thiểu | > 20 mm |

## 11. Quy ước

- Thêm hằng số vật lý → **chỉ** sửa `physical_params.py`, chạy `verify_physical_params.py`
- Thêm đại lượng dùng bởi ≥2 MDP term → đưa vào `PivotStateCache`, không tính tại chỗ
- Sửa logic gate an toàn → **chỉ** sửa `supervisor_gate.py`
- Sửa action box → chạy lại `sweep_clearance_actionbox.py` (±0.3 rad, toàn box)
- Gym ID: `Pivot-VQR-Train-v0`, `Pivot-VQR-Play-v0`, `Recover-VQR-v0`

## 12. Chưa hoàn thiện

- `*_HIP` không có collision geometry trong URDF → dùng capsule ảo trong `capsule_model.py`
- I_C hiện trích ở pose đứng, cần trích lại tại pose balance (`extract_inertia_at_pose.py`)
- Chưa có bumper đuôi trong model; ξ_max đang giữ bảo thủ
- Nguồn μ̂ khi deploy: operator nhập + observer hiệu chỉnh (`mu_estimator.py`)
- Chưa đo latency end-to-end; giữ 50 Hz