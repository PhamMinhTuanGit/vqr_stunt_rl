# AGENTS_yaw_fsm.md

Chỉ thị implement task `Flat-VQR-Wheel-Yaw-FSM`. Đọc hết file này trước khi sửa dòng code đầu tiên.

## 0. Ràng buộc bất khả xâm phạm

1. **Task gốc `Flat-VQR-Wheel-Yaw` phải chạy y hệt.** Đang train, có hard-check 18 term / weight 8.0 trong `train.py:206-216`. Mọi tham số mới default `None` → đi nhánh cũ.
2. **`mdp/commands.py` không đổi một ký tự.** Subclass đặt ở `mdp/fsm.py`.
3. Baseline phải **bit-identical**, chứng minh bằng test, không bằng lập luận.
4. Không đổi noise model của observation.
5. Không xoá/đổi tên accumulator mà task gốc và `_export_yaw_curriculum_state` trong `train.py` đang dùng.
6. **Không sửa `train.py`, runner/framework training, checkpoint loader hoặc critic warm-up machinery.** Task FSM được train mới hoàn toàn từ iteration 0 bằng luồng training hiện có.

Nếu một yêu cầu trong file này mâu thuẫn với 6 điểm trên → **dừng, hỏi lại**, đừng tự quyết.

---

## 1. Kiến trúc

FSM sống trong `CommandTerm`, không phải event, không phải wrapper.

```
YawFSMCommand(YawRateCommand)
  _update_command():  super()._update_command()   ← hành vi cũ nguyên vẹn
                      self._step_fsm()            ← chỉ cộng thêm
  command property:   KHÔNG override → vẫn (N,1) yaw rate
  attribute phụ:      fsm_state, support_diagonal, state_time,
                      just_switched, yaw_entry_pos
```

Hệ quả: mọi term đang dùng `generated_commands("yaw_fsm")` hoạt động không sửa. FSM state ra ngoài qua `command_manager.get_term(name)`.

**Cấm** gate `command` theo state (kiểu trả 0 khi TRANSITION). Giữ raw.

### Thứ tự trong `CommandTerm.compute()` — verify với version IsaacLab đang dùng

```
_update_metrics()   ← đọc fsm_state CŨ 1 step
time_left -= dt
_resample(ids)      ← lệnh mới
_update_command()   ← FSM update tại đây
```

Vì vậy: **accumulator curriculum đặt trong reward term, không đặt trong `_update_metrics()`.**

Trong `env.step()`: reward đọc state cũ, obs thấy state mới, cùng một step. Không sai — nhưng khi log per-state phải chọn **một** nguồn, nếu không tỷ lệ `time_in_state` lệch reward breakdown đúng 1 step mỗi lần chuyển.

---

## 2. FSM 7 trạng thái

**Thứ tự triển khai bắt buộc:** hoàn thiện transition graph và cho test toàn bộ
trajectory pass trước khi nối bất kỳ reward/gate FSM nào. Reward không được dùng
để che hoặc bù cho một graph chuyển trạng thái chưa đúng.

```
FOUR_STAND ──yaw>+0.10──► TRANSITION_POS ──pose_ready──► YAW_POS
     ▲                          │                           │
     │                          └──── abort (|yaw|<0.05) ───┤
     └──── four_stand_ready ──── RETURN_TO_4 ◄──────────────┘

(nhánh NEG đối xứng gương, yaw < −0.10)
SAFE_RECOVERY ← unsafe, từ MỌI state
```

- Hysteresis: enter `0.10`, exit `0.05`, limit `±0.25` (giữ như `VQRYawFSM` scalar hiện có).
- **Không có cạnh `YAW_POS ↔ YAW_NEG` và không có cạnh `RETURN_TO_4 → TRANSITION_*`.** Sign-flip bắt buộc đi đủ trajectory `YAW_POS → RETURN_TO_4 → FOUR_STAND → TRANSITION_NEG` (nhánh ngược lại đối xứng).
- Trong `RETURN_TO_4`, khi `four_stand_ready=True` thì state kế tiếp **luôn là `FOUR_STAND`**, bất kể `yaw_cmd` hiện tại. Chỉ ở lần update FSM kế tiếp, logic của `FOUR_STAND` mới được phép xét `yaw_cmd` để vào `TRANSITION_POS` hoặc `TRANSITION_NEG`.
- Command resampling chỉ thay `yaw_cmd`; **không bao giờ reset hoặc ghi trực tiếp FSM state**. FSM tự quyết định abort, return và sign-flip qua transition graph.
- `TRANSITION_* → RETURN_TO_4` (abort) là cạnh **bắt buộc**. Thiếu nó, lệnh về 0 giữa lúc nâng bánh sẽ không có đường ra hợp lệ.
- `transition_timeout = 3.0s` → `DoneTerm` (mặc định bật).

### Predicate (vectorized, tái dùng helper trong `observations.py`)

```
positive_pose_ready = contact(FL,HR) ∧ clearance(FR,HL) ≥ 0.8·target ∧ |roll,pitch| < θ
negative_pose_ready = mirror
four_stand_ready    = contact(4) ∧ |roll,pitch| < θ
unsafe              = torso_contact ∨ base_height < MIN ∨ |roll,pitch| > θ_max
```

Ngưỡng bắt đầu **hời hợt**, siết dần theo pha. Đừng tune chặt trước short run.

### Buffer

| Buffer | Shape | Reset trong `reset()` |
|---|---|---|
| `fsm_state` | (N,) long | `FOUR_STAND` |
| `support_diagonal` | (N,) long {−1,0,+1} | **`0`** |
| `state_time` | (N,) float | `0.0` |
| `just_switched` | (N,) bool | `False` |
| `yaw_entry_pos` | (N,2) float | `0.0` |

⚠️ Sót `support_diagonal` reset → env mới thừa hưởng chéo cũ, `RETURN_TO_4` hạ nhầm cặp bánh. **Không crash, chỉ tụt success rate.** Đây là bug tốn nhiều giờ nhất trong thiết kế này.

`reset()` phải `super().reset()` trước rồi merge extras.

`just_switched` + `state_time = 0` set trong `_update_command()` cuối step *t*, reward step *t+1* đọc.

---

## 3. Ba lớp gate — dùng nhầm lớp là nguồn gốc mọi bug reward

| Lớp | Áp dụng | Cơ chế |
|---|---|---|
| **Hard** | Term vô nghĩa ngoài state | nhân mask 0/1 |
| **Soft** | `four_stand_stability` | hệ số trượt 0→1 |
| **Không gate** | Toàn bộ penalty nền | luôn bật |

**Quy tắc: gate reward dương, không gate reward âm.**

Ngoại lệ hợp lệ duy nhất: `lifted_wheel_spin`, `rolling_slip` — gate vì đại lượng *không định nghĩa được* khi 4 bánh chạm đất, không phải vì ưu tiên.

Mask cache theo `common_step_counter` trong `mdp/fsm_gates.py`, hàm `fsm_gates(env, command_name) -> dict`. 22 term gọi chung một cache.

---

## 4. Reward — 22 term

### Nhóm 1 — Nền (10, không gate, weight cố định suốt run)

`balance` (+, trừ SAFE) · `torque` · `action_rate` · `joint_velocity` · `joint_limits` · `lateral_slip` · `undesired_contact` · `planar_velocity` · `low_base_height` · `downward_low_base_velocity`

Tất cả giữ weight baseline. Trong task FSM, `balance` bị gate khỏi
`SAFE_RECOVERY`; task gốc giữ nhánh `fsm_command_name=None` bit-identical.

**Không đổi weight nhóm 1 giữa run.**

### Nhóm 2 — Gated theo state (8, weight cố định)

`●` hard · `◐` soft · `−` tắt

| Term | W | FOUR | TRANS | YAW | RETURN | SAFE |
|---|---|---|---|---|---|---|
| `com_support` | baseline | − | ● | ● | ● | − |
| `com_inside_segment` | baseline | − | ● | ● | ● | − |
| `support_span_band` | baseline | − | ● | ● | ● | − |
| `lift_clearance` | baseline | − | ● | ● | − | − |
| `base_height` | 0.49 | − | − | ● | − | − |
| `lifted_wheel_spin` | baseline | − | − | ● | − | − |
| `rolling_slip` | baseline | − | − | ● | − | − |
| `four_stand_stability` | **3.0** | ● | ◐ out | − | ◐ in | − |

Soft gate:
```python
tau = state_time.clamp(0, 1)
w = f_four + f_trans*(1-tau) + f_return*tau
```

Chọn chéo bằng **`support_diagonal`**, không bằng `fsm_state` — vì `RETURN_TO_4` phải nhớ chéo đang đỡ.

### Nhóm 3 — Theo pha curriculum (4)

| Term | Pha A | Pha B | Pha C |
|---|---|---|---|
| `fsm_gated_tracking` | 8.0, `floor=0` | 8.0, `floor=0.25·(1−t/3s)` | 8.0, `floor=0` |
| `transition_progress` | 2.0 | 2.0 | 2.0 |
| `spin_center_drift` | 0.0 | −0.5 | **−2.0** |
| `safe_recovery_entry` | *terminate* | *terminate* | **−2.0** (một lần) |

- `transition_progress` là potential-based shaping → **không anneal**. Giảm dần chỉ mất tốc độ hội tụ, không được gì.
- Floor **chỉ** ở pha B. Bật ở pha C = mời policy stall trở lại.
- Scale thật: floor 0.25 × 8.0 × dt 0.02 ≈ 2.0/giây. Đủ lớn để câu giờ → bắt buộc có `decay`.

**Assertion đúng 22 term cho task FSM**, song song với 18 của task gốc.

---

## 5. Potential-based shaping — 2 dòng dễ sai

```python
class TransitionProgress(ManagerTermBase):
    def __call__(self, env, ...):
        phi_up = (clearance / target).clamp(0, 1)
        phi = torch.where(g["b_return"], 1 - phi_up, phi_up)   # RETURN đảo chiều
        r = self.gamma * phi - self.prev_phi
        self.prev_phi.copy_(phi)          # ← LUÔN update, kể cả ngoài gate
        r = torch.where(g["just_switched"], 0.0, r)   # ← zero ở bước chuyển
        return r * (g["f_trans"] + g["f_return"])
```

- Chỉ update `prev_phi` trong gate → lần vào state sau bắn xung giả.
- Không zero ở `just_switched` → hỏng telescoping, để lại bias.
- Phải là `ManagerTermBase` (cần `reset()`), không phải function.

---

## 6. Mirror POS/NEG — một term, hai nhánh

**Cấm tách thành `*_pos` / `*_neg`.** Lý do: số term vỡ, log bị pha loãng 50%, hai weight sớm muộn lệch nhau.

```python
# mdp/fsm_mirror.py — nguồn chân lý duy nhất
SUPPORT_POS = ["wheel_FL", "wheel_HR"]   # front-first, BẮT BUỘC
SUPPORT_NEG = ["wheel_FR", "wheel_HL"]
SWING_POS, SWING_NEG = SUPPORT_NEG, SUPPORT_POS

MIRROR = {
    "com_support":        (SUPPORT_POS, SUPPORT_NEG),
    "com_inside_segment": (SUPPORT_POS, SUPPORT_NEG),
    "support_span_band":  (SUPPORT_POS, SUPPORT_NEG),
    "rolling_slip":       (SUPPORT_POS, SUPPORT_NEG),
    "lift_clearance":     (SWING_POS,   SWING_NEG),
    "lifted_wheel_spin":  (SWING_POS,   SWING_NEG),
    "transition_progress":(SWING_POS,   SWING_NEG),
}
```

7 term mirror, 5 dùng chung (`base_height`, `four_stand_stability`, `fsm_gated_tracking`, `spin_center_drift`, `safe_recovery_entry`).

Pattern:
```python
r_pos = _core(env, asset_cfg)
if fsm_command_name is None: return r_pos      # baseline
r_neg = _core(env, asset_cfg_mirror)
return torch.where(g["diag_pos"], r_pos, r_neg) * g["f_geom"]
```

`_core` **không được** hard-code bất kỳ body trái/phải nào bên trong. Đối xứng phải là *cấu trúc*, không phải may mắn.

### Ba cạm bẫy mirror

1. **Thứ tự endpoint**: phần tử đầu luôn là bánh TRƯỚC. Nếu `com_inside_segment` có băng thưởng lệch trước (bù quán tính), viết `(HL, FR)` sẽ cho NEG băng lệch sau — lệch vài % rất khó truy.
2. **Đại lượng có dấu**: norm/khoảng cách bất biến qua gương. Kiểm `com_support` — nếu có thành phần signed lateral thì phải `×(−1)` cho NEG.
3. **`fsm_gated_tracking` đã tự đối xứng** vì so trực tiếp với `cmd` có dấu. Giữ nguyên. **Cấm** đổi thành `exp(−(|ω_z|−|cmd|)²/σ)` — mất ràng buộc chiều quay, quay ngược vẫn full điểm.

---

## 7. Curriculum 3 pha

| Pha | Nội dung | Gate lên |
|---|---|---|
| **A — lift** | yaw scale nhỏ, certify **cả 2 chéo** | đủ episode mỗi bên |
| **B — transition** | FOUR_STAND → `pose_ready` trong 3s | success ≥ 0.85, 2048 ep, 3 cửa sổ liên tiếp, **cả 2 hướng** |
| **C — yaw+DR** | scale ladder + domain rand | `yaw_track_err` + `drift` đạt ngưỡng |

Gate dùng **`min` hai chéo**, không dùng trung bình:
```python
ready = (score_pos > thr) & (score_neg > thr) & (n_ep_pos > 1024) & (n_ep_neg > 1024)
```
Trung bình → chéo tốt kéo chéo yếu lên level, bên yếu không bao giờ học đủ.

Metric `fsm_gated_tracking` chia cho **số step trong YAW_***, không chia tổng step (loãng 60–80%, curriculum không lên level).

### Đổi weight giữa pha

Mọi lần đổi weight nhóm 3 làm value function sai ngay → KL spike, `explained_variance` tụt ~50 iter. **Mặc định: ramp tuyến tính 100–200 iter ngay trong `yaw_fsm_task_levels`/reward config.** Không thêm critic-only warm-up hoặc logic đặc biệt vào framework training.

---

## 8. Train mới từ đầu (obs 55→62)

Task `Flat-VQR-Wheel-Yaw-FSM` được **train from-scratch từ iteration 0**. Không load checkpoint của `Flat-VQR-Wheel-Yaw`, không model surgery và không pad layer từ model cũ.

- Policy input mới: 55 + one-hot 7 = **62**.
- Critic input mới: 83 + one-hot 7 + 4 ready/unsafe flag + `state_time` = **95**.
- Actor và critic được khởi tạo mới theo runner/config hiện hữu; input dimension được suy ra từ environment.
- Dùng exploration noise khởi tạo bình thường của một run mới. Không có bước “reset `init_noise_std` từ checkpoint”.
- Không critic warm-up, không freeze actor và không sửa checkpoint loader.
- `resume=False`; checkpoint của task gốc không phải input của task FSM.

Nếu cần tách log, chỉ thay `experiment_name` bằng cấu hình khai báo sẵn hoặc CLI; không thêm nhánh xử lý task FSM vào `train.py`.

---

## 9. Bất biến — assert, đừng tin vào weight

Log `budget/<state>` = tổng reward **dương** trung bình/step theo state, kiểm mỗi ~100 iter:

```
budget[SAFE_RECOVERY] = 0
budget[TRANSITION]    ≤ budget[YAW_*]
budget[s] > 0  ∀ s != SAFE_RECOVERY
|budget[POS] − budget[NEG]| / mean < 0.05
```

Bất đẳng thức 4 phát hiện lỗi mirror sớm nhất. Vỡ → **kiểm `MIRROR` dict và thứ tự endpoint trước**. Chỉnh weight để che bất đối xứng là cách chắc chắn nhất để khoá cứng nó vào policy.

Log tách POS/NEG dù term dùng chung (CoM lệch, ma sát, motor strength randomize khác nhau):
```python
for tag, m in (("pos", b_yaw & diag_pos), ("neg", b_yaw & ~diag_pos)):
    extras[f"fsm/{tag}/{track_err, drift, budget, time_frac}"] = ...
```

---

## 10. Files

| File | Việc |
|---|---|
| `mdp/fsm.py` | + `YawFSMVectorized`, + `YawFSMCommand/Cfg`. Giữ `VQRYawFSM` scalar làm oracle |
| `mdp/fsm_gates.py` | **mới** — `fsm_gates()` cache |
| `mdp/fsm_mirror.py` | **mới** — `MIRROR` dict |
| `mdp/commands.py` | **KHÔNG ĐỔI** |
| `mdp/observations.py` | + `fsm_state_onehot`, `fsm_ready_flags` |
| `mdp/rewards.py` | param `fsm_command_name=None` trên 7 hàm + 4 term mới |
| `mdp/terminations.py` | + `fsm_transition_timeout` |
| `mdp/curriculums.py` | `yaw_fsm_task_levels` |
| `config/.../yaw_env_fsm_cfg.py` | điền `VQRWheelFlatEnvFSMCfg` |
| `agents/rsl_rl_ppo_cfg.py` | Không bắt buộc đổi; ưu tiên tái dùng runner hiện có. Chỉ cho phép config khai báo riêng nếu cần `experiment_name`, không thêm training logic |
| `scripts/.../train.py` | **KHÔNG ĐỔI** — không checkpoint injection, model surgery, task-specific warm-up hay nhánh FSM |
| `tests/test_yaw_fsm.py` | mới |

`YawFSMCommandCfg` phải có `class_type: type = YawFSMCommand`. Quên → chạy nhầm class cũ, **im lặng**.

---

## 11. Test (4 test đầu chạy không cần Isaac)

```python
test_baseline_bit_identical()   # fsm_command_name=None → torch.equal với ref
test_fsm_equivalence()          # fuzz 10k step × 1024 env, bit-exact TOÀN BỘ quỹ đạo
                                # state so với VQRYawFSM scalar — không chỉ state cuối
test_sign_flip_visits_four()    # YAW_POS → RETURN_TO_4 → FOUR_STAND → TRANSITION_NEG
                                # cấm RETURN_TO_4 → TRANSITION_NEG trong cùng update
test_resample_preserves_fsm()   # resample chỉ đổi yaw_cmd, không reset/ghi state
test_mirror_is_exact()          # lật y mọi body + dấu ω_z,cmd → allclose(r_pos, r_neg)
test_gate_coverage()            # mọi state có ≥1 term dương
test_pbs_no_spurious_spike()    # |r_progress| tại just_switched == 0
test_budget_invariants()        # 4 bất đẳng thức §9
test_mirror_dict_covers_all()   # term nào có asset_cfg_mirror đều phải trong MIRROR
```

---

## 12. Quy trình nghiệm thu

1. Unit test (không Isaac).
2. Smoke: `train.py --task Flat-VQR-Wheel-Yaw-FSM --num_envs 64 --max_iterations 5`.
3. **Regression**: chạy task gốc, xác nhận log vẫn in đúng 18 term / weight 8.0.
4. Short run 4096 env, `experiment_name = vqr_wheel_yaw_flat_fsm`.

Ngưỡng theo dõi:

| Metric | Mục tiêu |
|---|---|
| `transition_success_rate` (POS/NEG tách) | > 0.98 |
| `yaw_rate_MAE` trong YAW | < 0.15 rad/s |
| **`drift_radius_10s`** | **< 8 cm** |
| `swing_contact_rate` trong YAW | < 1% |
| `time_in_state` mỗi state | không state nào < 8% |
| `fsm_switch_rate` | < 0.5 Hz (cao → hysteresis hỏng) |
| `|budget_pos − budget_neg|` | < 5% |

---

## 13. Quyết định thiết kế

Decision #5 và #6 đã được chốt; implementation và test phải coi đây là contract:

| # | Quyết định | Mặc định đang dùng |
|---|---|---|
| 3 | obs 55→62: resume hay scratch | **Train from-scratch**, `resume=False`; không model surgery, không load checkpoint task gốc, không critic warm-up (§8) |
| 4 | SAFE_RECOVERY semantics | Pha A/B **terminate**; pha C bật, entry cost −2.0, **không term dương**, min dwell 0.5s rồi ép về FOUR_STAND (§4) |
| 5 | Command resampling | Chỉ thay `yaw_cmd`; không reset/ghi FSM state. Abort, return và sign-flip do transition graph quyết định |
| 6 | Recovery exit | `SAFE_RECOVERY → FOUR_STAND` chỉ khi đã safe và `four_stand_ready` liên tục đủ dwell; không exit thẳng sang `TRANSITION_*` |

---

## 14. Optional — làm sau khi Gate B pass

Event reset ~40% env thẳng vào tư thế 2 bánh chéo (base nghiêng sẵn, swing nâng tới target, `fsm_state` = `YAW_*`, `support_diagonal` tương ứng). Cắt đáng kể wall-clock pha C. **Không làm trước khi pha B certify.**

---

## 15. Ba lỗi kinh điển của FSM này

1. **Chattering biên deadband** → hysteresis 0.10/0.05 + `yaw_min_dwell`.
2. **Policy "gian lận" không nhấc bánh** (giữ 4 bánh vẫn quay được phần nào, an toàn hơn) → `lifted_wheel_spin` đủ nặng + termination khi swing chạm đất kéo dài + đưa `m_sup` vào obs.
3. **Trôi vòng tròn thay vì xoay tại chỗ** → `planar_velocity` tức thời KHÔNG đủ (0.03 m/s × 10s = 30 cm, khán giả thấy ngay). Bắt buộc `spin_center_drift` tích luỹ từ `yaw_entry_pos`.
