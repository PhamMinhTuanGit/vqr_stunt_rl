# Tài liệu thiết kế FSM cho task Flat-VQR-Wheel-Yaw-FSM

FSM gồm 7 trạng thái cho tác vụ chuyển từ bốn bánh sang hai bánh chéo, xoay tại
chỗ, trở về bốn bánh và recovery an toàn.

```python
class VQRFsmState(IntEnum):
    FOUR_STAND = 0
    TRANSITION_POS = 1
    YAW_POS = 2
    TRANSITION_NEG = 3
    YAW_NEG = 4
    RETURN_TO_4 = 5
    SAFE_RECOVERY = 6
```

## Thứ tự triển khai

Transition graph phải được implement và kiểm thử toàn bộ trajectory trước khi
nối reward/gate FSM. Reward không phải cơ chế sửa sai cho transition graph.

## Transition graph bắt buộc

```text
FOUR_STAND ── yaw_cmd > +enter ──> TRANSITION_POS ── positive_pose_ready ──> YAW_POS
     ^                                  |                                      |
     |                                  └── abort/exit ──> RETURN_TO_4 <────────┘
     |                                                         |
     └──────────────── four_stand_ready ────────────────────────┘

FOUR_STAND ── yaw_cmd < -enter ──> TRANSITION_NEG ── negative_pose_ready ──> YAW_NEG
     ^                                  |                                      |
     |                                  └── abort/exit ──> RETURN_TO_4 <────────┘
     |                                                         |
     └──────────────── four_stand_ready ────────────────────────┘

MỌI STATE ── unsafe ──> SAFE_RECOVERY
SAFE_RECOVERY ── safe và four_stand_ready liên tục đủ dwell ──> FOUR_STAND
```

Các bất biến bắt buộc:

- Không tồn tại cạnh `YAW_POS ↔ YAW_NEG`.
- Không tồn tại cạnh `RETURN_TO_4 → TRANSITION_POS/NEG`.
- Khi đang ở `RETURN_TO_4` và `four_stand_ready=True`, state kế tiếp luôn là
  `FOUR_STAND`, không phụ thuộc `yaw_cmd`.
- Sign-flip dương sang âm phải đi đúng trajectory
  `YAW_POS → RETURN_TO_4 → FOUR_STAND → TRANSITION_NEG`. Nhánh âm sang dương
  hoàn toàn đối xứng.
- `FOUR_STAND` chỉ xét lệnh yaw trong lần update FSM tiếp theo; không gộp cạnh
  `RETURN_TO_4 → FOUR_STAND → TRANSITION_*` vào cùng một update.
- Command resampling chỉ thay `yaw_cmd`, không reset hoặc ghi trực tiếp FSM
  state.
- `SAFE_RECOVERY` chỉ thoát về `FOUR_STAND` sau khi robot đã safe và
  `four_stand_ready` liên tục đủ dwell; không thoát thẳng sang transition.

## Quan hệ thời gian với reward và observation

Trong `env.step()`, reward đọc state cũ còn observation thấy state mới. Khi log
theo state phải dùng thống nhất một nguồn để tránh lệch một step tại mỗi lần
chuyển state.
