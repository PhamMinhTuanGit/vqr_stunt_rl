# Tài liệu thiết kế FSM cho task Flat-VQR-Wheel-Yaw-FSM

Tài liệu thiết kế bộ FSM gồm 4 trạng thái chính và 2 trạng thái chuyển tiếp cho tác vụ xoay 2 bánh tại chỗ

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

## I. Trạng thái đứng im:
Trạng thái đứng im yêu cầu robot đứng yên không di chuyển theo một pose có sẵn được định nghĩa trong 

## II. trạng thái TRANSITION_POS:
Robot nhận được trạng thái này khi đang ở trạng thái đứng yên và nhận lệnh yaw-cmd > deadband:
* Lệnh này ảnh hưởng đến reward như thế nào?
- Giảm dần trọng số của R_four 