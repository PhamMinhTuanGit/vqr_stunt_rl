# Rà soát FSM của `Flat-VQR-Wheel-Yaw-FSM`

Ngày rà soát: 24/09/2026. Mã nền: commit `040240e`, có thay đổi chưa commit ở `mdp/fsm.py` và `tests/test_yaw_fsm.py` (thêm thời gian giữ `pose_ready` 0,10 s). Báo cáo này mô tả **mã trong working tree** tại lúc rà soát. Số liệu training ở [báo cáo run 11:23:35](training_status_2026-09-24_11-23-35.md) là ảnh chụp iteration 629 của run trước thay đổi chưa commit đó.

## 1. Kết luận nhanh

FSM 7 trạng thái, đối xứng POS/NEG, đường sign flip qua `FOUR_STAND`, reset `support_diagonal`, và gate reward theo trạng thái đã được nối vào task. Mã hiện có một tham số không có tác dụng (`yaw_min_dwell`) và ba giới hạn cần lưu ý khi đọc kết quả training: watchdog swing contact có thể bị vô hiệu hóa bởi thời gian thoát YAW ngắn hơn; điểm bắt đầu đo drift được đặt lại mỗi lần tái nhập YAW; và phase B đánh giá thành công theo episode thay vì từng lần chuyển trạng thái. `RETURN_TO_4` không có watchdog riêng.

## 2. Đường đi của task và thứ tự thời gian

- Gym ID `Flat-VQR-Wheel-Yaw-FSM` trỏ tới `VQRWheelFlatEnvFSMCfg` và runner riêng `VQRWheelYawFlatFSMPPORunnerCfg`: [đăng ký](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/__init__.py#L59), [cấu hình](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py#L959), [runner](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/agents/rsl_rl_ppo_cfg.py#L94).
- Physics chạy ở 200 Hz (`dt=0,005 s`), decimation 4: command, reward và FSM ở 50 Hz (`step_dt=0,020 s`); episode 20 s. Command yaw là tensor `(N, 1)`, resample sau 4–6 s. Biên yaw ban đầu do curriculum đặt thành `[-0,25; +0,25] rad/s`, dù cấu hình command khai báo `[-1; +1]` trước khi curriculum cập nhật: [môi trường](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/velocity_yaw_env_cfg.py#L914), [command](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/commands.py#L191), [curriculum](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py#L1144).
- FSM sống trong `YawFSMCommand._update_command()`: command gốc được cập nhật trước, sau đó đọc sensor và gọi FSM. Resample đổi yaw command, không reset FSM: [mã command](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py#L508).
- Theo Isaac Lab đang dùng, một `env.step()` tính **termination → reward → reset các env hoàn tất → command/FSM → observation**. Vì thế reward và các watchdog đọc trạng thái FSM ở cuối step trước, còn observation trả về dùng trạng thái vừa cập nhật. Termination unsafe đọc sensor hiện tại riêng. Ở phase C, khi unsafe mới xuất hiện, reward của step đó vẫn có thể đọc state cũ; từ step kế tiếp SAFE gate mới có hiệu lực. Đây là độ trễ theo thứ tự framework, cần dùng cùng một mốc state khi đối chiếu log. Nguồn framework: `/home/robotics/datld5/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:204-238`.

## 3. Trạng thái, predicate và chuyển trạng thái

| State | Giá trị | Đường ra khi không unsafe |
|---|---:|---|
| `FOUR_STAND` | 0 | `yaw_cmd > +0,10` → `TRANSITION_POS`; `yaw_cmd < −0,10` → `TRANSITION_NEG` |
| `TRANSITION_POS` | 1 | `yaw_cmd < +0,05` → `RETURN_TO_4`; POS pose ready liên tục 0,10 s → `YAW_POS` |
| `YAW_POS` | 2 | `yaw_cmd < +0,05` → `RETURN_TO_4` ngay; POS pose không ready liên tục 0,10 s → `TRANSITION_POS` |
| `TRANSITION_NEG` | 3 | `yaw_cmd > −0,05` → `RETURN_TO_4`; NEG pose ready liên tục 0,10 s → `YAW_NEG` |
| `YAW_NEG` | 4 | `yaw_cmd > −0,05` → `RETURN_TO_4` ngay; NEG pose không ready liên tục 0,10 s → `TRANSITION_NEG` |
| `RETURN_TO_4` | 5 | `four_stand_ready` → `FOUR_STAND`; lệnh yaw chỉ được xét ở lần cập nhật sau |
| `SAFE_RECOVERY` | 6 | Hết unsafe và `four_stand_ready` liên tục 0,50 s → `FOUR_STAND` |

Unsafe có ưu tiên cao nhất và đưa mọi state sang `SAFE_RECOVERY`. POS support là FL+HR, swing là FR+HL; NEG đảo lại. `pose_ready` đòi cả hai bánh support có lực contact >1 N, cả hai bánh swing cao ít nhất `0,8 × target_clearance` tính từ đáy bánh, và `|roll|`, `|pitch| < 0,35 rad`. `four_stand_ready` đòi cả bốn bánh contact và giới hạn góc đó. Unsafe là torso contact >1 N, chiều cao base <0,35 m, hoặc `|roll|`/`|pitch| >0,80 rad`: [predicate](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/observations.py#L178), [FSM scalar/vector](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py#L38).

`support_diagonal` nhận `+1` cho POS, `−1` cho NEG, giữ giá trị qua TRANSITION/YAW/RETURN và về 0 khi vào `FOUR_STAND` hoặc reset. `state_time` về 0 khi state đổi; `just_switched` đánh dấu đúng lần đổi. `yaw_entry_pos` lấy vị trí XY tại mỗi lần **vào** `YAW_POS` hoặc `YAW_NEG`: [buffer và reset](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py#L246), [anchor](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py#L616).

## 4. Reward, observation, termination và curriculum

| Thành phần | Hành vi hiện tại |
|---|---|
| Observation | Actor thêm one-hot 7 state (tổng dự kiến 62 chiều). Critic thêm one-hot, bốn ready/unsafe flag, `state_time` và hai hình học support được chọn theo diagonal (tổng dự kiến 95 chiều). Noise của nhóm kế thừa không đổi. [Cấu hình](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py#L906) |
| Reward | 22 term: 10 nền; 8 term hình học/ổn định có gate; 4 term tracking, tiến độ transition, drift và entry SAFE. `fsm_gated_tracking` chỉ cho tracking yaw khi **cả hai** bánh support contact; POS/NEG chọn chung một term qua `support_diagonal`. Hình học CoM và lift vẫn có tín hiệu khi mất contact. [Cấu hình](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/vqr_wheel/yaw_env_fsm_cfg.py#L281), [tracking](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py#L680), [gate](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm_gates.py#L49) |
| Termination | Unsafe sau grace tính từ reset 0,15 s ở phase A/B; phase C đưa vào SAFE thay vì terminate. TRANSITION liên tục ≥3,0 s terminate. Một bánh swing contact liên tục ≥0,20 s **khi ở YAW** terminate. [Mã](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py#L62) |
| Curriculum | A: clearance `0,05→0,10→0,15→0,20 m`, đủ lift **và** support mỗi hướng; B: tỉ lệ episode từng hướng có TRANSITION rồi đến YAW ≥0,85; C: tăng yaw/online DR theo 6 mức, dùng tracking ratio và drift từng hướng. Mỗi hướng cần ≥1024 episode/cửa sổ và 3 cửa sổ pass; weight drift thay đổi bằng ramp 3600 step. [Mã](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py#L662) |

Reward dương bị tắt trong `SAFE_RECOVERY`; penalty nền vẫn tính. `transition_progress` dùng potential có `gamma=0,99`, cập nhật potential cả ngoài state được gate và zero reward ở bước vừa đổi state. `spin_center_drift` chỉ có weight ở phase B/C. Khi log, `drift_10s=0` mà `drift_10s_samples=0` nghĩa là **chưa đo được**, không phải drift bằng 0.

## 5. Phát hiện cần xử lý hoặc theo dõi

### F1 — Cao: watchdog swing contact có thể không đạt ngưỡng trên nền phẳng

Nếu bánh swing chạm nền, clearance của nó thường thấp hơn ngưỡng pose ready. FSM rời `YAW_*` về `TRANSITION_*` sau 0,10 s mất pose liên tục; timer của `SwingContactTimeout` chỉ chạy ở YAW và bị xóa ngay khi rời YAW. Ngưỡng termination là 0,20 s. Vì vậy một chuỗi contact trên nền phẳng có thể liên tục gây mất pose nhưng **không bao giờ đủ 0,20 s trong YAW** để kích hoạt watchdog. Đây là suy luận từ [pose predicate](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/observations.py#L230), [pose-loss grace](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py#L393), [timer](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py#L159); contact do va vào vật cao có thể khác. Nên kiểm tra watchdog bằng trajectory contact thật và quyết định tính thời gian qua cả `TRANSITION_*` hoặc đổi ngưỡng/điều kiện cho phù hợp.

### F2 — Trung bình: `yaw_min_dwell=0,20 s` không có tác dụng

Hai FSM lưu tham số nhưng không dùng nó trong predicate thoát. Một lệnh đi qua ngưỡng exit ngay sau khi vào YAW đưa state vào `RETURN_TO_4` ở lần cập nhật kế tiếp. [Mã](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py#L84), [test xác nhận thoát ngay](../../tests/test_yaw_fsm.py#L147). [Tài liệu FSM hiện có](fsm.md) lại nói chỉ thoát do command sau dwell 0,20 s. Cần chốt một semantics rồi đồng bộ mã, test và tài liệu; unsafe vẫn phải có ưu tiên tức thì.

### F3 — Trung bình: drift có thể bị che bởi việc tái nhập YAW

`YAW_* → TRANSITION_*` xảy ra sau 0,10 s pose loss; vào lại YAW sau 0,10 s pose ready, và mỗi lần vào lại sẽ ghi đè `yaw_entry_pos`. Reward drift đo từ anchor mới, còn metric `drift_10s` yêu cầu ở YAW liên tục 10 s. Do đó trajectory bị ngắt quãng có thể trôi xa khỏi điểm bắt đầu maneuver trong khi penalty drift nhỏ và `drift_10s_samples=0`: [chuyển state](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py#L393), [anchor](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py#L616), [reward/metric](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py#L978). Đây là rủi ro đo lường, chưa có bằng chứng policy đang cố tình khai thác. Nên đo thêm displacement từ đầu maneuver, giữ anchor đến khi trở lại `FOUR_STAND`, hoặc log số lần tái nhập YAW và tỷ lệ mẫu 10 s.

### F4 — Trung bình: phase B đếm thành công theo episode, không theo từng attempt

Reward ghi hai boolean `transition_attempted` và `transition_succeeded` cho **mỗi episode và mỗi hướng**. Curriculum đánh giá `success=transitioned` khi `phase==1`; một attempt abort rồi một attempt sau thành công trong cùng episode vẫn được tính một lần thành công. Vì thế metric không chứng minh mọi lần chuyển trạng thái đạt YAW trong 3 s. Watchdog 3 s chỉ bắt chuỗi TRANSITION liên tục: [ghi cờ](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py#L847), [đánh giá](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py#L844), [watchdog](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py#L173). Nếu tiêu chí nghiệm thu là tỉ lệ thành công **mỗi attempt**, cần đếm attempt và thời gian tương ứng.

### F5 — Trung bình: `RETURN_TO_4` không có giới hạn thời gian riêng

`RETURN_TO_4` chỉ thoát khi cả bốn bánh contact và roll/pitch đủ nhỏ; watchdog 3 s chỉ áp dụng cho `TRANSITION_*`, còn watchdog swing chỉ cho `YAW_*`. Nếu điều kiện bốn bánh không xảy ra, env có thể ở RETURN đến episode timeout 20 s: [cạnh RETURN](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/fsm.py#L445), [watchdog](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/terminations.py#L173). Nên log thời gian RETURN và tỉ lệ episode timeout trong RETURN trước khi quyết định thêm timeout hoặc recovery path.

### F6 — Chất lượng học: support contact vẫn là nút thắt trong run được ghi nhận

Ở iteration 629 của [run cũ](training_status_2026-09-24_11-23-35.md), support score POS/NEG là 0,383/0,421 và `fail_support_rate=100%` cả hai hướng, trong khi lift progress khoảng 0,92. `com_support` và `com_inside_segment` vẫn thưởng hình học khi support contact mất; `fsm_gated_tracking` mới yêu cầu cả hai support contact cho phần tracking và cho partial bonus. Vì vậy reward tổng tăng chưa chứng minh robot giữ đúng hai bánh support. Đây là **quan sát training cũ cộng với rủi ro thiết kế reward**, không phải lỗi transition graph mới được chứng minh. Nên ưu tiên xem `mean_support_score`, contact từng bánh, `support_loss_max_dwell_s` và episode pass phase A ở run với mã hiện tại.

## 6. Đã xác nhận và giới hạn kiểm tra

- Chạy `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .../env_isaaclab_tuanpm48_vqr/bin/python -m pytest -q tests/test_yaw_fsm.py tests/test_yaw_fsm_training_contract.py tests/test_yaw_fsm_observations.py tests/test_yaw_curriculum.py`: **32 passed**. Các test CPU kiểm scalar/vector trajectory, reset, mirror, gate reward, contract đăng ký/runner, critic geometry, và curriculum hai hướng.
- Đối chiếu trực tiếp thứ tự `CommandTerm.compute()` và `ManagerBasedRLEnv.step()` trong bản Isaac Lab trên máy. Chưa chạy smoke simulation/GPU với working tree hiện tại; do đó các suy luận về contact thực tế và hiệu quả training cần đo trong simulator.
- Không sửa mã FSM hay test trong lần rà soát này. Các file `mdp/fsm.py` và `tests/test_yaw_fsm.py` đang có thay đổi chưa commit của working tree và được đọc đúng trạng thái đó.
