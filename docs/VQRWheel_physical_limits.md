# VQRWheel — Giới hạn vật lý & thông số kỹ thuật

Nguồn: `deep_robotics_model/VQRWheel/VQRWheel_urdf/urdf/VQRWheel.urdf` (robot `VQR_01`, xuất từ SolidWorks URDF Exporter).

## Tổng quan hệ thống

- **Cấu hình**: quadruped, mỗi chân 3 khớp lái (HipX, HipY, Knee) + 1 bánh xe → **12 khớp lái + 4 bánh xe (continuous)**
- **Tổng khối lượng**: 17.46 (TORSO) + 4 × (1.0 HIP + 1.5 THIGH + 0.15 SHANK + 1.2 WHEEL) = 17.46 + 15.4 = **32.86 kg**
- **Ký hiệu chân**: FL, FR, HL, HR (Front-Left, Front-Right, Hind-Left, Hind-Right)

## Giới hạn khớp (Joint Limits)

| Khớp | Loại | Trục | Lower [rad] | Upper [rad] | Effort [N·m] | Velocity [rad/s] |
|---|---|---|---|---|---|---|
| `*_HipX_joint` (4 chân) | revolute | (-1, 0, 0) | **-0.7854** | **0.7854** (±45°) | **60** | **14.66** |
| `*_HipY_joint` (4 chân) | revolute | (0, -1, 0) | **-3.4** | **2.4** | **60** | **14.66** |
| `*_Knee_joint` (4 chân) | revolute | (0, -1, 0) | **0.8116** | **2.7663** | **60** | **14.66** |
| `*_WHEEL` (4 bánh) | continuous | (0, -1, 0) | không giới hạn (quay tự do) | — | — | — |

> Lưu ý: bánh xe là khớp `continuous` nên URDF không khai báo limit/effort/velocity; effort/velocity của bánh lấy từ cấu hình actuator khi convert sang USD/Isaac (kiểm tra `VQRWheel_usd/` hoặc env cfg).

## Thông số link (khối lượng & quán tính)

| Link | Khối lượng [kg] | Trọng tâm (x, y, z) [m] | Ixx | Iyy | Izz [kg·m²] |
|---|---|---|---|---|---|
| TORSO | 17.46 | (0, 0, 0) | 0.241554 | 0.524066 | 0.447501 |
| HIP (mỗi bên) | 1.0 | (0, 0, 0) | 0.00078043 | 0.00068757 | 0.00078043 |
| THIGH (mỗi bên) | 1.5 | (0, ±0.06, -0.08) | 0.0104 | 0.0082 | 0.0033 |
| SHANK (mỗi bên) | 0.15 | (0, ±0.01, -0.08) | 0.00089039 | 0.00090672 | 3.1266e-05 |
| WHEEL (mỗi bánh) | 1.2 | (0, 0, 0) | 0.0026443 | 0.0049686 | 0.0026443 |

Dấu y đổi theo bên: FL/HL = +y, FR/HR = -y. Các product of inertia (Ixy, Ixz, Iyz) đều bằng 0.

## Kích thước hình học chính (theo URDF)

| Đại lượng | Giá trị [m] |
|---|---|
| HipX origin FL/FR tại TORSO | (±0.1924, ±0.07, -0.0255) |
| HipX origin HL/HR tại TORSO | (∓0.1924, ±0.07, -0.0255) |
| HipY offset so với HIP | (+0.056 trước / -0.056 sau, ±0.03, 0) |
| Knee offset so với THIGH | (0, ±0.094, -0.212) → **độ dài đùi L1 = 0.212 m** |
| Wheel offset so với SHANK | (0, ±0.0425, -0.213) → **độ dài cẳng L2 = 0.213 m** |
| Bán kính bánh xe | **0.091 m** (collision cylinder dày 0.04 m) |
| TORSO collision box | 0.7 × 0.08 × 0.18 m |
| THIGH collision box | 0.04 × 0.045 × 0.24 m |
| SHANK collision cylinder | r = 0.015 m, dài 0.12 m |

## Động học / hình học đạo hàm

- **Wheelbase (HipX trước ↔ sau)**: 2 × 0.1924 = **0.3848 m**
- **Track (HipX trái ↔ phải)**: 2 × 0.07 = **0.14 m** (tại TORSO; giữa hai trục Knee là 0.14 + 2×0.03 = 0.20 m)
- **Tổng dài chân (HipY → tiếp đất)**: 0.212 + 0.213 + 0.091 ≈ **0.516 m** (knee duỗi thẳng); chiều cao đứng tối đa từ TORSO ≈ 0.0255 + 0.516 ≈ **0.54 m**
- Knee giới hạn 0.8116–2.7663 rad (~46.5°–158.5°): **không bao giờ duỗi thẳng hoàn toàn** → tránh kỳ dị (singularity) và giới hạn vùng workspace của chân

## Quy ước trục khớp

- HipX trục `(-1, 0, 0)`: góc dương xoay quanh trục -x (ab/adduction)
- HipY, Knee, Wheel trục `(0, -1, 0)`: góc dương xoay quanh trục -y
- Damping/friction trong URDF = 0; ma sát/khối riêng tiền tệ được cấu hình lại khi convert sang USD/Isaac

## Ghi chú

- File backup: `VQRWheel.urdf.bak` cạnh file URDF
- Datasheet motor tham chiếu: `SEAF70A16NG01关节电机规格书20250723.pdf` (repo root)
