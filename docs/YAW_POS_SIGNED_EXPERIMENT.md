# POS signed-ground and stationary-CoM experiment

This experiment changes only `Flat-VQR-Wheel-Yaw-POS`.
It resumes `2026-10-02_15-05-31/model_36999.pt` for 1,000 PPO updates,
with 4,096 environments and seed 42. RSL-RL labels these updates
36999–37998; the final checkpoint is `model_37998.pt` in the new run.

The old combined differential term is replaced by three active-only terms:

| Term | Definition | Weight |
|---|---|---:|
| `signed_ground_participation` | `min_i clip(measured_i / target_i, 0, 1)` for FL/HR; both contacts and valid geometry required | 2 |
| `rolling_tracking` | `1 / (1 + max_i ((measured_i - target_i) / scale_i)^2)`; `scale_i = max(0.05, 0.25 * abs(target_i))` | 2 |
| `active_com_stationary` | `1 / (1 + (mass_weighted_CoM_XY_speed / 0.05)^2)` | 3 |

Motor magnitude is used only for telemetry and the existing strict
differential certificate. All certificate predicates, thresholds, settle
time, curriculum levels, neutral reward functions/parameters and actor
observation dimensions are preserved. Existing rolling-slip shaping is
also retained.

On resume, only FL/HR action std is set to 0.12. Only their Adam moments
are cleared; leg and FR/HL std, their moments, actor/critic tensors,
optimizer groups, iteration and curriculum state are preserved. This is
an initial override; PPO may subsequently learn the std values.

Policy joint-velocity observations retain their order and dimension.
Leg noise stays uniform ±3 rad/s, and all four wheels use ±0.5 rad/s,
before the unchanged 0.05 observation scaling. POS online interval push
is now ±0.15 m/s at DR=1 and scales with DR at earlier levels. Other
tasks retain their existing push/noise configuration. Runtime event
settings are applied by EventManager; the saved initial `env.yaml`
event parameters can still show zero ranges.

The new TensorBoard diagnostics under `Curriculum/task_levels/` are
`differential_{fl,hr}_signed_ratio`, `differential_{fl,hr}_wrong_sign_pct`,
`differential_any_wrong_sign_pct`, `differential_{fl,hr}_overspeed` and
`differential_{fl,hr}_overspeed_pct`. Wrong-sign means measured and target
have opposing signs. Overspeed is excess signed ratio above 1; its
percentage counts samples with ratio >1. Ratios and percentages average
active samples, including transitions; the unchanged differential
certificate excludes the first 0.5 seconds. Existing `active_com_planar_speed`,
`certification/differential_pass_rate` and `heading_error` provide CoM XY,
certificate and active-yaw MAE.

Validation: 179 focused CPU tests pass, including a real checkpoint
resume, joint/order permutations, optimizer-moment isolation, wrong-sign
and overspeed telemetry, CoM motion, noise bounds and unchanged strict
certification. An AST comparison with the pre-change workspace confirms
unchanged neutral functions, sample certificate, settle mask and all
POS curriculum threshold parameters. `git diff --check` and compilation pass.

Run artifacts are in `outputs/yaw_pos_signed_experiment/`: `training.log`,
`launch.json`, `manifest.json`, the experiment diff, and a source snapshot.
The completion monitor writes `result.json`, `result.md` and the requested
scalar series after training exits, verifying exactly 1,000 updates,
the final checkpoint and finite scalars/model tensors.
