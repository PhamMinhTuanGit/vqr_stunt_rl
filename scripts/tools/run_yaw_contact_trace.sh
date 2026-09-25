#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
isaaclab_launcher="${1:-}"
checkpoint="${2:-$repo_dir/logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_11-23-35/model_500.pt}"
trace_csv="${3:-$repo_dir/logs/rsl_rl/vqr_wheel_yaw_flat_fsm/2026-09-24_11-23-35/contact_trace_model_500.csv}"



python3 "$repo_dir/scripts/reinforcement_learning/rsl_rl/play.py" \
    --task Flat-VQR-Wheel-Yaw-FSM \
    --checkpoint "$checkpoint" \
    --num_envs 1 \
    --max_steps 2000 \
    --headless \
    --yaw_contact_trace "$trace_csv" \
    --yaw_contact_limit 0.25

python3 "$repo_dir/scripts/tools/analyze_yaw_contact_trace.py" "$trace_csv"
