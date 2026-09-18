#!/usr/bin/env bash
# Solve the HL-HR rear-axle equilibrium pose using the vqr_poseopt conda env.
#
# Usage:
#   ./run_rear_axle.sh                       # defaults below
#   ./run_rear_axle.sh --clearance 0.04      # extra flags pass through
#
# --rear-straight-deg 50: the URDF knee range cannot produce a straighter
# rear leg than ~48.5 deg thigh/shank bend (audited), so 5 deg default fails.
set -euo pipefail

POSE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /home/robotics/miniconda3/etc/profile.d/conda.sh
conda activate vqr_poseopt

python "${POSE_DIR}/apps/rear_axle_equilibrium.py" \
    --config "${POSE_DIR}/config/equilibrium_relaxed.yaml" \
    --rear-straight-deg 50 \
    "$@"
