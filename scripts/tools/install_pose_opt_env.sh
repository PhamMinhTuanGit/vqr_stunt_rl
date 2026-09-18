#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="vqr_poseopt"
PYTHON_VERSION="3.11"
PINOCCHIO_VERSION="4.1.0"
CASADI_VERSION="3.7.2"

echo "========================================"
echo " VQR Pose Optimization Environment"
echo "========================================"

# ------------------------------------------------------------
# 0. Check conda
# ------------------------------------------------------------

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found."
    echo "Install Miniconda/Miniforge first."
    exit 1
fi

# Make conda activation available inside bash script
eval "$(conda shell.bash hook)"

# ------------------------------------------------------------
# 1. Remove old environment
# ------------------------------------------------------------

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[INFO] Removing old environment: ${ENV_NAME}"
    conda deactivate 2>/dev/null || true
    conda env remove -n "${ENV_NAME}" -y
fi

# ------------------------------------------------------------
# 2. Create clean environment
# ------------------------------------------------------------

echo "[INFO] Creating ${ENV_NAME}..."

conda create \
    -n "${ENV_NAME}" \
    -c conda-forge \
    -y \
    python="${PYTHON_VERSION}" \
    pip \
    numpy \
    pyyaml \
    cmake \
    ninja \
    pkg-config \
    cxx-compiler

conda activate "${ENV_NAME}"

# ------------------------------------------------------------
# 3. Install Pinocchio
# ------------------------------------------------------------

echo "[INFO] Installing Pinocchio ${PINOCCHIO_VERSION}..."

conda install \
    -c conda-forge \
    -y \
    "pinocchio=${PINOCCHIO_VERSION}"

# ------------------------------------------------------------
# 4. Install CasADi
# ------------------------------------------------------------

echo "[INFO] Installing CasADi ${CASADI_VERSION}..."

python -m pip install \
    --upgrade pip setuptools wheel

python -m pip install \
    "casadi==${CASADI_VERSION}"

# ------------------------------------------------------------
# 5. Python import tests
# ------------------------------------------------------------

echo
echo "========================================"
echo " Testing Python packages"
echo "========================================"

python - <<'PY'
import sys
import numpy as np
import pinocchio as pin
import casadi as ca
import yaml

print("Python    :", sys.version.split()[0])
print("NumPy     :", np.__version__)
print("Pinocchio :", pin.__version__)
print("CasADi    :", ca.__version__)
print("PyYAML    :", yaml.__version__)

assert hasattr(pin, "buildModelFromUrdf")
assert hasattr(pin, "JointModelFreeFlyer")

print()
print("[PASS] Pinocchio Python API")
PY

# ------------------------------------------------------------
# 6. Test CasADi + IPOPT
# ------------------------------------------------------------

echo
echo "========================================"
echo " Testing CasADi + IPOPT"
echo "========================================"

python - <<'PY'
import casadi as ca

x = ca.MX.sym("x")

nlp = {
    "x": x,
    "f": (x - 2.0)**2,
}

solver = ca.nlpsol(
    "test_solver",
    "ipopt",
    nlp,
    {
        "ipopt.print_level": 0,
        "print_time": False,
    },
)

sol = solver(x0=0.0)

x_opt = float(sol["x"])

print("IPOPT solution =", x_opt)

if abs(x_opt - 2.0) > 1e-5:
    raise RuntimeError(
        f"IPOPT test failed: x={x_opt}"
    )

print("[PASS] CasADi + IPOPT")
PY

# ------------------------------------------------------------
# 7. Test CMake Pinocchio discovery
# ------------------------------------------------------------

echo
echo "========================================"
echo " Testing CMake Pinocchio discovery"
echo "========================================"

TMP_DIR="$(mktemp -d)"

cat > "${TMP_DIR}/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.16)

project(pinocchio_test LANGUAGES CXX)

find_package(pinocchio REQUIRED)

add_executable(
    pinocchio_test
    main.cpp
)

target_link_libraries(
    pinocchio_test
    PRIVATE
    pinocchio::pinocchio
)
EOF

cat > "${TMP_DIR}/main.cpp" <<'EOF'
#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>

#include <iostream>

int main()
{
    pinocchio::Model model;

    std::cout
        << "Pinocchio C++ OK"
        << std::endl;

    return 0;
}
EOF

cmake \
    -S "${TMP_DIR}" \
    -B "${TMP_DIR}/build" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release

cmake \
    --build "${TMP_DIR}/build"

"${TMP_DIR}/build/pinocchio_test"

rm -rf "${TMP_DIR}"

echo
echo "[PASS] Pinocchio CMake/C++"

# ------------------------------------------------------------
# 8. Final information
# ------------------------------------------------------------

echo
echo "========================================"
echo " Installation complete"
echo "========================================"

echo
echo "Activate with:"
echo
echo "    conda activate ${ENV_NAME}"
echo
echo "Environment:"
echo "    CONDA_PREFIX=${CONDA_PREFIX}"
echo
echo "Python:"
which python
echo
echo "CMake:"
which cmake